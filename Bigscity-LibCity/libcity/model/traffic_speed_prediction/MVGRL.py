"""
MVGRL (Multi-View Graph Representation Learning) adapted for LibCity traffic state prediction.

Original paper: "Contrastive Multi-View Representation Learning on Graphs" (ICML 2020)
Original implementation: https://github.com/kavehhassani/mvgrl

Adaptation notes:
- The original MVGRL is an unsupervised contrastive learning model for graph representation
- This adaptation converts it to a supervised traffic prediction model while preserving
  the core multi-view (adjacency + PPR diffusion) spatial encoding mechanism
- The model uses two GCN branches: one for the original adjacency, one for PPR diffusion
- Temporal modeling is added via a temporal convolution layer or GRU
- The contrastive loss can optionally be used as an auxiliary loss during training

Key changes from original:
1. Adapted to LibCity's AbstractTrafficStateModel interface
2. Added temporal modeling layer for time-series traffic data
3. Modified input handling for LibCity's batch format (batch, time, nodes, features)
4. Added prediction head for traffic forecasting
5. PPR diffusion matrix is computed from the adjacency matrix at initialization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from scipy.linalg import fractional_matrix_power, inv

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def compute_ppr_matrix(adj_mx, alpha=0.2, self_loop=True):
    """
    Compute Personalized PageRank (PPR) diffusion matrix.

    PPR = alpha * (I - (1-alpha) * A_tilde)^(-1)
    where A_tilde = D^(-1/2) * A^ * D^(-1/2) and A^ = A + I

    Args:
        adj_mx: numpy adjacency matrix (N, N)
        alpha: teleport probability (default 0.2)
        self_loop: whether to add self-loops

    Returns:
        PPR diffusion matrix (N, N)
    """
    a = adj_mx.copy()
    if self_loop:
        a = a + np.eye(a.shape[0])

    # Compute degree matrix
    d = np.diag(np.sum(a, axis=1))

    # Handle zero degrees
    d_inv_sqrt = np.zeros_like(d)
    non_zero = np.diag(d) > 0
    d_inv_sqrt[non_zero, non_zero] = np.power(np.diag(d)[non_zero], -0.5)

    # Normalized adjacency: A_tilde = D^(-1/2) * A * D^(-1/2)
    a_tilde = np.matmul(np.matmul(d_inv_sqrt, a), d_inv_sqrt)

    # PPR matrix: alpha * (I - (1-alpha) * A_tilde)^(-1)
    n = a.shape[0]
    try:
        ppr = alpha * inv(np.eye(n) - (1 - alpha) * a_tilde)
    except np.linalg.LinAlgError:
        # If matrix is singular, use pseudo-inverse
        ppr = alpha * np.linalg.pinv(np.eye(n) - (1 - alpha) * a_tilde)

    return ppr.astype(np.float32)


def normalize_adj_matrix(adj_mx, self_loop=True):
    """
    Symmetrically normalize adjacency matrix.
    A_norm = D^(-1/2) * A * D^(-1/2)

    Args:
        adj_mx: numpy adjacency matrix (N, N)
        self_loop: whether to add self-loops

    Returns:
        Normalized adjacency matrix (N, N)
    """
    a = adj_mx.copy()
    if self_loop:
        a = a + np.eye(a.shape[0])

    # Compute degree
    d = np.sum(a, axis=1)
    d_inv_sqrt = np.zeros_like(d)
    non_zero = d > 0
    d_inv_sqrt[non_zero] = np.power(d[non_zero], -0.5)
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)

    # Normalize
    a_norm = np.matmul(np.matmul(d_mat_inv_sqrt, a), d_mat_inv_sqrt)

    return a_norm.astype(np.float32)


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer adapted from MVGRL.
    Performs: out = activation(A * X * W + b)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, adj):
        """
        Args:
            x: node features (batch, nodes, features)
            adj: adjacency matrix (nodes, nodes) or (batch, nodes, nodes)

        Returns:
            out: (batch, nodes, out_features)
        """
        # Linear transformation
        x = self.fc(x)  # (batch, nodes, out_features)

        # Graph convolution
        if adj.dim() == 2:
            # Same adjacency for all batches
            out = torch.matmul(adj, x)  # (batch, nodes, out_features)
        else:
            # Different adjacency per batch
            out = torch.bmm(adj, x)  # (batch, nodes, out_features)

        if self.bias is not None:
            out = out + self.bias

        return self.act(out)


class MVGRLEncoder(nn.Module):
    """
    Multi-View GRL Encoder with two GCN branches.
    One branch for original adjacency, one for PPR diffusion.
    """

    def __init__(self, in_features, hidden_dim, num_layers=2):
        super(MVGRLEncoder, self).__init__()
        self.num_layers = num_layers

        # GCN branch for adjacency view
        self.gcn_adj = nn.ModuleList()
        self.gcn_adj.append(GCNLayer(in_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.gcn_adj.append(GCNLayer(hidden_dim, hidden_dim))

        # GCN branch for diffusion view
        self.gcn_diff = nn.ModuleList()
        self.gcn_diff.append(GCNLayer(in_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.gcn_diff.append(GCNLayer(hidden_dim, hidden_dim))

    def forward(self, x, adj, diff):
        """
        Args:
            x: node features (batch, nodes, features)
            adj: normalized adjacency matrix (nodes, nodes)
            diff: PPR diffusion matrix (nodes, nodes)

        Returns:
            h_adj: node embeddings from adjacency view (batch, nodes, hidden_dim)
            h_diff: node embeddings from diffusion view (batch, nodes, hidden_dim)
        """
        # Adjacency view
        h_adj = x
        for layer in self.gcn_adj:
            h_adj = layer(h_adj, adj)

        # Diffusion view
        h_diff = x
        for layer in self.gcn_diff:
            h_diff = layer(h_diff, diff)

        return h_adj, h_diff


class Readout(nn.Module):
    """Graph-level readout by mean pooling."""

    def forward(self, x, mask=None):
        """
        Args:
            x: node features (batch, nodes, features)
            mask: optional node mask

        Returns:
            graph-level representation (batch, features)
        """
        if mask is None:
            return torch.mean(x, dim=1)
        else:
            mask = mask.unsqueeze(-1)
            return torch.sum(x * mask, dim=1) / torch.sum(mask, dim=1)


class Discriminator(nn.Module):
    """
    Bilinear discriminator for contrastive learning.
    Computes scores between local (node) and global (graph) representations.
    """

    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Bilinear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, h_local, g_global):
        """
        Args:
            h_local: local node embeddings (batch, nodes, hidden_dim)
            g_global: global graph embedding (batch, hidden_dim)

        Returns:
            scores: (batch, nodes)
        """
        # Expand global to match local dimensions
        g_expand = g_global.unsqueeze(1).expand_as(h_local)
        scores = self.bilinear(h_local, g_expand).squeeze(-1)
        return scores


class TemporalConv(nn.Module):
    """Temporal convolution layer for processing time-series data."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (batch * nodes, time, features)

        Returns:
            out: (batch * nodes, time, out_channels)
        """
        # Transpose for conv1d: (batch*nodes, features, time)
        x = x.transpose(1, 2)
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        # Transpose back: (batch*nodes, time, out_channels)
        return out.transpose(1, 2)


class MVGRL(AbstractTrafficStateModel):
    """
    MVGRL adapted for traffic state prediction in LibCity.

    This model uses multi-view graph representation learning with:
    - Two spatial views: adjacency and PPR diffusion
    - Temporal modeling via temporal convolution or GRU
    - Optional contrastive auxiliary loss

    Config parameters:
        hidden_dim: hidden dimension for GCN layers (default: 64)
        num_gcn_layers: number of GCN layers per view (default: 2)
        ppr_alpha: teleport probability for PPR computation (default: 0.2)
        use_contrastive_loss: whether to add contrastive auxiliary loss (default: False)
        contrastive_weight: weight for contrastive loss (default: 0.1)
        temporal_type: type of temporal modeling ('conv' or 'gru') (default: 'conv')
        dropout: dropout probability (default: 0.1)
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        # Data features
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)
        self._scaler = data_feature.get('scaler')
        self._logger = getLogger()

        # Model config
        self.hidden_dim = config.get('hidden_dim', 64)
        self.num_gcn_layers = config.get('num_gcn_layers', 2)
        self.ppr_alpha = config.get('ppr_alpha', 0.2)
        self.use_contrastive_loss = config.get('use_contrastive_loss', False)
        self.contrastive_weight = config.get('contrastive_weight', 0.1)
        self.temporal_type = config.get('temporal_type', 'conv')
        self.dropout_prob = config.get('dropout', 0.1)

        # Window sizes
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)

        self.device = config.get('device', torch.device('cpu'))

        # Get adjacency matrix and compute PPR diffusion matrix
        adj_mx = data_feature.get('adj_mx')
        if adj_mx is None:
            self._logger.warning('No adjacency matrix provided. Using identity matrix.')
            adj_mx = np.eye(self.num_nodes)

        # Normalize adjacency and compute PPR
        self.adj_norm = torch.FloatTensor(normalize_adj_matrix(adj_mx)).to(self.device)
        self.diff_mx = torch.FloatTensor(compute_ppr_matrix(adj_mx, self.ppr_alpha)).to(self.device)

        self._logger.info(f'MVGRL initialized with {self.num_nodes} nodes, '
                          f'hidden_dim={self.hidden_dim}, num_gcn_layers={self.num_gcn_layers}')

        # Build model components
        self._build_model()

    def _build_model(self):
        """Build model architecture."""

        # Input projection
        self.input_proj = nn.Linear(self.feature_dim, self.hidden_dim)

        # Multi-view GCN encoder
        self.encoder = MVGRLEncoder(
            in_features=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_gcn_layers
        )

        # Temporal modeling
        if self.temporal_type == 'gru':
            self.temporal = nn.GRU(
                input_size=self.hidden_dim * 2,  # concat of two views
                hidden_size=self.hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=self.dropout_prob
            )
            temporal_out_dim = self.hidden_dim
        else:  # conv
            self.temporal = nn.Sequential(
                TemporalConv(self.hidden_dim * 2, self.hidden_dim),
                TemporalConv(self.hidden_dim, self.hidden_dim),
            )
            temporal_out_dim = self.hidden_dim

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(temporal_out_dim * self.input_window, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim, self.output_window * self.output_dim)
        )

        # Contrastive learning components (optional)
        if self.use_contrastive_loss:
            self.readout = Readout()
            self.discriminator = Discriminator(self.hidden_dim)
            self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, batch):
        """
        Forward pass for traffic prediction.

        Args:
            batch: dict with 'X' tensor of shape (batch, time_in, num_nodes, features)

        Returns:
            predictions: (batch, output_window, num_nodes, output_dim)
        """
        x = batch['X']  # (batch, time_in, num_nodes, features)
        batch_size, time_steps, num_nodes, _ = x.shape

        # Project input
        x = self.input_proj(x)  # (batch, time, nodes, hidden_dim)

        # Process each time step with multi-view GCN
        h_list = []
        for t in range(time_steps):
            x_t = x[:, t, :, :]  # (batch, nodes, hidden_dim)
            h_adj, h_diff = self.encoder(x_t, self.adj_norm, self.diff_mx)
            # Concatenate two views
            h_t = torch.cat([h_adj, h_diff], dim=-1)  # (batch, nodes, hidden_dim*2)
            h_list.append(h_t)

        # Stack temporal features: (batch, time, nodes, hidden_dim*2)
        h = torch.stack(h_list, dim=1)

        # Reshape for temporal modeling: (batch*nodes, time, hidden_dim*2)
        h = h.permute(0, 2, 1, 3).contiguous()
        h = h.view(batch_size * num_nodes, time_steps, -1)

        # Temporal modeling
        if self.temporal_type == 'gru':
            h, _ = self.temporal(h)  # (batch*nodes, time, hidden_dim)
        else:
            h = self.temporal(h)  # (batch*nodes, time, hidden_dim)

        # Flatten temporal dimension
        h = h.contiguous().view(batch_size * num_nodes, -1)  # (batch*nodes, time*hidden_dim)

        # Prediction
        out = self.pred_head(h)  # (batch*nodes, output_window*output_dim)
        out = out.view(batch_size, num_nodes, self.output_window, self.output_dim)
        out = out.permute(0, 2, 1, 3)  # (batch, output_window, num_nodes, output_dim)

        return out

    def _compute_contrastive_loss(self, x):
        """
        Compute contrastive loss between two views.

        Args:
            x: input features (batch, nodes, features)

        Returns:
            contrastive loss tensor
        """
        batch_size = x.shape[0]

        # Get embeddings from both views
        h_adj, h_diff = self.encoder(x, self.adj_norm, self.diff_mx)

        # Graph-level representations
        g_adj = self.sigmoid(self.readout(h_adj))
        g_diff = self.sigmoid(self.readout(h_diff))

        # Create corrupted (shuffled) input for negative samples
        idx = torch.randperm(batch_size)
        x_shuf = x[idx]
        h_adj_neg, h_diff_neg = self.encoder(x_shuf, self.adj_norm, self.diff_mx)

        # Discriminator scores
        # Positive: local from view1 with global from view2
        pos_1 = self.discriminator(h_adj, g_diff)  # (batch, nodes)
        pos_2 = self.discriminator(h_diff, g_adj)  # (batch, nodes)

        # Negative: corrupted local with global
        neg_1 = self.discriminator(h_adj_neg, g_diff)
        neg_2 = self.discriminator(h_diff_neg, g_adj)

        # Binary cross-entropy loss
        ones = torch.ones_like(pos_1)
        zeros = torch.zeros_like(neg_1)

        loss = F.binary_cross_entropy_with_logits(pos_1, ones) + \
               F.binary_cross_entropy_with_logits(pos_2, ones) + \
               F.binary_cross_entropy_with_logits(neg_1, zeros) + \
               F.binary_cross_entropy_with_logits(neg_2, zeros)

        return loss / 4.0

    def calculate_loss(self, batch):
        """
        Calculate training loss.

        Args:
            batch: dict with 'X' and 'y' tensors

        Returns:
            loss tensor
        """
        y_true = batch['y']  # (batch, output_window, num_nodes, features)
        y_pred = self.forward(batch)  # (batch, output_window, num_nodes, output_dim)

        # Inverse transform for loss computation
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_pred = self._scaler.inverse_transform(y_pred[..., :self.output_dim])

        # Main prediction loss
        pred_loss = loss.masked_mse_torch(y_pred, y_true)

        # Optional contrastive auxiliary loss
        if self.use_contrastive_loss and self.training:
            x = batch['X']
            # Use middle time step for contrastive loss
            mid_t = x.shape[1] // 2
            x_mid = self.input_proj(x[:, mid_t, :, :])
            cont_loss = self._compute_contrastive_loss(x_mid)
            total_loss = pred_loss + self.contrastive_weight * cont_loss
            return total_loss

        return pred_loss

    def predict(self, batch):
        """
        Generate predictions.

        Args:
            batch: dict with 'X' tensor

        Returns:
            predictions: (batch, output_window, num_nodes, output_dim)
        """
        return self.forward(batch)
