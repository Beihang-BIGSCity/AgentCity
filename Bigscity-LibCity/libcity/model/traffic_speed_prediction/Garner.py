"""
Garner Model Adapted for LibCity Framework

Original Paper: Garner - Multi-view Graph Representation Learning for Road Networks
Original Repository: repos/Garner

Adaptations Made:
1. Converted from self-supervised contrastive learning to supervised traffic prediction
2. Added temporal encoding for time-series traffic data
3. Replaced DGL-based graph operations with PyTorch-based adjacency operations
4. Adapted input format from OSM road features to LibCity's batch dict format
5. Implemented forward(), predict(), calculate_loss() as required by LibCity
6. Made SVI embeddings optional (focus on road network structure)
7. Maintained the multi-view graph approach (original + diffusion graphs)

Key Components:
- Multi-view GNN encoders (original graph, diffusion graph, optional similarity graph)
- Temporal attention for time-series modeling
- Spatio-temporal feature fusion
- MSE/MAE loss for traffic prediction

Required Config Parameters:
- hidden_dim: Hidden dimension for GNN encoders (default: 64)
- num_gnn_layers: Number of GNN layers (default: 2)
- diffusion_k: Number of diffusion propagation steps (default: 10)
- diffusion_alpha: Diffusion propagation factor (default: 0.2)
- diffusion_epsilon: Edge mask threshold for diffusion graph (default: 0.01)
- use_similarity_graph: Whether to use the third similarity graph view (default: False)
- temporal_hidden_dim: Hidden dimension for temporal modeling (default: 64)
- num_temporal_layers: Number of temporal attention layers (default: 2)
- dropout: Dropout rate (default: 0.1)
"""

import math
import numpy as np
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class GraphConvolution(nn.Module):
    """
    Simple GCN layer adapted from DGL's GraphConv
    Supports weighted edges via edge_weight parameter
    """
    def __init__(self, in_features, out_features, bias=True, activation=None, norm='both'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.norm = norm
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, x, edge_weight=None):
        """
        Args:
            adj: Adjacency matrix (num_nodes, num_nodes)
            x: Node features (batch, num_nodes, in_features) or (num_nodes, in_features)
            edge_weight: Optional edge weights (num_nodes, num_nodes)
        Returns:
            Node embeddings with same batch dimensions
        """
        # Handle batched input
        if x.dim() == 3:
            batch_size = x.size(0)
            # x: (batch, num_nodes, in_features)
            support = torch.matmul(x, self.weight)  # (batch, num_nodes, out_features)

            # Apply edge weights if provided
            if edge_weight is not None:
                adj = adj * edge_weight

            # Normalize adjacency
            if self.norm == 'both':
                adj = self._normalize_adj(adj)
            elif self.norm == 'right':
                adj = self._row_normalize(adj)

            # Expand adj for batch matmul
            adj_expanded = adj.unsqueeze(0).expand(batch_size, -1, -1)
            output = torch.matmul(adj_expanded, support)  # (batch, num_nodes, out_features)
        else:
            # x: (num_nodes, in_features)
            support = torch.matmul(x, self.weight)

            if edge_weight is not None:
                adj = adj * edge_weight

            if self.norm == 'both':
                adj = self._normalize_adj(adj)
            elif self.norm == 'right':
                adj = self._row_normalize(adj)

            output = torch.matmul(adj, support)

        if self.bias is not None:
            output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output

    def _normalize_adj(self, adj):
        """Symmetric normalization: D^(-1/2) A D^(-1/2)"""
        rowsum = adj.sum(dim=-1)
        d_inv_sqrt = torch.pow(rowsum + 1e-8, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    def _row_normalize(self, adj):
        """Row normalization: D^(-1) A"""
        rowsum = adj.sum(dim=-1)
        d_inv = torch.pow(rowsum + 1e-8, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat = torch.diag(d_inv)
        return torch.mm(d_mat, adj)


class MultiViewGNNEncoder(nn.Module):
    """
    Multi-view GNN encoder inspired by Garner's MVGRL architecture.
    Uses multiple graph views (original, diffusion, similarity) to learn
    comprehensive node representations.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2,
                 use_similarity_graph=False, dropout=0.1):
        super(MultiViewGNNEncoder, self).__init__()
        self.num_layers = num_layers
        self.use_similarity_graph = use_similarity_graph

        # Encoder 1: Original graph view
        self.encoder1_layers = nn.ModuleList()
        self.encoder1_layers.append(
            GraphConvolution(in_dim, hidden_dim, activation=nn.PReLU(), norm='both')
        )
        for _ in range(num_layers - 1):
            self.encoder1_layers.append(
                GraphConvolution(hidden_dim, hidden_dim, activation=nn.PReLU(), norm='both')
            )

        # Encoder 2: Diffusion graph view
        self.encoder2_layers = nn.ModuleList()
        self.encoder2_layers.append(
            GraphConvolution(in_dim, hidden_dim, activation=nn.PReLU(), norm='none')
        )
        for _ in range(num_layers - 1):
            self.encoder2_layers.append(
                GraphConvolution(hidden_dim, hidden_dim, activation=nn.PReLU(), norm='none')
            )

        # Encoder 3: Similarity graph view (optional)
        if use_similarity_graph:
            self.encoder3_layers = nn.ModuleList()
            self.encoder3_layers.append(
                GraphConvolution(in_dim, hidden_dim, activation=nn.PReLU(), norm='both')
            )
            for _ in range(num_layers - 1):
                self.encoder3_layers.append(
                    GraphConvolution(hidden_dim, hidden_dim, activation=nn.PReLU(), norm='both')
                )

        # Output projection
        if use_similarity_graph:
            self.output_proj = nn.Linear(hidden_dim * 3, out_dim)
        else:
            self.output_proj = nn.Linear(hidden_dim * 2, out_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x, adj, diff_adj, diff_weight=None, sim_adj=None):
        """
        Args:
            x: Node features (batch, num_nodes, in_features) or (num_nodes, in_features)
            adj: Original adjacency matrix (num_nodes, num_nodes)
            diff_adj: Diffusion adjacency matrix (num_nodes, num_nodes)
            diff_weight: Optional diffusion edge weights
            sim_adj: Optional similarity adjacency matrix
        Returns:
            Node embeddings combining all graph views
        """
        # Encoder 1: Original graph
        h1 = x
        for layer in self.encoder1_layers:
            h1 = layer(adj, h1)
            h1 = self.dropout(h1)

        # Encoder 2: Diffusion graph
        h2 = x
        for layer in self.encoder2_layers:
            h2 = layer(diff_adj, h2, edge_weight=diff_weight)
            h2 = self.dropout(h2)

        # Combine views
        if self.use_similarity_graph and sim_adj is not None:
            # Encoder 3: Similarity graph
            h3 = x
            for layer in self.encoder3_layers:
                h3 = layer(sim_adj, h3)
                h3 = self.dropout(h3)
            h_combined = torch.cat([h1, h2, h3], dim=-1)
        else:
            h_combined = torch.cat([h1, h2], dim=-1)

        # Output projection
        output = self.output_proj(h_combined)
        output = self.layer_norm(output)

        return output


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for modeling temporal dependencies
    in traffic time series data.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, time, num_nodes, hidden_dim)
        Returns:
            Temporal-attended features: (batch, time, num_nodes, hidden_dim)
        """
        batch_size, time_steps, num_nodes, hidden_dim = x.shape

        # Reshape for temporal attention: (batch * num_nodes, time, hidden_dim)
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, time_steps, hidden_dim)

        # Compute Q, K, V
        q = self.query(x_reshaped).view(batch_size * num_nodes, time_steps, self.num_heads, self.head_dim)
        k = self.key(x_reshaped).view(batch_size * num_nodes, time_steps, self.num_heads, self.head_dim)
        v = self.value(x_reshaped).view(batch_size * num_nodes, time_steps, self.num_heads, self.head_dim)

        # Transpose for attention: (batch * num_nodes, num_heads, time, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch * num_nodes, num_heads, time, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, time_steps, hidden_dim)
        out = self.out_proj(out)

        # Residual connection and layer norm
        out = self.layer_norm(x_reshaped + out)

        # Reshape back: (batch, time, num_nodes, hidden_dim)
        out = out.view(batch_size, num_nodes, time_steps, hidden_dim).permute(0, 2, 1, 3)

        return out


class TemporalEncoder(nn.Module):
    """
    Temporal encoder using stacked temporal attention layers
    for capturing temporal patterns in traffic data.
    """
    def __init__(self, hidden_dim, num_layers=2, num_heads=4, dropout=0.1):
        super(TemporalEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TemporalAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, time, num_nodes, hidden_dim)
        Returns:
            Temporally encoded features: (batch, time, num_nodes, hidden_dim)
        """
        for layer in self.layers:
            x = layer(x)

        # Apply FFN with residual
        out = self.ffn(x)
        out = self.layer_norm(x + out)

        return out


class SpatioTemporalFusion(nn.Module):
    """
    Fuses spatial (graph) and temporal representations
    for traffic prediction.
    """
    def __init__(self, spatial_dim, temporal_dim, hidden_dim, dropout=0.1):
        super(SpatioTemporalFusion, self).__init__()
        self.spatial_proj = nn.Linear(spatial_dim, hidden_dim)
        self.temporal_proj = nn.Linear(temporal_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, spatial_feat, temporal_feat):
        """
        Args:
            spatial_feat: (batch, time, num_nodes, spatial_dim)
            temporal_feat: (batch, time, num_nodes, temporal_dim)
        Returns:
            Fused features: (batch, time, num_nodes, hidden_dim)
        """
        s = self.spatial_proj(spatial_feat)
        t = self.temporal_proj(temporal_feat)

        combined = torch.cat([s, t], dim=-1)
        fused = self.fusion(combined)
        fused = self.layer_norm(fused)

        return fused


class Garner(AbstractTrafficStateModel):
    """
    Garner: Multi-view Graph Representation Learning for Traffic Speed Prediction

    This is an adaptation of the Garner model for traffic prediction tasks
    within the LibCity framework. The original Garner model was designed for
    self-supervised road representation learning using contrastive learning.

    This adaptation:
    - Converts to supervised traffic prediction
    - Adds temporal modeling for time-series data
    - Uses LibCity's data format and conventions
    - Maintains the multi-view graph approach

    Input shape: (batch_size, input_window, num_nodes, feature_dim)
    Output shape: (batch_size, output_window, num_nodes, output_dim)
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        # Data features
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self._scaler = self.data_feature.get('scaler')
        self._logger = getLogger()

        # Model hyperparameters
        self.hidden_dim = config.get('hidden_dim', 64)
        self.num_gnn_layers = config.get('num_gnn_layers', 2)
        self.diffusion_k = config.get('diffusion_k', 10)
        self.diffusion_alpha = config.get('diffusion_alpha', 0.2)
        self.diffusion_epsilon = config.get('diffusion_epsilon', 0.01)
        self.use_similarity_graph = config.get('use_similarity_graph', False)
        self.temporal_hidden_dim = config.get('temporal_hidden_dim', 64)
        self.num_temporal_layers = config.get('num_temporal_layers', 2)
        self.num_heads = config.get('num_heads', 4)
        self.dropout = config.get('dropout', 0.1)
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.device = config.get('device', torch.device('cpu'))

        # Get adjacency matrix and create graphs
        adj_mx = data_feature.get('adj_mx')
        if adj_mx is not None:
            self.adj = self._create_adjacency(adj_mx)
            self.diff_adj, self.diff_weight = self._create_diffusion_graph(adj_mx)
        else:
            self._logger.warning('No adjacency matrix provided, using identity matrix')
            self.adj = torch.eye(self.num_nodes).to(self.device)
            self.diff_adj = torch.eye(self.num_nodes).to(self.device)
            self.diff_weight = None

        self._logger.info(f'Garner initialized with num_nodes={self.num_nodes}, '
                         f'hidden_dim={self.hidden_dim}, num_gnn_layers={self.num_gnn_layers}')

        # Input projection
        self.input_proj = nn.Linear(self.feature_dim, self.hidden_dim)

        # Multi-view GNN encoder (spatial)
        self.spatial_encoder = MultiViewGNNEncoder(
            in_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            num_layers=self.num_gnn_layers,
            use_similarity_graph=self.use_similarity_graph,
            dropout=self.dropout
        )

        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_temporal_layers,
            num_heads=self.num_heads,
            dropout=self.dropout
        )

        # Spatio-temporal fusion
        self.st_fusion = SpatioTemporalFusion(
            spatial_dim=self.hidden_dim,
            temporal_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        )

        # Output projection
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

        # For multi-step prediction
        self.prediction_head = nn.Linear(self.input_window, self.output_window)

    def _create_adjacency(self, adj_mx):
        """Convert adjacency matrix to torch tensor and add self-loops."""
        if isinstance(adj_mx, np.ndarray):
            adj = torch.FloatTensor(adj_mx)
        else:
            adj = adj_mx.clone().float()

        # Add self-loops
        adj = adj + torch.eye(self.num_nodes)

        # Normalize
        rowsum = adj.sum(dim=1)
        d_inv_sqrt = torch.pow(rowsum + 1e-8, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

        return adj.to(self.device)

    def _create_diffusion_graph(self, adj_mx):
        """
        Create diffusion graph using APPNP-style propagation.
        This implements the graph diffusion from the original Garner paper.
        """
        if isinstance(adj_mx, np.ndarray):
            adj = torch.FloatTensor(adj_mx)
        else:
            adj = adj_mx.clone().float()

        n = adj.shape[0]

        # Add self-loops
        adj = adj + torch.eye(n)

        # Row normalize: D^-1 * A
        rowsum = adj.sum(dim=1)
        d_inv = torch.pow(rowsum + 1e-8, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat = torch.diag(d_inv)
        norm_adj = torch.mm(d_mat, adj)

        # APPNP-style diffusion: (1-alpha) * (I - alpha * A_norm)^-1
        # Approximate using power iteration
        alpha = self.diffusion_alpha
        k = self.diffusion_k

        # Power iteration approximation
        diff = torch.eye(n)
        prop = torch.eye(n)
        for _ in range(k):
            prop = (1 - alpha) * torch.mm(norm_adj, prop) + alpha * torch.eye(n)
            diff = diff + prop
        diff = diff / (k + 1)

        # Apply threshold
        epsilon = self.diffusion_epsilon
        diff_weight = diff.clone()
        diff[diff < epsilon] = 0

        # Normalize weights
        max_val = diff_weight.max()
        if max_val > 0:
            diff_weight = diff_weight / max_val

        diff_adj = (diff > 0).float()

        return diff_adj.to(self.device), diff_weight.to(self.device)

    def forward(self, batch):
        """
        Forward pass for traffic speed prediction.

        Args:
            batch: Dictionary containing:
                - 'X': Input tensor of shape (batch_size, input_window, num_nodes, feature_dim)

        Returns:
            Predictions of shape (batch_size, output_window, num_nodes, output_dim)
        """
        x = batch['X']  # (batch_size, input_window, num_nodes, feature_dim)
        batch_size, time_steps, num_nodes, feature_dim = x.shape

        # Input projection
        x = self.input_proj(x)  # (batch_size, input_window, num_nodes, hidden_dim)

        # Apply spatial encoding for each time step
        spatial_out = []
        for t in range(time_steps):
            x_t = x[:, t, :, :]  # (batch_size, num_nodes, hidden_dim)
            h_t = self.spatial_encoder(x_t, self.adj, self.diff_adj, self.diff_weight)
            spatial_out.append(h_t)

        spatial_feat = torch.stack(spatial_out, dim=1)  # (batch_size, input_window, num_nodes, hidden_dim)

        # Apply temporal encoding
        temporal_feat = self.temporal_encoder(x)  # (batch_size, input_window, num_nodes, hidden_dim)

        # Fuse spatial and temporal features
        fused = self.st_fusion(spatial_feat, temporal_feat)  # (batch_size, input_window, num_nodes, hidden_dim)

        # Output projection
        out = self.output_layer(fused)  # (batch_size, input_window, num_nodes, output_dim)

        # Multi-step prediction: project from input_window to output_window
        out = out.permute(0, 2, 3, 1)  # (batch_size, num_nodes, output_dim, input_window)
        out = self.prediction_head(out)  # (batch_size, num_nodes, output_dim, output_window)
        out = out.permute(0, 3, 1, 2)  # (batch_size, output_window, num_nodes, output_dim)

        return out

    def calculate_loss(self, batch):
        """
        Calculate the training loss.

        Args:
            batch: Dictionary containing 'X' and 'y'

        Returns:
            Scalar loss tensor
        """
        y_true = batch['y']  # (batch_size, output_window, num_nodes, feature_dim)
        y_predicted = self.forward(batch)  # (batch_size, output_window, num_nodes, output_dim)

        # Inverse transform for loss calculation
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        return loss.masked_mse_torch(y_predicted, y_true)

    def predict(self, batch):
        """
        Generate predictions for the input batch.

        Args:
            batch: Dictionary containing 'X'

        Returns:
            Predictions of shape (batch_size, output_window, num_nodes, output_dim)
        """
        return self.forward(batch)
