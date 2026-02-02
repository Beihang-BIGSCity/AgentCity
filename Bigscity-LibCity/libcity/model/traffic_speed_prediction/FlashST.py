"""
FlashST: A Universal Framework for Spatio-Temporal Prediction

This is an adaptation of the FlashST model for the LibCity framework.
FlashST uses a prompt-based learning approach for spatio-temporal prediction,
with learnable spatial and temporal embeddings that can be adapted to new datasets.

Original source: repos/FlashST/model/FlashST.py, repos/FlashST/model/PromptNet.py

Key Features:
- PromptNet for learning spatio-temporal embeddings
- GCN layers for graph-based spatial modeling
- MLP layers for feature transformation
- Laplacian positional encoding for spatial information
- Time-of-day and day-of-week temporal embeddings

Key Changes from Original:
1. Inherit from AbstractTrafficStateModel instead of nn.Module
2. Replace hardcoded .cuda() calls with self.device
3. Handle LibCity batch format (dict with 'X' and 'y' keys)
4. Implement calculate_loss and predict methods
5. Build standalone predictor instead of wrapping external models
6. Compute Laplacian positional encoding from adjacency matrix

Paper: FlashST: A Simple and Universal Prompt-Tuning Framework for Traffic Prediction
"""

from logging import getLogger
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def calculate_normalized_laplacian(adj):
    """
    Calculate normalized Laplacian matrix.
    L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2

    Args:
        adj: Adjacency matrix (numpy array)

    Returns:
        Normalized Laplacian matrix
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return normalized_laplacian


def calculate_laplacian_positional_encoding(adj, pos_enc_dim):
    """
    Calculate Laplacian Positional Encoding for graph nodes.
    Uses eigenvectors of the normalized Laplacian matrix.

    Args:
        adj: Adjacency matrix (numpy array)
        pos_enc_dim: Dimension of positional encoding

    Returns:
        Positional encoding tensor of shape (num_nodes, pos_enc_dim)
    """
    # Calculate normalized Laplacian
    lap = calculate_normalized_laplacian(adj)

    # Convert to dense if sparse
    if sp.issparse(lap):
        lap = lap.toarray()

    num_nodes = adj.shape[0]

    # Compute eigenvalues and eigenvectors
    try:
        # Use sparse eigenvalue computation for efficiency
        if pos_enc_dim + 1 < num_nodes:
            eigenvalues, eigenvectors = linalg.eigsh(
                sp.csr_matrix(lap), k=pos_enc_dim + 1, which='SM', tol=1e-2
            )
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(lap)
    except Exception:
        # Fallback to dense computation
        eigenvalues, eigenvectors = np.linalg.eigh(lap)

    # Sort by eigenvalue and take first pos_enc_dim eigenvectors (skip first constant eigenvector)
    idx = eigenvalues.argsort()
    eigenvectors = eigenvectors[:, idx]

    # Skip first eigenvector (constant) and take pos_enc_dim eigenvectors
    if eigenvectors.shape[1] > pos_enc_dim:
        pos_enc = eigenvectors[:, 1:pos_enc_dim + 1]
    else:
        # Pad with zeros if not enough eigenvectors
        pos_enc = np.zeros((num_nodes, pos_enc_dim))
        available = min(eigenvectors.shape[1] - 1, pos_enc_dim)
        if available > 0:
            pos_enc[:, :available] = eigenvectors[:, 1:available + 1]

    return pos_enc.astype(np.float32)


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


class GCN(nn.Module):
    """Graph Convolutional Network layer with residual connection."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True
        )
        self.act = nn.LeakyReLU()

    def forward(self, input_data, nadj, use_gnn=True):
        """
        Forward pass of GCN layer.

        Args:
            input_data: Input tensor of shape (B, D, N, E)
            nadj: Normalized adjacency matrix of shape (N, N)
            use_gnn: Whether to use graph convolution

        Returns:
            Output tensor with residual connection
        """
        if use_gnn and nadj is not None:
            # Graph convolution: aggregate neighbor features
            gcn_out = self.act(torch.einsum('nk,bdke->bdne', nadj, self.fc1(input_data)))
        else:
            gcn_out = self.act(self.fc1(input_data))
        return gcn_out + input_data


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual connection."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data):
        """
        Forward pass with residual connection.

        Args:
            input_data: Input tensor

        Returns:
            Output tensor with residual connection
        """
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        return hidden + input_data


class PromptNet(nn.Module):
    """
    Prompt Network for FlashST.

    Learns spatio-temporal embeddings that can be adapted to new datasets
    through prompt-based learning.

    Args:
        num_nodes: Number of nodes in the graph
        input_len: Input sequence length
        embed_dim: Embedding dimension
        num_layer: Number of MLP layers in encoder
        node_dim: Dimension of spatial positional encoding
        temp_dim_tid: Dimension of time-in-day embedding
        temp_dim_diw: Dimension of day-in-week embedding
        input_base_dim: Base input dimension (usually 1 for traffic speed)
        if_time_in_day: Whether to use time-of-day embedding
        if_day_in_week: Whether to use day-of-week embedding
        if_spatial: Whether to use spatial positional encoding
        device: Computation device
    """

    def __init__(self, num_nodes, input_len, embed_dim, num_layer, node_dim,
                 temp_dim_tid, temp_dim_diw, input_base_dim,
                 if_time_in_day=True, if_day_in_week=True, if_spatial=True,
                 device=torch.device('cpu')):
        super().__init__()

        self.node_dim = node_dim
        self.input_len = input_len
        self.embed_dim = embed_dim
        self.num_layer = num_layer
        self.temp_dim_tid = temp_dim_tid
        self.temp_dim_diw = temp_dim_diw
        self.input_base_dim = input_base_dim
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week
        self.if_spatial = if_spatial
        self.device = device

        # Spatial embeddings (Laplacian positional encoding)
        if self.if_spatial:
            self.LaplacianPE1 = nn.Linear(self.node_dim, self.node_dim)
            self.LaplacianPE2 = nn.Linear(self.node_dim, self.node_dim)

        # Temporal embeddings
        # 288 = 24 * 60 / 5 (assuming 5-minute intervals for a full day)
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Embedding(289, self.temp_dim_tid)  # 288 + 1 for padding
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Embedding(8, self.temp_dim_diw)  # 7 + 1 for padding

        # Time series embedding layer - projects each time step's features
        # Changed from projecting temporal dim to projecting feature dim per timestep
        self.time_series_emb_layer = nn.Linear(self.input_base_dim, self.embed_dim, bias=True)

        # Calculate hidden dimension
        self.hidden_dim = (
            self.embed_dim +
            self.node_dim * int(self.if_spatial) +
            self.temp_dim_tid * int(self.if_day_in_week) +
            self.temp_dim_diw * int(self.if_time_in_day)
        )

        # Encoder layers
        self.encoder1 = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )
        self.encoder2 = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )

        # GCN layers
        self.gcn1 = GCN(self.hidden_dim)
        self.gcn2 = GCN(self.hidden_dim)

        self.act = nn.LeakyReLU()

    def forward(self, history_data, source_full, nadj=None, lpls=None, use_gnn=True):
        """
        Forward pass of PromptNet.

        Args:
            history_data: Historical traffic data, shape (B, T, N, D_base)
            source_full: Full input data including temporal features, shape (B, T, N, D_full)
            nadj: Normalized adjacency matrix, shape (N, N)
            lpls: Laplacian positional encoding, shape (N, node_dim)
            use_gnn: Whether to use GCN layers

        Returns:
            Prompt embeddings, shape (B, T, N, hidden_dim)
        """
        input_data = history_data
        batch_size, _, num_nodes, _ = input_data.shape

        # Get temporal embeddings
        time_in_day_emb = None
        day_in_week_emb = None

        if self.if_time_in_day:
            # Time-of-day is assumed to be at index input_base_dim (normalized 0-1)
            # Convert to integer index (0-287)
            t_i_d_data = source_full[:, 0, :, self.input_base_dim]  # (B, N)
            # Scale from 0-1 to 0-287 and convert to int
            t_i_d_indices = (t_i_d_data * 288).long().clamp(0, 288)
            time_in_day_emb = self.time_in_day_emb(t_i_d_indices)  # (B, N, temp_dim_tid)

        if self.if_day_in_week:
            # Day-of-week is assumed to be at index input_base_dim + 1
            d_i_w_data = source_full[:, 0, :, self.input_base_dim + 1]  # (B, N)
            d_i_w_indices = d_i_w_data.long().clamp(0, 7)
            day_in_week_emb = self.day_in_week_emb(d_i_w_indices)  # (B, N, temp_dim_diw)

        # Time series embedding: project feature dimension per timestep
        # input_data: (B, T, N, D) -> project D -> (B, T, N, embed_dim)
        time_series_emb = self.time_series_emb_layer(
            input_data[..., :self.input_base_dim]
        )  # (B, T, N, embed_dim)

        # Spatial embedding (Laplacian positional encoding)
        node_emb = []
        if self.if_spatial and lpls is not None:
            lap_pos_enc = self.LaplacianPE2(self.act(self.LaplacianPE1(lpls)))  # (N, node_dim)
            tensor_neb = lap_pos_enc.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, node_dim)
            # Expand to match temporal dimension: (B, T, N, node_dim)
            tensor_neb = tensor_neb.unsqueeze(1).expand(-1, self.input_len, -1, -1)
            node_emb.append(tensor_neb)

        # Temporal embeddings - expand to match temporal dimension
        tem_emb = []
        if time_in_day_emb is not None:
            # (B, N, temp_dim_tid) -> (B, T, N, temp_dim_tid)
            tem_emb.append(time_in_day_emb.unsqueeze(1).expand(-1, self.input_len, -1, -1))
        if day_in_week_emb is not None:
            # (B, N, temp_dim_diw) -> (B, T, N, temp_dim_diw)
            tem_emb.append(day_in_week_emb.unsqueeze(1).expand(-1, self.input_len, -1, -1))

        # Concatenate all embeddings along feature dimension
        # time_series_emb: (B, T, N, embed_dim)
        # node_emb: list of (B, T, N, node_dim)
        # tem_emb: list of (B, T, N, temp_dim)
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=-1)  # (B, T, N, hidden_dim)
        hidden = hidden.permute(0, 3, 2, 1)  # (B, hidden_dim, N, T)

        # Encoding with GCN and MLP layers
        hidden_gcn = self.gcn1(hidden, nadj, use_gnn)
        hidden = self.encoder1(hidden_gcn)
        hidden_gcn = self.gcn2(hidden, nadj, use_gnn)
        hidden = self.encoder2(hidden_gcn)

        # Transform back: (B, hidden_dim, N, T) -> (B, T, N, hidden_dim)
        x_prompt = hidden.permute(0, 3, 2, 1)  # (B, T, N, hidden_dim)
        x_prompt = F.normalize(x_prompt, dim=-1)

        return x_prompt  # (B, T, N, hidden_dim)


class SimplePredictor(nn.Module):
    """
    Simple predictor network for traffic prediction.

    Uses temporal convolutions and MLPs to predict future traffic values
    from the prompt embeddings.

    Args:
        input_dim: Input feature dimension (from PromptNet)
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (usually 1 for traffic speed)
        input_window: Input sequence length
        output_window: Output sequence length
        num_nodes: Number of nodes
    """

    def __init__(self, input_dim, hidden_dim, output_dim, input_window, output_window, num_nodes):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_window = input_window
        self.output_window = output_window
        self.num_nodes = num_nodes

        # Temporal embedding
        self.temporal_conv = nn.Conv2d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=(1, input_window),
            padding=0
        )

        # Spatial mixing
        self.spatial_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Output projection
        self.output_conv = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=output_window * output_dim,
            kernel_size=(1, 1)
        )

    def forward(self, x):
        """
        Forward pass of predictor.

        Args:
            x: Input from PromptNet, shape (B, T, N, D)

        Returns:
            Predictions, shape (B, T_out, N, D_out)
        """
        batch_size = x.shape[0]

        # (B, T, N, D) -> (B, D, N, T)
        x = x.permute(0, 3, 2, 1)

        # Temporal convolution: (B, D, N, T) -> (B, hidden, N, 1)
        x = self.temporal_conv(x)
        x = F.relu(x)

        # Spatial MLP: (B, hidden, N, 1) -> (B, N, hidden) -> (B, N, hidden) -> (B, hidden, N, 1)
        x = x.squeeze(-1).permute(0, 2, 1)  # (B, N, hidden)
        x = self.spatial_mlp(x)
        x = x.permute(0, 2, 1).unsqueeze(-1)  # (B, hidden, N, 1)

        # Output projection: (B, hidden, N, 1) -> (B, T_out * D_out, N, 1)
        x = self.output_conv(x)

        # Reshape to (B, T_out, N, D_out)
        x = x.squeeze(-1)  # (B, T_out * D_out, N)
        x = x.permute(0, 2, 1)  # (B, N, T_out * D_out)
        x = x.reshape(batch_size, self.num_nodes, self.output_window, self.output_dim)
        x = x.permute(0, 2, 1, 3)  # (B, T_out, N, D_out)

        return x


class FlashST(AbstractTrafficStateModel):
    """
    FlashST: A Universal Framework for Spatio-Temporal Prediction.

    This model uses prompt-based learning to adapt to traffic prediction tasks.
    It combines spatial (Laplacian positional encoding) and temporal (time-of-day,
    day-of-week) embeddings with a learnable prompt network.

    Args:
        config: Configuration dictionary containing model hyperparameters
        data_feature: Data feature dictionary containing dataset information

    Configuration Parameters:
        - input_window: Number of input time steps (default: 12)
        - output_window: Number of output time steps (default: 12)
        - embed_dim: Embedding dimension for time series (default: 32)
        - hidden_dim: Hidden dimension for predictor (default: 64)
        - num_layer: Number of MLP layers in PromptNet encoder (default: 2)
        - node_dim: Dimension of spatial positional encoding (default: 32)
        - temp_dim_tid: Dimension of time-in-day embedding (default: 32)
        - temp_dim_diw: Dimension of day-in-week embedding (default: 32)
        - if_time_in_day: Whether to use time-of-day embedding (default: True)
        - if_day_in_week: Whether to use day-of-week embedding (default: True)
        - if_spatial: Whether to use spatial positional encoding (default: True)
        - use_gnn: Whether to use GCN layers (default: True)
        - dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        # Logger
        self._logger = getLogger()

        # Data features
        self.adj_mx = data_feature.get('adj_mx')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)
        self._scaler = data_feature.get('scaler')

        # Configuration parameters
        self.device = config.get('device', torch.device('cpu'))
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)

        # Model hyperparameters
        self.embed_dim = config.get('embed_dim', 32)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.num_layer = config.get('num_layer', 2)
        self.node_dim = config.get('node_dim', 32)
        self.temp_dim_tid = config.get('temp_dim_tid', 32)
        self.temp_dim_diw = config.get('temp_dim_diw', 32)
        self.input_base_dim = config.get('input_base_dim', 1)
        self.if_time_in_day = config.get('if_time_in_day', True)
        self.if_day_in_week = config.get('if_day_in_week', True)
        self.if_spatial = config.get('if_spatial', True)
        self.use_gnn = config.get('use_gnn', True)
        self.dropout = config.get('dropout', 0.1)

        # Calculate prompt dimension (output of PromptNet)
        self.prompt_dim = (
            self.embed_dim +
            self.node_dim * int(self.if_spatial) +
            self.temp_dim_tid * int(self.if_day_in_week) +
            self.temp_dim_diw * int(self.if_time_in_day)
        )

        # Initialize PromptNet
        self.prompt_net = PromptNet(
            num_nodes=self.num_nodes,
            input_len=self.input_window,
            embed_dim=self.embed_dim,
            num_layer=self.num_layer,
            node_dim=self.node_dim,
            temp_dim_tid=self.temp_dim_tid,
            temp_dim_diw=self.temp_dim_diw,
            input_base_dim=self.input_base_dim,
            if_time_in_day=self.if_time_in_day,
            if_day_in_week=self.if_day_in_week,
            if_spatial=self.if_spatial,
            device=self.device
        )

        # Initialize Predictor
        self.predictor = SimplePredictor(
            input_dim=self.prompt_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            input_window=self.input_window,  # Use temporal window size (e.g., 12)
            output_window=self.output_window,
            num_nodes=self.num_nodes
        )

        # Prepare adjacency matrix and Laplacian positional encoding
        self._init_graph_features()

        self._logger.info(f'FlashST initialized with {self.num_nodes} nodes, '
                         f'prompt_dim={self.prompt_dim}, hidden_dim={self.hidden_dim}')

    def _init_graph_features(self):
        """Initialize graph-related features: normalized adjacency and Laplacian PE."""
        if self.adj_mx is not None:
            # Normalize adjacency matrix
            nadj = sym_adj(self.adj_mx + np.eye(self.num_nodes))
            self.register_buffer('nadj', torch.tensor(nadj, dtype=torch.float32))

            # Compute Laplacian positional encoding
            lpls = calculate_laplacian_positional_encoding(self.adj_mx, self.node_dim)
            self.register_buffer('lpls', torch.tensor(lpls, dtype=torch.float32))
        else:
            # Use identity matrix if no adjacency provided
            self.register_buffer('nadj', torch.eye(self.num_nodes, dtype=torch.float32))
            # Use random positional encoding if no graph
            self.register_buffer('lpls', torch.randn(self.num_nodes, self.node_dim))

    def forward(self, batch):
        """
        Forward pass of FlashST.

        Args:
            batch: Dictionary containing:
                - 'X': Input tensor of shape (B, T, N, D)
                - 'y': Target tensor of shape (B, T_out, N, D_out)

        Returns:
            Predictions of shape (B, T_out, N, D_out)
        """
        x = batch['X']  # (B, T, N, D)

        # Extract base input (traffic values only)
        x_base = x[..., :self.input_base_dim]  # (B, T, N, D_base)

        # Get prompt embeddings from PromptNet
        x_prompt = self.prompt_net(
            history_data=x_base,
            source_full=x,
            nadj=self.nadj,
            lpls=self.lpls,
            use_gnn=self.use_gnn
        )  # (B, T, N, prompt_dim)

        # Predict future values - x_prompt is already in (B, T, N, D) format
        predictions = self.predictor(x_prompt)  # (B, T_out, N, D_out)

        return predictions

    def predict(self, batch):
        """
        Generate predictions for a batch.

        Args:
            batch: Input batch dictionary

        Returns:
            Predictions with shape (B, T_out, N, D_out)
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate the masked MAE loss.

        Args:
            batch: Dictionary containing 'X' and 'y'

        Returns:
            Scalar loss tensor
        """
        y_true = batch['y']
        y_predicted = self.predict(batch)

        # Inverse transform to original scale
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        return loss.masked_mae_torch(y_predicted, y_true, null_val=0.0)
