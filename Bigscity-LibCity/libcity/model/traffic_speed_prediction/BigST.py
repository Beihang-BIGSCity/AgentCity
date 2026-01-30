"""
BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting

This is a LibCity-adapted implementation of the BigST model.
Original paper: "BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks"

Key adaptations from original implementation:
- Inherits from AbstractTrafficStateModel
- Accepts LibCity's config and data_feature parameters
- Implements required methods: forward, predict, calculate_loss
- Handles LibCity's batch format (batch['X'], batch['y'])
- Uses LibCity's scaler from data_feature
- Replaced deprecated torch.qr with torch.linalg.qr
- Input shape adapted from (B, N, T, D) to LibCity's (B, T, N, D)
"""

import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def create_products_of_givens_rotations(dim, seed):
    """
    Create a random rotation matrix using products of Givens rotations.

    Args:
        dim: Dimension of the rotation matrix
        seed: Random seed for reproducibility

    Returns:
        torch.Tensor: Rotation matrix of shape (dim, dim)
    """
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    np.random.seed(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(dim, 2)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.cos(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return torch.tensor(q, dtype=torch.float32)


def create_random_matrix(m, d, seed=0, scaling=0, struct_mode=False):
    """
    Create a random projection matrix for random feature approximation.

    Args:
        m: Number of random features
        d: Input dimension
        seed: Random seed
        scaling: Scaling mode (0 or 1)
        struct_mode: Whether to use structured random matrices

    Returns:
        torch.Tensor: Random projection matrix of shape (m, d)
    """
    nb_full_blocks = int(m / d)
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            # Updated: Use torch.linalg.qr instead of deprecated torch.qr
            q, _ = torch.linalg.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q)
        current_seed += 1
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            # Updated: Use torch.linalg.qr instead of deprecated torch.qr
            q, _ = torch.linalg.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q[0:remaining_rows])
    final_matrix = torch.vstack(block_list)

    current_seed += 1
    torch.manual_seed(current_seed)
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d)), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

    return torch.matmul(torch.diag(multiplier), final_matrix)


def random_feature_map(data, is_query, projection_matrix=None, numerical_stabilizer=0.000001):
    """
    Apply random feature map for linearized attention approximation.

    Args:
        data: Input tensor of shape (B, N, H, D)
        is_query: Whether this is the query tensor
        projection_matrix: Random projection matrix
        numerical_stabilizer: Small constant for numerical stability

    Returns:
        torch.Tensor: Transformed features
    """
    data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32)))
    data = data_normalizer * data
    ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32))
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)
    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=len(data.shape) - 1)
    diag_data = diag_data / 2.0
    diag_data = torch.unsqueeze(diag_data, dim=len(data.shape) - 1)
    last_dims_t = len(data_dash.shape) - 1
    attention_dims_t = len(data_dash.shape) - 3
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash, dim=last_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                    dim=attention_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )
    return data_dash


def linear_kernel(x, node_vec1, node_vec2):
    """
    Compute linear kernel attention with random features.

    Args:
        x: Input tensor of shape (B, N, 1, nhid)
        node_vec1: Query features of shape (B, N, 1, r)
        node_vec2: Key features of shape (B, N, 1, r)

    Returns:
        torch.Tensor: Output of shape (B, N, 1, nhid)
    """
    # x: [B, N, 1, nhid] node_vec1: [B, N, 1, r], node_vec2: [B, N, 1, r]
    node_vec1 = node_vec1.permute(1, 0, 2, 3)  # [N, B, 1, r]
    node_vec2 = node_vec2.permute(1, 0, 2, 3)  # [N, B, 1, r]
    x = x.permute(1, 0, 2, 3)  # [N, B, 1, nhid]

    v2x = torch.einsum("nbhm,nbhd->bhmd", node_vec2, x)
    out1 = torch.einsum("nbhm,bhmd->nbhd", node_vec1, v2x)  # [N, B, 1, nhid]

    one_matrix = torch.ones([node_vec2.shape[0]]).to(node_vec1.device)
    node_vec2_sum = torch.einsum("nbhm,n->bhm", node_vec2, one_matrix)
    out2 = torch.einsum("nbhm,bhm->nbh", node_vec1, node_vec2_sum)  # [N, 1]

    out1 = out1.permute(1, 0, 2, 3)  # [B, N, 1, nhid]
    out2 = out2.permute(1, 0, 2)
    out2 = torch.unsqueeze(out2, len(out2.shape))
    out = out1 / out2  # [B, N, 1, nhid]

    return out


class ConvApproximation(nn.Module):
    """
    Convolution approximation layer using random features for linear complexity.

    This layer approximates the attention mechanism using random feature maps
    to achieve O(N) complexity instead of O(N^2).
    """

    def __init__(self, dropout, tau, random_feature_dim):
        """
        Args:
            dropout: Dropout rate
            tau: Temperature parameter for softmax approximation
            random_feature_dim: Dimension of random features
        """
        super(ConvApproximation, self).__init__()
        self.tau = tau
        self.random_feature_dim = random_feature_dim
        self.activation = nn.ReLU()
        self.dropout = dropout

    def forward(self, x, node_vec1, node_vec2):
        """
        Forward pass of convolution approximation.

        Args:
            x: Input tensor of shape (B, N, 1, nhid)
            node_vec1: Query vectors of shape (B, N, 1, d)
            node_vec2: Key vectors of shape (B, N, 1, d)

        Returns:
            tuple: (output, transformed_node_vec1, transformed_node_vec2)
        """
        B = x.size(0)  # (B, N, 1, nhid)
        dim = node_vec1.shape[-1]  # (N, 1, d)

        random_seed = torch.ceil(torch.abs(torch.sum(node_vec1) * 1e8)).to(torch.int32)
        random_matrix = create_random_matrix(self.random_feature_dim, dim, seed=random_seed).to(node_vec1.device)

        node_vec1 = node_vec1 / math.sqrt(self.tau)
        node_vec2 = node_vec2 / math.sqrt(self.tau)
        node_vec1_prime = random_feature_map(node_vec1, True, random_matrix)  # [B, N, 1, r]
        node_vec2_prime = random_feature_map(node_vec2, False, random_matrix)  # [B, N, 1, r]

        x = linear_kernel(x, node_vec1_prime, node_vec2_prime)

        return x, node_vec1_prime, node_vec2_prime


class LinearizedConv(nn.Module):
    """
    Linearized spatial convolution layer.

    Uses gated linear units with random feature approximation
    for efficient spatial message passing.
    """

    def __init__(self, in_dim, hid_dim, dropout, tau=1.0, random_feature_dim=64):
        """
        Args:
            in_dim: Input dimension
            hid_dim: Hidden dimension
            dropout: Dropout rate
            tau: Temperature for random feature approximation
            random_feature_dim: Dimension of random features
        """
        super(LinearizedConv, self).__init__()

        self.dropout = dropout
        self.tau = tau
        self.random_feature_dim = random_feature_dim

        self.input_fc = nn.Conv2d(in_channels=in_dim, out_channels=hid_dim, kernel_size=(1, 1), bias=True)
        self.output_fc = nn.Conv2d(in_channels=in_dim, out_channels=hid_dim, kernel_size=(1, 1), bias=True)
        self.activation = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(p=dropout)

        self.conv_app_layer = ConvApproximation(self.dropout, self.tau, self.random_feature_dim)

    def forward(self, input_data, node_vec1, node_vec2):
        """
        Forward pass of linearized convolution.

        Args:
            input_data: Input tensor of shape (B, dim, N, 1)
            node_vec1: Query vectors
            node_vec2: Key vectors

        Returns:
            tuple: (output, transformed_node_vec1, transformed_node_vec2)
        """
        x = self.input_fc(input_data)
        x = self.activation(x) * self.output_fc(input_data)
        x = self.dropout_layer(x)

        x = x.permute(0, 2, 3, 1)  # (B, N, 1, dim*4)
        x, node_vec1_prime, node_vec2_prime = self.conv_app_layer(x, node_vec1, node_vec2)
        x = x.permute(0, 3, 1, 2)  # (B, dim*4, N, 1)

        return x, node_vec1_prime, node_vec2_prime


class BigST(AbstractTrafficStateModel):
    """
    BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting.

    This model uses random feature approximation to achieve linear complexity O(N)
    instead of quadratic complexity O(N^2) in spatial attention mechanisms.

    Key components:
    - Node embeddings for spatial identity
    - Time-of-day and day-of-week embeddings for temporal patterns
    - Linearized spatial convolution with random feature approximation
    - Optional residual connections and batch normalization

    Config parameters:
        - input_window: Number of input time steps (default: 12)
        - output_window: Number of output time steps (default: 12)
        - num_layers: Number of linearized conv layers (default: 1)
        - hid_dim: Hidden dimension (default: 32)
        - node_dim: Node embedding dimension (default: 32)
        - time_dim: Time embedding dimension (default: 32)
        - tau: Temperature for random feature approximation (default: 1.0)
        - random_feature_dim: Dimension of random features (default: 64)
        - dropout: Dropout rate (default: 0.1)
        - use_residual: Whether to use residual connections (default: True)
        - use_bn: Whether to use layer normalization (default: True)
        - use_spatial: Whether to compute spatial loss (default: False)
        - use_long: Whether to use long-term features (default: False)
    """

    def __init__(self, config, data_feature):
        """
        Initialize BigST model.

        Args:
            config: LibCity configuration dictionary
            data_feature: Data feature dictionary containing num_nodes, scaler, etc.
        """
        super().__init__(config, data_feature)
        self._logger = getLogger()

        # Device configuration
        self.device = config.get('device', torch.device('cpu'))

        # Data features
        self.num_nodes = data_feature.get('num_nodes')
        self._scaler = data_feature.get('scaler')
        self.feature_dim = data_feature.get('feature_dim', 3)
        self.output_dim = data_feature.get('output_dim', 1)

        # Model hyperparameters from config
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.num_layers = config.get('num_layers', 1)
        self.hid_dim = config.get('hid_dim', 32)
        self.node_dim = config.get('node_dim', 32)
        self.time_dim = config.get('time_dim', 32)
        self.tau = config.get('tau', 1.0)
        self.random_feature_dim = config.get('random_feature_dim', 64)
        self.dropout = config.get('dropout', 0.1)

        # Model options
        self.use_residual = config.get('use_residual', True)
        self.use_bn = config.get('use_bn', True)
        self.use_spatial = config.get('use_spatial', False)
        self.use_long = config.get('use_long', False)

        # Time embedding sizes
        # LibCity uses normalized time-of-day (0-1) which maps to 288 intervals for 5-min data
        self.time_num = config.get('time_num', 288)  # Number of time-of-day intervals
        self.week_num = config.get('week_num', 7)    # Days in a week

        # Input dimension: traffic value + time features
        self.in_dim = self.output_dim  # Only use traffic value for embedding

        self.activation = nn.ReLU()

        # Node embedding layer
        self.node_emb_layer = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
        nn.init.xavier_uniform_(self.node_emb_layer)

        # Time embedding layers
        self.time_emb_layer = nn.Parameter(torch.empty(self.time_num, self.time_dim))
        nn.init.xavier_uniform_(self.time_emb_layer)
        self.week_emb_layer = nn.Parameter(torch.empty(self.week_num, self.time_dim))
        nn.init.xavier_uniform_(self.week_emb_layer)

        # Input embedding layer
        # Maps (output_window * in_dim) to hid_dim
        self.input_emb_layer = nn.Conv2d(
            self.input_window * self.in_dim,
            self.hid_dim,
            kernel_size=(1, 1),
            bias=True
        )

        # Projection layers for node+time embeddings
        embedding_dim = self.node_dim + self.time_dim * 2
        self.W_1 = nn.Conv2d(embedding_dim, self.hid_dim, kernel_size=(1, 1), bias=True)
        self.W_2 = nn.Conv2d(embedding_dim, self.hid_dim, kernel_size=(1, 1), bias=True)

        # Linearized convolution layers
        self.linear_conv = nn.ModuleList()
        self.bn = nn.ModuleList()

        layer_dim = self.hid_dim * 4  # Concatenated: input_emb + node_emb + time_emb + week_emb
        for i in range(self.num_layers):
            self.linear_conv.append(
                LinearizedConv(layer_dim, layer_dim, self.dropout, self.tau, self.random_feature_dim)
            )
            self.bn.append(nn.LayerNorm(layer_dim))

        # Regression layer
        if self.use_long:
            # With long-term features
            regression_in_dim = layer_dim * 2 + self.hid_dim + self.output_window
        else:
            # Without long-term features
            regression_in_dim = layer_dim * 2

        self.regression_layer = nn.Conv2d(
            regression_in_dim,
            self.output_window,
            kernel_size=(1, 1),
            bias=True
        )

        self._logger.info(f"BigST initialized with {self.num_nodes} nodes, "
                         f"{self.num_layers} layers, hidden_dim={self.hid_dim}")

    def forward(self, batch):
        """
        Forward pass of BigST model.

        Args:
            batch: Dictionary containing:
                - 'X': Input tensor of shape (B, T, N, D) where D includes:
                    - Feature 0: Traffic speed value
                    - Feature 1: Time-of-day (normalized 0-1)
                    - Feature 2: Day-of-week (0-6)

        Returns:
            torch.Tensor: Predictions of shape (B, output_window, N, 1)
        """
        x = batch['X']  # (B, T, N, D) in LibCity format

        # Convert from LibCity format (B, T, N, D) to original format (B, N, T, D)
        x = x.permute(0, 2, 1, 3)  # (B, N, T, D)
        B, N, T, D = x.size()

        # Extract time embeddings from the last time step
        # Time-of-day: Feature index 1, normalized to [0, 1]
        # In LibCity, time-in-day is already normalized to [0, 1]
        time_indices = (x[:, :, -1, 1] * self.time_num).long().clamp(0, self.time_num - 1)
        time_emb = self.time_emb_layer[time_indices]  # (B, N, time_dim)

        # Day-of-week: Feature index 2
        # Handle both one-hot encoded (LibCity) and integer format (original)
        if D > 3:
            # One-hot encoded day-of-week (LibCity format with add_day_in_week=True)
            week_indices = torch.argmax(x[:, :, -1, 2:], dim=-1).long()
        else:
            # Integer format
            week_indices = x[:, :, -1, 2].long().clamp(0, self.week_num - 1)
        week_emb = self.week_emb_layer[week_indices]  # (B, N, time_dim)

        # Input embedding: use only traffic value (feature 0)
        x_val = x[..., :self.in_dim]  # (B, N, T, in_dim)
        x_val = x_val.contiguous().view(B, N, -1).transpose(1, 2).unsqueeze(-1)  # (B, T*in_dim, N, 1)
        input_emb = self.input_emb_layer(x_val)  # (B, hid_dim, N, 1)

        # Node embeddings
        node_emb = self.node_emb_layer.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1)  # (B, node_dim, N, 1)

        # Time embeddings reshape
        time_emb = time_emb.transpose(1, 2).unsqueeze(-1)  # (B, time_dim, N, 1)
        week_emb = week_emb.transpose(1, 2).unsqueeze(-1)  # (B, time_dim, N, 1)

        # Concatenate embeddings for graph convolution
        x_g = torch.cat([node_emb, time_emb, week_emb], dim=1)  # (B, node_dim+time_dim*2, N, 1)

        # Full feature concatenation
        x = torch.cat([input_emb, node_emb, time_emb, week_emb], dim=1)  # (B, hid_dim+node_dim+time_dim*2, N, 1)

        # Linearized spatial convolution
        x_pool = [x]  # (B, dim*4, N, 1)

        node_vec1 = self.W_1(x_g)  # (B, hid_dim, N, 1)
        node_vec2 = self.W_2(x_g)  # (B, hid_dim, N, 1)
        node_vec1 = node_vec1.permute(0, 2, 3, 1)  # (B, N, 1, hid_dim)
        node_vec2 = node_vec2.permute(0, 2, 3, 1)  # (B, N, 1, hid_dim)

        for i in range(self.num_layers):
            if self.use_residual:
                residual = x
            x, node_vec1_prime, node_vec2_prime = self.linear_conv[i](x, node_vec1, node_vec2)

            if self.use_residual:
                x = x + residual

            if self.use_bn:
                x = x.permute(0, 2, 3, 1)  # (B, N, 1, dim*4)
                x = self.bn[i](x)
                x = x.permute(0, 3, 1, 2)  # (B, dim*4, N, 1)

        x_pool.append(x)
        x = torch.cat(x_pool, dim=1)  # (B, dim*4*2, N, 1)

        x = self.activation(x)

        if self.use_long:
            # Long-term feature handling (not implemented in this single-stage version)
            # This would require additional preprocessing
            pass

        # Regression
        x = self.regression_layer(x)  # (B, output_window, N, 1)
        x = x.squeeze(-1).permute(0, 2, 1)  # (B, N, output_window)

        # Convert back to LibCity format: (B, output_window, N, 1)
        x = x.permute(0, 2, 1).unsqueeze(-1)  # (B, output_window, N, 1)

        return x

    def predict(self, batch):
        """
        Generate predictions for a batch.

        Args:
            batch: Dictionary containing 'X' tensor

        Returns:
            torch.Tensor: Predictions of shape (B, output_window, N, output_dim)
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate training loss.

        Args:
            batch: Dictionary containing 'X' and 'y' tensors

        Returns:
            torch.Tensor: Scalar loss value
        """
        y_true = batch['y']  # (B, output_window, N, output_dim)
        y_predicted = self.predict(batch)  # (B, output_window, N, 1)

        # Inverse transform to original scale
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        # Masked MAE loss (standard in LibCity)
        return loss.masked_mae_torch(y_predicted, y_true, null_val=0.0)
