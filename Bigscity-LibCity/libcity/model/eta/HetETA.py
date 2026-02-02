"""
HetETA Model adapted for LibCity framework.

Original source: repos/HetETA/codes/model/
Paper: Heterogeneous Information Network Embedding for Estimated Time of Arrival

Key adaptations from TensorFlow to PyTorch:
1. All TensorFlow operations converted to PyTorch equivalents
2. tf.SparseTensor operations replaced with PyTorch sparse tensors
3. tf.Variable/tf.get_variable replaced with nn.Parameter/nn.Linear
4. Layer class replaced with nn.Module
5. Graph convolutions adapted for dense tensor operations
6. Inherits from AbstractTrafficStateModel for LibCity integration

Architecture:
- Heterogeneous Information Networks (road network + vehicle trajectory)
- Chebyshev polynomial graph filters (K-hop diffusion)
- Multi-period temporal patterns (recent/daily/weekly)
- Multi-head attention for different relation types
- Road network: 7 relation types
- Vehicle network: 1 type

Author: Model Adaptation Agent
"""

import math
import numpy as np
from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


# ==============================================================================
# Utility Functions
# ==============================================================================

def calculate_scaled_laplacian(adj_mx):
    """
    Calculate the scaled Laplacian matrix for Chebyshev polynomial approximation.
    L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    L' = 2L/lambda_max - I

    Args:
        adj_mx: Adjacency matrix (numpy array)

    Returns:
        Scaled Laplacian matrix
    """
    n = adj_mx.shape[0]
    d = np.sum(adj_mx, axis=1)
    lap = np.diag(d) - adj_mx  # L = D - A

    # Normalized Laplacian
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                lap[i, j] /= np.sqrt(d[i] * d[j])

    lap[np.isinf(lap)] = 0
    lap[np.isnan(lap)] = 0

    # Compute eigenvalue for scaling
    eigenvalues = np.linalg.eigvals(lap)
    lambda_max = eigenvalues.max().real

    # Avoid division by zero
    if lambda_max < 1e-6:
        lambda_max = 2.0

    return 2 * lap / lambda_max - np.eye(n)


def calculate_cheb_polynomials(lap, K):
    """
    Calculate Chebyshev polynomials T_0(L) to T_{K-1}(L).
    T_0(L) = I
    T_1(L) = L
    T_k(L) = 2*L*T_{k-1}(L) - T_{k-2}(L)

    Args:
        lap: Scaled Laplacian matrix
        K: Order of polynomials

    Returns:
        List of K polynomial matrices
    """
    n = lap.shape[0]
    cheb_polynomials = [np.eye(n), lap.copy()]

    for i in range(2, K):
        cheb_polynomials.append(
            2 * np.matmul(lap, cheb_polynomials[-1]) - cheb_polynomials[-2]
        )

    return cheb_polynomials[:K]


# ==============================================================================
# Layer Normalization
# ==============================================================================

class NormLayer(nn.Module):
    """
    Layer normalization for spatio-temporal data.
    Normalizes across node and feature dimensions.
    """

    def __init__(self, num_nodes, feature_dim):
        super(NormLayer, self).__init__()
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.gamma = nn.Parameter(torch.ones(1, 1, num_nodes, feature_dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_nodes, feature_dim))

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, num_nodes, feature_dim]

        Returns:
            Normalized tensor [batch_size, seq_len, num_nodes, feature_dim]
        """
        mu = x.mean(dim=[2, 3], keepdim=True)
        sigma = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-6) * self.gamma + self.beta


# ==============================================================================
# Temporal Convolution Layer
# ==============================================================================

class TemporalConvLayer(nn.Module):
    """
    Temporal convolutional layer with GLU or ReLU activation.
    Performs 1D convolution along the time dimension.

    Adapted from TensorFlow implementation:
    - tf.nn.conv2d replaced with nn.Conv2d
    - Manual weight management replaced with nn.Linear/Conv2d
    """

    def __init__(self, Kt, input_dim, output_dim, act_func='relu'):
        """
        Args:
            Kt: Kernel size for temporal convolution
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            act_func: Activation function ('GLU', 'relu', 'sigmoid', 'linear')
        """
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act_func = act_func

        # Alignment layer for residual connection
        if input_dim > output_dim:
            self.align_conv = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        else:
            self.align_conv = None

        # Temporal convolution
        if act_func == 'GLU':
            # GLU needs 2x output channels
            self.conv = nn.Conv2d(
                input_dim, 2 * output_dim,
                kernel_size=(Kt, 1),
                padding=0
            )
        else:
            self.conv = nn.Conv2d(
                input_dim, output_dim,
                kernel_size=(Kt, 1),
                padding=0
            )

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, num_nodes, input_dim]

        Returns:
            Output tensor [batch_size, seq_len-Kt+1, num_nodes, output_dim]
        """
        batch_size, seq_len, num_nodes, c_in = x.shape
        assert c_in == self.input_dim

        # Align input for residual connection
        # Convert to [batch, channels, time, nodes] for conv2d
        x_t = x.permute(0, 3, 1, 2)  # [B, C, T, N]

        if self.align_conv is not None:
            x_input = self.align_conv(x_t)
        elif self.input_dim < self.output_dim:
            # Pad channels
            padding = torch.zeros(
                batch_size, self.output_dim - self.input_dim, seq_len, num_nodes,
                device=x.device
            )
            x_input = torch.cat([x_t, padding], dim=1)
        else:
            x_input = x_t

        # Keep aligned sequence length for residual
        x_input = x_input[:, :, self.Kt - 1:, :]  # [B, C, T-Kt+1, N]

        # Apply temporal convolution
        x_conv = self.conv(x_t)  # [B, C_out or 2*C_out, T-Kt+1, N]

        if self.act_func == 'GLU':
            # Gated Linear Unit
            gate = torch.sigmoid(x_conv[:, self.output_dim:, :, :])
            output = (x_conv[:, :self.output_dim, :, :] + x_input) * gate
        elif self.act_func == 'relu':
            output = F.relu(x_conv + x_input)
        elif self.act_func == 'sigmoid':
            output = torch.sigmoid(x_conv)
        elif self.act_func == 'linear':
            output = x_conv
        else:
            raise ValueError(f'Unknown activation: {self.act_func}')

        # Convert back to [B, T, N, C]
        output = output.permute(0, 2, 3, 1)
        return output


# ==============================================================================
# Attention Mechanism for Chebyshev Graph Convolution
# ==============================================================================

class AttentionHeadCheb(nn.Module):
    """
    Attention head for heterogeneous graph attention with Chebyshev polynomials.
    Computes attention scores for different relation types and applies graph convolution.

    Adapted from TensorFlow:
    - tf.sparse_tensor_dense_matmul replaced with torch.sparse.mm
    - Sparse hadamard product adapted for PyTorch sparse tensors
    """

    def __init__(self, input_dim, output_dim, num_supports, num_atten_supports,
                 in_drop=0.0, is_residual=False, device=None):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            num_supports: Number of Chebyshev polynomial supports (K-order)
            num_atten_supports: Number of attention adjacency matrices (relation types)
            in_drop: Input dropout rate
            is_residual: Whether to use residual connection
            device: Computation device
        """
        super(AttentionHeadCheb, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_supports = num_supports
        self.num_atten_supports = num_atten_supports
        self.in_drop = in_drop
        self.is_residual = is_residual
        self.device = device if device is not None else torch.device('cpu')

        # Transformation weights for each Chebyshev order
        self.W_transform = nn.ModuleList([
            nn.Linear(input_dim, output_dim, bias=False)
            for _ in range(num_supports)
        ])

        # Attention weights: left and right for each support and each relation type
        self.W_left = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(output_dim, 1, bias=False)
                for _ in range(num_atten_supports + 1)  # +1 for high-order neighbors
            ])
            for _ in range(num_supports)
        ])

        self.W_right = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(output_dim, 1, bias=False)
                for _ in range(num_atten_supports + 1)
            ])
            for _ in range(num_supports)
        ])

        # Residual connection weights
        if is_residual:
            self.W_residual = nn.Linear(input_dim + output_dim, output_dim, bias=False)

        self.dropout = nn.Dropout(in_drop) if in_drop > 0 else None

    def forward(self, x, supports, atten_supports):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            supports: List of Chebyshev polynomial matrices [K] of shape [N, N]
            atten_supports: List of relation-specific adjacency matrices [num_relations]

        Returns:
            Output features [num_nodes, output_dim]
        """
        if self.dropout is not None:
            x = self.dropout(x)

        output_list = []

        for k in range(self.num_supports):
            # Transform features for this polynomial order
            wx = self.W_transform[k](x)  # [N, output_dim]

            # Get support matrix for this order
            support_k = supports[k]  # [N, N]

            # Compute attention scores from all relation types
            edges_scores = None

            for i in range(self.num_atten_supports):
                # Attention scores
                a_left = self.W_left[k][i](wx).squeeze(-1)  # [N]
                a_right = self.W_right[k][i](wx).squeeze(-1)  # [N]

                # Get relation-specific adjacency
                adj_i = atten_supports[i]  # [N, N] dense or sparse

                # Compute edge attention: a_left[i] + a_right[j] for edge (i,j)
                if adj_i.is_sparse:
                    # For sparse adjacency
                    indices = adj_i.coalesce().indices()
                    row_idx = indices[0]
                    col_idx = indices[1]
                    edge_attn = a_left[row_idx] + a_right[col_idx]

                    # Create sparse attention tensor
                    attn_sparse = torch.sparse_coo_tensor(
                        indices, edge_attn, adj_i.shape, device=self.device
                    )
                else:
                    # For dense adjacency
                    attn_sparse = (a_left.unsqueeze(1) + a_right.unsqueeze(0)) * (adj_i != 0).float()

                if edges_scores is None:
                    edges_scores = attn_sparse
                else:
                    if adj_i.is_sparse:
                        edges_scores = edges_scores + attn_sparse
                    else:
                        edges_scores = edges_scores + attn_sparse

            # Add high-order neighbor attention
            a_left_ho = self.W_left[k][self.num_atten_supports](wx).squeeze(-1)
            a_right_ho = self.W_right[k][self.num_atten_supports](wx).squeeze(-1)

            if isinstance(support_k, torch.Tensor) and support_k.is_sparse:
                indices = support_k.coalesce().indices()
                row_idx = indices[0]
                col_idx = indices[1]
                edge_attn_ho = a_left_ho[row_idx] + a_right_ho[col_idx]
                attn_ho = torch.sparse_coo_tensor(
                    indices, edge_attn_ho, support_k.shape, device=self.device
                )
            else:
                # Dense support
                if isinstance(support_k, np.ndarray):
                    support_k = torch.FloatTensor(support_k).to(self.device)
                attn_ho = (a_left_ho.unsqueeze(1) + a_right_ho.unsqueeze(0)) * (support_k != 0).float()

            # Combine all attention scores
            if edges_scores is not None:
                if isinstance(edges_scores, torch.Tensor) and edges_scores.is_sparse:
                    all_scores = edges_scores.to_dense() + attn_ho.to_dense() if attn_ho.is_sparse else edges_scores.to_dense() + attn_ho
                else:
                    all_scores = edges_scores + (attn_ho.to_dense() if isinstance(attn_ho, torch.Tensor) and attn_ho.is_sparse else attn_ho)
            else:
                all_scores = attn_ho.to_dense() if isinstance(attn_ho, torch.Tensor) and attn_ho.is_sparse else attn_ho

            # Softmax attention
            all_scores = all_scores.masked_fill(all_scores == 0, float('-inf'))
            attn_probs = F.softmax(all_scores, dim=-1)
            attn_probs = torch.where(
                torch.isnan(attn_probs),
                torch.zeros_like(attn_probs),
                attn_probs
            )

            # Apply attention
            output_k = torch.matmul(attn_probs, wx)
            output_list.append(output_k)

        # Sum outputs from all polynomial orders
        output = sum(output_list)

        # Residual connection
        if self.is_residual:
            output = self.W_residual(torch.cat([x, output], dim=-1))

        return F.elu(output)


class MultiAttentionCheb(nn.Module):
    """
    Multi-head attention for heterogeneous graph convolution.
    Combines multiple attention heads for different relation types.
    """

    def __init__(self, input_dim, output_dim, num_supports, num_atten_supports,
                 heads_num=1, is_concat=False, in_drop=0.0, is_residual=False, device=None):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension per head
            num_supports: Number of Chebyshev polynomial supports
            num_atten_supports: Number of attention adjacency matrices
            heads_num: Number of attention heads
            is_concat: Whether to concatenate heads (True) or average (False)
            in_drop: Input dropout rate
            is_residual: Whether to use residual connection
            device: Computation device
        """
        super(MultiAttentionCheb, self).__init__()
        self.heads_num = heads_num
        self.is_concat = is_concat
        self.device = device

        self.heads = nn.ModuleList([
            AttentionHeadCheb(
                input_dim, output_dim, num_supports, num_atten_supports,
                in_drop, is_residual, device
            )
            for _ in range(heads_num)
        ])

    def forward(self, x, supports, atten_supports):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            supports: Chebyshev polynomial matrices
            atten_supports: Relation-specific adjacency matrices

        Returns:
            Output features [num_nodes, output_dim * heads or output_dim]
        """
        head_results = [head(x, supports, atten_supports) for head in self.heads]

        if self.is_concat:
            output = torch.cat(head_results, dim=-1)
        else:
            output = sum(head_results) / self.heads_num

        return output


# ==============================================================================
# Spatial Graph Convolution Layer with Chebyshev Polynomials
# ==============================================================================

class SpatioConvLayerCheb(nn.Module):
    """
    Spatial graph convolution layer using Chebyshev polynomials and multi-head attention.
    Handles heterogeneous graph structures with different relation types.
    """

    def __init__(self, Ks, input_dim, output_dim, num_atten_supports, heads_num, device, name=''):
        """
        Args:
            Ks: Order of Chebyshev polynomials
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            num_atten_supports: Number of relation-specific adjacency matrices
            heads_num: Number of attention heads
            device: Computation device
            name: Layer name for identification
        """
        super(SpatioConvLayerCheb, self).__init__()
        self.Ks = Ks
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        # Multi-head attention convolution
        self.atten_conv = MultiAttentionCheb(
            input_dim, output_dim,
            num_supports=Ks + 1,  # T_0 to T_Ks
            num_atten_supports=num_atten_supports,
            heads_num=heads_num,
            is_concat=False,
            in_drop=0.0,
            is_residual=False,
            device=device
        )

        # Alignment for residual connection
        if input_dim > output_dim:
            self.align_conv = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        else:
            self.align_conv = None

    def forward(self, x, supports, atten_supports):
        """
        Args:
            x: Input tensor [batch_size, seq_len, num_nodes, input_dim]
            supports: Chebyshev polynomial matrices (list of K+1 tensors)
            atten_supports: Relation-specific adjacency matrices

        Returns:
            Output tensor [batch_size, seq_len, num_nodes, output_dim]
        """
        batch_size, seq_len, num_nodes, c_in = x.shape
        assert c_in == self.input_dim

        # Align for residual connection
        x_t = x.permute(0, 3, 1, 2)  # [B, C, T, N]
        if self.align_conv is not None:
            x_input = self.align_conv(x_t)
        elif self.input_dim < self.output_dim:
            padding = torch.zeros(
                batch_size, self.output_dim - self.input_dim, seq_len, num_nodes,
                device=self.device
            )
            x_input = torch.cat([x_t, padding], dim=1)
        else:
            x_input = x_t
        x_input = x_input.permute(0, 2, 3, 1)  # [B, T, N, C]

        # Reshape for graph convolution: [B*T, N, C]
        x_reshaped = x.reshape(batch_size * seq_len, num_nodes, self.input_dim)

        # Apply graph convolution to each batch-time element
        gc_outputs = []
        for bt in range(batch_size * seq_len):
            x_bt = x_reshaped[bt]  # [N, C]
            gc_out = self.atten_conv(x_bt, supports, atten_supports)  # [N, C_out]
            gc_outputs.append(gc_out)

        x_gc = torch.stack(gc_outputs, dim=0)  # [B*T, N, C_out]
        x_gc = x_gc.reshape(batch_size, seq_len, num_nodes, self.output_dim)

        return F.relu(x_gc + x_input)


# ==============================================================================
# Spatio-Temporal Convolutional Block
# ==============================================================================

class STConvBlock(nn.Module):
    """
    Spatio-Temporal Convolutional Block.
    Contains two temporal convolutions with a spatial graph convolution in between.
    Supports heterogeneous networks (road + vehicle).
    """

    def __init__(self, Ks, Kt, num_nodes, road_num, car_num,
                 road_atten_num, car_atten_num, channels, dropout, heads_num, device,
                 act_func='GLU'):
        """
        Args:
            Ks: Spatial kernel size (Chebyshev order)
            Kt: Temporal kernel size
            num_nodes: Number of nodes in the graph
            road_num: Number of road network supports
            car_num: Number of vehicle network supports
            road_atten_num: Number of road relation types
            car_atten_num: Number of vehicle relation types
            channels: [input_dim, hidden_dim, output_dim]
            dropout: Dropout rate
            heads_num: Number of attention heads
            device: Computation device
            act_func: Activation function
        """
        super(STConvBlock, self).__init__()
        self.road_num = road_num
        self.car_num = car_num
        self.device = device

        # Temporal convolution 1
        self.temporal_layer1 = TemporalConvLayer(Kt, channels[0], channels[1], act_func)

        # Spatial convolutions for each network type
        if road_num > 0:
            self.road_spatio_layer = SpatioConvLayerCheb(
                Ks, channels[1], channels[1],
                num_atten_supports=road_atten_num,
                heads_num=heads_num,
                device=device,
                name='road'
            )

        if car_num > 0:
            self.car_spatio_layer = SpatioConvLayerCheb(
                Ks, channels[1], channels[1],
                num_atten_supports=car_atten_num,
                heads_num=heads_num,
                device=device,
                name='car'
            )

        # Determine hidden dimension based on network types
        hidden_num = (1 if road_num > 0 else 0) + (1 if car_num > 0 else 0)

        # Temporal convolution 2
        self.temporal_layer2 = TemporalConvLayer(Kt, hidden_num * channels[1], channels[2])

        # Normalization
        self.norm_layer = NormLayer(num_nodes, channels[2])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, road_supports, road_atten_supports, car_supports, car_atten_supports):
        """
        Args:
            x: Input tensor [batch_size, seq_len, num_nodes, input_dim]
            road_supports: Road network Chebyshev matrices
            road_atten_supports: Road relation adjacency matrices
            car_supports: Vehicle network Chebyshev matrices
            car_atten_supports: Vehicle relation adjacency matrices

        Returns:
            Output tensor [batch_size, seq_len-2*(Kt-1), num_nodes, output_dim]
        """
        # First temporal convolution
        x = self.temporal_layer1(x)

        # Spatial convolutions
        outputs = []
        if self.road_num > 0:
            x_road = self.road_spatio_layer(x, road_supports, road_atten_supports)
            outputs.append(x_road)

        if self.car_num > 0:
            x_car = self.car_spatio_layer(x, car_supports, car_atten_supports)
            outputs.append(x_car)

        if len(outputs) > 1:
            x = torch.cat(outputs, dim=-1)
        else:
            x = outputs[0]

        # Second temporal convolution
        x = self.temporal_layer2(x)

        # Normalization and dropout
        x = self.norm_layer(x)
        x = self.dropout(x)

        return x


# ==============================================================================
# Output Layers
# ==============================================================================

class STLastLayer(nn.Module):
    """
    Final temporal aggregation layer.
    Reduces temporal dimension to 1.
    """

    def __init__(self, seq_len, num_nodes, dim, act_func='GLU'):
        super(STLastLayer, self).__init__()
        self.temporal_layer = TemporalConvLayer(seq_len, dim, dim, act_func)
        self.norm_layer = NormLayer(num_nodes, dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, num_nodes, dim]

        Returns:
            Output tensor [batch_size, 1, num_nodes, dim]
        """
        x = self.temporal_layer(x)
        x = self.norm_layer(x)
        return x


class STPredictLayer(nn.Module):
    """
    Final prediction layer.
    Produces per-node speed predictions.
    """

    def __init__(self, num_nodes, input_dim, hidden_dim):
        super(STPredictLayer, self).__init__()
        self.temporal_layer = TemporalConvLayer(1, input_dim, hidden_dim, act_func='sigmoid')
        self.conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.bias = nn.Parameter(torch.zeros(num_nodes, 1))

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, 1, num_nodes, input_dim]

        Returns:
            last_state: Hidden state [batch_size, 1, num_nodes, hidden_dim]
            output: Predictions [batch_size, 1, num_nodes, 1]
        """
        last_state = self.temporal_layer(x)  # [B, 1, N, hidden_dim]

        # Convert to conv2d format
        x_t = last_state.permute(0, 3, 1, 2)  # [B, hidden_dim, 1, N]
        output = self.conv(x_t)  # [B, 1, 1, N]
        output = output.permute(0, 2, 3, 1)  # [B, 1, N, 1]
        output = output + self.bias  # Add per-node bias

        return last_state, output


# ==============================================================================
# Main HetETA Model
# ==============================================================================

class HetETA(AbstractTrafficStateModel):
    """
    Heterogeneous ETA Model for Traffic Speed Prediction and Travel Time Estimation.

    Architecture:
    1. Multi-period temporal patterns (recent/daily/weekly)
    2. Heterogeneous graph convolutions (road network + vehicle trajectory)
    3. Chebyshev polynomial graph filters
    4. Multi-head attention for different relation types
    5. Speed-based ETA calculation

    LibCity Adaptations:
    - Inherits from AbstractTrafficStateModel
    - Uses config dict for hyperparameters
    - Device-agnostic operations
    - Implements predict() and calculate_loss() methods

    Config Parameters:
    - max_diffusion_step: Chebyshev polynomial order (default: 2)
    - rnn_units: Hidden dimension (default: 11)
    - seq_len: Recent time steps (default: 4)
    - days: Daily pattern steps (default: 4)
    - weeks: Weekly pattern steps (default: 4)
    - road_net_num: Road network relation types (default: 7)
    - car_net_num: Vehicle network relation types (default: 1)
    - heads_num: Attention heads (default: 1)
    - dropout: Dropout rate (default: 0.0)
    - regular_rate: L2 regularization rate (default: 0.0005)
    """

    def __init__(self, config, data_feature):
        super(HetETA, self).__init__(config, data_feature)
        self._logger = getLogger()
        self.config = config
        self.data_feature = data_feature

        # Device
        self.device = config.get('device', torch.device('cpu'))

        # Data dimensions
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)
        self._scaler = data_feature.get('scaler')

        # Model hyperparameters
        self.max_diffusion_step = config.get('max_diffusion_step', 2)
        self.rnn_units = config.get('rnn_units', 11)
        self.seq_len = config.get('seq_len', 4)
        self.days = config.get('days', 4)
        self.weeks = config.get('weeks', 4)
        self.road_net_num = config.get('road_net_num', 7)
        self.car_net_num = config.get('car_net_num', 1)
        self.heads_num = config.get('heads_num', 1)
        self.dropout = config.get('dropout', 0.0)
        self.regular_rate = config.get('regular_rate', 0.0005)

        self.input_window = config.get('input_window', self.seq_len + self.days + self.weeks)
        self.output_window = config.get('output_window', 1)

        # Block configuration
        self.blocks = [[self.feature_dim, 8, self.rnn_units]]

        # Temporal kernel configurations
        self.Ko = {
            'recent': self.seq_len,
            'days': self.days,
            'weeks': self.weeks
        }
        self.Ks = self.max_diffusion_step
        self.Kt = 2  # Temporal kernel size

        # Time period indices
        self.Istart = {
            'weeks': 0,
            'days': self.weeks,
            'recent': self.weeks + self.days
        }
        self.Iend = {
            'weeks': self.weeks,
            'days': self.weeks + self.days,
            'recent': self.weeks + self.days + self.seq_len
        }

        # Get active time types
        self.types_list = self._get_type_list()

        # Build graph supports
        self._build_graph_supports(data_feature)

        # Build ST convolutional blocks for each time type
        self.st_blocks = nn.ModuleDict()
        self.last_layers = nn.ModuleDict()

        for time_type in self.types_list:
            # Calculate output length after temporal convolutions
            Ko_type = self.Ko[time_type]

            # ST conv blocks
            block_list = nn.ModuleList()
            for i, channels in enumerate(self.blocks):
                block = STConvBlock(
                    Ks=self.Ks,
                    Kt=self.Kt,
                    num_nodes=self.num_nodes,
                    road_num=self.road_net_num,
                    car_num=self.car_net_num,
                    road_atten_num=self.road_net_num,
                    car_atten_num=self.car_net_num,
                    channels=channels,
                    dropout=self.dropout,
                    heads_num=self.heads_num,
                    device=self.device
                )
                block_list.append(block)
                Ko_type -= 2 * (self.Kt - 1)

            self.st_blocks[time_type] = block_list

            # Last temporal layer if needed
            if Ko_type > 1:
                self.last_layers[time_type] = STLastLayer(
                    Ko_type, self.num_nodes, self.blocks[-1][-1]
                )

        # Final prediction layer
        total_channels = self.blocks[-1][-1] * len(self.types_list)
        self.predict_layer = STPredictLayer(
            self.num_nodes, total_channels, self.blocks[-1][-1]
        )

        self._init_weights()
        self._logger.info(f'HetETA model initialized with {self.num_nodes} nodes, '
                         f'{len(self.types_list)} time types')

    def _get_type_list(self):
        """Get list of active temporal pattern types."""
        types = []
        if self.seq_len > 0:
            types.append('recent')
        if self.days > 0:
            types.append('days')
        if self.weeks > 0:
            types.append('weeks')
        assert len(types) > 0, 'At least one time type must be active'
        return types

    def _build_graph_supports(self, data_feature):
        """
        Build Chebyshev polynomial supports from adjacency matrices.
        """
        # Get adjacency matrix from data_feature
        adj_mx = data_feature.get('adj_mx', None)

        if adj_mx is not None:
            # Calculate scaled Laplacian
            lap = calculate_scaled_laplacian(adj_mx)

            # Calculate Chebyshev polynomials
            cheb_polys = calculate_cheb_polynomials(lap, self.Ks + 1)

            # Convert to tensors
            self.road_supports = [
                torch.FloatTensor(poly).to(self.device)
                for poly in cheb_polys
            ]
        else:
            # Create identity supports if no adjacency provided
            self._logger.warning('No adjacency matrix provided, using identity')
            self.road_supports = [
                torch.eye(self.num_nodes).to(self.device)
                for _ in range(self.Ks + 1)
            ]

        # For heterogeneous relations, use the same supports by default
        # In practice, these would come from different relation types
        self.road_atten_supports = [
            torch.FloatTensor(adj_mx).to(self.device) if adj_mx is not None
            else torch.eye(self.num_nodes).to(self.device)
            for _ in range(self.road_net_num)
        ]

        self.car_supports = self.road_supports.copy()
        self.car_atten_supports = [
            torch.FloatTensor(adj_mx).to(self.device) if adj_mx is not None
            else torch.eye(self.num_nodes).to(self.device)
            for _ in range(self.car_net_num)
        ]

    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'bias' in name:
                init.zeros_(param)
            elif 'weight' in name:
                init.xavier_uniform_(param)

    def forward(self, batch):
        """
        Forward pass of HetETA.

        Args:
            batch: Dict containing 'X' tensor [batch_size, input_length, num_nodes, feature_dim]

        Returns:
            pred_speed: Predicted speed [batch_size, 1, num_nodes, 1]
        """
        x = batch['X']  # [B, T, N, F]

        all_outputs = []

        for time_type in self.types_list:
            # Extract time slice
            x_slice = x[:, self.Istart[time_type]:self.Iend[time_type], :, :]

            # Apply ST conv blocks
            for block in self.st_blocks[time_type]:
                x_slice = block(
                    x_slice,
                    self.road_supports,
                    self.road_atten_supports,
                    self.car_supports,
                    self.car_atten_supports
                )

            # Apply last layer if needed
            if time_type in self.last_layers:
                x_slice = self.last_layers[time_type](x_slice)

            all_outputs.append(x_slice)

        # Concatenate all time type outputs
        concat_state = torch.cat(all_outputs, dim=-1)  # [B, 1, N, C*len(types)]

        # Final prediction
        last_state, pred = self.predict_layer(concat_state)

        # Apply sigmoid and scale to speed (0-120 km/h -> m/s)
        pred_speed = torch.sigmoid(pred.squeeze(-1))  # [B, 1, N]
        pred_speed = pred_speed * (120 / 3.6)  # Convert to m/s
        pred_speed = torch.clamp(pred_speed, min=0.1, max=120/3.6)
        pred_speed = pred_speed.unsqueeze(-1)  # [B, 1, N, 1]

        return pred_speed

    def predict(self, batch):
        """
        Make predictions on a batch.

        For ETA task: returns travel time predictions in seconds.
        The method handles both traffic state format (X/y) and ETA format (with link_distances).

        Args:
            batch: Input batch dictionary
                - For ETA: contains 'X', 'link_distances', 'gt_eta_time'
                - For traffic state: contains 'X', 'y'

        Returns:
            For ETA task: predictions [batch_size, 1] - travel time in seconds
            For traffic state: predictions [batch_size, output_window, num_nodes, output_dim]
        """
        # Check if this is ETA format (has link_distances)
        # Use batch.data for key checking since Batch class doesn't implement __contains__
        batch_keys = batch.data.keys() if hasattr(batch, 'data') else []
        if 'link_distances' in batch_keys:
            # ETA task: compute travel time from predicted speeds
            return self._predict_eta(batch)
        else:
            # Traffic state prediction: multi-step speed prediction
            return self._predict_traffic_state(batch)

    def _predict_eta(self, batch):
        """
        Predict ETA from link speeds.

        ETA = sum(link_distance / predicted_speed) for all links in route

        Args:
            batch: Batch with 'X' and 'link_distances'

        Returns:
            eta: [batch_size, 1] travel time in seconds
        """
        # Get input features
        x = batch['X']  # [B, T, N, F]

        # Ensure proper tensor format
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension

        # Predict link speeds
        batch_tmp = {'X': x}
        pred_speed = self.forward(batch_tmp)  # [B, 1, N, 1]
        pred_speed = pred_speed.squeeze(-1).squeeze(1)  # [B, N]

        # Clamp speed to avoid division issues
        pred_speed = torch.clamp(pred_speed, min=0.1)  # Minimum 0.1 m/s

        # Get link distances
        link_distances = batch['link_distances']  # [B, N]
        if not isinstance(link_distances, torch.Tensor):
            link_distances = torch.FloatTensor(link_distances).to(self.device)
        if link_distances.dim() == 1:
            link_distances = link_distances.unsqueeze(0)

        # Calculate ETA: sum(distance / speed) for each sample
        # Only sum for links with non-zero distance
        time_per_link = link_distances / pred_speed  # [B, N]
        time_per_link = torch.where(
            link_distances > 0,
            time_per_link,
            torch.zeros_like(time_per_link)
        )
        eta = time_per_link.sum(dim=1, keepdim=True)  # [B, 1]

        return eta

    def _predict_traffic_state(self, batch):
        """
        Multi-step traffic state (speed) prediction.

        Args:
            batch: Batch with 'X' and optionally 'y'

        Returns:
            predictions: [batch_size, output_window, num_nodes, output_dim]
        """
        x = batch['X']
        y = batch.get('y', None)

        y_preds = []
        x_ = x.clone()

        for i in range(self.output_window):
            batch_tmp = {'X': x_}
            y_ = self.forward(batch_tmp)  # [B, 1, N, output_dim]
            y_preds.append(y_.clone())

            if y is not None and y_.shape[-1] < x_.shape[-1]:
                # Append extra features from ground truth
                y_ = torch.cat([y_, y[:, i:i+1, :, self.output_dim:]], dim=3)

            # Shift input window
            x_ = torch.cat([x_[:, 1:, :, :], y_], dim=1)

        y_preds = torch.cat(y_preds, dim=1)  # [B, output_window, N, output_dim]
        return y_preds

    def calculate_loss(self, batch):
        """
        Calculate loss for training.

        Handles both ETA format (time) and traffic state format (y).

        Args:
            batch: Input batch
                - For ETA: contains 'X', 'link_distances', 'time'
                - For traffic state: contains 'X', 'y'

        Returns:
            loss: Scalar loss tensor
        """
        # Check if this is ETA format (has link_distances and time)
        # Use batch.data for key checking since Batch class doesn't implement __contains__
        batch_keys = batch.data.keys() if hasattr(batch, 'data') else []
        if 'link_distances' in batch_keys and 'time' in batch_keys:
            # ETA task: compute loss on travel time
            y_true = batch['time']  # [B, 1]
            if not isinstance(y_true, torch.Tensor):
                y_true = torch.FloatTensor(y_true).to(self.device)
            if y_true.dim() == 1:
                y_true = y_true.unsqueeze(1)

            y_pred = self._predict_eta(batch)  # [B, 1]

            # Compute loss (MAPE for ETA task)
            mape = torch.abs(y_pred - y_true) / (y_true + 1e-6)
            return mape.mean()

        else:
            # Traffic state prediction
            y_true = batch['y']  # [B, T_out, N, F]
            y_pred = self._predict_traffic_state(batch)  # [B, T_out, N, output_dim]

            # Apply inverse transform if scaler available
            if self._scaler is not None:
                y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
                y_pred = self._scaler.inverse_transform(y_pred[..., :self.output_dim])
            else:
                y_true = y_true[..., :self.output_dim]

            # Masked MSE loss
            return loss.masked_mse_torch(y_pred, y_true)

    def calculate_eta(self, batch, link_distances):
        """
        Calculate ETA from predicted speeds and link distances.

        Args:
            batch: Input batch
            link_distances: Sparse tensor of link distances [batch_size, num_nodes]

        Returns:
            eta: Estimated time of arrival for each trajectory
        """
        pred_speed = self.forward(batch)  # [B, 1, N, 1]
        pred_speed = pred_speed.squeeze()  # [B, N] or [N] for batch_size=1

        if pred_speed.dim() == 1:
            pred_speed = pred_speed.unsqueeze(0)

        # ETA = sum(distance / speed) for each link in trajectory
        # link_distances: [B, N] where each row is distances for one trajectory
        if isinstance(link_distances, torch.Tensor):
            if link_distances.is_sparse:
                # For sparse distances
                eta = torch.sparse.sum(
                    link_distances / pred_speed, dim=1
                )
            else:
                eta = (link_distances / pred_speed).sum(dim=1)
        else:
            link_distances = torch.FloatTensor(link_distances).to(self.device)
            eta = (link_distances / pred_speed).sum(dim=1)

        return eta
