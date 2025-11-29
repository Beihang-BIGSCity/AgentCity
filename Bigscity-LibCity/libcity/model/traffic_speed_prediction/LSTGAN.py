"""
Long-term Spatio-Temporal Graph Attention Network (LSTGAN)

Paper: Long-term spatio-temporal graph attention network for traffic forecasting
Publication: Expert Systems with Applications, 2025
Authors: Brahim Remmouche, Doulkifli Boukraa, Anastasia Zakharova, Thierry Delot

This implementation is adapted for LibCity framework.
Original repository: Not available (to be added when/if published)

The model addresses long-term traffic forecasting by:
1. Using graph attention mechanisms to capture spatial dependencies
2. Employing temporal convolutional layers to model long-term patterns
3. Integrating spatio-temporal features for improved predictions
"""

from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for capturing spatial dependencies
    """
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # Linear transformation
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention mechanism parameters
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        Args:
            h: input features (batch_size, num_nodes, in_features)
            adj: adjacency matrix (num_nodes, num_nodes)
        Returns:
            output: (batch_size, num_nodes, out_features)
        """
        # Linear transformation
        Wh = torch.matmul(h, self.W)  # (batch_size, num_nodes, out_features)

        # Attention mechanism
        batch_size = h.size(0)
        num_nodes = h.size(1)

        # Prepare for attention computation
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        # Mask attention based on adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Apply attention weights
        h_prime = torch.matmul(attention, Wh)

        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        batch_size = Wh.size(0)
        num_nodes = Wh.size(1)

        # Repeat features for concatenation
        Wh_repeated_in_chunks = Wh.repeat_interleave(num_nodes, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, num_nodes, 1)

        # Concatenate features
        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2
        )

        return all_combinations_matrix.view(batch_size, num_nodes, num_nodes, 2 * self.out_features)


class TemporalConvBlock(nn.Module):
    """
    Temporal Convolutional Block for capturing long-term temporal patterns
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super(TemporalConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, seq_len)
        Returns:
            output: (batch_size, out_channels, seq_len)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class STBlock(nn.Module):
    """
    Spatio-Temporal Block combining graph attention and temporal convolution
    """
    def __init__(self, in_features, hidden_dim, num_nodes, dropout=0.1):
        super(STBlock, self).__init__()

        # Spatial component (Graph Attention)
        self.gat = GraphAttentionLayer(in_features, hidden_dim, dropout)

        # Temporal component
        self.temporal_conv = TemporalConvBlock(
            num_nodes * hidden_dim,
            num_nodes * hidden_dim,
            kernel_size=3,
            dropout=dropout
        )

        # Residual connection
        if in_features != hidden_dim:
            self.residual = nn.Linear(in_features, hidden_dim)
        else:
            self.residual = None

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj):
        """
        Args:
            x: (batch_size, seq_len, num_nodes, in_features)
            adj: (num_nodes, num_nodes)
        Returns:
            output: (batch_size, seq_len, num_nodes, hidden_dim)
        """
        batch_size, seq_len, num_nodes, in_features = x.size()

        # Apply graph attention at each time step
        spatial_outputs = []
        for t in range(seq_len):
            spatial_out = self.gat(x[:, t, :, :], adj)
            spatial_outputs.append(spatial_out)
        spatial_out = torch.stack(spatial_outputs, dim=1)  # (batch_size, seq_len, num_nodes, hidden_dim)

        # Reshape for temporal convolution
        temp_input = spatial_out.reshape(batch_size, seq_len, -1).transpose(1, 2)
        temp_out = self.temporal_conv(temp_input)
        temp_out = temp_out.transpose(1, 2).reshape(batch_size, seq_len, num_nodes, -1)

        # Residual connection
        if self.residual is not None:
            residual = self.residual(x)
        else:
            residual = x

        out = self.layer_norm(temp_out + residual)
        return out


class LSTGAN(AbstractTrafficStateModel):
    """
    Long-term Spatio-Temporal Graph Attention Network for traffic forecasting
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        # Get data features
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.adj_mx = self.data_feature.get('adj_mx')

        # Get model configuration
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.device = config.get('device', torch.device('cpu'))

        # Model hyperparameters
        self.hidden_dim = config.get('hidden_dim', 64)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.1)
        self.num_heads = config.get('num_heads', 8)

        self._logger = getLogger()
        self._logger.info('LSTGAN initialized with {} nodes, {} input steps, {} output steps'.format(
            self.num_nodes, self.input_window, self.output_window))

        # Convert adjacency matrix to tensor
        if self.adj_mx is not None:
            self.adj_mx = torch.FloatTensor(self.adj_mx).to(self.device)
        else:
            # Create a fully connected graph if no adjacency matrix provided
            self.adj_mx = torch.ones(self.num_nodes, self.num_nodes).to(self.device)

        # Input projection
        self.input_proj = nn.Linear(self.feature_dim, self.hidden_dim)

        # Spatio-Temporal blocks
        self.st_blocks = nn.ModuleList([
            STBlock(self.hidden_dim, self.hidden_dim, self.num_nodes, self.dropout)
            for _ in range(self.num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

        # Temporal projection to generate multi-step predictions
        self.temporal_proj = nn.Linear(self.input_window, self.output_window)

    def forward(self, batch):
        """
        Args:
            batch: dict with key 'X' of shape (batch_size, input_window, num_nodes, feature_dim)
        Returns:
            predictions: (batch_size, output_window, num_nodes, output_dim)
        """
        x = batch['X']  # (batch_size, input_window, num_nodes, feature_dim)

        # Input projection
        x = self.input_proj(x)  # (batch_size, input_window, num_nodes, hidden_dim)

        # Apply spatio-temporal blocks
        for st_block in self.st_blocks:
            x = st_block(x, self.adj_mx)

        # Temporal projection for multi-step prediction
        # Reshape: (batch_size, num_nodes, hidden_dim, input_window)
        x = x.permute(0, 2, 3, 1)
        x = self.temporal_proj(x)
        # Reshape back: (batch_size, output_window, num_nodes, hidden_dim)
        x = x.permute(0, 3, 1, 2)

        # Output projection
        output = self.output_proj(x)  # (batch_size, output_window, num_nodes, output_dim)

        return output

    def predict(self, batch):
        """
        Prediction interface required by LibCity
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate loss for training
        Args:
            batch: dict with 'X' and 'y'
        Returns:
            loss: scalar tensor
        """
        y_true = batch['y']  # (batch_size, output_window, num_nodes, output_dim)
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0.0)
