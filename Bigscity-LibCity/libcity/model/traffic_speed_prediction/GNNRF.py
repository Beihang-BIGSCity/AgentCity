"""
GNNRF Model for LibCity Framework

This module adapts the NodeEncodedGCN_1l model from the GNNRF repository for use
in the LibCity traffic prediction framework.

Original Model: NodeEncodedGCN_1l
Source: /home/wangwenrui/shk/AgentCity/repos/GNNRF/src/models/gcn_model.py
Task: ETA/Arrival Time Prediction for autonomous shuttles

Key Adaptations:
1. Converted from PyTorch Lightning to LibCity's AbstractTrafficStateModel
2. Adapted PyTorch Geometric operations to work with LibCity's batch format
3. Replaced external Standardize transform with LibCity's scaler
4. Preserved core architecture: node encoding, GCN layer, skip connections

Architecture:
- Single-layer GCN with node one-hot encoding
- Input: Lag features, vehicle ID, temporal features (dow/tod sin/cos), weather features
- Skip connection merging original features with GCN output
- Output: Single value (predicted travel/dwell time) per node
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from logging import getLogger

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


class GCNConvLayer(nn.Module):
    """
    Simple GCN Convolution Layer implementation.

    This replaces PyTorch Geometric's GCNConv to avoid external dependencies.
    Implements: H' = sigma(D^(-1/2) * A * D^(-1/2) * H * W)

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        aggr: Aggregation function (default 'mean')
    """

    def __init__(self, in_channels, out_channels, aggr='mean'):
        super(GCNConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr

        self.linear = nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x, adj_norm):
        """
        Forward pass for GCN convolution.

        Args:
            x: Node features of shape (batch_size * num_nodes, in_channels)
               or (batch_size, num_nodes, in_channels)
            adj_norm: Normalized adjacency matrix of shape (num_nodes, num_nodes)

        Returns:
            Updated node features
        """
        # x shape: (batch_size, num_nodes, in_channels) or flattened
        if x.dim() == 2:
            # Reshape to (batch_size, num_nodes, features)
            # This case handles when x is passed as flattened
            pass

        # Apply linear transformation first
        x_transformed = self.linear(x)  # (batch, num_nodes, out_channels)

        # Apply adjacency multiplication (message passing)
        # adj_norm: (num_nodes, num_nodes)
        # x_transformed: (batch, num_nodes, out_channels)
        if x_transformed.dim() == 3:
            out = torch.einsum('nm,bmf->bnf', adj_norm, x_transformed)
        else:
            out = torch.mm(adj_norm, x_transformed)

        return out


class GNNRF(AbstractTrafficStateModel):
    """
    Graph Neural Network for ETA/Travel Time Prediction.

    This model uses a single-layer GCN with node encoding and skip connections
    to predict travel times or dwell times for autonomous shuttles.

    The model architecture:
    1. Concatenate lag features, global features, and node one-hot encoding
    2. First linear layer with ReLU and dropout
    3. GCN layer with ReLU
    4. Skip connection concatenating GCN output with original merged features
    5. Merge linear layer with ReLU and dropout
    6. Final linear layer for output

    Args:
        config: Configuration dictionary
        data_feature: Data features dictionary containing adj_mx, num_nodes, etc.
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        # Get data features
        self.adj_mx = data_feature.get('adj_mx')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)
        self._scaler = data_feature.get('scaler')
        self._logger = getLogger()

        # Get config parameters
        self.device = config.get('device', torch.device('cpu'))
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)

        # Model hyperparameters (from original model)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.drop_prob = config.get('dropout', 0.1)
        self.aggregation_function = config.get('aggregation_function', 'mean')

        # Additional feature dimensions for global features
        # Original model uses: vehicle_id (one-hot), time features (4: dow_sin, dow_cos, tod_sin, tod_cos),
        # weather features (3: temp, prcp, wspd)
        self.num_vehicles = config.get('num_vehicles', 1)
        self.num_time_features = config.get('num_time_features', 4)  # dow_sin, dow_cos, tod_sin, tod_cos
        self.num_weather_features = config.get('num_weather_features', 3)  # temp, prcp, wspd

        # Whether to use node encoding (one-hot)
        self.use_node_encoding = config.get('use_node_encoding', True)

        # Calculate input sizes
        # Lag features from X: typically (num_lags * num_lag_features)
        # In LibCity format, X shape is (batch, time_in, num_nodes, features)
        # We use the last time step's features or flatten all
        self.num_lags = config.get('num_lags', 2)
        self.lag_feature_dim = config.get('lag_feature_dim', 1)  # features per lag

        # Input size calculation:
        # - lag features: extracted from X's first channel
        # - global features: vehicle_id + time + weather (from external or X)
        # - node encoding: num_nodes (one-hot)

        # For LibCity, we adapt the input based on available features
        # Input from X: (batch, time, num_nodes, feature_dim)
        # We'll use feature_dim as the per-node features

        # Global feature dimension (extracted from batch or config)
        self.global_feat_dim = self.num_vehicles + self.num_time_features + self.num_weather_features

        # Node encoding dimension
        self.node_encoding_dim = self.num_nodes if self.use_node_encoding else 0

        # Calculate merged feature size
        # lag features (1 value per node) + global features + node encoding
        self.merged_input_size = self.lag_feature_dim + self.global_feat_dim + self.node_encoding_dim

        # Build normalized adjacency matrix
        self._build_adj_matrix()

        # Model layers (following original NodeEncodedGCN_1l architecture)
        # First linear layer: merged features -> hidden
        self.fc_in1 = nn.Linear(self.merged_input_size, self.hidden_dim)

        # GCN layer: hidden -> hidden
        self.conv1 = GCNConvLayer(self.hidden_dim, self.hidden_dim, aggr=self.aggregation_function)

        # Merge layer after skip connection: hidden + merged_input_size -> hidden
        self.fc_merge = nn.Linear(self.hidden_dim + self.merged_input_size, self.hidden_dim)

        # Output layer: hidden -> output_dim
        self.fc_last = nn.Linear(self.hidden_dim, self.output_dim)

        # Regularization layers
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.act_func = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(self.hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(self.hidden_dim)

        # Loss function
        self.loss_func = nn.L1Loss()

        self._logger.info(f'GNNRF model initialized with {self.num_nodes} nodes, '
                         f'hidden_dim={self.hidden_dim}, dropout={self.drop_prob}')
        self._logger.info(f'Input size: lag_features={self.lag_feature_dim}, '
                         f'global_features={self.global_feat_dim}, '
                         f'node_encoding={self.node_encoding_dim}')

    def _build_adj_matrix(self):
        """
        Build normalized adjacency matrix from adj_mx.
        Uses symmetric normalization: D^(-1/2) * A * D^(-1/2)
        """
        if self.adj_mx is None:
            # Create identity matrix if no adjacency provided
            self._logger.warning('No adjacency matrix provided, using identity matrix')
            adj = np.eye(self.num_nodes)
        else:
            adj = np.array(self.adj_mx)

        # Add self-loops
        adj = adj + np.eye(self.num_nodes)

        # Compute degree matrix
        d = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(d, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)

        # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        adj_norm = np.dot(np.dot(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

        self.adj_norm = torch.FloatTensor(adj_norm).to(self.device)

    def _create_node_encoding(self, batch_size):
        """
        Create one-hot node encoding for each node.

        Args:
            batch_size: Number of samples in batch

        Returns:
            Node encoding tensor of shape (batch_size, num_nodes, num_nodes)
        """
        # One-hot encoding for each node
        node_encoding = torch.eye(self.num_nodes, device=self.device)  # (num_nodes, num_nodes)
        # Expand for batch: (batch_size, num_nodes, num_nodes)
        node_encoding = node_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        return node_encoding

    def _extract_features(self, batch):
        """
        Extract and format features from LibCity batch format.

        In LibCity:
        - X shape: (batch_size, input_window, num_nodes, feature_dim)
        - y shape: (batch_size, output_window, num_nodes, output_dim)

        Original GNNRF expects:
        - x: lag features per node (num_nodes, num_lags, features)
        - u: global features (batch, global_feat_dim) - vehicle, time, weather
        - node_encoding: one-hot (num_nodes, num_nodes)

        We adapt by:
        1. Using X's last time step or aggregated features as lag features
        2. Creating global features from available data or using placeholders
        3. Creating node one-hot encoding

        Args:
            batch: Dictionary with 'X', 'y', and optional other keys

        Returns:
            Tuple of (lag_features, global_features, node_encoding)
        """
        x = batch['X']  # (batch_size, input_window, num_nodes, feature_dim)
        batch_size = x.shape[0]

        # Extract lag features from X
        # Use the first feature channel and last time step as primary feature
        # Shape: (batch_size, num_nodes, 1)
        lag_features = x[:, -1, :, 0:self.lag_feature_dim]  # (batch, num_nodes, lag_feature_dim)

        # Create global features
        # In original model: vehicle_id (one-hot) + time (4) + weather (3)
        # Here we create placeholder or use available external features
        # Note: LibCity's Batch class does not support 'in' operator, use try-except instead
        try:
            global_features = batch['global_feat']  # (batch, global_feat_dim)
        except KeyError:
            # Create default global features (zeros)
            global_features = torch.zeros(batch_size, self.global_feat_dim, device=x.device)

            # Try to extract time features if available in X (e.g., additional channels)
            if self.feature_dim > 1:
                # Assume extra channels might contain time/external features
                # This is a heuristic - actual extraction depends on data format
                pass

        # Create node encoding (one-hot)
        if self.use_node_encoding:
            node_encoding = self._create_node_encoding(batch_size)  # (batch, num_nodes, num_nodes)
        else:
            node_encoding = torch.zeros(batch_size, self.num_nodes, 0, device=x.device)

        return lag_features, global_features, node_encoding

    def forward(self, batch):
        """
        Forward pass following original NodeEncodedGCN_1l architecture.

        Args:
            batch: Dictionary with 'X' tensor of shape (batch_size, input_window, num_nodes, feature_dim)

        Returns:
            Predictions of shape (batch_size, output_window, num_nodes, output_dim)
        """
        x_input = batch['X']  # (batch_size, input_window, num_nodes, feature_dim)
        batch_size = x_input.shape[0]

        # Extract features from batch
        lag_features, global_features, node_encoding = self._extract_features(batch)

        # lag_features: (batch, num_nodes, lag_feature_dim)
        # global_features: (batch, global_feat_dim)
        # node_encoding: (batch, num_nodes, num_nodes)

        # Expand global features to match node dimension
        # (batch, global_feat_dim) -> (batch, num_nodes, global_feat_dim)
        global_features_expanded = global_features.unsqueeze(1).expand(-1, self.num_nodes, -1)

        # Concatenate all features: [lag_features, global_features, node_encoding]
        # Shape: (batch, num_nodes, merged_input_size)
        merged_features = torch.cat([lag_features, global_features_expanded, node_encoding], dim=-1)

        # First linear layer with activation and dropout
        h = self.fc_in1(merged_features)  # (batch, num_nodes, hidden_dim)
        h = self.act_func(h)
        h = self.dropout(h)

        # Apply batch normalization (reshape for BatchNorm1d)
        h = h.transpose(1, 2)  # (batch, hidden_dim, num_nodes)
        h = self.batch_norm1(h)
        h = h.transpose(1, 2)  # (batch, num_nodes, hidden_dim)

        # GCN layer
        h = self.conv1(h, self.adj_norm)  # (batch, num_nodes, hidden_dim)
        h = self.act_func(h)

        # Skip connection: concatenate with original merged features
        h = torch.cat([h, merged_features], dim=-1)  # (batch, num_nodes, hidden_dim + merged_input_size)

        # Merge layer with activation and dropout
        h = self.fc_merge(h)  # (batch, num_nodes, hidden_dim)
        h = self.act_func(h)
        h = self.dropout(h)

        # Apply second batch normalization
        h = h.transpose(1, 2)  # (batch, hidden_dim, num_nodes)
        h = self.batch_norm2(h)
        h = h.transpose(1, 2)  # (batch, num_nodes, hidden_dim)

        # Final output layer
        output = self.fc_last(h)  # (batch, num_nodes, output_dim)

        # Reshape to LibCity format: (batch, output_window, num_nodes, output_dim)
        # For single-step output
        output = output.unsqueeze(1)  # (batch, 1, num_nodes, output_dim)

        # If output_window > 1, repeat or handle multi-step prediction
        if self.output_window > 1:
            output = output.expand(-1, self.output_window, -1, -1)

        return output

    def predict(self, batch):
        """
        Make predictions for a batch.

        Args:
            batch: Dictionary with 'X' tensor

        Returns:
            Predictions of shape (batch_size, output_window, num_nodes, output_dim)
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate training loss.

        Uses L1 (MAE) loss following the original model's default.

        Args:
            batch: Dictionary with 'X' and 'y' tensors

        Returns:
            Loss tensor
        """
        y_true = batch['y']  # (batch_size, output_window, num_nodes, feature_dim)
        y_pred = self.predict(batch)  # (batch_size, output_window, num_nodes, output_dim)

        # Inverse transform if scaler is available
        if self._scaler is not None:
            y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
            y_pred = self._scaler.inverse_transform(y_pred[..., :self.output_dim])
        else:
            y_true = y_true[..., :self.output_dim]
            y_pred = y_pred[..., :self.output_dim]

        # Use masked MAE loss (standard in LibCity)
        return loss.masked_mae_torch(y_pred, y_true, 0)
