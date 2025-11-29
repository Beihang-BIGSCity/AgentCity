"""
EAC: Expand and Compress: Exploring Tuning Principles for Continual Spatio-Temporal Graph Forecasting
Paper: https://openreview.net/pdf?id=FRzCIlkM7I
Original Code: https://github.com/Onedean/EAC
Venue: ICLR 2025

This implementation adapts the EAC model for LibCity framework.
EAC is a continual learning method for spatio-temporal graph forecasting that uses
prompt tuning with expand and compress principles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


class BatchGCNConv(nn.Module):
    """
    Simple GCN layer for batch processing
    """
    def __init__(self, in_features, out_features, bias=True, gcn=True):
        super(BatchGCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_neigh = nn.Linear(in_features, out_features, bias=bias)
        if not gcn:
            self.weight_self = nn.Linear(in_features, out_features, bias=False)
        else:
            self.register_parameter('weight_self', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight_neigh.reset_parameters()
        if self.weight_self is not None:
            self.weight_self.reset_parameters()

    def forward(self, x, adj):
        """
        Args:
            x: [bs, N, in_features]
            adj: [N, N]
        Returns:
            output: [bs, N, out_features]
        """
        # Compute neighbor aggregation
        input_x = torch.matmul(adj, x)  # [bs, N, in_features]
        output = self.weight_neigh(input_x)  # [bs, N, out_features]

        # Add self connection if using non-GCN variant
        if self.weight_self is not None:
            output += self.weight_self(x)

        return output


class EACCore(nn.Module):
    """
    Core EAC model with expand and compress mechanism
    """
    def __init__(self, args):
        super(EACCore, self).__init__()
        self.dropout = args.get('dropout', 0.0)
        self.rank = args.get('rank', 6)

        # GCN layers
        in_channel = args.get('gcn_in_channel', 12)
        hidden_channel = args.get('gcn_hidden_channel', 64)
        out_channel = args.get('gcn_out_channel', 12)

        self.gcn1 = BatchGCNConv(in_channel, hidden_channel, bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(hidden_channel, out_channel, bias=True, gcn=False)

        # TCN layer
        tcn_kernel_size = args.get('tcn_kernel_size', 3)
        tcn_dilation = args.get('tcn_dilation', 1)
        padding = int((tcn_kernel_size - 1) * tcn_dilation / 2)

        self.tcn1 = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=tcn_kernel_size,
            dilation=tcn_dilation,
            padding=padding
        )

        # Output layer
        y_len = args.get('output_window', 12)
        self.fc = nn.Linear(out_channel, y_len)
        self.activation = nn.GELU()

        # Expand and Compress: low-rank adaptive parameters
        num_nodes = args.get('num_nodes', 207)
        self.U = nn.Parameter(torch.empty(num_nodes, self.rank).uniform_(-0.1, 0.1))
        self.V = nn.Parameter(torch.empty(self.rank, in_channel).uniform_(-0.1, 0.1))

        self.num_nodes = num_nodes
        self.in_channel = in_channel

    def forward(self, x, adj):
        """
        Args:
            x: [bs, N, feature]
            adj: [N, N] normalized adjacency matrix
        Returns:
            output: [bs * N, output_window]
        """
        B, N, T = x.shape

        # Expand: Compute adaptive parameters using low-rank matrices
        adaptive_params = torch.mm(self.U[:N, :], self.V)  # [N, feature_dim]
        x = x + adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape)  # [bs, N, feature]

        # Spatial: GCN layer 1
        x = F.relu(self.gcn1(x, adj))  # [bs, N, hidden_channel]
        x = x.reshape((-1, 1, x.shape[-1]))  # [bs * N, 1, hidden_channel]

        # Temporal: TCN layer
        x = self.tcn1(x)  # [bs * N, 1, hidden_channel]

        # Spatial: GCN layer 2
        x = x.reshape((B, N, -1))  # [bs, N, hidden_channel]
        x = self.gcn2(x, adj)  # [bs, N, out_channel]
        x = x.reshape((-1, x.shape[-1]))  # [bs * N, out_channel]

        # Output projection
        x = self.fc(self.activation(x))  # [bs * N, output_window]
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def expand_adaptive_params(self, new_num_nodes):
        """
        Expand adaptive parameters for new nodes (for continual learning scenarios)
        """
        if new_num_nodes > self.num_nodes:
            new_params = nn.Parameter(
                torch.empty(
                    new_num_nodes - self.num_nodes,
                    self.rank,
                    dtype=self.U.dtype,
                    device=self.U.device
                ).uniform_(-0.1, 0.1)
            )
            self.U = nn.Parameter(torch.cat([self.U, new_params], dim=0))
            self.num_nodes = new_num_nodes


class EAC(AbstractTrafficStateModel):
    """
    EAC model adapted for LibCity framework
    """
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        # Get data features
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 207)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.adj_mx = self.data_feature.get('adj_mx')
        self._logger = getLogger()

        # Get model config
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.device = config.get('device', torch.device('cpu'))

        # EAC specific parameters
        self.dropout = config.get('dropout', 0.0)
        self.rank = config.get('rank', 6)  # Low-rank dimension for expand/compress

        # GCN parameters
        self.gcn_in_channel = self.input_window * self.feature_dim
        self.gcn_hidden_channel = config.get('gcn_hidden_channel', 64)
        self.gcn_out_channel = config.get('gcn_out_channel', self.gcn_in_channel)

        # TCN parameters
        self.tcn_kernel_size = config.get('tcn_kernel_size', 3)
        self.tcn_dilation = config.get('tcn_dilation', 1)

        # Prepare model arguments
        model_args = {
            'dropout': self.dropout,
            'rank': self.rank,
            'gcn_in_channel': self.gcn_in_channel,
            'gcn_hidden_channel': self.gcn_hidden_channel,
            'gcn_out_channel': self.gcn_out_channel,
            'tcn_kernel_size': self.tcn_kernel_size,
            'tcn_dilation': self.tcn_dilation,
            'output_window': self.output_window,
            'num_nodes': self.num_nodes
        }

        # Build model
        self.model = EACCore(model_args)

        # Process adjacency matrix
        if self.adj_mx is not None:
            self.adj_mx = self._normalize_adj(self.adj_mx)
            self.adj_mx = torch.FloatTensor(self.adj_mx).to(self.device)
        else:
            # Create identity matrix if no adjacency matrix provided
            self.adj_mx = torch.eye(self.num_nodes).to(self.device)

        self._logger.info('EAC model initialized')
        self._logger.info(f'Num nodes: {self.num_nodes}, Input window: {self.input_window}, '
                         f'Output window: {self.output_window}')
        self._logger.info(f'Rank: {self.rank}, GCN hidden: {self.gcn_hidden_channel}')

    def _normalize_adj(self, adj):
        """
        Normalize adjacency matrix: A_norm = A / (rowsum(A) + eps)
        """
        adj = adj.astype(np.float32)
        rowsum = np.sum(adj, axis=1, keepdims=True)
        adj_normalized = adj / (rowsum + 1e-6)
        return adj_normalized

    def forward(self, batch):
        """
        Forward pass

        Args:
            batch: dict with key 'X' of shape [batch_size, input_window, num_nodes, feature_dim]

        Returns:
            torch.Tensor: predictions of shape [batch_size, output_window, num_nodes, output_dim]
        """
        # Get input: [batch_size, input_window, num_nodes, feature_dim]
        x = batch['X'].to(self.device)
        batch_size, input_window, num_nodes, feature_dim = x.shape

        # Reshape to [batch_size, num_nodes, input_window * feature_dim]
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, num_nodes, input_window, feature_dim]
        x = x.reshape(batch_size, num_nodes, -1)  # [batch_size, num_nodes, input_window * feature_dim]

        # Forward through model: [batch_size * num_nodes, output_window]
        output = self.model(x, self.adj_mx)

        # Reshape to [batch_size, num_nodes, output_window]
        output = output.reshape(batch_size, num_nodes, self.output_window)

        # Reshape to [batch_size, output_window, num_nodes, 1]
        output = output.permute(0, 2, 1).unsqueeze(-1)

        return output

    def predict(self, batch):
        """
        Prediction interface for LibCity
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate loss

        Args:
            batch: dict with keys 'X' and 'y'

        Returns:
            torch.Tensor: loss value
        """
        y_true = batch['y'].to(self.device)  # [batch_size, output_window, num_nodes, output_dim]
        y_predicted = self.predict(batch)  # [batch_size, output_window, num_nodes, output_dim]

        # Inverse transform if scaler is available
        if self._scaler is not None:
            y_true = self._scaler.inverse_transform(y_true)
            y_predicted = self._scaler.inverse_transform(y_predicted)

        # Calculate MAE loss (as used in original EAC paper)
        return loss.masked_mae_torch(y_predicted, y_true, null_val=0.0)
