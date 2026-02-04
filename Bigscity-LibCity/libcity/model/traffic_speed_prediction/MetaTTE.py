# -*- coding: utf-8 -*-
"""
MetaTTE Model for LibCity Framework

Adapted from: /home/wangwenrui/shk/AgentCity/repos/MetaTTE/models/mstte_model.py
Original class: MSMTTEGRUAttModel (TensorFlow-based)

Key Changes:
1. Converted from TensorFlow to PyTorch
2. Inherit from AbstractTrafficStateModel
3. Vectorized batch processing (replaced TensorArray loop with batched operations)
4. Implemented LibCity interface methods: forward, predict, calculate_loss
5. Added configurable parameters via config dict

Model Architecture:
- Input: Trajectory points with (lat_diff, lng_diff, time_id, week_id)
- Embedding: Time (24 hours -> time_emb_dim), Week (7 days -> week_emb_dim)
- Three parallel GRU branches: Spatial, Hour temporal, Week temporal
- Attention mechanism over 3 branches
- MLP head: hidden_size -> 1024 -> 512 -> 256 -> hidden_size -> 1 with residual connection

Author: LibCity Adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class BranchAttention(nn.Module):
    """
    Attention mechanism over three branches (spatial, hour temporal, week temporal).
    Computes attention weights and returns weighted sum of branch features.
    """

    def __init__(self, hidden_size):
        super(BranchAttention, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 3)

    def forward(self, spatial_features, hour_features, week_features):
        """
        Args:
            spatial_features: [batch_size, hidden_size]
            hour_features: [batch_size, hidden_size]
            week_features: [batch_size, hidden_size]

        Returns:
            scored_features: [batch_size, hidden_size]
        """
        # Stack features: [batch_size, hidden_size, 3]
        all_features = torch.stack([spatial_features, hour_features, week_features], dim=2)

        # Compute attention scores using ReLU (as in original TF model)
        # Input to attention: [batch_size, hidden_size]
        # We use spatial_features as query (could also use concatenation)
        scores = F.relu(self.attention_weights(spatial_features))  # [batch_size, 3]
        scores = F.softmax(scores, dim=1)  # [batch_size, 3]

        # Apply attention: [batch_size, hidden_size, 3] * [batch_size, 1, 3] -> sum over dim 2
        scores = scores.unsqueeze(1)  # [batch_size, 1, 3]
        scored_features = torch.sum(all_features * scores, dim=2)  # [batch_size, hidden_size]

        return scored_features


class MLPHead(nn.Module):
    """
    Multi-layer perceptron head with residual connection.
    Architecture: input -> 1024 -> 512 -> 256 -> hidden_size -> 1
    Residual connection adds input to final hidden layer before output.
    """

    def __init__(self, hidden_size, mlp_layers=None):
        super(MLPHead, self).__init__()

        if mlp_layers is None:
            mlp_layers = [1024, 512, 256, hidden_size]

        self.fc1 = nn.Linear(hidden_size, mlp_layers[0])
        self.fc2 = nn.Linear(mlp_layers[0], mlp_layers[1])
        self.fc3 = nn.Linear(mlp_layers[1], mlp_layers[2])
        self.fc4 = nn.Linear(mlp_layers[2], mlp_layers[3])
        self.fc_out = nn.Linear(mlp_layers[3], 1)

        # For residual connection, we need to match dimensions
        self.residual_proj = None
        if hidden_size != mlp_layers[3]:
            self.residual_proj = nn.Linear(hidden_size, mlp_layers[3])

    def forward(self, x):
        """
        Args:
            x: [batch_size, hidden_size]

        Returns:
            output: [batch_size, 1]
        """
        x_shortcut = x

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # Residual connection
        if self.residual_proj is not None:
            x_shortcut = self.residual_proj(x_shortcut)

        output = self.fc_out(x + x_shortcut)

        return output


class MetaTTE(AbstractTrafficStateModel):
    """
    MetaTTE: Multi-Scale Spatio-Temporal Travel Time Estimation Model

    This model uses three parallel GRU branches to capture:
    1. Spatial patterns from lat/lng differences
    2. Hour-of-day temporal patterns
    3. Day-of-week temporal patterns

    An attention mechanism combines the three branches, followed by
    an MLP head with residual connection for final travel time prediction.

    Config Parameters:
        hidden_size (int): Hidden dimension for GRU and embeddings. Default: 128
        time_emb_dim (int): Embedding dimension for hour of day. Default: 128
        week_emb_dim (int): Embedding dimension for day of week. Default: 128
        num_hours (int): Number of hours in a day (for embedding). Default: 24
        num_weekdays (int): Number of days in a week (for embedding). Default: 7
        spatial_input_dim (int): Dimension of spatial input (lat_diff, lng_diff). Default: 2
        num_gru_layers (int): Number of GRU layers. Default: 1
        dropout (float): Dropout rate. Default: 0.0
        bidirectional (bool): Use bidirectional GRU. Default: False
        rnn_type (str): Type of RNN ('GRU' or 'LSTM'). Default: 'GRU'

    Data Feature Requirements:
        scaler: Scaler object for normalization/denormalization
    """

    def __init__(self, config, data_feature):
        super(MetaTTE, self).__init__(config, data_feature)

        self.config = config
        self.data_feature = data_feature
        self.device = config.get('device', torch.device('cpu'))

        # Model hyperparameters
        self.hidden_size = config.get('hidden_size', 128)
        self.time_emb_dim = config.get('time_emb_dim', 128)
        self.week_emb_dim = config.get('week_emb_dim', 128)
        self.num_hours = config.get('num_hours', 24)
        self.num_weekdays = config.get('num_weekdays', 7)
        self.spatial_input_dim = config.get('spatial_input_dim', 2)
        self.num_gru_layers = config.get('num_gru_layers', 1)
        self.dropout = config.get('dropout', 0.0)
        self.bidirectional = config.get('bidirectional', False)
        self.rnn_type = config.get('rnn_type', 'GRU').upper()

        # MLP configuration
        self.mlp_layers = config.get('mlp_layers', [1024, 512, 256, self.hidden_size])

        # Scaler for normalization
        self._scaler = self.data_feature.get('scaler', None)

        # Embeddings for temporal features
        self.time_embedding = nn.Embedding(self.num_hours, self.time_emb_dim)
        self.week_embedding = nn.Embedding(self.num_weekdays, self.week_emb_dim)

        # GRU multiplier for bidirectional
        gru_multiplier = 2 if self.bidirectional else 1

        # Three parallel GRU branches
        rnn_class = nn.GRU if self.rnn_type == 'GRU' else nn.LSTM

        self.spatial_gru = rnn_class(
            input_size=self.spatial_input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_gru_layers,
            batch_first=True,
            dropout=self.dropout if self.num_gru_layers > 1 else 0,
            bidirectional=self.bidirectional
        )

        self.hour_temporal_gru = rnn_class(
            input_size=self.time_emb_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_gru_layers,
            batch_first=True,
            dropout=self.dropout if self.num_gru_layers > 1 else 0,
            bidirectional=self.bidirectional
        )

        self.week_temporal_gru = rnn_class(
            input_size=self.week_emb_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_gru_layers,
            batch_first=True,
            dropout=self.dropout if self.num_gru_layers > 1 else 0,
            bidirectional=self.bidirectional
        )

        # Projection layers for bidirectional GRU output
        gru_output_size = self.hidden_size * gru_multiplier
        if self.bidirectional:
            self.spatial_proj = nn.Linear(gru_output_size, self.hidden_size)
            self.hour_proj = nn.Linear(gru_output_size, self.hidden_size)
            self.week_proj = nn.Linear(gru_output_size, self.hidden_size)

        # Branch attention mechanism
        self.branch_attention = BranchAttention(self.hidden_size)

        # MLP head with residual connection
        self.mlp_head = MLPHead(self.hidden_size, self.mlp_layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)

    def forward(self, batch):
        """
        Forward pass of the MetaTTE model.

        Args:
            batch (dict): Dictionary containing input tensors:
                - 'X': Input tensor of shape [batch_size, seq_len, 4]
                       where the 4 features are (lat_diff, lng_diff, time_id, week_id)
                - Alternatively, individual keys:
                  - 'lat_diff': [batch_size, seq_len]
                  - 'lng_diff': [batch_size, seq_len]
                  - 'time_id': [batch_size, seq_len]
                  - 'week_id': [batch_size, seq_len]

        Returns:
            output: Travel time prediction of shape [batch_size, 1]
        """
        # Extract input data
        # Note: LibCity's Batch class does not implement __contains__, so we use try/except
        try:
            x = batch['X']  # [batch_size, seq_len, 4]
            # Handle different input formats
            if x.dim() == 4:
                # Shape: [batch_size, seq_len, num_nodes, features]
                # For trajectory data, num_nodes might be 1
                x = x.squeeze(2)  # [batch_size, seq_len, features]

            spatial_data = x[:, :, :2].float()  # [batch_size, seq_len, 2] - lat_diff, lng_diff
            time_ids = x[:, :, 2].long()  # [batch_size, seq_len] - time_id (hour)
            week_ids = x[:, :, 3].long()  # [batch_size, seq_len] - week_id (day of week)
        except KeyError:
            # Alternative format with separate keys
            try:
                lat_diff = batch['lat_diff']
            except KeyError:
                lat_diff = batch['current_lati']
            try:
                lng_diff = batch['lng_diff']
            except KeyError:
                lng_diff = batch['current_longi']
            try:
                time_ids = batch['time_id'].long()
            except KeyError:
                time_ids = batch['timeid'].long()
            try:
                week_ids = batch['week_id'].long()
            except KeyError:
                week_ids = batch['weekid'].long()
            spatial_data = torch.stack([lat_diff, lng_diff], dim=-1).float()

        batch_size = spatial_data.size(0)

        # Clamp time_ids and week_ids to valid range
        time_ids = torch.clamp(time_ids, 0, self.num_hours - 1)
        week_ids = torch.clamp(week_ids, 0, self.num_weekdays - 1)

        # Get embeddings for temporal features
        time_embeddings = self.time_embedding(time_ids)  # [batch_size, seq_len, time_emb_dim]
        week_embeddings = self.week_embedding(week_ids)  # [batch_size, seq_len, week_emb_dim]

        # Process through three parallel GRU branches
        # Spatial branch
        spatial_output, spatial_hidden = self.spatial_gru(spatial_data)
        # Use last hidden state
        if self.rnn_type == 'LSTM':
            spatial_hidden = spatial_hidden[0]  # Get h_n from (h_n, c_n)

        if self.bidirectional:
            # Concatenate forward and backward hidden states and project
            spatial_hidden = spatial_hidden.view(self.num_gru_layers, 2, batch_size, self.hidden_size)
            spatial_hidden = torch.cat([spatial_hidden[-1, 0], spatial_hidden[-1, 1]], dim=-1)
            spatial_features = self.spatial_proj(spatial_hidden)
        else:
            spatial_features = spatial_hidden[-1]  # [batch_size, hidden_size]

        # Hour temporal branch
        hour_output, hour_hidden = self.hour_temporal_gru(time_embeddings)
        if self.rnn_type == 'LSTM':
            hour_hidden = hour_hidden[0]

        if self.bidirectional:
            hour_hidden = hour_hidden.view(self.num_gru_layers, 2, batch_size, self.hidden_size)
            hour_hidden = torch.cat([hour_hidden[-1, 0], hour_hidden[-1, 1]], dim=-1)
            hour_features = self.hour_proj(hour_hidden)
        else:
            hour_features = hour_hidden[-1]  # [batch_size, hidden_size]

        # Week temporal branch
        week_output, week_hidden = self.week_temporal_gru(week_embeddings)
        if self.rnn_type == 'LSTM':
            week_hidden = week_hidden[0]

        if self.bidirectional:
            week_hidden = week_hidden.view(self.num_gru_layers, 2, batch_size, self.hidden_size)
            week_hidden = torch.cat([week_hidden[-1, 0], week_hidden[-1, 1]], dim=-1)
            week_features = self.week_proj(week_hidden)
        else:
            week_features = week_hidden[-1]  # [batch_size, hidden_size]

        # Apply branch attention
        scored_features = self.branch_attention(spatial_features, hour_features, week_features)

        # MLP head with residual connection
        output = self.mlp_head(scored_features)  # [batch_size, 1]

        return output

    def predict(self, batch):
        """
        Generate predictions for a batch of data.

        Args:
            batch (dict): Input batch dictionary

        Returns:
            torch.Tensor: Predicted travel times of shape [batch_size, 1]
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate the training loss for a batch.

        Uses Mean Squared Error (MSE) loss between predictions and ground truth.

        Args:
            batch (dict): Input batch dictionary containing:
                - 'X': Input features
                - 'y': Ground truth travel times

        Returns:
            torch.Tensor: Scalar loss value
        """
        predictions = self.forward(batch)

        # Get ground truth
        # Note: LibCity's Batch class does not implement __contains__, so we use try/except
        targets = None
        try:
            targets = batch['y']
            # Handle different target shapes
            if targets.dim() == 4:
                # Shape: [batch_size, time_out, num_nodes, features]
                targets = targets.squeeze()  # Try to match prediction shape
            if targets.dim() > 1:
                targets = targets.view(predictions.size(0), -1)
                if targets.size(1) > 1:
                    # Take mean or first value if multiple time steps
                    targets = targets[:, 0:1]
        except KeyError:
            pass

        if targets is None:
            try:
                targets = batch['time']
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
            except KeyError:
                pass

        if targets is None:
            try:
                targets = batch['travel_time']
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
            except KeyError:
                raise KeyError("Batch must contain 'y', 'time', or 'travel_time' key for targets")

        targets = targets.float().to(predictions.device)

        # Ensure shapes match
        if predictions.shape != targets.shape:
            targets = targets.view_as(predictions)

        # Calculate MSE loss
        loss = F.mse_loss(predictions, targets)

        return loss
