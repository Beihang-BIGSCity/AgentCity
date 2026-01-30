# -*- coding: utf-8 -*-
"""
MTSTAN (Multi-Task Spatio-Temporal Attention Network) for Travel Time Estimation

This model is adapted from the TensorFlow implementation to PyTorch for LibCity.

Original paper: Multi-Task Spatio-Temporal Attention Network for Travel Time Estimation
Original implementation: TensorFlow 1.12.0

Key components:
- SpatialTransformer: Multi-head attention across spatial dimension
- TemporalTransformer: Multi-head attention across temporal dimension
- ST_Block: Combines spatial and temporal attention with gated fusion
- BridgeTransformer: Connects encoder with decoder for future prediction
- InferenceClass: CNN and dense layers for output

Adaptation Notes:
- Converted TensorFlow operations to PyTorch
- Adapted data format to LibCity's batch dictionary format
- Implemented required AbstractTrafficStateModel methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class LayerNorm(nn.Module):
    """
    Layer normalization module.
    Adapted from TensorFlow's normalize function.
    """
    def __init__(self, hidden_size, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    Adapted from TensorFlow implementation in spatial_attention.py and temporal_attention.py.

    Args:
        num_units: Attention dimension (embedding size)
        num_heads: Number of attention heads
        dropout_rate: Dropout probability
        use_residual: Whether to use residual connection from queries
        return_weights: Whether to return attention weights
    """
    def __init__(self, num_units, num_heads=4, dropout_rate=0.0,
                 use_residual=True, return_weights=False):
        super(MultiHeadAttention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.d_k = num_units // num_heads
        self.use_residual = use_residual
        self.return_weights = return_weights

        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(num_units, num_units)
        self.W_K = nn.Linear(num_units, num_units)
        self.W_V = nn.Linear(num_units, num_units)

        # Residual projection
        self.W_residual = nn.Linear(num_units, num_units)

        # Layer normalization
        self.layer_norm = LayerNorm(num_units)

        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.W_Q.weight, std=0.01)
        nn.init.trunc_normal_(self.W_K.weight, std=0.01)
        nn.init.trunc_normal_(self.W_V.weight, std=0.01)
        nn.init.trunc_normal_(self.W_residual.weight, std=0.01)

    def forward(self, queries, keys, values=None):
        """
        Args:
            queries: [batch, T_q, dim]
            keys: [batch, T_k, dim]
            values: [batch, T_k, dim] (optional, defaults to keys)
        Returns:
            outputs: [batch, T_q, dim]
            attention_weights: [batch, num_heads, T_q, T_k] (if return_weights=True)
        """
        if values is None:
            values = keys

        batch_size = queries.size(0)

        # Linear projections with ReLU activation (as in original TF code)
        Q = F.relu(self.W_Q(queries))  # [batch, T_q, dim]
        K = F.relu(self.W_K(keys))      # [batch, T_k, dim]
        V = F.relu(self.W_V(values))    # [batch, T_k, dim]

        # Split into heads: [batch, T, dim] -> [batch * num_heads, T, d_k]
        Q_ = torch.cat(Q.chunk(self.num_heads, dim=-1), dim=0)
        K_ = torch.cat(K.chunk(self.num_heads, dim=-1), dim=0)
        V_ = torch.cat(V.chunk(self.num_heads, dim=-1), dim=0)

        # Scaled dot-product attention
        # [batch * num_heads, T_q, T_k]
        attn_scores = torch.bmm(Q_, K_.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(self.d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply dropout
        attn_weights = self.dropout(attn_weights)

        # Weighted sum: [batch * num_heads, T_q, d_k]
        outputs = torch.bmm(attn_weights, V_)

        # Concatenate heads: [batch, T_q, dim]
        outputs = torch.cat(outputs.chunk(self.num_heads, dim=0), dim=-1)

        # Residual connection
        if self.use_residual:
            outputs = outputs + F.relu(self.W_residual(queries))

        # Layer normalization
        outputs = self.layer_norm(outputs)

        if self.return_weights:
            # Reshape attention weights: [batch, num_heads, T_q, T_k]
            attn_weights = attn_weights.view(self.num_heads, batch_size, -1, attn_weights.size(-1))
            attn_weights = attn_weights.permute(1, 0, 2, 3)
            return outputs, attn_weights

        return outputs


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network with Conv1D.
    Adapted from TensorFlow's feedforward function.
    """
    def __init__(self, input_dim, hidden_dim=None, output_dim=None,
                 use_residual=True, use_norm=True):
        super(FeedForward, self).__init__()
        hidden_dim = hidden_dim or 4 * input_dim
        output_dim = output_dim or input_dim

        self.use_residual = use_residual
        self.use_norm = use_norm

        # Using Conv1d as in original TF code
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

        if use_norm:
            self.layer_norm = LayerNorm(output_dim)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, dim]
        Returns:
            outputs: [batch, seq_len, dim]
        """
        residual = x

        # [batch, dim, seq_len]
        x = x.transpose(-1, -2)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = x.transpose(-1, -2)  # [batch, seq_len, dim]

        if self.use_residual:
            x = x + residual

        if self.use_norm:
            x = self.layer_norm(x)

        return x


class SpatialTransformer(nn.Module):
    """
    Spatial Transformer encoder.
    Applies multi-head attention across the spatial dimension (sites/nodes).

    Adapted from model/spatial_attention.py
    """
    def __init__(self, emb_size, num_heads=4, num_blocks=1, dropout_rate=0.0):
        super(SpatialTransformer, self).__init__()
        self.num_blocks = num_blocks

        self.attention_layers = nn.ModuleList()
        self.feedforward_layers = nn.ModuleList()

        for _ in range(num_blocks):
            self.attention_layers.append(
                MultiHeadAttention(emb_size, num_heads, dropout_rate,
                                   use_residual=True, return_weights=False)
            )
            self.feedforward_layers.append(
                FeedForward(emb_size, 4 * emb_size, emb_size)
            )

    def forward(self, inputs):
        """
        Args:
            inputs: [batch * time, site_num, emb_size * 2] (concatenated with STE)
        Returns:
            outputs: [batch * time, site_num, emb_size]
        """
        enc = inputs
        for i in range(self.num_blocks):
            enc = self.attention_layers[i](enc, enc)
            enc = self.feedforward_layers[i](enc)
        return enc


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer encoder.
    Applies multi-head attention across the temporal dimension.

    Adapted from model/temporal_attention.py
    """
    def __init__(self, emb_size, num_heads=4, num_blocks=1, dropout_rate=0.0):
        super(TemporalTransformer, self).__init__()
        self.num_blocks = num_blocks

        self.attention_layers = nn.ModuleList()
        self.feedforward_layers = nn.ModuleList()

        for _ in range(num_blocks):
            self.attention_layers.append(
                MultiHeadAttention(emb_size, num_heads, dropout_rate,
                                   use_residual=True, return_weights=True)
            )
            self.feedforward_layers.append(
                FeedForward(emb_size, 4 * emb_size, emb_size)
            )

    def forward(self, hiddens, hidden):
        """
        Args:
            hiddens: encoder hidden states [batch * site_num, input_length, emb_size]
            hidden: query [batch * site_num, input_length, emb_size]
        Returns:
            enc: [batch * site_num, input_length, emb_size]
            attention_weights: list of attention weights from each layer
        """
        attention_weights = []
        enc = hidden
        dec = hiddens

        for i in range(self.num_blocks):
            enc, weights = self.attention_layers[i](enc, dec)
            enc = self.feedforward_layers[i](enc)
            attention_weights.append(weights)

        return enc, attention_weights


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism to combine spatial and temporal features.
    Adapted from model/utils.py gatedFusion function.
    """
    def __init__(self, emb_size):
        super(GatedFusion, self).__init__()
        self.fc_xs = nn.Linear(emb_size, emb_size, bias=True)
        self.fc_xt = nn.Linear(emb_size, emb_size, bias=True)
        self.fc_h = nn.Linear(emb_size, emb_size, bias=False)

    def forward(self, x_s, x_t):
        """
        Args:
            x_s: spatial features [batch, time, site_num, emb_size]
            x_t: temporal features [batch, time, site_num, emb_size]
        Returns:
            fused: [batch, time, site_num, emb_size]
        """
        z = torch.sigmoid(self.fc_xs(x_s) + self.fc_xt(x_t))
        h = torch.tanh(self.fc_h(z * x_s))
        fused = z * h + (1 - z) * x_t
        return fused


class STBlock(nn.Module):
    """
    Spatio-Temporal Block combining spatial and temporal attention.

    Adapted from model/st_block.py

    Architecture:
    1. Temporal attention on transposed input
    2. Spatial attention on concatenated input (with STE)
    3. Gated fusion of spatial and temporal outputs
    """
    def __init__(self, emb_size, site_num, input_length,
                 num_heads=4, num_blocks=1, dropout=0.0):
        super(STBlock, self).__init__()
        self.emb_size = emb_size
        self.site_num = site_num
        self.input_length = input_length

        # Temporal transformer
        self.temporal_transformer = TemporalTransformer(
            emb_size, num_heads, num_blocks, dropout
        )

        # Spatial transformer (input is concatenated with STE, so 2 * emb_size)
        self.spatial_transformer = SpatialTransformer(
            emb_size * 2, num_heads, num_blocks, dropout
        )

        # Feature fusion layers for temporal output
        self.temporal_fc1 = nn.Linear(emb_size, emb_size)
        self.temporal_fc2 = nn.Linear(emb_size, emb_size)

        # Feature fusion layers for spatial output
        self.spatial_fc1 = nn.Linear(emb_size * 2, emb_size)
        self.spatial_fc2 = nn.Linear(emb_size, emb_size)

        # Gated fusion
        self.gated_fusion = GatedFusion(emb_size)

    def forward(self, speed, STE):
        """
        Args:
            speed: [batch, input_length, site_num, emb_size]
            STE: Spatio-temporal embedding [batch, input_length, site_num, emb_size]
        Returns:
            x_f: fused features [batch, input_length, site_num, emb_size]
        """
        batch_size = speed.size(0)

        # Concatenate speed with STE for spatial attention input
        x = torch.cat([speed, STE], dim=-1)  # [batch, time, site_num, 2*emb_size]

        # Temporal correlation
        # Transpose: [batch, time, site_num, emb] -> [batch, site_num, time, emb]
        x_t = speed.permute(0, 2, 1, 3)
        # Reshape for temporal attention: [batch * site_num, time, emb]
        x_t = x_t.reshape(-1, self.input_length, self.emb_size)

        x_t, _ = self.temporal_transformer(x_t, x_t)

        # Feature fusion for temporal
        x_t = F.relu(self.temporal_fc1(x_t))
        x_t = self.temporal_fc2(x_t)

        # Reshape back: [batch, site_num, time, emb] -> [batch, time, site_num, emb]
        x_t = x_t.view(batch_size, self.site_num, self.input_length, self.emb_size)
        x_t = x_t.permute(0, 2, 1, 3)

        # Spatial correlation
        # Reshape: [batch * time, site_num, 2*emb_size]
        x_s = x.reshape(-1, self.site_num, self.emb_size * 2)
        x_s = self.spatial_transformer(x_s)

        # Feature fusion for spatial
        x_s = F.relu(self.spatial_fc1(x_s))
        x_s = self.spatial_fc2(x_s)

        # Reshape back: [batch, time, site_num, emb]
        x_s = x_s.view(batch_size, self.input_length, self.site_num, self.emb_size)

        # Gated fusion
        x_f = self.gated_fusion(x_s, x_t)

        return x_f


class BridgeTransformer(nn.Module):
    """
    Bridge Transformer for connecting encoder outputs with future predictions.
    Uses cross-attention between future queries and historical encodings.

    Adapted from model/bridge.py
    """
    def __init__(self, emb_size, site_num, input_length, output_length,
                 num_heads=4, num_blocks=1, dropout=0.0):
        super(BridgeTransformer, self).__init__()
        self.emb_size = emb_size
        self.site_num = site_num
        self.input_length = input_length
        self.output_length = output_length
        self.num_blocks = num_blocks

        self.attention_layers = nn.ModuleList()
        self.feedforward_layers = nn.ModuleList()

        for _ in range(num_blocks):
            # Cross-attention: Q from future, K and V from history
            self.attention_layers.append(
                MultiHeadAttention(emb_size, num_heads, dropout,
                                   use_residual=False, return_weights=False)
            )
            self.feedforward_layers.append(
                FeedForward(emb_size, 4 * emb_size, emb_size,
                           use_residual=False, use_norm=False)
            )

    def forward(self, X, X_P, X_Q):
        """
        Args:
            X: encoder outputs [batch, input_length, site_num, emb_size]
            X_P: positional encoding for history [batch, input_length, site_num, emb_size]
            X_Q: query for future [batch, output_length, site_num, emb_size]
        Returns:
            outputs: [batch, output_length, site_num, emb_size]
        """
        batch_size = X.size(0)

        # Transpose and reshape for attention
        # [batch, time, site, emb] -> [batch, site, time, emb] -> [batch*site, time, emb]
        X = X.permute(0, 2, 1, 3).reshape(-1, self.input_length, self.emb_size)
        X_P = X_P.permute(0, 2, 1, 3).reshape(-1, self.input_length, self.emb_size)
        X_Q = X_Q.permute(0, 2, 1, 3).reshape(-1, self.output_length, self.emb_size)

        for i in range(self.num_blocks):
            # Cross-attention: Q=future query, K=historical position, V=historical values
            X_Q = self.attention_layers[i](X_Q, X_P, X)
            X_Q = self.feedforward_layers[i](X_Q)

        # Reshape back
        X_out = X_Q.view(batch_size, self.site_num, self.output_length, self.emb_size)
        X_out = X_out.permute(0, 2, 1, 3)  # [batch, output_length, site_num, emb_size]

        return X_out


class InferenceModule(nn.Module):
    """
    Inference module with CNN and dense layers for final output.

    Adapted from model/inference.py
    """
    def __init__(self, emb_size, site_num, output_length):
        super(InferenceModule, self).__init__()
        self.emb_size = emb_size
        self.site_num = site_num
        self.output_length = output_length

        # CNN layer for feature extraction
        self.conv = nn.Conv2d(
            in_channels=output_length,
            out_channels=64,
            kernel_size=(3, emb_size),
            padding=(1, 0)
        )

        # Dense layers for final prediction
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, out_hiddens):
        """
        Args:
            out_hiddens: [batch, output_length, site_num, emb_size]
        Returns:
            results: [batch, site_num, output_length]
        """
        # Transpose for dense layers: [batch, site_num, output_length, emb_size]
        x = out_hiddens.permute(0, 2, 1, 3)

        # Dense layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Squeeze last dimension: [batch, site_num, output_length]
        x = x.squeeze(-1)

        return x


class EmbeddingLayer(nn.Module):
    """
    Embedding layer for temporal features (week, day, hour, minute) and position.

    Adapted from model/embedding.py
    """
    def __init__(self, emb_size, site_num):
        super(EmbeddingLayer, self).__init__()
        self.emb_size = emb_size
        self.site_num = site_num

        # Position embedding
        self.position_emb = nn.Embedding(site_num, emb_size)

        # Temporal embeddings
        self.week_emb = nn.Embedding(5, emb_size)    # 5 weeks (0-4)
        self.day_emb = nn.Embedding(31, emb_size)    # 31 days
        self.hour_emb = nn.Embedding(24, emb_size)   # 24 hours
        self.minute_emb = nn.Embedding(4, emb_size)  # 4 quarter-hours (0-3)

        self._init_weights()

    def _init_weights(self):
        for emb in [self.position_emb, self.week_emb, self.day_emb,
                    self.hour_emb, self.minute_emb]:
            nn.init.trunc_normal_(emb.weight, mean=0, std=1)

    def forward(self, batch_size, total_length, position_ids,
                week_ids, day_ids, hour_ids, minute_ids, device):
        """
        Args:
            batch_size: int
            total_length: input_length + output_length
            position_ids: [site_num]
            week_ids: [batch, total_length * site_num]
            day_ids: [batch, total_length * site_num]
            hour_ids: [batch, total_length * site_num]
            minute_ids: [batch, total_length * site_num]
        Returns:
            STE: Spatio-temporal embedding [batch, total_length, site_num, emb_size]
        """
        # Position embedding: [site_num, emb_size]
        p_emb = self.position_emb(position_ids)
        # Expand to [batch, total_length, site_num, emb_size]
        p_emb = p_emb.unsqueeze(0).unsqueeze(0).expand(
            batch_size, total_length, -1, -1
        )

        # Temporal embeddings
        # Reshape time indices: [batch, total_length, site_num]
        w_emb = self.week_emb(week_ids.view(batch_size, total_length, self.site_num))
        d_emb = self.day_emb(day_ids.view(batch_size, total_length, self.site_num))
        h_emb = self.hour_emb(hour_ids.view(batch_size, total_length, self.site_num))
        m_emb = self.minute_emb(minute_ids.view(batch_size, total_length, self.site_num))

        # Combine embeddings: STE = position + hour + minute
        # (Following original implementation which uses hour and minute primarily)
        STE = p_emb + h_emb + m_emb

        return STE


class FC(nn.Module):
    """
    Fully connected layer with optional batch normalization.
    Adapted from model/utils.py FC function.
    """
    def __init__(self, input_dim, units, activations=None, bn=False):
        super(FC, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = activations or [None] * len(units)
        self.bn = bn

        in_dim = input_dim
        for i, out_dim in enumerate(units):
            self.layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        if bn:
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm1d(u) for u in units
            ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.activations[i] is not None:
                if self.activations[i] == 'relu':
                    x = F.relu(x)
                elif self.activations[i] == 'tanh':
                    x = torch.tanh(x)
                elif self.activations[i] == 'sigmoid':
                    x = torch.sigmoid(x)
        return x


class MTSTAN(AbstractTrafficStateModel):
    """
    Multi-Task Spatio-Temporal Attention Network (MT-STAN) for Travel Time Estimation.

    This model combines:
    - Spatio-temporal attention blocks for learning traffic patterns
    - Bridge transformer for connecting historical and future predictions
    - Multi-task learning for both speed prediction and travel time estimation

    Args:
        config: Configuration dictionary
        data_feature: Data feature dictionary from dataset
    """

    def __init__(self, config, data_feature):
        super(MTSTAN, self).__init__(config, data_feature)

        self.device = config.get('device', torch.device('cpu'))

        # Model hyperparameters
        self.emb_size = config.get('emb_size', 64)
        self.site_num = data_feature.get('num_nodes', config.get('site_num', 108))
        self.input_length = config.get('input_length', 12)
        self.output_length = config.get('output_length', 6)
        self.num_heads = config.get('num_heads', 4)
        self.num_blocks = config.get('num_blocks', 1)
        self.dropout = config.get('dropout', 0.0)
        self.feature_dim = config.get('feature_dim', data_feature.get('feature_dim', 1))

        # Multi-task learning weights
        self.alpha1 = config.get('alpha1', 0.3)  # Speed prediction weight
        self.alpha2 = config.get('alpha2', 0.4)  # Total travel time weight
        self.alpha3 = config.get('alpha3', 0.3)  # Segment travel time weight

        # Get scaler from data feature for normalization
        self._scaler = data_feature.get('scaler')

        # Embedding layers
        self.embedding = EmbeddingLayer(self.emb_size, self.site_num)

        # Input projection: project input features to embedding size
        self.input_fc = FC(
            self.feature_dim,
            [self.emb_size, self.emb_size],
            activations=['relu', None]
        )

        # ST Block for spatio-temporal encoding
        self.st_block = STBlock(
            self.emb_size, self.site_num, self.input_length,
            self.num_heads, self.num_blocks, self.dropout
        )

        # Bridge transformer for future prediction
        self.bridge = BridgeTransformer(
            self.emb_size, self.site_num, self.input_length, self.output_length,
            self.num_heads, self.num_blocks, self.dropout
        )

        # Inference module for speed prediction
        self.inference = InferenceModule(
            self.emb_size, self.site_num, self.output_length
        )

        # Travel time prediction layers
        self.travel_time_fc1 = nn.Linear(self.emb_size, 64)
        self.travel_time_fc2 = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _create_temporal_ids(self, batch):
        """
        Create temporal indices from batch data.
        For LibCity, we need to extract or generate temporal features.
        """
        batch_size = batch['X'].size(0)
        total_length = self.input_length + self.output_length

        # Generate default temporal IDs if not provided
        # These can be overridden by actual data from batch
        position_ids = torch.arange(self.site_num, device=self.device)

        # Default temporal indices (can be customized based on actual data)
        week_ids = torch.zeros(batch_size, total_length * self.site_num,
                               dtype=torch.long, device=self.device)
        day_ids = torch.zeros(batch_size, total_length * self.site_num,
                              dtype=torch.long, device=self.device)
        hour_ids = torch.zeros(batch_size, total_length * self.site_num,
                               dtype=torch.long, device=self.device)
        minute_ids = torch.zeros(batch_size, total_length * self.site_num,
                                 dtype=torch.long, device=self.device)

        # Try to extract temporal features from batch if available
        # Note: LibCity's Batch class requires checking batch.data for key existence
        if 'week' in batch.data:
            week_ids = batch['week'].long().to(self.device)
        if 'day' in batch.data:
            day_ids = batch['day'].long().to(self.device)
        if 'hour' in batch.data:
            hour_ids = batch['hour'].long().to(self.device)
        if 'minute' in batch.data:
            minute_ids = batch['minute'].long().to(self.device)

        return position_ids, week_ids, day_ids, hour_ids, minute_ids

    def forward(self, batch):
        """
        Forward pass of MTSTAN model.

        Args:
            batch: Dictionary containing:
                - 'X': Input tensor [batch, input_length, num_nodes, feature_dim]
                - 'y': Target tensor [batch, output_length, num_nodes, feature_dim]
                - Optional temporal features: 'week', 'day', 'hour', 'minute'

        Returns:
            pre_s: Speed predictions [batch, num_nodes, output_length]
        """
        # Extract input data
        x = batch['X']  # [batch, input_length, num_nodes, feature_dim]
        batch_size = x.size(0)

        # Project input to embedding size
        # [batch, input_length, site_num, emb_size]
        speed = self.input_fc(x)

        # Create temporal embeddings
        position_ids, week_ids, day_ids, hour_ids, minute_ids = self._create_temporal_ids(batch)

        total_length = self.input_length + self.output_length
        STE = self.embedding(
            batch_size, total_length, position_ids,
            week_ids, day_ids, hour_ids, minute_ids, self.device
        )

        # ST Block encoding
        encoder_outs = self.st_block(speed, STE[:, :self.input_length, :, :])

        # Bridge transformer for future prediction
        bridge_outs = self.bridge(
            encoder_outs,
            encoder_outs,
            STE[:, self.input_length:, :, :]
        )

        # Speed prediction through inference module
        pre_s = self.inference(bridge_outs)  # [batch, site_num, output_length]

        return pre_s

    def calculate_loss(self, batch):
        """
        Calculate training loss.

        Args:
            batch: Dictionary containing input data and labels

        Returns:
            loss: Combined loss tensor
        """
        # Forward pass
        pre_s = self.forward(batch)

        # Get target
        y_true = batch['y']  # [batch, output_length, num_nodes, feature_dim]

        # Reshape predictions to match target format
        # pre_s: [batch, num_nodes, output_length] -> [batch, output_length, num_nodes, 1]
        pre_s = pre_s.permute(0, 2, 1).unsqueeze(-1)

        # Calculate MAE loss for speed prediction
        loss = F.l1_loss(pre_s, y_true)

        return loss

    def predict(self, batch):
        """
        Make predictions.

        Args:
            batch: Dictionary containing input data

        Returns:
            predictions: Predicted values in same format as target
        """
        pre_s = self.forward(batch)

        # Reshape: [batch, num_nodes, output_length] -> [batch, output_length, num_nodes, 1]
        pre_s = pre_s.permute(0, 2, 1).unsqueeze(-1)

        return pre_s
