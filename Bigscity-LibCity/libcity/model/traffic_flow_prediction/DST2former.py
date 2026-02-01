"""
DST2former: Dynamic Spatio-Temporal Trend Fusion Transformer for Traffic Prediction

This model is adapted from the original DSTRformer implementation for the LibCity framework.

Original source files:
- Repository: repos/DST2former
- Model file: baselines/DSTRformer/arch/DSTRformer_arch.py
- Helper module: baselines/DSTRformer/arch/mlp.py

Key adaptations for LibCity:
1. Inherits from AbstractTrafficStateModel instead of nn.Module
2. Extracts parameters from config and data_feature dictionaries
3. Handles adjacency matrix format conversion (doubletransition format)
4. Implements forward(), predict(), and calculate_loss() methods
5. Uses LibCity's device handling via config.get('device') instead of hardcoded cuda:0
6. Uses LibCity's scaler for inverse transformation in loss calculation

Input/Output shapes:
- Input: batch['X'] with shape [batch_size, in_steps, num_nodes, input_dim=3]
- Output: predictions with shape [batch_size, out_steps, num_nodes, output_dim=1]

Core components preserved from original:
- Dynamic Trend Fusion Module (graph encoding via GraphMLP)
- Multiple embeddings (TOD, DOW, time series, adaptive)
- Dual attention paths (temporal + spatial via SelfAttentionLayer)
- Transformer layers with augmented attention

Config parameters:
- input_window: number of input time steps (default: 12)
- output_window: number of output time steps (default: 12)
- steps_per_day: time steps per day for embeddings (default: 288)
- input_dim: input feature dimension (default: 1)
- input_embedding_dim: dimension for input projection (default: 24)
- tod_embedding_dim: time-of-day embedding dimension (default: 24)
- dow_embedding_dim: day-of-week embedding dimension (default: 24)
- ts_embedding_dim: time series embedding dimension (default: 28)
- time_embedding_dim: combined time embedding dimension (default: 0)
- adaptive_embedding_dim: adaptive spatial embedding dimension (default: 100)
- node_dim: graph encoder hidden dimension (default: 64)
- feed_forward_dim: attention feed-forward dimension (default: 256)
- out_feed_forward_dim: output attention feed-forward dimension (default: 256)
- num_heads: number of attention heads (default: 4)
- num_layers: number of temporal/spatial attention layers (default: 2)
- num_layers_m: number of augmented attention layers (default: 1)
- mlp_num_layers: number of fusion MLP layers (default: 2)
- dropout: dropout rate (default: 0.1)
- use_mixed_proj: use mixed projection for output (default: True)
"""
import torch
import torch.nn as nn
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


class MultiLayerPerceptron(nn.Module):
    """MLP with residual connection."""
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True),
        )

    def forward(self, input_data):
        hidden = self.fc(input_data)
        hidden = hidden + input_data  # residual
        return hidden


class GraphMLP(nn.Module):
    """Graph MLP for adjacency matrix encoding."""
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x + self.fc2(x)


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.
    But must `src length == K length == V length`.
    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (query @ key) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)

        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)
        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    """Self attention layer with optional cross attention."""
    def __init__(
            self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()
        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.argumented_linear = nn.Linear(model_dim, model_dim)
        self.act1 = nn.GELU()
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, y=None, dim=-2, c=None, augment=False):
        x = x.transpose(dim, -2)
        augmented = None
        # x: (batch_size, ..., length, model_dim)
        if c is not None:
            residual = c
        else:
            residual = x
        if y is None:
            out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
            if augment is True:
                augmented = self.act1(self.argumented_linear(residual))
        else:
            y = y.transpose(dim, -2)
            out = self.attn(y, x, x)

        out = self.dropout1(out)

        if augmented is not None and augment is not False:
            out = self.ln1(residual + out + augmented)
        else:
            out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class DST2former(AbstractTrafficStateModel):
    """
    DST2former: Dual-Stream Spatio-Temporal Transformer for Traffic Forecasting.

    Adapted from DSTRformer for LibCity framework.

    Based on STAEformer architecture with:
    - Multi-head attention with temporal, spatial, and cross attention
    - Graph processing with forward/backward adjacency encoders
    - Multiple embedding types: input projection, ToD, DoW, time series, adaptive
    - Autoregressive attention layers
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        # Data features
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.ext_dim = self.data_feature.get('ext_dim', 0)
        self.output_dim = self.data_feature.get('output_dim', 1)

        # Get adjacency matrix from data_feature
        adj_mx = self.data_feature.get('adj_mx')
        self.adj_mx = self._process_adj_mx(adj_mx)

        # Device handling - use config instead of hardcoded cuda:0
        self.device = config.get('device', torch.device('cpu'))

        # Model hyperparameters from config
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.in_steps = self.input_window
        self.out_steps = self.output_window

        # Steps per day for time embeddings (typically 288 for 5-min intervals)
        self.steps_per_day = config.get('steps_per_day', 288)

        # Input dimension (traffic flow/speed only, excluding time features)
        self.input_dim = config.get('input_dim', self.output_dim)

        # Embedding dimensions
        self.input_embedding_dim = config.get('input_embedding_dim', 24)
        self.tod_embedding_dim = config.get('tod_embedding_dim', 24)
        self.dow_embedding_dim = config.get('dow_embedding_dim', 24)
        self.ts_embedding_dim = config.get('ts_embedding_dim', 28)
        self.time_embedding_dim = config.get('time_embedding_dim', 0)
        self.adaptive_embedding_dim = config.get('adaptive_embedding_dim', 100)
        self.node_dim = config.get('node_dim', 64)

        # Attention hyperparameters
        self.feed_forward_dim = config.get('feed_forward_dim', 256)
        self.out_feed_forward_dim = config.get('out_feed_forward_dim', 256)
        self.num_heads = config.get('num_heads', 4)
        self.num_layers = config.get('num_layers', 2)
        self.num_layers_m = config.get('num_layers_m', 1)
        self.mlp_num_layers = config.get('mlp_num_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        self.use_mixed_proj = config.get('use_mixed_proj', True)

        # Calculate model dimension
        self.model_dim = (
            self.input_embedding_dim
            + self.tod_embedding_dim
            + self.dow_embedding_dim
            + self.adaptive_embedding_dim
            + self.ts_embedding_dim
            + self.time_embedding_dim
        )

        # Build model layers
        self._build_model()

    def _process_adj_mx(self, adj_mx):
        """Process adjacency matrix to forward/backward format (doubletransition).

        LibCity typically provides a single adjacency matrix.
        DSTRformer expects a list of [forward_adj, backward_adj].

        Args:
            adj_mx: Adjacency matrix from LibCity data_feature

        Returns:
            list: [forward_adj_tensor, backward_adj_tensor]
        """
        if adj_mx is None:
            self._logger.warning('No adjacency matrix provided. Using identity matrix.')
            identity = torch.eye(self.num_nodes)
            return [identity, identity]

        if isinstance(adj_mx, list) and len(adj_mx) >= 2:
            # Already in doubletransition format
            forward_adj = torch.FloatTensor(adj_mx[0]) if not isinstance(adj_mx[0], torch.Tensor) else adj_mx[0]
            backward_adj = torch.FloatTensor(adj_mx[1]) if not isinstance(adj_mx[1], torch.Tensor) else adj_mx[1]
            return [forward_adj, backward_adj]

        # Convert to tensor if needed
        if not isinstance(adj_mx, torch.Tensor):
            adj_mx = torch.FloatTensor(adj_mx)

        # Create forward and backward adjacency (transpose)
        forward_adj = adj_mx
        backward_adj = adj_mx.T

        return [forward_adj, backward_adj]

    def _build_model(self):
        """Build all model layers."""
        # Input projection
        if self.input_embedding_dim > 0:
            self.input_proj = nn.Linear(self.input_dim, self.input_embedding_dim)

        # Time embeddings
        if self.tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
        if self.dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        if self.time_embedding_dim > 0:
            self.time_embedding = nn.Embedding(7 * self.steps_per_day, self.time_embedding_dim)

        # Adaptive embedding
        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(self.in_steps, self.num_nodes, self.adaptive_embedding_dim))
            )

        # Graph adjacency encoders
        self.adj_mx_forward_encoder = nn.Sequential(
            GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim)
        )
        self.adj_mx_backward_encoder = nn.Sequential(
            GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim)
        )

        # Output projection
        if self.use_mixed_proj:
            self.output_proj = nn.Linear(
                self.in_steps * self.model_dim, self.out_steps * self.output_dim
            )
        else:
            self.temporal_proj = nn.Linear(self.in_steps, self.out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        # Temporal attention layers
        self.attn_layers_t = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, self.feed_forward_dim, self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ])

        # Spatial attention layers
        self.attn_layers_s = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, self.feed_forward_dim, self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ])

        # Cross attention layers
        self.attn_layers_c = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, self.feed_forward_dim, self.num_heads, self.dropout)
        ])

        # Autoregressive attention layers
        self.ar_attn = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, self.out_feed_forward_dim, self.num_heads, self.dropout)
            for _ in range(self.num_layers_m)
        ])

        # Time series embedding layer
        if self.ts_embedding_dim > 0:
            self.time_series_emb_layer = nn.Conv2d(
                in_channels=self.input_dim * self.in_steps,
                out_channels=self.ts_embedding_dim,
                kernel_size=(1, 1),
                bias=True
            )

        # Fusion model for graph and adaptive embeddings
        self.fusion_model = nn.Sequential(
            *[MultiLayerPerceptron(
                input_dim=self.adaptive_embedding_dim + 2 * self.node_dim,
                hidden_dim=self.adaptive_embedding_dim + 2 * self.node_dim,
                dropout=0.2
            ) for _ in range(self.mlp_num_layers)],
            nn.Linear(
                in_features=self.adaptive_embedding_dim + 2 * self.node_dim,
                out_features=self.adaptive_embedding_dim,
                bias=True
            )
        )

    def forward(self, batch):
        """Forward pass adapted for LibCity batch format.

        Args:
            batch: dict with keys 'X' and 'y'
                - X shape: (batch_size, in_steps, num_nodes, feature_dim)
                  where feature_dim includes [traffic_value, tod, dow, ...]

        Returns:
            torch.Tensor: predictions of shape (batch_size, out_steps, num_nodes, output_dim)
        """
        # Extract input from batch
        x = batch['X']  # (batch_size, in_steps, num_nodes, feature_dim)
        batch_size, _, num_nodes, _ = x.shape

        # Extract time features if available
        # LibCity format: [..., 0:output_dim] is traffic data
        # [..., -2] is typically time_of_day (normalized 0-1)
        # [..., -1] is typically day_of_week (0-6)
        if x.shape[-1] > self.input_dim:
            if self.tod_embedding_dim > 0:
                tod = x[..., self.input_dim]  # Assume tod is at position input_dim
            if self.dow_embedding_dim > 0:
                dow = x[..., self.input_dim + 1] if x.shape[-1] > self.input_dim + 1 else torch.zeros_like(x[..., 0])
            if self.time_embedding_dim > 0:
                tod = x[..., self.input_dim]
                dow = x[..., self.input_dim + 1] if x.shape[-1] > self.input_dim + 1 else torch.zeros_like(x[..., 0])
        else:
            # No time features available, use zeros
            tod = torch.zeros(batch_size, self.in_steps, num_nodes, device=x.device)
            dow = torch.zeros(batch_size, self.in_steps, num_nodes, device=x.device)

        # Extract traffic data only
        x = x[..., :self.input_dim]

        # Time series embedding
        if self.ts_embedding_dim > 0:
            input_data = x.transpose(1, 2).contiguous()
            input_data = input_data.view(
                batch_size, self.num_nodes, -1
            ).transpose(1, 2).unsqueeze(-1)
            # B L*input_dim N 1
            time_series_emb = self.time_series_emb_layer(input_data)
            time_series_emb = time_series_emb.transpose(1, -1).expand(
                batch_size, self.in_steps, self.num_nodes, self.ts_embedding_dim
            )

        # Input projection
        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]

        # Add time series embedding
        if self.ts_embedding_dim > 0:
            features.append(time_series_emb)

        # Add time-of-day embedding
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long().clamp(0, self.steps_per_day - 1)
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)

        # Add day-of-week embedding
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long().clamp(0, 6)
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)

        # Add combined time embedding
        if self.time_embedding_dim > 0:
            time_emb = self.time_embedding(
                ((tod + dow * 7) * self.steps_per_day).long().clamp(0, 7 * self.steps_per_day - 1)
            )
            features.append(time_emb)

        # Add adaptive embedding
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)

        # Concatenate all features
        temporal_x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        spatial_x = temporal_x.clone()

        # Apply temporal and spatial attention layers
        for attn_t, attn_s in zip(self.attn_layers_t, self.attn_layers_s):
            temporal_x = attn_t(temporal_x, dim=1)
            spatial_x = attn_s(spatial_x, dim=2)

        # Cross attention
        for attn in self.attn_layers_c:
            x = attn(temporal_x, spatial_x, dim=2)

        # Graph processing
        if self.node_dim > 0:
            adp_graph = x[..., -self.adaptive_embedding_dim:]
            x = x[..., :self.model_dim - self.adaptive_embedding_dim]

            # Forward adjacency encoding
            node_forward = self.adj_mx[0].to(self.device)
            node_forward = self.adj_mx_forward_encoder(node_forward.unsqueeze(0)).expand(
                batch_size, self.in_steps, -1, -1
            )

            # Backward adjacency encoding
            node_backward = self.adj_mx[1].to(self.device)
            node_backward = self.adj_mx_backward_encoder(node_backward.unsqueeze(0)).expand(
                batch_size, self.in_steps, -1, -1
            )

            # Fuse graph and adaptive embeddings
            graph = torch.cat([adp_graph, node_forward, node_backward], dim=-1)
            graph = self.fusion_model(graph)

            x = torch.cat([x, graph], dim=-1)

        # Autoregressive attention
        for attn in self.ar_attn:
            x = attn(x, dim=2, augment=True)

        # Output projection
        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(out)  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(out.transpose(1, 3))  # (batch_size, out_steps, num_nodes, output_dim)

        return out

    def predict(self, batch):
        """Make predictions for a batch.

        Args:
            batch: LibCity batch dict

        Returns:
            torch.Tensor: predictions
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """Calculate training loss.

        Args:
            batch: LibCity batch dict with 'X' and 'y'

        Returns:
            torch.Tensor: loss value
        """
        y_true = batch['y']
        y_predicted = self.predict(batch)

        # Inverse transform to original scale
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        return loss.masked_mae_torch(y_predicted, y_true, 0)
