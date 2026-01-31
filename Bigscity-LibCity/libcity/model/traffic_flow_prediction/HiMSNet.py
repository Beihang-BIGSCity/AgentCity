"""
HiMSNet: Hierarchical Multi-Source Spatiotemporal Network
Adapted for LibCity Framework

Original Paper: Multi-source hierarchical spatiotemporal network for traffic prediction
Original Source: /home/wangwenrui/shk/AgentCity/repos/HiMSNet/skytraffic/models/himsnet.py

Key Adaptations:
1. Inherits from AbstractTrafficStateModel for LibCity compatibility
2. Uses LibCity's batch format (batch['X'], batch['y'])
3. Simplified to single-source mode (use_drone=True, use_ld=False, use_global=True)
4. Uses LibCity's scaler for normalization
5. All layer components are embedded in this file for portability

Required Config Parameters:
- d_model: hidden dimension (default: 64)
- dropout: dropout rate (default: 0.1)
- adjacency_hop: K-hop for adjacency matrix (default: 5)
- num_regions: number of regions for attention aggregation (default: 4)
- layernorm: whether to use layer normalization (default: True)
- use_global: whether to use global spatial module (default: True)

Dependencies: torch, torch_geometric, einops
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_geometric.nn as gnn
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

try:
    from einops import rearrange
    HAS_EINOPS = True
except ImportError:
    HAS_EINOPS = False

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss

logger = logging.getLogger('libcity')


# ============================================================================
# Layer Components (ported from original HiMSNet)
# ============================================================================

class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding layer.

    Assumed input shape: (batch_size, seq_len, feature_dim) if batch_first is True.
    """
    def __init__(self, d_model, dropout=0.1, batch_first=True, max_len=500):
        super().__init__()
        self.batch_first = batch_first
        self.batch_dim, self.att_dim = (0, 1) if batch_first else (1, 0)
        init_values = torch.rand(size=(max_len, d_model))
        self.encoding_dict = nn.Parameter(init_values)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        encoding = self.encoding_dict[:x.size(self.att_dim), :].unsqueeze(self.batch_dim)
        x = x + encoding
        return self.dropout(x)


class MLP(nn.Module):
    """Simple MLP with one hidden layer."""

    def __init__(self, in_dim, hid_dim, out_dim, dropout, layernorm=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.norm1 = nn.LayerNorm(hid_dim) if layernorm else nn.Identity()
        self.linear2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = self.norm1(self.linear1(x))
        x = self.dropout(torch.relu(x))
        return self.linear2(x)


class ValueEmbedding(nn.Module):
    """Value embedding layer for traffic time series data.

    Handles missing values by replacing them with learnable tokens.
    For LibCity adaptation, we use a simplified version that assumes clean input.
    """
    def __init__(self, d_model, assume_clean_input=True):
        super().__init__()
        self.d_model = d_model
        self.assume_clean_input = assume_clean_input
        # For single-feature input, we use a linear projection
        self.value_proj = nn.Linear(1, d_model)
        # Learnable tokens for missing values
        self.empty_token = nn.Parameter(torch.randn(d_model))

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (N, T, P, C) where C is feature dimension
        Returns:
            Embedded tensor of shape (N, T, P, d_model)
        """
        N, T, P, C = x.shape
        # Take only the first feature channel and project
        x_val = x[..., 0:1]  # (N, T, P, 1)
        emb = self.value_proj(x_val)  # (N, T, P, d_model)
        return emb

    @property
    def device(self):
        return list(self.parameters())[0].device


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


# ============================================================================
# HiMSNet Model (LibCity Adaptation)
# ============================================================================

class HiMSNet(AbstractTrafficStateModel):
    """
    HiMSNet: Hierarchical Multi-Source Spatiotemporal Network

    This is a LibCity-compatible adaptation of the original HiMSNet model.
    The model is simplified to single-source mode for compatibility with
    LibCity's standard traffic flow prediction task.

    Architecture:
    1. Value Embedding: Projects input features to hidden dimension
    2. Temporal Encoding: Learned positional encoding for temporal dimension
    3. Spatial Encoding: Learned positional encoding for spatial dimension
    4. Temporal Module: LSTM for capturing temporal patterns
    5. Global Spatial Module: GCN for capturing spatial dependencies (optional)
    6. Feature Aggregation: Attention-based aggregation for regional features
    7. Prediction: MLP for final prediction
    """

    def __init__(self, config, data_feature):
        # Extract data features before calling super().__init__
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)
        self.adj_mx = data_feature.get('adj_mx', None)

        super().__init__(config, data_feature)

        # Model hyperparameters from config
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.d_model = config.get('d_model', 64)
        self.dropout = config.get('dropout', 0.1)
        self.adjacency_hop = config.get('adjacency_hop', 5)
        self.num_regions = config.get('num_regions', 4)
        self.layernorm = config.get('layernorm', True)
        self.use_global = config.get('use_global', True)
        self.global_downsample_factor = config.get('global_downsample_factor', 1)
        self.attn_agg = config.get('attn_agg', True)

        # Device configuration
        self.device = config.get('device', torch.device('cpu'))
        self._scaler = data_feature.get('scaler')

        # Initialize edge index from adjacency matrix
        self.edge_index = None
        self._init_edge_index()

        # Initialize cluster IDs for regional aggregation
        self._init_cluster_id()

        # Build model layers
        self._build_model()

        logger.info(f"HiMSNet initialized with {self.num_params} parameters")

    def _init_edge_index(self):
        """Initialize edge index from adjacency matrix with K-hop expansion."""
        if self.adj_mx is None:
            # Create a fully connected adjacency matrix if not provided
            adj = torch.ones(self.num_nodes, self.num_nodes)
        else:
            adj = torch.tensor(self.adj_mx, dtype=torch.float32)

        # Apply K-hop expansion
        if isinstance(self.adjacency_hop, int) and self.adjacency_hop > 1:
            adj_iter = adj.clone()
            adj_init = adj.clone()
            for _ in range(self.adjacency_hop - 1):
                adj_iter = torch.mm(adj_iter.float(), adj_init.float())
                adj_iter = (adj_iter > 0).float()
            adj = adj_iter

        # Convert to edge index format
        self.edge_index = torch.nonzero(adj, as_tuple=False).T
        logger.info(f"Edge index initialized with shape: {self.edge_index.shape}")

    def _init_cluster_id(self):
        """Initialize cluster IDs for regional aggregation."""
        # Simple uniform clustering based on node indices
        self.cluster_id = torch.arange(self.num_nodes) % self.num_regions

    def _build_model(self):
        """Build all model components."""
        d_model = self.d_model
        dropout = self.dropout

        # Spatial encoding (shared across modalities)
        self.spatial_encoding = LearnedPositionalEncoding(
            d_model=d_model,
            max_len=max(self.num_nodes + 10, 1600),
            dropout=dropout
        )

        # Input embedding and temporal encoding
        self.input_embedding = ValueEmbedding(d_model=d_model, assume_clean_input=True)
        self.temporal_encoding = LearnedPositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=self.input_window + 10
        )

        # Temporal module (LSTM)
        self.temporal_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=3,
            batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(d_model) if self.layernorm else nn.Identity()

        # Global spatial module (GCN-based)
        if self.use_global and HAS_TORCH_GEOMETRIC:
            global_dim = d_model // self.global_downsample_factor
            self.channel_down_sample = nn.Linear(d_model, global_dim)
            self.dropout_glb = nn.Dropout(p=dropout)
            self.relu = nn.ReLU()
            self.gcn_1 = gnn.GCNConv(in_channels=global_dim, out_channels=global_dim, node_dim=1)
            self.gcn_2 = gnn.GCNConv(in_channels=global_dim, out_channels=global_dim, node_dim=1)
            self.gcn_3 = gnn.GCNConv(in_channels=global_dim, out_channels=global_dim, node_dim=1)
            self.global_norm1 = nn.LayerNorm(global_dim) if self.layernorm else nn.Identity()
            self.global_norm2 = nn.LayerNorm(global_dim) if self.layernorm else nn.Identity()
            self.channel_up_sample = nn.Linear(global_dim, d_model)
        elif self.use_global:
            logger.warning("torch_geometric not available, global spatial module disabled")
            self.use_global = False

        # Feature dimension after concatenation
        num_modalities = 1 + (1 if self.use_global else 0)  # input + global
        feature_dim = num_modalities * d_model

        # Attention-based regional aggregation
        if self.attn_agg:
            self.query_regional = nn.Parameter(torch.randn(self.num_regions, feature_dim))
            self.feature_aggregator = MultiHeadAttention(
                n_head=num_modalities,
                d_model=feature_dim,
                d_k=d_model,
                d_v=d_model,
                dropout=dropout
            )

        # Prediction heads
        self.prediction = MLP(
            in_dim=feature_dim,
            hid_dim=int(d_model * 2),
            out_dim=self.output_window * self.output_dim,
            dropout=dropout
        )

    @property
    def num_params(self):
        """Return the number of trainable parameters."""
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def forward(self, batch):
        """
        Forward pass for LibCity batch format.

        Args:
            batch: Dictionary containing:
                - 'X': Input tensor of shape (B, T_in, N, C)
                - 'y': Target tensor of shape (B, T_out, N, C) (optional)

        Returns:
            Prediction tensor of shape (B, T_out, N, output_dim)
        """
        x = batch['X']  # (B, T_in, N, C)
        B, T, N, C = x.shape

        # Ensure edge_index is on the correct device
        if self.edge_index is not None and self.edge_index.device != x.device:
            self.edge_index = self.edge_index.to(x.device)
        if self.cluster_id.device != x.device:
            self.cluster_id = self.cluster_id.to(x.device)

        # 1. Input embedding
        x_emb = self.input_embedding(x)  # (B, T, N, d_model)

        # 2. Spatial encoding
        # Reshape for spatial encoding: (B*T, N, d_model)
        x_spatial = x_emb.view(B * T, N, -1)
        x_spatial = self.spatial_encoding(x_spatial)

        # 3. Temporal encoding and LSTM
        # Reshape for temporal processing: (B*N, T, d_model)
        x_temporal = x_spatial.view(B, T, N, -1).permute(0, 2, 1, 3).contiguous()
        x_temporal = x_temporal.view(B * N, T, -1)
        x_temporal = self.temporal_encoding(x_temporal)
        x_temporal, _ = self.temporal_lstm(x_temporal)
        x_temporal = self.temporal_norm(x_temporal)

        # Take the last time step output: (B*N, d_model) -> (B, N, d_model)
        x_out = x_temporal[:, -1, :].view(B, N, -1)

        all_features = [x_out]

        # 4. Global spatial module (GCN)
        if self.use_global and HAS_TORCH_GEOMETRIC:
            x_global = self.channel_down_sample(x_out)
            x_inter = self.dropout_glb(self.relu(self.global_norm1(self.gcn_1(x_global, self.edge_index))))
            x_inter = self.dropout_glb(self.relu(self.global_norm2(self.gcn_2(x_inter, self.edge_index))))
            x_global = x_global + x_inter
            x_global = self.gcn_3(x_global, self.edge_index)
            x_global = self.channel_up_sample(x_global)
            all_features.append(x_global)

        # 5. Concatenate features
        fused_features = torch.cat(all_features, dim=-1)  # (B, N, feature_dim)

        # 6. Prediction
        pred = self.prediction(fused_features)  # (B, N, T_out * output_dim)

        # Reshape to (B, T_out, N, output_dim)
        pred = pred.view(B, N, self.output_window, self.output_dim)
        pred = pred.permute(0, 2, 1, 3).contiguous()  # (B, T_out, N, output_dim)

        return pred

    def predict(self, batch):
        """
        Predict method for LibCity compatibility.

        Args:
            batch: Dictionary containing 'X' and optionally 'y'

        Returns:
            Prediction tensor of shape (B, T_out, N, output_dim)
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate loss for training.

        Args:
            batch: Dictionary containing 'X' and 'y'

        Returns:
            Loss tensor
        """
        y_true = batch['y']
        y_predicted = self.predict(batch)

        # Inverse transform predictions and targets
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        # Use masked MAE loss (standard for traffic prediction)
        return loss.masked_mae_torch(y_predicted, y_true, 0)
