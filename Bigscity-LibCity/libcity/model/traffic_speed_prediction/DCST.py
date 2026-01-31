"""
DCST: Dual Cross-Scale Transformer for Traffic Speed Prediction

This model implements a Dual Cross-Scale Transformer architecture that captures
multi-scale temporal and spatial dependencies in traffic data.

Original Paper: "Make Graph Neural Networks Great Again: A Generic Integration Paradigm of
                Topology-Free Patterns for Traffic Speed Prediction" (IJCAI 2024)

Key Components:
- ViewMerging: Temporal segment merging using window-based concatenation
- Temporal_scale: Cross-attention between original sequence and merged temporal segments
- TemporalATT: Multi-scale temporal attention with configurable segment sizes (1,2,3,4)
- node2grid_encoder: Node-to-grid spatial encoding with learnable per-node projections
- Spatial_scale: Cross-attention between node features and grid representations
- Spatial_ATT: Multi-scale spatial attention (node + configurable grid scales)
- AttentionLayer: Core multi-head attention mechanism
- SelfAttentionLayer: Self-attention with feed-forward network

Adaptations for LibCity:
- Inherits from AbstractTrafficStateModel
- Accepts (config, data_feature) parameters
- Implements predict() and calculate_loss() methods
- Removes hardcoded device assignments
- Makes grid-based spatial processing optional
- Handles LibCity Batch data format

Source Repository: repos/DCST
Main model file: repos/DCST/model/DCST.py
"""

from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class AttentionLayer(nn.Module):
    """Multi-head attention layer for DCST.

    Performs attention across the specified dimension with support for
    different source and target lengths.

    Args:
        model_dim: Model dimension
        num_heads: Number of attention heads
        mask: Whether to apply causal mask
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

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

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
    """Self-attention layer with feed-forward network.

    Args:
        model_dim: Model dimension
        feed_forward_dim: Feed-forward network hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        mask: Whether to apply causal mask
    """

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
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class ViewMerging(nn.Module):
    """Temporal segment merging module.

    Merges temporal points into segments by concatenating every win_size-th element
    and projecting back to model dimension.

    Args:
        win_size: Window size for segment merging
        model_dim: Model dimension
    """

    def __init__(self, win_size, model_dim):
        super(ViewMerging, self).__init__()

        self.win_size = win_size
        self.model_dim = model_dim

        self.temporal_merge = nn.Linear(win_size * model_dim, model_dim)
        self.norm = nn.LayerNorm(win_size * model_dim)

    def forward(self, x):
        # x: (batch_size, num_nodes, time_steps, model_dim)
        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        x = torch.cat(seg_to_merge, -1)
        x = self.norm(x)
        x = self.temporal_merge(x)
        return x


class Temporal_scale(nn.Module):
    """Multi-scale temporal feature extraction module.

    Performs cross-attention between original temporal sequence and
    merged temporal segments.

    Args:
        win_size: Window size for segment merging
        model_dim: Model dimension
        feed_forward_dim: Feed-forward network hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        mask: Whether to apply causal mask
    """

    def __init__(
        self, win_size, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        # Merge Temporal Points to Temporal Segment
        self.merge_layer = ViewMerging(win_size, model_dim)

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)

        # Temporal Segment
        x_seg = self.merge_layer(x)

        residual = x
        out = self.attn(x, x_seg, x_seg)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class TemporalATT(nn.Module):
    """Multi-scale temporal attention module.

    Stacks multiple Temporal_scale blocks with different segment sizes.

    Args:
        model_dim: Model dimension
        ST_scale: Number of temporal scales (1-4)
        feed_forward_dim: Feed-forward network hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        mask: Whether to apply causal mask
    """

    def __init__(
        self, model_dim, ST_scale, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        # Length of Temporal Segment
        if ST_scale == 4:
            self.temporal_size = [1, 2, 3, 4]
        elif ST_scale == 3:
            self.temporal_size = [1, 2, 3]
        elif ST_scale == 2:
            self.temporal_size = [1, 2]
        elif ST_scale == 1:
            self.temporal_size = [1]
        else:
            self.temporal_size = [1, 2, 3, 4]

        self.temporal_blocks = nn.ModuleList()

        for i in range(len(self.temporal_size)):
            self.temporal_blocks.append(
                Temporal_scale(self.temporal_size[i], model_dim, feed_forward_dim, num_heads, dropout, mask)
            )

    def forward(self, x, dim=-2):
        for block in self.temporal_blocks:
            x = block(x, dim)
        return x


class node2grid_encoder(nn.Module):
    """Node to grid spatial encoding module.

    Performs per-node linear transformation followed by grid aggregation
    using the provided view (grid-node mapping) matrix.

    Args:
        view: Grid-node mapping tensor of shape (num_grids, num_nodes)
        d_model: Model dimension
    """

    def __init__(self, view, d_model):
        super(node2grid_encoder, self).__init__()
        self.view = view
        self.d_model = d_model

        num_nodes = view.shape[1]

        # One linear layer per node (using einsum for efficiency)
        self.N2Gencoder_w = nn.Parameter(torch.randn(num_nodes, d_model, d_model) * 0.02)
        self.N2Gencoder_b = nn.Parameter(torch.randn(1, num_nodes, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch = x.shape[0]
        # x: (batch, time, num_nodes, d_model)
        x = torch.einsum("btni,nio->btno", x, self.N2Gencoder_w)

        # Add bias
        t_num = x.shape[1]
        x = x.reshape(batch * t_num, x.shape[2], self.d_model)
        x = x + self.N2Gencoder_b
        x = x.reshape(batch, t_num, x.shape[1], self.d_model)

        # Aggregate to grid
        x_grid_embed = torch.einsum("btnd,gn->btgd", x, self.view)
        x_grid_embed = self.norm(x_grid_embed)
        return x_grid_embed


class Spatial_scale(nn.Module):
    """Multi-scale spatial feature extraction module.

    Performs cross-attention between node features and grid representations.

    Args:
        view: Grid-node mapping tensor (num_grids, num_nodes). If view has size (1, ...), no grid encoding is used.
        model_dim: Model dimension
        feed_forward_dim: Feed-forward network hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        mask: Whether to apply causal mask
    """

    def __init__(
        self, view, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        # Spatial Node to Spatial Grid
        if view.size(0) == 1:
            self.node2grid = None
        else:
            self.node2grid = node2grid_encoder(view, model_dim)

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)

        if self.node2grid is not None:
            x_grid = self.node2grid(x)
        else:
            x_grid = x

        residual = x
        out = self.attn(x, x_grid, x_grid)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class Spatial_ATT(nn.Module):
    """Multi-scale spatial attention module.

    Stacks node-level attention followed by grid-level attentions at different scales.

    Args:
        model_dim: Model dimension
        ST_scale: Number of spatial scales (2-4)
        num_nodes: Number of nodes
        feed_forward_dim: Feed-forward network hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        mask: Whether to apply causal mask
        grid_views: Optional pre-computed grid-node mapping tensors
        device: Torch device
    """

    def __init__(
        self, model_dim, ST_scale, num_nodes, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False,
        grid_views=None, device=None
    ):
        super().__init__()

        self.device = device if device is not None else torch.device('cpu')
        self.num_nodes = num_nodes
        self.ST_scale = ST_scale
        self.grid_views = grid_views

        self.spatial_blocks = nn.ModuleList()

        # First block: node-level self-attention (view size 1)
        self.spatial_blocks.append(
            Spatial_scale(torch.tensor([1.0]), model_dim, feed_forward_dim, num_heads, dropout, mask)
        )

        # Subsequent blocks: grid-level cross-attention
        if grid_views is not None and len(grid_views) > 0:
            for view in grid_views:
                self.spatial_blocks.append(
                    Spatial_scale(view, model_dim, feed_forward_dim, num_heads, dropout, mask)
                )
        else:
            # If no grid views provided, use learned spatial groupings
            self._create_learned_spatial_blocks(model_dim, ST_scale, num_nodes, feed_forward_dim, num_heads, dropout, mask)

    def _create_learned_spatial_blocks(self, model_dim, ST_scale, num_nodes, feed_forward_dim, num_heads, dropout, mask):
        """Create spatial blocks with learned node groupings when grid views are not available."""
        # Size of Spatial Grid (based on original implementation)
        if ST_scale >= 4:
            view_sizes = [160, 80, 40]
        elif ST_scale == 3:
            view_sizes = [160, 80]
        elif ST_scale == 2:
            view_sizes = [160]
        else:
            view_sizes = []

        # Clip view sizes to be at most num_nodes
        view_sizes = [min(vs, num_nodes) for vs in view_sizes if vs <= num_nodes * 2]

        # Create learned grouping matrices
        for view_size in view_sizes:
            # Create a learnable grouping matrix
            grouping_matrix = torch.randn(view_size, num_nodes) * 0.02
            grouping_matrix = F.softmax(grouping_matrix, dim=-1)
            view = nn.Parameter(grouping_matrix)
            self.register_parameter(f'view_{view_size}', view)

            # Create spatial block with this view
            self.spatial_blocks.append(
                Spatial_scale(view, model_dim, feed_forward_dim, num_heads, dropout, mask)
            )

    def forward(self, x, dim=-2):
        for block in self.spatial_blocks:
            x = block(x, dim)
        return x


class DCST(AbstractTrafficStateModel):
    """Dual Cross-Scale Transformer for Traffic Speed Prediction.

    This model captures multi-scale temporal and spatial dependencies using
    a dual cross-scale transformer architecture.

    Config Parameters:
        input_window (int): Input sequence length. Default: 12
        output_window (int): Output prediction length. Default: 12
        steps_per_day (int): Number of time steps per day. Default: 288
        input_dim (int): Input feature dimension. Default: 3
        output_dim (int): Output feature dimension. Default: 1
        input_embedding_dim (int): Input embedding dimension. Default: 24
        tod_embedding_dim (int): Time-of-day embedding dimension. Default: 24
        dow_embedding_dim (int): Day-of-week embedding dimension. Default: 24
        spatial_embedding_dim (int): Spatial embedding dimension. Default: 0
        adaptive_embedding_dim (int): Adaptive embedding dimension. Default: 80
        feed_forward_dim (int): Feed-forward hidden dimension. Default: 256
        num_heads (int): Number of attention heads. Default: 4
        num_layers (int): Not used directly, kept for compatibility. Default: 3
        ST_scale (int): Number of spatio-temporal scales (1-4). Default: 4
        dropout (float): Dropout rate. Default: 0.1
        use_mixed_proj (bool): Whether to use mixed projection for output. Default: True
        use_grid (bool): Whether to use grid-based spatial processing. Default: False
        grid_data_path (str): Path to grid-node mapping files. Default: None

    Data Feature Parameters:
        num_nodes: Number of nodes in the traffic network
        feature_dim: Input feature dimension
        output_dim: Output feature dimension
        scaler: Data scaler for normalization

    Args:
        config: Configuration dictionary
        data_feature: Data feature dictionary
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()

        # Device configuration (from config, not hardcoded)
        self.device = config.get('device', torch.device('cpu'))

        # Data features
        self.num_nodes = data_feature.get('num_nodes')
        self._scaler = data_feature.get('scaler')
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.ext_dim = data_feature.get('ext_dim', 0)

        # Model configuration
        self.in_steps = config.get('input_window', 12)
        self.out_steps = config.get('output_window', 12)
        self.steps_per_day = config.get('steps_per_day', 288)

        # Input/output dimensions
        self.input_dim = config.get('input_dim', 3)
        self.output_dim = config.get('output_dim', 1)

        # Embedding dimensions
        self.input_embedding_dim = config.get('input_embedding_dim', 24)
        self.tod_embedding_dim = config.get('tod_embedding_dim', 24)
        self.dow_embedding_dim = config.get('dow_embedding_dim', 24)
        self.spatial_embedding_dim = config.get('spatial_embedding_dim', 0)
        self.adaptive_embedding_dim = config.get('adaptive_embedding_dim', 80)

        # Calculate model dimension
        self.model_dim = (
            self.input_embedding_dim
            + self.tod_embedding_dim
            + self.dow_embedding_dim
            + self.spatial_embedding_dim
            + self.adaptive_embedding_dim
        )

        # Transformer configuration
        self.num_heads = config.get('num_heads', 4)
        self.dropout = config.get('dropout', 0.1)
        self.feed_forward_dim = config.get('feed_forward_dim', 256)
        self.use_mixed_proj = config.get('use_mixed_proj', True)

        # Scale configuration
        self.ST_scale = config.get('ST_scale', 4)

        # Grid configuration (optional)
        self.use_grid = config.get('use_grid', False)
        self.grid_data_path = config.get('grid_data_path', None)
        self.dataset = config.get('dataset', None)

        # Time feature flags
        self.add_time_in_day = config.get('add_time_in_day', True)
        self.add_day_in_week = config.get('add_day_in_week', True)

        # Load grid views if configured
        self.grid_views = self._load_grid_views() if self.use_grid else None

        # Build model layers
        self._build_model()

        self._logger.info(
            f"DCST initialized with {self.num_nodes} nodes, "
            f"model_dim={self.model_dim}, ST_scale={self.ST_scale}, "
            f"use_grid={self.use_grid}"
        )

    def _load_grid_views(self):
        """Load grid-node mapping matrices from files."""
        if self.grid_data_path is None and self.dataset is None:
            self._logger.warning("Grid enabled but no grid_data_path or dataset specified. Using learned groupings.")
            return None

        # Determine grid sizes based on ST_scale
        if self.ST_scale >= 4:
            view_sizes = [160, 80, 40]
        elif self.ST_scale == 3:
            view_sizes = [160, 80]
        elif self.ST_scale == 2:
            view_sizes = [160]
        else:
            return None

        views = []
        try:
            import pandas as pd

            base_path = self.grid_data_path if self.grid_data_path else f"../data/{self.dataset}"

            for view_size in view_sizes:
                grid_node_path = f"grid_node_{view_size}.csv"
                full_path = os.path.join(base_path, grid_node_path)

                if os.path.exists(full_path):
                    grid_node = pd.read_csv(full_path)
                    grid_node = grid_node.values[:, 1:]  # Skip index column

                    # Normalize grid-node mapping
                    grid_sum = np.sum(grid_node, axis=1, keepdims=True)
                    grid_sum = np.where(grid_sum == 0, 1, grid_sum)  # Avoid division by zero
                    grid_node = grid_node / grid_sum

                    grid_node = torch.from_numpy(grid_node).float().to(self.device)
                    views.append(grid_node)
                else:
                    self._logger.warning(f"Grid file not found: {full_path}. Skipping this scale.")

        except ImportError:
            self._logger.warning("pandas not available for loading grid files. Using learned groupings.")
            return None
        except Exception as e:
            self._logger.warning(f"Error loading grid files: {e}. Using learned groupings.")
            return None

        return views if len(views) > 0 else None

    def _build_model(self):
        """Build the model architecture."""
        # Input projection
        self.input_proj = nn.Linear(self.input_dim, self.input_embedding_dim)

        # Time embeddings
        if self.tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
        if self.dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)

        # Spatial embedding (optional)
        if self.spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)

        # Adaptive embedding (learnable per time step and node)
        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(self.in_steps, self.num_nodes, self.adaptive_embedding_dim))
            )

        # Output projection
        if self.use_mixed_proj:
            self.output_proj = nn.Linear(
                self.in_steps * self.model_dim, self.out_steps * self.output_dim
            )
        else:
            self.temporal_proj = nn.Linear(self.in_steps, self.out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        # DCST: Dual Cross-Scale Transformer
        self.attn_layers_t = TemporalATT(
            self.model_dim, self.ST_scale, self.feed_forward_dim, self.num_heads, self.dropout
        )
        self.attn_layers_s = Spatial_ATT(
            self.model_dim, self.ST_scale, self.num_nodes, self.feed_forward_dim,
            self.num_heads, self.dropout, grid_views=self.grid_views, device=self.device
        )

    def forward(self, batch):
        """Forward pass.

        Args:
            batch: Dictionary containing 'X' with shape (batch, time, nodes, features)

        Returns:
            Predictions with shape (batch, out_steps, nodes, output_dim)
        """
        x = batch['X']
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        # Extract temporal features if present
        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]

        # Extract input features
        x = x[..., :self.input_dim]

        # Input projection
        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)

        # Build feature list
        features = [x]

        # Add time embeddings
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)

        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)

        # Add spatial embedding
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)

        # Add adaptive embedding
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)

        # Concatenate all features
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        # DCST: Temporal then Spatial attention
        x = self.attn_layers_t(x, dim=1)
        x = self.attn_layers_s(x, dim=2)

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
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out

    def predict(self, batch):
        """Make predictions.

        Args:
            batch: Input batch dictionary

        Returns:
            Predictions tensor
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """Calculate the training loss.

        Args:
            batch: Input batch dictionary containing 'X' and 'y'

        Returns:
            Loss tensor
        """
        y_true = batch['y']
        y_predicted = self.predict(batch)

        # Inverse transform for proper loss calculation
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        return loss.masked_mae_torch(y_predicted, y_true, null_val=0.0)
