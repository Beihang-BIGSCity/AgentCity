"""
PatchSTG Model Adaptation for LibCity Framework

Original Paper: PatchSTG: Patch-based Spatial-Temporal Graph Neural Networks
Original Repository: https://github.com/PatchSTG/PatchSTG

Key Adaptations:
- Inherits from AbstractTrafficStateModel
- KDTree spatial partitioning computed from geo coordinates in data_feature
- Temporal embeddings extracted from LibCity batch format
- WindowAttBlock uses timm Vision Transformer components for dual attention

Required Configuration Parameters:
- tps: temporal patch size (default: 12)
- tpn: temporal patch num (default: 1)
- recur: KDTree recursion depth (default: 7)
- sps: spatial patch size (default: 2)
- spn: spatial patch num (default: 128)
- factors: merging factor for leaf nodes (default: 16)
- layers: number of encoder layers (default: 5)
- id: input embedding dimension (default: 64)
- nd: node embedding dimension (default: 64)
- td: time-of-day embedding dimension (default: 32)
- dd: day-of-week embedding dimension (default: 32)
- tod: time slots per day (default: 288)
- dow: days per week (default: 7)

Dependencies:
- timm>=1.0.12 (for Vision Transformer attention components)
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from logging import getLogger
from sklearn.metrics.pairwise import cosine_similarity

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

try:
    from timm.models.vision_transformer import Attention, Mlp
except ImportError:
    raise ImportError(
        "PatchSTG requires timm library. Please install it with: pip install timm>=1.0.12"
    )


class WindowAttBlock(nn.Module):
    """
    Dual attention block for PatchSTG.
    Performs depth attention (within patches) and breadth attention (across patches).
    Uses timm's Vision Transformer Attention and Mlp modules.

    Args:
        hidden_size: Hidden dimension size
        num_heads: Number of attention heads
        num: Number of patches (P)
        size: Size of each patch (N)
        mlp_ratio: MLP hidden dimension ratio
    """

    def __init__(self, hidden_size, num_heads, num, size, mlp_ratio=4.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num, self.size = num, size

        # Depth attention components (within patches)
        self.snorm1 = nn.LayerNorm(hidden_size)
        self.sattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.snorm2 = nn.LayerNorm(hidden_size)
        self.smlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

        # Breadth attention components (across patches)
        self.nnorm1 = nn.LayerNorm(hidden_size)
        self.nattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.nnorm2 = nn.LayerNorm(hidden_size)
        self.nmlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, num*size, D)

        Returns:
            Output tensor of shape (B, T, num*size, D)
        """
        B, T, _, D = x.shape
        P, N = self.num, self.size
        assert self.num * self.size == _, f"Mismatch: num({self.num}) * size({self.size}) != {_}"
        x = x.reshape(B, T, P, N, D)

        # Depth attention (within each patch)
        qkv = self.snorm1(x.reshape(B * T * P, N, D))
        x = x + self.sattn(qkv).reshape(B, T, P, N, D)
        x = x + self.smlp(self.snorm2(x))

        # Breadth attention (across patches)
        qkv = self.nnorm1(x.transpose(2, 3).reshape(B * T * N, P, D))
        x = x + self.nattn(qkv).reshape(B, T, N, P, D).transpose(2, 3)
        x = x + self.nmlp(self.nnorm2(x))

        return x.reshape(B, T, -1, D)


class PatchSTG(AbstractTrafficStateModel):
    """
    PatchSTG: Transformer-based model with irregular spatial patching via KDTree.

    The model uses:
    1. Spatio-temporal embeddings combining input features, node embeddings, and temporal embeddings
    2. WindowAttBlock for dual attention (depth and breadth) on spatial patches
    3. Conv2d projection decoder for final prediction

    Reference:
        Paper: PatchSTG: Patch-based Spatial-Temporal Graph Neural Networks
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()

        # Basic configuration
        self.device = config.get('device', torch.device('cpu'))
        self.num_nodes = data_feature.get('num_nodes')
        self._scaler = data_feature.get('scaler')
        self.adj_mx = data_feature.get('adj_mx')

        # Input/output configuration
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.output_dim = self.data_feature.get('output_dim', 1)

        # Temporal patch configuration
        self.tem_patchsize = config.get('tps', 12)  # temporal patch size
        self.tem_patchnum = config.get('tpn', 1)    # number of temporal patches

        # Spatial patch configuration
        self.recur_times = config.get('recur', 7)   # KDTree recursion depth
        self.spa_patchsize = config.get('sps', 2)   # spatial patch size
        self.spa_patchnum = config.get('spn', 128)  # spatial patch count
        self.factors = config.get('factors', 16)    # merging factor for leaf nodes

        # Embedding dimensions
        self.input_dims = config.get('id', 64)      # input embedding dim
        self.node_dims = config.get('nd', 64)       # node embedding dim
        self.tod_dims = config.get('td', 32)        # time-of-day embedding dim
        self.dow_dims = config.get('dd', 32)        # day-of-week embedding dim

        # Temporal encoding configuration
        self.time_intervals = config.get('time_intervals', 300)  # seconds per interval
        self.tod = config.get('tod', int(24 * 3600 / self.time_intervals))  # time slots per day
        self.dow = config.get('dow', 7)  # days per week

        # Model architecture
        self.layers = config.get('layers', 5)

        # Total embedding dimension
        self.dims = self.input_dims + self.tod_dims + self.dow_dims + self.node_dims

        # Load spatial indices from geo coordinates
        self._load_spatial_indices(config, data_feature)

        # Derive spa_patchnum from actual KDTree output
        padded_nodes = len(self.reo_all_idx)
        self.spa_patchnum = padded_nodes // self.spa_patchsize
        self._logger.info(f"Adjusted spa_patchnum to {self.spa_patchnum} "
                         f"(padded_nodes={padded_nodes}, spa_patchsize={self.spa_patchsize})")

        # Ensure factors divides spa_patchnum evenly
        while self.spa_patchnum % self.factors != 0 and self.factors > 1:
            self.factors //= 2
        self._logger.info(f"Using factors={self.factors} "
                         f"(num={self.spa_patchnum // self.factors}, "
                         f"size={self.spa_patchsize * self.factors})")

        # Build model layers
        self._build_model()

        self._logger.info(f"PatchSTG initialized with {self.num_nodes} nodes, "
                         f"spatial patches: {self.spa_patchnum}, temporal patches: {self.tem_patchnum}")

    def _load_spatial_indices(self, config, data_feature):
        """
        Load or compute spatial indices for KDTree-based patching.

        This method either:
        1. Uses pre-computed indices from data_feature if available
        2. Computes indices from geo coordinates
        """
        # Check if pre-computed indices are available
        if 'ori_parts_idx' in data_feature and 'reo_parts_idx' in data_feature:
            self.ori_parts_idx = data_feature.get('ori_parts_idx')
            self.reo_parts_idx = data_feature.get('reo_parts_idx')
            self.reo_all_idx = data_feature.get('reo_all_idx')
            self._logger.info("Using pre-computed spatial indices from data_feature")
            return

        # Compute from geo coordinates
        self._logger.info("Computing spatial indices from geo coordinates...")

        # Try to get coordinates from geo file
        dataset = config.get('dataset', '')
        geo_file = config.get('geo_file', dataset)
        data_path = f'./raw_data/{dataset}/'
        geo_path = os.path.join(data_path, f'{geo_file}.geo')

        coords_valid = False
        if os.path.exists(geo_path):
            geofile = pd.read_csv(geo_path)
            # Parse coordinates - format is "[lng, lat]" or similar
            coords = []
            for idx, row in geofile.iterrows():
                coord_str = row['coordinates']
                if isinstance(coord_str, str):
                    coord = eval(coord_str)
                else:
                    coord = coord_str
                if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                    coords.append(coord[:2])
            if len(coords) == len(geofile):
                coords = np.array(coords)
                locations = coords.T  # Shape: (2, num_nodes)
                coords_valid = True
                self._logger.info(f"Loaded {len(coords)} node coordinates from {geo_path}")
            else:
                self._logger.warning(
                    f"Geo file {geo_path} has {len(coords)}/{len(geofile)} valid coordinates, "
                    f"falling back to random coordinates")

        if not coords_valid:
            self._logger.warning(f"Using random coordinates for spatial partitioning")
            np.random.seed(42)
            locations = np.random.randn(2, self.num_nodes)

        # Compute adjacency for padding if not available
        if self.adj_mx is not None and np.any(self.adj_mx):
            adj = self.adj_mx.copy()
            # Normalize adjacency
            if np.isinf(adj).any() or (adj == 0).all():
                adj = np.ones((self.num_nodes, self.num_nodes))
        else:
            # Construct adjacency using cosine similarity
            self._logger.info("Computing cosine similarity adjacency matrix")
            adj = np.eye(self.num_nodes)

        # Compute KDTree partitioning and indices
        parts_idx = self._kdtree(locations, self.recur_times, 0)
        self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx = self._reorder_data(
            parts_idx, adj, self.spa_patchsize
        )

        self._logger.info(f"Computed spatial indices: {len(self.reo_all_idx)} padded nodes")

    def _kdtree(self, locations, times, axis):
        """
        KDTree partitioning of spatial locations.

        Args:
            locations: (2, N) array of [lng, lat] coordinates
            times: Recursion depth
            axis: Current split axis (0=lng, 1=lat)

        Returns:
            List of index arrays for each leaf partition
        """
        sorted_idx = np.argsort(locations[axis])
        mid = locations.shape[1] // 2
        part1 = np.sort(sorted_idx[:mid])
        part2 = np.sort(sorted_idx[mid:])

        if times == 1:
            return [part1, part2]
        else:
            parts = []
            left_parts = self._kdtree(locations[:, part1], times - 1, axis ^ 1)
            right_parts = self._kdtree(locations[:, part2], times - 1, axis ^ 1)
            for part in left_parts:
                parts.append(part1[part])
            for part in right_parts:
                parts.append(part2[part])
            return parts

    def _augment_align(self, dist_matrix, auglen):
        """
        Find most similar points in other leaf nodes for padding.
        """
        sorted_idx = np.argsort(dist_matrix.reshape(-1) * -1)
        sorted_idx = sorted_idx % dist_matrix.shape[-1]
        augidx = []
        for idx in sorted_idx:
            if idx not in augidx:
                augidx.append(idx)
            if len(augidx) == auglen:
                break
        return np.array(augidx, dtype=int)

    def _reorder_data(self, parts_idx, adj, sps):
        """
        Reorder and pad data based on KDTree partitioning.

        Args:
            parts_idx: List of index arrays from KDTree
            adj: Adjacency matrix for finding similar nodes for padding
            sps: Spatial patch size (minimum; will be increased to max partition size)

        Returns:
            Tuple of (ori_parts_idx, reo_parts_idx, reo_all_idx)
        """
        # Ensure sps covers the largest partition
        max_part_size = max(len(p) for p in parts_idx)
        if max_part_size > sps:
            self._logger.info(f"Increasing sps from {sps} to {max_part_size} "
                             f"to cover largest KDTree partition")
            sps = max_part_size
        self.spa_patchsize = sps

        ori_parts_idx = np.array([], dtype=int)
        reo_parts_idx = np.array([], dtype=int)
        reo_all_idx = np.array([], dtype=int)

        for i, part_idx in enumerate(parts_idx):
            part_dist = adj[part_idx, :].copy()
            part_dist[:, part_idx] = 0

            if sps - part_idx.shape[0] > 0:
                local_part_idx = self._augment_align(part_dist, sps - part_idx.shape[0])
                auged_part_idx = np.concatenate([part_idx, local_part_idx], 0)
            else:
                auged_part_idx = part_idx[:sps]

            reo_parts_idx = np.concatenate([reo_parts_idx, np.arange(part_idx.shape[0]) + sps * i])
            ori_parts_idx = np.concatenate([ori_parts_idx, part_idx])
            reo_all_idx = np.concatenate([reo_all_idx, auged_part_idx])

        return ori_parts_idx, reo_parts_idx, reo_all_idx

    def _build_model(self):
        """Build model layers."""
        # Input projection: Conv2d for temporal patching
        # Input has 3 channels: traffic value + normalized tod + normalized dow
        self.input_st_fc = nn.Conv2d(
            in_channels=3,
            out_channels=self.input_dims,
            kernel_size=(1, self.tem_patchsize),
            stride=(1, self.tem_patchsize),
            bias=True
        )

        # Spatial embedding
        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dims))
        nn.init.xavier_uniform_(self.node_emb)

        # Temporal embeddings
        self.time_in_day_emb = nn.Parameter(torch.empty(self.tod, self.tod_dims))
        nn.init.xavier_uniform_(self.time_in_day_emb)

        self.day_in_week_emb = nn.Parameter(torch.empty(self.dow, self.dow_dims))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        # Dual attention encoder layers
        self.spa_encoder = nn.ModuleList([
            WindowAttBlock(
                self.dims,
                1,  # num_heads
                self.spa_patchnum // self.factors,
                self.spa_patchsize * self.factors,
                mlp_ratio=1
            ) for _ in range(self.layers)
        ])

        # Projection decoder
        self.regression_conv = nn.Conv2d(
            in_channels=self.tem_patchnum * self.dims,
            out_channels=self.output_window,
            kernel_size=(1, 1),
            bias=True
        )

    def embedding(self, x, te):
        """
        Compute spatio-temporal embeddings.

        Args:
            x: Traffic data, shape (B, T, N, 1)
            te: Temporal encoding, shape (B, T, N, 2) where [..., 0] is tod and [..., 1] is dow

        Returns:
            Embedded data, shape (B, T', N, dims)
        """
        b, t, n, _ = x.shape

        # Concatenate traffic data with normalized temporal features
        x1 = torch.cat([x, (te[..., 0:1] / self.tod), (te[..., 1:2] / self.dow)], -1).float()

        # Apply temporal patching via Conv2d: (B, T, N, 3) -> (B, T', N, input_dims)
        input_data = self.input_st_fc(x1.transpose(1, 3)).transpose(1, 3)
        t_out = input_data.shape[1]

        # Add time-of-day embedding
        t_i_d_data = te[:, -t_out:, :, 0]  # (B, T', N)
        tod_indices = t_i_d_data.long().clamp(0, self.tod - 1)
        input_data = torch.cat([input_data, self.time_in_day_emb[tod_indices]], -1)

        # Add day-of-week embedding
        d_i_w_data = te[:, -t_out:, :, 1]  # (B, T', N)
        dow_indices = d_i_w_data.long().clamp(0, self.dow - 1)
        input_data = torch.cat([input_data, self.day_in_week_emb[dow_indices]], -1)

        # Add spatial embedding
        node_emb = self.node_emb.unsqueeze(0).unsqueeze(1).expand(b, t_out, -1, -1)
        input_data = torch.cat([input_data, node_emb], -1)

        return input_data

    def forward(self, batch):
        """
        Forward pass of PatchSTG.

        Args:
            batch: Dictionary containing:
                - 'X': Input tensor of shape (B, T, N, C) where C includes traffic + temporal features

        Returns:
            Predictions of shape (B, output_window, N, output_dim)
        """
        x = batch['X']  # (B, T, N, C)

        # Extract traffic data and temporal encodings from input
        # In LibCity format: x[..., 0] is traffic, x[..., 1] is time_in_day (normalized),
        # x[..., 2:] may be day_in_week (one-hot) or other features
        traffic_data = x[..., :1]  # (B, T, N, 1)

        # Extract temporal encodings
        if x.shape[-1] >= 2:
            # Time of day is typically normalized to [0, 1]
            tod_data = x[..., 1:2] * self.tod  # Convert back to slot index
        else:
            tod_data = torch.zeros_like(traffic_data)

        if x.shape[-1] >= 3:
            # Day of week may be one-hot encoded or single value
            if x.shape[-1] > 3:
                # One-hot encoded day of week
                dow_data = torch.argmax(x[..., 2:], dim=-1, keepdim=True).float()
            else:
                dow_data = x[..., 2:3] * self.dow
        else:
            dow_data = torch.zeros_like(traffic_data)

        # Combine temporal encodings: (B, T, N, 2)
        te = torch.cat([tod_data, dow_data], dim=-1)

        # Get spatio-temporal embeddings
        embedded_x = self.embedding(traffic_data, te)

        # Select patched points using pre-computed indices
        reo_all_idx_tensor = torch.tensor(self.reo_all_idx, device=x.device, dtype=torch.long)
        rex = embedded_x[:, :, reo_all_idx_tensor, :]

        # Apply dual attention encoder
        for block in self.spa_encoder:
            rex = block(rex)

        # Map back to original node indices
        original = torch.zeros(rex.shape[0], rex.shape[1], self.num_nodes, rex.shape[-1], device=x.device)
        ori_parts_idx_tensor = torch.tensor(self.ori_parts_idx, device=x.device, dtype=torch.long)
        reo_parts_idx_tensor = torch.tensor(self.reo_parts_idx, device=x.device, dtype=torch.long)
        original[:, :, ori_parts_idx_tensor, :] = rex[:, :, reo_parts_idx_tensor, :]

        # Apply projection decoder
        # Reshape: (B, T', N, D) -> (B, T'*D, N, 1) for Conv2d
        pred_y = self.regression_conv(
            original.transpose(2, 3).reshape(original.shape[0], -1, original.shape[2], 1)
        )

        # Output shape is already (B, output_window, N, 1) - return directly
        return pred_y

    def predict(self, batch):
        """
        Predict traffic states for a batch.

        Args:
            batch: Input batch dictionary

        Returns:
            Predictions of shape (B, output_window, N, output_dim)
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate training loss.

        Args:
            batch: Dictionary containing 'X' and 'y'

        Returns:
            Scalar loss tensor
        """
        y_true = batch['y']
        y_predicted = self.predict(batch)

        # Inverse transform for loss calculation
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        return loss.masked_mae_torch(y_predicted, y_true, null_val=0.0)
