import torch
import torch.nn as nn
import numpy as np
from logging import getLogger
from timm.models.vision_transformer import Attention, Mlp
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


class WindowAttBlock(nn.Module):
    """
    Window Attention Block with dual attention mechanism:
    - Depth attention: attention within each spatial patch
    - Breadth attention: attention across different spatial patches
    """
    def __init__(self, hidden_size, num_heads, num, size, mlp_ratio=4.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num, self.size = num, size

        # Depth attention (within patches)
        self.nnorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.nnorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nmlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

        # Breadth attention (across patches)
        self.snorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.snorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.smlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

    def forward(self, x):
        B, T, _, D = x.shape
        # P: patch num and N: patch size
        P, N = self.num, self.size
        assert self.num * self.size == _
        x = x.reshape(B, T, P, N, D)

        # Depth attention
        qkv = self.snorm1(x.reshape(B*T*P, N, D))
        x = x + self.sattn(qkv).reshape(B, T, P, N, D)
        x = x + self.smlp(self.snorm2(x))

        # Breadth attention
        qkv = self.nnorm1(x.transpose(2, 3).reshape(B*T*N, P, D))
        x = x + self.nattn(qkv).reshape(B, T, N, P, D).transpose(2, 3)
        x = x + self.nmlp(self.nnorm2(x))

        return x.reshape(B, T, -1, D)


class PatchSTG(AbstractTrafficStateModel):
    """
    Paper: Efficient Large-Scale Traffic Forecasting with Transformers: A Spatial Data Management Perspective
    Official Code: https://github.com/LMissher/PatchSTG
    Venue: SIGKDD 2025

    This model uses irregular spatial patching to reduce complexity in attention mechanism
    for efficient large-scale traffic forecasting.
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        # Get data features
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.adj_mx = self.data_feature.get('adj_mx')
        self.ext_dim = self.data_feature.get('ext_dim', 1)
        self._logger = getLogger()

        # Get model config
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.device = config.get('device', torch.device('cpu'))

        # PatchSTG specific parameters
        self.tem_patchsize = config.get('tps', 12)  # temporal patch size
        self.tem_patchnum = config.get('tpn', 1)    # temporal patch number
        self.recur_times = config.get('recur', 7)   # KDTree depth
        self.spa_patchsize = config.get('sps', 2)   # spatial patch size (leaf size)
        self.spa_patchnum = config.get('spn', 128)  # spatial patch number (2^recur)
        self.layers = config.get('layers', 5)
        self.factors = config.get('factors', 16)    # factor for merging patches

        # Embedding dimensions
        self.input_dims = config.get('id', 64)
        self.node_dims = config.get('nd', 64)
        self.tod_dims = config.get('td', 32)
        self.dow_dims = config.get('dd', 32)

        # Time dimensions
        self.tod = config.get('tod', 96)  # time of day slots (e.g., 96 for 15-min intervals)
        self.dow = config.get('dow', 7)   # day of week

        # Total embedding dimension
        dims = self.input_dims + self.tod_dims + self.dow_dims + self.node_dims

        # Get geo data for spatial patching
        geo = data_feature.get('geo')
        if geo is not None:
            self._logger.info("Constructing spatial patches using geo information...")
            self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx = self._construct_spatial_patches(geo)
        else:
            self._logger.warning("No geo information found, using sequential patching...")
            self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx = self._construct_sequential_patches()

        # Spatio-temporal embedding layers (section 4.1 in paper)
        # Input embedding with temporal patching
        self.input_st_fc = nn.Conv2d(
            in_channels=3,  # traffic value + time of day + day of week
            out_channels=self.input_dims,
            kernel_size=(1, self.tem_patchsize),
            stride=(1, self.tem_patchsize),
            bias=True
        )

        # Spatial embedding (node embedding)
        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dims))
        nn.init.xavier_uniform_(self.node_emb)

        # Temporal embeddings
        self.time_in_day_emb = nn.Parameter(torch.empty(self.tod, self.tod_dims))
        nn.init.xavier_uniform_(self.time_in_day_emb)

        self.day_in_week_emb = nn.Parameter(torch.empty(self.dow, self.dow_dims))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        # Dual attention encoder (section 4.3 in paper)
        self.spa_encoder = nn.ModuleList([
            WindowAttBlock(dims, 1, self.spa_patchnum//self.factors,
                          self.spa_patchsize*self.factors, mlp_ratio=1)
            for _ in range(self.layers)
        ])

        # Projection decoder (section 4.4 in paper)
        self.regression_conv = nn.Conv2d(
            in_channels=self.tem_patchnum * dims,
            out_channels=self.output_window,
            kernel_size=(1, 1),
            bias=True
        )

        self._logger.info(f"PatchSTG initialized with {self.num_nodes} nodes, "
                         f"input_window={self.input_window}, output_window={self.output_window}")
        self._logger.info(f"Spatial patches: {self.spa_patchnum}, patch size: {self.spa_patchsize}")

    def _construct_spatial_patches(self, geo):
        """
        Construct spatial patches using KD-Tree on geo locations.

        Args:
            geo: DataFrame with geo information including 'coordinates' column

        Returns:
            ori_parts_idx: original indices of nodes
            reo_parts_idx: reordered indices within patches
            reo_all_idx: all indices including padded nodes
        """
        from sklearn.metrics.pairwise import cosine_similarity

        # Extract coordinates
        coordinates = []
        for idx in range(len(geo)):
            coord = eval(geo.loc[idx, 'coordinates']) if isinstance(geo.loc[idx, 'coordinates'], str) else geo.loc[idx, 'coordinates']
            if isinstance(coord, (list, tuple)):
                coordinates.append(coord)
            else:
                coordinates.append([0, 0])  # Default coordinates if parsing fails

        coordinates = np.array(coordinates)
        if coordinates.shape[1] >= 2:
            locations = coordinates[:, :2].T  # [2, N] array (lng, lat)
        else:
            locations = coordinates.T

        # Build KD-Tree for spatial partitioning
        parts_idx, mxlen = self._kdTree(locations, self.recur_times, 0)

        # Construct adjacency matrix for padding if not available
        if self.adj_mx is None:
            self._logger.warning("No adjacency matrix found, creating identity matrix")
            adj = np.eye(self.num_nodes)
        else:
            adj = self.adj_mx

        # Reorder and pad data
        ori_parts_idx, reo_parts_idx, reo_all_idx = self._reorderData(parts_idx, mxlen, adj, self.spa_patchsize)

        return ori_parts_idx, reo_parts_idx, reo_all_idx

    def _construct_sequential_patches(self):
        """
        Construct sequential patches when geo information is not available.
        """
        # Simple sequential patching
        ori_parts_idx = np.arange(self.num_nodes)
        reo_parts_idx = np.arange(self.num_nodes)

        # Pad to match spa_patchnum * spa_patchsize
        total_size = self.spa_patchnum * self.spa_patchsize
        if self.num_nodes < total_size:
            padding = np.arange(self.num_nodes) % self.num_nodes
            padding = padding[:total_size - self.num_nodes]
            reo_all_idx = np.concatenate([ori_parts_idx, padding])
        else:
            reo_all_idx = ori_parts_idx[:total_size]

        return ori_parts_idx, reo_parts_idx, reo_all_idx

    def _kdTree(self, locations, times, axis):
        """
        KD-Tree partitioning for spatial data.

        Args:
            locations: [2, N] array of coordinates
            times: recursion depth
            axis: 0 for lng, 1 for lat

        Returns:
            parts: list of node indices in each partition
            mxlen: maximum partition size
        """
        sorted_idx = np.argsort(locations[axis])
        part1 = np.sort(sorted_idx[:locations.shape[1]//2])
        part2 = np.sort(sorted_idx[locations.shape[1]//2:])

        if times == 1:
            return [part1, part2], max(part1.shape[0], part2.shape[0])
        else:
            left_parts, lmxlen = self._kdTree(locations[:, part1], times-1, axis^1)
            right_parts, rmxlen = self._kdTree(locations[:, part2], times-1, axis^1)
            parts = []
            for part in left_parts:
                parts.append(part1[part])
            for part in right_parts:
                parts.append(part2[part])
            return parts, max(lmxlen, rmxlen)

    def _augmentAlign(self, dist_matrix, auglen):
        """Find the most similar points in other leaf nodes for padding."""
        sorted_idx = np.argsort(dist_matrix.reshape(-1) * -1)
        sorted_idx = sorted_idx % dist_matrix.shape[-1]
        augidx = []
        for idx in sorted_idx:
            if idx not in augidx:
                augidx.append(idx)
            if len(augidx) == auglen:
                break
        return np.array(augidx, dtype=int)

    def _reorderData(self, parts_idx, mxlen, adj, sps):
        """
        Reorder and pad data based on spatial partitions.

        Args:
            parts_idx: list of node indices in each partition
            mxlen: maximum partition size
            adj: adjacency matrix
            sps: spatial patch size for padding

        Returns:
            ori_parts_idx: original indices
            reo_parts_idx: reordered indices within patches
            reo_all_idx: all indices including padded
        """
        ori_parts_idx = np.array([], dtype=int)
        reo_parts_idx = np.array([], dtype=int)
        reo_all_idx = np.array([], dtype=int)

        for i, part_idx in enumerate(parts_idx):
            part_dist = adj[part_idx, :].copy()
            part_dist[:, part_idx] = 0

            if sps - part_idx.shape[0] > 0:
                # Need padding
                local_part_idx = self._augmentAlign(part_dist, sps - part_idx.shape[0])
                auged_part_idx = np.concatenate([part_idx, local_part_idx], 0)
            else:
                auged_part_idx = part_idx

            reo_parts_idx = np.concatenate([reo_parts_idx, np.arange(part_idx.shape[0]) + sps*i])
            ori_parts_idx = np.concatenate([ori_parts_idx, part_idx])
            reo_all_idx = np.concatenate([reo_all_idx, auged_part_idx])

        return ori_parts_idx.astype(int), reo_parts_idx.astype(int), reo_all_idx.astype(int)

    def embedding(self, x, te):
        """
        Spatio-temporal embedding (section 4.1 in paper).

        Args:
            x: [B, T, N, 1] input traffic
            te: [B, T, N, 2] time information (time of day, day of week)

        Returns:
            input_data: [B, T, N, D] embedded data
        """
        b, t, n, _ = x.shape

        # Concatenate traffic value + normalized time features
        x1 = torch.cat([x, (te[..., 0:1]/self.tod), (te[..., 1:2]/self.dow)], -1).float()
        # Apply temporal patching through Conv2d
        input_data = self.input_st_fc(x1.transpose(1, 3)).transpose(1, 3)
        t, d = input_data.shape[1], input_data.shape[-1]

        # Add time of day embedding
        t_i_d_data = te[:, -input_data.shape[1]:, :, 0]
        input_data = torch.cat([input_data, self.time_in_day_emb[(t_i_d_data).type(torch.LongTensor)]], -1)

        # Add day of week embedding
        d_i_w_data = te[:, -input_data.shape[1]:, :, 1]
        input_data = torch.cat([input_data, self.day_in_week_emb[(d_i_w_data).type(torch.LongTensor)]], -1)

        # Add spatial embedding
        node_emb = self.node_emb.unsqueeze(0).unsqueeze(1).expand(b, t, -1, -1)
        input_data = torch.cat([input_data, node_emb], -1)

        return input_data

    def forward(self, batch):
        """
        Forward pass of PatchSTG.

        Args:
            batch: dict with keys 'X' and optionally temporal features

        Returns:
            pred_y: [B, T_out, N, 1] predicted traffic
        """
        # Extract input from batch
        # batch['X']: [B, T_in, N, F] where F includes traffic + temporal features
        x = batch['X']  # [B, T_in, N, F]
        batch_size, input_len, num_nodes, feature_dim = x.shape

        # Extract traffic values (first feature) and temporal information
        traffic = x[..., :1]  # [B, T_in, N, 1]

        # Extract or construct temporal embeddings
        if feature_dim >= 3:
            # Time of day and day of week are included in the data
            te = x[..., 1:3]  # [B, T_in, N, 2]
        else:
            # Generate default temporal embeddings (this is a fallback)
            te = torch.zeros(batch_size, input_len, num_nodes, 2, device=x.device)
        # Spatio-temporal embedding (section 4.1 in paper)
        embeded_x = self.embedding(traffic, te)  # [B, T, N, D]

        # Select patched points
        rex = embeded_x[:, :, self.reo_all_idx, :]  # [B, T, N_patched, D]

        # Dual attention encoder (section 4.3 in paper)
        for block in self.spa_encoder:
            rex = block(rex)

        # Map back to original indices
        orginal = torch.zeros(rex.shape[0], rex.shape[1], self.num_nodes, rex.shape[-1],
                             device=x.device, dtype=rex.dtype)
        orginal[:, :, self.ori_parts_idx, :] = rex[:, :, self.reo_parts_idx, :]

        # Projection decoder (section 4.4 in paper)
        # orginal shape: [B, T_patch, N, D]
        # Reshape to [B, T_patch*D, N, 1] for conv
        pred_y = self.regression_conv(
            orginal.transpose(1, 3).reshape(orginal.shape[0], -1, orginal.shape[2], 1)
        )
        # pred_y shape after conv: [B, T_out, N, 1]

        return pred_y

    def calculate_loss(self, batch):
        """
        Calculate loss between predictions and ground truth.

        Args:
            batch: dict with keys 'X' (input) and 'y' (ground truth)

        Returns:
            loss: scalar tensor
        """
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        """
        Make predictions.

        Args:
            batch: dict with keys 'X' (input)

        Returns:
            predictions: [B, T_out, N, F] predicted traffic
        """
        return self.forward(batch)
