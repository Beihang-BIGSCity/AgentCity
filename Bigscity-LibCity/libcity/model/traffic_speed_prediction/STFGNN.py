# -*- coding: utf-8 -*-
"""
STFGNN: Spatial-Temporal Fusion Graph Neural Networks for Traffic Flow Forecasting (AAAI 2021)

PyTorch re-authoring for the LibCity framework.

The original reference implementation (https://github.com/MengzhangLI/STFGNN,
file ``models/stsgcn_4n_res.py``) is written in the **Apache MXNet symbolic API**
(``mx.sym.*``); it has no ``nn.Module`` and no exportable weights.  This file
therefore *re-implements* the architecture natively in PyTorch, following the
1:1 architecture spec in ``documentation/STFGNN_repo_analysis.md``.

Architecture summary (3 stacked STSGCL blocks):
    input (B, T_in, N, C)
      -> First embedding FC (C -> first_layer_embedding_size) + ReLU
      -> 3 x STSGCL, each:
           * learnable temporal + spatial position embeddings
           * gated (GLU) dilated temporal convolution, kernel (1,2), dilation (1,3),
             producing T-(strides-1) steps  (residual branch)
           * for every sliding window of `strides`(=4) consecutive time steps:
               reshape -> (B, 4N, C) -> STSGCM(spatial-temporal fusion GCN) -> (B, N, C')
           * residual add:  STSGCM output + temporal-conv output
      -> output head (per horizon step) : FC (T*C -> 128) + ReLU -> FC (128 -> 1)
      -> concat over horizon -> (B, T_out, N, output_dim)

This mirrors ``libcity/model/traffic_flow_prediction/STSGCN.py`` (the direct
predecessor model already ported in LibCity) extended with the two STFGNN
innovations: (1) the gated temporal-convolution residual branch and (2) the
(4N, 4N) *fusion graph* that combines a spatial adjacency with a DTW-based
temporal adjacency.

NOTE: no dataset is in scope for this migration round; the fusion-graph builder
is therefore implemented purely model-side and degrades gracefully (identity /
self-loop temporal graph) when a DTW temporal graph is not supplied.
"""

import os
import numpy as np
from logging import getLogger

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


# ---------------------------------------------------------------------------
# Fusion-graph construction utilities
# ---------------------------------------------------------------------------
def construct_adj_fusion(a_spatial, a_dtw, steps=4):
    """Build the STFGNN (4N, 4N) fusion graph.

    Faithful reproduction of ``utils_4n0_3layer_12T_res.py::construct_adj_fusion``
    from the original repo.  ``steps`` must be 4 for the STFGNN paper model (this
    is ``strides`` = the number of time steps combined into one local
    spatio-temporal graph).

    Block pattern (T = temporal/DTW graph, S = spatial graph, I = identity band):

        [ T, I, I, T
          I, S, I, I
          I, I, S, I
          T, I, I, T ]

    Args:
        a_spatial: np.ndarray, shape (N, N), spatial (connectivity) adjacency.
        a_dtw:     np.ndarray, shape (N, N), temporal (DTW) adjacency.
        steps:     int, must be 4.

    Returns:
        np.ndarray, shape (4N, 4N).
    """
    assert steps == 4, "STFGNN fusion graph is defined for steps=4 (strides=4)."
    a_spatial = np.asarray(a_spatial, dtype=np.float32)
    a_dtw = np.asarray(a_dtw, dtype=np.float32)
    n = a_spatial.shape[0]
    adj = np.zeros([n * steps] * 2, dtype=np.float32)

    # (1) diagonal blocks: block1, block2 <- spatial ; block0, block3 <- DTW
    for i in range(steps):
        if i == 1 or i == 2:
            adj[i * n:(i + 1) * n, i * n:(i + 1) * n] = a_spatial
        else:
            adj[i * n:(i + 1) * n, i * n:(i + 1) * n] = a_dtw

    # (2) temporal band: connect the same node across adjacent time blocks
    for i in range(n):
        for k in range(steps - 1):
            adj[k * n + i, (k + 1) * n + i] = 1
            adj[(k + 1) * n + i, k * n + i] = 1

    # (3) corner blocks <- DTW
    adj[3 * n:4 * n, 0:n] = a_dtw
    adj[0:n, 3 * n:4 * n] = a_dtw

    # (4) off-diagonal identity-like blocks (blk(2,0), blk(0,2), blk(1,3), blk(3,1))
    adj[2 * n:3 * n, 0:n] = adj[0 * n:1 * n, 1 * n:2 * n]
    adj[0:n, 2 * n:3 * n] = adj[0 * n:1 * n, 1 * n:2 * n]
    adj[1 * n:2 * n, 3 * n:4 * n] = adj[0 * n:1 * n, 1 * n:2 * n]
    adj[3 * n:4 * n, 1 * n:2 * n] = adj[0 * n:1 * n, 1 * n:2 * n]

    # (5) self loops
    for i in range(len(adj)):
        adj[i, i] = 1

    return adj


def _load_adjacency(path):
    """Load an (N, N) adjacency matrix from a .npy / csv / txt file."""
    if path.endswith('.npy'):
        adj = np.load(path)
    else:
        adj = np.genfromtxt(path, delimiter=',')
    return np.asarray(adj, dtype=np.float32)


def _dtw_distance(a, b, order=1, lag=12):
    """Constrained (Sakoe-Chiba band) DTW distance between two 1-D series.

    Reproduces the original repo's ``compute_dtw``: local cost L^order, a band
    ``j in (i - lag, i + lag]`` and a final ``sqrt[order]`` of the path cost.
    """
    length = len(a)
    inf = float('inf')
    d = np.full((length + 1, length + 1), inf, dtype=np.float64)
    d[0, 0] = 0.0
    for i in range(1, length + 1):
        j_lo = max(1, i - lag)
        j_hi = min(length, i + lag)
        for j in range(j_lo, j_hi + 1):
            cost = abs(a[i - 1] - b[j - 1]) if order == 1 else abs(a[i - 1] - b[j - 1]) ** order
            d[i, j] = cost + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return d[length, length] ** (1.0 / order)


def build_dtw_temporal_adjacency(series, adj_percent=0.01, order=1, lag=12):
    """Optional helper: build the DTW temporal graph used by the fusion graph.

    Reproduces the documented pipeline (``data/fastDTW_adj_gen.py`` /
    ``data/stsgcn-dtw.py``): DTW(order=1, lag=12) on (z-scored) train signals,
    symmetrize, keep the top ``adj_percent`` (1%) nearest neighbours per row,
    symmetrize again and set self-loops.

    This function is *not* used by the model's default path (no dataset is in
    scope); it is provided so a caller can pre-compute ``adj_dtw`` and feed it
    to the model via the ``adj_dtw`` / ``adj_dtw_filename`` config keys.

    Args:
        series: np.ndarray, shape (L, N), per-node training signals (z-scored).
        adj_percent: float, fraction of neighbours kept per node (default 1%).
        order: int, Minkowski order of the local cost (paper uses 1).
        lag: int, Sakoe-Chiba band / time lag (paper uses 12).

    Returns:
        np.ndarray, shape (N, N), binary temporal adjacency with self-loops.
    """
    series = np.asarray(series, dtype=np.float64)
    n = series.shape[1]
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = _dtw_distance(series[:, i], series[:, j], order=order, lag=lag)
    dist = dist + dist.T  # symmetrize (as in the original script)
    k = max(1, int(n * adj_percent))
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        idx = np.argsort(dist[i])[:k]
        adj[i, idx] = 1
    adj = ((adj + adj.T) > 0).astype(np.float32)  # make undirected
    np.fill_diagonal(adj, 1)                       # self loops
    return adj


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------
class PositionEmbedding(nn.Module):
    """Learnable temporal (1, T, 1, C) and spatial (1, 1, N, C) embeddings."""

    def __init__(self, input_length, num_of_vertices, embedding_size,
                 temporal=True, spatial=True, device=torch.device('cpu')):
        super().__init__()
        self.temporal = temporal
        self.spatial = spatial
        if temporal:
            self.temporal_emb = nn.Parameter(
                torch.zeros(1, input_length, 1, embedding_size, device=device))
            xavier_uniform_(self.temporal_emb)
        if spatial:
            self.spatial_emb = nn.Parameter(
                torch.zeros(1, 1, num_of_vertices, embedding_size, device=device))
            xavier_uniform_(self.spatial_emb)

    def forward(self, data):
        # data: (B, T, N, C)
        if self.temporal:
            data = data + self.temporal_emb
        if self.spatial:
            data = data + self.spatial_emb
        return data


class GcnOperation(nn.Module):
    """Graph convolution: A @ X followed by GLU / ReLU.

    Mirrors the MXNet ``gcn_operation``: ``dot(adj, data)`` then a fully
    connected layer; GLU splits into two halves and returns ``lhs * sigmoid(rhs)``.
    """

    def __init__(self, num_of_filter, num_of_features, num_of_vertices, activation):
        super().__init__()
        assert activation in {'GLU', 'relu'}
        self.num_of_filter = num_of_filter
        self.num_of_features = num_of_features
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        if activation == 'GLU':
            self.layer = nn.Linear(num_of_features, 2 * num_of_filter)
        else:
            self.layer = nn.Linear(num_of_features, num_of_filter)

    def forward(self, data, adj):
        # data: (B, n, C) ; adj: (n, n) -> (B, n, C)
        # `adj` (2D) is broadcast over the batch dim, matching LibCity's STSGCN convention.
        data = torch.matmul(adj, data)
        if self.activation == 'GLU':
            data = self.layer(data)
            lhs, rhs = data.split(self.num_of_filter, -1)
            return lhs * torch.sigmoid(rhs)
        return torch.relu(self.layer(data))


class Stsgcm(nn.Module):
    """STSGCM: stack GCN ops, crop each to the middle N vertices and max-pool."""

    def __init__(self, filters, num_of_features, num_of_vertices, activation):
        super().__init__()
        self.filters = filters
        self.num_of_features = num_of_features
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        self.layers = nn.ModuleList()
        for i in range(len(filters)):
            self.layers.append(
                GcnOperation(filters[i], num_of_features, num_of_vertices, activation))
            num_of_features = filters[i]

    def forward(self, data, adj):
        # data: (B, 4N, C) -> returns (B, N, C')
        need_concat = []
        for layer in self.layers:
            data = layer(data, adj)
            # crop to the middle block [N:2N] of the 4N vertices
            need_concat.append(data[:, self.num_of_vertices:2 * self.num_of_vertices, :])
        # stack over the GCN sub-layers and take the element-wise max
        return torch.max(torch.stack(need_concat, dim=0), dim=0).values


class STSGCL(nn.Module):
    """One spatio-temporal synchronous graph convolution layer (STSGCL).

    Combines position embeddings, a gated dilated temporal convolution residual
    branch and a set of STSGCM blocks over sliding windows of `strides` steps.
    """

    def __init__(self, t, num_of_vertices, num_of_features, filters, strides,
                 activation='GLU', temporal_emb=True, spatial_emb=True,
                 device=torch.device('cpu')):
        super().__init__()
        self.T = t
        self.strides = strides
        self.history = t
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.in_dim = num_of_features
        self.filters = filters
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb

        self.position_embedding = PositionEmbedding(
            t, num_of_vertices, num_of_features, temporal_emb, spatial_emb, device)

        # gated dilated temporal convolution: kernel (1, 2), dilation (1, strides-1)
        # -> effective receptive field = strides, output length T-(strides-1)
        self.conv_left = nn.Conv2d(
            num_of_features, filters[-1], kernel_size=(1, 2), stride=(1, 1),
            dilation=(1, strides - 1))
        self.conv_right = nn.Conv2d(
            num_of_features, filters[-1], kernel_size=(1, 2), stride=(1, 1),
            dilation=(1, strides - 1))

        # one STSGCM per sliding window (module_type = 'individual')
        self.stsgcms = nn.ModuleList()
        for _ in range(t - strides + 1):
            self.stsgcms.append(
                Stsgcm(filters, num_of_features, num_of_vertices, activation))

    def forward(self, data, adj):
        # data: (B, T, N, C) -> (B, T-(strides-1), N, C')
        data = self.position_embedding(data)

        # residual branch: gated dilated temporal convolution over the time axis
        data_temp = data.permute(0, 3, 2, 1)                       # (B, C, N, T)
        data_left = torch.sigmoid(self.conv_left(data_temp))
        data_right = torch.tanh(self.conv_right(data_temp))
        data_res = (data_left * data_right).permute(0, 3, 2, 1)    # (B, T-k+1, N, C')

        batch_size = data.shape[0]
        need_concat = []
        for i in range(self.history - self.strides + 1):
            t = data[:, i:i + self.strides, :, :]                  # (B, strides, N, C)
            # merge time and node axes -> (B, strides*N, C), matching the (4N) fusion graph
            t = t.reshape(batch_size, self.strides * self.num_of_vertices, self.in_dim)
            t = self.stsgcms[i](t, adj)                            # (B, N, C')
            need_concat.append(t.unsqueeze(1))                     # (B, 1, N, C')

        mid_out = torch.cat(need_concat, dim=1)                    # (B, T-k+1, N, C')
        return mid_out + data_res


class OutputLayer(nn.Module):
    """Output head applied per prediction step: FC(T*C -> hidden) + ReLU -> FC(hidden -> 1)."""

    def __init__(self, num_of_vertices, input_length, num_of_features,
                 num_of_filters=128, predict_length=1, output_dim=1):
        super().__init__()
        self.num_of_vertices = num_of_vertices
        self.input_length = input_length
        self.num_of_features = num_of_features
        self.predict_length = predict_length
        self.output_dim = output_dim
        self.hidden_layer = nn.Linear(input_length * num_of_features, num_of_filters)
        self.output_layer = nn.Linear(num_of_filters, predict_length * output_dim)

    def forward(self, data):
        # data: (B, T, N, C) -> (B, predict_length, N, output_dim)
        data = data.transpose(1, 2)                                # (B, N, T, C)
        data = data.reshape(-1, self.num_of_vertices, self.input_length * self.num_of_features)
        data = torch.relu(self.hidden_layer(data))                 # (B, N, hidden)
        data = self.output_layer(data).reshape(
            -1, self.num_of_vertices, self.predict_length, self.output_dim)
        return data.permute(0, 2, 1, 3)                            # (B, T', N, out)


# ---------------------------------------------------------------------------
# Top-level STFGNN model
# ---------------------------------------------------------------------------
class STFGNN(AbstractTrafficStateModel):
    """STFGNN (AAAI 2021) adapted to the LibCity framework."""

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self._scaler = self.data_feature.get('scaler')
        self._logger = getLogger()

        self.device = config.get('device', torch.device('cpu'))
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.num_of_vertices = self.num_nodes

        # ---- hyperparameters (paper defaults) ----
        self.module_type = config.get('module_type', 'individual')
        self.activation = config.get('act_type', 'GLU')
        self.temporal_emb = config.get('temporal_emb', True)
        self.spatial_emb = config.get('spatial_emb', True)
        self.use_mask = config.get('use_mask', True)
        self.strides = config.get('strides', 4)
        self.rho = config.get('rho', 1.0)
        self.out_layer_dim = config.get('out_layer_dim', 128)
        first_layer_embedding_size = config.get('first_layer_embedding_size', 64)

        if self.module_type != 'individual':
            self._logger.warning(
                "STFGNN only implements module_type='individual'; "
                "got '%s', falling back to 'individual'.", self.module_type)
            self.module_type = 'individual'

        # ---- fusion graph (4N, 4N) ----
        adj_spatial = self._build_spatial_adj()
        adj_dtw = self._build_temporal_adj(config, adj_spatial)
        fusion = construct_adj_fusion(adj_spatial, adj_dtw, steps=self.strides)
        self.register_buffer('adj_base', torch.tensor(fusion, dtype=torch.float32))
        self._logger.info('STFGNN fusion graph shape: %s', str(tuple(fusion.shape)))

        # learnable adjacency mask initialized to the binary support of the fusion graph
        if self.use_mask:
            self.mask = nn.Parameter((self.adj_base != 0).float().clone())
        else:
            self.mask = None

        # ---- input embedding ----
        self.embedding_dim = self.feature_dim
        self.num_of_features = self.feature_dim
        if first_layer_embedding_size:
            self.first_layer_embedding = nn.Linear(self.num_of_features, first_layer_embedding_size)
            self.num_of_features = first_layer_embedding_size
        else:
            self.first_layer_embedding = None

        # ---- 3 x STSGCL ----
        self.filter_list = config.get(
            'filters', [[64, 64, 64], [64, 64, 64], [64, 64, 64]])
        self.stsgcl_layers = nn.ModuleList()
        input_length = self.input_window
        for filters in self.filter_list:
            self.stsgcl_layers.append(
                STSGCL(input_length, self.num_of_vertices, self.num_of_features,
                       filters, self.strides, activation=self.activation,
                       temporal_emb=self.temporal_emb, spatial_emb=self.spatial_emb,
                       device=self.device))
            input_length -= (self.strides - 1)
            self.num_of_features = filters[-1]
        self.final_input_length = input_length
        if self.final_input_length <= 0:
            raise ValueError(
                'input_window (%d) is too short for %d STSGCL layers with '
                'strides=%d.' % (self.input_window, len(self.filter_list), self.strides))

        # ---- output head: one head per prediction step ----
        self.outputs = nn.ModuleList()
        for _ in range(self.output_window):
            self.outputs.append(
                OutputLayer(self.num_of_vertices, self.final_input_length,
                            self.num_of_features, num_of_filters=self.out_layer_dim,
                            predict_length=1, output_dim=self.output_dim))

    # -- adjacency helpers --------------------------------------------------
    def _build_spatial_adj(self):
        """Use LibCity's provided spatial adjacency (binarized connectivity)."""
        n = self.num_of_vertices
        adj = self.data_feature.get('adj_mx', None)
        if adj is None:
            self._logger.warning(
                'adj_mx not found in data_feature; using identity (self-loop) adjacency.')
            return np.eye(n, dtype=np.float32)
        adj = np.asarray(adj, dtype=np.float32)
        if adj.ndim != 2 or adj.shape[0] != n or adj.shape[1] != n:
            self._logger.warning(
                'adj_mx shape %s incompatible with num_nodes=%d; using identity adjacency.',
                str(adj.shape), n)
            return np.eye(n, dtype=np.float32)
        adj = np.where(np.isfinite(adj), adj, 0.0)          # drop inf / nan (distance mats)
        return (adj != 0).astype(np.float32)                # binary connectivity (paper)

    def _build_temporal_adj(self, config, adj_spatial):
        """Obtain the DTW temporal graph, degrading gracefully when unavailable."""
        n = self.num_of_vertices
        adj = config.get('adj_dtw', None)
        if adj is not None:
            adj = np.asarray(adj, dtype=np.float32)
        else:
            path = config.get('adj_dtw_filename', config.get('adj_dtw_file', None))
            if path and os.path.exists(path):
                adj = _load_adjacency(path)
            else:
                default = str(config.get('adj_dtw_default', 'identity')).lower()
                if default == 'spatial':
                    self._logger.warning(
                        'DTW temporal graph unavailable; using the spatial adjacency '
                        'as the temporal graph.')
                    adj = np.asarray(adj_spatial, dtype=np.float32).copy()
                else:
                    self._logger.warning(
                        'DTW temporal graph unavailable; using identity (self-loop) '
                        'temporal graph.')
                    adj = np.eye(n, dtype=np.float32)
        if adj.ndim != 2 or adj.shape[0] != n or adj.shape[1] != n:
            self._logger.warning(
                'Temporal adjacency shape %s incompatible with num_nodes=%d; using identity.',
                str(adj.shape), n)
            adj = np.eye(n, dtype=np.float32)
        adj = np.where(np.isfinite(adj), adj, 0.0)
        return (adj != 0).astype(np.float32)

    # -- forward / loss -----------------------------------------------------
    def forward(self, batch):
        data = batch['X']  # (B, input_window, N, feature_dim)
        # only the first `embedding_dim` channels are used (paper: only channel 0 / flow)
        if data.shape[-1] > self.embedding_dim:
            data = data[:, :, :, 0:self.embedding_dim]

        if self.first_layer_embedding is not None:
            data = torch.relu(self.first_layer_embedding(data))

        adj = self.mask * self.adj_base if self.use_mask else self.adj_base

        for stsgcl_layer in self.stsgcl_layers:
            data = stsgcl_layer(data, adj)

        need_concat = [output_layer(data) for output_layer in self.outputs]
        data = torch.cat(need_concat, dim=1)  # (B, output_window, N, output_dim)
        return data

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        # Huber loss (rho configurable, paper rho=1)
        return loss.huber_loss(y_predicted, y_true, delta=self.rho)
