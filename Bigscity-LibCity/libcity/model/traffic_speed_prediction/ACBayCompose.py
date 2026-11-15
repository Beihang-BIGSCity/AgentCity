# -*- coding: utf-8 -*-
"""
ACBayCompose: dataset-conditioned composed model for PeMS-Bay short-term speed prediction.

This model is *composed* from atomic modules of the AgentCity module library
(``/home/wangwenrui/shk/AgentCity/data/modlib``) rather than being a port of a
published architecture.  Every selected module implementation is inlined below
verbatim (only documentation/whitespace adapted, plus one pure-performance cache
in ``ChebGCN``) so this file has no dependency on the module library path.

Composition (exactly one module per category), fixed by the measured profile of
PEMS_BAY (325 loop sensors, 5-min sampling, cv_global=0.1492, cv_node=0.1337,
spatial_heterogeneity=0.0415, lag-1 autocorrelation rho1=0.9735, cv_window_std
=0.0101, null_ratio=0.0):

    encoder  : ``feature_embedding_encoder`` / ``FeatureEmbeddingEncoder``
               -> provenance: LibCity ``traffic_flow_prediction/STSGCN.py``
                  (``STSGCN.first_layer_embedding``, Linear(C -> d) + ReLU)
    spatial  : ``cheb_gcn`` / ``ChebGCN`` (Ks=2, fixed adjacency from
               PEMS_BAY.rel + PEMS_BAY.geo, bidir_adj_mx=true)
               -> provenance: LibCity ``traffic_flow_prediction/ASTGCN.py``
                  (``ChebConvWithSAt``)
    temporal : ``lstm`` / ``LSTMEncoder`` (T' = T)
               -> provenance: LibCity ``traffic_speed_prediction/RNN.py``
                  (``rnn_type='LSTM'`` branch)
    fusion   : ``add`` / ``AddFusion`` (parameter-free when widths match)
               -> provenance: LibCity ``traffic_flow_prediction/ASTGCN.py``
                  (``FusionLayer``)
    head     : ``linear_head`` / ``LinearHead`` (direct multi-step, masked_mae)
               -> provenance: LibCity ``traffic_speed_prediction/STFGNN.py``
                  (``OutputLayer``)

Macro contract (mirrors ``traffic_speed_prediction/STFGNN.py``):
    X (B, input_window, N, C)
      -> encoder            (B, T, N, hidden_dim)
      -> parallel branches  spatial: (B, T, N, hidden_dim)   [needs adj (N, N)]
                            temporal: (B, T, N, hidden_dim)
      -> fusion             (B, T, N, hidden_dim)
      -> head               (B, output_window, N, output_dim)

Ablation switches (kept intact for the downstream frozen-architecture benchmark):
    ``spatial_module='none'``   -> the spatial branch is replaced by
                                  ``torch.zeros_like(encoded)``.
    ``temporal_module='none'``  -> the temporal branch is replaced by
                                  ``torch.zeros_like(encoded)``.
    In both cases the fusion/head parameter shapes are unchanged, because the
    branches all live at width ``hidden_dim`` and ``AddFusion`` only creates
    projections when the widths differ.

Loss: the framework path -- inverse-transform y_true *and* y_pred with the
dataset scaler (one global mean/std, train-split fitted), then
``masked_mae_torch(..., null_val=0)`` so the optimised quantity is in km/h.
"""

import numpy as np
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


# ---------------------------------------------------------------------------
# Atomic module: encoder / feature_embedding_encoder
# (source: module library code/encoder/feature_embedding_encoder.py,
#  provenance LibCity traffic_flow_prediction/STSGCN.py)
# ---------------------------------------------------------------------------
class FeatureEmbeddingEncoder(nn.Module):
    """Project ``C`` input channels to ``hidden_dim`` hidden channels."""

    def __init__(self, input_dim=3, hidden_dim=64, dropout=0.0, bias=True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.proj = nn.Linear(self.input_dim, self.hidden_dim, bias=bias)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if self.proj.in_features != x.shape[-1]:
            self.proj = nn.Linear(x.shape[-1], self.hidden_dim).to(device=x.device, dtype=x.dtype)
        return self.dropout(self.act(self.proj(x)))


# ---------------------------------------------------------------------------
# Atomic module: spatial / cheb_gcn
# (source: module library code/spatial/cheb_gcn.py,
#  provenance LibCity traffic_flow_prediction/ASTGCN.py :: ChebConvWithSAt)
# ---------------------------------------------------------------------------
class ChebGCN(nn.Module):
    """K-order Chebyshev graph convolution."""

    def __init__(self, hidden_dim=64, num_nodes=207, ks=2, dropout=0.0, bias=True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_nodes = int(num_nodes)
        self.ks = max(1, int(ks))
        self.dropout = float(dropout)
        self.in_proj = None
        self.weight = nn.Parameter(torch.empty(self.ks, self.hidden_dim, self.hidden_dim))
        self.bias = nn.Parameter(torch.zeros(self.hidden_dim)) if bias else None
        nn.init.xavier_uniform_(self.weight)
        # pure-performance cache: the Chebyshev supports depend only on the fixed
        # adjacency, so the (expensive) eigendecomposition is done once instead of
        # once per forward call.  Invalidated automatically when adj/device/dtype change.
        self._supports_key = None
        self._supports_value = None

    def _align(self, x):
        if x.shape[-1] == self.hidden_dim:
            return x
        if self.in_proj is None or self.in_proj.in_features != x.shape[-1]:
            self.in_proj = nn.Linear(x.shape[-1], self.hidden_dim).to(device=x.device, dtype=x.dtype)
        return self.in_proj(x)

    def _supports(self, adj, num_nodes, device, dtype):
        key = (
            num_nodes,
            str(device),
            str(dtype),
            None if adj is None else (tuple(adj.shape), adj.data_ptr()),
        )
        if self._supports_key == key and self._supports_value is not None:
            return self._supports_value

        eye = torch.eye(num_nodes, device=device, dtype=dtype)
        if adj is None:
            zero = torch.zeros_like(eye)
            supports = [eye] + [zero for _ in range(self.ks - 1)]
        else:
            adj = torch.nan_to_num(adj.to(device=device, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
            adj = adj.clamp(min=0.0)
            adj = 0.5 * (adj + adj.transpose(0, 1)) + eye
            degree = adj.sum(dim=-1)
            scale = torch.where(degree > 0, degree.clamp(min=1e-6).rsqrt(), torch.zeros_like(degree))
            norm = scale.unsqueeze(-1) * adj * scale.unsqueeze(0)
            lap = eye - norm
            lambda_max = torch.clamp(torch.linalg.eigvalsh(lap.detach()).max(), min=1e-5)
            scaled = 2.0 * lap / lambda_max - eye
            supports = [eye, scaled]
            for _ in range(2, self.ks):
                supports.append(2.0 * scaled @ supports[-1] - supports[-2])
            supports = supports[: self.ks]
        self._supports_key = key
        self._supports_value = supports
        return supports

    def forward(self, x, adj=None):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self._align(x)
        supports = self._supports(adj, x.shape[2], x.device, x.dtype)
        terms = torch.stack([torch.einsum("nm,btmc->btnc", s, x) for s in supports], dim=0)
        out = torch.einsum("kbtnc,kco->btno", terms, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return F.dropout(out, p=self.dropout, training=self.training)


# ---------------------------------------------------------------------------
# Atomic module: temporal / lstm
# (source: module library code/temporal/lstm.py,
#  provenance LibCity traffic_speed_prediction/RNN.py :: rnn_type='LSTM')
# ---------------------------------------------------------------------------
class LSTMEncoder(nn.Module):
    """Long short-term memory encoder with ``T' == T``."""

    def __init__(self, hidden_dim=64, num_layers=1, dropout=0.0, bidirectional=False, input_dim=None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = max(1, int(num_layers))
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1
        in_dim = self.hidden_dim if input_dim is None else int(input_dim)
        self.in_proj = nn.Identity() if in_dim == self.hidden_dim else nn.Linear(in_dim, self.hidden_dim)
        self.rnn = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        self.out_proj = (
            nn.Identity()
            if self.num_directions == 1
            else nn.Linear(self.num_directions * self.hidden_dim, self.hidden_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.in_proj(x)
        batch, steps, nodes, channels = x.shape
        sequence = x.permute(1, 0, 2, 3).reshape(steps, batch * nodes, channels)
        out, _ = self.rnn(sequence)
        out = out.reshape(steps, batch, nodes, out.shape[-1]).permute(1, 0, 2, 3)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Atomic module: fusion / add
# (source: module library code/fusion/add.py,
#  provenance LibCity traffic_flow_prediction/ASTGCN.py :: FusionLayer)
# ---------------------------------------------------------------------------
class AddFusion(nn.Module):
    """Sum two ``(B, T, N, d)`` representations."""

    def __init__(self, hidden_dim=64, dropout=0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.proj_sp = None
        self.proj_tm = None

    def _project(self, h, spatial):
        if h.shape[-1] == self.hidden_dim:
            return h
        proj = self.proj_sp if spatial else self.proj_tm
        if proj is None or proj.in_features != h.shape[-1]:
            proj = nn.Linear(h.shape[-1], self.hidden_dim).to(device=h.device, dtype=h.dtype)
            if spatial:
                self.proj_sp = proj
            else:
                self.proj_tm = proj
        return proj(h)

    def _align(self, h_sp, h_tm):
        if h_sp.shape[-1] == h_tm.shape[-1] == self.hidden_dim:
            return h_sp, h_tm
        return self._project(h_sp, True), self._project(h_tm, False)

    def forward(self, h_sp, h_tm):
        h_sp, h_tm = self._align(h_sp, h_tm)
        return self.dropout(h_sp + h_tm)


# ---------------------------------------------------------------------------
# Atomic module: head / linear_head
# (source: module library code/head/linear_head.py,
#  provenance LibCity traffic_speed_prediction/STFGNN.py :: OutputLayer)
# ---------------------------------------------------------------------------
class LinearHead(nn.Module):
    """Flatten-then-linear head producing ``(B, H, N, output_dim)``."""

    DEFAULT_LOSS = "masked_mae"

    def __init__(self, hidden_dim=64, input_len=12, horizon=12, output_dim=1, dropout=0.0, bias=True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input_len = int(input_len)
        self.horizon = int(horizon)
        self.output_dim = int(output_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.fc = nn.Linear(self.input_len * self.hidden_dim, self.horizon * self.output_dim, bias=bias)

    def _ensure_layer(self, in_features, device, dtype):
        if self.fc.in_features != in_features:
            self.fc = nn.Linear(in_features, self.horizon * self.output_dim).to(device=device, dtype=dtype)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        batch, steps, nodes, channels = x.shape
        self._ensure_layer(steps * channels, x.device, x.dtype)
        flat = x.permute(0, 2, 1, 3).reshape(batch, nodes, steps * channels)
        out = self.fc(self.dropout(flat))
        return out.view(batch, nodes, self.horizon, self.output_dim).permute(0, 2, 1, 3)

    def loss(self, pred, target, null_val=0.0, loss_type=None):
        loss_type = (loss_type or self.DEFAULT_LOSS).lower()
        target = target.to(pred.dtype)
        mask = (target != null_val).to(pred.dtype)
        diff = pred - target
        if loss_type in ("masked_mae", "mae"):
            error = diff.abs()
        elif loss_type in ("masked_mse", "mse"):
            error = diff * diff
        elif loss_type in ("masked_mape", "mape"):
            error = diff.abs() / target.abs().clamp_min(1e-5)
        else:
            raise ValueError("unsupported loss_type: {}".format(loss_type))
        masked = error * mask
        return masked.sum() / mask.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# Composed model
# ---------------------------------------------------------------------------
class ACBayCompose(AbstractTrafficStateModel):
    """PeMS-Bay composed model: FeatureEmbedding -> [ChebGCN || LSTM] -> Add -> LinearHead."""

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        # NOTE: libcity's TrafficStateDataset sets feature_dim = x_train.shape[-1]
        # (i.e. traffic channels + external channels) and ext_dim = feature_dim - output_dim.
        self.feature_dim = int(self.data_feature.get('feature_dim', 1))
        self.ext_dim = int(self.data_feature.get('ext_dim', 0))
        self.output_dim = int(self.data_feature.get('output_dim', 1))
        self._scaler = self.data_feature.get('scaler')

        self.device = config.get('device', torch.device('cpu'))
        self.input_window = int(config.get('input_window', 12))
        self.output_window = int(config.get('output_window', 12))
        self.hidden_dim = int(config.get('hidden_dim', 32))
        self.depth = max(1, int(config.get('depth', 1)))
        self.null_value = float(config.get('null_value', 0))
        self.loss_type = str(config.get('loss_type', LinearHead.DEFAULT_LOSS)).lower()

        # ablation switches ('none' -> zero branch, parameter shapes unchanged)
        self.spatial_module = str(config.get('spatial_module', 'cheb_gcn')).lower()
        self.temporal_module = str(config.get('temporal_module', 'lstm')).lower()

        # module hyperparameters
        self.cheb_ks = max(1, int(config.get('cheb_ks', config.get('ks', 2))))
        self.rnn_layers = max(1, int(config.get('rnn_layers', config.get('num_rnn_layers', 1))))
        self.dropout = float(config.get('dropout', 0.0))

        # ---- fixed graph support -------------------------------------------
        self.register_buffer('adj_mx', self._build_adjacency())

        # ---- encoder: single traffic_speed channel -> hidden_dim -----------
        self.encoder = FeatureEmbeddingEncoder(
            input_dim=self.feature_dim, hidden_dim=self.hidden_dim, dropout=self.dropout, bias=True)

        # ---- depth x (spatial || temporal -> fusion) blocks ----------------
        self.spatial_layers = nn.ModuleList()
        self.temporal_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        for _ in range(self.depth):
            self.spatial_layers.append(ChebGCN(
                hidden_dim=self.hidden_dim, num_nodes=self.num_nodes, ks=self.cheb_ks,
                dropout=self.dropout, bias=True))
            self.temporal_layers.append(LSTMEncoder(
                hidden_dim=self.hidden_dim, num_layers=self.rnn_layers,
                dropout=self.dropout, bidirectional=False, input_dim=self.hidden_dim))
            self.fusion_layers.append(AddFusion(hidden_dim=self.hidden_dim, dropout=self.dropout))

        # ---- direct multi-step regression head -----------------------------
        self.head = LinearHead(
            hidden_dim=self.hidden_dim, input_len=self.input_window,
            horizon=self.output_window, output_dim=self.output_dim,
            dropout=self.dropout, bias=True)

        self._logger.info(
            'ACBayCompose: encoder=feature_embedding_encoder(input_dim=%d -> %d), '
            'spatial=%s(ks=%d), temporal=%s(layers=%d), fusion=add, head=linear_head, '
            'N=%d, depth=%d, adj nnz=%d',
            self.feature_dim, self.hidden_dim, self.spatial_module, self.cheb_ks,
            self.temporal_module, self.rnn_layers, self.num_nodes, self.depth,
            int(torch.count_nonzero(self.adj_mx).item()))

    # -- adjacency ---------------------------------------------------------
    def _build_adjacency(self):
        """Take LibCity's runtime (N, N) adjacency, sanitising non-finite entries."""
        n = self.num_nodes
        adj = self.data_feature.get('adj_mx', None)
        if adj is None:
            self._logger.warning('adj_mx missing in data_feature; ACBayCompose uses an identity graph.')
            return torch.eye(n, dtype=torch.float32)
        adj = np.asarray(adj, dtype=np.float32)
        if adj.ndim != 2 or adj.shape[0] != n or adj.shape[1] != n:
            self._logger.warning(
                'adj_mx shape %s incompatible with num_nodes=%d; using identity graph.', str(adj.shape), n)
            return torch.eye(n, dtype=torch.float32)
        # distance-derived weights may contain inf for unreachable pairs (init_weight_inf_or_zero='inf')
        adj = np.where(np.isfinite(adj), adj, 0.0)
        adj = np.clip(adj, a_min=0.0, a_max=None)
        return torch.from_numpy(np.ascontiguousarray(adj))

    # -- named module attributes -------------------------------------------
    # Contract: expose ``encoder``, ``spatial``, ``temporal``, ``fusion`` and
    # ``head`` as named attributes.  ``encoder``/``head`` are plain attributes;
    # the per-branch modules live in ModuleLists so the same source supports
    # depth > 1, and these read-only views expose the active (first) block.
    # They add no parameters and do not change the forward path.
    @property
    def spatial(self):
        return self.spatial_layers[0]

    @property
    def temporal(self):
        return self.temporal_layers[0]

    @property
    def fusion(self):
        return self.fusion_layers[0]

    # -- forward / predict / loss -----------------------------------------
    def forward(self, batch):
        data = batch['X']                       # (B, input_window, N, feature_dim)
        if data.dim() == 3:
            data = data.unsqueeze(-1)
        # the dataset yields float64 tensors; the module parameters are float32
        if data.dtype != self.encoder.proj.weight.dtype:
            data = data.to(self.encoder.proj.weight.dtype)

        adj = self.adj_mx.to(device=data.device, dtype=data.dtype)

        hidden = self.encoder(data)             # (B, T, N, hidden_dim)
        for i in range(self.depth):
            if self.spatial_module == 'none':
                h_sp = torch.zeros_like(hidden)
            else:
                h_sp = self.spatial_layers[i](hidden, adj)
            if self.temporal_module == 'none':
                h_tm = torch.zeros_like(hidden)
            else:
                h_tm = self.temporal_layers[i](hidden)
            hidden = self.fusion_layers[i](h_sp, h_tm)

        return self.head(hidden)                # (B, output_window, N, output_dim)

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        # inverse-transform BOTH sides first: the masked loss lives in km/h
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        y_true = y_true.to(y_predicted.dtype)
        return loss.masked_mae_torch(y_predicted, y_true, null_val=self.null_value)
