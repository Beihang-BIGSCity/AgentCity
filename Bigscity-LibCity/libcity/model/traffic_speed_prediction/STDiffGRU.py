# -*- coding: utf-8 -*-
"""
STDiffGRU: a composed spatio-temporal model for METR_LA traffic-speed prediction.

This file is *generated* by the model-composition stage of the AgentCity
adaptation pipeline.  It is deliberately self-contained: the five atomic
modules selected from the module library are inlined below (source copied from
``data/modlib/code/...``) so the file has **no dependency on the module-library
path**.  The unified ``forward`` interfaces of the source modules are preserved.

Scenario (METR_LA, 207 freeway loop detectors, 5-min sampling):
    x : (B, input_window=12, N=207, C=2)   # traffic_speed + time_in_day
    y : (B, output_window=12, N=207, 1)

Architecture (encoder -> spatial + temporal -> fusion -> head):
    X (B,12,N,2)
      -> [encoder]  FeatureEmbeddingEncoder   : Linear(C -> hidden) + ReLU
      -> H (B,12,N,h)
      -> [spatial]  DiffusionGCN              : 1 shallow bidirectional random-walk
                                                 diffusion layer over the 207x207
                                                 Gauss-kernel adjacency -> H_sp
      -> [temporal] GRUEncoder                : 1-layer GRU, hidden_size=32 (config: hidden_dim), no dropout
                                                 -> H_tm
      -> [fusion]   ConcatFusion              : concat(H_sp, H_tm) -> Linear(2h -> h)
      -> [head]     LinearHead                : flatten(T*h) -> Linear -> 12 values/node
      -> Y_hat (B,12,N,1)   # LibCity layout [batch, output_window, num_nodes, output_dim]

Module provenance (atomic module library -> inlined class here):
    encoder  : code/encoder/feature_embedding_encoder.py  (FeatureEmbeddingEncoder)
               <- LibCity traffic_flow_prediction/STSGCN.py (STSGCN.first_layer_embedding)
    spatial  : code/spatial/diffusion_gcn.py              (DiffusionGCN)
               <- LibCity traffic_speed_prediction/DCRNN.py (GCONV, filter_type='dual_random_walk')
    temporal : code/temporal/gru.py                       (GRUEncoder)
               <- LibCity traffic_speed_prediction/DCRNN.py (DCGRUCell) / RNN.py
    fusion   : code/fusion/concat.py                      (ConcatFusion)
               <- LibCity traffic_speed_prediction/GWNET.py (end_conv channel fusion)
    head     : code/head/linear_head.py                   (LinearHead)
               <- LibCity traffic_speed_prediction/STFGNN.py (OutputLayer)

The model contract mirrors the already-working
``libcity/model/traffic_speed_prediction/STFGNN.py``: ``__init__(self, config,
data_feature)``, ``get_data`` (inherited), ``predict``, ``calculate_loss`` and
``forward``.
"""

from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


# ===========================================================================
# Inlined atomic modules (module-library source, unified interfaces preserved)
# ===========================================================================

class FeatureEmbeddingEncoder(nn.Module):
    """[encoder] Project ``C`` input channels to ``hidden_dim`` hidden channels.

    Source: module library ``code/encoder/feature_embedding_encoder.py``.
    Interface: ``forward(x: (B, T, N, C)) -> (B, T, N, d)`` with ``d = hidden_dim``.
    """

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


class DiffusionGCN(nn.Module):
    """[spatial] Bidirectional random-walk diffusion convolution (DCRNN style).

    Source: module library ``code/spatial/diffusion_gcn.py``.
    Interface: ``forward(x: (B, T, N, d), adj: (N, N) | None) -> (B, T, N, d)``.
    """

    def __init__(self, hidden_dim=64, num_nodes=207, num_diffusion_steps=2, dropout=0.0, bias=True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_nodes = int(num_nodes)
        self.num_diffusion_steps = max(1, int(num_diffusion_steps))
        self.num_directions = 2
        self.dropout = float(dropout)
        self.in_proj = None
        self.blocks = 1 + self.num_directions * self.num_diffusion_steps
        self.proj = nn.Linear(self.blocks * self.hidden_dim, self.hidden_dim, bias=bias)

    def _align(self, x):
        if x.shape[-1] == self.hidden_dim:
            return x
        if self.in_proj is None or self.in_proj.in_features != x.shape[-1]:
            self.in_proj = nn.Linear(x.shape[-1], self.hidden_dim).to(device=x.device, dtype=x.dtype)
        return self.in_proj(x)

    def _supports(self, adj, num_nodes, device, dtype):
        eye = torch.eye(num_nodes, device=device, dtype=dtype)
        if adj is None:
            return [eye, eye]
        adj = torch.nan_to_num(adj.to(device=device, dtype=dtype), nan=0.0).clamp(min=0.0)

        def random_walk(matrix):
            degree = matrix.sum(dim=-1, keepdim=True)
            inverse = torch.where(degree > 0, 1.0 / degree.clamp(min=1e-6), torch.zeros_like(degree))
            return inverse * matrix

        return [random_walk(adj), random_walk(adj.transpose(0, 1))]

    def forward(self, x, adj=None):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self._align(x)
        supports = self._supports(adj, x.shape[2], x.device, x.dtype)
        blocks = [x]
        for support in supports:
            propagated = x
            for _ in range(self.num_diffusion_steps):
                propagated = torch.einsum("nm,btmc->btnc", support, propagated)
                blocks.append(propagated)
        out = self.proj(torch.cat(blocks, dim=-1))
        return F.dropout(out, p=self.dropout, training=self.training)


class GRUEncoder(nn.Module):
    """[temporal] Gated recurrent encoder with ``T' == T``.

    Source: module library ``code/temporal/gru.py``.
    Interface: ``forward(x: (B, T, N, d)) -> (B, T', N, d)`` with ``T' == T``.
    """

    def __init__(self, hidden_dim=64, num_layers=1, dropout=0.0, bidirectional=False, input_dim=None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = max(1, int(num_layers))
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1
        in_dim = self.hidden_dim if input_dim is None else int(input_dim)
        self.in_proj = nn.Identity() if in_dim == self.hidden_dim else nn.Linear(in_dim, self.hidden_dim)
        self.rnn = nn.GRU(
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


class ConcatFusion(nn.Module):
    """[fusion] Concatenate two representations and project them back.

    Source: module library ``code/fusion/concat.py``.
    Interface: ``forward(h_sp: (B, T, N, d), h_tm: (B, T, N, d)) -> (B, T, N, d)``.
    """

    def __init__(self, hidden_dim=64, dropout=0.0, bias=True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.proj = nn.Linear(2 * self.hidden_dim, self.hidden_dim, bias=bias)
        self.proj_sp = None
        self.proj_tm = None

    def _project(self, h, spatial):
        if h.shape[-1] == self.hidden_dim:
            return h
        branch = self.proj_sp if spatial else self.proj_tm
        if branch is None or branch.in_features != h.shape[-1]:
            branch = nn.Linear(h.shape[-1], self.hidden_dim).to(device=h.device, dtype=h.dtype)
            if spatial:
                self.proj_sp = branch
            else:
                self.proj_tm = branch
        return branch(h)

    def _align(self, h_sp, h_tm):
        if h_sp.shape[-1] == h_tm.shape[-1] == self.hidden_dim:
            return h_sp, h_tm
        return self._project(h_sp, True), self._project(h_tm, False)

    def forward(self, h_sp, h_tm):
        h_sp, h_tm = self._align(h_sp, h_tm)
        return self.dropout(self.proj(torch.cat([h_sp, h_tm], dim=-1)))


class LinearHead(nn.Module):
    """[head] Flatten-then-linear head producing ``(B, H, N, output_dim)``.

    Source: module library ``code/head/linear_head.py``.
    Interface: ``forward(x: (B, T', N, d)) -> (B, H, N, output_dim)``.
    """

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


# ===========================================================================
# Top-level composed model
# ===========================================================================

class STDiffGRU(AbstractTrafficStateModel):
    """Composed encoder + shallow diffusion-GCN + GRU + concat-fusion + linear head."""

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

        # ---- hyperparameters (kept RNN-scale for the CPU smoke budget) ----
        self.hidden_dim = config.get('hidden_dim', config.get('hidden_size', 64))
        self.num_layers = config.get('num_layers', 1)
        self.dropout = config.get('dropout', 0.0)
        self.num_diffusion_steps = config.get('num_diffusion_steps', 2)
        self.loss_type = config.get('loss_type', 'masked_mae')
        self.null_value = config.get('null_value', 0)

        # ---- optional branch ablation (config-driven; defaults keep the full composition) ----
        # A branch is disabled by setting its module name to "none" in the run config
        # (see libcity/config/model/traffic_state_pred/STDiffGRU_abl_*.json). The
        # disabled branch contributes an all-zero representation of the same shape,
        # so the fusion projection and the head keep identical shapes/parameter
        # counts; the only information available then comes from the retained branch.
        self.spatial_module = str(config.get('spatial_module', 'diffusion_gcn')).lower()
        self.temporal_module = str(config.get('temporal_module', 'gru')).lower()
        self.use_spatial = self.spatial_module not in ('none', 'off', 'disabled')
        self.use_temporal = self.temporal_module not in ('none', 'off', 'disabled')
        if not self.use_spatial and not self.use_temporal:
            self._logger.warning('both spatial and temporal branches are disabled; the model is a plain encoder+head')

        # ---- encoder: raw feature channels -> hidden width ----
        self.encoder = FeatureEmbeddingEncoder(
            input_dim=self.feature_dim, hidden_dim=self.hidden_dim, dropout=self.dropout)

        # ---- graph adjacency from the dataset (207x207 Gauss kernel) ----
        adj = self.data_feature.get('adj_mx', None)
        if adj is not None:
            adj = np.asarray(adj, dtype=np.float32)
            if adj.ndim == 2 and adj.shape[0] == self.num_nodes and adj.shape[1] == self.num_nodes:
                adj = np.where(np.isfinite(adj), adj, 0.0)
                self.register_buffer('adj', torch.tensor(adj, dtype=torch.float32))
            else:
                self._logger.warning(
                    'adj_mx shape %s incompatible with num_nodes=%d; spatial layer degrades '
                    'to identity supports.', str(getattr(adj, 'shape', None)), self.num_nodes)
                self.adj = None
        else:
            self._logger.warning('adj_mx not found in data_feature; spatial layer degrades.')
            self.adj = None

        # ---- spatial / temporal branches over the shared hidden representation ----
        self.spatial = (
            DiffusionGCN(
                hidden_dim=self.hidden_dim, num_nodes=self.num_nodes,
                num_diffusion_steps=self.num_diffusion_steps, dropout=self.dropout)
            if self.use_spatial
            else None
        )
        self.temporal = (
            GRUEncoder(hidden_dim=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout)
            if self.use_temporal
            else None
        )

        # ---- cheap explicit fusion ----
        self.fusion = ConcatFusion(hidden_dim=self.hidden_dim, dropout=self.dropout)

        # ---- single shared linear head ----
        self.head = LinearHead(
            hidden_dim=self.hidden_dim, input_len=self.input_window,
            horizon=self.output_window, output_dim=self.output_dim, dropout=self.dropout)

        self._logger.info(
            'STDiffGRU composed: hidden_dim=%d, num_layers=%d, diffusion_steps=%d, '
            'in=%d, out=%d, loss=%s, spatial_module=%s, temporal_module=%s',
            self.hidden_dim, self.num_layers, self.num_diffusion_steps,
            self.input_window, self.output_window, self.loss_type,
            self.spatial_module, self.temporal_module)

    def forward(self, batch):
        x = batch['X']                       # (B, input_window, N, feature_dim)
        h = self.encoder(x)                  # (B, T, N, d)
        h_sp = self.spatial(h, self.adj) if self.use_spatial else torch.zeros_like(h)
        h_tm = self.temporal(h) if self.use_temporal else torch.zeros_like(h)
        h_fused = self.fusion(h_sp, h_tm)    # (B, T, N, d)
        y_hat = self.head(h_fused)           # (B, output_window, N, output_dim)
        return y_hat

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, self.null_value)
