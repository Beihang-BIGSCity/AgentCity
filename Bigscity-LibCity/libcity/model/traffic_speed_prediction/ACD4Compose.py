# -*- coding: utf-8 -*-
"""
ACD4Compose: dataset-conditioned composed model for PEMSD4 short-term traffic-flow prediction.

This model is *composed* from atomic modules of the AgentCity module library
(``/home/wangwenrui/shk/AgentCity/data/modlib``) instead of being a port of a
single published architecture.  Every selected module implementation is inlined
below verbatim (only the module docstring is shortened; the code is unchanged)
so this file has no dependency on the module library path.

Composition (exactly one module per category), fixed by the MEASURED training
profile of PEMSD4 (307 loop sensors, 5-min sampling, 16992 steps, 10195 training
timesteps, cv_global=0.735, cv_node=0.570, spatial_heterogeneity=0.429,
lag-1 autocorrelation rho1=0.945, cv_window_std=0.028, null_ratio=1.67%):

    encoder  : ``feature_embedding_encoder`` / ``FeatureEmbeddingEncoder``
               (Linear(C=1 -> hidden_dim=32) + ReLU; no internal z-score, because
               LibCity already applies one global StandardScaler fitted on the
               train split)
               -> module library: code/encoder/feature_embedding_encoder.py
               -> provenance: LibCity ``traffic_flow_prediction/STSGCN.py``
                  (``STSGCN.first_layer_embedding``)
    spatial  : ``adaptive_gcn`` / ``AdaptiveGCN`` (ks=2, embed_dim=10, N=307)
               justified by spatial_heterogeneity=0.429 over 307 heterogeneous
               sensors combined with an extremely sparse shipped topology
               (raw_data/PEMSD4/PEMSD4.rel: 340 directed edges over 307 nodes,
               average degree ~1.1), so a learned node-embedding adjacency mixes
               information beyond the single neighbour a fixed graph offers
               (``spatial_use_prior_adj=true`` switches to the .rel graph as the
               Neumann base for ablation).
               -> module library: code/spatial/adaptive_gcn.py
               -> provenance: LibCity ``traffic_flow_prediction/AGCRN.py``
                  (``AVWGCN``: softmax(relu(E E^T)) + K-order polynomial)
    temporal : ``gated_dilated_tcn`` / ``GatedDilatedTCN`` (GLU, num_layers=3,
               kernel_size=2, dilations 2/4/8 -> receptive field 1+(2+4+8)=15 >= 12)
               justified by rho1=0.945 with input_window=12: the whole 12-step
               persistence structure is inside one bounded causal dilated
               receptive field, while temporal attention over 12 steps would only
               add O(T^2) cost.
               -> module library: code/temporal/gated_dilated_tcn.py
               -> provenance: LibCity ``traffic_flow_prediction/STSGCN.py``
                  (``STSGCL``: temporal convolution + GLU, a * sigmoid(b))
    fusion   : ``concat`` / ``ConcatFusion`` (Linear(2*32 -> 32))
               the two branches are complementary (graph mixing vs. temporal
               filtering), so the combination is learned rather than averaged;
               it also degrades gracefully under ablation because a zero branch
               is still a valid projection input.
               -> module library: code/fusion/concat.py
               -> provenance: LibCity ``traffic_speed_prediction/GWNET.py``
                  (channel stacking + linear fusion; STSGCN ``OutputLayer``)
    head     : ``linear_head`` / ``LinearHead`` (flatten 12*32 -> 12*1)
               the training regime is stationary (cv_window_std=0.028), so one
               shared linear map over the flattened history is enough, and its
               DEFAULT_LOSS is already ``masked_mae``.
               -> module library: code/head/linear_head.py
               -> provenance: LibCity ``traffic_speed_prediction/STFGNN.py``
                  (``OutputLayer``)

Macro contract (mirrors ``traffic_speed_prediction/STFGNN.py``):
    X (B, input_window, N, C)
      -> encoder            (B, T, N, hidden_dim)
      -> parallel branches  spatial : (B, T, N, hidden_dim)   [optional adj (N, N)]
                            temporal: (B, T, N, hidden_dim)
      -> fusion             (B, T, N, hidden_dim)
      -> head               (B, output_window, N, output_dim)

Ablation switches (kept intact for the downstream frozen-architecture benchmark):
    ``spatial_module='none'``   -> the spatial branch is replaced by
                                   ``torch.zeros_like(encoded_hidden)``.
    ``temporal_module='none'``  -> the temporal branch is replaced by
                                   ``torch.zeros_like(encoded_hidden)``.
    Both zeros tensors are fed into the SAME ``fusion`` / ``head`` path, so the
    parameter shapes never change and no branch is short-circuited.

Loss: the framework path -- inverse-transform y_true *and* y_pred with the
dataset scaler (one global mean/std, train-split fitted), then
``loss.masked_mae_torch(..., null_val=0.0)``, i.e. the optimised quantity is in
vehicles per 5 min, matching the reported metric.
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
# (source: module library code/encoder/feature_embedding_encoder.py;
#  provenance LibCity traffic_flow_prediction/STSGCN.py :: first_layer_embedding)
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
# Atomic module: spatial / adaptive_gcn
# (source: module library code/spatial/adaptive_gcn.py;
#  provenance LibCity traffic_flow_prediction/AGCRN.py :: AVWGCN)
# ---------------------------------------------------------------------------
class AdaptiveGCN(nn.Module):
    """Graph convolution over a node-embedding-derived adaptive adjacency."""

    def __init__(self, hidden_dim=64, num_nodes=207, embed_dim=10, ks=2, dropout=0.0, bias=True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_nodes = int(num_nodes)
        self.embed_dim = int(embed_dim)
        self.ks = max(1, int(ks))
        self.dropout = float(dropout)
        self.in_proj = None
        self.node_embeddings = nn.Parameter(0.1 * torch.randn(self.num_nodes, self.embed_dim))
        self.weight = nn.Parameter(torch.empty(self.ks, self.hidden_dim, self.hidden_dim))
        self.bias = nn.Parameter(torch.zeros(self.hidden_dim)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def _align(self, x):
        if x.shape[-1] == self.hidden_dim:
            return x
        if self.in_proj is None or self.in_proj.in_features != x.shape[-1]:
            self.in_proj = nn.Linear(x.shape[-1], self.hidden_dim).to(device=x.device, dtype=x.dtype)
        return self.in_proj(x)

    def _supports(self, adj, num_nodes, device, dtype):
        eye = torch.eye(num_nodes, device=device, dtype=dtype)
        if adj is not None:
            adj = torch.nan_to_num(adj.to(device=device, dtype=dtype), nan=0.0).clamp(min=0.0)
            degree = adj.sum(dim=-1, keepdim=True)
            base = torch.where(degree > 0, adj / degree.clamp(min=1e-6), torch.zeros_like(adj))
        elif num_nodes == self.num_nodes:
            similarity = self.node_embeddings @ self.node_embeddings.transpose(0, 1)
            base = F.softmax(F.relu(similarity), dim=1).to(device=device, dtype=dtype)
        else:
            zero = torch.zeros_like(eye)
            return [eye] + [zero for _ in range(self.ks - 1)]
        supports = [eye, base]
        for _ in range(2, self.ks):
            supports.append(base @ supports[-1])
        return supports[: self.ks]

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
# Atomic module: temporal / gated_dilated_tcn
# (source: module library code/temporal/gated_dilated_tcn.py;
#  provenance LibCity traffic_flow_prediction/STSGCN.py :: STSGCL temporal branch)
# ---------------------------------------------------------------------------
class GatedDilatedTCN(nn.Module):
    """GLU-gated dilated causal temporal convolution stack."""

    def __init__(self, hidden_dim=64, num_layers=2, kernel_size=2, dropout=0.0, input_dim=None, base_dilation=1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = max(1, int(num_layers))
        self.kernel_size = max(2, int(kernel_size))
        self.dropout = float(dropout)
        self.dilations = [max(1, int(base_dilation)) * (2 ** i) for i in range(self.num_layers)]
        in_dim = self.hidden_dim if input_dim is None else int(input_dim)
        self.in_proj = nn.Identity() if in_dim == self.hidden_dim else nn.Linear(in_dim, self.hidden_dim)
        self.convs = nn.ModuleList(
            nn.Conv2d(self.hidden_dim, 2 * self.hidden_dim, (1, self.kernel_size), dilation=(1, d))
            for d in self.dilations
        )
        self.skip_convs = nn.ModuleList(nn.Conv2d(self.hidden_dim, self.hidden_dim, (1, 1)) for _ in self.dilations)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        hidden = self.in_proj(x).permute(0, 3, 2, 1)
        for index, dilation in enumerate(self.dilations):
            residual = hidden
            pad = dilation * (self.kernel_size - 1)
            if pad > 0:
                hidden = F.pad(hidden, (pad, 0))
            content, gate = torch.chunk(self.convs[index](hidden), 2, dim=1)
            hidden = self.skip_convs[index](content * torch.sigmoid(gate))
            hidden = self.dropout_layer(hidden) + residual[:, :, :, -hidden.shape[-1]:]
        return hidden.permute(0, 3, 2, 1)


# ---------------------------------------------------------------------------
# Atomic module: fusion / concat
# (source: module library code/fusion/concat.py;
#  provenance LibCity traffic_speed_prediction/GWNET.py :: end_conv fusion)
# ---------------------------------------------------------------------------
class ConcatFusion(nn.Module):
    """Concatenate two representations and project them back."""

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


# ---------------------------------------------------------------------------
# Atomic module: head / linear_head
# (source: module library code/head/linear_head.py;
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
class ACD4Compose(AbstractTrafficStateModel):
    """PEMSD4 composed model: FeatureEmbedding -> [AdaptiveGCN || GatedDilatedTCN] -> Concat -> LinearHead."""

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self.num_nodes = int(self.data_feature.get('num_nodes', 1))
        # libcity's TrafficStateDataset sets feature_dim = traffic channels + ext channels
        # (add_time_in_day=true appends one time-of-day channel) and ext_dim = feature_dim - output_dim.
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
        # only the modelled traffic channels enter the model; the external
        # time-of-day channel supplied by LibCity is not used by this composition.
        self.input_channels = max(1, int(config.get('input_channels', self.output_dim)))
        self.dropout = float(config.get('dropout', 0.0))

        # composition ids (recorded so a run log always shows what actually ran)
        self.encoder_module = str(config.get('encoder_module', 'feature_embedding_encoder')).lower()
        self.spatial_module = str(config.get('spatial_module', 'adaptive_gcn')).lower()
        self.temporal_module = str(config.get('temporal_module', 'gated_dilated_tcn')).lower()
        self.fusion_module = str(config.get('fusion_module', 'concat')).lower()
        self.head_module = str(config.get('head_module', 'linear_head')).lower()

        # module hyperparameters
        self.spatial_embed_dim = max(1, int(config.get('spatial_embed_dim', 10)))
        self.spatial_ks = max(1, int(config.get('spatial_ks', config.get('ks', 2))))
        # False -> the adaptive graph is learned from node embeddings (default);
        # True  -> the dataset's (N, N) distance graph is used as the Neumann base.
        self.spatial_use_prior_adj = bool(config.get('spatial_use_prior_adj', False))
        self.temporal_layers = max(1, int(config.get('temporal_layers', 3)))
        self.temporal_kernel_size = max(2, int(config.get('temporal_kernel_size', 2)))
        self.temporal_base_dilation = max(1, int(config.get('temporal_base_dilation', 2)))

        # ---- optional fixed graph support (only used when spatial_use_prior_adj) --
        self.register_buffer('adj_mx', self._build_adjacency())

        # ---- encoder: traffic channel(s) -> hidden_dim ---------------------
        self.encoder = FeatureEmbeddingEncoder(
            input_dim=self.input_channels, hidden_dim=self.hidden_dim,
            dropout=self.dropout, bias=True)

        # ---- depth x (spatial || temporal -> fusion) blocks ----------------
        self.spatial_layers = nn.ModuleList()
        self.temporal_layers_list = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        for _ in range(self.depth):
            self.spatial_layers.append(AdaptiveGCN(
                hidden_dim=self.hidden_dim, num_nodes=self.num_nodes,
                embed_dim=self.spatial_embed_dim, ks=self.spatial_ks,
                dropout=self.dropout, bias=True))
            self.temporal_layers_list.append(GatedDilatedTCN(
                hidden_dim=self.hidden_dim, num_layers=self.temporal_layers,
                kernel_size=self.temporal_kernel_size, dropout=self.dropout,
                input_dim=self.hidden_dim, base_dilation=self.temporal_base_dilation))
            self.fusion_layers.append(ConcatFusion(hidden_dim=self.hidden_dim, dropout=self.dropout))

        # ---- direct multi-step regression head -----------------------------
        self.head = LinearHead(
            hidden_dim=self.hidden_dim, input_len=self.input_window,
            horizon=self.output_window, output_dim=self.output_dim,
            dropout=self.dropout, bias=True)

        self._logger.info(
            'ACD4Compose: encoder=%s(%d -> %d), spatial=%s(ks=%d, embed_dim=%d, prior_adj=%s), '
            'temporal=%s(layers=%d, k=%d), fusion=%s, head=%s, N=%d, depth=%d, loss=%s',
            self.encoder_module, self.input_channels, self.hidden_dim,
            self.spatial_module, self.spatial_ks, self.spatial_embed_dim, self.spatial_use_prior_adj,
            self.temporal_module, self.temporal_layers, self.temporal_kernel_size,
            self.fusion_module, self.head_module, self.num_nodes, self.depth, self.loss_type)

    # -- adjacency ---------------------------------------------------------
    def _build_adjacency(self):
        """Take LibCity's runtime (N, N) adjacency, sanitising non-finite entries."""
        n = self.num_nodes
        adj = self.data_feature.get('adj_mx', None)
        if adj is None:
            self._logger.warning(
                'adj_mx missing in data_feature; ACD4Compose falls back to the learned adaptive graph.')
            return torch.eye(n, dtype=torch.float32)
        adj = np.asarray(adj, dtype=np.float32)
        if adj.ndim != 2 or adj.shape[0] != n or adj.shape[1] != n:
            self._logger.warning(
                'adj_mx shape %s incompatible with num_nodes=%d; using identity graph.', str(adj.shape), n)
            return torch.eye(n, dtype=torch.float32)
        # distance-derived weights may contain inf for unreachable pairs
        adj = np.where(np.isfinite(adj), adj, 0.0)
        adj = np.clip(adj, a_min=0.0, a_max=None)
        return torch.from_numpy(np.ascontiguousarray(adj))

    # -- named module attributes -------------------------------------------
    # Contract: expose ``encoder``, ``spatial``, ``temporal``, ``fusion`` and
    # ``head`` as named attributes.  ``encoder`` / ``head`` are plain attributes;
    # the per-branch modules live in ModuleLists so the same source supports
    # depth > 1, and these read-only views expose the active (first) block.
    # They add no parameters and do not change the forward path.
    @property
    def spatial(self):
        return self.spatial_layers[0]

    @property
    def temporal(self):
        return self.temporal_layers_list[0]

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
        if data.shape[-1] > self.input_channels:
            data = data[..., :self.input_channels]

        adj = self.adj_mx.to(device=data.device, dtype=data.dtype)
        prior_adj = adj if self.spatial_use_prior_adj else None

        hidden = self.encoder(data)             # (B, T, N, hidden_dim) = encoded_hidden
        for i in range(self.depth):
            # ablation: a disabled branch is exactly torch.zeros_like(encoded_hidden)
            # and is fed into the SAME fusion / head path (no shortcut).
            if self.spatial_module == 'none':
                h_sp = torch.zeros_like(hidden)
            else:
                h_sp = self.spatial_layers[i](hidden, prior_adj)
            if self.temporal_module == 'none':
                h_tm = torch.zeros_like(hidden)
            else:
                h_tm = self.temporal_layers_list[i](hidden)
            hidden = self.fusion_layers[i](h_sp, h_tm)

        return self.head(hidden)                # (B, output_window, N, output_dim)

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        y_true = batch['y'][..., :self.output_dim]
        y_predicted = self.predict(batch)[..., :self.output_dim]
        # inverse-transform BOTH sides first: the masked loss lives in vehicles / 5 min
        y_true = self._scaler.inverse_transform(y_true)
        y_predicted = self._scaler.inverse_transform(y_predicted)
        y_true = y_true.to(y_predicted.dtype)
        if self.loss_type in ('masked_mse', 'mse'):
            return loss.masked_mse_torch(y_predicted, y_true, null_val=self.null_value)
        if self.loss_type in ('masked_mape', 'mape'):
            return loss.masked_mape_torch(y_predicted, y_true, null_val=self.null_value, eps=1e-3)
        return loss.masked_mae_torch(y_predicted, y_true, null_val=self.null_value)
