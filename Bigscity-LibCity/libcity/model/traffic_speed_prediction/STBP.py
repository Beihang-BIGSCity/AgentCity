"""
STBP: LibCity adaptation of the STBP model.

Reference:
    "When Prompt Meets Drift: Continual Spatio-Temporal Learning with Dynamic
    Pattern Banks" (STBP).

Original source:
    repos/STBP/src/model/model.py

Key adaptation notes:
    1. torch_geometric dependency removed. The original model consumes a
       ``torch_geometric.data.Data`` object whose ``data.x`` has shape
       ``[B*N, T_in]``. Here we consume LibCity's ``batch['X']`` of shape
       ``[B, T_in, N, F]`` directly and reshape to ``[B, N, T_in*F]``.
    2. The ``adj`` argument of the original forward is a pure pass-through:
       the DLGA attention is kernel-based and never uses the adjacency
       matrix, so we drop it entirely.
    3. Continual-learning trainer-side behaviour (yearly node expansion /
       parameter freezing) is intentionally not ported. The pattern bank is
       initialised once using ``data_feature['num_nodes']``. The
       ``expand_adaptive_params`` helper from the original is preserved as a
       method should downstream users want to reuse it.
    4. Loss follows LibCity conventions: masked MAE with ``null_val=0``
       computed in the original (unscaled) space via ``self._scaler``.

Config / data_feature mapping:
    - ``config['in_channel']``      <- original ``args.model['in_channel']`` (default 12)
    - ``config['hidden_channel']``  <- original ``args.model['hidden_channel']`` (default 192)
    - ``config['out_channel']``     <- original ``args.model['out_channel']`` (default 12)
    - ``config['output_window']``   <- original ``args.y_len`` (default 12)
    - ``config['dropout']``         <- original ``args.dropout`` (default 0.0)
    - ``data_feature['num_nodes']`` <- original ``args.base_node_size``
    - ``data_feature['scaler']``    used for inverse transform in loss/predict
"""

import math
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


# ---------------------------------------------------------------------------
# Building blocks (ported verbatim from repos/STBP/src/model/model.py, with
# only cosmetic cleanups -- torch_geometric is not used here).
# ---------------------------------------------------------------------------


class LayerNorm(nn.Module):
    """Channel + node LayerNorm used by the original STBP encoder."""

    def __init__(self, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(1, 2), keepdim=True)
        variance = x.var(dim=(1, 2), unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(variance + self.eps)


class Mlp(nn.Module):
    """Conv2d-based pointwise MLP used as the feed-forward block."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.2):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=(1, 1))
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=(1, 1))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Conv(nn.Module):
    """1x1 Conv + ReLU + Dropout, used for Q/K/V projections inside DLGA."""

    def __init__(self, features, dropout=0.2):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class DLGA(nn.Module):
    """Dynamic Linear Graph Attention (kernel trick -> O(N))."""

    def __init__(self, d_model, head, seq_length=1):
        super(DLGA, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.head = head
        self.q = Conv(d_model)
        self.k = Conv(d_model)
        self.v = Conv(d_model)
        self.concat = Conv(d_model)

    def forward(self, x, node_emb):
        _, d_model, num_nodes, seq_length = x.shape
        query, key, value = self.q(x), self.k(x), self.v(x)

        query = query.view(
            query.shape[0], -1, self.d_k, query.shape[2], seq_length
        ).permute(0, 1, 4, 3, 2)
        key = key.view(
            key.shape[0], -1, self.d_k, key.shape[2], seq_length
        ).permute(0, 1, 4, 3, 2)
        value = value.view(
            value.shape[0], -1, self.d_k, value.shape[2], seq_length
        ).permute(0, 1, 4, 3, 2)
        node_emb = node_emb.view(
            node_emb.shape[0], -1, self.d_k, node_emb.shape[2], seq_length
        ).permute(0, 1, 4, 3, 2)

        key = torch.softmax(key / math.sqrt(self.d_k), dim=-1)
        query = torch.softmax(query / math.sqrt(self.d_k), dim=-1)
        node_emb = torch.softmax(node_emb / math.sqrt(self.d_k), dim=-1)

        kv = torch.einsum("bhlnx, bhlny->bhlxy", key, value)
        attn_qkv1 = torch.einsum("bhlnx, bhlxy->bhlny", query, kv)

        node_emb_v = torch.einsum("bhlnx, bhlny->bhlxy", node_emb, value)
        attn_qkv2 = torch.einsum("bhlnx, bhlxy->bhlny", query, node_emb_v)

        attn_qkv = attn_qkv1 + attn_qkv2
        out = (
            attn_qkv.permute(0, 1, 4, 3, 2)
            .contiguous()
            .view(attn_qkv.shape[0], d_model, num_nodes, seq_length)
        )
        out = self.concat(out)
        return out


def modulate(x, shift, scale):
    """AdaLN-style modulation: x * (1 + scale) + shift."""
    return x * (1 + scale) + shift


class Encoder(nn.Module):
    """STBP ST-module: DLGA attention + pointwise FFW with gated modulation."""

    def __init__(self, d_model, head, seq_length=1):
        super(Encoder, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.head = head
        self.seq_length = seq_length
        self.d_model = d_model

        self.attn = DLGA(d_model, head, seq_length)
        self.FFW = Mlp(d_model)

        self.norm1 = LayerNorm()
        self.norm2 = LayerNorm()

    def forward(self, x, node_emb):
        gate, scale, memory = torch.chunk(node_emb, 3, dim=1)
        x = x + gate * self.attn(modulate(self.norm1(x), 0, scale), memory)
        x = x + gate * self.FFW(modulate(self.norm2(x), 0, scale))
        return x


class FreNetwork(nn.Module):
    """Frequency-domain filter applied along the temporal axis (rFFT)."""

    def __init__(self, time_dim=64):
        super(FreNetwork, self).__init__()
        self.time_dim = time_dim
        self.complex_weight = nn.Parameter(
            torch.randn(1, 1, time_dim // 2 + 1, 2, dtype=torch.float32) * 0.02
        )
        nn.init.xavier_uniform_(self.complex_weight)

    def forward(self, x):
        # x: [B, N, T]
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        t_bank = torch.view_as_complex(self.complex_weight)
        x = x * t_bank
        x = torch.fft.irfft(x, n=self.time_dim, dim=2, norm='ortho')
        return x


class FConv(nn.Module):
    """Linear projection + frequency-domain filter, optionally with a residual linear."""

    def __init__(self, in_features, out_features, bias=True, fc=True):
        super(FConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_neigh = nn.Linear(in_features, out_features, bias=bias)
        self.fre_network = FreNetwork(out_features)
        if not fc:
            self.weight_self = nn.Linear(in_features, out_features, bias=False)
        else:
            self.register_parameter('weight_self', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight_neigh.reset_parameters()
        if self.weight_self is not None:
            self.weight_self.reset_parameters()

    def forward(self, x):
        output = self.fre_network(self.weight_neigh(x))
        if self.weight_self is not None:
            output = output + self.weight_self(x)
        return output


# ---------------------------------------------------------------------------
# LibCity STBP wrapper
# ---------------------------------------------------------------------------


class STBP(AbstractTrafficStateModel):
    """LibCity-compatible STBP model.

    Expected batch format:
        batch['X']: (B, T_in, N, F_in)
        batch['y']: (B, T_out, N, F_out)

    Returned prediction shape:
        (B, T_out, N, output_dim)
    """

    def __init__(self, config, data_feature):
        super(STBP, self).__init__(config, data_feature)
        self._logger = getLogger()

        # --- data_feature ----------------------------------------------------
        self._scaler = data_feature.get('scaler')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)

        # --- config ---------------------------------------------------------
        self.device = config.get('device', torch.device('cpu'))
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.dropout = config.get('dropout', 0.0)

        # Original STBP works on length-12 scalar series (feature_dim=1).
        # For LibCity we default ``in_channel`` to ``input_window`` so that
        # only the first (target) feature is fed into the model, matching the
        # original behaviour. If ``feature_dim`` > 1, extra features are
        # ignored unless the caller overrides ``in_channel`` explicitly.
        self.in_channel = config.get('in_channel', self.input_window)
        self.hidden_channel = config.get('hidden_channel', 192)
        self.out_channel = config.get('out_channel', self.in_channel)
        self.head = config.get('head', 1)

        # Final projection produces ``output_window * output_dim`` scalars
        # per node, which we then reshape back to (T_out, output_dim).
        self.y_len = self.output_window * self.output_dim

        # --- layers ---------------------------------------------------------
        self.fconv1 = FConv(self.in_channel, self.hidden_channel, bias=True, fc=False)
        self.stmodule = Encoder(
            d_model=self.hidden_channel,
            head=self.head,
            seq_length=1,
        )
        self.fconv2 = FConv(self.hidden_channel, self.out_channel, bias=True, fc=False)
        self.fc = nn.Linear(self.out_channel, self.y_len)
        self.activation = nn.GELU()

        # Continual-learning pattern bank (gate / scale / memory per node).
        self.pattern_bank = nn.Parameter(
            torch.empty(self.num_nodes, self.hidden_channel * 3).uniform_(-0.1, 0.1)
        )

        self._logger.info(
            'STBP initialised: num_nodes=%d in_channel=%d hidden=%d out_channel=%d y_len=%d',
            self.num_nodes, self.in_channel, self.hidden_channel, self.out_channel, self.y_len,
        )

    # ------------------------------------------------------------------ utils
    def expand_adaptive_params(self, new_num_nodes):
        """Expand the pattern bank in place (kept for parity with the original).

        In standard LibCity benchmarks the node count is fixed, so this is a
        no-op unless explicitly called by user code (e.g. continual setups).
        """
        if new_num_nodes > self.num_nodes:
            extra = torch.empty(
                new_num_nodes - self.num_nodes,
                self.hidden_channel * 3,
                dtype=self.pattern_bank.dtype,
                device=self.pattern_bank.device,
            ).uniform_(-0.1, 0.1)
            self.pattern_bank = nn.Parameter(
                torch.cat([self.pattern_bank.data, extra], dim=0)
            )
            self.num_nodes = new_num_nodes

    # ---------------------------------------------------------------- forward
    def forward(self, batch):
        """Run a forward pass.

        Args:
            batch: dict-like with key ``'X'`` of shape [B, T_in, N, F_in].

        Returns:
            Tensor of shape [B, T_out, N, output_dim].
        """
        x = batch['X']                        # [B, T_in, N, F_in]
        bsz, t_in, num_nodes, f_in = x.shape

        # Use only the target (first) channel, mirroring the original STBP
        # single-series formulation. Shape becomes [B, N, T_in].
        x_series = x[..., 0].permute(0, 2, 1).contiguous()  # [B, N, T_in]

        if x_series.shape[-1] != self.in_channel:
            # Fall back to raising so that config drift is caught early.
            raise ValueError(
                f"STBP expects in_channel={self.in_channel} time steps per node, "
                f"but got {x_series.shape[-1]} (input_window={t_in})."
            )

        # Keep an input copy for the residual connection.
        x_in = x_series                                      # [B, N, T_in]

        # FConv 1: in_channel -> hidden_channel.
        h = F.relu(self.fconv1(x_series))                    # [B, N, hidden]

        # Build node embeddings from the pattern bank:
        # [num_nodes, hidden*3] -> [B, hidden*3, N, 1]
        adaptive_params = self.pattern_bank
        node_emb = (
            adaptive_params.unsqueeze(0)
            .expand(bsz, *adaptive_params.shape)
            .permute(0, 2, 1)
            .unsqueeze(-1)
            .contiguous()
        )

        # ST module expects [B, C, N, 1].
        h = h.permute(0, 2, 1).unsqueeze(-1)                 # [B, hidden, N, 1]
        h = self.stmodule(h, node_emb)                       # [B, hidden, N, 1]
        h = h.squeeze(-1).permute(0, 2, 1)                   # [B, N, hidden]

        # FConv 2: hidden_channel -> out_channel.
        h = self.fconv2(h)                                   # [B, N, out_channel]

        # Residual with the original series (requires out_channel == in_channel).
        if h.shape[-1] == x_in.shape[-1]:
            h = h + x_in

        # Output projection: [B, N, out_channel] -> [B, N, y_len].
        h = self.fc(self.activation(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Reshape to LibCity format [B, T_out, N, output_dim].
        out = h.view(bsz, num_nodes, self.output_window, self.output_dim)
        out = out.permute(0, 2, 1, 3).contiguous()
        return out

    # ---------------------------------------------------------------- predict
    def predict(self, batch):
        return self.forward(batch)

    # --------------------------------------------------------- loss
    def calculate_loss(self, batch):
        y_true = batch['y']
        y_pred = self.predict(batch)

        # Align feature dim with ``output_dim`` before inverse scaling.
        y_true = y_true[..., :self.output_dim]
        y_pred = y_pred[..., :self.output_dim]

        if self._scaler is not None:
            y_true = self._scaler.inverse_transform(y_true)
            y_pred = self._scaler.inverse_transform(y_pred)

        # Masked MAE with null_val=0.0 as in the original STBP pipeline.
        return loss.masked_mae_torch(y_pred, y_true, 0.0)
