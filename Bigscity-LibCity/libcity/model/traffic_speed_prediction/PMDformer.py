from logging import getLogger
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class SelfAttention(nn.Module):
    """Self-attention used by the original PMDformer PVA encoder."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, tgt_len, _ = x.shape
        src_len = x.shape[1]
        num_heads = self.n_heads

        queries = self.query_projection(x).view(batch_size, tgt_len, num_heads, -1)
        keys = self.key_projection(x).view(batch_size, src_len, num_heads, -1)
        values = self.value_projection(x).view(batch_size, src_len, num_heads, -1)

        scale = 1.0 / sqrt(queries.size(-1))
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        values = torch.einsum("bhls,bshd->blhd", attn, values)

        values = values.reshape(batch_size, tgt_len, -1)
        out = self.out_projection(values)
        return out, attn


class TrendRestorationAttention(nn.Module):
    """Trend restoration attention from the original PMDformer implementation."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, means):
        batch_size, tgt_len, _ = x.shape
        src_len = x.shape[1]
        num_heads = self.n_heads

        queries = self.query_projection(x).view(batch_size, tgt_len, num_heads, -1)
        keys = self.key_projection(x).view(batch_size, src_len, num_heads, -1)
        values = self.value_projection(x).view(batch_size, src_len, num_heads, -1) + means.unsqueeze(2)

        scale = 1.0 / sqrt(queries.size(-1))
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        values = torch.einsum("bhls,bshd->blhd", attn, values)

        values = values.reshape(batch_size, tgt_len, -1)
        out = self.out_projection(values)
        return out, attn


class EncoderLayerPVA(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x):
        new_x, attn = self.attention(x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        return self.norm2(x + y), attn


class EncoderPVA(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class EncoderLayerTRA(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x, means):
        new_x, attn = self.attention(x, means)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        return self.norm2(x + y), attn


class EncoderTRA(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, means):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, means)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class PMDformerCore(nn.Module):
    """
    Core PMDformer adapted from the external implementation.

    Expected input/output:
    - input:  [B, seq_len, num_series]
    - output: [B, pred_len, num_series]
    """

    def __init__(self, seq_len, pred_len, d_model, enc_in, e_layers, v_layers,
                 n_heads, d_ff, dropout, patch_size, use_norm):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.enc_in = enc_in
        self.e_layers = e_layers
        self.v_layers = v_layers
        self.use_norm = use_norm
        self.patch_size = patch_size
        self.stride = patch_size

        if seq_len < patch_size:
            raise ValueError(
                f"PMDformer requires input_window >= patch_size, got input_window={seq_len}, patch_size={patch_size}."
            )

        self.patch_num = (seq_len - patch_size) // self.stride + 1
        if self.patch_num <= 0:
            raise ValueError(
                f"Invalid patch_num={self.patch_num}. Check input_window={seq_len} and patch_size={patch_size}."
            )

        self.embedding_layer = nn.Linear(patch_size, d_model)
        self.predicting_layer = nn.Linear(d_model * self.patch_num, pred_len)

        self.encoder_tra = EncoderTRA(
            [
                EncoderLayerTRA(
                    TrendRestorationAttention(d_model=d_model, n_heads=n_heads, dropout=dropout),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.encoder_pva = EncoderPVA(
            [
                EncoderLayerPVA(
                    SelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout
                )
                for _ in range(v_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.patch_pos = nn.Parameter(torch.zeros(self.patch_num, d_model))

    def forward(self, x_enc):
        x = x_enc
        batch_size, seq_len, num_series = x.shape

        x = x.permute(0, 2, 1).contiguous()  # [B, N, L]
        if self.use_norm:
            means = x.mean(-1, keepdim=True)
            stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5)
            x = (x - means) / stdev
        else:
            means = None
            stdev = None

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # [B, N, P, S]
        x = x.reshape(batch_size * num_series, self.patch_num, self.patch_size)  # [B*N, P, S]

        patch_means = x.mean(-1, keepdim=True)
        x = x - patch_means

        x = self.embedding_layer(x)
        x = x + self.patch_pos.unsqueeze(0)

        x_main = x[:, :-1, :]
        x_last = x[:, -1:, :]
        x_last = x_last.reshape(batch_size, num_series, 1, self.d_model)
        x_last = x_last.permute(0, 2, 1, 3).reshape(batch_size, num_series, self.d_model)
        x_last, _ = self.encoder_pva(x_last)
        x_last = x_last.reshape(batch_size, 1, num_series, self.d_model)
        x_last = x_last.permute(0, 2, 1, 3).reshape(batch_size * num_series, 1, self.d_model)
        x = torch.cat([x_main, x_last], dim=1)

        x, _ = self.encoder_tra(x, patch_means)
        x = x + patch_means

        x = x.reshape(batch_size, num_series, self.patch_num * self.d_model)
        x = self.predicting_layer(x)  # [B, N, T]

        if self.use_norm:
            x = x * stdev + means

        x = x.permute(0, 2, 1).contiguous()  # [B, T, N]
        return x


class PMDformer(AbstractTrafficStateModel):
    """
    LibCity-compatible PMDformer.

    Adaptation notes:
    - Original PMDformer consumes [B, L, N] and predicts [B, pred_len, N].
    - LibCity provides [B, L, num_nodes, feature_dim].
    - This adapter uses the first `output_dim` channels from LibCity input, flattens
      node/channel dimensions into PMDformer's variable axis, and reshapes the output
      back to [B, output_window, num_nodes, output_dim].
    - This preserves the original model's non-graph multivariate forecasting behavior.
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))

        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', config.get('input_dim', 1))
        self.output_dim = int(config.get('output_dim', data_feature.get('output_dim', 1)))
        self.input_window = int(config.get('input_window', 12))
        self.output_window = int(config.get('output_window', 12))
        self._scaler = data_feature.get('scaler')

        if self.output_dim > self.feature_dim:
            raise ValueError(
                f"PMDformer expects output_dim <= feature_dim, got output_dim={self.output_dim}, "
                f"feature_dim={self.feature_dim}."
            )

        self.series_dim = self.output_dim
        self.enc_in = self.num_nodes * self.series_dim

        # Original PMDformer hyperparameters.
        self.d_model = int(config.get('d_model', 512))
        self.n_heads = int(config.get('n_heads', 8))
        self.e_layers = int(config.get('e_layers', 2))
        self.v_layers = int(config.get('v_layers', 1))
        self.d_ff = int(config.get('d_ff', 2048))
        self.dropout = float(config.get('dropout', 0.1))
        self.patch_size = int(config.get('patch_size', 12))
        self.use_norm = bool(config.get('use_norm', True))

        self.core = PMDformerCore(
            seq_len=self.input_window,
            pred_len=self.output_window,
            d_model=self.d_model,
            enc_in=self.enc_in,
            e_layers=self.e_layers,
            v_layers=self.v_layers,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            patch_size=self.patch_size,
            use_norm=self.use_norm,
        )

        self._logger.info(
            "PMDformer initialized: num_nodes=%s, feature_dim=%s, output_dim=%s, enc_in=%s, "
            "input_window=%s, output_window=%s, d_model=%s, n_heads=%s, e_layers=%s, v_layers=%s, "
            "d_ff=%s, patch_size=%s, use_norm=%s",
            self.num_nodes, self.feature_dim, self.output_dim, self.enc_in,
            self.input_window, self.output_window, self.d_model, self.n_heads,
            self.e_layers, self.v_layers, self.d_ff, self.patch_size, self.use_norm
        )

    def _prepare_input(self, x):
        """
        Convert LibCity input [B, T, N, F] to PMDformer input [B, T, N*output_dim].
        """
        x = x[..., :self.series_dim]
        batch_size, time_steps, num_nodes, channels = x.shape
        return x.reshape(batch_size, time_steps, num_nodes * channels)

    def _restore_output(self, x):
        """
        Convert PMDformer output [B, T_out, N*output_dim] back to [B, T_out, N, output_dim].
        """
        batch_size, time_steps, _ = x.shape
        return x.reshape(batch_size, time_steps, self.num_nodes, self.output_dim)

    def forward(self, batch):
        x = batch['X']
        x = self._prepare_input(x)
        x = self.core(x)
        x = self._restore_output(x)
        return x

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        y_true = batch['y'][..., :self.output_dim]
        y_pred = self.predict(batch)

        if self._scaler is not None:
            y_true = self._scaler.inverse_transform(y_true)
            y_pred = self._scaler.inverse_transform(y_pred)

        return loss.masked_mae_torch(y_pred, y_true, null_val=0.0)
