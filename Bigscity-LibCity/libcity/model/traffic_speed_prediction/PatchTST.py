"""
PatchTST Model for LibCity Traffic Speed Prediction

Adapted from: https://github.com/yuqinie98/PatchTST
Paper: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR 2023)

Original Files:
- models/PatchTST.py
- layers/PatchTST_backbone.py
- layers/PatchTST_layers.py
- layers/RevIN.py

Key Adaptations:
1. Inherits from AbstractTrafficStateModel
2. Reshapes LibCity's [batch, seq_len, num_nodes, features] to [batch, seq_len, num_nodes * features]
3. Implements predict() and calculate_loss() methods
4. Uses LibCity's config and data_feature pattern
5. All layers consolidated into single file

Hyperparameters (from config):
- input_window: Input sequence length (default: 12)
- output_window: Prediction length (default: 12)
- e_layers: Number of encoder layers (default: 3)
- n_heads: Number of attention heads (default: 16)
- d_model: Model dimension (default: 128)
- d_ff: Feed-forward dimension (default: 256)
- patch_len: Patch length (default: 16)
- stride: Stride for patching (default: 8)
- dropout: Dropout rate (default: 0.2)
- fc_dropout: FC dropout (default: 0.2)
- head_dropout: Head dropout (default: 0.0)
- revin: Use RevIN normalization (default: 1)
- affine: RevIN affine transformation (default: 0)
- decomposition: Series decomposition (default: 0)
- kernel_size: Decomposition kernel size (default: 25)
- individual: Individual head per channel (default: 0)
- padding_patch: Patch padding ('end' or None, default: 'end')
"""

from logging import getLogger
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


# ====================== RevIN Layer ======================
class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time series
    From: https://github.com/ts-kim/RevIN
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


# ====================== Utility Layers ======================
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


# ====================== Decomposition Layers ======================
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# ====================== Positional Encoding ======================
def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * \
              (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        if abs(cpe.mean()) <= eps:
            break
        elif cpe.mean() > eps:
            x += .001
        else:
            x -= .001
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe, learn_pe, q_len, d_model):
    if pe is None:
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos':
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(f"{pe} is not a valid pe (positional encoder)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)


# ====================== Attention Layers ======================
class _ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention with optional residual attention
    """
    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        attn_scores = torch.matmul(q, k) * self.scale

        if prev is not None:
            attn_scores = attn_scores + prev

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False,
                 attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout,
                                                   res_attention=self.res_attention, lsa=lsa)

        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None,
                prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None):
        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q

        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                               key_padding_mask=key_padding_mask,
                                                               attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask,
                                                  attn_mask=attn_mask)

        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu",
                 res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout,
                                              proj_dropout=dropout, res_attention=res_attention)

        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:
        if self.pre_norm:
            src = self.norm_attn(src)
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask,
                                                 attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        if self.pre_norm:
            src = self.norm_ffn(src)
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v,
                                                      d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                                      dropout=dropout, activation=activation,
                                                      res_attention=res_attention, pre_norm=pre_norm,
                                                      store_attn=store_attn) for _ in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask,
                                     attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers:
                output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


class TSTiEncoder(nn.Module):
    """Channel-independent encoder"""
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)
        self.seq_len = q_len

        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        self.dropout = nn.Dropout(dropout)

        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                   attn_dropout=attn_dropout, dropout=dropout, pre_norm=pre_norm,
                                   activation=act, res_attention=res_attention, n_layers=n_layers,
                                   store_attn=store_attn)

    def forward(self, x) -> Tensor:
        n_vars = x.shape[1]
        x = x.permute(0, 1, 3, 2)  # [bs, nvars, patch_num, patch_len]
        x = self.W_P(x)  # [bs, nvars, patch_num, d_model]

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # [bs*nvars, patch_num, d_model]
        u = self.dropout(u + self.W_pos)

        z = self.encoder(u)  # [bs*nvars, patch_num, d_model]
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))  # [bs, nvars, patch_num, d_model]
        z = z.permute(0, 1, 3, 2)  # [bs, nvars, d_model, patch_num]

        return z


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs, nvars, d_model, patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # [bs, nvars, target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


# ====================== PatchTST Backbone ======================
class PatchTST_backbone(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int,
                 max_seq_len: Optional[int] = 1024, n_layers: int = 3, d_model=128, n_heads=16,
                 d_k: Optional[int] = None, d_v: Optional[int] = None, d_ff: int = 256,
                 norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0., act: str = "gelu",
                 key_padding_mask: bool = 'auto', padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None, res_attention: bool = True, pre_norm: bool = False,
                 store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True, fc_dropout: float = 0.,
                 head_dropout=0, padding_patch=None, pretrain_head: bool = False, head_type='flatten',
                 individual=False, revin=True, affine=True, subtract_last=False, verbose: bool = False, **kwargs):
        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Backbone
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                     n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v,
                                     d_ff=d_ff, attn_dropout=attn_dropout, dropout=dropout, act=act,
                                     key_padding_mask=key_padding_mask, padding_var=padding_var,
                                     attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                     store_attn=store_attn, pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout)
        elif head_type == 'flatten':
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                     head_dropout=head_dropout)

    def forward(self, z):  # z: [bs, nvars, seq_len]
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [bs, nvars, patch_num, patch_len]
        z = z.permute(0, 1, 3, 2)  # [bs, nvars, patch_len, patch_num]

        # model
        z = self.backbone(z)  # [bs, nvars, d_model, patch_num]
        z = self.head(z)  # [bs, nvars, target_window]

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout), nn.Conv1d(head_nf, vars, 1))


# ====================== Main PatchTST Model for LibCity ======================
class PatchTST(AbstractTrafficStateModel):
    """
    PatchTST model adapted for LibCity traffic speed prediction.

    Data format transformation:
    - LibCity input: [batch, seq_len, num_nodes, features]
    - PatchTST expects: [batch, seq_len, channels] where channels = num_nodes * features
    - Output: [batch, pred_len, num_nodes, output_dim]
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))

        # Data features
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self._scaler = self.data_feature.get('scaler')

        # Model parameters from config
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.input_dim = self.data_feature.get('feature_dim', config.get('input_dim', 1))
        self.output_dim = config.get('output_dim', 1)

        # PatchTST specific parameters
        self.e_layers = config.get('e_layers', 3)
        self.n_heads = config.get('n_heads', 16)
        self.d_model = config.get('d_model', 128)
        self.d_ff = config.get('d_ff', 256)
        self.patch_len = config.get('patch_len', 16)
        self.stride = config.get('stride', 8)
        self.dropout = config.get('dropout', 0.2)
        self.fc_dropout = config.get('fc_dropout', 0.2)
        self.head_dropout = config.get('head_dropout', 0.0)

        # RevIN parameters
        self.revin = config.get('revin', 1)
        self.affine = config.get('affine', 0)
        self.subtract_last = config.get('subtract_last', 0)

        # Decomposition parameters
        self.decomposition = config.get('decomposition', 0)
        self.kernel_size = config.get('kernel_size', 25)

        # Other parameters
        self.individual = config.get('individual', 0)
        self.padding_patch = config.get('padding_patch', 'end')
        self.pe = config.get('pe', 'zeros')
        self.learn_pe = config.get('learn_pe', True)
        self.attn_dropout = config.get('attn_dropout', 0.)
        self.res_attention = config.get('res_attention', True)
        self.pre_norm = config.get('pre_norm', False)
        self.norm = config.get('norm', 'BatchNorm')

        # Calculate total channels (num_nodes * input_dim for traffic data)
        # For traffic prediction: treat each node's feature as a channel
        self.c_in = self.num_nodes * self.input_dim

        # Build model
        self._build_model()

        self._logger.info(f"PatchTST initialized with: num_nodes={self.num_nodes}, "
                          f"input_window={self.input_window}, output_window={self.output_window}, "
                          f"c_in={self.c_in}, d_model={self.d_model}, n_heads={self.n_heads}, "
                          f"e_layers={self.e_layers}, patch_len={self.patch_len}, stride={self.stride}")

    def _build_model(self):
        """Build the PatchTST model components"""
        self.decomposition_flag = self.decomposition
        if self.decomposition_flag:
            self.decomp_module = series_decomp(self.kernel_size)
            self.model_trend = PatchTST_backbone(
                c_in=self.c_in,
                context_window=self.input_window,
                target_window=self.output_window,
                patch_len=self.patch_len,
                stride=self.stride,
                n_layers=self.e_layers,
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                fc_dropout=self.fc_dropout,
                head_dropout=self.head_dropout,
                individual=self.individual,
                revin=self.revin,
                affine=self.affine,
                subtract_last=self.subtract_last,
                padding_patch=self.padding_patch,
                pe=self.pe,
                learn_pe=self.learn_pe,
                attn_dropout=self.attn_dropout,
                res_attention=self.res_attention,
                pre_norm=self.pre_norm,
                norm=self.norm
            )
            self.model_res = PatchTST_backbone(
                c_in=self.c_in,
                context_window=self.input_window,
                target_window=self.output_window,
                patch_len=self.patch_len,
                stride=self.stride,
                n_layers=self.e_layers,
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                fc_dropout=self.fc_dropout,
                head_dropout=self.head_dropout,
                individual=self.individual,
                revin=self.revin,
                affine=self.affine,
                subtract_last=self.subtract_last,
                padding_patch=self.padding_patch,
                pe=self.pe,
                learn_pe=self.learn_pe,
                attn_dropout=self.attn_dropout,
                res_attention=self.res_attention,
                pre_norm=self.pre_norm,
                norm=self.norm
            )
        else:
            self.model = PatchTST_backbone(
                c_in=self.c_in,
                context_window=self.input_window,
                target_window=self.output_window,
                patch_len=self.patch_len,
                stride=self.stride,
                n_layers=self.e_layers,
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                fc_dropout=self.fc_dropout,
                head_dropout=self.head_dropout,
                individual=self.individual,
                revin=self.revin,
                affine=self.affine,
                subtract_last=self.subtract_last,
                padding_patch=self.padding_patch,
                pe=self.pe,
                learn_pe=self.learn_pe,
                attn_dropout=self.attn_dropout,
                res_attention=self.res_attention,
                pre_norm=self.pre_norm,
                norm=self.norm
            )

    def forward(self, batch):
        """
        Forward pass

        Args:
            batch: dict with 'X' key containing input tensor
                   X shape: [batch, input_window, num_nodes, input_dim]

        Returns:
            output: [batch, output_window, num_nodes, output_dim]
        """
        x = batch['X']  # [batch, input_window, num_nodes, input_dim]
        batch_size = x.shape[0]

        # Extract only the feature dimensions needed (first input_dim features)
        x = x[..., :self.input_dim]  # [batch, input_window, num_nodes, input_dim]

        # Reshape from [batch, seq_len, num_nodes, features] to [batch, seq_len, num_nodes * features]
        x = x.reshape(batch_size, self.input_window, -1)  # [batch, input_window, num_nodes * input_dim]

        # Apply model
        if self.decomposition_flag:
            res_init, trend_init = self.decomp_module(x)
            res_init = res_init.permute(0, 2, 1)  # [batch, channels, input_window]
            trend_init = trend_init.permute(0, 2, 1)
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0, 2, 1)  # [batch, output_window, channels]
        else:
            x = x.permute(0, 2, 1)  # [batch, channels, input_window]
            x = self.model(x)
            x = x.permute(0, 2, 1)  # [batch, output_window, channels]

        # Reshape back from [batch, output_window, num_nodes * input_dim] to [batch, output_window, num_nodes, output_dim]
        # Note: We only predict output_dim features (typically speed only)
        x = x.reshape(batch_size, self.output_window, self.num_nodes, self.input_dim)

        # Select only output_dim features
        x = x[..., :self.output_dim]

        return x  # [batch, output_window, num_nodes, output_dim]

    def predict(self, batch):
        """
        Predict for a batch

        Args:
            batch: dict with 'X' key

        Returns:
            predictions: [batch, output_window, num_nodes, output_dim]
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate training loss

        Args:
            batch: dict with 'X' and 'y' keys

        Returns:
            loss: scalar tensor
        """
        y_true = batch['y']  # [batch, output_window, num_nodes, features]
        y_predicted = self.predict(batch)

        # Apply inverse scaler for proper loss calculation
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        return loss.masked_mae_torch(y_predicted, y_true, null_val=0.0)
