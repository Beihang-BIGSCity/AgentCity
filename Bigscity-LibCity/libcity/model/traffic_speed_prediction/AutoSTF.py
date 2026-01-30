"""
AutoSTF Model for LibCity

Adapted from: https://github.com/AutoSTF
Original paper: Automated Spatio-Temporal Graph Contrastive Learning (WWW 2023)

This model performs Neural Architecture Search (NAS) for spatio-temporal forecasting.
In LibCity, we run the model in inference mode (ONE_PATH_FIXED) to bypass NAS search.

Key adaptations:
1. Inherit from AbstractTrafficStateModel
2. Convert input format from LibCity's [B, T, N, C] to AutoSTF's [B, C, N, T]
3. Implement predict() and calculate_loss() methods
4. Handle adjacency matrix from data_feature
5. Use LibCity's scaler for inverse transform
"""

import copy
import enum
import math
from logging import getLogger
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import ModuleList, Linear, Dropout, LayerNorm
from torch.nn.init import constant_, xavier_uniform_

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


# ============================================================================
# Mode Enum (from mode.py)
# ============================================================================
class Mode(enum.Enum):
    NONE = 0
    ONE_PATH_FIXED = 1
    ONE_PATH_RANDOM = 2
    TWO_PATHS = 3
    ALL_PATHS = 4


def get_mode(name):
    name2mode = {
        'NONE': Mode.NONE,
        'ONE_PATH_FIXED': Mode.ONE_PATH_FIXED,
        'ONE_PATH_RANDOM': Mode.ONE_PATH_RANDOM,
        'TWO_PATHS': Mode.TWO_PATHS,
        'ALL_PATHS': Mode.ALL_PATHS
    }
    return name2mode.get(name, Mode.NONE)


# ============================================================================
# Transformer Components (from transformer.py)
# ============================================================================
class LearnedPositionalEncoding(nn.Embedding):
    def __init__(self, d_model, dropout=0.1, max_len=207):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0), :]
        return self.dropout(x)


class LScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=0.1, groups=2):
        super(LScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model // groups, h * d_k // groups)
        self.fc_k = nn.Linear(d_model // groups, h * d_k // groups)
        self.fc_v = nn.Linear(d_model // groups, h * d_v // groups)
        self.fc_o = nn.Linear(h * d_v // groups, d_model // groups)
        self.dropout = nn.Dropout(dropout)
        self.groups = groups
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.fc_q.weight)
        xavier_uniform_(self.fc_k.weight)
        xavier_uniform_(self.fc_v.weight)
        xavier_uniform_(self.fc_o.weight)
        constant_(self.fc_q.bias, 0)
        constant_(self.fc_k.bias, 0)
        constant_(self.fc_v.bias, 0)
        constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        queries = queries.permute(1, 0, 2)
        keys = keys.permute(1, 0, 2)
        values = values.permute(1, 0, 2)
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries.view(b_s, nq, self.groups, -1)).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys.view(b_s, nk, self.groups, -1)).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values.view(b_s, nk, self.groups, -1)).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out.view(b_s, nq, self.groups, -1)).view(b_s, nq, -1)
        return out.permute(1, 0, 2)


class LMultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1, batch_first=False, groups=2, device=None, dtype=None):
        super(LMultiHeadAttention, self).__init__()
        self.attention = LScaledDotProductAttention(
            d_model=d_model, groups=groups, d_k=d_model // h, d_v=d_model // h, h=h, dropout=dropout)

    def forward(self, queries, keys, values, attn_mask=None, key_padding_mask=None,
                need_weights=False, attention_weights=None):
        out = self.attention(queries, keys, values, attn_mask, attention_weights)
        return out, out


class LinearFormerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.gelu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LinearFormerLayer, self).__init__()
        self.self_attn = LMultiHeadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward // 2, d_model // 2, **factory_kwargs)
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        if isinstance(activation, str):
            self.activation = F.gelu
        else:
            self.activation = activation

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        b, l, d = x.size()
        x = self.linear2(self.dropout(self.activation(self.linear1(x))).view(b, l, 2, d * 4 // 2))
        x = x.view(b, l, d)
        return self.dropout2(x)


class LinearFormer(nn.Module):
    def __init__(self, attention_layer, num_layers, norm=None):
        super(LinearFormer, self).__init__()
        self.layers = ModuleList([copy.deepcopy(attention_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for i, mod in enumerate(self.layers):
            if i % 2 == 0:
                output = mod(output)
            else:
                output = mod(output, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


# ============================================================================
# Linear Layer Components (from LinearLayer.py)
# ============================================================================
class LightLayer(nn.Module):
    def __init__(self, hid_dim, dim_feedforward):
        super(LightLayer, self).__init__()
        layer_norm_eps = 1e-5
        self.hid_dim = hid_dim
        self.linear1 = nn.Linear(hid_dim, dim_feedforward)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_feedforward // 2, self.hid_dim // 2)
        self.norm1 = nn.LayerNorm(self.hid_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.hid_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.activation = F.gelu

    def forward(self, inputs):
        x = inputs
        x1 = self.norm1(x)
        b, l, d = x1.size()
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x1))).view(b, l, 2, d * 4 // 2))
        x2 = x2.view(b, l, d)
        x2 = self.dropout2(x2)
        x = self.norm2(x1 + x2)
        return x


class LightLinear(nn.Module):
    def __init__(self, hidden_channels, num_linear_layers):
        super(LightLinear, self).__init__()
        self.hid_dim = hidden_channels
        self.layers = num_linear_layers
        self.LightLayers = nn.ModuleList()
        for _ in range(self.layers):
            self.LightLayers.append(LightLayer(self.hid_dim, self.hid_dim * 4))

    def forward(self, inputs):
        x = inputs.permute(1, 0, 2)
        for i, layer in enumerate(self.LightLayers):
            x = layer(x)
        output = x.permute(1, 0, 2)
        return output


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        hidden = hidden + input_data
        return hidden


# ============================================================================
# Candidate Operations (from CandidateOpration.py)
# ============================================================================
def gconv(x, A):
    A = A.to(torch.float32)
    x = torch.einsum('bnh,nn->bnh', (x, A))
    return x.contiguous()


class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, inputs, **kwargs):
        return inputs.mul(0.)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs, **kwargs):
        return inputs


class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self._padding = (kernel_size[-1] - 1) * dilation
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=(0, self._padding), dilation=dilation, groups=groups, bias=bias)

    def forward(self, inputs):
        result = super(CausalConv2d, self).forward(inputs)
        if self._padding != 0:
            return result[:, :, :, :-self._padding]
        return result


class DCCLayer(nn.Module):
    """Dilated causal convolution layer with GLU function."""

    def __init__(self, hidden_channels, kernel_size=(1, 2), stride=1, dilation=1):
        super(DCCLayer, self).__init__()
        c_in = hidden_channels
        c_out = hidden_channels
        self.relu = nn.ReLU()
        self.filter_conv = CausalConv2d(c_in, c_out, kernel_size, stride, dilation=dilation)
        self.gate_conv = CausalConv2d(c_in, c_out, kernel_size, stride, dilation=dilation)
        self.bn = nn.BatchNorm2d(c_out, affine=False)

    def forward(self, x, **kwargs):
        x = self.relu(x)
        filter = torch.tanh(self.filter_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        output = filter * gate
        output = self.bn(output)
        return output


class GNN_fixed(nn.Module):
    """K-order diffusion convolution layer with fixed adjacency matrix."""

    def __init__(self, hidden_channels, num_hop, adj_mx, device, use_bn=True, dropout=0.15):
        super(GNN_fixed, self).__init__()
        self.use_bn = use_bn
        self.dropout = dropout
        self.supports = []
        self.num_hop = num_hop
        adj_mx_dcrnn = adj_mx[0]
        supports = [adj_mx_dcrnn]
        for support in supports:
            self.supports.append(torch.tensor(support).to(device))
        self.linear = nn.Conv2d(hidden_channels * (len(self.supports) * self.num_hop + 1), hidden_channels,
                                kernel_size=(1, 1), stride=(1, 1))
        if use_bn:
            self.bn = nn.BatchNorm2d(hidden_channels)

    def forward(self, inputs, **kwargs):
        x = torch.relu(inputs)
        outputs = [x]
        for support in self.supports:
            for j in range(self.num_hop):
                x = gconv(x, support)
                outputs += [x]
        h = torch.cat(outputs, dim=2)
        h = h.unsqueeze(dim=1)
        h = h.transpose(1, 3)
        h = self.linear(h)
        if self.use_bn:
            h = self.bn(h)
        if self.dropout > 0:
            h = F.dropout(h, self.dropout, training=self.training)
        h = h.transpose(1, 3)
        h = h.squeeze(dim=1)
        return h


class GNN_adap(nn.Module):
    """K-order diffusion convolution layer with adaptive adjacency matrix."""

    def __init__(self, hidden_channels, num_hop, node_vec1, node_vec2, use_bn=True, dropout=0.15):
        super(GNN_adap, self).__init__()
        self.use_bn = use_bn
        self.dropout = dropout
        self.num_hop = num_hop
        self.node_vec1 = node_vec1
        self.node_vec2 = node_vec2
        self.linear = nn.Conv2d(hidden_channels * (self.num_hop + 1), hidden_channels, kernel_size=(1, 1), stride=(1, 1))
        if use_bn:
            self.bn = nn.BatchNorm2d(hidden_channels)

    def forward(self, inputs, **kwargs):
        x = torch.relu(inputs)
        adp = F.relu(torch.mm(self.node_vec1, self.node_vec2))
        mask = torch.zeros_like(adp) - 10 ** 10
        adp = torch.where(adp == 0, mask, adp)
        adp = F.softmax(adp, dim=1)
        outputs = [x]
        for _ in range(self.num_hop):
            x = gconv(x, adp)
            outputs += [x]
        h = torch.cat(outputs, dim=2)
        h = h.unsqueeze(dim=1)
        h = h.transpose(1, 3)
        h = self.linear(h)
        if self.use_bn:
            h = self.bn(h)
        if self.dropout > 0:
            h = F.dropout(h, self.dropout, training=self.training)
        h = h.transpose(1, 3)
        h = h.squeeze(dim=1)
        return h


class GNN_att(nn.Module):
    """GNN layer with attention mechanism."""

    def __init__(self, hidden_channels, num_att_layers, num_sensors):
        super(GNN_att, self).__init__()
        self.heads = 8
        self.layers = num_att_layers
        self.hid_dim = hidden_channels
        self.attention_layer = LinearFormerLayer(self.hid_dim, self.heads, self.hid_dim * 4)
        self.attention_norm = nn.LayerNorm(self.hid_dim)
        self.attention = LinearFormer(self.attention_layer, self.layers, self.attention_norm)
        self.lpos = LearnedPositionalEncoding(self.hid_dim, max_len=num_sensors)

    def forward(self, input, mask):
        x = input.permute(1, 0, 2)
        x = self.lpos(x)
        output = self.attention(x, mask)
        output = output.permute(1, 0, 2)
        return output


# ============================================================================
# Informer Components (from CandidateOpration.py)
# ============================================================================
class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=3, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = torch.softmax(scores, dim=-1)
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class InformerLayer(nn.Module):
    def __init__(self, d_model=32, d_ff=32, dropout=0., n_heads=4, activation="relu", output_attention=False):
        super(InformerLayer, self).__init__()
        self.attention = AttentionLayer(
            ProbAttention(False, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.d_model = d_model

    def forward(self, x, attn_mask=None, **kwargs):
        b, C, N, T = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, T, C)
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x + y)
        output = output.reshape(b, -1, T, C)
        output = output.permute(0, 3, 1, 2)
        return output


# ============================================================================
# Mixed Operations (from MixedOpration.py)
# ============================================================================
def create_op(op_name, node_embedding_1, node_embedding_2, adj_mx, hidden_channels, num_hop, num_att_layers,
              num_sensors, device):
    """Create operation by name."""
    if op_name == 'Zero':
        return Zero()
    elif op_name == 'Identity':
        return Identity()
    elif op_name == 'Informer':
        return InformerLayer(d_model=hidden_channels, d_ff=hidden_channels)
    elif op_name == 'DCC_2':
        return DCCLayer(hidden_channels, dilation=2)
    elif op_name == 'GNN_fixed':
        return GNN_fixed(hidden_channels, num_hop, adj_mx, device)
    elif op_name == 'GNN_adap':
        return GNN_adap(hidden_channels, num_hop, node_embedding_1, node_embedding_2)
    elif op_name == 'GNN_att':
        return GNN_att(hidden_channels, num_att_layers, num_sensors)
    else:
        raise Exception(f'Unknown operation name: {op_name}')


class TemporalLayerMixedOp(nn.Module):
    def __init__(self, node_embedding_1, node_embedding_2, adj_mx, hidden_channels, num_hop,
                 num_att_layers, num_sensors, device, used_operations):
        super(TemporalLayerMixedOp, self).__init__()
        self._num_ops = len(used_operations)
        self._candidate_ops = nn.ModuleList()
        for op_name in used_operations:
            self._candidate_ops += [create_op(op_name, node_embedding_1, node_embedding_2, adj_mx,
                                              hidden_channels, num_hop, num_att_layers, num_sensors, device)]
        self._candidate_alphas = nn.Parameter(torch.zeros(self._num_ops), requires_grad=True)
        self.set_mode(Mode.NONE)

    def set_mode(self, mode):
        self._mode = mode
        if mode == Mode.NONE:
            self._sample_idx = None
        elif mode == Mode.ONE_PATH_FIXED:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            op = torch.argmax(probs).item()
            self._sample_idx = np.array([op], dtype=np.int32)
        elif mode == Mode.ONE_PATH_RANDOM:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            self._sample_idx = torch.multinomial(probs, 1, replacement=True).cpu().numpy()
        elif mode == Mode.TWO_PATHS:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            self._sample_idx = torch.multinomial(probs, 2, replacement=True).cpu().numpy()
        elif mode == Mode.ALL_PATHS:
            self._sample_idx = np.arange(self._num_ops)

    def forward(self, inputs, mask):
        inputs = inputs[0]
        a = self._candidate_alphas[self._sample_idx]
        probs = F.softmax(a, dim=0)
        output = 0
        for i, idx in enumerate(self._sample_idx):
            output += probs[i] * self._candidate_ops[idx](inputs, mask=mask)
        return output

    def arch_parameters(self):
        yield self._candidate_alphas

    def weight_parameters(self):
        for i in range(self._num_ops):
            for p in self._candidate_ops[i].parameters():
                yield p


class SpatialLayerMixedOp(nn.Module):
    def __init__(self, node_embedding_1, node_embedding_2, adj_mx, hidden_channels, num_hop,
                 num_att_layers, num_sensors, device, used_operations):
        super(SpatialLayerMixedOp, self).__init__()
        self._num_ops = len(used_operations)
        self._candidate_ops = nn.ModuleList()
        for op_name in used_operations:
            self._candidate_ops += [create_op(op_name, node_embedding_1, node_embedding_2, adj_mx,
                                              hidden_channels, num_hop, num_att_layers, num_sensors, device)]

    def forward(self, inputs, candidate_alphas, mask):
        inputs = inputs[0]
        probs = F.softmax(candidate_alphas, dim=0)
        sample_idx = torch.multinomial(probs, 2, replacement=True).cpu().numpy()
        a = candidate_alphas[sample_idx]
        p = F.softmax(a, dim=0)
        output = 0
        for i, idx in enumerate(sample_idx):
            output += p[i] * self._candidate_ops[idx](inputs, mask=mask)
        return output

    def weight_parameters(self):
        for i in range(self._num_ops):
            for p in self._candidate_ops[i].parameters():
                yield p


# ============================================================================
# Search Layers (from STLayers.py)
# ============================================================================
class TemporalSearchLayer(nn.Module):
    def __init__(self, node_embedding_1, node_embedding_2, adj_mx, hidden_channels, num_hop,
                 num_att_layers, num_sensors, num_temporal_search_node, temporal_operations, device):
        super(TemporalSearchLayer, self).__init__()
        self._mixed_ops = nn.ModuleList()
        num_search_node = num_temporal_search_node
        self._num_mixed_ops = self.get_num_mixed_ops(num_search_node)
        for i in range(self._num_mixed_ops):
            self._mixed_ops += [
                TemporalLayerMixedOp(node_embedding_1, node_embedding_2, adj_mx, hidden_channels, num_hop,
                                     num_att_layers, num_sensors, device, temporal_operations)]
        self.set_mode(Mode.NONE)

    def forward(self, x, mask):
        node_idx = 0
        current_output = 0
        node_outputs = [x]
        for i in range(self._num_mixed_ops):
            a = self._mixed_ops[i]
            b = [node_outputs[node_idx]]
            c = a(b, mask)
            current_output += c
            if node_idx + 1 >= len(node_outputs):
                node_outputs += [current_output]
                current_output = 0
                node_idx = 0
            else:
                node_idx += 1
        if node_idx != 0:
            node_outputs += [current_output]
        ret = 0
        for x in node_outputs[:]:
            ret = ret + x
        return ret

    def set_mode(self, mode):
        self._mode = mode
        for op in self._mixed_ops:
            op.set_mode(mode)

    def arch_parameters(self):
        for i in range(self._num_mixed_ops):
            for p in self._mixed_ops[i].arch_parameters():
                yield p

    def weight_parameters(self):
        for i in range(self._num_mixed_ops):
            for p in self._mixed_ops[i].weight_parameters():
                yield p

    def get_num_mixed_ops(self, num):
        i = 1
        s = 0
        while i < num:
            s += i
            i += 1
        return s


class SpatialSearchLayer(nn.Module):
    def __init__(self, node_embedding_1, node_embedding_2, adj_mx, hidden_channels, num_hop,
                 num_att_layers, num_sensors, num_temporal_search_node, spatial_operations, scale_list, device):
        super(SpatialSearchLayer, self).__init__()
        self.scale_list = scale_list
        num_search_node = num_temporal_search_node
        self._num_mixed_ops = self.get_num_mixed_ops(num_search_node)
        self._mixed_ops = nn.ModuleList()
        for i in range(self._num_mixed_ops):
            self._mixed_ops += [
                SpatialLayerMixedOp(node_embedding_1, node_embedding_2, adj_mx, hidden_channels, num_hop,
                                    num_att_layers, num_sensors, device, spatial_operations)]
        self.spatial_dag = []
        for dag_num in range(len(self.scale_list)):
            dag_ops = []
            for i in range(self._num_mixed_ops):
                dag_ops += [nn.Parameter(torch.zeros(len(spatial_operations)), requires_grad=True)]
            self.spatial_dag.append(dag_ops)

    def forward(self, x, dag_i, mask):
        node_idx = 0
        current_output = 0
        node_outputs = [x]
        for i in range(self._num_mixed_ops):
            a = self._mixed_ops[i]
            b = [node_outputs[node_idx]]
            c = a(b, self.spatial_dag[dag_i][i], mask)
            current_output += c
            if node_idx + 1 >= len(node_outputs):
                node_outputs += [current_output]
                current_output = 0
                node_idx = 0
            else:
                node_idx += 1
        if node_idx != 0:
            node_outputs += [current_output]
        ret = 0
        for x in node_outputs[:]:
            ret = ret + x
        return ret

    def set_mode(self, mode):
        self._mode = mode

    def arch_parameters(self):
        for dag in self.spatial_dag:
            for param in dag:
                yield param

    def weight_parameters(self):
        for i in range(self._num_mixed_ops):
            for p in self._mixed_ops[i].weight_parameters():
                yield p

    def get_num_mixed_ops(self, num):
        i = 1
        s = 0
        while i < num:
            s += i
            i += 1
        return s


# ============================================================================
# Main AutoSTF Model (from TrafficForecasting.py)
# ============================================================================
def create_layer(name, node_embedding_1, node_embedding_2, adj_mx, hidden_channels, num_hop, num_att_layers,
                 num_sensors, num_temporal_search_node, temporal_operations, spatial_operations, scale_list,
                 in_length, device):
    """Create search layer by name."""
    if name == 'TemporalSearch':
        return TemporalSearchLayer(node_embedding_1, node_embedding_2, adj_mx, hidden_channels, num_hop,
                                   num_att_layers, num_sensors, num_temporal_search_node, temporal_operations, device)
    if name == 'SpatialSearch':
        return SpatialSearchLayer(node_embedding_1, node_embedding_2, adj_mx, hidden_channels, num_hop,
                                  num_att_layers, num_sensors, num_temporal_search_node, spatial_operations,
                                  scale_list, device)
    if name == 'ConvPooling':
        return nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                         kernel_size=(1, 3), padding=(0, 1), stride=(1, 2))
    if name == 'AvgPooling':
        return nn.AvgPool2d(kernel_size=(1, 3), padding=(0, 1), stride=(1, 2))
    raise Exception(f'Unknown layer name: {name}')


class AutoSTFCore(nn.Module):
    """Core AutoSTF model architecture."""

    def __init__(self, in_length, out_length, mask_support_adj, adj_mx, num_sensors,
                 in_channels, out_channels, hidden_channels, end_channels, layer_names,
                 scale_list, num_mlp_layers, num_linear_layers, num_hop, num_att_layers,
                 num_temporal_search_node, temporal_operations, spatial_operations,
                 IsUseLinear, device):
        super(AutoSTFCore, self).__init__()

        self.scale_list = scale_list
        self.scale_step = int(in_length / len(scale_list))
        self.in_length = in_length
        self.out_length = out_length
        self.num_sensors = num_sensors
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.end_channels = end_channels
        self.mask_support_adj = mask_support_adj
        self.layer_names = layer_names
        self.IsUseLinear = IsUseLinear
        self.device = device

        # MLP node and temporal dimensions
        self.mlp_node_dim = 32
        self.temp_dim_tid = 32
        self.temp_dim_diw = 32
        self.useDayTime = True
        self.useWeekTime = True

        if self.in_channels == 1:
            self.useDayTime = False
            self.useWeekTime = False
            self.temp_dim_tid = 0
            self.temp_dim_diw = 0
        elif self.in_channels == 2:
            self.useWeekTime = False
            self.temp_dim_diw = 0

        self.num_mlp_layer = num_mlp_layers
        self.time_of_day_size = 288
        self.day_of_week_size = 7

        # Node embeddings
        self.node_emb = nn.Parameter(torch.empty(self.num_sensors, self.mlp_node_dim))
        nn.init.xavier_uniform_(self.node_emb)

        # Temporal embeddings
        if self.useDayTime:
            self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.useWeekTime:
            self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # Feature embeddings
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.in_channels * self.in_length,
            out_channels=self.hidden_channels, kernel_size=(1, 1), bias=True)

        # Fusion layer
        self.hidden_dim = self.hidden_channels + self.mlp_node_dim + self.temp_dim_tid + self.temp_dim_diw
        self.start_encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_mlp_layer)])
        self.fusion_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.hidden_channels, kernel_size=(1, 1), bias=True)

        # Light linear layer
        if self.IsUseLinear:
            self.light_linear_layer = LightLinear(hidden_channels, num_linear_layers)
            self.start_scale_light = nn.Conv2d(in_channels=self.scale_step, out_channels=1,
                                               kernel_size=(1, 1), bias=True)
            self.start_linear_light = nn.Conv2d(in_channels=self.hidden_channels * 2,
                                                out_channels=self.hidden_channels,
                                                kernel_size=(1, 1), bias=True)

        # Spatial Layer mask
        mask0 = mask_support_adj[0].detach()
        mask1 = mask_support_adj[1].detach()
        mask = mask0 + mask1
        self.mask = mask == 0

        # Node vectors for adaptive graph
        self.node_vec1 = nn.Parameter(torch.randn(self.num_sensors, 10).to(device), requires_grad=True).to(device)
        self.node_vec2 = nn.Parameter(torch.randn(10, self.num_sensors).to(device), requires_grad=True).to(device)

        # Search layers
        self.TemporalSearchLayers = nn.ModuleList()
        self.SpatialSearchLayers = nn.ModuleList()
        for name in layer_names:
            if name == 'TemporalSearch':
                self.TemporalSearchLayers += [create_layer(
                    name, node_embedding_1=self.node_vec1, node_embedding_2=self.node_vec2,
                    adj_mx=adj_mx, hidden_channels=hidden_channels, num_hop=num_hop,
                    num_att_layers=num_att_layers, num_sensors=num_sensors,
                    num_temporal_search_node=num_temporal_search_node,
                    temporal_operations=temporal_operations, spatial_operations=spatial_operations,
                    scale_list=scale_list, in_length=in_length, device=device)]
                self.start_temporal = nn.Conv2d(in_channels=1, out_channels=self.hidden_channels,
                                                kernel_size=(1, 1), bias=True)
            elif name == 'SpatialSearch':
                self.SpatialSearchLayers += [create_layer(
                    name, node_embedding_1=self.node_vec1, node_embedding_2=self.node_vec2,
                    adj_mx=adj_mx, hidden_channels=hidden_channels, num_hop=num_hop,
                    num_att_layers=num_att_layers, num_sensors=num_sensors,
                    num_temporal_search_node=num_temporal_search_node,
                    temporal_operations=temporal_operations, spatial_operations=spatial_operations,
                    scale_list=scale_list, in_length=in_length, device=device)]

        self.spatial_fusion = nn.Linear(self.hidden_channels * len(self.scale_list), self.hidden_channels)

        # End layers
        self.end_conv1 = nn.Linear(self.hidden_channels * 2, self.end_channels)
        self.end_conv2 = nn.Linear(self.end_channels, self.out_length * self.out_channels)

    def forward(self, inputs, mode):
        batch_size, num_features, num_nodes, num_timestep = inputs.shape
        self.set_mode(mode)

        # Input Layer
        history_data = inputs.transpose(1, 3)

        # Temporal embedding
        input_data = history_data[..., range(self.in_channels)]
        if self.useDayTime:
            t_i_d_data = history_data[..., 1]
            day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]
        if self.useWeekTime:
            d_i_w_data = history_data[..., 2]
            week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]

        # Feature embedding
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        # Concatenate embeddings
        embeddings_list = [time_series_emb]
        embeddings_list += [self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)]
        if self.useDayTime:
            embeddings_list += [day_emb.transpose(1, 2).unsqueeze(-1)]
        if self.useWeekTime:
            embeddings_list += [week_emb.transpose(1, 2).unsqueeze(-1)]

        hidden = torch.cat(embeddings_list, dim=1)

        # Encoding
        x = self.start_encoder(hidden)

        # Fusion
        x = self.fusion_layer(x).squeeze(dim=-1).transpose(1, 2)
        mlp_residual = x

        # Temporal Search
        if 'TemporalSearch' in self.layer_names:
            x_t = inputs[:, 0:1, :, :]
            x_t = self.start_temporal(x_t)
            temporal_residual = 0
            for TLayer in self.TemporalSearchLayers:
                x_t = TLayer(x_t, self.mask)
                temporal_residual += x_t
            x_t = temporal_residual.transpose(1, 3)

        # Multi-scale spatial processing
        start_scale = 0
        spatial_embedding = []
        for i in range(len(self.scale_list)):
            x_scale = x_t[:, start_scale:start_scale + self.scale_step, :, :]
            start_scale = start_scale + self.scale_step
            x_scale = self.start_scale_light(x_scale).squeeze(dim=1)
            x = torch.cat([mlp_residual] + [x_scale], dim=-1).unsqueeze(dim=-1).transpose(1, 2)
            x = self.start_linear_light(x).squeeze(dim=-1).transpose(1, 2)
            x = self.light_linear_layer(x)
            x = self.SpatialSearchLayers[0](x, i, self.mask)
            spatial_embedding.append(x)
        x = torch.cat(spatial_embedding, dim=-1)
        x = self.spatial_fusion(x)
        spatial_residual = x

        # Final outputs
        outputs = [mlp_residual, spatial_residual]
        output = torch.cat(outputs, dim=-1)

        # End conv
        x = self.end_conv1(output)
        x = F.relu(x)
        x = self.end_conv2(x)
        x = x.unsqueeze(dim=1)

        self.set_mode(Mode.NONE)
        return x

    def set_mode(self, mode):
        self._mode = mode
        if 'TemporalSearch' in self.layer_names:
            for l in self.TemporalSearchLayers:
                l.set_mode(mode)
        for l in self.SpatialSearchLayers:
            l.set_mode(mode)

    def asym_adj(self, adj):
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        return d_mat.dot(adj).astype(np.float32).todense()


# ============================================================================
# LibCity Adapter for AutoSTF
# ============================================================================
class AutoSTF(AbstractTrafficStateModel):
    """
    AutoSTF model adapted for LibCity framework.

    This model performs Neural Architecture Search (NAS) for spatio-temporal forecasting.
    In inference mode, we use ONE_PATH_FIXED to bypass NAS search and use the best architecture.

    Reference:
        AutoSTF: Automated Spatio-Temporal Graph Forecasting (WWW 2023)

    Args:
        config: LibCity configuration object
        data_feature: Dictionary containing data features like num_nodes, adj_mx, scaler, etc.
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()

        # Data features
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self._scaler = self.data_feature.get('scaler')

        # Model hyperparameters from config
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.hidden_channels = config.get('hidden_channels', 32)
        self.end_channels = config.get('end_channels', 512)
        self.num_mlp_layers = config.get('num_mlp_layers', 2)
        self.num_linear_layers = config.get('num_linear_layers', 2)
        self.num_hop = config.get('num_hop', 2)
        self.num_att_layers = config.get('num_att_layers', 2)
        self.num_temporal_search_node = config.get('num_temporal_search_node', 3)

        # NAS operations
        self.temporal_operations = config.get('temporal_operations', ['Zero', 'Identity', 'Informer', 'DCC_2'])
        self.spatial_operations = config.get('spatial_operations', ['Zero', 'Identity', 'GNN_fixed', 'GNN_adap', 'GNN_att'])

        # Layer configuration
        self.layer_names = config.get('layer_names', ['TemporalSearch', 'SpatialSearch'])
        self.scale_list = config.get('scale_list', [1, 2, 3, 4])
        self.IsUseLinear = config.get('IsUseLinear', True)

        # Mode for NAS (use ONE_PATH_FIXED for inference)
        self.mode_name = config.get('mode', 'ONE_PATH_FIXED')
        self.mode = get_mode(self.mode_name)

        self.device = config.get('device', torch.device('cpu'))

        # Get adjacency matrix
        adj_mx = self.data_feature.get('adj_mx')
        if adj_mx is None:
            self._logger.warning('No adjacency matrix provided, using identity matrix')
            adj_mx = np.eye(self.num_nodes)

        # Process adjacency matrix
        adj_mx_processed = self._process_adj_mx(adj_mx)
        mask_support_adj = self._get_mask_support_adj(adj_mx_processed)

        # Initialize the core model
        self.model = AutoSTFCore(
            in_length=self.input_window,
            out_length=self.output_window,
            mask_support_adj=mask_support_adj,
            adj_mx=adj_mx_processed,
            num_sensors=self.num_nodes,
            in_channels=self.feature_dim,
            out_channels=self.output_dim,
            hidden_channels=self.hidden_channels,
            end_channels=self.end_channels,
            layer_names=self.layer_names,
            scale_list=self.scale_list,
            num_mlp_layers=self.num_mlp_layers,
            num_linear_layers=self.num_linear_layers,
            num_hop=self.num_hop,
            num_att_layers=self.num_att_layers,
            num_temporal_search_node=self.num_temporal_search_node,
            temporal_operations=self.temporal_operations,
            spatial_operations=self.spatial_operations,
            IsUseLinear=self.IsUseLinear,
            device=self.device
        )

        self._logger.info(f'AutoSTF model initialized with {self.num_nodes} nodes, '
                          f'input_window={self.input_window}, output_window={self.output_window}')

    def _process_adj_mx(self, adj_mx):
        """Process adjacency matrix for the model."""
        if isinstance(adj_mx, np.ndarray):
            adj_mx = adj_mx.astype(np.float32)
        else:
            adj_mx = np.array(adj_mx, dtype=np.float32)

        # Normalize adjacency matrix
        adj_mx_normalized = self._asym_adj(adj_mx)
        adj_mx_normalized_t = self._asym_adj(adj_mx.T)

        return [adj_mx_normalized, adj_mx_normalized_t]

    def _asym_adj(self, adj):
        """Compute asymmetric normalized adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        return d_mat.dot(adj).astype(np.float32).todense()

    def _get_mask_support_adj(self, adj_mx_list):
        """Get mask support adjacency matrices."""
        mask_support_adj = []
        for adj in adj_mx_list:
            adj_tensor = torch.tensor(adj, dtype=torch.float32).to(self.device)
            mask_support_adj.append(adj_tensor)
        return mask_support_adj

    def forward(self, batch):
        """
        Forward pass of the model.

        Args:
            batch: Dictionary containing 'X' with shape [B, T, N, C]

        Returns:
            Predictions with shape [B, T_out, N, C_out]
        """
        x = batch['X']  # [B, T, N, C]

        # Convert from LibCity format [B, T, N, C] to AutoSTF format [B, C, N, T]
        x = x.permute(0, 3, 2, 1)  # [B, C, N, T]

        # Forward through core model
        output = self.model(x, self.mode)  # [B, 1, N, T_out * C_out]

        # Reshape output
        batch_size = output.shape[0]
        output = output.squeeze(1)  # [B, N, T_out * C_out]
        output = output.view(batch_size, self.num_nodes, self.output_window, self.output_dim)  # [B, N, T_out, C_out]

        # Convert back to LibCity format [B, T_out, N, C_out]
        output = output.permute(0, 2, 1, 3)  # [B, T_out, N, C_out]

        return output

    def predict(self, batch):
        """
        Predict method for evaluation.

        Args:
            batch: Dictionary containing 'X' and 'y'

        Returns:
            Predictions with shape [B, T_out, N, C_out]
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate training loss using masked MAE.

        Args:
            batch: Dictionary containing 'X' and 'y'

        Returns:
            Loss tensor
        """
        y_true = batch['y']  # [B, T_out, N, C]
        y_predicted = self.predict(batch)  # [B, T_out, N, C_out]

        # Inverse transform for proper loss calculation
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        # Use masked MAE loss
        return loss.masked_mae_torch(y_predicted, y_true)
