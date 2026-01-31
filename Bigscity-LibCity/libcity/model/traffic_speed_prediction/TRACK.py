"""
TRACK (DRNTRL) Model Adaptation for LibCity

Original Model: TRACK (Dynamic Road Network and Trajectory Representation Learning)
Source Repository: ./repos/TRACK
Original Files:
    - Main Model: ./repos/TRACK/libcity/model/rn_emb/DRNTRL.py
    - Traffic State Prediction: ./repos/TRACK/libcity/model/traffic_speed_prediction/DRNTRLTSP.py
    - Traffic Encoder: ./repos/TRACK/libcity/model/traf_emb/DRNRL.py

Key Adaptations:
    - Inherits from AbstractTrafficStateModel for LibCity compatibility
    - Implements forward(), predict(), and calculate_loss() methods
    - Handles LibCity batch dict format {'X': tensor, 'y': tensor, ...}
    - All components are self-contained in this file

Model Architecture:
    - DataEmbedding: Encodes traffic features with spatial and temporal embeddings
    - DRNRL: Dynamic Road Network Representation Learning encoder blocks
    - GAT/DGAT: Graph Attention Network for spatial feature extraction
    - CoTransformerBlock: Co-attention mechanism between traffic and trajectory embeddings
    - Time-aware embeddings for temporal patterns

Config Parameters:
    - embed_dim: Embedding dimension (default: 64)
    - d_model: Model dimension (default: 64)
    - enc_depth: Number of encoder blocks (default: 6)
    - s_num_heads: Spatial attention heads (default: 2)
    - t_num_heads: Temporal attention heads (default: 6)
    - gat_heads_per_layer: GAT heads per layer (default: [8, 16, 1])
    - gat_features_per_layer: GAT features per layer (default: [16, 16, 64])
    - coattn_heads: Co-attention heads (default: 8)
    - co_n_layers: Number of co-attention layers (default: 1)
    - input_window: Input time steps (default: 12)
    - output_window: Output time steps (default: 12)
    - add_time_in_day: Add time-of-day embedding (default: True)
    - add_day_in_week: Add day-of-week embedding (default: True)

Limitations:
    - Requires adjacency matrix information in data_feature
    - Simplified version without trajectory co-attention (standalone traffic prediction)
    - For full TRACK with trajectory data, additional data preprocessing is needed
"""

import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
import numpy as np

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


# ==================== Utility Classes ====================

class Mlp(nn.Module):
    """MLP block with GELU activation and dropout."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path_func(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """DropPath module for stochastic depth."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_func(x, self.drop_prob, self.training)


# ==================== Embedding Classes ====================

class TokenEmbedding(nn.Module):
    """Token embedding layer."""

    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequence."""

    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()


# ==================== GAT Layer Classes ====================

class GATLayer(nn.Module):
    """Base GAT layer."""

    head_dim = 1

    def __init__(
        self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
        dropout_prob=0.1, add_skip_connection=True, bias=True, load_trans_prob=False,
    ):
        super().__init__()
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection
        self.load_trans_prob = load_trans_prob

        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        if self.load_trans_prob:
            self.linear_proj_tran_prob = nn.Linear(1, num_of_heads * num_out_features, bias=False)
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        if self.load_trans_prob:
            self.scoring_trans_prob = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)
        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        if self.load_trans_prob:
            nn.init.xavier_uniform_(self.linear_proj_tran_prob.weight)
            nn.init.xavier_uniform_(self.scoring_trans_prob)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def skip_concat_bias(self, in_nodes_features, out_nodes_features):
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class SparseGATLayer(GATLayer):
    """Sparse GAT layer for efficient computation."""

    src_nodes_dim = 0
    trg_nodes_dim = 1
    nodes_dim = 0
    head_dim = 1

    def __init__(
        self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
        dropout_prob=0.1, add_skip_connection=True, bias=True, load_trans_prob=False,
    ):
        super().__init__(
            num_in_features, num_out_features, num_of_heads, concat, activation,
            dropout_prob, add_skip_connection, bias, load_trans_prob,
        )

    def forward(self, data):
        in_nodes_features, edge_index, edge_prob = data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        in_nodes_features = self.dropout(in_nodes_features)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)
        if self.load_trans_prob:
            trans_prob_proj = self.linear_proj_tran_prob(edge_prob).view(-1, self.num_of_heads, self.num_out_features)
            trans_prob_proj = self.dropout(trans_prob_proj)

        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        if self.load_trans_prob:
            scores_trans_prob = (trans_prob_proj * self.scoring_trans_prob).sum(dim=-1)
            scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted + scores_trans_prob)
        else:
            scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        attentions_per_edge = self.dropout(attentions_per_edge)
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)
        out_nodes_features = self.skip_concat_bias(in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index, edge_prob)

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)
        size = list(exp_scores_per_edge.shape)
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)
        size[self.nodes_dim] = num_of_nodes
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)
        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)
        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)
        return this.expand_as(other)


class GAT(nn.Module):
    """Graph Attention Network."""

    def __init__(
        self, d_model, in_feature, num_heads_per_layer, num_features_per_layer,
        add_skip_connection=True, bias=True, dropout=0.1, load_trans_prob=False, avg_last=True,
    ):
        super().__init__()
        assert len(num_heads_per_layer) == len(num_features_per_layer), f'Enter valid arch params.'

        num_features_per_layer = [in_feature] + num_features_per_layer
        num_heads_per_layer = [1] + num_heads_per_layer
        if avg_last:
            assert num_features_per_layer[-1] == d_model
        else:
            assert num_features_per_layer[-1] * num_heads_per_layer[-1] == d_model
        num_of_layers = len(num_heads_per_layer) - 1

        gat_layers = []
        for i in range(num_of_layers):
            concat_input = True
            if i == num_of_layers - 1 and avg_last:
                concat_input = False
            layer = SparseGATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],
                num_out_features=num_features_per_layer[i+1], num_of_heads=num_heads_per_layer[i+1],
                concat=concat_input,
                activation=nn.ELU() if i < num_of_layers - 1 else None,
                dropout_prob=dropout, add_skip_connection=add_skip_connection, bias=bias, load_trans_prob=load_trans_prob,
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(*gat_layers)

    def forward(self, node_features, edge_index_input, edge_prob_input):
        """
        Args:
            node_features: (N, fea_dim)
            edge_index_input: (2, E)
            edge_prob_input: (E, 1)
        Returns:
            (N, d_model)
        """
        data = (node_features, edge_index_input, edge_prob_input)
        (node_fea_emb, edge_index, edge_prob) = self.gat_net(data)
        return node_fea_emb


# ==================== Traffic GAT Layer Classes ====================

class TrafGATLayer(nn.Module):
    """Traffic GAT layer with 4D input support."""

    head_dim = 3

    def __init__(
        self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
        dropout_prob=0.3, add_skip_connection=True, bias=False, load_trans_prob=False,
    ):
        super().__init__()
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection
        self.load_trans_prob = False

        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, 1, 1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, 1, 1, num_of_heads, num_out_features))
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)
        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def skip_concat_bias(self, in_nodes_features, out_nodes_features):
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()
        B, T, N, _, _ = out_nodes_features.shape

        if self.add_skip_connection:
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:
                out_nodes_features += in_nodes_features.unsqueeze(self.head_dim)
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(B, T, N, self.num_of_heads, self.num_out_features)

        if self.concat:
            out_nodes_features = out_nodes_features.view(B, T, N, self.num_of_heads * self.num_out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class TrafSparseGATLayer(TrafGATLayer):
    """Traffic sparse GAT layer."""

    src_nodes_dim = 0
    trg_nodes_dim = 1
    nodes_dim = 2
    head_dim = 3

    def __init__(
        self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
        dropout_prob=0.3, add_skip_connection=True, bias=False, load_trans_prob=False,
    ):
        super().__init__(
            num_in_features, num_out_features, num_of_heads, concat, activation,
            dropout_prob, add_skip_connection, bias, load_trans_prob,
        )

    def forward(self, in_nodes_features, edge_index, edge_prob=None):
        B, T, N, _ = in_nodes_features.shape
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        in_nodes_features = self.dropout(in_nodes_features)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(B, T, N, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)

        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], N)
        attentions_per_edge = self.dropout(attentions_per_edge)
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, N)
        out_nodes_features = self.skip_concat_bias(in_nodes_features, out_nodes_features)
        return out_nodes_features

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        trg_index_broadcasted = trg_index.unsqueeze(-1).unsqueeze(0).unsqueeze(0).expand_as(exp_scores_per_edge)
        size = list(exp_scores_per_edge.shape)
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)
        size[self.nodes_dim] = num_of_nodes
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        trg_index_broadcasted = edge_index[self.trg_nodes_dim].unsqueeze(-1).unsqueeze(-1).unsqueeze(0).unsqueeze(0).expand_as(nodes_features_proj_lifted_weighted)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)
        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)
        return scores_source, scores_target, nodes_features_matrix_proj_lifted


# ==================== DRN Attention and Encoder Classes ====================

class DRNAttention(nn.Module):
    """Dynamic Road Network Attention combining spatial and temporal attention."""

    def __init__(
        self, dim, s_num_heads=2, t_num_heads=6, qkv_bias=False,
        attn_drop=0., proj_drop=0., device=torch.device('cpu'), output_dim=1,
    ):
        super().__init__()
        assert dim % (s_num_heads + t_num_heads) == 0
        self.s_num_heads = s_num_heads
        self.t_num_heads = t_num_heads
        self.head_dim = dim // (s_num_heads + t_num_heads)
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.s_ratio = s_num_heads / (s_num_heads + t_num_heads)
        self.t_ratio = 1 - self.s_ratio
        self.output_dim = output_dim

        if self.t_ratio != 0:
            self.t_q_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
            self.t_k_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
            self.t_v_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
            self.t_attn_drop = nn.Dropout(attn_drop)

        if self.s_ratio != 0:
            self.gat = TrafSparseGATLayer(dim, self.head_dim, s_num_heads, dropout_prob=attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, edge_index, edge_prob=None):
        B, T, N, D = x.shape

        if self.t_ratio != 0:
            t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
            t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
            t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
            t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
            t_attn = t_attn.softmax(dim=-1)
            t_attn = self.t_attn_drop(t_attn)
            t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, int(D * self.t_ratio)).transpose(1, 2)

        if self.s_ratio != 0:
            s_x = self.gat(x, edge_index, edge_prob)

        if self.t_ratio == 0:
            x = s_x
        elif self.s_ratio == 0:
            x = t_x
        else:
            x = torch.cat([t_x, s_x], dim=-1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DRNEncoderBlock(nn.Module):
    """Dynamic Road Network Encoder Block."""

    def __init__(
        self, dim, s_num_heads=2, t_num_heads=6, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, device=torch.device('cpu'), type_ln="post", output_dim=1,
    ):
        super().__init__()
        self.type_ln = type_ln
        self.norm1 = norm_layer(dim)
        self.drn_attn = DRNAttention(
            dim, s_num_heads=s_num_heads, t_num_heads=t_num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, device=device, output_dim=output_dim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, edge_index, edge_prob=None):
        if self.type_ln == 'pre':
            x = x + self.drop_path(self.drn_attn(self.norm1(x), edge_index, edge_prob))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            x = self.norm1(x + self.drop_path(self.drn_attn(x, edge_index, edge_prob)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


# ==================== Data Embedding ====================

class DataEmbedding(nn.Module):
    """Data embedding layer combining value, position, time, and spatial embeddings."""

    def __init__(
        self, feature_dim, embed_dim, num_nodes, node_fea_dim,
        gat_heads_per_layer=[8, 8, 1], gat_features_per_layer=[8, 8, 64], gat_dropout=0.1, drop=0.,
        add_time_in_day=False, add_day_in_week=False, time_intervals=1800,
    ):
        super().__init__()
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        self.feature_dim = feature_dim

        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim)
        if self.add_time_in_day:
            self.minute_size = int(24 * 3600 // time_intervals)
            self.daytime_embedding = nn.Embedding(self.minute_size + 1, embed_dim)
        if self.add_day_in_week:
            self.weekday_embedding = nn.Embedding(8, embed_dim)
        self.ada_spatial_embedding = nn.Parameter(torch.Tensor(num_nodes, embed_dim))
        if node_fea_dim > 0:
            self.sta_spatial_embedding = nn.Linear(node_fea_dim, embed_dim)
            self.spatial_embedding = GAT(
                d_model=embed_dim, in_feature=embed_dim, num_heads_per_layer=gat_heads_per_layer, num_features_per_layer=gat_features_per_layer,
                add_skip_connection=True, bias=True, dropout=gat_dropout, load_trans_prob=False, avg_last=True,
            )
            self.use_node_features = True
        else:
            self.use_node_features = False
        self.dropout = nn.Dropout(drop)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.ada_spatial_embedding)

    def forward(self, x, node_features=None, edge_index=None, edge_prob=None):
        origin_x = x
        x = self.value_embedding(origin_x[:, :, :, :self.feature_dim])
        x = x + self.position_encoding(x)
        if self.add_time_in_day and origin_x.shape[-1] > self.feature_dim:
            time_idx = (origin_x[:, :, :, self.feature_dim] * self.minute_size).round().long()
            time_idx = time_idx.clamp(0, self.minute_size)
            x = x + self.daytime_embedding(time_idx)
        if self.add_day_in_week and origin_x.shape[-1] > self.feature_dim + 1:
            day_idx = origin_x[:, :, :, self.feature_dim + 1].round().long()
            day_idx = day_idx.clamp(0, 7)
            x = x + self.weekday_embedding(day_idx)
        if self.use_node_features and node_features is not None and edge_index is not None:
            spatial_emb = self.spatial_embedding(
                node_features=self.sta_spatial_embedding(node_features) + self.ada_spatial_embedding,
                edge_index_input=edge_index, edge_prob_input=edge_prob,
            )
            x = x + spatial_emb
        else:
            x = x + self.ada_spatial_embedding
        x = self.dropout(x)
        return x


# ==================== DRNRL Module ====================

class DRNRL(nn.Module):
    """Dynamic Road Network Representation Learning module."""

    def __init__(self, config, data_feature, edge_index, edge_prob):
        super().__init__()
        drop_path = config.get("drop_path", 0.3)
        enc_depth = config.get("enc_depth", 6)
        embed_dim = config.get('embed_dim', 64)
        s_num_heads = config.get('s_num_heads', 2)
        t_num_heads = config.get('t_num_heads', 6)
        mlp_ratio = config.get("mlp_ratio", 4)
        qkv_bias = config.get("qkv_bias", True)
        drop = config.get("traf_drop", 0.)
        attn_drop = config.get("traf_attn_drop", 0.)
        device = config.get('device', torch.device('cpu'))
        type_ln = config.get("type_ln", "post")
        output_dim = config.get('output_dim', 1)
        d_model = config.get("d_model", 64)

        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_prob', edge_prob)

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, enc_depth)]
        self.encoder_blocks = nn.ModuleList([
            DRNEncoderBlock(
                dim=embed_dim, s_num_heads=s_num_heads, t_num_heads=t_num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=enc_dpr[i], act_layer=nn.GELU,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), device=device, type_ln=type_ln, output_dim=output_dim,
            ) for i in range(enc_depth)
        ])
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=embed_dim, out_channels=d_model, kernel_size=1,
            ) for _ in range(enc_depth)
        ])

    def forward(self, X):
        skip = 0
        for i, encoder_block in enumerate(self.encoder_blocks):
            X = encoder_block(X, self.edge_index, self.edge_prob)
            skip = skip + self.skip_convs[i](X.permute(0, 3, 2, 1))
        return skip.permute(0, 3, 2, 1)


# ==================== TRACK Model (Main) ====================

class TRACK(AbstractTrafficStateModel):
    """
    TRACK (Dynamic Road Network and Trajectory Representation Learning) model for traffic state prediction.

    This is a simplified version adapted for LibCity that performs standalone traffic prediction
    without requiring trajectory co-attention (which would need additional trajectory data).

    The model combines:
    - Spatial embeddings using Graph Attention Networks
    - Temporal embeddings using positional encoding and time-of-day/day-of-week embeddings
    - Dynamic Road Network Representation Learning (DRNRL) encoder blocks
    - Skip connections for multi-scale feature aggregation
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()

        # Extract data features
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.ext_dim = data_feature.get('ext_dim', 0)
        self.output_dim = data_feature.get('output_dim', 1)
        self._scaler = data_feature.get('scaler')

        # Model configuration
        self.device = config.get('device', torch.device('cpu'))
        self.embed_dim = config.get('embed_dim', 64)
        self.d_model = config.get('d_model', 64)
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.add_time_in_day = config.get('add_time_in_day', True)
        self.add_day_in_week = config.get('add_day_in_week', True)
        self.time_intervals = config.get('time_intervals', 300)  # 5 minutes default
        self.traf_drop = config.get('traf_drop', 0.)

        # GAT configuration
        self.gat_heads_per_layer = config.get('gat_heads_per_layer', [8, 16, 1])
        self.gat_features_per_layer = config.get('gat_features_per_layer', [16, 16, 64])
        self.gat_dropout = config.get('gat_dropout', 0.1)

        # Build adjacency information
        adj_mx = data_feature.get('adj_mx')
        if adj_mx is not None:
            edge_index, edge_prob = self._build_edge_index(adj_mx)
        else:
            # Create self-loop only graph if no adjacency matrix
            edge_index = torch.stack([
                torch.arange(self.num_nodes),
                torch.arange(self.num_nodes)
            ]).long()
            edge_prob = torch.ones(self.num_nodes, 1).float()
            self._logger.warning("No adjacency matrix found, using self-loop only graph")

        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_prob', edge_prob)

        # Node features
        node_features = data_feature.get('node_features')
        if node_features is not None:
            self.register_buffer('node_features', torch.FloatTensor(node_features))
            node_fea_dim = node_features.shape[-1]
        else:
            self.node_features = None
            node_fea_dim = 0

        # Build model components
        input_dim = self.feature_dim - self.ext_dim

        self.enc_embed_layer = DataEmbedding(
            input_dim, self.embed_dim, self.num_nodes, node_fea_dim, drop=self.traf_drop,
            gat_heads_per_layer=self.gat_heads_per_layer, gat_features_per_layer=self.gat_features_per_layer,
            gat_dropout=self.gat_dropout,
            add_time_in_day=self.add_time_in_day, add_day_in_week=self.add_day_in_week,
            time_intervals=self.time_intervals,
        )

        # DRNRL encoder
        self.drnrl = DRNRL(config, data_feature, edge_index, edge_prob)

        # Output convolutions
        self.traf_emb_conv = nn.Conv2d(
            in_channels=self.input_window, out_channels=1, kernel_size=1,
        )
        self.end_conv1 = nn.Conv2d(in_channels=1, out_channels=self.output_window, kernel_size=1)
        self.end_conv2 = nn.Linear(self.d_model, self.output_dim)

        self._logger.info(f"TRACK model initialized with {self.num_nodes} nodes, "
                         f"input_window={self.input_window}, output_window={self.output_window}")

    def _build_edge_index(self, adj_mx):
        """Build edge index and edge probability from adjacency matrix."""
        if isinstance(adj_mx, np.ndarray):
            adj_mx = torch.FloatTensor(adj_mx)

        # Get edge indices
        src, dst = torch.where(adj_mx > 0)
        edge_index = torch.stack([src, dst]).long()

        # Get edge probabilities (weights)
        edge_prob = adj_mx[src, dst].unsqueeze(-1).float()

        return edge_index, edge_prob

    def forward(self, batch):
        """
        Forward pass of the TRACK model.

        Args:
            batch: dict with 'X' key containing input tensor of shape (B, T, N, F)

        Returns:
            predictions: tensor of shape (B, output_window, N, output_dim)
        """
        x = batch['X']  # (B, T, N, F)

        # Get traffic embeddings
        if self.node_features is not None:
            traf_enc = self.enc_embed_layer(x, self.node_features, self.edge_index, self.edge_prob)
        else:
            traf_enc = self.enc_embed_layer(x)

        # Apply DRNRL encoder
        traf_emb = self.drnrl(traf_enc)  # (B, T, N, d_model)

        # Temporal aggregation
        traf_rn_emb = self.traf_emb_conv(traf_emb)  # (B, 1, N, d_model)

        # Output projection
        pred_traf = self.end_conv1(traf_rn_emb)  # (B, output_window, N, d_model)
        pred_traf = self.end_conv2(pred_traf)  # (B, output_window, N, output_dim)

        return pred_traf

    def predict(self, batch):
        """
        Predict method for LibCity compatibility.

        Args:
            batch: dict with 'X' key

        Returns:
            predictions: tensor of shape (B, output_window, N, output_dim)
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate loss for training.

        Args:
            batch: dict with 'X' and 'y' keys

        Returns:
            loss: scalar tensor
        """
        y_true = batch['y']
        y_predicted = self.predict(batch)

        # Inverse transform for loss calculation
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        return loss.masked_mae_torch(y_predicted, y_true, 0)
