"""
TGraphormer: Traffic Graphormer for Spatiotemporal Traffic Speed Prediction

Adapted from: https://github.com/QTMdeHUB/TGraphormer
Original paper: TGraphormer: A Graph Transformer for Traffic Forecasting

This adaptation ports the TGraphormer model to LibCity conventions:
- Inherits from AbstractTrafficStateModel
- Accepts config and data_feature parameters
- Implements predict() and calculate_loss() methods
- Handles LibCity batch format: batch['X'] shape [B, T_in, N, F]

Key changes made during adaptation:
1. Integrated all required modules (GraphormerGraphEncoder, GraphormerLayers, MultiheadAttention)
2. Added automatic graph preprocessing (Floyd-Warshall for spatial positions)
3. Adapted data format from PyG to LibCity batch dict format
4. Added config-based hyperparameter support
5. Uses LibCity's scaler for inverse transform in loss calculation

Model variants available (via 'model_size' config parameter):
- micro: encoder_embed_dim=64, encoder_depth=6, num_heads=2
- mini: encoder_embed_dim=128, encoder_depth=6, num_heads=4 (default)
- small: encoder_embed_dim=192, encoder_depth=8, num_heads=6
- med: encoder_embed_dim=384, encoder_depth=10, num_heads=8
- big: encoder_embed_dim=768, encoder_depth=12, num_heads=12
- large: encoder_embed_dim=1024, encoder_depth=24, num_heads=16
- xl: encoder_embed_dim=1280, encoder_depth=32, num_heads=16

Required config parameters:
- input_window: number of input time steps (default: 12)
- output_window: number of output time steps (default: 12)
- model_size: one of 'micro', 'mini', 'small', 'med', 'big', 'large', 'xl' (default: 'mini')
- encoder_embed_dim: transformer embedding dimension (overrides model_size)
- encoder_depth: number of transformer layers (overrides model_size)
- num_heads: number of attention heads (overrides model_size)
- dropout: dropout rate (default: 0.1)
"""

import math
import numpy as np
from functools import partial
from typing import Optional, Tuple, Callable, Union
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


# ============================================================================
# Graph Algorithms: Floyd-Warshall for shortest path computation
# ============================================================================

def floyd_warshall(adjacency_matrix):
    """
    Compute all-pairs shortest paths using Floyd-Warshall algorithm.
    O(n^3) complexity.

    Args:
        adjacency_matrix: numpy array of shape [N, N]

    Returns:
        M: shortest path distances matrix [N, N]
        path: path reconstruction matrix [N, N]
    """
    nrows, ncols = adjacency_matrix.shape
    assert nrows == ncols
    n = nrows

    M = adjacency_matrix.astype(np.float64, order='C', casting='safe', copy=True)
    path = np.zeros((n, n), dtype=np.int64)

    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = 0
            elif M[i, j] == 0:
                M[i, j] = np.inf
    assert (np.diagonal(M) == 0.0).all()

    for k in range(n):
        for i in range(n):
            for j in range(n):
                cost_ikkj = M[i, k] + M[k, j]
                if M[i, j] > cost_ikkj:
                    M[i, j] = cost_ikkj
                    path[i, j] = k

    M[M >= 510] = 510
    path[M >= 510] = 510

    return M, path


# ============================================================================
# Multi-Head Attention Module
# ============================================================================

class MultiheadAttention(nn.Module):
    """Multi-headed attention with graph attention bias support."""

    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=True,
            self_attention=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            attn_bias: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel"""
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        attn_weights = torch.bmm(q, k.contiguous().transpose(1, 2))

        if attn_bias is not None:
            attn_weights += attn_bias.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        attn_weights_ret: Optional[Tensor] = None
        if need_weights:
            attn_weights_ret = attn_weights_float.contiguous().view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                attn_weights_ret = attn_weights_ret.mean(dim=0)

        return attn, attn_weights_ret


# ============================================================================
# Graphormer Encoder Layer
# ============================================================================

class GraphormerGraphEncoderLayer(nn.Module):
    """Single Graphormer encoder layer with self-attention and FFN."""

    def __init__(
            self,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 3072,
            num_attention_heads: int = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = "gelu",
            pre_layernorm: bool = True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.pre_layernorm = pre_layernorm

        self.dropout_module = nn.Dropout(dropout)
        self.activation_dropout_module = nn.Dropout(activation_dropout)

        self.activation_fn = nn.GELU() if activation_fn == "gelu" else nn.ReLU()

        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, eps=1e-8)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = LayerNorm(self.embedding_dim, eps=1e-8)

    def forward(
            self,
            x: torch.Tensor,
            self_attn_bias: Optional[torch.Tensor] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            get_attn_scores=False,
    ):
        residual = x
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=get_attn_scores,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.final_layer_norm(x)
        return x, attn


# ============================================================================
# Graphormer Node Feature and Attention Bias Modules
# ============================================================================

class Conv2D(nn.Module):
    """2D convolution block for node feature projection."""

    def __init__(
            self,
            input_dims: int,
            output_dims: int,
            kernel_size: Union[tuple, list],
            stride: Union[tuple, list] = (1, 1),
            use_bias: bool = False,
            activation: Optional[Callable] = F.gelu,
    ):
        super(Conv2D, self).__init__()
        self.activation = activation
        self.conv = nn.Conv2d(
            input_dims,
            output_dims,
            kernel_size,
            stride=stride,
            padding=0,
            bias=use_bias,
        )
        self.batch_norm = nn.BatchNorm2d(output_dims)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class FC(nn.Module):
    """Fully connected layer using 1x1 convolutions."""

    def __init__(self, input_dims, units, activations, use_bias):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        self.convs = nn.ModuleList([
            Conv2D(
                input_dims=input_dim,
                output_dims=num_unit,
                kernel_size=[1, 1],
                stride=[1, 1],
                use_bias=use_bias,
                activation=activation,
            )
            for input_dim, num_unit, activation in zip(input_dims, units, activations)
        ])

    def forward(self, x):
        x = x.contiguous().permute(0, 3, 2, 1)
        for conv in self.convs:
            x = conv(x)
        x = x.contiguous().permute(0, 3, 2, 1)
        return x


class GraphNodeFeature(nn.Module):
    """Compute node features with centrality encoding."""

    def __init__(
            self,
            node_feature_dim,
            num_heads,
            num_nodes,
            num_in_degree,
            num_out_degree,
            hidden_dim,
            start_conv=True,
            centrality_encoding=True,
            act_fn='gelu',
    ):
        super(GraphNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.num_nodes = num_nodes
        self.start_conv = start_conv
        self.centrality_encoding = centrality_encoding

        self.activation = nn.GELU() if act_fn == 'gelu' else nn.ReLU()
        if self.start_conv:
            self.fc = FC(
                input_dims=[node_feature_dim, hidden_dim],
                units=[hidden_dim, hidden_dim],
                activations=[self.activation, None],
                use_bias=False,
            )
            if centrality_encoding:
                self.in_degree_encoder = nn.Embedding(
                    num_in_degree + 1, hidden_dim, padding_idx=0
                )
                self.out_degree_encoder = nn.Embedding(
                    num_out_degree + 1, hidden_dim, padding_idx=0
                )

    def forward(self, x, in_degree, out_degree):
        if self.start_conv:
            x = self.fc(x)
        if self.centrality_encoding:
            centrality_encodings = (
                    self.in_degree_encoder(in_degree)
                    + self.out_degree_encoder(out_degree)
            )
            x = x + centrality_encodings.unsqueeze(1)
        return x


class GraphAttnBias(nn.Module):
    """Compute spatial position-based attention bias."""

    def __init__(
            self,
            num_heads,
            num_spatial,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.spatial_pos_encoder = nn.Embedding(num_spatial + 1, num_heads, padding_idx=0)

    def forward(self, attn_bias, spatial_pos):
        N = attn_bias.shape[0]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).contiguous().permute(0, 3, 1, 2)
        graph_attn_bias = graph_attn_bias + spatial_pos_bias
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)

        return graph_attn_bias


# ============================================================================
# Graphormer Graph Encoder
# ============================================================================

class GraphormerGraphEncoder(nn.Module):
    """Full Graphormer encoder with multiple layers."""

    def __init__(
            self,
            node_feature_dim: int,
            num_nodes: int,
            num_in_degree: int,
            num_out_degree: int,
            num_spatial: int,
            num_encoder_layers: int = 12,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 768,
            num_attention_heads: int = 32,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            pre_layernorm: bool = True,
            activation_fn: str = "gelu",
            centrality_encoding: bool = True,
            attention_bias: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.centrality_encoding = centrality_encoding
        self.attention_bias = attention_bias

        self.dropout_module = nn.Dropout(dropout)

        self.graph_node_feature = GraphNodeFeature(
            node_feature_dim=node_feature_dim,
            num_heads=num_attention_heads,
            num_nodes=num_nodes,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            start_conv=True,
            centrality_encoding=centrality_encoding,
            act_fn=activation_fn,
        )

        if attention_bias:
            self.graph_attn_bias = GraphAttnBias(
                num_heads=num_attention_heads,
                num_spatial=num_spatial,
            )

        self.emb_layer_norm = LayerNorm(embedding_dim, eps=1e-8)

        self.layers = nn.ModuleList([
            GraphormerGraphEncoderLayer(
                embedding_dim=embedding_dim,
                ffn_embedding_dim=ffn_embedding_dim,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
                pre_layernorm=pre_layernorm,
            )
            for _ in range(num_encoder_layers)
        ])

    def compute_mods(self, x, in_degree, out_degree, attn_bias, spatial_pos):
        """Compute node features and attention bias."""
        x = self.graph_node_feature(x, in_degree, out_degree)

        attn_bias_out = None
        if self.attention_bias:
            attn_bias_out = self.graph_attn_bias(attn_bias, spatial_pos)
        return x, attn_bias_out

    def forward(
            self,
            x,
            attn_bias=None,
            last_state_only: bool = True,
            attn_mask: Optional[torch.Tensor] = None,
            get_attn_scores=False,
    ):
        if attn_bias is None and self.attention_bias:
            raise ValueError('Missing graph attention bias')
        if get_attn_scores:
            last_state_only = False

        B, T, D = x.shape

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)
        x = self.dropout_module(x)

        x = x.contiguous().transpose(0, 1)

        inner_states, attn_scores = [], []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, attn = layer(
                x,
                self_attn_padding_mask=None,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
                get_attn_scores=get_attn_scores,
            )
            if not last_state_only:
                inner_states.append(x)
                attn_scores.append(attn)

        if last_state_only:
            inner_states = [x]
            attn_scores = [attn]

        return inner_states, attn_scores


# ============================================================================
# TGraphormer Model for LibCity
# ============================================================================

class TGraphormer(AbstractTrafficStateModel):
    """
    TGraphormer: Traffic Graphormer for Spatiotemporal Forecasting.

    This model uses a Graph Transformer architecture for traffic speed prediction,
    incorporating spatial graph structure through attention bias and centrality encoding.
    """

    # Model size configurations (from original TGraphormer implementation)
    MODEL_CONFIGS = {
        'micro': {'encoder_embed_dim': 64, 'encoder_depth': 6, 'num_heads': 2},
        'mini': {'encoder_embed_dim': 128, 'encoder_depth': 6, 'num_heads': 4},
        'small': {'encoder_embed_dim': 192, 'encoder_depth': 8, 'num_heads': 6},
        'med': {'encoder_embed_dim': 384, 'encoder_depth': 10, 'num_heads': 8},
        'big': {'encoder_embed_dim': 768, 'encoder_depth': 12, 'num_heads': 12},
        'large': {'encoder_embed_dim': 1024, 'encoder_depth': 24, 'num_heads': 16},
        'xl': {'encoder_embed_dim': 1280, 'encoder_depth': 32, 'num_heads': 16},
    }

    def __init__(self, config, data_feature):
        # Get data features before calling super().__init__
        self.adj_mx = data_feature.get('adj_mx')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)

        super().__init__(config, data_feature)

        # Model configuration
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.output_dim = data_feature.get('output_dim', 1)
        self.device = config.get('device', torch.device('cpu'))

        # Model size selection (supports 'micro', 'mini', 'small', 'med', 'big', 'large', 'xl')
        model_size = config.get('model_size', 'mini')
        if model_size in self.MODEL_CONFIGS:
            default_config = self.MODEL_CONFIGS[model_size]
        else:
            default_config = self.MODEL_CONFIGS['mini']

        # Graphormer hyperparameters (can override model_size defaults)
        self.encoder_embed_dim = config.get('encoder_embed_dim', default_config['encoder_embed_dim'])
        self.encoder_depth = config.get('encoder_depth', default_config['encoder_depth'])
        self.num_heads = config.get('num_heads', default_config['num_heads'])
        self.dropout = config.get('dropout', 0.1)
        self.end_channel = config.get('end_channel', 512)
        self.use_conv = config.get('use_conv', True)
        self.act_fn = config.get('act_fn', 'gelu')

        # Graph encoding parameters
        self.num_spatial = config.get('num_spatial', 512)
        self.num_in_degree = config.get('num_in_degree', 512)
        self.num_out_degree = config.get('num_out_degree', 512)
        self.spatial_pos_max = config.get('spatial_pos_max', 20)

        # CLS token for graph-level representation
        self.cls_token = config.get('cls_token', True)
        self.attention_bias = config.get('attention_bias', True)
        self.centrality_encoding = config.get('centrality_encoding', True)
        self.sep_pos_embed = config.get('sep_pos_embed', False)

        self._logger = getLogger()
        self._scaler = data_feature.get('scaler')
        self._model_size = model_size

        # Precompute graph structure features
        self._precompute_graph_features()

        # Build model components
        self._build_model()

        self._logger.info(f'TGraphormer initialized: size={model_size}, '
                          f'embed_dim={self.encoder_embed_dim}, depth={self.encoder_depth}, '
                          f'heads={self.num_heads}, nodes={self.num_nodes}, '
                          f'params={self._count_parameters()}')

    def _count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _precompute_graph_features(self):
        """Precompute graph structural features using Floyd-Warshall."""
        if self.adj_mx is not None:
            adj = self.adj_mx.copy()
            adj = (adj > 0).astype(np.float64)

            shortest_path, _ = floyd_warshall(adj)
            shortest_path = np.clip(shortest_path, 0, self.spatial_pos_max)

            self.register_buffer('spatial_pos',
                torch.from_numpy(shortest_path).long() + 1)

            in_degree = adj.sum(axis=1).astype(np.int64)
            out_degree = adj.sum(axis=0).astype(np.int64)
            in_degree = np.clip(in_degree, 0, self.num_in_degree)
            out_degree = np.clip(out_degree, 0, self.num_out_degree)

            self.register_buffer('in_degree', torch.from_numpy(in_degree).long() + 1)
            self.register_buffer('out_degree', torch.from_numpy(out_degree).long() + 1)

            attn_bias = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
            attn_bias[shortest_path >= self.spatial_pos_max] = float('-inf')
            self.register_buffer('base_attn_bias', torch.from_numpy(attn_bias))
        else:
            self._logger.warning('No adjacency matrix provided, using identity graph structure')
            self.register_buffer('spatial_pos',
                torch.ones(self.num_nodes, self.num_nodes).long())
            self.register_buffer('in_degree',
                torch.ones(self.num_nodes).long())
            self.register_buffer('out_degree',
                torch.ones(self.num_nodes).long())
            self.register_buffer('base_attn_bias',
                torch.zeros(self.num_nodes, self.num_nodes))

    def _build_model(self):
        """Build the TGraphormer model architecture."""
        # CLS token for graph-level representation
        if self.cls_token:
            self.cls_token_embed = nn.Parameter(
                torch.zeros(1, 1, self.encoder_embed_dim)
            )
            if self.attention_bias:
                self.cls_token_virtual_distance = nn.Embedding(1, self.num_heads)

        # Position embeddings
        if self.sep_pos_embed:
            self.pos_embed_time = nn.Parameter(
                torch.zeros(1, self.input_window, self.encoder_embed_dim)
            )
            self.pos_embed_space = nn.Parameter(
                torch.zeros(1, self.num_nodes, self.encoder_embed_dim)
            )
            if self.cls_token:
                self.pos_embed_cls = nn.Parameter(
                    torch.zeros(1, 1, self.encoder_embed_dim)
                )
        else:
            if self.cls_token:
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, self.input_window * self.num_nodes + 1, self.encoder_embed_dim)
                )
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, self.input_window * self.num_nodes, self.encoder_embed_dim)
                )

        # Graphormer encoder
        self.blocks = GraphormerGraphEncoder(
            node_feature_dim=self.feature_dim,
            num_nodes=self.num_nodes,
            num_in_degree=self.num_in_degree,
            num_out_degree=self.num_out_degree,
            num_spatial=self.num_spatial,
            num_encoder_layers=self.encoder_depth,
            embedding_dim=self.encoder_embed_dim,
            ffn_embedding_dim=self.encoder_embed_dim * 4,
            num_attention_heads=self.num_heads,
            dropout=self.dropout,
            attention_dropout=self.dropout,
            activation_dropout=self.dropout,
            pre_layernorm=True,
            activation_fn=self.act_fn,
            centrality_encoding=self.centrality_encoding,
            attention_bias=self.attention_bias,
        )

        self.norm = nn.LayerNorm(self.encoder_embed_dim, eps=1e-8)
        self.activation = nn.GELU() if self.act_fn == 'gelu' else nn.ReLU()

        # Prediction head
        if self.output_window > self.input_window:
            self.fc_project = nn.Linear(
                self.input_window * self.encoder_embed_dim,
                self.output_window * self.end_channel
            )
            self.layer_norm = nn.LayerNorm(self.end_channel)
            self.end_conv_2 = nn.Conv2d(
                in_channels=self.end_channel,
                out_channels=self.output_dim,
                kernel_size=(1, 1),
                bias=True
            )
        else:
            if self.use_conv:
                self.end_conv_1 = nn.Conv2d(
                    in_channels=self.encoder_embed_dim,
                    out_channels=self.end_channel,
                    kernel_size=(1, 1),
                    bias=False,
                )
                self.batch_norm = nn.BatchNorm2d(self.end_channel)
                self.end_conv_2 = nn.Conv2d(
                    in_channels=self.end_channel,
                    out_channels=self.output_dim,
                    kernel_size=(1, 1),
                    bias=True
                )
            else:
                self.fc_channel = nn.Linear(self.encoder_embed_dim, self.output_dim)
                self.mlp_pred_dropout = nn.Dropout(0.5)
                self.fc_his = nn.Linear(self.input_window, self.output_window)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        if self.sep_pos_embed:
            nn.init.trunc_normal_(self.pos_embed_time, std=0.02)
            nn.init.trunc_normal_(self.pos_embed_space, std=0.02)
            if self.cls_token:
                nn.init.trunc_normal_(self.pos_embed_cls, std=0.02)
        else:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if self.cls_token:
            nn.init.trunc_normal_(self.cls_token_embed, std=0.02)
            if self.attention_bias:
                nn.init.trunc_normal_(self.cls_token_virtual_distance.weight, std=0.02)

        if self.centrality_encoding:
            nn.init.trunc_normal_(
                self.blocks.graph_node_feature.in_degree_encoder.weight,
                mean=0.0, std=0.02
            )
            nn.init.trunc_normal_(
                self.blocks.graph_node_feature.out_degree_encoder.weight,
                mean=0.0, std=0.02
            )

        if self.attention_bias:
            nn.init.trunc_normal_(
                self.blocks.graph_attn_bias.spatial_pos_encoder.weight,
                mean=0.0, std=0.02
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize individual layer weights."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _add_cls_token_distance(self, attn_bias: torch.Tensor) -> torch.Tensor:
        """Add virtual distance for CLS token to attention bias."""
        if not self.attention_bias:
            return None

        N, n_heads, L, _ = attn_bias.shape

        attn_bias = torch.cat(
            [torch.zeros(N, n_heads, 1, L, device=attn_bias.device), attn_bias],
            dim=2
        )
        attn_bias = torch.cat(
            [torch.zeros(N, n_heads, L + 1, 1, device=attn_bias.device), attn_bias],
            dim=3
        )

        t = self.cls_token_virtual_distance.weight.repeat(N, 1).contiguous().view(-1, n_heads, 1)
        attn_bias[:, :, 1:, 0] = attn_bias.clone()[:, :, 1:, 0] + t
        attn_bias[:, :, 0, :] = attn_bias.clone()[:, :, 0, :] + t

        return attn_bias

    def forward_encoder(self, x: torch.Tensor, attn_bias: torch.Tensor):
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor [B, T, V, D] after node feature processing
            attn_bias: Attention bias [B, n_heads, V, V]

        Returns:
            latent: Encoded representation
            graph_rep: Graph-level representation (from CLS token)
            x_shape: Original shape for reconstruction
        """
        if not self.attention_bias:
            attn_bias = None

        N, T, V, D = x.shape
        x_shape = [N, T, V, D]

        # Flatten spatial-temporal dimensions
        x = x.contiguous().view(N, -1, D)

        # Add CLS token
        if self.cls_token:
            cls_embeds = self.cls_token_embed.expand(N, -1, -1)
            x = torch.cat((cls_embeds, x), dim=1)

        # Expand attention bias for temporal dimension
        if attn_bias is not None:
            attn_bias = attn_bias.repeat(1, 1, T, T)

        # Add CLS token distance
        if self.cls_token and attn_bias is not None:
            attn_bias = self._add_cls_token_distance(attn_bias)

        # Add position embeddings
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_space.repeat(1, T, 1) + \
                       torch.repeat_interleave(self.pos_embed_time, V, dim=1)
            pos_embed = pos_embed.expand(N, -1, -1)
            if self.cls_token:
                pos_embed_token = self.pos_embed_cls.expand(N, -1, -1)
                pos_embed = torch.cat([pos_embed_token, pos_embed], dim=1)
        else:
            pos_embed = self.pos_embed[:, :, :].expand(N, -1, -1)

        x = x + pos_embed.contiguous().view(N, -1, D)

        # Forward through transformer blocks
        inner_states, _ = self.blocks(x, attn_bias)
        x = inner_states[-1].contiguous().transpose(0, 1)

        # Extract graph representation and node features
        if self.cls_token:
            graph_rep = x[:, :1, :]
            x = x[:, 1:, :]
        else:
            graph_rep = None

        return x, graph_rep, x_shape

    def forward_pred(self, x: torch.Tensor, x_shape: list):
        """
        Forward pass through prediction head.

        Args:
            x: Encoded features [N, T*V, D]
            x_shape: Original shape [N, T, V, D]

        Returns:
            Predictions [N, output_window, V, output_dim]
        """
        N, T, V, D = x_shape

        if self.output_window > self.input_window:
            x = x.contiguous().view(N, V, T, D).view(N, V, T * D)
            x = self.fc_project(x)
            x = x.view(N, V, self.output_window, self.end_channel)
            x = self.layer_norm(x)
            x = self.activation(x)
            x = x.contiguous().transpose(1, 3)
            x = self.end_conv_2(x)
            x = x.contiguous().permute(0, 2, 3, 1)
        else:
            x = x.contiguous().view(N, T, V, D).transpose(1, 3)
            if self.use_conv:
                x = self.end_conv_1(x)
                x = self.batch_norm(x)
                x = self.activation(x)
                x = self.end_conv_2(x)
                x = x.transpose(1, 3)
            else:
                x = self.fc_his(x).transpose(1, 3)
                x = self.activation(x)
                x = self.mlp_pred_dropout(x)
                x = self.fc_channel(x)

        return x

    def forward(self, batch):
        """
        Forward pass for LibCity batch.

        Args:
            batch: Dict containing 'X' with shape [B, T_in, N, F]

        Returns:
            Predictions with shape [B, T_out, N, output_dim]
        """
        x = batch['X']  # [B, T, N, F]
        B = x.shape[0]

        # Expand graph features to batch size
        in_degree = self.in_degree.unsqueeze(0).expand(B, -1)
        out_degree = self.out_degree.unsqueeze(0).expand(B, -1)
        spatial_pos = self.spatial_pos.unsqueeze(0).expand(B, -1, -1)
        attn_bias = self.base_attn_bias.unsqueeze(0).expand(B, -1, -1)

        # Compute node features and attention bias
        x, attn_bias = self.blocks.compute_mods(x, in_degree, out_degree, attn_bias, spatial_pos)

        # Forward through encoder
        latent, _, x_shape = self.forward_encoder(x, attn_bias)
        latent = self.norm(latent)

        # Forward through prediction head
        pred = self.forward_pred(latent, x_shape)

        return pred

    def predict(self, batch):
        """
        Predict method for LibCity compatibility.

        Args:
            batch: Dict containing 'X' with shape [B, T_in, N, F]

        Returns:
            Predictions with shape [B, T_out, N, output_dim]
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate training loss using Huber loss.

        Args:
            batch: Dict containing 'X' and 'y'

        Returns:
            loss: Scalar loss tensor
        """
        y_true = batch['y']
        y_predicted = self.predict(batch)

        # Apply inverse transform for proper loss calculation
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        # Use Huber loss (smooth L1) as in original TGraphormer
        return loss.huber_loss(y_predicted, y_true)
