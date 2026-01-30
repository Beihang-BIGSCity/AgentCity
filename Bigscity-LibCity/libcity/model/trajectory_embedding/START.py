"""
START: Self-supervised Trajectory Representation learning with Contrastive Pre-training

This module implements the START model for trajectory embedding learning,
based on BERT architecture with Graph Attention Network (GAT) embeddings
and contrastive learning objectives.

Original paper:
    "START: Self-supervised Trajectory Representation learning with Contrastive Pre-training"

Adapted for LibCity framework.

Key Components:
    - BERT: Base transformer encoder for trajectory sequences
    - BERTEmbedding: Embedding layer with optional GAT-based token embedding
    - GAT: Graph Attention Network for learning location embeddings
    - BERTContrastiveLM: Main model combining contrastive learning with masked LM
    - Downstream models: LinearETA, LinearClassify, LinearSim, LinearNextLoc

Data Format:
    Input batch is a dictionary containing:
        - 'contra_view1': First augmented view for contrastive learning
        - 'contra_view2': Second augmented view for contrastive learning
        - 'masked_input': Masked input for MLM task
        - 'padding_masks': Boolean mask for padding positions
        - 'batch_temporal_mat': Temporal distance matrix between positions
        - 'targets': Target labels for MLM
        - 'target_masks': Mask for MLM targets
        - 'graph_dict': Dictionary containing graph information for GAT

Authors: Adapted from START repository for LibCity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import copy
import random
from logging import getLogger

from libcity.model.abstract_model import AbstractModel


def drop_path_func(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample when applied in main path of residual blocks.

    Args:
        x: Input tensor
        drop_prob: Probability of dropping a path
        training: Whether in training mode

    Returns:
        Tensor with dropped paths during training
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample when applied in main path of residual blocks."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_func(x, self.drop_prob, self.training)


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for transformer models.

    Args:
        d_model: Dimension of the model
        max_len: Maximum sequence length
    """

    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x, position_ids=None):
        """
        Args:
            x: (B, T, d_model) input tensor
            position_ids: (B, T) optional custom position ids

        Returns:
            (1, T, d_model) or (B, T, d_model) positional embeddings
        """
        if position_ids is None:
            return self.pe[:, :x.size(1)].detach()
        else:
            batch_size, seq_len = position_ids.shape
            pe = self.pe[:, :seq_len, :]
            pe = pe.expand((position_ids.shape[0], -1, -1))
            pe = pe.reshape(-1, self.d_model)
            position_ids = position_ids.reshape(-1, 1).squeeze(1)
            output_pe = pe[position_ids].reshape(batch_size, seq_len, self.d_model).detach()
            return output_pe


class GATLayer(nn.Module):
    """Base class for Graph Attention Network layers.

    Args:
        num_in_features: Number of input features per node
        num_out_features: Number of output features per node
        num_of_heads: Number of attention heads
        concat: Whether to concatenate head outputs or average
        activation: Activation function
        dropout_prob: Dropout probability
        add_skip_connection: Whether to add residual connection
        bias: Whether to use bias
        load_trans_prob: Whether to load transition probabilities
    """

    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, load_trans_prob=True):
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
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        if self.load_trans_prob:
            nn.init.xavier_uniform_(self.linear_proj_tran_prob.weight)
            nn.init.xavier_uniform_(self.scoring_trans_prob)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, in_nodes_features, out_nodes_features):
        """Apply skip connection, concatenation, and bias."""
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


class GATLayerImp3(GATLayer):
    """Implementation of GAT layer inspired by PyTorch Geometric.

    Suitable for both transductive and inductive settings.
    """

    src_nodes_dim = 0
    trg_nodes_dim = 1
    nodes_dim = 0
    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, load_trans_prob=True):
        super().__init__(num_in_features, num_out_features, num_of_heads, concat, activation, dropout_prob,
                         add_skip_connection, bias, load_trans_prob)

    def forward(self, data):
        """
        Args:
            data: Tuple of (node_features, edge_index, edge_prob)
                - node_features: (N, FIN)
                - edge_index: (2, E)
                - edge_prob: (E, 1)

        Returns:
            Tuple of (out_features, edge_index, edge_prob)
        """
        in_nodes_features, edge_index, edge_prob = data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        in_nodes_features = self.dropout(in_nodes_features)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)

        if self.load_trans_prob:
            trans_prob_proj = self.linear_proj_tran_prob(edge_prob).view(
                -1, self.num_of_heads, self.num_out_features)
            trans_prob_proj = self.dropout(trans_prob_proj)

        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(
            scores_source, scores_target, nodes_features_proj, edge_index)

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
        """Compute softmax over neighborhoods."""
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()

        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        """Sum edge scores within each neighborhood."""
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        size = list(exp_scores_per_edge.shape)
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        """Aggregate neighborhood features."""
        size = list(nodes_features_proj_lifted_weighted.shape)
        size[self.nodes_dim] = num_of_nodes
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """Lift node features/scores to edge level based on edge index."""
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        """Explicitly broadcast tensor to match dimensions."""
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)
        return this.expand_as(other)


class GAT(nn.Module):
    """Graph Attention Network for learning node embeddings.

    Args:
        d_model: Output dimension
        in_feature: Input feature dimension
        num_heads_per_layer: List of attention heads per layer
        num_features_per_layer: List of feature dimensions per layer
        add_skip_connection: Whether to use skip connections
        bias: Whether to use bias
        dropout: Dropout probability
        load_trans_prob: Whether to use transition probabilities
        avg_last: Whether to average the last layer output
    """

    def __init__(self, d_model, in_feature, num_heads_per_layer, num_features_per_layer,
                 add_skip_connection=True, bias=True, dropout=0.6, load_trans_prob=True, avg_last=True):
        super().__init__()
        self.d_model = d_model
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
            if i == num_of_layers - 1:
                concat_input = not avg_last
            else:
                concat_input = True
            layer = GATLayerImp3(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],
                num_out_features=num_features_per_layer[i+1],
                num_of_heads=num_heads_per_layer[i+1],
                concat=concat_input,
                activation=nn.ELU() if i < num_of_layers - 1 else None,
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                load_trans_prob=load_trans_prob
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(*gat_layers)

    def forward(self, node_features, edge_index_input, edge_prob_input, x):
        """
        Args:
            node_features: (vocab_size, fea_dim)
            edge_index_input: (2, E)
            edge_prob_input: (E, 1)
            x: (B, T) location indices

        Returns:
            (B, T, d_model) embedded features
        """
        data = (node_features, edge_index_input, edge_prob_input)
        (node_fea_emb, edge_index, edge_prob) = self.gat_net(data)
        batch_size, seq_len = x.shape
        node_fea_emb = node_fea_emb.expand((batch_size, -1, -1))
        node_fea_emb = node_fea_emb.reshape(-1, self.d_model)
        x = x.reshape(-1, 1).squeeze(1)
        out_node_fea_emb = node_fea_emb[x].reshape(batch_size, seq_len, self.d_model)
        return out_node_fea_emb


class BERTEmbedding(nn.Module):
    """Embedding layer for BERT combining token, position, time and day embeddings.

    Args:
        d_model: Embedding dimension
        dropout: Dropout probability
        add_time_in_day: Whether to add time-of-day embedding
        add_day_in_week: Whether to add day-of-week embedding
        add_pe: Whether to add positional embedding
        node_fea_dim: Node feature dimension for GAT
        add_gat: Whether to use GAT for token embedding
        gat_heads_per_layer: GAT heads per layer
        gat_features_per_layer: GAT features per layer
        gat_dropout: GAT dropout
        load_trans_prob: Whether to use transition probabilities
        avg_last: Whether to average GAT last layer
    """

    def __init__(self, d_model, dropout=0.1, add_time_in_day=False, add_day_in_week=False,
                 add_pe=True, node_fea_dim=10, add_gat=True,
                 gat_heads_per_layer=None, gat_features_per_layer=None, gat_dropout=0.6,
                 load_trans_prob=True, avg_last=True):
        super().__init__()
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        self.add_pe = add_pe
        self.add_gat = add_gat

        if self.add_gat:
            self.token_embedding = GAT(d_model=d_model, in_feature=node_fea_dim,
                                       num_heads_per_layer=gat_heads_per_layer,
                                       num_features_per_layer=gat_features_per_layer,
                                       add_skip_connection=True, bias=True, dropout=gat_dropout,
                                       load_trans_prob=load_trans_prob, avg_last=avg_last)
        else:
            # Fallback token embedding when GAT is disabled
            # vocab_size needs to be passed or inferred; using a large default
            self.token_embedding = nn.Embedding(100000, d_model, padding_idx=0)
        if self.add_pe:
            self.position_embedding = PositionalEmbedding(d_model=d_model)
        if self.add_time_in_day:
            self.daytime_embedding = nn.Embedding(1441, d_model, padding_idx=0)
        if self.add_day_in_week:
            self.weekday_embedding = nn.Embedding(8, d_model, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, sequence, position_ids=None, graph_dict=None):
        """
        Args:
            sequence: (B, T, F) [loc, ts, mins, weeks, usr]
            position_ids: (B, T) or None
            graph_dict: Dict containing graph info for GAT

        Returns:
            (B, T, d_model) embedded sequence
        """
        if self.add_gat:
            x = self.token_embedding(node_features=graph_dict['node_features'],
                                     edge_index_input=graph_dict['edge_index'],
                                     edge_prob_input=graph_dict['loc_trans_prob'],
                                     x=sequence[:, :, 0])
        else:
            # Use fallback embedding when GAT is disabled
            # Extract location indices from sequence (first feature)
            x = self.token_embedding(sequence[:, :, 0].long())
        if self.add_pe:
            x += self.position_embedding(x, position_ids)
        if self.add_time_in_day:
            x += self.daytime_embedding(sequence[:, :, 2])
        if self.add_day_in_week:
            x += self.weekday_embedding(sequence[:, :, 3])
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    """Multi-headed attention with optional temporal bias.

    Args:
        num_heads: Number of attention heads
        d_model: Model dimension
        dim_out: Output dimension
        attn_drop: Attention dropout
        proj_drop: Projection dropout
        add_cls: Whether CLS token is used
        device: Device for tensors
        add_temporal_bias: Whether to add temporal bias
        temporal_bias_dim: Dimension of temporal bias
        use_mins_interval: Whether to use minutes for temporal intervals
    """

    def __init__(self, num_heads, d_model, dim_out, attn_drop=0., proj_drop=0.,
                 add_cls=True, device=torch.device('cpu'), add_temporal_bias=True,
                 temporal_bias_dim=64, use_mins_interval=False):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.device = device
        self.add_cls = add_cls
        self.scale = self.d_k ** -0.5
        self.add_temporal_bias = add_temporal_bias
        self.temporal_bias_dim = temporal_bias_dim
        self.use_mins_interval = use_mins_interval

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.dropout = nn.Dropout(p=attn_drop)

        self.proj = nn.Linear(d_model, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.add_temporal_bias:
            if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
                self.temporal_mat_bias_1 = nn.Linear(1, self.temporal_bias_dim, bias=True)
                self.temporal_mat_bias_2 = nn.Linear(self.temporal_bias_dim, 1, bias=True)
            elif self.temporal_bias_dim == -1:
                self.temporal_mat_bias = nn.Parameter(torch.Tensor(1, 1))
                nn.init.xavier_uniform_(self.temporal_mat_bias)

    def forward(self, x, padding_masks, future_mask=True, output_attentions=False, batch_temporal_mat=None):
        """
        Args:
            x: (B, T, d_model)
            padding_masks: (B, 1, T, T) padding mask
            future_mask: Whether to apply causal mask
            batch_temporal_mat: (B, T, T) temporal distances

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, d_model = x.shape

        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (x, x, x))]

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if self.add_temporal_bias:
            if self.use_mins_interval:
                batch_temporal_mat = 1.0 / torch.log(
                    torch.exp(torch.tensor(1.0).to(self.device)) +
                    (batch_temporal_mat / torch.tensor(60.0).to(self.device)))
            else:
                batch_temporal_mat = 1.0 / torch.log(
                    torch.exp(torch.tensor(1.0).to(self.device)) + batch_temporal_mat)
            if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
                batch_temporal_mat = self.temporal_mat_bias_2(F.leaky_relu(
                    self.temporal_mat_bias_1(batch_temporal_mat.unsqueeze(-1)),
                    negative_slope=0.2)).squeeze(-1)
            if self.temporal_bias_dim == -1:
                batch_temporal_mat = batch_temporal_mat * self.temporal_mat_bias.expand((1, seq_len, seq_len))
            batch_temporal_mat = batch_temporal_mat.unsqueeze(1)
            scores += batch_temporal_mat

        if padding_masks is not None:
            scores.masked_fill_(padding_masks == 0, float('-inf'))

        if future_mask:
            mask_postion = torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).bool().to(self.device)
            if self.add_cls:
                mask_postion[:, 0, :] = 0
            scores.masked_fill_(mask_postion, float('-inf'))

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        out = torch.matmul(p_attn, value)

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        out = self.proj(out)
        out = self.proj_drop(out)

        if output_attentions:
            return out, p_attn
        else:
            return out, None


class Mlp(nn.Module):
    """Position-wise Feed-Forward Network."""

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


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feed-forward network.

    Args:
        d_model: Model dimension
        attn_heads: Number of attention heads
        feed_forward_hidden: Hidden dimension of FFN
        drop_path: Drop path probability
        attn_drop: Attention dropout
        dropout: General dropout
        type_ln: LayerNorm type ('pre' or 'post')
        add_cls: Whether CLS token is used
        device: Device for computation
        add_temporal_bias: Whether to add temporal bias
        temporal_bias_dim: Temporal bias dimension
        use_mins_interval: Whether to use minutes for temporal
    """

    def __init__(self, d_model, attn_heads, feed_forward_hidden, drop_path,
                 attn_drop, dropout, type_ln='pre', add_cls=True,
                 device=torch.device('cpu'), add_temporal_bias=True,
                 temporal_bias_dim=64, use_mins_interval=False):
        super().__init__()
        self.attention = MultiHeadedAttention(num_heads=attn_heads, d_model=d_model, dim_out=d_model,
                                              attn_drop=attn_drop, proj_drop=dropout, add_cls=add_cls,
                                              device=device, add_temporal_bias=add_temporal_bias,
                                              temporal_bias_dim=temporal_bias_dim,
                                              use_mins_interval=use_mins_interval)
        self.mlp = Mlp(in_features=d_model, hidden_features=feed_forward_hidden,
                       out_features=d_model, act_layer=nn.GELU, drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.type_ln = type_ln

    def forward(self, x, padding_masks, future_mask=True, output_attentions=False, batch_temporal_mat=None):
        """
        Args:
            x: (B, T, d_model)
            padding_masks: (B, 1, T, T)
            future_mask: Whether to use causal mask
            batch_temporal_mat: (B, T, T)

        Returns:
            Tuple of (output, attention_weights)
        """
        if self.type_ln == 'pre':
            attn_out, attn_score = self.attention(self.norm1(x), padding_masks=padding_masks,
                                                  future_mask=future_mask, output_attentions=output_attentions,
                                                  batch_temporal_mat=batch_temporal_mat)
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            attn_out, attn_score = self.attention(x, padding_masks=padding_masks,
                                                  future_mask=future_mask, output_attentions=output_attentions,
                                                  batch_temporal_mat=batch_temporal_mat)
            x = self.norm1(x + self.drop_path(attn_out))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            raise ValueError('Error type_ln {}'.format(self.type_ln))
        return x, attn_score


class BERT(nn.Module):
    """BERT model for trajectory embedding.

    A bidirectional encoder based on transformer architecture, adapted for
    trajectory sequence modeling with optional graph attention embeddings.

    Args:
        config: Configuration dictionary
        data_feature: Dictionary containing data features like vocab_size
    """

    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config

        self.vocab_size = data_feature.get('vocab_size')
        self.usr_num = data_feature.get('usr_num')
        self.node_fea_dim = data_feature.get('node_fea_dim')

        self.d_model = self.config.get('d_model', 768)
        self.n_layers = self.config.get('n_layers', 12)
        self.attn_heads = self.config.get('attn_heads', 12)
        self.mlp_ratio = self.config.get('mlp_ratio', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.drop_path = self.config.get('drop_path', 0.3)
        self.lape_dim = self.config.get('lape_dim', 256)
        self.attn_drop = self.config.get('attn_drop', 0.1)
        self.type_ln = self.config.get('type_ln', 'pre')
        self.future_mask = self.config.get('future_mask', False)
        self.add_cls = self.config.get('add_cls', False)
        self.device = self.config.get('device', torch.device('cpu'))
        self.cutoff_row_rate = self.config.get('cutoff_row_rate', 0.2)
        self.cutoff_column_rate = self.config.get('cutoff_column_rate', 0.2)
        self.cutoff_random_rate = self.config.get('cutoff_random_rate', 0.2)
        self.sample_rate = self.config.get('sample_rate', 0.2)
        self.add_time_in_day = self.config.get('add_time_in_day', True)
        self.add_day_in_week = self.config.get('add_day_in_week', True)
        self.add_pe = self.config.get('add_pe', True)
        self.add_gat = self.config.get('add_gat', True)
        self.gat_heads_per_layer = self.config.get('gat_heads_per_layer', [8, 1])
        self.gat_features_per_layer = self.config.get('gat_features_per_layer', [16, self.d_model])
        self.gat_dropout = self.config.get('gat_dropout', 0.6)
        self.gat_avg_last = self.config.get('gat_avg_last', True)
        self.load_trans_prob = self.config.get('load_trans_prob', False)
        self.add_temporal_bias = self.config.get('add_temporal_bias', True)
        self.temporal_bias_dim = self.config.get('temporal_bias_dim', 64)
        self.use_mins_interval = self.config.get('use_mins_interval', False)

        self.feed_forward_hidden = self.d_model * self.mlp_ratio

        # Embedding layer
        self.embedding = BERTEmbedding(d_model=self.d_model, dropout=self.dropout,
                                       add_time_in_day=self.add_time_in_day, add_day_in_week=self.add_day_in_week,
                                       add_pe=self.add_pe, node_fea_dim=self.node_fea_dim, add_gat=self.add_gat,
                                       gat_heads_per_layer=self.gat_heads_per_layer,
                                       gat_features_per_layer=self.gat_features_per_layer, gat_dropout=self.gat_dropout,
                                       load_trans_prob=self.load_trans_prob, avg_last=self.gat_avg_last)

        # Transformer blocks with stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.n_layers)]
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model=self.d_model, attn_heads=self.attn_heads,
                              feed_forward_hidden=self.feed_forward_hidden, drop_path=enc_dpr[i],
                              attn_drop=self.attn_drop, dropout=self.dropout,
                              type_ln=self.type_ln, add_cls=self.add_cls,
                              device=self.device, add_temporal_bias=self.add_temporal_bias,
                              temporal_bias_dim=self.temporal_bias_dim,
                              use_mins_interval=self.use_mins_interval) for i in range(self.n_layers)])

    def _shuffle_position_ids(self, x, padding_masks, position_ids):
        """Shuffle position IDs for data augmentation."""
        batch_size, seq_len, feat_dim = x.shape
        if position_ids is None:
            position_ids = torch.arange(512).expand((batch_size, -1))[:, :seq_len].to(device=self.device)

        shuffled_pid = []
        for bsz_id in range(batch_size):
            sample_pid = position_ids[bsz_id]
            sample_mask = padding_masks[bsz_id]
            num_tokens = sample_mask.sum().int().item()
            indexes = list(range(num_tokens))
            random.shuffle(indexes)
            rest_indexes = list(range(num_tokens, seq_len))
            total_indexes = indexes + rest_indexes
            shuffled_pid.append(torch.index_select(sample_pid, 0, torch.tensor(total_indexes).to(
                device=self.device)).unsqueeze(0))
        shuffled_pid = torch.cat(shuffled_pid, 0).to(device=self.device)
        return shuffled_pid

    def _sample_span(self, x, padding_masks, sample_rate=0.2):
        """Sample a contiguous span from the sequence for augmentation."""
        batch_size, seq_len, feat_dim = x.shape

        if sample_rate > 0:
            true_seq_len = padding_masks.sum(1).cpu().numpy()
            mask = []
            for true_len in true_seq_len:
                sample_len = max(int(true_len * (1 - sample_rate)), 1)
                start_id = np.random.randint(0, high=true_len - sample_len + 1)
                tmp = [1] * seq_len
                for idx in range(start_id, start_id + sample_len):
                    tmp[idx] = 0
                mask.append(tmp)
            mask = torch.ByteTensor(mask).bool().to(self.device)
            x = x.masked_fill(mask.unsqueeze(-1), value=0.)
            padding_masks = padding_masks.masked_fill(mask, value=0)

        return x, padding_masks

    def _cutoff_embeddings(self, embedding_output, padding_masks, direction, cutoff_rate=0.2):
        """Apply cutoff augmentation to embeddings."""
        batch_size, seq_len, d_model = embedding_output.shape
        cutoff_embeddings = []
        for bsz_id in range(batch_size):
            sample_embedding = embedding_output[bsz_id]
            sample_mask = padding_masks[bsz_id]
            if direction == "row":
                num_dimensions = sample_mask.sum().int().item()
                dim_index = 0
            elif direction == "column":
                num_dimensions = d_model
                dim_index = 1
            elif direction == "random":
                num_dimensions = sample_mask.sum().int().item() * d_model
                dim_index = 0
            else:
                raise ValueError(f"direction should be either row or column, but got {direction}")
            num_cutoff_indexes = int(num_dimensions * cutoff_rate)
            if num_cutoff_indexes < 0 or num_cutoff_indexes > num_dimensions:
                raise ValueError(f"number of cutoff dimensions should be in (0, {num_dimensions}), but got {num_cutoff_indexes}")
            indexes = list(range(num_dimensions))
            random.shuffle(indexes)
            cutoff_indexes = indexes[:num_cutoff_indexes]
            if direction == "random":
                sample_embedding = sample_embedding.reshape(-1)
            cutoff_embedding = torch.index_fill(sample_embedding, dim_index, torch.tensor(
                cutoff_indexes, dtype=torch.long).to(device=self.device), 0.0)
            if direction == "random":
                cutoff_embedding = cutoff_embedding.reshape(seq_len, d_model)
            cutoff_embeddings.append(cutoff_embedding.unsqueeze(0))
        cutoff_embeddings = torch.cat(cutoff_embeddings, 0).to(device=self.device)
        assert cutoff_embeddings.shape == embedding_output.shape
        return cutoff_embeddings

    def _shuffle_embeddings(self, embedding_output, padding_masks):
        """Shuffle embeddings for augmentation."""
        batch_size, seq_len, d_model = embedding_output.shape
        shuffled_embeddings = []
        for bsz_id in range(batch_size):
            sample_embedding = embedding_output[bsz_id]
            sample_mask = padding_masks[bsz_id]
            num_tokens = sample_mask.sum().int().item()
            indexes = list(range(num_tokens))
            random.shuffle(indexes)
            rest_indexes = list(range(num_tokens, seq_len))
            total_indexes = indexes + rest_indexes
            shuffled_embeddings.append(torch.index_select(sample_embedding, 0, torch.tensor(total_indexes).to(
                device=self.device)).unsqueeze(0))
        shuffled_embeddings = torch.cat(shuffled_embeddings, 0).to(device=self.device)
        return shuffled_embeddings

    def forward(self, x, padding_masks, batch_temporal_mat=None, argument_methods=None, graph_dict=None,
                output_hidden_states=False, output_attentions=False):
        """
        Args:
            x: (batch_size, seq_length, feat_dim) input features
            padding_masks: (batch_size, seq_length) boolean mask, 1=keep, 0=padding
            batch_temporal_mat: (batch_size, seq_length, seq_length) temporal distances
            argument_methods: List of augmentation methods to apply
            graph_dict: Dictionary with graph information
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights

        Returns:
            Tuple of (output, hidden_states, attention_weights)
        """
        position_ids = None

        if argument_methods is not None:
            if 'shuffle_position' in argument_methods:
                position_ids = self._shuffle_position_ids(
                    x=x, padding_masks=padding_masks, position_ids=None)
            if 'span' in argument_methods:
                x, attention_mask = self._sample_span(
                    x=x, padding_masks=padding_masks, sample_rate=self.sample_rate)

        # Embed the sequence
        embedding_output = self.embedding(sequence=x, position_ids=position_ids,
                                          graph_dict=graph_dict)

        if argument_methods is not None:
            if 'cutoff_row' in argument_methods:
                embedding_output = self._cutoff_embeddings(
                    embedding_output=embedding_output, padding_masks=padding_masks,
                    direction='row', cutoff_rate=self.cutoff_row_rate)
            if 'cutoff_column' in argument_methods:
                embedding_output = self._cutoff_embeddings(
                    embedding_output=embedding_output, padding_masks=padding_masks,
                    direction='column', cutoff_rate=self.cutoff_column_rate)
            if 'cutoff_random' in argument_methods:
                embedding_output = self._cutoff_embeddings(
                    embedding_output=embedding_output, padding_masks=padding_masks,
                    direction='random', cutoff_rate=self.cutoff_random_rate)
            if 'shuffle_embedding' in argument_methods:
                embedding_output = self._shuffle_embeddings(
                    embedding_output=embedding_output, padding_masks=padding_masks)

        padding_masks_input = padding_masks.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # Run through transformer blocks
        all_hidden_states = [embedding_output] if output_hidden_states else None
        all_self_attentions = [] if output_attentions else None
        for transformer in self.transformer_blocks:
            embedding_output, attn_score = transformer.forward(
                x=embedding_output, padding_masks=padding_masks_input,
                future_mask=self.future_mask, output_attentions=output_attentions,
                batch_temporal_mat=batch_temporal_mat)
            if output_hidden_states:
                all_hidden_states.append(embedding_output)
            if output_attentions:
                all_self_attentions.append(attn_score)

        return embedding_output, all_hidden_states, all_self_attentions


class MLPLayer(nn.Module):
    """MLP layer for CLS pooling."""

    def __init__(self, d_model):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)
        return x


class BERTPooler(nn.Module):
    """Pooler for BERT outputs with multiple pooling strategies."""

    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config
        self.pooling = self.config.get('pooling', 'mean')
        self.add_cls = self.config.get('add_cls', True)
        self.d_model = self.config.get('d_model', 768)
        self.linear = MLPLayer(d_model=self.d_model)

        self._logger = getLogger()
        self._logger.info("Building BERTPooler model")

    def forward(self, bert_output, padding_masks, hidden_states=None):
        """
        Args:
            bert_output: (batch_size, seq_length, d_model)
            padding_masks: (batch_size, seq_length)
            hidden_states: List of hidden states

        Returns:
            (batch_size, d_model) pooled output
        """
        token_emb = bert_output
        if self.pooling == 'cls':
            if self.add_cls:
                return self.linear(token_emb[:, 0, :])
            else:
                raise ValueError('No use cls!')
        elif self.pooling == 'cls_before_pooler':
            if self.add_cls:
                return token_emb[:, 0, :]
            else:
                raise ValueError('No use cls!')
        elif self.pooling == 'mean':
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(token_emb.size()).float()
            sum_embeddings = torch.sum(token_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask
        elif self.pooling == 'max':
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(token_emb.size()).float()
            token_emb[input_mask_expanded == 0] = float('-inf')
            max_over_time = torch.max(token_emb, 1)[0]
            return max_over_time
        elif self.pooling == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            avg_emb = (first_hidden + last_hidden) / 2.0
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(avg_emb.size()).float()
            sum_embeddings = torch.sum(avg_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask
        elif self.pooling == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            avg_emb = (second_last_hidden + last_hidden) / 2.0
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(avg_emb.size()).float()
            sum_embeddings = torch.sum(avg_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask
        else:
            raise ValueError('Error pooling type {}'.format(self.pooling))


class MaskedLanguageModel(nn.Module):
    """Masked Language Model head for predicting masked tokens."""

    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class BERTDownstream(nn.Module):
    """BERT model for downstream tasks with pooled output."""

    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config
        self.vocab_size = data_feature.get('vocab_size')
        self.usr_num = data_feature.get('usr_num')
        self.pooling = self.config.get('pooling', 'mean')
        self.d_model = self.config.get('d_model', 768)
        self.add_cls = self.config.get('add_cls', True)

        self._logger = getLogger()
        self._logger.info("Building BERTDownstream model")

        self.bert = BERT(config, data_feature)

    def forward(self, x, padding_masks, batch_temporal_mat=None,
                graph_dict=None, output_hidden_states=False, output_attentions=False):
        """
        Args:
            x: (batch_size, seq_length, feat_dim)
            padding_masks: (batch_size, seq_length)

        Returns:
            (batch_size, d_model) pooled output
        """
        if self.pooling in ['avg_first_last', 'avg_top2']:
            output_hidden_states = True
        token_emb, hidden_states, _ = self.bert(x=x, padding_masks=padding_masks,
                                                batch_temporal_mat=batch_temporal_mat, graph_dict=graph_dict,
                                                output_hidden_states=output_hidden_states,
                                                output_attentions=output_attentions)
        if self.pooling == 'cls' or self.pooling == 'cls_before_pooler':
            if self.add_cls:
                return token_emb[:, 0, :]
            else:
                raise ValueError('No use cls!')
        elif self.pooling == 'mean':
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(token_emb.size()).float()
            sum_embeddings = torch.sum(token_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask
        elif self.pooling == 'max':
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(token_emb.size()).float()
            token_emb[input_mask_expanded == 0] = float('-inf')
            max_over_time = torch.max(token_emb, 1)[0]
            return max_over_time
        elif self.pooling == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            avg_emb = (first_hidden + last_hidden) / 2.0
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(avg_emb.size()).float()
            sum_embeddings = torch.sum(avg_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask
        elif self.pooling == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            avg_emb = (second_last_hidden + last_hidden) / 2.0
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(avg_emb.size()).float()
            sum_embeddings = torch.sum(avg_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask
        else:
            raise ValueError('Error pooling type {}'.format(self.pooling))


class BERTLM(nn.Module):
    """BERT with Masked Language Model head."""

    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config
        self.vocab_size = data_feature.get('vocab_size')
        self.usr_num = data_feature.get('usr_num')
        self.d_model = self.config.get('d_model', 768)

        self._logger = getLogger()
        self._logger.info("Building BERTLM model")

        self.bert = BERT(config, data_feature)
        self.mask_l = MaskedLanguageModel(self.d_model, self.vocab_size)

    def forward(self, x, padding_masks, batch_temporal_mat=None,
                graph_dict=None, output_hidden_states=False, output_attentions=False):
        """
        Args:
            x: (batch_size, seq_length, feat_dim)
            padding_masks: (batch_size, seq_length)

        Returns:
            (batch_size, seq_length, vocab_size) MLM predictions
        """
        x, _, _ = self.bert(x=x, padding_masks=padding_masks,
                            batch_temporal_mat=batch_temporal_mat, graph_dict=graph_dict,
                            output_hidden_states=output_hidden_states,
                            output_attentions=output_attentions)
        return self.mask_l(x)


class BERTContrastive(nn.Module):
    """BERT with contrastive learning objective."""

    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config
        self.pooling = self.config.get('pooling', 'mean')

        self._logger = getLogger()
        self._logger.info("Building BERTContrastive model")

        self.bert = BERT(config, data_feature)
        self.pooler = BERTPooler(config, data_feature)

    def forward(self, x, padding_masks, argument_methods, batch_temporal_mat=None,
                graph_dict=None, output_hidden_states=False, output_attentions=False):
        """
        Args:
            x: (batch_size, seq_length, feat_dim)
            padding_masks: (batch_size, seq_length)
            argument_methods: Augmentation methods

        Returns:
            (batch_size, d_model) pooled representation
        """
        x, hidden_states, _ = self.bert(x=x, padding_masks=padding_masks,
                                        argument_methods=argument_methods,
                                        batch_temporal_mat=batch_temporal_mat, graph_dict=graph_dict,
                                        output_hidden_states=output_hidden_states,
                                        output_attentions=output_attentions)
        x = self.pooler(bert_output=x, padding_masks=padding_masks, hidden_states=hidden_states)
        return x


class BERTContrastiveLM(nn.Module):
    """BERT with both contrastive learning and masked language model objectives.

    This is the main START model combining:
    - Contrastive learning between two augmented views
    - Masked language modeling for self-supervised learning

    Args:
        config: Configuration dictionary
        data_feature: Dictionary containing data features
    """

    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config
        self.vocab_size = data_feature.get('vocab_size')
        self.usr_num = data_feature.get('usr_num')
        self.d_model = self.config.get('d_model', 768)
        self.pooling = self.config.get('pooling', 'mean')

        self._logger = getLogger()
        self._logger.info("Building BERTContrastiveLM model")

        self.bert = BERT(config, data_feature)
        self.mask_l = MaskedLanguageModel(self.d_model, self.vocab_size)
        self.pooler = BERTPooler(config, data_feature)

    def forward(self, contra_view1, contra_view2, argument_methods1,
                argument_methods2, masked_input, padding_masks,
                batch_temporal_mat, padding_masks1=None, padding_masks2=None,
                batch_temporal_mat1=None, batch_temporal_mat2=None,
                graph_dict=None, output_hidden_states=False, output_attentions=False):
        """
        Args:
            contra_view1: (batch_size, seq_length, feat_dim) first contrastive view
            contra_view2: (batch_size, seq_length, feat_dim) second contrastive view
            argument_methods1: Augmentation methods for view 1
            argument_methods2: Augmentation methods for view 2
            masked_input: (batch_size, seq_length, feat_dim) masked input for MLM
            padding_masks: (batch_size, seq_length)
            batch_temporal_mat: (batch_size, seq_length, seq_length)
            graph_dict: Dictionary with graph information

        Returns:
            Tuple of (pooled_view1, pooled_view2, mlm_predictions)
        """
        if self.pooling in ['avg_first_last', 'avg_top2']:
            output_hidden_states = True
        if padding_masks1 is None:
            padding_masks1 = padding_masks
        if padding_masks2 is None:
            padding_masks2 = padding_masks
        if batch_temporal_mat1 is None:
            batch_temporal_mat1 = batch_temporal_mat
        if batch_temporal_mat2 is None:
            batch_temporal_mat2 = batch_temporal_mat

        # Process first contrastive view
        out_view1, hidden_states1, _ = self.bert(x=contra_view1, padding_masks=padding_masks1,
                                                 batch_temporal_mat=batch_temporal_mat1,
                                                 argument_methods=argument_methods1, graph_dict=graph_dict,
                                                 output_hidden_states=output_hidden_states,
                                                 output_attentions=output_attentions)
        pool_out_view1 = self.pooler(bert_output=out_view1, padding_masks=padding_masks1,
                                     hidden_states=hidden_states1)

        # Process second contrastive view
        out_view2, hidden_states2, _ = self.bert(x=contra_view2, padding_masks=padding_masks2,
                                                 batch_temporal_mat=batch_temporal_mat2,
                                                 argument_methods=argument_methods2, graph_dict=graph_dict,
                                                 output_hidden_states=output_hidden_states,
                                                 output_attentions=output_attentions)
        pool_out_view2 = self.pooler(bert_output=out_view2, padding_masks=padding_masks2,
                                     hidden_states=hidden_states2)

        # Process masked input for MLM
        bert_output, _, _ = self.bert(x=masked_input, padding_masks=padding_masks,
                                      batch_temporal_mat=batch_temporal_mat, graph_dict=graph_dict,
                                      output_hidden_states=output_hidden_states,
                                      output_attentions=output_attentions)
        mlm_output = self.mask_l(bert_output)

        return pool_out_view1, pool_out_view2, mlm_output


class START(AbstractModel):
    """START: Self-supervised Trajectory Representation learning with Contrastive Pre-training.

    This is the main model class adapted for LibCity framework, wrapping BERTContrastiveLM
    and providing the standard LibCity interface (predict, calculate_loss).

    Args:
        config: Configuration dictionary
        data_feature: Dictionary containing data features like vocab_size, usr_num, node_fea_dim
    """

    def __init__(self, config, data_feature):
        super(START, self).__init__(config, data_feature)

        self.config = config
        self.data_feature = data_feature

        self.vocab_size = data_feature.get('vocab_size')
        self.usr_num = data_feature.get('usr_num', 0)
        self.d_model = config.get('d_model', 768)
        self.device = config.get('device', torch.device('cpu'))

        # Loss parameters
        self.mlm_ratio = config.get('mlm_ratio', 1.0)
        self.contra_ratio = config.get('contra_ratio', 1.0)
        self.temperature = config.get('temperature', 0.05)
        self.contra_loss_type = config.get('contra_loss_type', 'simclr').lower()

        # Data augmentation parameters
        self.data_argument1 = config.get('data_argument1', ['shuffle_position'])
        self.data_argument2 = config.get('data_argument2', ['shuffle_position'])

        self._logger = getLogger()
        self._logger.info("Building START model")

        # Build the core model
        self.model = BERTContrastiveLM(config, data_feature)

        # Loss functions
        self.criterion_mlm = nn.NLLLoss(ignore_index=0, reduction='none')
        self.criterion_contra = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, batch):
        """Forward pass through the model.

        Args:
            batch: Dictionary containing input data

        Returns:
            Tuple of (pooled_view1, pooled_view2, mlm_predictions)
        """
        contra_view1 = batch['contra_view1'].to(self.device)
        contra_view2 = batch['contra_view2'].to(self.device)
        masked_input = batch['masked_input'].to(self.device)
        padding_masks = batch['padding_masks'].to(self.device)
        batch_temporal_mat = batch['batch_temporal_mat'].to(self.device)

        # Optional separate masks for contrastive views
        padding_masks1 = batch.get('padding_masks1')
        padding_masks2 = batch.get('padding_masks2')
        batch_temporal_mat1 = batch.get('batch_temporal_mat1')
        batch_temporal_mat2 = batch.get('batch_temporal_mat2')

        if padding_masks1 is not None:
            padding_masks1 = padding_masks1.to(self.device)
        if padding_masks2 is not None:
            padding_masks2 = padding_masks2.to(self.device)
        if batch_temporal_mat1 is not None:
            batch_temporal_mat1 = batch_temporal_mat1.to(self.device)
        if batch_temporal_mat2 is not None:
            batch_temporal_mat2 = batch_temporal_mat2.to(self.device)

        graph_dict = batch.get('graph_dict')
        if graph_dict is not None:
            graph_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                          for k, v in graph_dict.items()}

        return self.model(
            contra_view1=contra_view1,
            contra_view2=contra_view2,
            argument_methods1=self.data_argument1,
            argument_methods2=self.data_argument2,
            masked_input=masked_input,
            padding_masks=padding_masks,
            batch_temporal_mat=batch_temporal_mat,
            padding_masks1=padding_masks1,
            padding_masks2=padding_masks2,
            batch_temporal_mat1=batch_temporal_mat1,
            batch_temporal_mat2=batch_temporal_mat2,
            graph_dict=graph_dict
        )

    def predict(self, batch):
        """Generate predictions for a batch.

        Args:
            batch: Dictionary containing input data

        Returns:
            Dictionary with pooled embeddings and MLM predictions
        """
        pool_out_view1, pool_out_view2, mlm_output = self.forward(batch)
        return {
            'embedding_view1': pool_out_view1,
            'embedding_view2': pool_out_view2,
            'mlm_predictions': mlm_output
        }

    def calculate_loss(self, batch):
        """Calculate combined contrastive and MLM loss.

        Args:
            batch: Dictionary containing input data and targets

        Returns:
            Total loss tensor
        """
        pool_out_view1, pool_out_view2, mlm_output = self.forward(batch)

        # MLM loss
        targets = batch['targets'].to(self.device)
        target_masks = batch['target_masks'].to(self.device)
        targets_l = targets[..., 0]
        target_masks_l = target_masks[..., 0]

        batch_loss = self.criterion_mlm(mlm_output.transpose(1, 2), targets_l)
        batch_loss = torch.sum(batch_loss)
        num_active = target_masks_l.sum()
        mlm_loss = batch_loss / num_active if num_active > 0 else batch_loss

        # Contrastive loss
        contra_loss = self._contrastive_loss(pool_out_view1, pool_out_view2)

        # Combined loss
        total_loss = self.mlm_ratio * mlm_loss + self.contra_ratio * contra_loss

        return total_loss

    def _contrastive_loss(self, z1, z2):
        """Calculate contrastive loss between two views.

        Args:
            z1: (batch_size, d_model) embeddings from view 1
            z2: (batch_size, d_model) embeddings from view 2

        Returns:
            Contrastive loss tensor
        """
        if self.contra_loss_type == 'simclr':
            return self._contrastive_loss_simclr(z1, z2)
        elif self.contra_loss_type == 'simsce':
            return self._contrastive_loss_simsce(z1, z2)
        else:
            return self._contrastive_loss_simclr(z1, z2)

    def _contrastive_loss_simclr(self, z1, z2):
        """SimCLR contrastive loss."""
        assert z1.shape == z2.shape
        batch_size, d_model = z1.shape
        features = torch.cat([z1, z2], dim=0)

        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        logits = logits / self.temperature

        return self.criterion_contra(logits, labels)

    def _contrastive_loss_simsce(self, z1, z2):
        """SimCSE contrastive loss."""
        assert z1.shape == z2.shape
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        similarity_matrix = torch.matmul(z1, z2.T)
        similarity_matrix /= self.temperature

        labels = torch.arange(similarity_matrix.shape[0]).long().to(self.device)
        return self.criterion_contra(similarity_matrix, labels)


# Downstream task models

class LinearNextLoc(nn.Module):
    """Linear layer for next location prediction downstream task."""

    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config
        self.vocab_size = data_feature.get('vocab_size')
        self.usr_num = data_feature.get('usr_num')
        self.pooling = self.config.get('pooling', 'mean')
        self.d_model = self.config.get('d_model', 768)

        self._logger = getLogger()
        self._logger.info("Building Downstream LinearNextLoc model")

        self.model = BERTDownstream(config, data_feature)
        self.linear = nn.Linear(self.d_model, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, padding_masks, batch_temporal_mat=None,
                graph_dict=None, output_hidden_states=False, output_attentions=False):
        traj_emb = self.model(x=x, padding_masks=padding_masks,
                              batch_temporal_mat=batch_temporal_mat, graph_dict=graph_dict,
                              output_hidden_states=output_hidden_states,
                              output_attentions=output_attentions)
        nloc_pred = self.softmax(self.linear(traj_emb))
        return nloc_pred


class LinearETA(nn.Module):
    """Linear layer for ETA prediction downstream task."""

    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config
        self.vocab_size = data_feature.get('vocab_size')
        self.usr_num = data_feature.get('usr_num')
        self.pooling = self.config.get('pooling', 'mean')
        self.d_model = self.config.get('d_model', 768)

        self._logger = getLogger()
        self._logger.info("Building Downstream LinearETA model")

        self.model = BERTDownstream(config, data_feature)
        self.linear = nn.Linear(self.d_model, 1)

    def forward(self, x, padding_masks, batch_temporal_mat=None,
                graph_dict=None, output_hidden_states=False, output_attentions=False):
        traj_emb = self.model(x=x, padding_masks=padding_masks,
                              batch_temporal_mat=batch_temporal_mat, graph_dict=graph_dict,
                              output_hidden_states=output_hidden_states,
                              output_attentions=output_attentions)
        eta_pred = self.linear(traj_emb)
        return eta_pred


class LinearClassify(nn.Module):
    """Linear layer for classification downstream task."""

    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config
        self.usr_num = data_feature.get('usr_num')
        self.d_model = self.config.get('d_model', 768)
        self.dataset = self.config.get('dataset', '')
        self.classify_label = self.config.get('classify_label', 'vflag')

        self._logger = getLogger()
        self._logger.info("Building Downstream LinearClassify model")

        self.model = BERTDownstream(config, data_feature)
        if self.classify_label == 'vflag':
            self.linear = nn.Linear(self.d_model, 2)
            if self.dataset == 'geolife':
                self.linear = nn.Linear(self.d_model, 4)
        elif self.classify_label == 'usrid':
            self.linear = nn.Linear(self.d_model, self.usr_num)
        else:
            raise ValueError('Error classify_label = {}'.format(self.classify_label))
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, padding_masks, batch_temporal_mat=None,
                graph_dict=None, output_hidden_states=False, output_attentions=False):
        traj_emb = self.model(x=x, padding_masks=padding_masks,
                              batch_temporal_mat=batch_temporal_mat, graph_dict=graph_dict,
                              output_hidden_states=output_hidden_states,
                              output_attentions=output_attentions)
        cls_pred = self.softmax(self.linear(traj_emb))
        return cls_pred


class LinearSim(nn.Module):
    """Linear layer for similarity downstream task."""

    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config
        self.usr_num = data_feature.get('usr_num')
        self.d_model = self.config.get('d_model', 768)

        self._logger = getLogger()
        self._logger.info("Building Downstream LinearSim model")

        self.model = BERTDownstream(config, data_feature)

    def forward(self, x, padding_masks, batch_temporal_mat=None,
                graph_dict=None, output_hidden_states=False, output_attentions=False):
        traj_emb = self.model(x=x, padding_masks=padding_masks,
                              batch_temporal_mat=batch_temporal_mat, graph_dict=graph_dict,
                              output_hidden_states=output_hidden_states,
                              output_attentions=output_attentions)
        return traj_emb
