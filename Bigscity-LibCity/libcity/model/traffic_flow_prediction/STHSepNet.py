"""
STHSepNet: Spatio-Temporal Hypergraph Separation Network for Traffic Flow Prediction

This is a LibCity-compatible adaptation of the STHSepNet model.
Original repository: STHSepNet

Key Features:
- Temporal Module: Optional LLM-based or lightweight temporal encoder
- Spatial Module: Adaptive hypergraph neural networks
- Fusion: Learnable gating mechanisms (alpha, beta, gamma, theta)

Adaptations for LibCity:
- Inherits from AbstractTrafficStateModel
- Uses LibCity's batch format (X, y tensors)
- Device handling via config
- Adjacency matrix from data_feature
- Removed Accelerate/DeepSpeed dependencies
- Made LLM integration optional with lightweight default

Authors: Adapted for LibCity framework
"""

import math
import numbers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from logging import getLogger

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# Helper Layers: Graph and Hypergraph Constructors
# =============================================================================

class GraphConstructor(nn.Module):
    """Adaptive graph constructor for first-order interactions."""

    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(GraphConstructor, self).__init__()
        self.nnodes = nnodes
        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

    def forward(self, idx):
        device = self.emb1.weight.device
        idx = idx.to(device)

        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))

        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask = torch.zeros(idx.size(0), idx.size(0)).to(device)
        mask.fill_(0)
        s1 = s1.float()
        s1.fill_(1)
        s1 = s1.to(device)
        t1 = t1.to(device)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask

        return adj


class HypergraphConstructor(nn.Module):
    """Adaptive hypergraph constructor for high-order interactions."""

    def __init__(self, nnodes, num_hyperedges, dim, device, scale_hyperedges=10,
                 alpha=3, static_feat=None, metric='knn'):
        super(HypergraphConstructor, self).__init__()
        self.nnodes = nnodes
        self.num_hyperedges = num_hyperedges
        self.device = device
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
        self.metric = metric
        self.scale_hyperedges = scale_hyperedges

        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)

    def forward(self, idx):
        device = self.emb1.weight.device
        idx = idx.to(device)

        if self.static_feat is None:
            node_features = self.emb1(idx)
        else:
            node_features = self.static_feat[idx, :]

        transformed_features = torch.tanh(self.alpha * self.lin1(node_features))

        if self.metric == 'knn' and SKLEARN_AVAILABLE:
            num_neighbors = self.scale_hyperedges
            transformed_features_np = transformed_features.to(torch.float32).cpu().detach().numpy()
            nn_model = NearestNeighbors(n_neighbors=num_neighbors, metric='euclidean')
            nn_model.fit(transformed_features_np)
            distances, indices = nn_model.kneighbors(transformed_features_np)

            hyperedges = set()
            for i in range(self.nnodes):
                hyperedge = frozenset(indices[i])
                hyperedges.add(hyperedge)

            unique_hyperedges = list(hyperedges)
            H = torch.zeros((len(unique_hyperedges), self.nnodes), device=device)

            for idx_h, hyperedge in enumerate(unique_hyperedges):
                for node in hyperedge:
                    H[idx_h, node] = 1

        elif self.metric == 'cluster' and SKLEARN_AVAILABLE:
            kmeans = KMeans(n_clusters=self.num_hyperedges)
            kmeans.fit(transformed_features.cpu().detach().numpy())
            cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.long, device=device)
            H = torch.zeros((self.num_hyperedges, self.nnodes), device=device)
            for i in range(self.nnodes):
                H[cluster_labels[i], i] = 1
        else:
            # Fallback: use similarity-based hypergraph
            sim = torch.mm(transformed_features, transformed_features.t())
            _, top_indices = torch.topk(sim, min(self.scale_hyperedges, self.nnodes), dim=1)
            H = torch.zeros((self.nnodes, self.nnodes), device=device)
            for i in range(self.nnodes):
                H[i, top_indices[i]] = 1

        H = H.t()
        return H


# =============================================================================
# Hypergraph Neural Network Layers
# =============================================================================

class HypergraphConvolution(nn.Module):
    """Hypergraph convolution layer."""

    def __init__(self, in_channels, out_channels):
        super(HypergraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.W = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.U = nn.Parameter(torch.Tensor(out_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.U)

    def forward(self, x, H):
        """
        Args:
            x: Node features (B, L, N, F) or (B, F, N, L)
            H: Hypergraph incidence matrix (N, M)
        """
        device = self.W.device
        x = x.to(device)
        H = H.to(device).float()

        # Node to hyperedge aggregation
        x_enc = torch.matmul(x, self.W)
        x_enc = torch.einsum('nm,blnf->blmf', H, x_enc)

        # Transformation on hyperedges
        x_enc = F.relu(torch.matmul(x_enc, self.U))

        # Hyperedge to node aggregation
        x_out = torch.einsum('mn,blmf->blnf', H.t(), x_enc)

        return x_out


class HypergraphAttention(nn.Module):
    """Hypergraph attention layer."""

    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(HypergraphAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.W_self = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.W_neigh = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.W_concat = nn.Parameter(torch.Tensor(2 * out_channels, out_channels))
        self.attention = nn.Parameter(torch.Tensor(2 * out_channels, 1))

        nn.init.xavier_uniform_(self.W_self)
        nn.init.xavier_uniform_(self.W_neigh)
        nn.init.xavier_uniform_(self.W_concat)
        nn.init.xavier_uniform_(self.attention)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, H):
        device = self.W_self.device
        x = x.to(device)
        H = H.to(device).float()

        x_self = torch.einsum('blnf,fd->blnd', x, self.W_self)

        hyperedge_features = torch.einsum('mn,blnf->blmf', H.t(), x)
        hyperedge_degrees = H.sum(dim=0).clamp(min=1)
        hyperedge_features = hyperedge_features / hyperedge_degrees.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        hyperedge_features = self.leaky_relu(hyperedge_features)

        x_neigh = torch.einsum('nm,blmf->blnf', H, hyperedge_features)
        node_degrees = H.sum(dim=1).clamp(min=1)
        x_neigh = x_neigh / node_degrees.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        x_neigh = torch.einsum('blnf,fd->blnd', x_neigh, self.W_neigh)

        x_concat = torch.cat([x_self, x_neigh], dim=-1)

        attention_scores = torch.einsum('blnd,df->bln', x_concat, self.attention)
        attention_weights = self.softmax(attention_scores)
        attention_weights = self.dropout_layer(attention_weights)

        x_attention = x_concat * attention_weights.unsqueeze(-1)
        x_out = torch.einsum('blnd,df->blnf', x_attention, self.W_concat)

        return x_out


class HypergraphSAGE(nn.Module):
    """Hypergraph SAGE layer."""

    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(HypergraphSAGE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.W_self = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.W_neigh = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.W_concat = nn.Parameter(torch.Tensor(2 * out_channels, out_channels))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_self)
        nn.init.xavier_uniform_(self.W_neigh)
        nn.init.xavier_uniform_(self.W_concat)

    def forward(self, x, H):
        device = self.W_self.device
        x = x.to(device)
        H = H.to(device).float()

        hyperedge_degrees = H.sum(dim=0).clamp(min=1)
        node_degrees = H.sum(dim=1).clamp(min=1)

        x_self = torch.einsum('blnf,fd->blnd', x, self.W_self)

        hyperedge_features = torch.einsum('mn,blnf->blmf', H.t(), x)
        hyperedge_features = hyperedge_features / hyperedge_degrees.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        hyperedge_features = self.relu(hyperedge_features)

        x_neigh = torch.einsum('nm,blmf->blnf', H, hyperedge_features)
        x_neigh = x_neigh / node_degrees.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        x_neigh = torch.einsum('blnf,fd->blnd', x_neigh, self.W_neigh)

        x_concat = torch.cat([x_self, x_neigh], dim=-1)
        x_out = torch.einsum('blnd,df->blnf', x_concat, self.W_concat)

        return x_out


# =============================================================================
# Graph Convolution Layers
# =============================================================================

class NConv(nn.Module):
    """Normalized graph convolution."""

    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()


class Linear(nn.Module):
    """Linear layer implemented as 1x1 convolution."""

    def __init__(self, c_in, c_out, bias=True):
        super(Linear, self).__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class MixProp(nn.Module):
    """Mixed propagation layer for GCN."""

    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(MixProp, self).__init__()
        self.nconv = NConv()
        self.mlp = Linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device).to(adj.dtype)
        d = adj.sum(1).clamp(min=1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


class DilatedInception(nn.Module):
    """Dilated inception module for temporal convolution."""

    def __init__(self, cin, cout, dilation_factor=2):
        super(DilatedInception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout_per_kernel = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout_per_kernel, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class STHGNNLayerNorm(nn.Module):
    """Layer normalization for STHGNN."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(STHGNNLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]),
                              self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)


# =============================================================================
# Fusion Gates
# =============================================================================

class AttentionFusion(nn.Module):
    """Attention-based fusion module."""

    def __init__(self, input_size):
        super(AttentionFusion, self).__init__()
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, dec_out, sthgnn_enc):
        query = self.query(dec_out)
        key = self.key(sthgnn_enc)
        value = self.value(sthgnn_enc)

        scores = torch.bmm(query, key.transpose(1, 2)) / (query.size(-1) ** 0.5)
        attention_weights = self.softmax(scores)
        attended_sthgnn_enc = torch.bmm(attention_weights, value)
        fused_output = dec_out + attended_sthgnn_enc

        return fused_output


class LSTMGate(nn.Module):
    """LSTM-based fusion gate."""

    def __init__(self, input_size):
        super(LSTMGate, self).__init__()
        self.lstm = nn.LSTM(input_size, input_size, batch_first=True)

    def forward(self, x, h):
        B, T, N = x.size()
        h = h.view(B, T, N).transpose(0, 1).contiguous()
        h0 = h[-1].unsqueeze(0)
        c0 = torch.zeros_like(h0)
        output, (h_n, c_n) = self.lstm(x, (h0, c0))
        fused_output = output.view(B, T, N)
        return fused_output


# =============================================================================
# Spatio-Temporal Hypergraph Neural Network
# =============================================================================

class STHGNN(nn.Module):
    """Spatio-Temporal Hypergraph Neural Network module."""

    def __init__(self, gcn_true, hgcn_true, hgat_true, buildA_true, buildH_true,
                 gcn_depth, num_nodes, num_hyperedges, device, adaptive_hyperhgnn,
                 temporl_true, static, predefined_A=None, static_feat=None,
                 dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1,
                 conv_channels=32, residual_channels=32, skip_channels=64,
                 end_channels=128, seq_length=12, in_dim=2, out_dim=12,
                 layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True,
                 true_adj=None, scale_hyperedges=10, alpha=0.1, beta=0.3,
                 gamma=0.5, theta=0.2):
        super(STHGNN, self).__init__()

        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.hgcn_true = hgcn_true
        self.hgat_true = hgat_true
        self.buildH_true = buildH_true
        self.adaptive_hyperhgnn = adaptive_hyperhgnn
        self.temporl_true = temporl_true
        self.static = static

        # Learnable fusion parameters
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        self.alpha_param = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float32))

        self.num_nodes = num_nodes
        self.num_hyperedges = num_hyperedges
        self.scale_hyperedges = scale_hyperedges
        self.dropout = dropout
        self.predefined_A = predefined_A

        if true_adj is not None:
            self.register_buffer('true_adj', torch.tensor(true_adj).float())
        else:
            self.register_buffer('true_adj', torch.zeros(num_nodes, num_nodes))

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.gconv3 = nn.ModuleList()
        self.norm = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))

        # Graph constructor
        self.gc = GraphConstructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        # Hypergraph constructor
        self.hgc = HypergraphConstructor(num_nodes, num_hyperedges, node_dim, device,
                                         scale_hyperedges=scale_hyperedges, alpha=tanhalpha,
                                         static_feat=static_feat, metric='knn')

        self.seq_length = seq_length
        kernel_size = 7

        if dilation_exponential > 1:
            self.receptive_field = int(1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(1 + i * (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1

            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(DilatedInception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(DilatedInception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=residual_channels, kernel_size=(1, 1)))

                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length - rf_size_j + 1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field - rf_size_j + 1)))

                if self.gcn_true:
                    self.gconv1.append(MixProp(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(MixProp(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv3.append(MixProp(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length > self.receptive_field:
                    self.norm.append(STHGNNLayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
                                                    elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(STHGNNLayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
                                                    elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True)

        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
                                  kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)

        self.idx = torch.arange(self.num_nodes)
        self.device = device

    def forward(self, input, idx=None):
        seq_len = input.size(3)

        if seq_len != self.seq_length:
            # Pad or truncate to match expected sequence length
            if seq_len < self.seq_length:
                input = F.pad(input, (self.seq_length - seq_len, 0, 0, 0))
            else:
                input = input[..., -self.seq_length:]

        if self.seq_length < self.receptive_field:
            input = F.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))

        # Build adaptive graph
        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx.to(input.device))
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A
            adp = adp.to(input.dtype)

        # Build adaptive hypergraph
        if self.hgcn_true:
            if self.buildH_true:
                if idx is None:
                    hadp = self.hgc(self.idx.to(input.device))
                else:
                    hadp = self.hgc(idx)
            else:
                hadp = None
            if hadp is not None:
                hadp = hadp.to(input.dtype)

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))

        for i in range(self.layers):
            residual = x

            if self.temporl_true:
                filter_out = self.filter_convs[i](x)
                filter_out = torch.tanh(filter_out)
                gate = self.gate_convs[i](x)
                gate = torch.sigmoid(gate)
                x = filter_out * gate
                x = F.dropout(x, self.dropout, training=self.training)
            else:
                x = self.filter_convs[i](x)

            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

            true_adj = self.true_adj.to(x.device)

            if self.static:
                x1 = self.gconv3[i](x, true_adj)
            elif self.gcn_true:
                x1 = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))
            else:
                x1 = self.residual_convs[i](x)

            b, l, f, n = x.shape
            if self.hgcn_true and hadp is not None:
                if self.adaptive_hyperhgnn == 'hgcn':
                    hypergraph_layer = HypergraphConvolution(in_channels=n, out_channels=n)
                    hypergraph_layer = hypergraph_layer.to(x.device)
                    x2 = hypergraph_layer(x, hadp)
                elif self.adaptive_hyperhgnn == 'hgat':
                    hypergraph_layer = HypergraphAttention(in_channels=n, out_channels=n)
                    hypergraph_layer = hypergraph_layer.to(x.device)
                    x2 = hypergraph_layer(x, hadp)
                elif self.adaptive_hyperhgnn == 'hsage':
                    hypergraph_layer = HypergraphSAGE(in_channels=n, out_channels=n)
                    hypergraph_layer = hypergraph_layer.to(x.device)
                    x2 = hypergraph_layer(x, hadp)
                else:
                    x2 = self.residual_convs[i](x)

                x = self.gamma * x1 + (1 - self.gamma) * x2.to(x1.device)
            else:
                x = x1

            x = x + residual[:, :, :, -x.size(3):]

            if idx is None:
                x = self.norm[i](x, self.idx.to(x.device))
            else:
                x = self.norm[i](x, idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x


# =============================================================================
# Lightweight Temporal Encoder (Alternative to LLM)
# =============================================================================

class LightweightTemporalEncoder(nn.Module):
    """Lightweight temporal encoder as alternative to LLM-based encoder."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes, seq_len, pred_len,
                 num_layers=2, dropout=0.1):
        super(LightweightTemporalEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Temporal encoding via transformer-like attention
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))

        # Multi-head self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * seq_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, T, N) or (B, T, N, F)
        Returns:
            Output tensor (B, pred_len, N)
        """
        if x.dim() == 4:
            B, T, N, F = x.shape
            x = x[..., 0]  # Take first feature
        else:
            B, T, N = x.shape

        # Process each node independently
        x = x.permute(0, 2, 1).contiguous()  # (B, N, T)
        x = x.view(B * N, T, 1)  # (B*N, T, 1)

        # Project to hidden dim
        x = self.input_projection(x)  # (B*N, T, hidden_dim)

        # Add positional encoding
        x = x + self.pos_encoding[:, :T, :]

        # Transformer encoding
        x = self.transformer_encoder(x)  # (B*N, T, hidden_dim)

        # Flatten and project to output
        x = x.view(B * N, -1)  # (B*N, T*hidden_dim)
        x = self.output_projection(x)  # (B*N, pred_len)

        # Reshape back
        x = x.view(B, N, self.pred_len)  # (B, N, pred_len)
        x = x.permute(0, 2, 1).contiguous()  # (B, pred_len, N)

        return x


# =============================================================================
# Main STHSepNet Model (LibCity Compatible)
# =============================================================================

class STHSepNet(AbstractTrafficStateModel):
    """
    Spatio-Temporal Hypergraph Separation Network for Traffic Flow Prediction.

    This model combines:
    - Temporal Module: Lightweight transformer or optional LLM-based encoder
    - Spatial Module: Adaptive hypergraph neural networks
    - Fusion: Learnable gating mechanisms

    LibCity-compatible implementation.
    """

    def __init__(self, config, data_feature):
        # Extract data features
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)

        super().__init__(config, data_feature)

        # Configuration
        self.device = config.get('device', torch.device('cpu'))
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)

        # Model hyperparameters
        self.hidden_dim = config.get('hidden_dim', 64)
        self.d_model = config.get('d_model', 32)
        self.d_ff = config.get('d_ff', 32)
        self.n_heads = config.get('n_heads', 4)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)

        # Hypergraph parameters
        self.gcn_true = config.get('gcn_true', True)
        self.hgcn_true = config.get('hgcn_true', True)
        self.hgat_true = config.get('hgat_true', False)
        self.buildA_true = config.get('buildA_true', True)
        self.buildH_true = config.get('buildH_true', True)
        self.adaptive_hyperhgnn = config.get('adaptive_hyperhgnn', 'hgcn')
        self.temporl_true = config.get('temporl_true', True)
        self.static = config.get('static', False)
        self.scale_hyperedges = config.get('scale_hyperedges', 10)

        # Fusion parameters
        self.alpha = config.get('alpha', 0.1)
        self.beta = config.get('beta', 0.3)
        self.gamma_init = config.get('gamma', 0.5)
        self.theta = config.get('theta', 0.2)
        self.fusion_gate = config.get('fusion_gate', 'adaptive')

        # Get adjacency matrix from data_feature
        adj_mx = data_feature.get('adj_mx', None)
        if adj_mx is not None:
            if isinstance(adj_mx, np.ndarray):
                self.adj_mx = torch.tensor(adj_mx, dtype=torch.float32)
            else:
                self.adj_mx = adj_mx.float()
        else:
            self.adj_mx = torch.zeros(self.num_nodes, self.num_nodes)

        # Scaler for inverse transform
        self._scaler = data_feature.get('scaler')
        self._logger = getLogger()

        # Build STHGNN module
        self.sthgnn = STHGNN(
            gcn_true=self.gcn_true,
            hgcn_true=self.hgcn_true,
            hgat_true=self.hgat_true,
            buildA_true=self.buildA_true,
            buildH_true=self.buildH_true,
            gcn_depth=2,
            num_nodes=self.num_nodes,
            num_hyperedges=self.num_nodes,
            device=self.device,
            adaptive_hyperhgnn=self.adaptive_hyperhgnn,
            temporl_true=self.temporl_true,
            static=self.static,
            predefined_A=self.adj_mx,
            static_feat=None,
            dropout=self.dropout,
            subgraph_size=min(20, self.num_nodes),
            node_dim=40,
            dilation_exponential=1,
            conv_channels=32,
            residual_channels=32,
            skip_channels=64,
            end_channels=128,
            seq_length=self.input_window,
            in_dim=self.feature_dim,
            out_dim=self.output_window,
            layers=3,
            propalpha=0.05,
            tanhalpha=3,
            layer_norm_affline=True,
            true_adj=self.adj_mx.numpy() if isinstance(self.adj_mx, torch.Tensor) else self.adj_mx,
            scale_hyperedges=self.scale_hyperedges,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma_init,
            theta=self.theta
        )

        # Lightweight temporal encoder (alternative to LLM)
        self.temporal_encoder = LightweightTemporalEncoder(
            input_dim=1,
            hidden_dim=self.hidden_dim,
            output_dim=self.d_ff,
            num_nodes=self.num_nodes,
            seq_len=self.input_window,
            pred_len=self.output_window,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

        # Fusion layers
        self.gate_linear = nn.Linear(self.num_nodes * 2, self.num_nodes)
        self.attention_gate = AttentionFusion(self.num_nodes)
        self.lstm_gate = LSTMGate(self.num_nodes)

        # Output projection
        self.output_projection = nn.Linear(self.num_nodes, self.num_nodes * self.output_dim)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, batch):
        """
        Forward pass of the model.

        Args:
            batch: Dictionary containing 'X' with shape (B, T, N, F)

        Returns:
            Predictions with shape (B, T_out, N, output_dim)
        """
        x = batch['X']  # (B, T, N, F)
        B, T, N, F = x.shape

        # Temporal encoding branch
        temporal_out = self.temporal_encoder(x)  # (B, pred_len, N)

        # Spatial encoding branch via STHGNN
        # STHGNN expects (B, F, N, T)
        x_sthgnn = x.permute(0, 3, 2, 1).contiguous()  # (B, F, N, T)
        sthgnn_out = self.sthgnn(x_sthgnn)  # (B, out_dim, N, 1)
        sthgnn_out = sthgnn_out.squeeze(-1)  # (B, out_dim, N)
        sthgnn_out = sthgnn_out.permute(0, 1, 2).contiguous()  # (B, pred_len, N)

        # Fusion of temporal and spatial outputs
        if self.fusion_gate == 'adaptive':
            # Concatenate and apply gating
            dec_out_expanded = temporal_out  # (B, pred_len, N)
            combined_input = torch.cat((dec_out_expanded, sthgnn_out), dim=2)  # (B, pred_len, 2N)
            gate_values = torch.sigmoid(self.gate_linear(combined_input))  # (B, pred_len, N)
            fused_output = gate_values * dec_out_expanded + (1 - gate_values) * sthgnn_out

        elif self.fusion_gate == 'hyperstgnn':
            # Only use STHGNN output
            fused_output = sthgnn_out

        elif self.fusion_gate == 'attentiongate':
            fused_output = self.attention_gate(temporal_out, sthgnn_out)

        elif self.fusion_gate == 'lstmgate':
            fused_output = self.lstm_gate(temporal_out, sthgnn_out)

        else:
            # Default: average fusion
            fused_output = (temporal_out + sthgnn_out) / 2

        # Output projection to match output_dim
        output = fused_output.unsqueeze(-1)  # (B, pred_len, N, 1)

        if self.output_dim > 1:
            output = output.expand(-1, -1, -1, self.output_dim)

        return output

    def predict(self, batch):
        """
        Generate predictions for a batch.

        Args:
            batch: Dictionary containing 'X' with shape (B, T, N, F)

        Returns:
            Predictions with shape (B, T_out, N, output_dim)
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate training loss.

        Args:
            batch: Dictionary containing 'X' and 'y' tensors

        Returns:
            Loss tensor
        """
        y_true = batch['y']
        y_predicted = self.predict(batch)

        # Inverse transform if scaler is available
        if self._scaler is not None:
            y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
            y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        return loss.masked_mae_torch(y_predicted, y_true, 0)
