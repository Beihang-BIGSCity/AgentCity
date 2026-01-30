"""
LoTNext: Long-tail Next POI Prediction

This module implements the LoTNext model adapted from the original repository
to the LibCity framework for trajectory location prediction.

Original Paper: LoTNext: Long-tail Next POI Prediction
Key Components:
- Flashback RNN with spatial/temporal weighting
- Time2Vec temporal encoding
- Transformer-based sequence encoder
- Denoising GCN for user-POI interactions
- Multi-task learning (location + time slot prediction)

Adaptations for LibCity:
- Inherits from AbstractModel
- Uses config.get() for parameters
- Adapts to LibCity's batch format
- Implements predict() and calculate_loss() methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from enum import Enum
import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, identity

from libcity.model.abstract_model import AbstractModel

try:
    from torch_geometric.nn import GCNConv
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    # Fallback: Define a simple GCN layer without torch_geometric
    class GCNConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(GCNConv, self).__init__()
            self.linear = nn.Linear(in_channels, out_channels)

        def forward(self, x, edge_index):
            # Simple message passing without torch_geometric
            return self.linear(x)


# ============================================================================
# Utility Functions
# ============================================================================

def sparse_matrix_to_tensor(graph):
    """Convert scipy sparse matrix to PyTorch sparse tensor."""
    graph = coo_matrix(graph)
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = graph.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape))


def calculate_random_walk_matrix(adj_mx):
    """Calculate the random walk normalized adjacency matrix: D^-1 * W."""
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def haversine(s1, s2):
    """
    Calculate the Haversine distance between two batches of geographic locations.

    Args:
        s1, s2: Tensors of shape [batch_size, 2] containing latitude and longitude.

    Returns:
        Tensor of shape [batch_size] containing distances in kilometers.
    """
    s1 = s1 * math.pi / 180
    s2 = s2 * math.pi / 180

    lat1, lon1 = s1[:, 0], s1[:, 1]
    lat2, lon2 = s2[:, 0], s2[:, 1]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(a.clamp(min=1e-10)))
    r = 6371  # Earth's radius in kilometers
    return c * r


def haversines(s1, s2):
    """
    Calculate pairwise Haversine distances for sequence positions.

    Args:
        s1, s2: Tensors of shape [batch_size, seq_len, 2].

    Returns:
        Tensor of shape [batch_size, seq_len, seq_len] containing distances.
    """
    s1 = s1 * math.pi / 180
    s2 = s2 * math.pi / 180

    s1_expanded = s1.unsqueeze(2)
    s2_expanded = s2.unsqueeze(1)

    dlat = s2_expanded[..., 0] - s1_expanded[..., 0]
    dlon = s2_expanded[..., 1] - s1_expanded[..., 1]

    a = torch.sin(dlat / 2)**2 + torch.cos(s1_expanded[..., 0]) * torch.cos(s2_expanded[..., 0]) * torch.sin(dlon / 2)**2
    c = 2 * torch.asin(torch.sqrt(a.clamp(min=1e-10)))
    r = 6371
    return c * r


def mask_value(epoch, T, v_min=-100, v_max=-1e9):
    """Compute dynamic mask value for attention based on training progress."""
    if T == 0:
        return v_min
    return -10 ** (np.log10(-v_min) + (np.log10(-v_max) - np.log10(-v_min)) * (epoch / T))


# ============================================================================
# RNN Factory
# ============================================================================

class Rnn(Enum):
    """Enumeration of available RNN types."""
    RNN = 0
    GRU = 1
    LSTM = 2

    @staticmethod
    def from_string(name):
        name = name.lower()
        if name == 'rnn':
            return Rnn.RNN
        if name == 'gru':
            return Rnn.GRU
        if name == 'lstm':
            return Rnn.LSTM
        raise ValueError('{} not supported in rnn_type'.format(name))


class RnnFactory:
    """Factory class to create the desired RNN unit."""

    def __init__(self, rnn_type_str):
        self.rnn_type = Rnn.from_string(rnn_type_str)

    def is_lstm(self):
        return self.rnn_type == Rnn.LSTM

    def create(self, hidden_size):
        if self.rnn_type == Rnn.RNN:
            return nn.RNN(hidden_size, hidden_size)
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size)


# ============================================================================
# Transformer Components
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeedForwardNetwork(nn.Module):
    """Feed-forward network for transformer encoder."""

    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional masking."""

    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None, current_epoch=0, mask=True):
        orig_q_size = q.size()
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)

        q = q * self.scale
        x = torch.matmul(q, k)

        if attn_bias is not None:
            x = x + attn_bias

        if mask is not None:
            seq_len = x.size(-1)
            mask_tensor = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(1)
            mask_tensor = mask_tensor.expand(batch_size, 1, seq_len, seq_len).to(x.device)
            x = x.masked_fill(mask_tensor, mask_value(current_epoch, 100))

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * d_v)
        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention and FFN."""

    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm1 = nn.LayerNorm(hidden_size)
        self.ffn_norm2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, epoch=0, mask=None):
        y = self.self_attention(x, x, x, attn_bias, epoch, mask=1)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm1(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        x = self.ffn_norm2(x)
        return x


class TransformerModel(nn.Module):
    """Full transformer encoder model."""

    def __init__(self, embed_size, nhead, nhid, nlayers, max_len, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, max_len, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        return x


# ============================================================================
# Time2Vec Components
# ============================================================================

def t2v(tau, f, out_features, w, b, w0, b0):
    """Time2Vec transformation."""
    v1 = f(tau.unsqueeze(-1) * w + b)
    v2 = tau.unsqueeze(-1) * w0 + b0
    return torch.cat([v1, v2], -1)


class SineActivation1(nn.Module):
    """Sine-based Time2Vec activation."""

    def __init__(self, seq_len, out_features):
        super(SineActivation1, self).__init__()
        self.out_features = out_features
        self.l1 = nn.Linear(1, out_features - 1, bias=True)
        self.l2 = nn.Linear(1, 1, bias=True)
        self.f = torch.sin

    def forward(self, tau):
        v1 = self.l1(tau.unsqueeze(-1))
        v1 = self.f(v1)
        v2 = self.l2(tau.unsqueeze(-1))
        return torch.cat([v1, v2], -1)


class CosineActivation(nn.Module):
    """Cosine-based Time2Vec activation."""

    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(in_features, 1))
        self.w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    """Time2Vec temporal encoding module."""

    def __init__(self, activation, batch_size, seq_len, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation1(seq_len, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(seq_len, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x


# ============================================================================
# Embedding Fusion
# ============================================================================

class FuseEmbeddings(nn.Module):
    """Fuse location and temporal embeddings."""

    def __init__(self, loc_embed_dim, time_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = loc_embed_dim + time_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, loc_embed, time_embed):
        x = self.fuse_embed(torch.cat((loc_embed, time_embed), 2))
        x = self.leaky_relu(x)
        return x


# ============================================================================
# GCN Components for Denoising
# ============================================================================

class AttentionLayer(nn.Module):
    """Attention layer for computing edge weights in user-POI graph."""

    def __init__(self, user_dim, item_dim):
        super(AttentionLayer, self).__init__()
        self.attention_fc = nn.Sequential(
            nn.Linear(user_dim + item_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )
        for layer in self.attention_fc:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)

    def forward(self, user_embeddings, item_embeddings, edge_index):
        user_indices = edge_index[0]
        item_indices = edge_index[1]
        user_feats = user_embeddings[user_indices]
        item_feats = item_embeddings[item_indices]

        edge_feats = torch.cat([user_feats, item_feats], dim=1)
        edge_weights = torch.sigmoid(self.attention_fc(edge_feats)).squeeze()
        return edge_weights


class DenoisingLayer(nn.Module):
    """Denoising layer for filtering graph edges."""

    def __init__(self):
        super(DenoisingLayer, self).__init__()

    def forward(self, edge_weights, edge_index, threshold=0.8):
        mask = edge_weights > threshold
        if mask.sum() == 0:
            mask[edge_weights.argmax()] = True
        denoised_edge_index = edge_index[:, mask]
        denoised_edge_weights = edge_weights[mask]
        return denoised_edge_index, denoised_edge_weights


class GCNLayer(nn.Module):
    """Graph convolutional layer."""

    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        if HAS_TORCH_GEOMETRIC:
            self.conv1 = GCNConv(in_channels, out_channels)
        else:
            self.conv1 = nn.Linear(in_channels, out_channels)
            self._use_fallback = True

    def forward(self, x, edge_index):
        if hasattr(self, '_use_fallback') and self._use_fallback:
            return self.conv1(x)
        return self.conv1(x, edge_index)


class DenoisingGCNNet(nn.Module):
    """Complete denoising GCN network for user-POI graph."""

    def __init__(self, user_dim, item_dim, out_channels):
        super(DenoisingGCNNet, self).__init__()
        self.attention_layer = AttentionLayer(user_dim, item_dim)
        self.denoising_layer = DenoisingLayer()
        self.gcn_layer = GCNLayer(user_dim, out_channels)

    def forward(self, user_embeddings, item_embeddings, edge_index):
        edge_weights = self.attention_layer(user_embeddings, item_embeddings, edge_index)
        denoised_edge_index, denoised_edge_weights = self.denoising_layer(edge_weights, edge_index)
        gcn_input = torch.cat([user_embeddings, item_embeddings], dim=0)
        gcn_output = self.gcn_layer(gcn_input, denoised_edge_index)
        return gcn_output, denoised_edge_index, denoised_edge_weights


class GraphConvolution(nn.Module):
    """Standard graph convolution layer."""

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output


# ============================================================================
# Hidden State Initialization Strategies
# ============================================================================

class H0Strategy:
    """Base class for hidden state initialization."""

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def on_init(self, user_len, device):
        pass

    def on_reset(self, user):
        pass

    def on_reset_test(self, user, device):
        return self.on_reset(user)


class FixNoiseStrategy(H0Strategy):
    """Use fixed normal noise as initialization."""

    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        mu = 0
        sd = 1 / self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu

    def on_init(self, user_len, device):
        hs = [self.h0 for _ in range(user_len)]
        return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(device)

    def on_reset(self, user):
        return self.h0


class LstmStrategy(H0Strategy):
    """Creates h0 and c0 using inner strategies for LSTM."""

    def __init__(self, hidden_size, h_strategy, c_strategy):
        super().__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy

    def on_init(self, user_len, device):
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return h, c

    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return h, c


def create_h0_strategy(hidden_size, is_lstm):
    """Factory function for hidden state strategies."""
    if is_lstm:
        return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))
    return FixNoiseStrategy(hidden_size)


# ============================================================================
# Main LoTNext Model
# ============================================================================

class LoTNext(AbstractModel):
    """
    LoTNext: Long-tail Next POI Prediction Model

    This model combines:
    - RNN-based sequence encoding
    - Transformer with Time2Vec temporal encoding
    - Denoising GCN for user-POI interactions
    - Spatial and temporal attention weighting
    - Multi-task learning (location + time prediction)

    Args:
        config: LibCity configuration dictionary
        data_feature: Data feature dictionary containing loc_size, uid_size, etc.
    """

    def __init__(self, config, data_feature):
        super(LoTNext, self).__init__(config, data_feature)

        # Extract data features
        self.loc_size = data_feature.get('loc_size', 1000)
        self.uid_size = data_feature.get('uid_size', 100)
        self.loc_pad = data_feature.get('loc_pad', 0)
        self.uid_pad = data_feature.get('uid_pad', 0)

        # Model hyperparameters from config
        self.hidden_size = config.get('hidden_size', 128)
        self.loc_emb_size = config.get('loc_emb_size', 128)
        self.time_emb_size = config.get('time_emb_size', 6)
        self.rnn_type = config.get('rnn_type', 'LSTM')
        self.device = config.get('device', 'cpu')
        self.batch_size = config.get('batch_size', 32)
        self.sequence_length = config.get('sequence_length', 20)

        # Transformer parameters
        self.transformer_nhid = config.get('transformer_nhid', 256)
        self.transformer_nhead = config.get('transformer_nhead', 4)
        self.transformer_dropout = config.get('transformer_dropout', 0.1)
        self.attention_dropout_rate = config.get('attention_dropout_rate', 0.1)

        # Graph parameters
        self.lambda_loc = config.get('lambda_loc', 1.0)
        self.lambda_user = config.get('lambda_user', 1.0)
        self.use_graph_user = config.get('use_graph_user', False)
        self.use_spatial_graph = config.get('use_spatial_graph', False)

        # Loss weights
        self.loc_loss_weight = config.get('loc_loss_weight', 1.0)
        self.time_loss_weight = config.get('time_loss_weight', 0.1)

        # Time slot parameters (e.g., 24 hours * 7 days = 168 slots)
        self.num_time_slots = config.get('num_time_slots', 168)

        # Initialize embeddings
        self.encoder = nn.Embedding(self.loc_size, self.hidden_size, padding_idx=self.loc_pad)
        self.user_encoder = nn.Embedding(self.uid_size, self.hidden_size, padding_idx=self.uid_pad)

        # Initialize RNN
        self.rnn_factory = RnnFactory(self.rnn_type)
        self.rnn = self.rnn_factory.create(self.hidden_size)

        # Transformer encoder layer
        self.seq_model = EncoderLayer(
            self.hidden_size + self.time_emb_size,
            self.transformer_nhid,
            self.transformer_dropout,
            self.attention_dropout_rate,
            self.transformer_nhead
        )

        # Time2Vec temporal encoding
        self.time_embed_model = Time2Vec('sin', self.batch_size, self.sequence_length, out_dim=self.time_emb_size)

        # Embedding fusion
        self.embed_fuse_model = FuseEmbeddings(self.hidden_size, self.time_emb_size)

        # Decoders
        self.decoder = nn.Linear(self.hidden_size + self.time_emb_size, self.hidden_size)
        self.fc = nn.Linear(2 * self.hidden_size, self.loc_size)
        self.time_decoder = nn.Linear(self.hidden_size + self.time_emb_size, 1)

        # Denoising GCN
        self.denoise = DenoisingGCNNet(self.hidden_size, self.hidden_size, self.hidden_size)

        # Spatial decay parameter
        self.spatial_decay = config.get('spatial_decay', 100.0)

        # Graph matrices (initialized as None, can be set later)
        self.graph = None
        self.spatial_graph = None
        self.interact_graph = None

        # Initialize hidden state strategy
        self.h0_strategy = create_h0_strategy(self.hidden_size, self.rnn_factory.is_lstm())

        # Training epoch counter
        self.current_epoch = 0

    def set_graphs(self, transition_graph=None, spatial_graph=None, interact_graph=None):
        """
        Set graph matrices for the model.

        Args:
            transition_graph: POI transition graph (scipy sparse matrix)
            spatial_graph: Spatial proximity graph (scipy sparse matrix)
            interact_graph: User-POI interaction graph (scipy sparse matrix)
        """
        if transition_graph is not None:
            I = identity(transition_graph.shape[0], format='coo')
            self.graph = sparse_matrix_to_tensor(
                calculate_random_walk_matrix((transition_graph * self.lambda_loc + I).astype(np.float32))
            )

        if spatial_graph is not None:
            self.spatial_graph = spatial_graph

        if interact_graph is not None:
            self.interact_graph = sparse_matrix_to_tensor(
                calculate_random_walk_matrix(interact_graph)
            )

    def _compute_spatial_weight(self, distance, user_len):
        """Compute spatial weight using exponential decay."""
        return torch.exp(-distance / self.spatial_decay)

    def forward(self, batch):
        """
        Forward pass of the LoTNext model.

        Args:
            batch: LibCity batch dictionary containing:
                - 'current_loc': Current trajectory locations [batch_size, seq_len]
                - 'current_tim': Current trajectory timestamps [batch_size, seq_len]
                - 'uid': User IDs [batch_size]
                - 'current_coord': Coordinates (optional) [batch_size, seq_len, 2]
                - 'target': Target location [batch_size]
                - 'target_tim': Target time slot (optional) [batch_size]

        Returns:
            y_linear: Location prediction logits [batch_size, loc_size]
            out_time: Time prediction (optional)
            cosine_similarity: Cosine similarity scores (optional)
        """
        # Extract batch data
        if hasattr(batch, '__getitem__'):
            loc = batch.get('current_loc', batch.get('X', None))
            tim = batch.get('current_tim', None)
            uid = batch.get('uid', None)
            coord = batch.get('current_coord', None)
        else:
            loc = getattr(batch, 'current_loc', getattr(batch, 'X', None))
            tim = getattr(batch, 'current_tim', None)
            uid = getattr(batch, 'uid', None)
            coord = getattr(batch, 'current_coord', None)

        if loc is None:
            raise ValueError("Batch must contain 'current_loc' or 'X' field")

        # Handle dimensions
        if loc.dim() == 2:
            batch_size, seq_len = loc.size()
        else:
            batch_size = loc.size(0)
            seq_len = loc.size(1) if loc.dim() > 1 else 1
            loc = loc.view(batch_size, seq_len)

        device = loc.device

        # Get location embeddings
        x_emb = self.encoder(loc)  # [batch_size, seq_len, hidden_size]

        # Get user embeddings if available
        if uid is not None:
            if uid.dim() == 1:
                p_u = self.user_encoder(uid)  # [batch_size, hidden_size]
            else:
                p_u = self.user_encoder(uid.squeeze())
        else:
            p_u = torch.zeros(batch_size, self.hidden_size, device=device)

        # Apply GCN-enhanced embeddings if graph is available
        if self.graph is not None:
            graph = self.graph.to(device)
            loc_emb_all = self.encoder(torch.arange(self.loc_size, device=device))
            encoder_weight = torch.sparse.mm(graph, loc_emb_all)

            # Update location embeddings using graph
            new_x_emb = []
            for i in range(seq_len):
                temp_x = torch.index_select(encoder_weight, 0, loc[:, i])
                new_x_emb.append(temp_x)
            x_emb = torch.stack(new_x_emb, dim=1)

        # Apply denoising GCN if interaction graph is available
        if self.interact_graph is not None:
            interact_graph = self.interact_graph.to(device)

            loc_emb_all = self.encoder(torch.arange(self.loc_size, device=device))
            user_emb_all = self.user_encoder(torch.arange(
                min(self.uid_size, interact_graph.size(0)), device=device))

            encoder_weight_user = torch.sparse.mm(interact_graph, loc_emb_all)
            encoder_weight_poi = torch.sparse.mm(interact_graph.t(), user_emb_all)

            edge_index = interact_graph.coalesce().indices()

            gcn_output, _, _ = self.denoise(encoder_weight_user, encoder_weight_poi, edge_index)
            encoder_weight_poi = gcn_output[interact_graph.size(0):]

            new_x_emb = []
            for i in range(seq_len):
                loc_indices = loc[:, i].clamp(0, encoder_weight_poi.size(0) - 1)
                temp_x = torch.index_select(encoder_weight_poi, 0, loc_indices)
                new_x_emb.append(temp_x)
            x_emb_new = torch.stack(new_x_emb, dim=1)
            x_emb = (x_emb + x_emb_new) / 2

        # Time embedding
        if tim is not None:
            t_slot = tim.float() / self.num_time_slots
            t_emb = self.time_embed_model(t_slot)
            x_emb_fused = self.embed_fuse_model(x_emb, t_emb)
        else:
            t_emb = torch.zeros(batch_size, seq_len, self.time_emb_size, device=device)
            x_emb_fused = self.embed_fuse_model(x_emb, t_emb)

        # Transformer encoding
        out = self.seq_model(x_emb_fused, epoch=self.current_epoch)

        # Time prediction
        out_time = self.time_decoder(out)  # [batch_size, seq_len, 1]

        # Location decoding
        out_decoded = self.decoder(out)  # [batch_size, seq_len, hidden_size]

        # Spatial attention weighting
        if coord is not None:
            out_w = torch.zeros(batch_size, seq_len, self.hidden_size, device=device)
            for i in range(seq_len):
                sum_w = torch.zeros(batch_size, 1, device=device)
                for j in range(i + 1):
                    dist_s = haversine(coord[:, i], coord[:, j])
                    b_j = self._compute_spatial_weight(dist_s, batch_size).unsqueeze(1)
                    w_j = b_j + 1e-10
                    sum_w += w_j
                    out_w[:, i] += w_j * out_decoded[:, j]
                out_w[:, i] /= sum_w
        else:
            out_w = out_decoded

        # Get final output (last position)
        out_final = out_w[:, -1, :]  # [batch_size, hidden_size]

        # Concatenate with user embedding
        out_pu = torch.cat([out_final, p_u], dim=1)  # [batch_size, 2*hidden_size]

        # Compute cosine similarity
        cosine_similarity = F.linear(F.normalize(out_pu), F.normalize(self.fc.weight))

        # Final location prediction
        y_linear = self.fc(out_pu)  # [batch_size, loc_size]

        # Final time prediction (last position)
        out_time_final = out_time[:, -1, :]  # [batch_size, 1]

        return y_linear, out_time_final, cosine_similarity

    def predict(self, batch):
        """
        Predict the next POI location.

        Args:
            batch: LibCity batch dictionary

        Returns:
            score: Log-softmax scores for each location [batch_size, loc_size]
        """
        y_linear, _, _ = self.forward(batch)
        score = F.log_softmax(y_linear, dim=1)
        return score

    def calculate_loss(self, batch):
        """
        Calculate the multi-task loss (location + time prediction).

        Args:
            batch: LibCity batch dictionary containing:
                - 'target': Target location
                - 'target_tim': Target time slot (optional)

        Returns:
            loss: Combined loss tensor
        """
        y_linear, out_time, _ = self.forward(batch)

        # Location prediction loss
        if hasattr(batch, '__getitem__'):
            target = batch.get('target', batch.get('y', None))
            target_tim = batch.get('target_tim', None)
        else:
            target = getattr(batch, 'target', getattr(batch, 'y', None))
            target_tim = getattr(batch, 'target_tim', None)

        if target is None:
            raise ValueError("Batch must contain 'target' or 'y' field")

        # Handle target shape
        if target.dim() > 1:
            target = target.squeeze()

        # Cross-entropy loss for location prediction
        loc_criterion = nn.CrossEntropyLoss().to(self.device)
        loc_loss = loc_criterion(y_linear, target)

        total_loss = self.loc_loss_weight * loc_loss

        # Time prediction loss (MSE)
        if target_tim is not None and out_time is not None:
            if target_tim.dim() > 1:
                target_tim = target_tim.squeeze()
            time_criterion = nn.MSELoss().to(self.device)
            time_loss = time_criterion(out_time.squeeze(), target_tim.float())
            total_loss += self.time_loss_weight * time_loss

        return total_loss

    def set_epoch(self, epoch):
        """Set the current training epoch for dynamic masking."""
        self.current_epoch = epoch
