# coding: utf-8
"""
GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation

This module adapts the GETNext model from:
    Yang et al. "GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation" (SIGIR 2022)

Original repository: https://github.com/songyangme/GETNext

Key adaptations made for LibCity:
1. Combined multiple sub-models (GCN, embeddings, transformer) into a single class
2. Adapted data handling to use LibCity's batch dictionary format
3. Implemented predict() and calculate_loss() methods following LibCity conventions
4. Added graph data loading from data_feature

Required data_feature keys:
- loc_size: Number of POI locations
- uid_size: Number of users
- tim_size: Number of time slots
- cat_size: Number of categories
- graph_A: Adjacency matrix for POI graph (optional, can be built from data)
- graph_X: Node features for POI graph (optional)
- poi_idx2cat_idx: POI index to category index mapping

Required config parameters:
- poi_embed_dim: POI embedding dimension (default: 128)
- user_embed_dim: User embedding dimension (default: 128)
- time_embed_dim: Time embedding dimension (default: 32)
- cat_embed_dim: Category embedding dimension (default: 32)
- gcn_nhid: Hidden dimensions for GCN layers (default: [32, 64])
- gcn_dropout: Dropout rate for GCN (default: 0.3)
- transformer_nhid: Hidden dim in TransformerEncoder (default: 1024)
- transformer_nlayers: Number of TransformerEncoderLayers (default: 2)
- transformer_nhead: Number of attention heads (default: 2)
- transformer_dropout: Dropout rate for transformer (default: 0.3)
- node_attn_nhid: Node attention hidden dimensions (default: 128)
- time_loss_weight: Weight for time prediction loss (default: 10)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pad_sequence
from libcity.model.abstract_model import AbstractModel


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    """Time2Vec transformation function."""
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)


class SineActivation(nn.Module):
    """Sine activation for Time2Vec."""
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        # tau: (batch, seq_len, 1) or (batch, 1)
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    """Cosine activation for Time2Vec."""
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    """Time2Vec encoding for temporal features."""
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        # x: (batch, seq_len) -> (batch, seq_len, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        return self.l1(x)


class GraphConvolution(nn.Module):
    """Graph Convolution Layer."""
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
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
        # Use sparse matrix multiplication if adj is sparse
        if adj.is_sparse:
            output = torch.sparse.mm(adj, support)
        else:
            output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    """Graph Convolutional Network for POI embedding."""
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()
        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)
        return x


class NodeAttnMap(nn.Module):
    """Node Attention Map for graph-based prediction adjustment."""
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap, self).__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        Wh = torch.mm(X, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))

        A = A + 1  # shift from 0-1 to 1-2
        e = e * A
        return e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class UserEmbeddings(nn.Module):
    """User embedding layer."""
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()
        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )

    def forward(self, user_idx):
        return self.user_embedding(user_idx)


class CategoryEmbeddings(nn.Module):
    """Category embedding layer."""
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()
        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
        )

    def forward(self, cat_idx):
        return self.cat_embedding(cat_idx)


class FuseEmbeddings(nn.Module):
    """Fuse two embeddings with a linear layer."""
    def __init__(self, embed_dim1, embed_dim2):
        super(FuseEmbeddings, self).__init__()
        embed_dim = embed_dim1 + embed_dim2
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, embed1, embed2):
        x = self.fuse_embed(torch.cat((embed1, embed2), dim=-1))
        x = self.leaky_relu(x)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, embed_dim)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerSeqModel(nn.Module):
    """Transformer-based sequence model for multi-task prediction."""
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerSeqModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_time = nn.Linear(embed_size, 1)
        self.decoder_cat = nn.Linear(embed_size, num_cat)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        # src: (batch, seq_len, embed_dim)
        src = src * math.sqrt(self.embed_size)
        # Transpose for positional encoding: (seq_len, batch, embed_dim)
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        # Transpose back: (batch, seq_len, embed_dim)
        src = src.transpose(0, 1)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_cat = self.decoder_cat(x)
        return out_poi, out_time, out_cat


class GETNext(AbstractModel):
    """
    GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation

    This model combines:
    1. GCN-based POI embeddings using trajectory flow graph
    2. Time2Vec for temporal encoding
    3. User and category embeddings
    4. Transformer encoder for sequence modeling
    5. Node attention map for graph-adjusted predictions

    The model performs multi-task learning:
    - Next POI prediction
    - Next visit time prediction
    - Next POI category prediction
    """

    def __init__(self, config, data_feature):
        super(GETNext, self).__init__(config, data_feature)

        self.device = config.get('device', 'cpu')

        # Data dimensions from data_feature
        self.num_poi = data_feature.get('loc_size', 1000)
        self.num_users = data_feature.get('uid_size', 100)
        self.num_cats = data_feature.get('cat_size', 50)
        self.tim_size = data_feature.get('tim_size', 48)

        # POI to category mapping
        self.poi_idx2cat_idx = data_feature.get('poi_idx2cat_idx', None)
        if self.poi_idx2cat_idx is None:
            # Create a dummy mapping if not provided
            self.poi_idx2cat_idx = {i: i % self.num_cats for i in range(self.num_poi)}

        # Model hyperparameters from config
        self.poi_embed_dim = config.get('poi_embed_dim', 128)
        self.user_embed_dim = config.get('user_embed_dim', 128)
        self.time_embed_dim = config.get('time_embed_dim', 32)
        self.cat_embed_dim = config.get('cat_embed_dim', 32)

        self.gcn_nhid = config.get('gcn_nhid', [32, 64])
        self.gcn_dropout = config.get('gcn_dropout', 0.3)

        self.transformer_nhid = config.get('transformer_nhid', 1024)
        self.transformer_nlayers = config.get('transformer_nlayers', 2)
        self.transformer_nhead = config.get('transformer_nhead', 2)
        self.transformer_dropout = config.get('transformer_dropout', 0.3)

        self.node_attn_nhid = config.get('node_attn_nhid', 128)
        self.time_loss_weight = config.get('time_loss_weight', 10.0)

        # Total embedding dimension
        self.seq_input_embed = (self.poi_embed_dim + self.user_embed_dim +
                                self.time_embed_dim + self.cat_embed_dim)

        # Load or initialize graph data
        self._init_graph_data(data_feature)

        # Build sub-models
        self._build_models()

    def _init_graph_data(self, data_feature):
        """Initialize graph adjacency matrix and node features."""
        # Get graph data from data_feature
        graph_A = data_feature.get('graph_A', None)
        graph_X = data_feature.get('graph_X', None)

        if graph_A is not None:
            if isinstance(graph_A, np.ndarray):
                graph_A = torch.from_numpy(graph_A).float()
            self.register_buffer('graph_A', graph_A)
        else:
            # Create identity matrix as default
            self.register_buffer('graph_A', torch.eye(self.num_poi))

        if graph_X is not None:
            if isinstance(graph_X, np.ndarray):
                graph_X = torch.from_numpy(graph_X).float()
            self.register_buffer('graph_X', graph_X)
            self.gcn_nfeat = graph_X.shape[1]
        else:
            # Create one-hot features as default
            self.register_buffer('graph_X', torch.eye(self.num_poi))
            self.gcn_nfeat = self.num_poi

    def _build_models(self):
        """Build all sub-models."""
        # GCN for POI embeddings
        self.poi_embed_model = GCN(
            ninput=self.gcn_nfeat,
            nhid=self.gcn_nhid,
            noutput=self.poi_embed_dim,
            dropout=self.gcn_dropout
        )

        # Node attention map
        self.node_attn_model = NodeAttnMap(
            in_features=self.gcn_nfeat,
            nhid=self.node_attn_nhid,
            use_mask=False
        )

        # User embeddings
        self.user_embed_model = UserEmbeddings(self.num_users, self.user_embed_dim)

        # Time embeddings (Time2Vec)
        self.time_embed_model = Time2Vec('sin', self.time_embed_dim)

        # Category embeddings
        self.cat_embed_model = CategoryEmbeddings(self.num_cats, self.cat_embed_dim)

        # Fusion models
        self.embed_fuse_model1 = FuseEmbeddings(self.user_embed_dim, self.poi_embed_dim)
        self.embed_fuse_model2 = FuseEmbeddings(self.time_embed_dim, self.cat_embed_dim)

        # Transformer sequence model
        self.seq_model = TransformerSeqModel(
            num_poi=self.num_poi,
            num_cat=self.num_cats,
            embed_size=self.seq_input_embed,
            nhead=self.transformer_nhead,
            nhid=self.transformer_nhid,
            nlayers=self.transformer_nlayers,
            dropout=self.transformer_dropout
        )

        # Loss functions
        self.criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)
        self.criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)

    def _masked_mse_loss(self, input, target, mask_value=-1):
        """Masked MSE loss for time prediction."""
        mask = target == mask_value
        out = (input[~mask] - target[~mask]) ** 2
        if out.numel() == 0:
            return torch.tensor(0.0, device=input.device)
        return out.mean()

    def _get_poi_embeddings(self):
        """Get POI embeddings from GCN."""
        return self.poi_embed_model(self.graph_X, self.graph_A)

    def _get_node_attn_map(self):
        """Get node attention map."""
        return self.node_attn_model(self.graph_X, self.graph_A)

    def _create_input_embeddings(self, batch):
        """
        Create input embeddings from batch data.

        Expected batch keys:
        - 'current_loc': (batch_size, seq_len) - POI indices
        - 'current_tim': (batch_size, seq_len) - time values (normalized)
        - 'uid': (batch_size,) or (batch_size, 1) - user indices

        Returns:
        - input_embed: (batch_size, seq_len, embed_dim)
        """
        current_loc = batch['current_loc']  # (batch_size, seq_len)
        current_tim = batch['current_tim']  # (batch_size, seq_len)

        # Handle user index
        if 'uid' in batch.data:
            uid = batch['uid']
            if uid.dim() == 1:
                uid = uid.unsqueeze(1)
            uid = uid.expand(-1, current_loc.size(1))  # (batch_size, seq_len)
        else:
            uid = torch.zeros(current_loc.size(0), current_loc.size(1),
                              dtype=torch.long, device=current_loc.device)

        batch_size, seq_len = current_loc.size()

        # Get POI embeddings from GCN
        poi_embeddings = self._get_poi_embeddings()  # (num_poi, poi_embed_dim)

        # Index POI embeddings: (batch_size, seq_len, poi_embed_dim)
        # Clamp indices to valid range
        current_loc_clamped = torch.clamp(current_loc, 0, self.num_poi - 1)
        poi_embed = poi_embeddings[current_loc_clamped.view(-1)].view(
            batch_size, seq_len, self.poi_embed_dim)

        # Get user embeddings: (batch_size, seq_len, user_embed_dim)
        uid_clamped = torch.clamp(uid, 0, self.num_users - 1)
        user_embed = self.user_embed_model(uid_clamped)

        # Get time embeddings: (batch_size, seq_len, time_embed_dim)
        time_embed = self.time_embed_model(current_tim.float())

        # Get category embeddings: (batch_size, seq_len, cat_embed_dim)
        # Map POI indices to category indices
        cat_indices = torch.zeros_like(current_loc)
        for i in range(batch_size):
            for j in range(seq_len):
                poi_idx = current_loc[i, j].item()
                cat_indices[i, j] = self.poi_idx2cat_idx.get(poi_idx, 0)
        cat_indices = torch.clamp(cat_indices, 0, self.num_cats - 1)
        cat_embed = self.cat_embed_model(cat_indices)

        # Fuse embeddings
        # Fuse user + poi
        fused1 = self.embed_fuse_model1(user_embed, poi_embed)
        # Fuse time + cat
        fused2 = self.embed_fuse_model2(time_embed, cat_embed)

        # Concatenate fused embeddings
        input_embed = torch.cat([fused1, fused2], dim=-1)

        return input_embed

    def _adjust_pred_by_graph(self, y_pred_poi, batch):
        """Adjust POI prediction probabilities using node attention map."""
        attn_map = self._get_node_attn_map()  # (num_poi, num_poi)
        current_loc = batch['current_loc']  # (batch_size, seq_len)

        batch_size, seq_len, num_poi = y_pred_poi.shape
        y_pred_adjusted = torch.zeros_like(y_pred_poi)

        for i in range(batch_size):
            for j in range(seq_len):
                poi_idx = current_loc[i, j].item()
                if 0 <= poi_idx < self.num_poi:
                    y_pred_adjusted[i, j, :] = attn_map[poi_idx, :] + y_pred_poi[i, j, :]
                else:
                    y_pred_adjusted[i, j, :] = y_pred_poi[i, j, :]

        return y_pred_adjusted

    def forward(self, batch):
        """
        Forward pass.

        Args:
            batch: Dictionary containing:
                - 'current_loc': (batch_size, seq_len) - POI indices
                - 'current_tim': (batch_size, seq_len) - time values
                - 'uid': (batch_size,) - user indices

        Returns:
            - out_poi: (batch_size, seq_len, num_poi) - POI prediction logits
            - out_time: (batch_size, seq_len, 1) - time predictions
            - out_cat: (batch_size, seq_len, num_cats) - category prediction logits
        """
        # Create input embeddings
        input_embed = self._create_input_embeddings(batch)
        batch_size, seq_len, _ = input_embed.shape

        # Generate attention mask
        src_mask = self.seq_model.generate_square_subsequent_mask(seq_len, input_embed.device)

        # Forward through transformer
        out_poi, out_time, out_cat = self.seq_model(input_embed, src_mask)

        # Adjust POI predictions using graph attention
        out_poi_adjusted = self._adjust_pred_by_graph(out_poi, batch)

        return out_poi_adjusted, out_time, out_cat

    def predict(self, batch):
        """
        Predict next POI for each position in the sequence.

        Args:
            batch: Dictionary containing input data

        Returns:
            torch.Tensor: POI prediction scores (batch_size, num_poi)
                          Returns predictions for the last valid position of each sequence.
        """
        out_poi, out_time, out_cat = self.forward(batch)

        # Get predictions for the last position in each sequence
        if 'current_loc' in batch.data:
            # If we have sequence length info, use last position
            # out_poi: (batch_size, seq_len, num_poi)
            last_pred = out_poi[:, -1, :]  # (batch_size, num_poi)
        else:
            last_pred = out_poi[:, -1, :]

        return last_pred

    def calculate_loss(self, batch):
        """
        Calculate multi-task loss.

        Args:
            batch: Dictionary containing:
                - 'current_loc': (batch_size, seq_len) - input POI indices
                - 'current_tim': (batch_size, seq_len) - input time values
                - 'uid': (batch_size,) - user indices
                - 'target': (batch_size,) or (batch_size, seq_len) - target POI indices
                - 'target_tim': (batch_size,) or (batch_size, seq_len) - target time values (optional)
                - 'target_cat': (batch_size,) or (batch_size, seq_len) - target category indices (optional)

        Returns:
            torch.Tensor: Total loss (scalar)
        """
        out_poi, out_time, out_cat = self.forward(batch)
        batch_size, seq_len, _ = out_poi.shape

        # Get target labels
        target = batch['target']

        # Handle single target (next POI prediction)
        if target.dim() == 1:
            # Only use last position prediction
            out_poi_last = out_poi[:, -1, :]  # (batch_size, num_poi)
            out_cat_last = out_cat[:, -1, :]  # (batch_size, num_cats)
            out_time_last = out_time[:, -1, :].squeeze(-1)  # (batch_size,)

            loss_poi = self.criterion_poi(out_poi_last, target)

            # Time loss
            if 'target_tim' in batch.data:
                target_tim = batch['target_tim']
                loss_time = self._masked_mse_loss(out_time_last, target_tim.float())
            else:
                loss_time = torch.tensor(0.0, device=self.device)

            # Category loss
            if 'target_cat' in batch.data:
                target_cat = batch['target_cat']
                loss_cat = self.criterion_cat(out_cat_last, target_cat)
            else:
                # Compute category from target POI
                target_cat = torch.tensor([self.poi_idx2cat_idx.get(t.item(), 0)
                                           for t in target], device=target.device)
                loss_cat = self.criterion_cat(out_cat_last, target_cat)

        else:
            # Sequence target (for sequence-to-sequence prediction)
            # out_poi: (batch_size, seq_len, num_poi)
            # target: (batch_size, seq_len)
            loss_poi = self.criterion_poi(out_poi.transpose(1, 2), target)

            # Time loss
            if 'target_tim' in batch.data:
                target_tim = batch['target_tim']
                loss_time = self._masked_mse_loss(out_time.squeeze(-1), target_tim.float())
            else:
                loss_time = torch.tensor(0.0, device=self.device)

            # Category loss
            if 'target_cat' in batch.data:
                target_cat = batch['target_cat']
                loss_cat = self.criterion_cat(out_cat.transpose(1, 2), target_cat)
            else:
                loss_cat = torch.tensor(0.0, device=self.device)

        # Combined loss
        total_loss = loss_poi + self.time_loss_weight * loss_time + loss_cat

        return total_loss
