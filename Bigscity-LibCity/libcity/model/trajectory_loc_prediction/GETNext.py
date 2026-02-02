"""
GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation

This model is adapted from the original GETNext implementation:
https://github.com/songyangme/GETNext

Original paper:
"Graph-Enhanced Transformer for Next POI Recommendation" (SIGIR 2022)

Key Components:
1. GCN - POI embedding via graph convolution
2. UserEmbeddings - User representation
3. CategoryEmbeddings - POI category representation
4. Time2Vec - Time encoding using sine activation
5. FuseEmbeddings - Fuses user+POI and time+category embeddings
6. TransformerModel - Main sequence model
7. NodeAttnMap - Trajectory flow map attention mechanism

Adaptations for LibCity:
- Unified all 8 sub-models into a single GETNext class
- Adapted batch input format to LibCity's trajectory batch dictionary
- Integrated graph adjacency matrix and node features from data_feature
- Implemented predict() and calculate_loss() methods following LibCity conventions
- Multi-task loss: POI prediction (cross-entropy) + time prediction (MSE) + category prediction (cross-entropy)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils.rnn import pad_sequence

from libcity.model.abstract_model import AbstractModel


class GraphConvolution(nn.Module):
    """Graph Convolution Layer for POI embeddings."""

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
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    """Graph Convolutional Network for learning POI embeddings from trajectory flow graph."""

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
    """Node Attention Map for trajectory flow map attention mechanism."""

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
        embed = self.user_embedding(user_idx)
        return embed


class CategoryEmbeddings(nn.Module):
    """Category embedding layer."""

    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()
        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
        )

    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed


class FuseEmbeddings(nn.Module):
    """Fuse two embeddings (e.g., user+POI or time+category)."""

    def __init__(self, embed_dim1, embed_dim2):
        super(FuseEmbeddings, self).__init__()
        embed_dim = embed_dim1 + embed_dim2
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, embed1, embed2):
        x = self.fuse_embed(torch.cat((embed1, embed2), -1))
        x = self.leaky_relu(x)
        return x


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
    """Time2Vec: Learnable time representation."""

    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x


class PositionalEncoding(nn.Module):
    """Standard positional encoding for Transformer."""

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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder for sequence modeling with multi-task output."""

    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerEncoder, self).__init__()
        from torch.nn import TransformerEncoder as TorchTransformerEncoder
        from torch.nn import TransformerEncoderLayer

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TorchTransformerEncoder(encoder_layers, nlayers)
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
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_cat = self.decoder_cat(x)
        return out_poi, out_time, out_cat


class GETNext(AbstractModel):
    """
    GETNext: Graph-Enhanced Transformer for Next POI Recommendation

    This model combines graph neural networks with transformers for trajectory-based
    POI recommendation. It uses a trajectory flow map to capture transition patterns
    between POIs and applies attention mechanism to adjust predictions.

    Args:
        config (dict): Configuration dictionary containing model hyperparameters
        data_feature (dict): Data features including vocab sizes, graph data, etc.

    Required config parameters:
        - poi_embed_dim: POI embedding dimension (default: 128)
        - user_embed_dim: User embedding dimension (default: 128)
        - time_embed_dim: Time embedding dimension (default: 32)
        - cat_embed_dim: Category embedding dimension (default: 32)
        - gcn_nhid: List of hidden dims for GCN layers (default: [32, 64])
        - gcn_dropout: Dropout rate for GCN (default: 0.3)
        - transformer_nhid: Hidden dim in TransformerEncoder (default: 1024)
        - transformer_nlayers: Number of TransformerEncoderLayers (default: 2)
        - transformer_nhead: Number of attention heads (default: 2)
        - transformer_dropout: Dropout rate for transformer (default: 0.3)
        - node_attn_nhid: Node attention map hidden dimension (default: 128)
        - time_loss_weight: Weight for time prediction loss (default: 10)
        - time_activation: Time2Vec activation function (default: "sin")
        - use_graph_attn: Whether to use graph attention adjustment (default: True)

    Required data_feature:
        - loc_size: Number of POI locations
        - uid_size: Number of users
        - tim_size: Number of time slots (for category, use separately if available)
        - num_categories: Number of POI categories (optional, defaults to tim_size)
        - adj_matrix: Adjacency matrix of POI graph (optional)
        - node_features: Node feature matrix (optional)
        - poi_to_cat: Mapping from POI index to category index (optional)
    """

    def __init__(self, config, data_feature):
        super(GETNext, self).__init__(config, data_feature)

        self.device = config.get('device', 'cpu')

        # Data dimensions from data_feature
        self.num_pois = data_feature.get('loc_size', 1000)
        self.num_users = data_feature.get('uid_size', 100)
        self.num_cats = data_feature.get('num_categories', data_feature.get('tim_size', 48))

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
        self.time_activation = config.get('time_activation', 'sin')
        self.use_graph_attn = config.get('use_graph_attn', True)

        # Graph data from data_feature (can be None if not provided)
        self.adj_matrix = data_feature.get('adj_matrix', None)
        self.node_features = data_feature.get('node_features', None)
        self.poi_to_cat = data_feature.get('poi_to_cat', None)

        # Initialize graph data if provided
        if self.adj_matrix is not None:
            if isinstance(self.adj_matrix, torch.Tensor):
                self.register_buffer('A', self.adj_matrix.float())
            else:
                self.register_buffer('A', torch.tensor(self.adj_matrix, dtype=torch.float32))
        else:
            # Create identity matrix as default adjacency
            self.register_buffer('A', torch.eye(self.num_pois, dtype=torch.float32))

        if self.node_features is not None:
            if isinstance(self.node_features, torch.Tensor):
                self.register_buffer('X', self.node_features.float())
            else:
                self.register_buffer('X', torch.tensor(self.node_features, dtype=torch.float32))
            self.gcn_nfeat = self.X.shape[1]
        else:
            # Create simple one-hot features as default
            self.register_buffer('X', torch.eye(self.num_pois, dtype=torch.float32))
            self.gcn_nfeat = self.num_pois

        # Build POI to category mapping if not provided
        if self.poi_to_cat is None:
            # Default: each POI maps to category 0
            self.poi_to_cat = {i: 0 for i in range(self.num_pois)}

        # Initialize all model components
        self._build_model()

    def _build_model(self):
        """Build all model components."""

        # Component 1: GCN for POI embeddings
        self.poi_embed_model = GCN(
            ninput=self.gcn_nfeat,
            nhid=self.gcn_nhid,
            noutput=self.poi_embed_dim,
            dropout=self.gcn_dropout
        )

        # Component 2: Node Attention Map
        self.node_attn_model = NodeAttnMap(
            in_features=self.gcn_nfeat,
            nhid=self.node_attn_nhid,
            use_mask=False
        )

        # Component 3: User Embeddings
        self.user_embed_model = UserEmbeddings(
            num_users=self.num_users,
            embedding_dim=self.user_embed_dim
        )

        # Component 4: Time Embeddings (Time2Vec)
        self.time_embed_model = Time2Vec(
            activation=self.time_activation,
            out_dim=self.time_embed_dim
        )

        # Component 5: Category Embeddings
        self.cat_embed_model = CategoryEmbeddings(
            num_cats=self.num_cats,
            embedding_dim=self.cat_embed_dim
        )

        # Component 6: Embedding Fusion (user + POI)
        self.embed_fuse_model1 = FuseEmbeddings(
            embed_dim1=self.user_embed_dim,
            embed_dim2=self.poi_embed_dim
        )

        # Component 7: Embedding Fusion (time + category)
        self.embed_fuse_model2 = FuseEmbeddings(
            embed_dim1=self.time_embed_dim,
            embed_dim2=self.cat_embed_dim
        )

        # Component 8: Transformer Sequence Model
        self.seq_input_embed = (self.poi_embed_dim + self.user_embed_dim +
                                self.time_embed_dim + self.cat_embed_dim)
        self.seq_model = TransformerEncoder(
            num_poi=self.num_pois,
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

    def _get_poi_embeddings(self):
        """Get POI embeddings from GCN."""
        return self.poi_embed_model(self.X, self.A)

    def _get_node_attn_map(self):
        """Get node attention map."""
        return self.node_attn_model(self.X, self.A)

    def _get_category_for_poi(self, poi_idx):
        """Get category index for a POI."""
        if isinstance(poi_idx, torch.Tensor):
            poi_idx = poi_idx.item() if poi_idx.dim() == 0 else poi_idx.tolist()
        if isinstance(poi_idx, list):
            return [self.poi_to_cat.get(p, 0) for p in poi_idx]
        return self.poi_to_cat.get(poi_idx, 0)

    def _masked_mse_loss(self, input, target, mask_value=-1):
        """Masked MSE loss for time prediction."""
        mask = target == mask_value
        out = (input[~mask] - target[~mask]) ** 2
        if out.numel() == 0:
            return torch.tensor(0.0, device=input.device)
        loss = out.mean()
        return loss

    def _input_traj_to_embeddings(self, user_idx, input_seq_poi, input_seq_time,
                                   input_seq_cat, poi_embeddings):
        """
        Convert input trajectory to embeddings.

        Args:
            user_idx: User index (int or tensor)
            input_seq_poi: List/tensor of POI indices
            input_seq_time: List/tensor of time values
            input_seq_cat: List/tensor of category indices
            poi_embeddings: Pre-computed POI embeddings from GCN

        Returns:
            List of concatenated embeddings for each step
        """
        # User embedding
        user_input = torch.LongTensor([user_idx]).to(self.device)
        user_embedding = self.user_embed_model(user_input).squeeze(0)  # (user_embed_dim,)

        input_seq_embed = []
        for idx in range(len(input_seq_poi)):
            # POI embedding
            poi_idx = input_seq_poi[idx]
            if isinstance(poi_idx, torch.Tensor):
                poi_idx = poi_idx.item()
            poi_embedding = poi_embeddings[poi_idx]  # (poi_embed_dim,)

            # Time embedding
            time_val = input_seq_time[idx]
            if isinstance(time_val, torch.Tensor):
                time_val = time_val.item()
            time_input = torch.tensor([[time_val]], dtype=torch.float, device=self.device)
            time_embedding = self.time_embed_model(time_input).squeeze(0)  # (time_embed_dim,)

            # Category embedding
            cat_idx = input_seq_cat[idx]
            if isinstance(cat_idx, torch.Tensor):
                cat_idx = cat_idx.item()
            cat_input = torch.LongTensor([cat_idx]).to(self.device)
            cat_embedding = self.cat_embed_model(cat_input).squeeze(0)  # (cat_embed_dim,)

            # Fuse embeddings
            fused_embedding1 = self.embed_fuse_model1(user_embedding, poi_embedding)
            fused_embedding2 = self.embed_fuse_model2(time_embedding, cat_embedding)

            # Concatenate
            concat_embedding = torch.cat((fused_embedding1, fused_embedding2), dim=-1)
            input_seq_embed.append(concat_embedding)

        return input_seq_embed

    def _adjust_pred_prob_by_graph(self, y_pred_poi, batch_input_seqs, batch_seq_lens):
        """
        Adjust prediction probabilities using the node attention map.

        Args:
            y_pred_poi: Predicted POI logits (batch_size, seq_len, num_pois)
            batch_input_seqs: List of input POI sequences for each sample
            batch_seq_lens: List of sequence lengths

        Returns:
            Adjusted prediction logits
        """
        if not self.use_graph_attn:
            return y_pred_poi

        y_pred_poi_adjusted = torch.zeros_like(y_pred_poi)
        attn_map = self._get_node_attn_map()

        for i in range(len(batch_seq_lens)):
            traj_i_input = batch_input_seqs[i]
            for j in range(batch_seq_lens[i]):
                poi_idx = traj_i_input[j]
                if isinstance(poi_idx, torch.Tensor):
                    poi_idx = poi_idx.item()
                y_pred_poi_adjusted[i, j, :] = attn_map[poi_idx, :] + y_pred_poi[i, j, :]

        return y_pred_poi_adjusted

    def forward(self, batch):
        """
        Forward pass.

        Args:
            batch: LibCity Batch object containing trajectory data. Expected keys:
                - 'uid': User IDs (batch_size,)
                - 'current_loc': Location/POI sequences (batch_size, seq_len)
                - 'current_tim': Time sequences (batch_size, seq_len)
                - 'target': Target POI for prediction (used in calculate_loss)
            Sequence lengths obtained via batch.get_origin_len('current_loc')

        Returns:
            y_pred_poi: POI prediction logits (batch_size, seq_len, num_pois)
            y_pred_time: Time prediction (batch_size, seq_len, 1)
            y_pred_cat: Category prediction logits (batch_size, seq_len, num_cats)
        """
        # Extract data from batch using LibCity's dictionary-style access
        # LibCity Batch object uses string keys, not integer indices
        user_ids = batch['uid']
        loc_seq = batch['current_loc']
        time_seq = batch['current_tim']

        # Get sequence lengths using LibCity's get_origin_len method
        seq_lens = batch.get_origin_len('current_loc')

        # Convert to tensors if needed
        if not isinstance(loc_seq, torch.Tensor):
            loc_seq = torch.LongTensor(loc_seq)
        if not isinstance(time_seq, torch.Tensor):
            time_seq = torch.FloatTensor(time_seq)
        if not isinstance(user_ids, torch.Tensor):
            user_ids = torch.LongTensor(user_ids)

        loc_seq = loc_seq.to(self.device)
        time_seq = time_seq.to(self.device)
        user_ids = user_ids.to(self.device)

        if isinstance(seq_lens, torch.Tensor):
            seq_lens = seq_lens.tolist()

        batch_size = loc_seq.shape[0]
        max_seq_len = loc_seq.shape[1]

        # Get POI embeddings from GCN
        poi_embeddings = self._get_poi_embeddings()

        # Process each sample in batch
        batch_seq_embeds = []
        batch_input_seqs = []
        batch_seq_lens = []

        for b in range(batch_size):
            # Get user id
            user_idx = user_ids[b].item() if user_ids.dim() > 0 else user_ids.item()
            user_idx = min(user_idx, self.num_users - 1)  # Ensure valid index

            # Get sequence length
            seq_len = seq_lens[b] if isinstance(seq_lens, list) else seq_lens
            if isinstance(seq_len, torch.Tensor):
                seq_len = seq_len.item()
            seq_len = int(seq_len)

            # Get input sequences (exclude last step for prediction)
            input_seq_poi = loc_seq[b, :seq_len].tolist()
            input_seq_time = time_seq[b, :seq_len].tolist()

            # Get categories for each POI
            input_seq_cat = [self._get_category_for_poi(p) for p in input_seq_poi]

            # Convert to embeddings
            input_seq_embed = self._input_traj_to_embeddings(
                user_idx, input_seq_poi, input_seq_time, input_seq_cat, poi_embeddings
            )

            if len(input_seq_embed) > 0:
                batch_seq_embeds.append(torch.stack(input_seq_embed))
            else:
                # Handle empty sequence
                batch_seq_embeds.append(torch.zeros(1, self.seq_input_embed, device=self.device))

            batch_input_seqs.append(input_seq_poi)
            batch_seq_lens.append(seq_len)

        # Pad sequences
        batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=0)

        # Generate attention mask
        max_len = batch_padded.size(1)
        src_mask = self.seq_model.generate_square_subsequent_mask(max_len, self.device)

        # Forward through transformer
        y_pred_poi, y_pred_time, y_pred_cat = self.seq_model(batch_padded, src_mask)

        # Adjust predictions using graph attention
        y_pred_poi_adjusted = self._adjust_pred_prob_by_graph(
            y_pred_poi, batch_input_seqs, batch_seq_lens
        )

        return y_pred_poi_adjusted, y_pred_time, y_pred_cat, batch_seq_lens

    def predict(self, batch):
        """
        Prediction method for LibCity.

        Args:
            batch: Input batch dictionary

        Returns:
            POI prediction scores for the last timestep of each sequence
        """
        y_pred_poi, y_pred_time, y_pred_cat, batch_seq_lens = self.forward(batch)

        # Get predictions for the last valid timestep of each sequence
        batch_size = y_pred_poi.size(0)
        predictions = []

        for i in range(batch_size):
            last_idx = batch_seq_lens[i] - 1
            last_idx = max(0, min(last_idx, y_pred_poi.size(1) - 1))
            predictions.append(y_pred_poi[i, last_idx, :])

        return torch.stack(predictions)  # (batch_size, num_pois)

    def calculate_loss(self, batch):
        """
        Calculate multi-task loss for training.

        Args:
            batch: LibCity Batch object containing:
                - trajectory data for forward pass ('uid', 'current_loc', 'current_tim')
                - 'target': Target POI indices (batch_size,)
                - 'target_tim': Target time (optional)

        Returns:
            Combined loss (POI loss + weighted time loss + category loss)
        """
        y_pred_poi, y_pred_time, y_pred_cat, batch_seq_lens = self.forward(batch)

        # Get target POI using LibCity's dictionary-style access
        target_poi = batch['target']

        # Get optional target time using try/except (Batch doesn't have .get() method)
        try:
            target_tim = batch['target_tim']
        except KeyError:
            target_tim = None

        if not isinstance(target_poi, torch.Tensor):
            target_poi = torch.LongTensor(target_poi)
        target_poi = target_poi.to(self.device)

        batch_size = y_pred_poi.size(0)

        # Calculate POI loss (using last timestep prediction)
        poi_losses = []
        for i in range(batch_size):
            last_idx = batch_seq_lens[i] - 1
            last_idx = max(0, min(last_idx, y_pred_poi.size(1) - 1))
            pred = y_pred_poi[i, last_idx, :].unsqueeze(0)
            target = target_poi[i].unsqueeze(0)
            poi_losses.append(self.criterion_poi(pred, target))
        loss_poi = torch.stack(poi_losses).mean()

        # Calculate time loss if target time is provided
        loss_time = torch.tensor(0.0, device=self.device)
        if target_tim is not None:
            if not isinstance(target_tim, torch.Tensor):
                target_tim = torch.FloatTensor(target_tim)
            target_tim = target_tim.to(self.device)

            time_losses = []
            for i in range(batch_size):
                last_idx = batch_seq_lens[i] - 1
                last_idx = max(0, min(last_idx, y_pred_time.size(1) - 1))
                pred_time = y_pred_time[i, last_idx, 0]
                target_t = target_tim[i]
                time_losses.append((pred_time - target_t) ** 2)
            loss_time = torch.stack(time_losses).mean()

        # Calculate category loss (use POI to cat mapping for targets)
        target_cat = torch.LongTensor([self._get_category_for_poi(p.item())
                                        for p in target_poi]).to(self.device)
        cat_losses = []
        for i in range(batch_size):
            last_idx = batch_seq_lens[i] - 1
            last_idx = max(0, min(last_idx, y_pred_cat.size(1) - 1))
            pred = y_pred_cat[i, last_idx, :].unsqueeze(0)
            target = target_cat[i].unsqueeze(0)
            cat_losses.append(self.criterion_cat(pred, target))
        loss_cat = torch.stack(cat_losses).mean()

        # Combined loss
        total_loss = loss_poi + loss_time * self.time_loss_weight + loss_cat

        return total_loss
