"""
LoTNext Model Adapted for LibCity Framework

Original Paper: Long-Tail Adjusted Next POI Prediction
Original Repository: repos/LoTNext

This model adapts the Flashback architecture from LoTNext for trajectory location prediction.
Key components:
1. Location embedding with GCN-based graph propagation
2. User embedding
3. Time2Vec temporal encoding
4. Transformer-based sequence modeling
5. Spatial-temporal weighted aggregation
6. Denoising GCN for user-POI bipartite graph
7. Long-tail logit adjustment

Adapted from:
- repos/LoTNext/network.py (Flashback class, lines 314-518)
- repos/LoTNext/model.py (TransformerModel, EncoderLayer, Time2Vec, FuseEmbeddings, etc.)
- repos/LoTNext/trainer.py (loss computation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch_geometric.nn import GCNConv
import numpy as np
import math
import scipy.sparse as sp
from scipy.sparse import coo_matrix, identity

from libcity.model.abstract_model import AbstractModel


# ========================== Utility Functions ==========================

def sparse_matrix_to_tensor(graph):
    """Convert scipy sparse matrix to PyTorch sparse tensor."""
    graph = coo_matrix(graph)
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = graph.shape
    graph = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    return graph


def calculate_random_walk_matrix(adj_mx):
    """Calculate random walk matrix D^-1 * W."""
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def haversine(s1, s2):
    """
    Calculate Haversine distance between two batches of geographic locations.

    Args:
        s1, s2: Tensors of shape [batch_size, 2] containing latitude and longitude.

    Returns:
        Distance tensor of shape [batch_size].
    """
    s1 = s1 * math.pi / 180
    s2 = s2 * math.pi / 180

    lat1, lon1 = s1[:, 0], s1[:, 1]
    lat2, lon2 = s2[:, 0], s2[:, 1]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(a))

    r = 6371  # Earth radius in km
    return c * r


# ========================== Supporting Modules ==========================

class FeedForwardNetwork(nn.Module):
    """Feed-forward network with GELU activation."""

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


def mask_value(epoch, T, v_min=-100, v_max=-1e9):
    """Compute progressive mask value for attention."""
    return -10 ** (np.log10(-v_min) + (np.log10(-v_max) - np.log10(-v_min)) * (epoch / T))


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional progressive masking."""

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


class SineActivation(nn.Module):
    """Sine activation for Time2Vec."""

    def __init__(self, seq_len, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.l1 = nn.Linear(1, out_features - 1, bias=True)
        self.l2 = nn.Linear(1, 1, bias=True)
        self.f = torch.sin

    def forward(self, tau):
        v1 = self.l1(tau.unsqueeze(-1))
        v1 = self.f(v1)
        v2 = self.l2(tau.unsqueeze(-1))
        return torch.cat([v1, v2], -1)


class Time2Vec(nn.Module):
    """Time2Vec temporal encoding module."""

    def __init__(self, activation, batch_size, seq_len, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(seq_len, out_dim)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        return self.l1(x)


class FuseEmbeddings(nn.Module):
    """Fuse location and time embeddings."""

    def __init__(self, loc_embed_dim, time_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = loc_embed_dim + time_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, loc_embed, time_embed):
        x = self.fuse_embed(torch.cat((loc_embed, time_embed), 2))
        x = self.leaky_relu(x)
        return x


# ========================== Denoising GCN Modules ==========================

class AttentionLayer(nn.Module):
    """Attention layer for edge weight computation."""

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
    """Edge denoising layer with threshold filtering."""

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
    """GCN layer using PyTorch Geometric."""

    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x


class DenoisingGCNNet(nn.Module):
    """Complete denoising GCN network for user-POI bipartite graph."""

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


# ========================== RNN Factory ==========================

class RnnFactory:
    """Factory for creating RNN units."""

    def __init__(self, rnn_type_str):
        self.rnn_type = rnn_type_str.lower()

    def is_lstm(self):
        return self.rnn_type == 'lstm'

    def create(self, hidden_size):
        if self.rnn_type == 'rnn':
            return nn.RNN(hidden_size, hidden_size)
        if self.rnn_type == 'gru':
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == 'lstm':
            return nn.LSTM(hidden_size, hidden_size)
        raise ValueError(f"Unknown RNN type: {self.rnn_type}")


# ========================== Main LoTNext Model ==========================

class LoTNext(AbstractModel):
    """
    LoTNext: Long-Tail Adjusted Next POI Prediction Model.

    This model combines:
    - GCN-based location embedding propagation
    - User embedding
    - Time2Vec temporal encoding
    - Transformer-based sequence modeling
    - Spatial-temporal weighted aggregation
    - Denoising GCN for user-POI bipartite graph
    - Long-tail logit adjustment

    Adapted from the Flashback model in LoTNext repository.
    """

    def __init__(self, config, data_feature):
        super(LoTNext, self).__init__(config, data_feature)

        # Device configuration
        self.device = config.get('device', 'cpu')

        # Data features
        self.loc_size = data_feature.get('loc_size', data_feature.get('num_loc', 1000))
        self.user_count = data_feature.get('uid_size', data_feature.get('num_users', 100))
        self.tim_size = data_feature.get('tim_size', 168)

        # Model hyperparameters
        self.hidden_size = config.get('hidden_size', 128)
        self.time_emb_dim = config.get('time_emb_dim', 6)
        self.rnn_type = config.get('rnn_type', 'gru')
        self.lambda_loc = config.get('lambda_loc', 1.0)
        self.lambda_s = config.get('lambda_s', 100.0)
        self.use_graph_user = config.get('use_graph_user', False)
        self.use_spatial_graph = config.get('use_spatial_graph', False)
        self.dropout_p = config.get('dropout_p', 0.3)

        # Transformer parameters
        self.transformer_nhid = config.get('transformer_nhid', 256)
        self.transformer_dropout = config.get('transformer_dropout', 0.1)
        self.attention_dropout_rate = config.get('attention_dropout_rate', 0.1)
        self.transformer_nhead = config.get('transformer_nhead', 2)
        self.sequence_length = config.get('sequence_length', 20)
        self.batch_size = config.get('batch_size', 64)

        # Evaluation method
        self.evaluate_method = config.get('evaluate_method', 'all')

        # Embeddings
        self.encoder = nn.Embedding(self.loc_size, self.hidden_size)
        self.user_encoder = nn.Embedding(self.user_count, self.hidden_size)

        # RNN
        rnn_factory = RnnFactory(self.rnn_type)
        self.rnn = rnn_factory.create(self.hidden_size)
        self.is_lstm = rnn_factory.is_lstm()

        # Transformer encoder
        self.seq_model = EncoderLayer(
            self.hidden_size + self.time_emb_dim,
            self.transformer_nhid,
            self.transformer_dropout,
            self.attention_dropout_rate,
            self.transformer_nhead
        )

        # Time2Vec
        self.time_embed_model = Time2Vec('sin', self.batch_size, self.sequence_length, out_dim=self.time_emb_dim)

        # Embedding fusion
        self.embed_fuse_model = FuseEmbeddings(self.hidden_size, self.time_emb_dim)

        # Decoders
        self.decoder = nn.Linear(self.hidden_size + self.time_emb_dim, self.hidden_size)
        self.fc = nn.Linear(2 * self.hidden_size, self.loc_size)
        self.time_decoder = nn.Linear(self.hidden_size + self.time_emb_dim, 1)

        # Denoising GCN
        self.denoise = DenoisingGCNNet(self.hidden_size, self.hidden_size, self.hidden_size)

        # Dropout
        self.dropout = nn.Dropout(self.dropout_p)

        # Spatial decay function
        self.f_s = lambda delta_s, user_len: torch.exp(-(delta_s * self.lambda_s))

        # Load graph data from data_feature if available
        self._load_graph_data(data_feature)

        # Loss function
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.loss_weight = nn.Parameter(torch.ones(3))

        # Training epoch counter (for progressive masking)
        self.current_epoch = 0

    def _load_graph_data(self, data_feature):
        """Load and preprocess graph data from data_feature."""
        # Transition graph
        if 'transition_graph' in data_feature and data_feature['transition_graph'] is not None:
            graph = data_feature['transition_graph']
            I = identity(graph.shape[0], format='coo')
            self.graph = sparse_matrix_to_tensor(
                calculate_random_walk_matrix((graph * self.lambda_loc + I).astype(np.float32))
            )
        else:
            # Create a simple identity graph if not provided
            self.graph = sparse_matrix_to_tensor(
                identity(self.loc_size, format='coo').astype(np.float32)
            )

        # Spatial graph
        if 'spatial_graph' in data_feature and data_feature['spatial_graph'] is not None:
            self.spatial_graph = data_feature['spatial_graph']
        else:
            self.spatial_graph = None

        # User-POI interaction graph
        if 'interact_graph' in data_feature and data_feature['interact_graph'] is not None:
            interact_graph = data_feature['interact_graph']
            self.interact_graph = sparse_matrix_to_tensor(
                calculate_random_walk_matrix(interact_graph)
            )
            self.has_interact_graph = True
        else:
            # Create a dummy interaction graph
            self.interact_graph = None
            self.has_interact_graph = False

    def set_epoch(self, epoch):
        """Set current epoch for progressive masking."""
        self.current_epoch = epoch

    def _get_batch_item(self, batch, key, default=None):
        """Safely get item from batch with default value.

        LibCity's BatchPAD class does not support .get() method.
        This helper provides a safe way to access optional batch fields.

        Args:
            batch: LibCity batch object (BatchPAD)
            key: Key to access in the batch
            default: Default value if key is not present

        Returns:
            The value from batch[key] if key exists, otherwise default
        """
        if key in batch.data:
            return batch[key]
        return default

    def forward(self, batch):
        """
        Forward pass for LoTNext model.

        Args:
            batch: Dictionary containing:
                - 'current_loc': Location sequence (batch_size, seq_len)
                - 'current_tim': Time sequence (batch_size, seq_len)
                - 'target': Target location (batch_size,)
                - 'uid': User IDs (batch_size,)
                - 'current_tim_slot': Time slot sequence (batch_size, seq_len) [optional]
                - 'current_coord': Coordinate sequence (batch_size, seq_len, 2) [optional]

        Returns:
            y_linear: Prediction logits (batch_size, loc_size)
        """
        # Extract data from batch
        current_loc = batch['current_loc']  # (batch_size, seq_len)
        current_tim = self._get_batch_item(batch, 'current_tim', None)
        uid = self._get_batch_item(batch, 'uid', None)

        # Handle different batch formats
        if isinstance(current_loc, torch.Tensor):
            batch_size = current_loc.shape[0]
            seq_len = current_loc.shape[1]
        else:
            current_loc = torch.tensor(current_loc)
            batch_size = current_loc.shape[0]
            seq_len = current_loc.shape[1]

        current_loc = current_loc.to(self.device)

        # Get time slots
        if 'current_tim_slot' in batch.data:
            t_slot = batch['current_tim_slot'].to(self.device)  # (batch_size, seq_len)
        elif current_tim is not None:
            # Compute time slots from timestamps if not provided
            current_tim = current_tim.to(self.device) if isinstance(current_tim, torch.Tensor) else torch.tensor(current_tim).to(self.device)
            t_slot = (current_tim % (24 * 7)).float()  # Simple time slot computation
        else:
            # Default time slots
            t_slot = torch.zeros(batch_size, seq_len).to(self.device)

        # Get coordinates if available
        if 'current_coord' in batch.data:
            coords = batch['current_coord'].to(self.device)  # (batch_size, seq_len, 2)
        else:
            coords = None

        # Get user IDs
        if uid is not None:
            if isinstance(uid, torch.Tensor):
                active_user = uid.to(self.device)
            else:
                active_user = torch.tensor(uid).to(self.device)
        else:
            active_user = torch.zeros(batch_size, dtype=torch.long).to(self.device)

        # Transpose for seq_len first format
        x = current_loc.transpose(0, 1)  # (seq_len, batch_size)
        t_slot_t = t_slot.transpose(0, 1)  # (seq_len, batch_size)

        seq_len_actual, user_len = x.size()

        # Location embedding
        x_emb = self.encoder(x)  # (seq_len, batch_size, hidden_size)

        # User embedding
        p_u = self.user_encoder(active_user)  # (batch_size, hidden_size)
        p_u = p_u.view(user_len, self.hidden_size)

        # GCN-based location embedding propagation
        graph = self.graph.to(self.device)
        loc_emb = self.encoder(torch.LongTensor(list(range(self.loc_size))).to(self.device))
        encoder_weight = torch.sparse.mm(graph, loc_emb).to(self.device)

        # Apply spatial graph if available
        if self.use_spatial_graph and self.spatial_graph is not None:
            I = identity(self.spatial_graph.shape[0], format='coo')
            spatial_graph = (self.spatial_graph * self.lambda_loc + I).astype(np.float32)
            spatial_graph = calculate_random_walk_matrix(spatial_graph)
            spatial_graph = sparse_matrix_to_tensor(spatial_graph).to(self.device)
            encoder_weight = encoder_weight + torch.sparse.mm(spatial_graph, loc_emb).to(self.device)
            encoder_weight = encoder_weight / 2

        # Update location embeddings
        new_x_emb = []
        for i in range(seq_len_actual):
            temp_x = torch.index_select(encoder_weight, 0, x[i])
            new_x_emb.append(temp_x)
        x_emb = torch.stack(new_x_emb, dim=0)

        # User-POI interaction via denoising GCN
        if self.has_interact_graph and self.interact_graph is not None:
            interact_graph = self.interact_graph.to(self.device)

            # User-aggregated POI embeddings
            encoder_weight_user = torch.sparse.mm(interact_graph, loc_emb).to(self.device)

            # POI-aggregated user embeddings (need to handle user count)
            user_emb = self.encoder(torch.LongTensor(list(range(min(self.interact_graph.size(0), self.loc_size)))).to(self.device))
            encoder_weight_poi = torch.sparse.mm(interact_graph.t(), user_emb).to(self.device)

            edge_index = self.interact_graph.coalesce().indices().to(self.device)

            try:
                gcn_output, denoised_edge_index, denoised_edge_weights = self.denoise(
                    encoder_weight_user, encoder_weight_poi, edge_index
                )
                encoder_weight_poi = gcn_output[self.interact_graph.size(0):]

                # Update x_emb with denoised POI embeddings
                new_x_emb = []
                for i in range(seq_len_actual):
                    temp_x = torch.index_select(encoder_weight_poi, 0, x[i])
                    new_x_emb.append(temp_x)
                x_emb_new = torch.stack(new_x_emb, dim=0)
                x_emb = (x_emb + x_emb_new) / 2

                # User-location similarity
                user_preference = torch.index_select(encoder_weight_user, 0, active_user).unsqueeze(0)
                user_loc_similarity = torch.exp(-(torch.norm(user_preference - x_emb, p=2, dim=-1))).to(self.device)
                user_loc_similarity = user_loc_similarity.permute(1, 0)
            except Exception:
                # Fallback if denoising fails
                user_loc_similarity = torch.ones(user_len, seq_len_actual).to(self.device)
        else:
            user_loc_similarity = torch.ones(user_len, seq_len_actual).to(self.device)

        # Time2Vec encoding
        t_emb = self.time_embed_model(t_slot / 168).to(self.device)  # (batch_size, seq_len, time_emb_dim)
        x_emb_fused = self.embed_fuse_model(x_emb.transpose(0, 1), t_emb).to(self.device)  # (batch_size, seq_len, hidden+time)

        # Transformer encoding
        out = self.seq_model(x_emb_fused, epoch=self.current_epoch).to(self.device)  # (batch_size, seq_len, hidden+time)

        # Time prediction (auxiliary task)
        out_time = self.time_decoder(out).to(self.device)  # (batch_size, seq_len, 1)

        # Decode to hidden size
        out = self.decoder(out).to(self.device).transpose(0, 1)  # (seq_len, batch_size, hidden)

        # Spatial-temporal weighted aggregation
        out_w = torch.zeros(seq_len_actual, user_len, self.hidden_size, device=self.device)

        if coords is not None:
            coords_t = coords.transpose(0, 1)  # (seq_len, batch_size, 2)
            for i in range(seq_len_actual):
                sum_w = torch.zeros(user_len, 1, device=self.device)
                for j in range(i + 1):
                    dist_s = haversine(coords_t[i], coords_t[j])
                    b_j = self.f_s(dist_s, user_len)
                    b_j = b_j.unsqueeze(1)
                    w_j = b_j + 1e-10
                    w_j = w_j * user_loc_similarity[:, j].unsqueeze(1)
                    sum_w = sum_w + w_j
                    out_w[i] = out_w[i] + w_j * out[j]
                out_w[i] = out_w[i] / sum_w
        else:
            # Without coordinates, use simple averaging
            for i in range(seq_len_actual):
                out_w[i] = out[:i + 1].mean(dim=0)

        # Concatenate with user embedding
        out_pu = torch.zeros(seq_len_actual, user_len, 2 * self.hidden_size, device=self.device)
        for i in range(seq_len_actual):
            out_pu[i] = torch.cat([out_w[i], p_u], dim=1)

        # Final prediction
        y_linear = self.fc(out_pu)  # (seq_len, batch_size, loc_size)

        # Get the last timestep prediction
        # Get actual sequence lengths
        if hasattr(batch, 'get_origin_len'):
            origin_len = batch.get_origin_len('current_loc')
            final_out_index = torch.tensor(origin_len) - 1
        else:
            final_out_index = torch.full((user_len,), seq_len_actual - 1, dtype=torch.long)

        final_out_index = final_out_index.to(self.device)

        # Gather final outputs for both y_linear and out_pu
        final_out = []
        final_out_pu = []
        for i in range(user_len):
            final_out.append(y_linear[final_out_index[i], i, :])
            final_out_pu.append(out_pu[final_out_index[i], i, :])
        y_linear_final = torch.stack(final_out, dim=0)  # (batch_size, loc_size)
        out_pu_final = torch.stack(final_out_pu, dim=0)  # (batch_size, 2*hidden_size)

        return y_linear_final, out_pu_final

    def predict(self, batch):
        """
        Make predictions for a batch.

        Args:
            batch: Input batch dictionary.

        Returns:
            score: Prediction scores (batch_size, loc_size) or sampled scores.
        """
        score, _ = self.forward(batch)  # Unpack tuple, discard intermediate representation

        if self.evaluate_method == 'sample' and 'neg_loc' in batch.data:
            pos_neg_index = torch.cat((batch['target'].unsqueeze(1), batch['neg_loc']), dim=1)
            score = torch.gather(score, 1, pos_neg_index.to(self.device))

        return score

    def calculate_loss(self, batch):
        """
        Calculate training loss.

        Args:
            batch: Input batch dictionary containing 'target' for labels.

        Returns:
            loss: Combined loss tensor.
        """
        # Forward pass - now returns tuple (y_linear, out_pu)
        y_linear, out_pu = self.forward(batch)

        # Get target
        target = batch['target']
        if isinstance(target, torch.Tensor):
            target = target.to(self.device)
        else:
            target = torch.tensor(target).to(self.device)

        # Ensure target is 1D
        if target.dim() > 1:
            target = target.view(-1)

        # Compute cosine similarity for long-tail adjustment
        # Use out_pu (pre-fc representation) instead of y_linear (post-fc output)
        # out_pu shape: (batch_size, 2*hidden_size)
        # fc.weight shape: (loc_size, 2*hidden_size)
        # Result shape: (batch_size, loc_size)
        cosine_similarity = F.linear(F.normalize(out_pu), F.normalize(self.fc.weight))

        # Get target cosine values
        target_cosine = cosine_similarity.gather(1, target.unsqueeze(1)).view(-1)

        # Compute vector lengths for weighting
        vector_lengths = torch.where(target_cosine > 0, torch.ones_like(target_cosine), 1 - target_cosine)

        # Compute geometric mean length
        log_geom_mean_length = torch.log(vector_lengths + 1e-9).mean()
        geom_mean_length = torch.exp(log_geom_mean_length)

        # Compute length difference for weight adjustment
        length_diff = vector_lengths - geom_mean_length

        # Adjust weights based on length difference
        weights = torch.ones_like(vector_lengths)
        weights[length_diff > 0] = 1 + length_diff[length_diff > 0]

        # Apply softmax to get loss weights
        loss_weights = F.softmax(self.loss_weight, dim=0)

        # Loss 1: Weighted cross-entropy with long-tail adjustment
        l1 = (self.cross_entropy_loss(y_linear, target) * weights).mean()

        # Loss 2: Standard cross-entropy
        l2 = self.cross_entropy_loss(y_linear, target)

        # Loss 3: Time prediction MSE (if time prediction is available)
        # For now, we use a placeholder since we don't have target time
        l3 = torch.tensor(0.0, device=self.device)
        if 'target_tim_slot' in batch.data:
            # Would need to track out_time from forward pass
            pass

        # Combined loss
        loss = loss_weights[0] * l1 + loss_weights[1] * l2 + loss_weights[2] * l3

        return loss
