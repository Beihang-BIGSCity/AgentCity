"""
AGRAN: Adaptive Graph Relation-Aware Attention Network for Next POI Recommendation

This model is adapted from the original AGRAN implementation.

Key Components:
1. AGCN (Adaptive Graph Convolutional Network) - learns adaptive adjacency via cosine similarity
2. TimeAwareMultiHeadAttention - attention with time, distance, and positional embeddings
3. PointWiseFeedForward - feed-forward network using 1D convolutions
4. Multiple attention blocks with layer normalization

Adaptations for LibCity:
- Inherits from AbstractModel
- Adapted batch input format to LibCity's trajectory batch dictionary
- Implemented predict() and calculate_loss() methods following LibCity conventions
- Extracted hyperparameters from config dict
- Extracted data features from data_feature dict
- Added KL divergence regularization with optional prior graph

Original files:
- repos/AGRAN/model_ag.py (AGRAN, TimeAwareMultiHeadAttention, PointWiseFeedForward)
- repos/AGRAN/AGCN.py (Adaptive Graph Convolutional Network)
- repos/AGRAN/utils_ag.py (Haversine distance, relation matrices)
"""

import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model.abstract_model import AbstractModel

FLOAT_MIN = -sys.float_info.max


# ========================== AGCN Module ==========================

class AGCN(nn.Module):
    """
    Adaptive Graph Convolutional Network.

    Learns an adaptive adjacency matrix using weighted cosine similarity
    and performs multi-layer graph convolution.

    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        layer: Number of graph convolution layers
        dropout: Dropout rate
        bias: Whether to use bias
    """

    def __init__(self, input_dim, output_dim, layer=3, dropout=0.2, bias=False):
        super(AGCN, self).__init__()

        self.dropout = dropout
        self.layer_num = layer
        self.cos_weight = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty(input_dim, output_dim))
        )

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))

    def get_neighbor_hard_threshold(self, adj, epsilon=0, mask_value=0):
        """
        Apply hard threshold to adjacency matrix and normalize.

        Args:
            adj: Adjacency matrix
            epsilon: Threshold value
            mask_value: Value for masked entries

        Returns:
            Normalized adjacency matrix and raw adjacency matrix
        """
        mask = (adj > epsilon).detach().float()
        raw_adj = adj * mask + (1 - mask) * mask_value
        adj_dig = torch.clamp(
            torch.pow(torch.sum(raw_adj, dim=-1, keepdim=True), 0.5), min=1e-12
        )
        update_adj = raw_adj / adj_dig / adj_dig.transpose(-1, -2)
        return update_adj, raw_adj

    def get_neighbor_soft_row_threshold(self, adj, epsilon=100, device=None):
        """Apply soft row-wise top-k threshold."""
        top_k = min(epsilon, adj.size(-1))
        _, index = torch.topk(adj, top_k, dim=-1)
        update_adj = torch.zeros_like(adj).scatter_(-1, index, 1)
        return update_adj

    def get_neighbor_soft_threshold(self, adj, epsilon, device=None):
        """Apply soft global top-k threshold."""
        top_k = math.ceil(epsilon * adj.size(-1) ** 2)
        adj_float = adj.flatten()
        _, index = torch.topk(adj_float, top_k, dim=-1)
        update_adj = torch.zeros_like(adj_float).scatter_(-1, index, 1)
        update_adj = update_adj.reshape(adj.size(0), adj.size(1))
        return update_adj

    def cosine_matrix_div(self, emb):
        """Compute cosine similarity matrix."""
        node_norm = emb.div(torch.norm(emb, p=2, dim=-1, keepdim=True))
        cos_adj = torch.mm(node_norm, node_norm.transpose(-1, -2))
        return cos_adj

    def weight_cosine_matrix_div(self, emb):
        """Compute weighted cosine similarity matrix."""
        emb = torch.matmul(emb, self.cos_weight)
        node_norm = F.normalize(emb, p=2, dim=-1)
        cos_adj = torch.mm(node_norm, node_norm.transpose(-1, -2))
        return cos_adj

    def cos_matrix_sim(self, emb):
        """Compute cosine similarity using pairwise comparison."""
        cos_adj = torch.cosine_similarity(
            emb.unsqueeze(0), emb.unsqueeze(1), dim=-1
        ).detach()
        return cos_adj

    def forward(self, inputs):
        """
        Forward pass.

        Args:
            inputs: Embedding layer (nn.Embedding)

        Returns:
            Enhanced embeddings and support matrix for KL loss
        """
        # Get embeddings excluding padding token (index 0)
        x = inputs.weight[1:, :]

        # Compute adaptive adjacency via weighted cosine similarity
        support = self.weight_cosine_matrix_div(x)
        support, support_loss = self.get_neighbor_hard_threshold(support)

        if self.training:
            support = F.dropout(support, self.dropout)

        # Multi-layer graph convolution
        x_fin = [x]
        layer = x
        for f in range(self.layer_num):
            layer = torch.matmul(support, layer)
            layer = torch.tanh(layer)
            x_fin += [layer]

        x_fin = torch.stack(x_fin, dim=1)
        out = torch.sum(x_fin, dim=1)

        if self.bias is not None:
            out += self.bias

        # Concatenate padding embedding back
        fin_out = torch.cat(
            [inputs.weight[0, :].unsqueeze(dim=0), out], dim=0
        )

        return fin_out, support_loss


# ========================== Feed-Forward Module ==========================

class PointWiseFeedForward(nn.Module):
    """
    Point-wise Feed-Forward Network using 1D convolutions.

    Args:
        hidden_units: Hidden dimension size
        dropout_rate: Dropout probability
    """

    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        """
        Forward pass with residual connection.

        Args:
            inputs: Input tensor (batch, seq_len, hidden_units)

        Returns:
            Output tensor (batch, seq_len, hidden_units)
        """
        outputs = self.dropout2(
            self.conv2(
                self.relu(
                    self.dropout1(self.conv1(inputs.transpose(-1, -2)))
                )
            )
        )
        outputs = outputs.transpose(-1, -2)
        outputs += inputs  # Residual connection
        return outputs


# ========================== Time-Aware Attention Module ==========================

class TimeAwareMultiHeadAttention(nn.Module):
    """
    Time-Aware Multi-Head Self-Attention.

    Incorporates absolute positional embeddings, time interval embeddings,
    and distance embeddings into the attention mechanism.

    Args:
        hidden_size: Hidden dimension size
        head_num: Number of attention heads
        dropout_rate: Dropout probability
        device: Computing device
    """

    def __init__(self, hidden_size, head_num, dropout_rate, device):
        super(TimeAwareMultiHeadAttention, self).__init__()

        self.Q_w = nn.Linear(hidden_size, hidden_size)
        self.K_w = nn.Linear(hidden_size, hidden_size)
        self.V_w = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.dev = device

    def forward(self, queries, keys, time_mask, attn_mask,
                time_matrix_K, time_matrix_V,
                dis_matrix_K, dis_matrix_V,
                abs_pos_K, abs_pos_V):
        """
        Forward pass.

        Args:
            queries: Query tensor (batch, seq_len, hidden_size)
            keys: Key tensor (batch, seq_len, hidden_size)
            time_mask: Padding mask (batch, seq_len)
            attn_mask: Causal attention mask (seq_len, seq_len)
            time_matrix_K: Time relation embeddings for keys (batch, seq, seq, hidden)
            time_matrix_V: Time relation embeddings for values (batch, seq, seq, hidden)
            dis_matrix_K: Distance relation embeddings for keys (batch, seq, seq, hidden)
            dis_matrix_V: Distance relation embeddings for values (batch, seq, seq, hidden)
            abs_pos_K: Absolute position embeddings for keys (batch, seq, hidden)
            abs_pos_V: Absolute position embeddings for values (batch, seq, hidden)

        Returns:
            Attention output (batch, seq_len, hidden_size)
        """
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        # Split into heads
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)
        dis_matrix_K_ = torch.cat(torch.split(dis_matrix_K, self.head_size, dim=3), dim=0)
        dis_matrix_V_ = torch.cat(torch.split(dis_matrix_V, self.head_size, dim=3), dim=0)
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

        # Compute attention weights with multiple relation types
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)
        attn_weights += dis_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)

        # Scale
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        # Apply masks
        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)

        paddings = torch.ones(attn_weights.shape) * (-2**32 + 1)
        paddings = paddings.to(self.dev)

        attn_weights = torch.where(time_mask, paddings, attn_weights)
        attn_weights = torch.where(attn_mask, paddings, attn_weights)

        attn_weights = self.softmax(attn_weights)
        attn_weights = self.dropout(attn_weights)

        # Compute outputs with multiple relation types
        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)
        outputs += attn_weights.unsqueeze(2).matmul(dis_matrix_V_).reshape(outputs.shape).squeeze(2)

        # Concatenate heads
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2)

        return outputs


# ========================== Haversine Distance Function ==========================

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on earth (in km).

    Args:
        lon1, lat1: Longitude and latitude of point 1
        lon2, lat2: Longitude and latitude of point 2

    Returns:
        Distance in kilometers
    """
    from math import radians, cos, sin, asin, sqrt
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth radius in km
    return c * r


# ========================== Relation Matrix Computation ==========================

def compute_time_matrix(time_seq, time_span):
    """
    Compute pairwise time interval matrix.

    Args:
        time_seq: Time sequence array
        time_span: Maximum time span (for clipping)

    Returns:
        Time relation matrix
    """
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i] - time_seq[j])
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = int(span)
    return time_matrix


def compute_distance_matrix(coords, dis_span):
    """
    Compute pairwise distance matrix using Haversine formula.

    Args:
        coords: Coordinate array of shape (seq_len, 2) with [lat, lon]
        dis_span: Maximum distance span (for clipping)

    Returns:
        Distance relation matrix
    """
    size = len(coords)
    dis_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            if coords[i][0] == 0 and coords[i][1] == 0:
                dis_matrix[i][j] = dis_span
            elif coords[j][0] == 0 and coords[j][1] == 0:
                dis_matrix[i][j] = dis_span
            else:
                span = int(abs(haversine(
                    coords[i][1], coords[i][0],
                    coords[j][1], coords[j][0]
                )))
                if span > dis_span:
                    dis_matrix[i][j] = dis_span
                else:
                    dis_matrix[i][j] = span
    return dis_matrix


# ========================== Main AGRAN Model ==========================

class AGRAN(AbstractModel):
    """
    AGRAN: Adaptive Graph Relation-Aware Attention Network.

    This model combines:
    1. Adaptive Graph Convolutional Network (AGCN) for item embedding enhancement
    2. Time-aware multi-head self-attention with time and distance relations
    3. Multiple attention blocks with layer normalization
    4. KL divergence regularization with prior graph

    Args:
        config (dict): Configuration dictionary containing model hyperparameters
        data_feature (dict): Data features including vocab sizes, etc.

    Required config parameters:
        - hidden_units: Hidden dimension size (default: 64)
        - num_blocks: Number of attention blocks (default: 3)
        - num_heads: Number of attention heads (default: 2)
        - dropout_rate: Dropout probability (default: 0.3)
        - maxlen: Maximum sequence length (default: 50)
        - time_span: Maximum time span for discretization (default: 256)
        - dis_span: Maximum distance span for discretization (default: 256)
        - l2_emb: L2 regularization weight (default: 0.0001)
        - kl_reg: KL divergence regularization weight (default: 1.0)
        - gcn_layers: Number of GCN layers (default: 4)

    Required data_feature:
        - loc_size: Number of POI locations
        - uid_size: Number of users
        - tim_size: Number of time slots (optional)
        - prior_graph: Prior adjacency matrix for KL regularization (optional)
    """

    def __init__(self, config, data_feature):
        super(AGRAN, self).__init__(config, data_feature)

        self.device = config.get('device', 'cpu')

        # Data dimensions from data_feature
        self.num_items = data_feature.get('loc_size', 1000)
        self.num_users = data_feature.get('uid_size', 100)
        self.num_times = data_feature.get('tim_size', 256)

        # Model hyperparameters from config
        self.hidden_units = config.get('hidden_units', 64)
        self.num_blocks = config.get('num_blocks', 3)
        self.num_heads = config.get('num_heads', 2)
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.maxlen = config.get('maxlen', 50)
        self.time_span = config.get('time_span', 256)
        self.dis_span = config.get('dis_span', 256)
        self.l2_emb = config.get('l2_emb', 0.0001)
        self.kl_reg = config.get('kl_reg', 1.0)
        self.gcn_layers = config.get('gcn_layers', 4)

        # Prior graph for KL regularization (optional)
        prior_graph = data_feature.get('prior_graph', None)
        if prior_graph is not None:
            self.register_buffer('prior_graph', torch.FloatTensor(prior_graph))
        else:
            self.prior_graph = None

        # Build model components
        self._build_model()

        # Loss function
        self.ce_criterion = nn.CrossEntropyLoss()
        self.kl_criterion = nn.KLDivLoss(reduction='batchmean')

        # Initialize embedding cache for memory-efficient evaluation
        self._cached_item_embs = None
        self._cached_support = None

    def train(self, mode=True):
        """
        Override train() to invalidate embedding cache when switching modes.

        This ensures that:
        - When switching to eval mode, the cache will be recomputed on first forward pass
        - When switching back to training mode, gradients flow through fresh AGCN computation

        Args:
            mode: If True, set to training mode; if False, set to evaluation mode

        Returns:
            self
        """
        # Invalidate cache when mode changes to ensure fresh embeddings
        if mode != self.training:
            self._cached_item_embs = None
            self._cached_support = None
            self.item_embs = None
        return super().train(mode)

    def _build_model(self):
        """Build all model components."""

        # Item embedding layer (with padding token at index 0)
        self.item_emb = nn.Embedding(
            self.num_items + 1, self.hidden_units, padding_idx=0
        )
        self.item_emb_dropout = nn.Dropout(p=self.dropout_rate)

        # Adaptive Graph Convolutional Network
        self.gcn = AGCN(
            input_dim=self.hidden_units,
            output_dim=self.hidden_units,
            layer=self.gcn_layers,
            dropout=self.dropout_rate
        )

        # Positional embeddings
        self.abs_pos_K_emb = nn.Embedding(self.maxlen, self.hidden_units)
        self.abs_pos_V_emb = nn.Embedding(self.maxlen, self.hidden_units)

        # Time relation embeddings
        self.time_matrix_K_emb = nn.Embedding(self.time_span + 1, self.hidden_units)
        self.time_matrix_V_emb = nn.Embedding(self.time_span + 1, self.hidden_units)

        # Distance relation embeddings
        self.dis_matrix_K_emb = nn.Embedding(self.dis_span + 1, self.hidden_units)
        self.dis_matrix_V_emb = nn.Embedding(self.dis_span + 1, self.hidden_units)

        # Dropout layers for embeddings
        self.abs_pos_K_emb_dropout = nn.Dropout(p=self.dropout_rate)
        self.abs_pos_V_emb_dropout = nn.Dropout(p=self.dropout_rate)
        self.time_matrix_K_dropout = nn.Dropout(p=self.dropout_rate)
        self.time_matrix_V_dropout = nn.Dropout(p=self.dropout_rate)
        self.dis_matrix_K_dropout = nn.Dropout(p=self.dropout_rate)
        self.dis_matrix_V_dropout = nn.Dropout(p=self.dropout_rate)

        # Attention blocks
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        for _ in range(self.num_blocks):
            new_attn_layernorm = nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = TimeAwareMultiHeadAttention(
                self.hidden_units,
                self.num_heads,
                self.dropout_rate,
                self.device
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(
                self.hidden_units, self.dropout_rate
            )
            self.forward_layers.append(new_fwd_layer)

        self.last_layernorm = nn.LayerNorm(self.hidden_units, eps=1e-8)

    def _mask_adjacency(self, adj, epsilon=0, mask_value=-1e16):
        """Apply mask to adjacency matrix for softmax."""
        mask = (adj > epsilon).detach().float()
        update_adj = adj * mask + (1 - mask) * mask_value
        return update_adj

    def seq2feats(self, log_seqs, time_matrices, dis_matrices, item_embs):
        """
        Convert input sequences to feature representations.

        Args:
            log_seqs: Input location sequences (batch, maxlen)
            time_matrices: Time relation matrices (batch, maxlen, maxlen)
            dis_matrices: Distance relation matrices (batch, maxlen, maxlen)
            item_embs: Enhanced item embeddings from AGCN

        Returns:
            Sequence feature representations (batch, maxlen, hidden_units)
        """
        # Get sequence embeddings
        seqs = item_embs[log_seqs.long(), :]
        seqs *= self.hidden_units ** 0.5
        seqs = self.item_emb_dropout(seqs)

        # Get positional embeddings
        batch_size = log_seqs.shape[0]
        seq_len = log_seqs.shape[1]
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        positions = positions.to(self.device)

        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)

        # Get time relation embeddings
        time_matrices = time_matrices.long().to(self.device)
        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        # Get distance relation embeddings
        dis_matrices = dis_matrices.long().to(self.device)
        dis_matrix_K = self.dis_matrix_K_emb(dis_matrices)
        dis_matrix_V = self.dis_matrix_V_emb(dis_matrices)
        dis_matrix_K = self.dis_matrix_K_dropout(dis_matrix_K)
        dis_matrix_V = self.dis_matrix_V_dropout(dis_matrix_V)

        # Create masks
        timeline_mask = (log_seqs == 0).to(self.device)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))

        # Apply attention blocks
        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs = self.attention_layers[i](
                Q, seqs,
                timeline_mask, attention_mask,
                time_matrix_K, time_matrix_V,
                dis_matrix_K, dis_matrix_V,
                abs_pos_K, abs_pos_V
            )
            seqs = Q + mha_outputs

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def _compute_relation_matrices(self, batch):
        """
        Compute time and distance relation matrices from batch data.

        Args:
            batch: LibCity batch object

        Returns:
            time_matrices: Time relation matrices (batch, maxlen, maxlen)
            dis_matrices: Distance relation matrices (batch, maxlen, maxlen)
        """
        loc_seq = batch['current_loc']
        batch_size = loc_seq.shape[0]
        seq_len = loc_seq.shape[1]

        # Initialize matrices
        time_matrices = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.long)
        dis_matrices = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.long)

        # Get time sequence if available
        if 'current_tim' in batch.data:
            time_seq = batch['current_tim']
            for b in range(batch_size):
                t_seq = time_seq[b].cpu().numpy() if isinstance(time_seq, torch.Tensor) else np.array(time_seq[b])
                time_matrices[b] = torch.from_numpy(compute_time_matrix(t_seq, self.time_span))

        # Get coordinates if available
        if 'current_coord' in batch.data:
            coords = batch['current_coord']
            for b in range(batch_size):
                c = coords[b].cpu().numpy() if isinstance(coords, torch.Tensor) else np.array(coords[b])
                dis_matrices[b] = torch.from_numpy(compute_distance_matrix(c, self.dis_span))
        elif 'traj_spatial_mat' in batch.data:
            # Use pre-computed spatial matrix if available
            spatial_mat = batch['traj_spatial_mat']
            dis_matrices = torch.clamp(spatial_mat.long(), 0, self.dis_span)

        return time_matrices.to(self.device), dis_matrices.to(self.device)

    def _get_time_distance_matrices(self, batch):
        """
        Get time and distance matrices from batch, computing if not available.

        Args:
            batch: LibCity batch object

        Returns:
            time_matrices, dis_matrices
        """
        # Check if pre-computed matrices are available
        if 'time_matrix' in batch.data:
            time_matrices = batch['time_matrix']
            if not isinstance(time_matrices, torch.Tensor):
                time_matrices = torch.LongTensor(time_matrices)
            time_matrices = time_matrices.to(self.device)
        else:
            time_matrices = None

        if 'dis_matrix' in batch.data:
            dis_matrices = batch['dis_matrix']
            if not isinstance(dis_matrices, torch.Tensor):
                dis_matrices = torch.LongTensor(dis_matrices)
            dis_matrices = dis_matrices.to(self.device)
        else:
            dis_matrices = None

        # Compute if not available
        if time_matrices is None or dis_matrices is None:
            computed_time, computed_dis = self._compute_relation_matrices(batch)
            if time_matrices is None:
                time_matrices = computed_time
            if dis_matrices is None:
                dis_matrices = computed_dis

        return time_matrices, dis_matrices

    def forward(self, batch):
        """
        Forward pass.

        Args:
            batch: LibCity Batch object containing trajectory data.
                Required keys:
                    - 'current_loc': Location/POI sequences (batch_size, seq_len)
                Optional keys:
                    - 'current_tim': Time sequences (batch_size, seq_len)
                    - 'current_coord': GPS coordinates (batch_size, seq_len, 2)
                    - 'time_matrix': Pre-computed time matrices
                    - 'dis_matrix': Pre-computed distance matrices

        Returns:
            logits: POI prediction logits (batch * seq_len, num_items + 1)
            support: Learned adjacency matrix from AGCN
        """
        # Get location sequence
        loc_seq = batch['current_loc']
        if not isinstance(loc_seq, torch.Tensor):
            loc_seq = torch.LongTensor(loc_seq)
        loc_seq = loc_seq.to(self.device)

        # Clamp to valid range
        loc_seq = torch.clamp(loc_seq, 0, self.num_items)

        # Get time and distance matrices
        time_matrices, dis_matrices = self._get_time_distance_matrices(batch)

        # Apply AGCN to get enhanced embeddings
        # During evaluation (not training), reuse cached embeddings to save memory
        # This prevents CUDA OOM during validation when calculate_loss() is called
        if self.training:
            # Training mode: always recompute AGCN for gradient updates
            item_embs, support = self.gcn(self.item_emb)
            # Cache for potential use by predict() within same batch
            self._cached_item_embs = item_embs.detach()
            self._cached_support = support.detach()
        else:
            # Evaluation mode: reuse cached embeddings if available
            if hasattr(self, '_cached_item_embs') and self._cached_item_embs is not None:
                item_embs = self._cached_item_embs
                support = self._cached_support
            else:
                # Fallback: compute and cache if not available
                with torch.no_grad():
                    item_embs, support = self.gcn(self.item_emb)
                self._cached_item_embs = item_embs
                self._cached_support = support

        # Store for prediction (for backward compatibility)
        self.item_embs = item_embs

        # Get sequence features
        log_feats = self.seq2feats(loc_seq, time_matrices, dis_matrices, item_embs)

        # Compute logits for all positions
        fin_logits = log_feats.matmul(item_embs.transpose(0, 1))
        fin_logits = fin_logits.reshape(-1, fin_logits.shape[-1])

        return fin_logits, support

    def predict(self, batch):
        """
        Prediction method for LibCity.

        Args:
            batch: Input batch dictionary

        Returns:
            POI prediction scores for the last timestep of each sequence
        """
        # Get location sequence
        loc_seq = batch['current_loc']
        if not isinstance(loc_seq, torch.Tensor):
            loc_seq = torch.LongTensor(loc_seq)
        loc_seq = loc_seq.to(self.device)

        # Clamp to valid range
        loc_seq = torch.clamp(loc_seq, 0, self.num_items)

        # Get time and distance matrices
        time_matrices, dis_matrices = self._get_time_distance_matrices(batch)

        # Reuse cached embeddings to save memory during evaluation
        # Priority: _cached_item_embs (new cache) > item_embs (legacy cache) > recompute
        if hasattr(self, '_cached_item_embs') and self._cached_item_embs is not None:
            item_embs = self._cached_item_embs
        elif hasattr(self, 'item_embs') and self.item_embs is not None:
            item_embs = self.item_embs
        else:
            # Fallback: compute if cache is not available
            with torch.no_grad():
                item_embs, _ = self.gcn(self.item_emb)
            self._cached_item_embs = item_embs

        # Get sequence features
        log_feats = self.seq2feats(loc_seq, time_matrices, dis_matrices, item_embs)

        # Get final feature (last position)
        final_feat = log_feats[:, -1, :]

        # Compute logits
        logits = final_feat.matmul(item_embs.transpose(0, 1))

        # Return predictions excluding padding token (index 0)
        return logits[:, 1:]

    def calculate_loss(self, batch):
        """
        Calculate training loss.

        Loss = Cross-Entropy Loss + KL Divergence Regularization

        Args:
            batch: LibCity Batch object containing:
                - trajectory data for forward pass
                - 'target': Target POI indices (batch_size,)

        Returns:
            loss: Combined loss tensor
        """
        # Forward pass
        fin_logits, support = self.forward(batch)

        # Get target
        target = batch['target']
        if not isinstance(target, torch.Tensor):
            target = torch.LongTensor(target)
        target = target.to(self.device)

        # Get location sequence for masking
        loc_seq = batch['current_loc']
        if not isinstance(loc_seq, torch.Tensor):
            loc_seq = torch.LongTensor(loc_seq)
        loc_seq = loc_seq.to(self.device)

        # Reshape target for cross-entropy
        # The model predicts for each position, target should be the next position
        batch_size = loc_seq.shape[0]
        seq_len = loc_seq.shape[1]

        # Create position-wise targets (shifted sequence)
        pos_targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device)
        pos_targets[:, :-1] = loc_seq[:, 1:]
        pos_targets[:, -1] = target
        pos_targets = pos_targets.reshape(-1)

        # Find valid positions (non-padding)
        valid_mask = (pos_targets != 0)

        # Compute cross-entropy loss
        if valid_mask.sum() > 0:
            ce_loss = self.ce_criterion(
                fin_logits[valid_mask],
                pos_targets[valid_mask]
            )
        else:
            ce_loss = torch.tensor(0.0, device=self.device)

        # KL divergence regularization
        kl_loss = torch.tensor(0.0, device=self.device)
        if self.prior_graph is not None and self.kl_reg > 0:
            # Apply mask and compute softmax
            masked_support = self._mask_adjacency(support)
            masked_prior = self._mask_adjacency(self.prior_graph)

            log_support = torch.log(
                torch.softmax(masked_support, dim=-1) + 1e-9
            )
            prior_softmax = torch.softmax(masked_prior, dim=-1)

            kl_loss = self.kl_criterion(log_support, prior_softmax)

        # Combined loss
        loss = ce_loss + self.kl_reg * kl_loss

        return loss
