# coding: utf-8
"""
AGRAN: Attention-based Graph Recurrent Attention Network for Next POI Recommendation

This module adapts the AGRAN model for LibCity framework.

Key Innovations:
1. Adaptive Graph Convolutional Network (AGCN) for learning item relationships
2. Time-aware and distance-aware multi-head attention mechanism
3. KL divergence regularization with adaptive graph structure

Original Architecture:
- Item embeddings enhanced via AGCN (learns adaptive adjacency matrix from cosine similarity)
- Time-aware multi-head self-attention with relative time/distance encoding
- Position, time interval, and distance interval embeddings

Memory Optimization (v2.0):
- Sparse adjacency matrix computation using top-k neighbors per node
- Chunked similarity computation to avoid OOM on large graphs (60k+ POIs)
- Automatic switching between dense (small graphs) and sparse (large graphs) modes

Memory Optimization (v3.0):
- Adaptive AGCN gradient handling based on dataset size
- For large datasets (>40k POIs): Freeze AGCN gradients to avoid OOM during backward pass
- For small datasets: Keep gradient flow for better graph learning
- This eliminates OOM errors that occur when gradients flow through AGCN on 61k+ POI datasets

Required data_feature keys:
- loc_size: Number of POI locations (item_num)
- uid_size: Number of users (user_num)
- tim_size: Number of time slots (time_num)

Required config parameters:
- hidden_units: Hidden dimension size (default: 64)
- num_blocks: Number of transformer blocks (default: 3)
- num_heads: Number of attention heads (default: 2)
- dropout_rate: Dropout rate (default: 0.3)
- maxlen: Maximum sequence length (default: 50)
- time_span: Maximum time interval span for bucketing (default: 256)
- dis_span: Maximum distance interval span for bucketing (default: 256)
- gcn_layers: Number of AGCN layers (default: 4)
- kl_weight: Weight for KL divergence loss (default: 0.01)
- top_k_neighbors: Number of neighbors to keep per POI for sparse adjacency (default: 100)
- chunk_size: Chunk size for batched similarity computation (default: 1000)
- num_neg_samples: Number of negative samples for sampled softmax in training (default: 1000)
    This reduces memory from O(batch * seq * num_locations) to O(batch * seq * num_samples)
- agcn_refresh_epochs: How often to recompute the adaptive graph (default: 1)
    Setting this to N means the graph is recomputed every N epochs.
    Higher values speed up training but may reduce graph adaptiveness.
"""

import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from libcity.model.abstract_model import AbstractModel

FLOAT_MIN = -sys.float_info.max


# ==============================================================================
# Adaptive Graph Convolutional Network (AGCN)
# ==============================================================================

class AGCN(nn.Module):
    """
    Adaptive Graph Convolutional Network that learns the graph structure
    from item embeddings using weighted cosine similarity.

    This module dynamically computes an adjacency matrix based on the learned
    item embeddings, then applies multi-layer graph convolution to enhance
    item representations with neighborhood information.

    Memory Optimization:
        For large graphs (e.g., 60k+ POIs), computing the full NxN adjacency matrix
        would require ~15GB GPU memory. Instead, we use sparse top-k computation:
        - Only compute and store the top-k neighbors per node
        - Use sparse matrix representation for graph convolution
        - This reduces memory from O(N^2) to O(N*k)
    """

    def __init__(self, input_dim, output_dim, layer=3, dropout=0.2, bias=False,
                 device='cpu', top_k_neighbors=100, chunk_size=1000):
        """
        Initialize AGCN.

        Args:
            input_dim: Input embedding dimension
            output_dim: Output embedding dimension
            layer: Number of GCN propagation layers
            dropout: Dropout rate for adjacency matrix
            bias: Whether to use bias
            device: Device to run the computation on
            top_k_neighbors: Number of neighbors to keep per node (for sparse computation)
            chunk_size: Chunk size for computing similarity in batches (memory optimization)
        """
        super(AGCN, self).__init__()

        self.dropout = dropout
        self.layer_num = layer
        self.device = device
        self.top_k_neighbors = top_k_neighbors
        self.chunk_size = chunk_size

        # Learnable weight matrix for computing weighted cosine similarity
        self.cos_weight = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty(input_dim, output_dim))
        )

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))

    def get_neighbor_hard_threshold(self, adj, epsilon=0, mask_value=0):
        """
        Apply hard thresholding to adjacency matrix and normalize.

        Args:
            adj: Raw adjacency matrix (N, N)
            epsilon: Threshold value
            mask_value: Value to fill for masked entries

        Returns:
            Tuple of (normalized adjacency matrix, raw adjacency matrix)
        """
        # Create binary mask based on threshold
        mask = (adj > epsilon).detach().float()
        raw_adj = adj * mask + (1 - mask) * mask_value

        # Symmetric normalization: D^{-0.5} * A * D^{-0.5}
        adj_dig = torch.clamp(torch.pow(torch.sum(raw_adj, dim=-1, keepdim=True), 0.5), min=1e-12)
        update_adj = raw_adj / adj_dig / adj_dig.transpose(-1, -2)

        return update_adj, raw_adj

    def get_neighbor_soft_row_threshold(self, adj, epsilon=100, device=None):
        """Apply top-k row-wise thresholding."""
        top_k = min(epsilon, adj.size(-1))
        _, index = torch.topk(adj, top_k, dim=-1)
        update_adj = torch.zeros_like(adj).scatter_(-1, index, 1)
        return update_adj

    def get_neighbor_soft_threshold(self, adj, epsilon, device=None):
        """Apply top-k global thresholding."""
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
        """
        Compute weighted cosine similarity matrix.

        The learnable weight matrix transforms embeddings before computing
        cosine similarity, allowing the model to learn which dimensions
        are important for item relationships.

        WARNING: This method creates a dense NxN matrix and should only be used
        for small graphs. For large graphs, use compute_sparse_adjacency instead.
        """
        emb = torch.matmul(emb, self.cos_weight)
        node_norm = F.normalize(emb, p=2, dim=-1)
        cos_adj = torch.mm(node_norm, node_norm.transpose(-1, -2))
        return cos_adj

    def compute_sparse_adjacency(self, emb):
        """
        Compute sparse adjacency matrix using top-k neighbors per node.

        This is the memory-efficient version that only keeps top-k neighbors
        per node instead of the full NxN matrix.

        Args:
            emb: Node embeddings (N, hidden_dim)

        Returns:
            Tuple of (sparse adjacency matrix as COO tensor, support_loss tensor)
        """
        N = emb.shape[0]
        top_k = min(self.top_k_neighbors, N)

        # Transform embeddings with learnable weight
        emb_transformed = torch.matmul(emb, self.cos_weight)
        node_norm = F.normalize(emb_transformed, p=2, dim=-1)

        # Process in chunks to avoid OOM
        chunk_size = min(self.chunk_size, N)
        all_indices = []
        all_values = []

        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)
            chunk_emb = node_norm[start_idx:end_idx]  # (chunk_size, hidden)

            # Compute similarity of this chunk with all nodes
            sim_chunk = torch.mm(chunk_emb, node_norm.transpose(0, 1))  # (chunk_size, N)

            # Get top-k neighbors for each node in chunk
            topk_values, topk_indices = torch.topk(sim_chunk, top_k, dim=-1)  # (chunk_size, top_k)

            # Create row indices for this chunk
            chunk_rows = torch.arange(start_idx, end_idx, device=emb.device).unsqueeze(1)
            chunk_rows = chunk_rows.expand(-1, top_k).reshape(-1)

            # Column indices are the topk_indices
            chunk_cols = topk_indices.reshape(-1)

            # Values are the similarity scores (apply threshold > 0)
            chunk_vals = topk_values.reshape(-1)
            mask = chunk_vals > 0
            chunk_rows = chunk_rows[mask]
            chunk_cols = chunk_cols[mask]
            chunk_vals = chunk_vals[mask]

            all_indices.append(torch.stack([chunk_rows, chunk_cols], dim=0))
            all_values.append(chunk_vals)

        # Concatenate all chunks
        indices = torch.cat(all_indices, dim=1)
        values = torch.cat(all_values, dim=0)

        # Symmetric normalization: D^{-0.5} * A * D^{-0.5}
        # First compute degree for each node
        row_indices = indices[0]
        degree = torch.zeros(N, device=emb.device)
        degree.scatter_add_(0, row_indices, values)

        # Normalize: values / sqrt(degree[row]) / sqrt(degree[col])
        deg_inv_sqrt = torch.clamp(degree, min=1e-12).pow(-0.5)
        row_deg = deg_inv_sqrt[indices[0]]
        col_deg = deg_inv_sqrt[indices[1]]
        normalized_values = values * row_deg * col_deg

        # Create sparse tensor
        sparse_adj = torch.sparse_coo_tensor(
            indices, normalized_values, (N, N), device=emb.device
        ).coalesce()

        # Compute support_loss for KL regularization (mean of edge values)
        support_loss = values.mean()

        return sparse_adj, support_loss

    def sparse_graph_conv(self, sparse_adj, x):
        """
        Perform graph convolution using sparse adjacency matrix.

        Args:
            sparse_adj: Sparse adjacency matrix (N, N) in COO format
            x: Node features (N, hidden_dim)

        Returns:
            Updated node features (N, hidden_dim)
        """
        return torch.sparse.mm(sparse_adj, x)

    def forward(self, inputs, use_sparse=None):
        """
        Forward pass through AGCN.

        Args:
            inputs: nn.Embedding layer with shape (num_items+1, hidden_dim)
                   where index 0 is padding
            use_sparse: If None, automatically decide based on graph size.
                       If True, force sparse computation.
                       If False, force dense computation.

        Returns:
            Tuple of (enhanced embeddings, support matrix/loss for KL loss)
        """
        # Extract embeddings (excluding padding at index 0)
        x = inputs.weight[1:, :]
        N = x.shape[0]

        # Automatically decide whether to use sparse or dense computation
        # Threshold: if N > 10000, use sparse to avoid OOM
        if use_sparse is None:
            use_sparse = (N > 10000)

        if use_sparse:
            # Memory-efficient sparse computation
            sparse_adj, support_loss = self.compute_sparse_adjacency(x)

            # Apply dropout to adjacency values during training
            if self.training:
                # For sparse dropout, we randomly zero out some edges
                indices = sparse_adj.indices()
                values = sparse_adj.values()
                dropout_mask = torch.rand_like(values) > self.dropout
                values = values * dropout_mask.float() / (1 - self.dropout + 1e-12)
                sparse_adj = torch.sparse_coo_tensor(
                    indices, values, sparse_adj.shape, device=x.device
                ).coalesce()

            # Multi-layer graph convolution with residual connections
            x_fin = [x]
            layer = x
            for f in range(self.layer_num):
                layer = self.sparse_graph_conv(sparse_adj, layer)
                layer = torch.tanh(layer)
                x_fin.append(layer)

            # Stack and sum across layers
            x_fin = torch.stack(x_fin, dim=1)
            out = torch.sum(x_fin, dim=1)

            if self.bias is not None:
                out = out + self.bias

            # Prepend padding embedding back
            fin_out = torch.cat([inputs.weight[0, :].unsqueeze(dim=0), out], dim=0)

            return fin_out, support_loss
        else:
            # Original dense computation (for small graphs)
            support = self.weight_cosine_matrix_div(x)
            support, support_loss = self.get_neighbor_hard_threshold(support)

            # Apply dropout to adjacency matrix during training
            if self.training:
                support = F.dropout(support, self.dropout)

            # Multi-layer graph convolution with residual connections
            x_fin = [x]
            layer = x
            for f in range(self.layer_num):
                layer = torch.matmul(support, layer)
                layer = torch.tanh(layer)
                x_fin.append(layer)

            # Stack and sum across layers
            x_fin = torch.stack(x_fin, dim=1)
            out = torch.sum(x_fin, dim=1)

            if self.bias is not None:
                out = out + self.bias

            # Prepend padding embedding back
            fin_out = torch.cat([inputs.weight[0, :].unsqueeze(dim=0), out], dim=0)

            return fin_out, support_loss


# ==============================================================================
# Point-wise Feed Forward Network
# ==============================================================================

class PointWiseFeedForward(nn.Module):
    """
    Point-wise feed forward network as in Transformer.
    Uses 1D convolutions for efficiency.
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
            inputs: (batch, seq_len, hidden_units)

        Returns:
            Output with same shape as input
        """
        outputs = self.dropout2(
            self.conv2(
                self.relu(
                    self.dropout1(
                        self.conv1(inputs.transpose(-1, -2))
                    )
                )
            )
        )
        outputs = outputs.transpose(-1, -2)
        outputs += inputs  # Residual connection
        return outputs


# ==============================================================================
# Time-Aware Multi-Head Attention
# ==============================================================================

class TimeAwareMultiHeadAttention(nn.Module):
    """
    Time-aware and distance-aware multi-head self-attention.

    This attention mechanism incorporates:
    1. Absolute position embeddings
    2. Relative time interval embeddings
    3. Relative distance interval embeddings

    The attention scores are computed as:
        score = Q*K^T + Q*AbsPos^T + TimeMatrix*Q + DistanceMatrix*Q
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
        self.device = device

    def forward(self, queries, keys, time_mask, attn_mask,
                time_matrix_K, time_matrix_V, dis_matrix_K, dis_matrix_V,
                abs_pos_K, abs_pos_V):
        """
        Forward pass of time-aware multi-head attention.

        Args:
            queries: Query tensor (batch, seq_len, hidden)
            keys: Key tensor (batch, seq_len, hidden)
            time_mask: Mask for padding positions (batch, seq_len)
            attn_mask: Causal attention mask (seq_len, seq_len)
            time_matrix_K: Time interval embeddings for keys (batch, seq_len, seq_len, hidden)
            time_matrix_V: Time interval embeddings for values
            dis_matrix_K: Distance interval embeddings for keys
            dis_matrix_V: Distance interval embeddings for values
            abs_pos_K: Absolute position embeddings for keys (batch, seq_len, hidden)
            abs_pos_V: Absolute position embeddings for values

        Returns:
            Attention output (batch, seq_len, hidden)
        """
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        # Split into multiple heads
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        # Split position/time/distance embeddings for multi-head
        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)
        dis_matrix_K_ = torch.cat(torch.split(dis_matrix_K, self.head_size, dim=3), dim=0)
        dis_matrix_V_ = torch.cat(torch.split(dis_matrix_V, self.head_size, dim=3), dim=0)
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

        # Compute attention scores with time and distance awareness
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))  # Standard Q*K^T
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))  # + Q*AbsPos^T
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)  # + Time*Q
        attn_weights += dis_matrix_K_.matmul((Q_.unsqueeze(-1))).squeeze(-1)  # + Dist*Q

        # Scale by sqrt(d_k)
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        # Apply masks
        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)

        paddings = torch.ones(attn_weights.shape) * (-2**32 + 1)
        paddings = paddings.to(self.device)
        attn_weights = torch.where(time_mask, paddings, attn_weights)
        attn_weights = torch.where(attn_mask, paddings, attn_weights)

        attn_weights = self.softmax(attn_weights)
        attn_weights = self.dropout(attn_weights)

        # Compute output with time and distance-aware values
        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)
        outputs += attn_weights.unsqueeze(2).matmul(dis_matrix_V_).reshape(outputs.shape).squeeze(2)

        # Concatenate heads
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2)

        return outputs


# ==============================================================================
# Main AGRAN Model (LibCity Adapter)
# ==============================================================================

class AGRAN(AbstractModel):
    """
    AGRAN: Attention-based Graph Recurrent Attention Network for Next POI Recommendation

    This model combines:
    1. Adaptive Graph Convolutional Network (AGCN) for learning item relationships
    2. Time-aware and distance-aware multi-head self-attention
    3. KL divergence regularization with the learned graph structure

    The model is adapted for LibCity framework with proper handling of:
    - LibCity's batch data format
    - Configuration and data_feature dictionaries
    - predict() and calculate_loss() methods
    """

    def __init__(self, config, data_feature):
        super(AGRAN, self).__init__(config, data_feature)

        # Device configuration
        self.device = config.get('device', 'cpu')

        # Data dimensions from data_feature
        self.user_num = data_feature.get('uid_size', 1000)
        self.item_num = data_feature.get('loc_size', 5000)
        self.time_num = data_feature.get('tim_size', 48)

        # Model hyperparameters from config
        self.hidden_units = config.get('hidden_units', 64)
        self.num_blocks = config.get('num_blocks', 3)
        self.num_heads = config.get('num_heads', 2)
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.maxlen = config.get('maxlen', 50)
        self.time_span = config.get('time_span', 256)
        self.dis_span = config.get('dis_span', 256)
        self.gcn_layers = config.get('gcn_layers', 4)
        self.kl_weight = config.get('kl_weight', 0.01)
        # Sparse adjacency parameters for memory optimization
        self.top_k_neighbors = config.get('top_k_neighbors', 100)
        self.chunk_size = config.get('chunk_size', 1000)
        # Negative sampling parameter for memory-efficient training
        # Reduces memory from O(batch * seq * num_items) to O(batch * num_neg_samples)
        self.num_neg_samples = config.get('num_neg_samples', 1000)
        # AGCN caching parameters for training performance optimization
        # The graph is recomputed every agcn_refresh_epochs epochs instead of every batch
        self.agcn_refresh_epochs = config.get('agcn_refresh_epochs', 1)

        # AGCN cache attributes - compute graph once per epoch (or per N epochs)
        # instead of every batch to avoid heavy CPU computation bottleneck
        self.cached_item_embs = None
        self.cached_support = None
        self.cache_epoch = -1  # Track which epoch the cache was computed for
        self._current_epoch = 0  # Current training epoch (updated externally)

        # Item embedding layer (with padding at index 0)
        self.item_emb = nn.Embedding(
            self.item_num + 1,
            self.hidden_units,
            padding_idx=0
        )
        self.item_emb_dropout = nn.Dropout(p=self.dropout_rate)

        # Adaptive Graph Convolutional Network
        self.gcn = AGCN(
            input_dim=self.hidden_units,
            output_dim=self.hidden_units,
            layer=self.gcn_layers,
            dropout=self.dropout_rate,
            device=self.device,
            top_k_neighbors=self.top_k_neighbors,
            chunk_size=self.chunk_size
        )

        # Absolute position embeddings
        self.abs_pos_K_emb = nn.Embedding(self.maxlen, self.hidden_units)
        self.abs_pos_V_emb = nn.Embedding(self.maxlen, self.hidden_units)

        # Time interval embeddings
        self.time_matrix_K_emb = nn.Embedding(self.time_span + 1, self.hidden_units)
        self.time_matrix_V_emb = nn.Embedding(self.time_span + 1, self.hidden_units)

        # Distance interval embeddings
        self.dis_matrix_K_emb = nn.Embedding(self.dis_span + 1, self.hidden_units)
        self.dis_matrix_V_emb = nn.Embedding(self.dis_span + 1, self.hidden_units)

        # Dropout layers for embeddings
        self.abs_pos_K_emb_dropout = nn.Dropout(p=self.dropout_rate)
        self.abs_pos_V_emb_dropout = nn.Dropout(p=self.dropout_rate)
        self.time_matrix_K_dropout = nn.Dropout(p=self.dropout_rate)
        self.time_matrix_V_dropout = nn.Dropout(p=self.dropout_rate)
        self.dis_matrix_K_dropout = nn.Dropout(p=self.dropout_rate)
        self.dis_matrix_V_dropout = nn.Dropout(p=self.dropout_rate)

        # Transformer blocks
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(self.hidden_units, eps=1e-8)

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

            new_fwd_layer = PointWiseFeedForward(self.hidden_units, self.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        # Store item embeddings after GCN enhancement
        self.item_embs = None

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def set_epoch(self, epoch):
        """
        Set the current training epoch. Called by the trainer at the start of each epoch.
        This is used to determine when to refresh the AGCN cache.

        Args:
            epoch: Current epoch number (0-indexed)
        """
        self._current_epoch = epoch

    def invalidate_cache(self):
        """
        Invalidate the AGCN cache, forcing recomputation on next forward pass.
        Call this when you want to force graph recomputation (e.g., at epoch boundaries).
        """
        self.cached_item_embs = None
        self.cached_support = None
        self.cache_epoch = -1

    def _should_refresh_cache(self):
        """
        Determine if the AGCN cache should be refreshed.

        Returns:
            bool: True if cache needs to be refreshed
        """
        if self.cached_item_embs is None:
            return True
        # Refresh if we're in a new epoch boundary (based on refresh frequency)
        epochs_since_cache = self._current_epoch - self.cache_epoch
        return epochs_since_cache >= self.agcn_refresh_epochs

    def _get_agcn_embeddings(self, require_grad=True):
        """
        Get AGCN-enhanced item embeddings, using cache when possible.

        During training, the cache is refreshed based on agcn_refresh_epochs.
        During inference (eval mode), always use cached embeddings or compute once.

        Memory Optimization (v3.0):
            For large datasets (>40k POIs), AGCN gradient computation during backward
            pass can require 13+ GB memory, causing OOM. We use adaptive gradient handling:
            - Large datasets (>40k POIs): Disable gradients for AGCN to avoid OOM
            - Small datasets (<=40k POIs): Keep gradient flow for better graph learning

        Args:
            require_grad: Whether gradients are needed (True during cache refresh in training)

        Returns:
            Tuple of (item_embs, support) where:
                - item_embs: GCN-enhanced embeddings (num_items+1, hidden_dim)
                - support: Support loss for KL regularization
        """
        # Threshold for large dataset detection (40k POIs)
        # Above this threshold, AGCN backward pass causes OOM (~13.86 GB required)
        LARGE_DATASET_THRESHOLD = 40000

        # In eval mode, compute once and cache
        if not self.training:
            if self.cached_item_embs is None:
                with torch.no_grad():
                    item_embs, support = self.gcn(self.item_emb)
                    self.cached_item_embs = item_embs.detach()
                    self.cached_support = support.detach() if isinstance(support, torch.Tensor) else support
            return self.cached_item_embs, self.cached_support

        # In training mode, check if we need to refresh the cache
        if self._should_refresh_cache():
            # Adaptive gradient handling based on dataset size
            # For large datasets, disable gradients to prevent OOM during AGCN backward pass
            if self.item_num > LARGE_DATASET_THRESHOLD:
                # Large dataset: compute AGCN without gradients to avoid OOM
                # Graph structure is relatively stable for large POI sets,
                # so freezing AGCN gradients has minimal impact on model quality
                with torch.no_grad():
                    item_embs, support = self.gcn(self.item_emb)
                # Cache and return detached embeddings
                self.cached_item_embs = item_embs.detach()
                self.cached_support = support.detach() if isinstance(support, torch.Tensor) else support
                self.cache_epoch = self._current_epoch
                return self.cached_item_embs, self.cached_support
            else:
                # Small dataset: compute AGCN with gradients for better graph learning
                item_embs, support = self.gcn(self.item_emb)
                # Cache the detached version for subsequent batches in this epoch
                self.cached_item_embs = item_embs.detach()
                self.cached_support = support.detach() if isinstance(support, torch.Tensor) else support
                self.cache_epoch = self._current_epoch
                # Return the version with gradients for this batch
                return item_embs, support
        else:
            # Use cached embeddings (no gradient computation for AGCN)
            # The cached embeddings are detached, so AGCN won't receive gradients
            # This is intentional - we only update AGCN once per refresh period
            return self.cached_item_embs, self.cached_support

    def seq2feats(self, user_ids, log_seqs, time_matrices, dis_matrices, item_embs):
        """
        Convert input sequences to feature representations through
        time-aware multi-head self-attention.

        Args:
            user_ids: User IDs (batch,)
            log_seqs: Location sequences (batch, seq_len)
            time_matrices: Time interval matrices (batch, seq_len, seq_len)
            dis_matrices: Distance interval matrices (batch, seq_len, seq_len)
            item_embs: GCN-enhanced item embeddings (num_items+1, hidden)

        Returns:
            Sequence features (batch, seq_len, hidden)
        """
        # Get sequence embeddings from enhanced item embeddings
        seqs = item_embs[log_seqs.long(), :]
        seqs *= item_embs.shape[1] ** 0.5  # Scale by sqrt(d)

        seqs = self.item_emb_dropout(seqs)

        # Position embeddings
        batch_size, seq_len = log_seqs.shape
        positions = np.tile(np.array(range(seq_len)), [batch_size, 1])
        positions = torch.LongTensor(positions).to(self.device)
        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)

        # Time interval embeddings
        time_matrices = time_matrices.long().to(self.device)
        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        # Distance interval embeddings
        dis_matrices = dis_matrices.long().to(self.device)
        dis_matrix_K = self.dis_matrix_K_emb(dis_matrices)
        dis_matrix_V = self.dis_matrix_V_emb(dis_matrices)
        dis_matrix_K = self.dis_matrix_K_dropout(dis_matrix_K)
        dis_matrix_V = self.dis_matrix_V_dropout(dis_matrix_V)

        # Timeline mask (mask padding positions)
        timeline_mask = torch.BoolTensor(log_seqs.cpu().numpy() == 0).to(self.device)
        seqs *= ~timeline_mask.unsqueeze(-1)

        # Causal attention mask (upper triangular)
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))

        # Apply transformer blocks
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

    def forward(self, batch, pos_seqs=None, neg_seqs=None):
        """
        Forward pass through AGRAN.

        Args:
            batch: Dictionary containing:
                - 'current_loc': (batch_size, seq_len) - Location sequence
                - 'uid': (batch_size,) - User IDs
                - 'time_matrix': (batch_size, seq_len, seq_len) - Time intervals (optional)
                - 'dis_matrix': (batch_size, seq_len, seq_len) - Distance intervals (optional)
            pos_seqs: Positive samples for training (optional)
            neg_seqs: Negative samples for training (optional)

        Returns:
            Tuple of (pos_logits, neg_logits, fin_logits, padding_emb, support)
            or just fin_logits if not training with pos/neg samples
        """
        # Extract data from batch
        log_seqs = batch['current_loc']
        batch_size, seq_len = log_seqs.shape

        # Handle user IDs
        if 'uid' in batch.data:
            user_ids = batch['uid']
            if user_ids.dim() > 1:
                user_ids = user_ids.squeeze()
        else:
            user_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Get or generate time matrices
        if 'time_matrix' in batch.data:
            time_matrices = batch['time_matrix']
        else:
            # Generate dummy time matrices if not provided
            time_matrices = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.long, device=self.device)

        # Get or generate distance matrices
        if 'dis_matrix' in batch.data:
            dis_matrices = batch['dis_matrix']
        else:
            # Generate dummy distance matrices if not provided
            dis_matrices = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.long, device=self.device)

        # Clamp matrices to valid range
        time_matrices = torch.clamp(time_matrices, 0, self.time_span)
        dis_matrices = torch.clamp(dis_matrices, 0, self.dis_span)

        # Enhanced item embeddings through AGCN (using cache for performance)
        # The cache is refreshed based on agcn_refresh_epochs to avoid recomputing
        # the expensive sparse adjacency computation on every batch
        item_embs, support = self._get_agcn_embeddings()
        self.item_embs = item_embs

        # Get sequence features
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices, dis_matrices, item_embs)

        if pos_seqs is not None and neg_seqs is not None:
            # Training with positive and negative samples
            pos_embs = item_embs[pos_seqs.long().to(self.device), :]
            neg_embs = item_embs[neg_seqs.long().to(self.device), :]

            pos_logits = (log_feats * pos_embs).sum(dim=-1)
            neg_logits = (log_feats * neg_embs).sum(dim=-1)

            fin_logits = log_feats.matmul(item_embs.transpose(0, 1))
            fin_logits = fin_logits.reshape(-1, fin_logits.shape[-1])

            return pos_logits, neg_logits, fin_logits, self.item_embs[0], support
        else:
            # Inference mode - compute logits for all items
            fin_logits = log_feats.matmul(item_embs.transpose(0, 1))
            return fin_logits, support

    def predict(self, batch):
        """
        Predict next POI for each sequence.

        Args:
            batch: Dictionary containing input data

        Returns:
            torch.Tensor: POI prediction scores (batch_size, loc_size)
                          Returns predictions for the last position of each sequence.
        """
        fin_logits, _ = self.forward(batch)

        # Get predictions for the last position
        if fin_logits.dim() == 3:
            # (batch, seq_len, num_items) -> take last position
            last_pred = fin_logits[:, -1, :]
        else:
            # Already flattened
            batch_size = batch['current_loc'].shape[0]
            seq_len = batch['current_loc'].shape[1]
            fin_logits = fin_logits.view(batch_size, seq_len, -1)
            last_pred = fin_logits[:, -1, :]

        return last_pred

    def calculate_loss(self, batch):
        """
        Calculate combined loss using sampled softmax (cross-entropy + KL divergence).

        This method uses negative sampling to avoid computing the full softmax over
        all POIs, which would cause OOM errors on datasets with many locations (e.g., 61k POIs).

        Memory optimization:
            Instead of computing logits for all items (batch * seq * num_items),
            we only compute logits for:
            - The positive target (1 per sample)
            - K sampled negative locations (num_neg_samples per sample)
            This reduces memory from O(batch * num_items) to O(batch * num_neg_samples).

        Args:
            batch: Dictionary containing:
                - 'current_loc': (batch_size, seq_len) - Input location sequence
                - 'target': (batch_size,) - Target location
                - Other optional fields

        Returns:
            torch.Tensor: Combined loss (scalar)
        """
        # Extract data from batch
        log_seqs = batch['current_loc']
        batch_size, seq_len = log_seqs.shape

        # Handle user IDs
        if 'uid' in batch.data:
            user_ids = batch['uid']
            if user_ids.dim() > 1:
                user_ids = user_ids.squeeze()
        else:
            user_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Get or generate time matrices
        if 'time_matrix' in batch.data:
            time_matrices = batch['time_matrix']
        else:
            time_matrices = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.long, device=self.device)

        # Get or generate distance matrices
        if 'dis_matrix' in batch.data:
            dis_matrices = batch['dis_matrix']
        else:
            dis_matrices = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.long, device=self.device)

        # Clamp matrices to valid range
        time_matrices = torch.clamp(time_matrices, 0, self.time_span)
        dis_matrices = torch.clamp(dis_matrices, 0, self.dis_span)

        # Enhanced item embeddings through AGCN (using cache for performance)
        # The cache is refreshed based on agcn_refresh_epochs to avoid recomputing
        # the expensive sparse adjacency computation on every batch
        item_embs, support = self._get_agcn_embeddings()
        self.item_embs = item_embs

        # Get sequence features - only last position needed for loss
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices, dis_matrices, item_embs)
        # Take only last position: (batch_size, hidden_units)
        last_feats = log_feats[:, -1, :]

        # Get target
        target = batch['target']
        if target.dim() > 1:
            target = target.squeeze()
        target = target.long().to(self.device)

        # =====================================================================
        # Sampled Softmax with Negative Sampling
        # =====================================================================
        # Instead of computing logits for all items (causing OOM),
        # we compute logits only for positive + sampled negative items.

        # Number of negative samples (capped by available items)
        num_neg = min(self.num_neg_samples, self.item_num - 1)

        # Sample negative indices uniformly (excluding padding index 0)
        # Shape: (batch_size, num_neg)
        neg_indices = torch.randint(1, self.item_num + 1, (batch_size, num_neg), device=self.device)

        # Ensure negatives do not include the positive target
        # Replace any collision with a random replacement
        target_expanded = target.unsqueeze(1).expand(-1, num_neg)
        collision_mask = (neg_indices == target_expanded)
        # Generate replacement indices for collisions
        replacement = torch.randint(1, self.item_num + 1, (batch_size, num_neg), device=self.device)
        neg_indices = torch.where(collision_mask, replacement, neg_indices)

        # Get embeddings for positive and negative samples
        # Positive: (batch_size, hidden_units)
        pos_embs = item_embs[target]
        # Negative: (batch_size, num_neg, hidden_units)
        neg_embs = item_embs[neg_indices]

        # Compute scores
        # Positive score: (batch_size,)
        pos_scores = (last_feats * pos_embs).sum(dim=-1, keepdim=True)  # (batch_size, 1)
        # Negative scores: (batch_size, num_neg)
        neg_scores = torch.bmm(neg_embs, last_feats.unsqueeze(-1)).squeeze(-1)  # (batch_size, num_neg)

        # Concatenate: positive at index 0, negatives at indices 1 to num_neg
        # Shape: (batch_size, 1 + num_neg)
        all_scores = torch.cat([pos_scores, neg_scores], dim=1)

        # Cross-entropy loss: target is always index 0 (positive sample)
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        ce_loss = F.cross_entropy(all_scores, labels)

        # =====================================================================
        # KL Divergence Regularization
        # =====================================================================
        if support is not None and self.kl_weight > 0:
            eps = 1e-10
            # Support can be a scalar (sparse mode) or tensor (dense mode)
            if isinstance(support, torch.Tensor):
                if support.dim() == 0:
                    # Sparse mode: support is already mean similarity (scalar)
                    # Use it directly as regularization term
                    kl_loss = -support  # Encourage higher similarity among neighbors
                else:
                    # Dense mode: support is adjacency matrix
                    support_flat = support.flatten()
                    support_prob = F.softmax(support_flat, dim=0) + eps
                    uniform_prob = torch.ones_like(support_prob) / len(support_prob)
                    kl_loss = (support_prob * (torch.log(support_prob) - torch.log(uniform_prob))).sum()
            else:
                # Scalar value
                kl_loss = torch.tensor(0.0, device=self.device)

            total_loss = ce_loss + self.kl_weight * kl_loss
        else:
            total_loss = ce_loss

        return total_loss
