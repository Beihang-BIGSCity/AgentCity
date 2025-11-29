"""
Multi-Level Causal Attention Transformer (MLCAFormer)

Paper: Spatio-temporal transformer traffic prediction network based on multi-level causal attention
Publication: PLOS One, 2025
Authors: Hengyuan He, Zhengtao Long, Yingchao Zhang, Xiaofei Jiang
Institution: College of Big Data and Information Engineering, GuiZhou University

This implementation is adapted for LibCity framework.
Original repository: Not available (to be added when/if published)

The model features:
1. Multi-level Causal Attention (MLCA) for capturing long- and short-term dependencies
2. Node-Identity-Aware Spatial Attention for learning spatial correlations
3. Multi-dimensional input with cyclical patterns and spatio-temporal embeddings
"""

from logging import getLogger
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class NodeIdentityEmbedding(nn.Module):
    """
    Node-Identity-Aware embedding for spatial attention
    """
    def __init__(self, num_nodes, embed_dim):
        super(NodeIdentityEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_nodes, embed_dim)

    def forward(self, batch_size, device):
        """
        Generate node identity embeddings for a batch
        Returns:
            embeddings: (batch_size, num_nodes, embed_dim)
        """
        node_ids = torch.arange(self.num_nodes, device=device)
        embeddings = self.embedding(node_ids)
        embeddings = embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        return embeddings


class CausalMultiHeadAttention(nn.Module):
    """
    Multi-Level Causal Attention mechanism
    Ensures temporal causality while capturing dependencies
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(CausalMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: (batch_size, seq_len, d_model)
            mask: causal mask
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)
        seq_len = query.size(1)

        # Linear projections
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply causal mask
        if mask is None:
            mask = self._generate_causal_mask(seq_len, query.device)
        scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.W_o(context)

        return output

    def _generate_causal_mask(self, seq_len, device):
        """Generate causal mask to prevent attending to future positions"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)


class SpatialAttention(nn.Module):
    """
    Node-Identity-Aware Spatial Attention
    """
    def __init__(self, d_model, num_nodes, num_heads, dropout=0.1):
        super(SpatialAttention, self).__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.num_heads = num_heads

        self.node_embedding = NodeIdentityEmbedding(num_nodes, d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, num_nodes, d_model)
        Returns:
            output: (batch_size, seq_len, num_nodes, d_model)
        """
        batch_size, seq_len, num_nodes, d_model = x.size()

        # Get node identity embeddings
        node_embeds = self.node_embedding(batch_size, x.device)

        # Process each time step
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # (batch_size, num_nodes, d_model)

            # Add node identity information
            x_t_enhanced = x_t + node_embeds

            # Apply spatial attention
            attn_out, _ = self.attention(x_t_enhanced, x_t_enhanced, x_t_enhanced)
            attn_out = self.dropout(attn_out)
            out_t = self.layer_norm(x_t + attn_out)

            outputs.append(out_t)

        output = torch.stack(outputs, dim=1)
        return output


class FeedForward(nn.Module):
    """Feed-forward network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.layer_norm(residual + x)


class MLCABlock(nn.Module):
    """
    Multi-Level Causal Attention Block
    Combines temporal causal attention, spatial attention, and feed-forward
    """
    def __init__(self, d_model, num_nodes, num_heads, d_ff, dropout=0.1):
        super(MLCABlock, self).__init__()

        # Temporal causal attention
        self.temporal_attn = CausalMultiHeadAttention(d_model, num_heads, dropout)

        # Spatial attention
        self.spatial_attn = SpatialAttention(d_model, num_nodes, num_heads, dropout)

        # Feed-forward
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, num_nodes, d_model)
        Returns:
            output: (batch_size, seq_len, num_nodes, d_model)
        """
        batch_size, seq_len, num_nodes, d_model = x.size()

        # Temporal attention (process each node separately)
        temp_outputs = []
        for n in range(num_nodes):
            x_n = x[:, :, n, :]  # (batch_size, seq_len, d_model)
            temp_out = self.temporal_attn(x_n, x_n, x_n)
            temp_out = self.dropout(temp_out)
            temp_out = self.layer_norm1(x_n + temp_out)
            temp_outputs.append(temp_out)
        x = torch.stack(temp_outputs, dim=2)  # (batch_size, seq_len, num_nodes, d_model)

        # Spatial attention
        spatial_out = self.spatial_attn(x)
        x = self.layer_norm2(x + spatial_out)

        # Feed-forward
        x = self.ffn(x)

        return x


class MLCAFormer(AbstractTrafficStateModel):
    """
    Multi-Level Causal Attention Transformer for traffic prediction
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        # Get data features
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)

        # Get model configuration
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.device = config.get('device', torch.device('cpu'))

        # Model hyperparameters
        self.d_model = config.get('d_model', 64)
        self.num_layers = config.get('num_layers', 3)
        self.num_heads = config.get('num_heads', 8)
        self.d_ff = config.get('d_ff', 256)
        self.dropout = config.get('dropout', 0.1)

        self._logger = getLogger()
        self._logger.info('MLCAFormer initialized with {} nodes, {} input steps, {} output steps'.format(
            self.num_nodes, self.input_window, self.output_window))

        # Input embedding
        self.input_embedding = nn.Linear(self.feature_dim, self.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=self.input_window)

        # MLCA blocks
        self.mlca_blocks = nn.ModuleList([
            MLCABlock(self.d_model, self.num_nodes, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.output_dim)
        )

        # Temporal decoder for multi-step prediction
        self.temporal_decoder = nn.Linear(self.input_window, self.output_window)

    def forward(self, batch):
        """
        Args:
            batch: dict with key 'X' of shape (batch_size, input_window, num_nodes, feature_dim)
        Returns:
            predictions: (batch_size, output_window, num_nodes, output_dim)
        """
        x = batch['X']  # (batch_size, input_window, num_nodes, feature_dim)

        # Input embedding
        x = self.input_embedding(x)  # (batch_size, input_window, num_nodes, d_model)

        # Add positional encoding (for temporal dimension)
        batch_size, seq_len, num_nodes, d_model = x.size()
        x_reshaped = x.view(batch_size * num_nodes, seq_len, d_model)
        x_reshaped = self.pos_encoder(x_reshaped)
        x = x_reshaped.view(batch_size, seq_len, num_nodes, d_model)

        # Apply MLCA blocks
        for mlca_block in self.mlca_blocks:
            x = mlca_block(x)

        # Temporal projection for multi-step prediction
        # Reshape to (batch_size, num_nodes, d_model, seq_len)
        x = x.permute(0, 2, 3, 1)
        x = self.temporal_decoder(x)
        # Reshape back to (batch_size, output_window, num_nodes, d_model)
        x = x.permute(0, 3, 1, 2)

        # Output projection
        output = self.output_projection(x)  # (batch_size, output_window, num_nodes, output_dim)

        return output

    def predict(self, batch):
        """
        Prediction interface required by LibCity
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate loss for training
        Args:
            batch: dict with 'X' and 'y'
        Returns:
            loss: scalar tensor
        """
        y_true = batch['y']  # (batch_size, output_window, num_nodes, output_dim)
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0.0)
