"""
HSTWAVE: Hierarchical Spatial-Temporal Weaving Attention and Heterogeneous Graph Network

This model is adapted from the original HSTWAVE implementation for the LibCity framework.
Original model uses heterogeneous graph with highway and parallel road node types.
In this adaptation, we simulate heterogeneous graph structure using the adjacency matrix
and treat all nodes uniformly while preserving the core architecture.

Key Components:
- GTU (Gated Temporal Unit): Gated temporal convolution for multi-scale temporal modeling
- MSWT (Multi-Scale Weaving Transformer): Multi-scale temporal feature extraction with transformer
- CHGAN (Coupled Heterogeneous Graph Attention Network): Graph attention with edge type embeddings
- MSDTHGTEncoder: Main encoder combining MSWT and CHGAN
- SequenceAugmentor: Data augmentation for contrastive learning

Original file: /repos/HSTWAVE/model.py
Adapted for LibCity traffic flow prediction task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


class CausalConv2d(nn.Conv2d):
    """
    Causal 2D convolution that applies padding only on the left side for temporal causality.
    This ensures the model only uses past information for predictions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        if enable_padding:
            self.__padding = [int((kernel_size[i] - 1)) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, groups=groups, bias=bias)

    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)
        return result


class Align(nn.Module):
    """
    Alignment module to match channel dimensions between input and output.
    Uses 1x1 convolution for reduction and zero-padding for expansion.
    """
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        # x shape: (B, F, N, T)
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, node_num, timestep = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, node_num, timestep]).to(x)], dim=1)
        return x


class GTU(nn.Module):
    """
    Gated Temporal Unit (GTU) for temporal feature extraction.
    Uses gated convolution with learnable weights for adaptive temporal modeling.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Temporal kernel size (scale factor)
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(GTU, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.conv = CausalConv2d(
            in_channels=in_channels,
            out_channels=2 * out_channels,
            kernel_size=(1, kernel_size),
            enable_padding=True
        )

        self.w_conv = nn.Conv2d(in_channels, 1, kernel_size=(1, 1))
        self.align = Align(in_channels, out_channels)

    def forward(self, x):
        # Input: (B, N, T, F) -> (B, F, N, T)
        x = x.permute(0, 3, 1, 2)

        x_conv = self.conv(x)  # (B, 2*out_channels, N, T)
        x_p = x_conv[:, :self.out_channels, :, :]
        x_q = x_conv[:, -self.out_channels:, :, :]

        x_gtu = torch.mul(self.tanh(x_p), self.sigmoid(x_q))
        w = torch.sigmoid(self.w_conv(x)).mean(-1).unsqueeze(-1)  # (B, 1, N, 1)
        return w * x_gtu + (1 - w) * x_gtu


class MSWT(nn.Module):
    """
    Multi-Scale Weaving Transformer (MSWT) for multi-scale temporal feature extraction.
    Combines multiple GTUs with different scales and a transformer encoder for
    cross-scale attention.

    Args:
        in_channels: Number of input channels
        d: Hidden dimension
        num_scales: Number of temporal scales (default: 3)
    """
    def __init__(self, in_channels, d, num_scales=3):
        super(MSWT, self).__init__()
        self.d = d
        self.num_scales = num_scales

        # Multi-Scale GTU modules
        self.gtu_scales = nn.ModuleList([GTU(in_channels, d, scale) for scale in range(1, num_scales + 1)])

        # Transformer Encoder layers for each scale
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d, nhead=1, dim_feedforward=4 * d, batch_first=False)
            for _ in range(num_scales)
        ])
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d, num_heads=1, batch_first=False)

    def forward(self, node_features):
        """
        Args:
            node_features: (B, N, T, in_channels)
        Returns:
            output: (B, N, T, d)
        """
        B, N, T, in_channels = node_features.shape

        outputs = []
        for gtu in self.gtu_scales:
            scale_output = gtu(node_features)  # (B, d, N, T)
            outputs.append(scale_output)

        transformer_outputs = []
        for i, output in enumerate(outputs):
            # (B, d, N, T) -> (T, B*N, d)
            reshaped_output = output.permute(3, 0, 2, 1).reshape(T, B*N, self.d)
            transformed = self.transformer_layers[i](reshaped_output)
            # (T, B*N, d) -> (B, N, T, d)
            transformed = transformed.reshape(T, B, N, self.d).permute(1, 2, 0, 3)
            transformer_outputs.append(transformed)

        # Stack and apply cross-scale attention
        stacked_output = torch.cat(transformer_outputs, dim=1)  # (B, N*num_scales, T, d)
        stacked_output = stacked_output.permute(2, 1, 0, 3)  # (T, N*num_scales, B, d)
        stacked_output = stacked_output.reshape(T, -1, self.d)

        attn_output, _ = self.multihead_attn(stacked_output, stacked_output, stacked_output)
        attn_output = attn_output.reshape(T, self.num_scales, N, B, self.d)
        attn_output = attn_output.permute(3, 2, 1, 0, 4)  # (B, N, num_scales, T, d)

        # Mean across scales
        final_output = attn_output.mean(dim=2)  # (B, N, T, d)

        return final_output


class CHGANSimplified(nn.Module):
    """
    Simplified Coupled Heterogeneous Graph Attention Network (CHGAN).

    This is a simplified version that works with LibCity's standard adjacency matrix
    instead of heterogeneous graph structure. It uses multi-head attention with
    learnable node type embeddings and edge bias.

    Args:
        num_nodes: Number of nodes in the graph
        d: Hidden dimension
        num_heads: Number of attention heads
        lambda_decay: Decay factor for distance-based attention
        dropout: Dropout rate
    """
    def __init__(self, num_nodes, d, num_heads=4, lambda_decay=0.5, dropout=0.1):
        super(CHGANSimplified, self).__init__()

        self.num_nodes = num_nodes
        self.d = d
        self.num_heads = num_heads
        self.lambda_decay = lambda_decay
        self.d_head = d // num_heads

        assert d % num_heads == 0, "d must be divisible by num_heads"

        # Node type embeddings (simulate 2 types: highway and parallel roads)
        self.node_type_embed = nn.Parameter(torch.randn(2, d))

        # Multi-head Q, K, V projections
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)

        # Edge bias for graph structure
        self.edge_bias = nn.Parameter(torch.zeros(num_nodes, num_nodes))

        # Output projection
        self.out_proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, node_features, adj_mx=None):
        """
        Args:
            node_features: (B, N, T, d) or (N, T, d) for unbatched
            adj_mx: Optional adjacency matrix (N, N)
        Returns:
            output: Same shape as input
        """
        if node_features.dim() == 3:
            # Unbatched: (N, T, d)
            return self._forward_unbatched(node_features, adj_mx)
        else:
            # Batched: (B, N, T, d)
            B, N, T, d = node_features.shape
            # Process each sample in the batch
            outputs = []
            for b in range(B):
                out = self._forward_unbatched(node_features[b], adj_mx)
                outputs.append(out)
            return torch.stack(outputs, dim=0)

    def _forward_unbatched(self, node_features, adj_mx=None):
        """
        Process unbatched node features.
        Args:
            node_features: (N, T, d)
            adj_mx: Optional adjacency matrix (N, N)
        Returns:
            output: (N, T, d)
        """
        N, T, d = node_features.shape
        device = node_features.device

        # Add node type information (alternate between types)
        node_types = torch.zeros(N, dtype=torch.long, device=device)
        node_types[::2] = 1  # Simulate alternating node types
        type_embeds = self.node_type_embed[node_types]  # (N, d)

        # Add type embedding to features
        enhanced_features = node_features + type_embeds.unsqueeze(1)  # (N, T, d)

        # Compute Q, K, V
        Q = self.q_proj(enhanced_features)  # (N, T, d)
        K = self.k_proj(enhanced_features)
        V = self.v_proj(enhanced_features)

        # Reshape for multi-head attention
        Q = Q.view(N, T, self.num_heads, self.d_head).permute(2, 1, 0, 3)  # (num_heads, T, N, d_head)
        K = K.view(N, T, self.num_heads, self.d_head).permute(2, 1, 0, 3)
        V = V.view(N, T, self.num_heads, self.d_head).permute(2, 1, 0, 3)

        # Attention scores
        attn_scores = torch.einsum('htnd,htmd->htnm', Q, K) / (self.d_head ** 0.5)  # (num_heads, T, N, N)

        # Add edge bias from graph structure
        edge_bias = self.edge_bias[:N, :N]
        attn_scores = attn_scores + edge_bias.unsqueeze(0).unsqueeze(0)

        # Apply adjacency mask if provided
        if adj_mx is not None:
            adj_mask = (adj_mx[:N, :N] == 0)
            # Ensure self-loops are not masked (prevent all-masked rows -> NaN in softmax)
            self_loop_mask = torch.eye(N, dtype=torch.bool, device=device)
            adj_mask = adj_mask & ~self_loop_mask
            attn_scores = attn_scores.masked_fill(adj_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax and apply to values
        attn_weights = F.softmax(attn_scores, dim=-1)
        # Replace any NaN from softmax (all -inf rows) with zeros
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.einsum('htnm,htmd->htnd', attn_weights, V)  # (num_heads, T, N, d_head)
        output = output.permute(2, 1, 0, 3).contiguous().view(N, T, d)  # (N, T, d)

        # Output projection and residual connection
        output = self.out_proj(output)
        output = self.layer_norm(output + node_features)

        return output


class MSDTHGTEncoderSimplified(nn.Module):
    """
    Simplified Multi-Scale Diffusion Temporal Heterogeneous Graph Transformer Encoder.

    Combines MSWT and CHGAN for joint spatial-temporal feature learning.
    Adapted to work with LibCity's standard data format.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        seq_len: Input sequence length
        num_nodes: Number of nodes
    """
    def __init__(self, in_channels, out_channels, seq_len, num_nodes):
        super(MSDTHGTEncoderSimplified, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.num_nodes = num_nodes

        self.mswt1 = MSWT(in_channels=in_channels, d=out_channels)
        self.chgan = CHGANSimplified(num_nodes=num_nodes, d=out_channels, num_heads=4)
        self.mswt2 = MSWT(in_channels=out_channels, d=out_channels)

        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm([num_nodes, out_channels])
        self.dropout = nn.Dropout(p=0.3)

        # Alignment layer for residual connection
        self.align = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, adj_mx=None):
        """
        Args:
            x: (B, N, T, F) node features
            adj_mx: Optional adjacency matrix (N, N)
        Returns:
            output: (B, N, T, out_channels)
        """
        B, N, T, F = x.shape

        # Store initial features for residual
        init_x = self.align(x)  # (B, N, T, out_channels)

        # First MSWT
        x = self.mswt1(x)  # (B, N, T, out_channels)

        # CHGAN for spatial modeling
        x = self.chgan(x, adj_mx)  # (B, N, T, out_channels)

        # Second MSWT
        x = self.mswt2(x)  # (B, N, T, out_channels)

        # Residual connection
        x = x + init_x

        # Layer norm and dropout
        # (B, N, T, F) -> (B, T, N, F) -> apply LN -> (B, N, T, F)
        x = x.permute(0, 2, 1, 3)  # (B, T, N, F)
        x = self.ln(x)
        x = x.permute(0, 2, 1, 3)  # (B, N, T, F)
        x = self.dropout(x)

        return x


class SequenceAugmentor:
    """
    Data augmentation module for contrastive learning.
    Provides various augmentation operations including flip, mask, shift, and noise addition.
    """
    def __init__(self, noise_std=0.01):
        self.noise_std = noise_std

    def flip(self, sequence, m, max_value):
        """Flip operation, randomly invert the values of m positions"""
        seq_length = sequence.size()[0]
        indices = torch.randperm(seq_length)[:m].to(sequence.device)
        sequence[indices] = max_value - sequence[indices]
        return sequence

    def mask(self, sequence, m):
        """Mask operation, randomly set m positions' values to 0"""
        seq_length = sequence.size()[0]
        indices = torch.randperm(seq_length)[:m].to(sequence.device)
        sequence[indices] = 0
        return sequence

    def replace_with_noise(self, sequence, m):
        """Replace operation, randomly replace m positions with Gaussian noise"""
        seq_length = sequence.size()[0]
        indices = torch.randperm(seq_length)[:m].to(sequence.device)
        noise = torch.normal(0.0, self.noise_std, size=(m,)).to(sequence.device)
        sequence[indices] = noise
        return sequence

    def shift(self, sequence, m):
        """Shift operation: move the last m positions to the front"""
        seq_length = sequence.size()[0]
        shifted_sequence = torch.zeros_like(sequence)
        shifted_sequence[:seq_length - m] = sequence[m:]
        return shifted_sequence

    def add_noise(self, sequence, strength=1.0):
        """Add noise operation, add Gaussian noise to the entire sequence"""
        noise = torch.normal(0.0, strength * self.noise_std, size=(sequence.size()[0],)).to(sequence.device)
        return sequence + noise

    def augment_sequence(self, sequence, max_value):
        """Augment a single sequence: randomly choose two operations"""
        seq_length = sequence.size()[0]
        m = int(0.25 * seq_length)
        ops = [
            lambda seq: self.flip(seq, m, max_value),
            lambda seq: self.mask(seq, m),
            lambda seq: self.shift(seq, m),
            lambda seq: self.add_noise(seq)
        ]
        chosen_ops = torch.randperm(len(ops))[:2]
        for idx in chosen_ops:
            sequence = ops[idx.item()](sequence)
        return sequence

    def augment(self, sequences):
        """Augment a batch of sequences"""
        augmented_sequences = []
        max_value = torch.max(sequences)
        for sequence in sequences:
            augmented_sequences.append(self.augment_sequence(sequence, max_value))
        return torch.stack(augmented_sequences)


class HSTWAVE(AbstractTrafficStateModel):
    """
    HSTWAVE: Hierarchical Spatial-Temporal Weaving Attention and Heterogeneous Graph Network

    Adapted for LibCity framework from the original implementation.

    The model uses:
    - Multi-Scale Weaving Transformer (MSWT) for temporal feature extraction
    - Coupled Heterogeneous Graph Attention Network (CHGAN) for spatial modeling
    - Contrastive learning with SimCLR loss for better representations

    Args:
        config: Configuration dictionary
        data_feature: Data feature dictionary containing num_nodes, feature_dim, etc.
    """

    def __init__(self, config, data_feature):
        # Extract data features
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)

        super().__init__(config, data_feature)

        # Model configuration
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.device = config.get('device', torch.device('cpu'))

        # HSTWAVE specific parameters
        self.hidden_dim = config.get('hidden_dim', 64)
        self.num_layers = config.get('num_layers', 3)
        self.num_scales = config.get('num_scales', 3)
        self.num_heads = config.get('num_heads', 4)
        self.dropout = config.get('dropout', 0.3)
        self.use_contrastive = config.get('use_contrastive', True)
        self.contrastive_weight = config.get('contrastive_weight', 0.2)
        self.contrastive_temp = config.get('contrastive_temp', 500)
        self.noise_std = config.get('noise_std', 0.05)

        # Build hidden dimensions list
        hidden_dims = [self.hidden_dim] * self.num_layers

        # Learnable temporal embeddings
        self.series_tensor = nn.Parameter(
            torch.nn.init.kaiming_normal_(
                torch.empty(1, self.num_nodes, self.input_window, self.feature_dim)
            )
        )

        # Encoder blocks
        modules = []
        in_dim = self.feature_dim
        for i, out_dim in enumerate(hidden_dims):
            # Use doubled sequence length to account for temporal embedding concatenation
            modules.append(
                MSDTHGTEncoderSimplified(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    seq_len=self.input_window * 2,
                    num_nodes=self.num_nodes
                )
            )
            in_dim = out_dim
        self.stg_blocks = nn.ModuleList(modules)

        # Output layers
        # Concatenate outputs from all encoder layers along feature dimension
        total_out_dim = self.hidden_dim * self.num_layers
        self.out_linear = nn.Conv2d(
            in_channels=total_out_dim,
            out_channels=128,
            kernel_size=(1, self.input_window)  # Use input_window for temporal kernel
        )
        self.ln = nn.Sequential(nn.ReLU(), nn.LayerNorm(128), nn.Dropout(self.dropout))
        self.final_fc = nn.Linear(128, self.output_window * self.output_dim)

        # Logger and scaler
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        # Get adjacency matrix if available
        self.adj_mx = data_feature.get('adj_mx', None)
        if self.adj_mx is not None:
            self.register_buffer('_adj_mx', torch.tensor(self.adj_mx, dtype=torch.float32))
        else:
            self._adj_mx = None

        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters (only linear/conv weights, skip embeddings and norm layers)"""
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'layer_norm' not in name and 'ln' not in name \
                    and 'series_tensor' not in name and 'node_type_embed' not in name \
                    and 'edge_bias' not in name:
                nn.init.xavier_uniform_(p)

    def init_weight(self, size: Tuple):
        """Initialize learnable weight tensor"""
        weight = torch.empty(size, requires_grad=True)
        weight = torch.nn.init.kaiming_normal_(weight)
        return torch.nn.Parameter(weight, requires_grad=True)

    def forward(self, batch):
        """
        Forward pass of HSTWAVE model.

        Args:
            batch: Dictionary containing 'X' with shape (B, T, N, F)

        Returns:
            If training with contrastive learning: (predictions, aug_out1, aug_out2)
            Otherwise: predictions with shape (B, T_out, N, output_dim)
        """
        # Get input: (B, T, N, F)
        x = batch['X']
        B, T, N, F = x.shape

        # Transpose to (B, N, T, F) for processing
        x = x.permute(0, 2, 1, 3)  # (B, N, T, F)

        # Expand temporal embedding
        series_tensor = self.series_tensor.expand(B, -1, -1, -1)  # (B, N, T, F)

        # Create augmented views for contrastive learning during training
        if self.training and self.use_contrastive:
            # View 1: Original with temporal embedding
            x1 = torch.cat([x, series_tensor], dim=2)  # (B, N, 2T, F)

            # View 2: Augmented with light noise
            augmentor = SequenceAugmentor(noise_std=self.noise_std)
            x2 = x.clone()
            # Augment the first feature channel (typically flow)
            x2_aug = x2[:, :, :, 0].reshape(-1, T)  # (B*N, T)
            x2_aug = augmentor.augment(x2_aug)
            x2[:, :, :, 0] = x2_aug.reshape(B, N, T)
            x2 = torch.cat([x2, series_tensor], dim=2)  # (B, N, 2T, F)

            # Process through encoders
            need_concat_1 = []
            need_concat_2 = []

            adj_mx = self._adj_mx if self._adj_mx is not None else None

            for encoder in self.stg_blocks:
                x1 = encoder(x1, adj_mx)
                x2 = encoder(x2, adj_mx)
                # Take first T timesteps after encoding
                need_concat_1.append(x1[:, :, :self.input_window, :])  # (B, N, T, hidden)
                need_concat_2.append(x2[:, :, :self.input_window, :])

            # Concatenate encoder outputs along feature dimension
            final_x1 = torch.cat(need_concat_1, dim=3)  # (B, N, T, hidden*num_layers)
            final_x2 = torch.cat(need_concat_2, dim=3)

            # (B, N, T, hidden*num_layers) -> (B, hidden*num_layers, N, T)
            final_x1 = final_x1.permute(0, 3, 1, 2)
            final_x2 = final_x2.permute(0, 3, 1, 2)

            # Output projection
            out1 = self.out_linear(final_x1)[:, :, :, -1]  # (B, 128, N)
            out2 = self.out_linear(final_x2)[:, :, :, -1]

            out1 = out1.permute(0, 2, 1)  # (B, N, 128)
            out2 = out2.permute(0, 2, 1)

            out1 = self.ln(out1)
            out2 = self.ln(out2)

            # Final prediction from view 1
            pre = self.final_fc(out1)  # (B, N, T_out * output_dim)
            pre = pre.reshape(B, N, self.output_window, self.output_dim)
            pre = pre.permute(0, 2, 1, 3)  # (B, T_out, N, output_dim)

            # Flatten for contrastive loss
            out1_flat = out1.reshape(B, -1)  # (B, N*128)
            out2_flat = out2.reshape(B, -1)

            return pre, out1_flat, out2_flat
        else:
            # Inference mode: single view
            x = torch.cat([x, series_tensor], dim=2)  # (B, N, 2T, F)

            need_concat = []
            adj_mx = self._adj_mx if self._adj_mx is not None else None

            for encoder in self.stg_blocks:
                x = encoder(x, adj_mx)
                need_concat.append(x[:, :, :self.input_window, :])

            final_x = torch.cat(need_concat, dim=3)  # (B, N, T, hidden*num_layers)
            final_x = final_x.permute(0, 3, 1, 2)  # (B, hidden*num_layers, N, T)

            out = self.out_linear(final_x)[:, :, :, -1]  # (B, 128, N)
            out = out.permute(0, 2, 1)  # (B, N, 128)
            out = self.ln(out)

            pre = self.final_fc(out)  # (B, N, T_out * output_dim)
            pre = pre.reshape(B, N, self.output_window, self.output_dim)
            pre = pre.permute(0, 2, 1, 3)  # (B, T_out, N, output_dim)

            return pre

    def predict(self, batch):
        """
        Prediction method for inference.

        Args:
            batch: Dictionary containing 'X'

        Returns:
            Predictions with shape (B, T_out, N, output_dim)
        """
        # Set to eval mode temporarily if needed
        result = self.forward(batch)
        if isinstance(result, tuple):
            return result[0]
        return result

    def calculate_loss(self, batch):
        """
        Calculate training loss combining MAE and contrastive loss.

        Args:
            batch: Dictionary containing 'X' and 'y'

        Returns:
            Total loss tensor
        """
        y_true = batch['y']

        if self.training and self.use_contrastive:
            y_predicted, out1, out2 = self.forward(batch)

            # Inverse transform for loss calculation
            y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
            y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

            # MAE loss
            mae_loss = loss.masked_mae_torch(y_predicted, y_true, 0)

            # Contrastive loss (SimCLR)
            contrastive_loss = self._simclr_loss(out1, out2, batch['X'].shape[0])

            # Combined loss
            total_loss = (1 - self.contrastive_weight) * mae_loss + self.contrastive_weight * contrastive_loss

            return total_loss
        else:
            y_predicted = self.predict(batch)

            y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
            y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

            return loss.masked_mae_torch(y_predicted, y_true, 0)

    def _simclr_loss(self, out_1, out_2, batch_size, temperature=None):
        """
        SimCLR contrastive loss for learning better representations.

        Args:
            out_1: First view embeddings (B, D)
            out_2: Second view embeddings (B, D)
            batch_size: Batch size
            temperature: Temperature for softmax (default uses self.contrastive_temp)

        Returns:
            Contrastive loss
        """
        if temperature is None:
            temperature = self.contrastive_temp

        # L2 normalize embeddings to prevent overflow in exp(dot_product)
        out_1 = F.normalize(out_1, dim=-1)
        out_2 = F.normalize(out_2, dim=-1)

        out = torch.cat([out_1, out_2], dim=0)  # (2*B, D)
        # After normalization, dot products are in [-1, 1], so exp values are bounded
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)  # (2*B, 2*B)

        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)  # (2*B, 2*B-1)

        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)  # (2*B)

        contrastive_loss = (-torch.log(pos_sim / (sim_matrix.sum(dim=-1) + 1e-8))).mean()

        return contrastive_loss
