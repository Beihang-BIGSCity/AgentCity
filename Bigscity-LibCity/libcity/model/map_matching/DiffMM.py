"""
DiffMM: Diffusion-based Map Matching Model

This module adapts the DiffMM model from the original repository to the LibCity
framework for trajectory map matching tasks.

Original paper: "DiffMM: A Diffusion Model for Map Matching"

Key Components:
1. TrajEncoder - Encodes trajectory points with road segment embeddings
   - PointEncoder: Transformer-based encoder for GPS point sequences
   - Attention: Computes attention over candidate road segments per GPS point
   - 9D road segment features + road ID embeddings

2. ShortCut - One-step diffusion transformer for road segment prediction
   - DiT (Diffusion Transformer): Main denoising network with AdaLN modulation
   - Uses flow matching with bootstrapping for efficient training
   - Sinusoidal time embeddings for diffusion timestep conditioning

3. ModelAllShortCut - Combines TrajEncoder and ShortCut for end-to-end training

Architecture Overview:
- Input: GPS trajectory with candidate road segments per point
- TrajEncoder produces condition embeddings from trajectory and road features
- ShortCut diffusion model predicts road segment probability distributions
- One-step or multi-step diffusion inference for final predictions

Adaptations for LibCity:
- Inherits from AbstractModel following LibCity conventions
- Implements predict() and calculate_loss() methods
- Adapts batch input format to LibCity's Batch dictionary format
- Configuration parameters accessible via config dictionary
- Device handling through LibCity's config system

Required data_feature:
- id_size: Number of unique road segment IDs
- knn: Maximum number of candidate segments per GPS point

Required config parameters:
- hid_dim: Hidden dimension (default: 256)
- num_units: Number of units for denoiser (default: 512)
- transformer_layers: Number of transformer layers in PointEncoder (default: 2)
- depth: Number of DiT blocks (default: 2)
- timesteps: Diffusion timesteps for training (default: 2)
- samplingsteps: Inference steps for diffusion (default: 1)
- dropout: Dropout probability (default: 0.1)
- bootstrap_every: Bootstrap frequency for training (default: 8)
- num_heads: Number of attention heads (default: 4)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from logging import getLogger

from libcity.model.abstract_model import AbstractModel


# ============================================================================
# Helper Functions
# ============================================================================

def sequence_mask(X, valid_len, value=0.):
    """Mask irrelevant entries in sequences.

    Args:
        X: Input tensor (batch, seq_len, ...)
        valid_len: Valid lengths for each sequence (batch,)
        value: Value to fill masked positions

    Returns:
        Masked tensor
    """
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def sequence_mask3d(X, valid_len, valid_len2, value=0.):
    """Mask irrelevant entries in 3D sequences.

    Args:
        X: Input tensor (batch, seq_len, seq_len2)
        valid_len: Valid lengths for dimension 1 (batch,)
        valid_len2: Valid lengths for dimension 2 (batch,)
        value: Value to fill masked positions

    Returns:
        Masked tensor
    """
    maxlen = X.size(1)
    maxlen2 = X.size(2)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    mask2 = torch.arange((maxlen2), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len2[:, None]
    mask_fin = torch.bmm(mask.float().unsqueeze(-1), mask2.float().unsqueeze(-2)).bool()
    X[~mask_fin] = value
    return X


def modulate(x, shift, scale):
    """Apply adaptive layer normalization modulation.

    Args:
        x: Input tensor
        shift: Shift parameter
        scale: Scale parameter

    Returns:
        Modulated tensor: x * (1 + scale) + shift
    """
    return x * (1 + scale) + shift


# ============================================================================
# Layer Components
# ============================================================================

class Norm(nn.Module):
    """Layer normalization with learnable parameters.

    Args:
        d_model: Model dimension
        eps: Small constant for numerical stability
    """

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding.

    Args:
        d_model: Model dimension
        max_seq_len: Maximum sequence length (default: 500)
    """

    def __init__(self, d_model, max_seq_len=500):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).to(x.device)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.

    Args:
        heads: Number of attention heads
        d_model: Model dimension
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = self.dropout(scores)

        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network with residual connection.

    Args:
        d_model: Model dimension
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.norm = Norm(d_model)

    def forward(self, x):
        residual = x
        x = self.linear_2(F.relu(self.linear_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.norm(x)
        return x


class EncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention and feed-forward.

    Args:
        d_model: Model dimension
        heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff=d_model * 2)
        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x, mask):
        residual = x
        x = self.dropout_1(self.attn(x, x, x, mask))
        x2 = self.norm_1(residual + x)
        x = self.ff(x2)
        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers.

    Args:
        d_model: Model dimension
        N: Number of encoder layers
        heads: Number of attention heads
    """

    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, heads) for _ in range(N)
        ])
        self.norm = Norm(d_model)

    def forward(self, src, mask3d=None):
        x = self.pe(src)
        for i in range(self.N):
            x = self.layers[i](x, mask3d)
        return self.norm(x)


class Attention(nn.Module):
    """Additive attention for candidate road segment selection.

    Computes attention weights over candidate road segments for each GPS point.

    Args:
        hid_dim: Hidden dimension
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.attn = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.v = nn.Linear(self.hid_dim, 1, bias=False)

    def forward(self, query, key, value, attn_mask):
        """Compute attention over candidate segments.

        Args:
            query: Point embeddings (batch, src_len, hid_dim)
            key: Candidate segment embeddings (batch, src_len, num_cands, hid_dim)
            value: Same as key
            attn_mask: Mask for valid candidates (batch, src_len, num_cands)

        Returns:
            scores: Attention weights (batch, src_len, num_cands)
            weighted: Weighted segment embeddings (batch, src_len, hid_dim)
        """
        batch_size, src_len = query.shape[0], query.shape[1]
        seg_num = key.shape[-2]

        # Expand query to match key dimensions
        query = query.unsqueeze(-2).repeat(1, 1, seg_num, 1)

        energy = torch.tanh(self.attn(torch.cat((query, key), dim=-1)))
        attention = self.v(energy).squeeze(-1)
        attention = attention.masked_fill(attn_mask == 0, -1e10)

        scores = F.softmax(attention, dim=-1)
        weighted = torch.bmm(
            scores.reshape(batch_size * src_len, seg_num).unsqueeze(-2),
            value.reshape(batch_size * src_len, seg_num, -1)
        ).squeeze(-2)
        weighted = weighted.reshape(batch_size, src_len, -1)

        return scores, weighted


class PointEncoder(nn.Module):
    """Transformer encoder for GPS point sequences.

    Args:
        hid_dim: Hidden dimension
        transformer_layers: Number of transformer layers
        num_heads: Number of attention heads (default: 4)
    """

    def __init__(self, hid_dim, transformer_layers, num_heads=4):
        super().__init__()
        self.hid_dim = hid_dim

        # GPS points have 3 features: lat, lon, and possibly time
        input_dim = 3
        self.fc_point = nn.Linear(input_dim, hid_dim)
        self.transformer = TransformerEncoder(hid_dim, transformer_layers, heads=num_heads)

    def forward(self, src, src_len):
        """Encode GPS point sequence.

        Args:
            src: GPS points (batch, seq_len, 3)
            src_len: Sequence lengths (list or tensor)

        Returns:
            Point embeddings (batch, seq_len, hid_dim)
        """
        max_src_len = src.size(1)
        batch_size = src.size(0)

        src_len = torch.tensor(src_len, device=src.device)

        mask3d = torch.ones(batch_size, max_src_len, max_src_len, device=src.device)
        mask2d = torch.ones(batch_size, max_src_len, device=src.device)

        mask3d = sequence_mask3d(mask3d, src_len, src_len)
        mask2d = sequence_mask(mask2d, src_len).unsqueeze(-1).repeat(1, 1, self.hid_dim)

        src = self.fc_point(src)
        outputs = self.transformer(src, mask3d)

        assert outputs.size(1) == max_src_len
        outputs = outputs * mask2d

        return outputs


# ============================================================================
# TrajEncoder: Trajectory Encoder with Road Segment Attention
# ============================================================================

class TrajEncoder(nn.Module):
    """Trajectory encoder combining GPS points with road segment embeddings.

    Encodes GPS trajectory points and computes attention over candidate road
    segments for each point to produce context-aware trajectory embeddings.

    Args:
        id_size: Number of unique road segment IDs
        hid_dim: Hidden dimension
        transformer_layers: Number of transformer layers
        num_heads: Number of attention heads
        device: Computation device
    """

    def __init__(self, id_size, hid_dim, transformer_layers, num_heads, device):
        super().__init__()
        self.id_size = id_size
        self.hid_dim = hid_dim
        self.id_emb_dim = hid_dim

        # Learnable road segment ID embeddings
        self.emb_id = nn.Parameter(torch.rand(self.id_size, self.id_emb_dim))

        self.device = device

        # Road embedding: ID embedding + 9D features
        road_emb_input_dim = self.id_emb_dim + 9
        self.road_emb = nn.Sequential(
            nn.Linear(road_emb_input_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            Norm(self.hid_dim)
        )

        self.point_encoder = PointEncoder(hid_dim, transformer_layers, num_heads)
        self.attn = Attention(self.hid_dim)

        self.output = nn.Linear(2 * hid_dim, hid_dim)

    def forward(self, src, src_len, src_segs, segs_feat, segs_mask):
        """Encode trajectory with road segment attention.

        Args:
            src: GPS points (batch, seq_len, 3)
            src_len: Sequence lengths (list)
            src_segs: Candidate segment IDs (batch, seq_len, num_cands)
            segs_feat: Segment features (batch, seq_len, num_cands, 9)
            segs_mask: Valid segment mask (batch, seq_len, num_cands)

        Returns:
            Trajectory embeddings (batch, seq_len, 2*hid_dim)
        """
        # Get road segment ID embeddings
        src_id_emb = self.emb_id[src_segs]
        src_road_emb = torch.cat((src_id_emb, segs_feat), dim=-1)
        road_emb = self.road_emb(src_road_emb)

        # Encode GPS points
        point_encoder_output = self.point_encoder(src, src_len)

        # Compute attention over candidate segments
        _, attention = self.attn(point_encoder_output, road_emb, road_emb, segs_mask)

        # Concatenate point embedding with attended road embedding
        outputs = torch.cat((point_encoder_output, attention), dim=-1)

        return outputs


# ============================================================================
# Diffusion Components: DiT and ShortCut
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps.

    Args:
        dim: Embedding dimension
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiTBlock(nn.Module):
    """Diffusion Transformer block with AdaLN modulation.

    Implements a transformer block with adaptive layer normalization (AdaLN)
    conditioning for diffusion models.

    Args:
        hid_dim: Hidden dimension
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, hid_dim, num_heads=4, dropout=0.1):
        super(DiTBlock, self).__init__()
        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # AdaLN modulation: generates shift, scale, gate for both attention and MLP
        self.cond_linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hid_dim, 6 * hid_dim)
        )

        self.norm1 = Norm(hid_dim)
        self.norm2 = Norm(hid_dim)

        self.attn = MultiHeadAttention(num_heads, hid_dim, dropout)
        self.ff = FeedForward(hid_dim, d_ff=hid_dim * 2)

    def forward(self, x, c):
        """Forward pass with conditioning.

        Args:
            x: Input tensor (batch, seq_len, hid_dim)
            c: Conditioning tensor (batch, seq_len, hid_dim)

        Returns:
            Output tensor (batch, seq_len, hid_dim)
        """
        cond = self.cond_linear(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(cond, 6, dim=-1)

        # Attention with AdaLN
        x_norm1 = self.norm1(x)
        x_modulated = modulate(x_norm1, shift_msa, scale_msa)
        attn_x = self.attn(x_modulated, x_modulated, x_modulated)
        x = x + (gate_msa * attn_x)

        # MLP with AdaLN
        x_norm2 = self.norm2(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_x = self.ff(x_modulated2)
        x = x + (gate_mlp * mlp_x)

        return x


class OutputLayer(nn.Module):
    """Output layer with AdaLN modulation for DiT.

    Args:
        hid_dim: Hidden dimension
        out_dim: Output dimension
    """

    def __init__(self, hid_dim, out_dim):
        super(OutputLayer, self).__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.cond_linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hid_dim, 2 * hid_dim)
        )

        self.norm = Norm(hid_dim)
        self.output_linear = nn.Linear(hid_dim, out_dim)

    def forward(self, x, c):
        cond = self.cond_linear(c)
        shift, scale = torch.chunk(cond, 2, dim=-1)

        x_norm = self.norm(x)
        x_modulated = modulate(x_norm, shift, scale)
        x = self.output_linear(x_modulated)

        return x


class DiT(nn.Module):
    """Diffusion Transformer for denoising.

    A transformer-based denoising network that conditions on trajectory
    embeddings and diffusion timesteps.

    Args:
        out_dim: Output dimension (number of candidate segments)
        hid_dim: Hidden dimension
        depth: Number of DiT blocks
        cond_dim: Conditioning dimension (from trajectory encoder)
    """

    def __init__(self, out_dim, hid_dim, depth, cond_dim):
        super(DiT, self).__init__()
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.depth = depth

        sinu_pos_emb = SinusoidalPosEmb(hid_dim)
        fourier_dim = hid_dim
        time_dim = hid_dim

        self.pe = PositionalEncoder(hid_dim, max_seq_len=2000)

        # Time embedding for diffusion timestep t
        self.time_embedder = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Timestep embedding for step size dt
        self.timestep_embedder = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.cond_linear = nn.Linear(cond_dim, hid_dim)

        self.DiTBlocks = nn.ModuleList([
            DiTBlock(hid_dim) for _ in range(depth)
        ])

        self.noise_linear = nn.Sequential(
            nn.Linear(out_dim, hid_dim),
            nn.ReLU()
        )

        self.output = OutputLayer(hid_dim, out_dim)

    def forward(self, x, t, dt, cond, segs_mask):
        """Forward pass through DiT.

        Args:
            x: Noisy input (batch, seq_len, out_dim)
            t: Diffusion timestep (batch,)
            dt: Step size indicator (batch,)
            cond: Conditioning from trajectory encoder (batch, seq_len, cond_dim)
            segs_mask: Valid segment mask (batch, seq_len, out_dim)

        Returns:
            Velocity prediction (batch, seq_len, out_dim)
        """
        x = self.noise_linear(x)
        x = self.pe(x)

        c = self.cond_linear(cond)
        te = self.time_embedder(t)
        dte = self.timestep_embedder(dt)

        # Add time embeddings to conditioning
        c = c + te[:, None] + dte[:, None]

        for i in range(self.depth):
            x = self.DiTBlocks[i](x, c)

        x = self.output(x, cond)

        # Mask invalid positions
        x = x.masked_fill(segs_mask == 0, 0)
        return x


class ShortCut(nn.Module):
    """One-step diffusion shortcut model for efficient inference.

    Implements a flow matching approach with optional bootstrapping for
    distillation during training.

    Args:
        model: DiT denoising network
        infer_steps: Number of inference steps
        seq_length: Output sequence length (number of candidates)
        bootstrap_every: Bootstrap frequency (default: 8)
    """

    def __init__(self, model, infer_steps, seq_length, bootstrap_every=8):
        super().__init__()
        self.model = model
        self.infer_steps = infer_steps
        self.seq_length = seq_length
        self.bootstrap_every = bootstrap_every

    def forward(self, x_t, v_t, t, dt_base, cond, x_1, segs_mask):
        """Forward pass for training.

        Args:
            x_t: Noisy sample at timestep t
            v_t: Target velocity
            t: Timestep
            dt_base: Step size base
            cond: Conditioning
            x_1: Clean target
            segs_mask: Valid segment mask

        Returns:
            Combined loss (MSE + BCE)
        """
        v_pred = self.model(x_t, t, dt_base, cond, segs_mask)

        # Predicted clean sample
        x_pred = x_t + v_pred

        # MSE loss on velocity
        mse_loss = F.mse_loss(v_pred, v_t)

        # BCE loss on predicted probability distribution
        bce_loss = F.binary_cross_entropy(
            F.softmax(x_pred.masked_fill(segs_mask == 0, -1e9), dim=-1),
            x_1,
            reduction='mean'
        )

        loss = mse_loss + bce_loss
        return loss

    @torch.no_grad()
    def inference(self, batch_size, cond, segs_mask):
        """Generate predictions via diffusion sampling.

        Args:
            batch_size: Batch size
            cond: Conditioning tensor (batch, seq_len, cond_dim)
            segs_mask: Valid segment mask (batch, seq_len, out_dim)

        Returns:
            Probability predictions (batch, seq_len, out_dim)
        """
        device = cond.device
        seq_len = cond.size(1)

        # Start from noise
        eps = torch.randn((batch_size, seq_len, self.seq_length), device=device)

        delta_t = 1.0 / self.infer_steps
        x = eps.masked_fill(segs_mask == 0, 0)

        for ti in range(self.infer_steps):
            t = ti / self.infer_steps

            t_vector = torch.full((batch_size,), t, device=device)
            dt_base = torch.ones_like(t_vector) * math.log2(self.infer_steps)

            v = self.model(x, t_vector, dt_base, cond, segs_mask)

            x = x + v * delta_t

        # Convert to probabilities
        x = F.softmax(x.masked_fill(segs_mask == 0, -1e9), dim=-1)

        return x


# ============================================================================
# Helper function for bootstrapping target generation
# ============================================================================

def get_targets(model, inputs, cond, denoise_steps, device, segs_mask, bootstrap_every=8, force_t=-1, force_dt=-1):
    """Generate bootstrap targets for training.

    Creates training targets by combining flow-matching objectives with
    bootstrapped targets from the model itself.

    Args:
        model: DiT model
        inputs: Target one-hot distributions (batch, seq_len, num_cands)
        cond: Conditioning tensor
        denoise_steps: Number of denoising steps
        device: Computation device
        segs_mask: Valid segment mask
        bootstrap_every: Bootstrap frequency
        force_t: Force specific timestep (-1 for random)
        force_dt: Force specific step size (-1 for random)

    Returns:
        x_t: Noisy samples
        v_t: Target velocities
        t: Timesteps
        dt_base: Step size bases
    """
    model.eval()

    batch_size = inputs.shape[0]

    # Sample dt
    bootstrap_batchsize = batch_size // bootstrap_every
    log2_sections = int(math.log2(denoise_steps))

    dt_base = torch.repeat_interleave(
        log2_sections - 1 - torch.arange(log2_sections),
        bootstrap_batchsize // log2_sections
    )
    dt_base = torch.cat([dt_base, torch.zeros(bootstrap_batchsize - dt_base.shape[0],)])

    force_dt_vec = torch.ones(bootstrap_batchsize) * force_dt
    dt_base = torch.where(force_dt_vec != -1, force_dt_vec, dt_base).to(device)
    dt = 1 / (2 ** (dt_base))
    dt_base_bootstrap = dt_base + 1
    dt_bootstrap = dt / 2

    # Sample t
    dt_sections = 2 ** dt_base
    t = torch.cat([
        torch.randint(low=0, high=int(val.item()), size=(1,)).float() for val in dt_sections
    ]).to(device)
    t = t / dt_sections
    force_t_vec = torch.ones(bootstrap_batchsize, dtype=torch.float32).to(device) * force_t
    t = torch.where(force_t_vec != -1, force_t_vec, t).to(device)
    t_full = t[:, None, None]

    # Generate Bootstrap Targets
    x_1 = inputs[:bootstrap_batchsize]
    cond_bst = cond[:bootstrap_batchsize]
    segs_mask_bst = segs_mask[:bootstrap_batchsize]
    x_0 = torch.randn_like(x_1).masked_fill(segs_mask_bst == 0, 0)
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1

    with torch.no_grad():
        v_b1 = model(x_t, t, dt_base_bootstrap, cond_bst, segs_mask_bst)
    t2 = t + dt_bootstrap
    x_t2 = x_t + dt_bootstrap[:, None, None] * v_b1
    x_t2 = torch.clip(x_t2, -4, 4)

    with torch.no_grad():
        v_b2 = model(x_t2, t2, dt_base_bootstrap, cond_bst, segs_mask_bst)

    v_target = (v_b1 + v_b2) / 2
    v_target = torch.clip(v_target, -4, 4)
    v_target = v_target.masked_fill(segs_mask_bst == 0, 0)

    bst_v = v_target
    bst_dt = dt_base
    bst_t = t
    bst_xt = x_t

    # Generate Flow-Matching Targets
    t = torch.randint(low=0, high=denoise_steps, size=(inputs.shape[0],), dtype=torch.float32)
    t /= denoise_steps
    force_t_vec = torch.ones(inputs.shape[0]) * force_t
    t = torch.where(force_t_vec != -1, force_t_vec, t).to(device)
    t_full = t[:, None, None]

    # Sample flow pairs x_t, v_t
    x_0 = torch.randn_like(inputs).masked_fill(segs_mask == 0, 0)
    x_1 = inputs
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    v_t = x_1 - (1 - 1e-5) * x_0
    v_t = v_t.masked_fill(segs_mask == 0, 0)

    dt_flow = int(math.log2(denoise_steps))
    dt_base = (torch.ones(inputs.shape[0], dtype=torch.int32) * dt_flow).to(device)

    # Merge Flow and Bootstrap
    bst_size = batch_size // bootstrap_every
    bst_size_data = batch_size - bst_size
    x_t = torch.cat([bst_xt, x_t[-bst_size_data:]], dim=0)
    t = torch.cat([bst_t, t[-bst_size_data:]], dim=0)
    dt_base = torch.cat([bst_dt, dt_base[-bst_size_data:]], dim=0)
    v_t = torch.cat([bst_v, v_t[-bst_size_data:]], dim=0)

    return x_t, v_t, t, dt_base


# ============================================================================
# DiffMM: Main LibCity Model
# ============================================================================

class DiffMM(AbstractModel):
    """DiffMM: Diffusion-based Map Matching Model for LibCity.

    This model uses a diffusion transformer (DiT) to predict road segment
    probability distributions from GPS trajectory sequences. It combines
    a trajectory encoder with attention over candidate road segments and
    a one-step diffusion shortcut for efficient inference.

    Args:
        config (dict): Configuration dictionary containing model hyperparameters
        data_feature (dict): Data features including vocabulary sizes

    Required config parameters:
        - hid_dim: Hidden dimension (default: 256)
        - num_units: Number of units for denoiser (default: 512)
        - transformer_layers: Number of transformer layers (default: 2)
        - depth: Number of DiT blocks (default: 2)
        - timesteps: Diffusion timesteps for training (default: 2)
        - samplingsteps: Inference steps (default: 1)
        - dropout: Dropout probability (default: 0.1)
        - bootstrap_every: Bootstrap frequency (default: 8)
        - num_heads: Number of attention heads (default: 4)
        - num_cands: Number of candidate segments per point (default: 10)

    Required data_feature:
        - id_size: Number of unique road segment IDs
        - num_cands: Maximum number of candidate segments (if not in config)
    """

    def __init__(self, config, data_feature):
        super(DiffMM, self).__init__(config, data_feature)

        self._logger = getLogger()
        self.device = config.get('device', 'cpu')

        # Model hyperparameters
        self.hid_dim = config.get('hid_dim', 256)
        self.num_units = config.get('num_units', 512)
        self.transformer_layers = config.get('transformer_layers', 2)
        self.depth = config.get('depth', 2)
        self.timesteps = config.get('timesteps', 2)
        self.samplingsteps = config.get('samplingsteps', 1)
        self.dropout = config.get('dropout', 0.1)
        self.bootstrap_every = config.get('bootstrap_every', 8)
        self.num_heads = config.get('num_heads', 4)

        # Data dimensions
        self.id_size = data_feature.get('id_size', 10000)
        self.num_cands = config.get('num_cands', data_feature.get('num_cands', 10))

        # Build model components
        self._build_model()

        self._logger.info(
            f"DiffMM initialized: hid_dim={self.hid_dim}, "
            f"depth={self.depth}, timesteps={self.timesteps}, "
            f"samplingsteps={self.samplingsteps}, id_size={self.id_size}"
        )

    def _build_model(self):
        """Build model components."""

        # Trajectory encoder
        self.encoder = TrajEncoder(
            id_size=self.id_size,
            hid_dim=self.hid_dim,
            transformer_layers=self.transformer_layers,
            num_heads=self.num_heads,
            device=self.device
        )

        # DiT denoising network
        # Output dim is number of candidates per point
        # Conditioning dim is 2*hid_dim from encoder
        self.dit = DiT(
            out_dim=self.num_cands,
            hid_dim=self.hid_dim,
            depth=self.depth,
            cond_dim=2 * self.hid_dim
        )

        # ShortCut diffusion wrapper
        self.shortcut = ShortCut(
            model=self.dit,
            infer_steps=self.samplingsteps,
            seq_length=self.num_cands,
            bootstrap_every=self.bootstrap_every
        )

    def forward(self, batch):
        """Forward pass for training.

        Args:
            batch: LibCity Batch object containing:
                - 'src': GPS points (batch, seq_len, 3)
                - 'src_len': Sequence lengths (list)
                - 'src_segs': Candidate segment IDs (batch, seq_len, num_cands)
                - 'segs_feat': Segment features (batch, seq_len, num_cands, 9)
                - 'segs_mask': Valid segment mask (batch, seq_len, num_cands)
                - 'target': Target one-hot distribution (batch, seq_len, num_cands)

        Returns:
            loss: Training loss tensor
        """
        # Extract data from batch
        src = batch['src'].to(self.device).float()
        src_len = batch['src_len']
        if isinstance(src_len, torch.Tensor):
            src_len = src_len.tolist()
        src_segs = batch['src_segs'].to(self.device).long()
        segs_feat = batch['segs_feat'].to(self.device).float()
        segs_mask = batch['segs_mask'].to(self.device)
        target = batch['target'].to(self.device).float()

        # Encode trajectory
        cond = self.encoder(src, src_len, src_segs, segs_feat, segs_mask)

        # Get diffusion targets
        x_t, v_t, t, dt_base = get_targets(
            model=self.dit,
            inputs=target,
            cond=cond,
            denoise_steps=self.timesteps,
            device=self.device,
            segs_mask=segs_mask,
            bootstrap_every=self.bootstrap_every
        )

        # Compute loss
        loss = self.shortcut(x_t, v_t, t, dt_base, cond, target, segs_mask)

        return loss

    def infer(self, batch):
        """Inference to get predicted road segment probabilities.

        Args:
            batch: LibCity Batch object

        Returns:
            predictions: Probability distributions (batch, seq_len, num_cands)
        """
        # Extract data from batch
        src = batch['src'].to(self.device).float()
        src_len = batch['src_len']
        if isinstance(src_len, torch.Tensor):
            src_len = src_len.tolist()
        src_segs = batch['src_segs'].to(self.device).long()
        segs_feat = batch['segs_feat'].to(self.device).float()
        segs_mask = batch['segs_mask'].to(self.device)

        batch_size = src.size(0)

        # Encode trajectory
        cond = self.encoder(src, src_len, src_segs, segs_feat, segs_mask)

        # Generate predictions via diffusion sampling
        predictions = self.shortcut.inference(batch_size, cond, segs_mask)

        return predictions

    def predict(self, batch):
        """Prediction method for LibCity evaluation.

        Args:
            batch: Input batch dictionary

        Returns:
            Predicted road segment probabilities or indices
        """
        self.eval()
        with torch.no_grad():
            probs = self.infer(batch)

            # Return argmax predictions for evaluation
            predictions = probs.argmax(dim=-1)

            return predictions

    def calculate_loss(self, batch):
        """Calculate training loss for LibCity.

        Args:
            batch: LibCity Batch object

        Returns:
            loss: Training loss tensor
        """
        return self.forward(batch)
