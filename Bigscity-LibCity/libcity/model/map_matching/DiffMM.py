"""
DiffMM: Efficient Method for Accurate Noisy and Sparse Trajectory Map Matching via One Step Diffusion

This model is adapted from the original DiffMM implementation for LibCity framework.

Original paper:
"DiffMM: Efficient Method for Accurate Noisy and Sparse Trajectory Map Matching via One Step Diffusion" (AAAI)

Repository: https://github.com/decisionintelligence/DiffMM

Key Components:
1. TrajEncoder: Encodes GPS trajectories with candidate road segment attention
2. DiT (Diffusion Transformer): DiT blocks with adaptive layer normalization
3. ShortCut: One-step diffusion model with flow matching and bootstrap training

Adaptations for LibCity:
- Replaced standalone model classes with LibCity's AbstractModel
- Adapted __init__() to use LibCity's config and data_feature pattern
- Implemented predict() and calculate_loss() methods following LibCity conventions
- Extracted hyperparameters to config parameters
- Handled device management through LibCity's config
- Preserved flow matching and bootstrap training mechanisms
- Moved to map_matching task category (correct task type for GPS-to-road matching)

Original files:
- repos/DiffMM/models/model.py (TrajEncoder, ModelAllShortCut)
- repos/DiffMM/models/short_cut.py (ShortCut, DiT, DiTBlock, get_targets)
- repos/DiffMM/models/layers.py (PositionalEncoder, MultiHeadAttention, etc.)

Dependencies:
- torch
- einops (optional, for tensor operations)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from libcity.model.abstract_model import AbstractModel


# ========================== Helper Functions ==========================

def modulate(x, shift, scale):
    """Adaptive layer normalization modulation."""
    return x * (1 + scale) + shift


def sequence_mask(X, valid_len, value=0.):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def sequence_mask3d(X, valid_len, valid_len2, value=0.):
    """Mask irrelevant entries in 3D sequences."""
    maxlen = X.size(1)
    maxlen2 = X.size(2)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    mask2 = torch.arange((maxlen2), dtype=torch.float32,
                         device=X.device)[None, :] < valid_len2[:, None]
    mask_fin = torch.bmm(mask.float().unsqueeze(-1), mask2.float().unsqueeze(-2)).bool()
    X[~mask_fin] = value
    return X


# ========================== Layer Norm ==========================

class Norm(nn.Module):
    """Layer normalization with learnable parameters."""

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


# ========================== Positional Encoding ==========================

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_seq_len=500):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).to(x.device)
        return x


# ========================== Multi-Head Attention ==========================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

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


# ========================== Feed Forward ==========================

class FeedForward(nn.Module):
    """Feed-forward network with residual connection."""

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


# ========================== Encoder Layer ==========================

class EncoderLayer(nn.Module):
    """Transformer encoder layer."""

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


# ========================== Transformer Encoder ==========================

class TransformerEncoder(nn.Module):
    """Transformer encoder with positional encoding."""

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


# ========================== Point Encoder ==========================

class PointEncoder(nn.Module):
    """Encoder for GPS point sequences using Transformer."""

    def __init__(self, hid_dim, transformer_layers, dropout=0.1):
        super().__init__()
        self.hid_dim = hid_dim

        input_dim = 3  # [normalized_lat, normalized_lng, normalized_time]
        self.fc_point = nn.Linear(input_dim, hid_dim)
        self.transformer = TransformerEncoder(hid_dim, transformer_layers, heads=4)

    def forward(self, src, src_len):
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


# ========================== Cross Attention ==========================

class Attention(nn.Module):
    """Cross-attention between trajectory and road segments."""

    def __init__(self, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim

        self.attn = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.v = nn.Linear(self.hid_dim, 1, bias=False)

    def forward(self, query, key, value, attn_mask):
        batch_size, src_len = query.shape[0], query.shape[1]
        seg_num = key.shape[-2]

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


# ========================== Trajectory Encoder ==========================

class TrajEncoder(nn.Module):
    """
    Trajectory encoder combining GPS point encoding with road segment attention.

    Encodes GPS trajectories and their candidate road segments into a unified
    representation for the diffusion model.
    """

    def __init__(self, id_size, hid_dim, transformer_layers, dropout=0.1):
        super().__init__()
        self.id_size = id_size
        self.hid_dim = hid_dim
        self.id_emb_dim = hid_dim

        # Road segment ID embedding
        self.emb_id = nn.Parameter(torch.rand(self.id_size, self.id_emb_dim))

        # Road embedding: ID embedding + 9 features
        road_emb_input_dim = self.id_emb_dim + 9
        self.road_emb = nn.Sequential(
            nn.Linear(road_emb_input_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            Norm(self.hid_dim)
        )

        # Point encoder
        self.point_encoder = PointEncoder(hid_dim, transformer_layers, dropout)

        # Cross-attention
        self.attn = Attention(self.hid_dim)

        # Output projection
        self.output = nn.Linear(2 * hid_dim, hid_dim)

    def forward(self, src, src_len, src_segs, segs_feat, segs_mask):
        """
        Forward pass.

        Args:
            src: GPS sequences (batch_size, seq_len, 3) - [lat, lng, time]
            src_len: Sequence lengths
            src_segs: Candidate segment IDs (batch_size, seq_len, max_cand)
            segs_feat: Segment features (batch_size, seq_len, max_cand, 9)
            segs_mask: Candidate mask (batch_size, seq_len, max_cand)

        Returns:
            Encoded trajectory representations (batch_size, seq_len, 2*hid_dim)
        """
        # Get road segment embeddings
        src_id_emb = self.emb_id[src_segs.long()]
        src_road_emb = torch.cat((src_id_emb, segs_feat), dim=-1)
        road_emb = self.road_emb(src_road_emb)

        # Encode GPS points
        point_encoder_output = self.point_encoder(src, src_len)

        # Cross-attention between trajectory and road segments
        _, attention = self.attn(point_encoder_output, road_emb, road_emb, segs_mask)

        # Concatenate point encoding and attention output
        outputs = torch.cat((point_encoder_output, attention), dim=-1)

        return outputs


# ========================== Sinusoidal Position Embedding ==========================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding for diffusion timesteps."""

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


# ========================== DiT Block ==========================

class DiTBlock(nn.Module):
    """Diffusion Transformer block with adaptive layer normalization."""

    def __init__(self, hid_dim, num_heads=4, dropout=0.1):
        super(DiTBlock, self).__init__()
        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # Condition projection for modulation
        self.cond_linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hid_dim, 6 * hid_dim)
        )

        self.norm1 = Norm(hid_dim)
        self.norm2 = Norm(hid_dim)

        self.attn = MultiHeadAttention(num_heads, hid_dim, dropout)
        self.ff = FeedForward(hid_dim, d_ff=hid_dim * 2)

    def forward(self, x, c):
        # Generate modulation parameters from condition
        cond = self.cond_linear(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(cond, 6, dim=-1)

        # Modulated self-attention
        x_norm1 = self.norm1(x)
        x_modulated = modulate(x_norm1, shift_msa, scale_msa)
        attn_x = self.attn(x_modulated, x_modulated, x_modulated)
        x = x + (gate_msa * attn_x)

        # Modulated feed-forward
        x_norm2 = self.norm2(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_x = self.ff(x_modulated2)
        x = x + (gate_mlp * mlp_x)

        return x


# ========================== Output Layer ==========================

class OutputLayer(nn.Module):
    """Final output layer with modulation."""

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


# ========================== DiT (Diffusion Transformer) ==========================

class DiT(nn.Module):
    """
    Diffusion Transformer for one-step flow matching.

    Combines noisy input with trajectory condition and timestep embedding
    to predict the velocity field for denoising.
    """

    def __init__(self, out_dim, hid_dim, depth, cond_dim, num_heads=4, dropout=0.1):
        super(DiT, self).__init__()
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.depth = depth

        # Positional encoding
        self.pe = PositionalEncoder(hid_dim, max_seq_len=2000)

        # Time embeddings
        sinu_pos_emb = SinusoidalPosEmb(hid_dim)
        self.time_embedder = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(hid_dim, hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim, hid_dim)
        )

        # Timestep (dt) embeddings
        self.timestep_embedder = nn.Sequential(
            SinusoidalPosEmb(hid_dim),
            nn.Linear(hid_dim, hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim, hid_dim)
        )

        # Condition projection
        self.cond_linear = nn.Linear(cond_dim, hid_dim)

        # DiT blocks
        self.DiTBlocks = nn.ModuleList([
            DiTBlock(hid_dim, num_heads, dropout) for _ in range(depth)
        ])

        # Noise projection
        self.noise_linear = nn.Sequential(
            nn.Linear(out_dim, hid_dim),
            nn.ReLU()
        )

        # Output layer
        self.output = OutputLayer(hid_dim, out_dim)

    def forward(self, x, t, dt, cond, segs_mask):
        """
        Forward pass.

        Args:
            x: Noisy input (batch_size, 1, out_dim)
            t: Diffusion timestep (batch_size,)
            dt: Step size base (batch_size,)
            cond: Trajectory condition (batch_size, 1, cond_dim)
            segs_mask: Candidate segment mask (batch_size, 1, out_dim)

        Returns:
            Predicted velocity (batch_size, 1, out_dim)
        """
        x = self.noise_linear(x)
        x = self.pe(x)

        # Condition embedding
        c = self.cond_linear(cond)
        te = self.time_embedder(t)
        dte = self.timestep_embedder(dt)

        c = c + te[:, None] + dte[:, None]

        # DiT blocks
        for i in range(self.depth):
            x = self.DiTBlocks[i](x, c)

        # Output
        x = self.output(x, cond)

        # Apply mask
        x = x.masked_fill(segs_mask == 0, 0)
        return x


# ========================== Bootstrap Target Generation ==========================

def get_targets(model, inputs, cond, denoise_steps, device, segs_mask, bootstrap_every=8, force_t=-1, force_dt=-1):
    """
    Generate bootstrap and flow-matching targets for training.

    This implements the self-consistency training mechanism where the model
    is used to generate its own training targets at coarser time scales.

    Args:
        model: The DiT model
        inputs: Ground truth one-hot encoded segments
        cond: Trajectory conditions
        denoise_steps: Number of denoising steps
        device: Computation device
        segs_mask: Candidate segment mask
        bootstrap_every: Ratio of bootstrap samples (1/bootstrap_every)
        force_t: Force specific timestep (for debugging)
        force_dt: Force specific dt (for debugging)

    Returns:
        x_t: Noisy inputs at sampled timesteps
        v_t: Target velocities
        t: Sampled timesteps
        dt_base: Step size bases
    """
    model.eval()

    batch_size = inputs.shape[0]

    # 1. Sample dt for bootstrap samples
    bootstrap_batchsize = batch_size // bootstrap_every
    log2_sections = int(math.log2(denoise_steps))

    dt_base = torch.repeat_interleave(
        log2_sections - 1 - torch.arange(log2_sections),
        bootstrap_batchsize // log2_sections
    )
    dt_base = torch.cat([dt_base, torch.zeros(bootstrap_batchsize - dt_base.shape[0], )])

    force_dt_vec = torch.ones(bootstrap_batchsize) * force_dt
    dt_base = torch.where(force_dt_vec != -1, force_dt_vec, dt_base).to(device)
    dt = 1 / (2 ** (dt_base))
    dt_base_bootstrap = dt_base + 1
    dt_bootstrap = dt / 2

    # 2. Sample t for bootstrap samples
    dt_sections = 2 ** dt_base
    t = torch.cat([
        torch.randint(low=0, high=int(val.item()), size=(1,)).float() for val in dt_sections
    ]).to(device)
    t = t / dt_sections
    force_t_vec = torch.ones(bootstrap_batchsize, dtype=torch.float32).to(device) * force_t
    t = torch.where(force_t_vec != -1, force_t_vec, t).to(device)
    t_full = t[:, None, None]

    # 3. Generate bootstrap targets
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

    # 4. Generate flow-matching targets
    t = torch.randint(low=0, high=denoise_steps, size=(inputs.shape[0],), dtype=torch.float32)
    t /= denoise_steps
    force_t_vec = torch.ones(inputs.shape[0]) * force_t
    t = torch.where(force_t_vec != -1, force_t_vec, t).to(device)
    t_full = t[:, None, None]

    x_0 = torch.randn_like(inputs).masked_fill(segs_mask == 0, 0)
    x_1 = inputs
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    v_t = x_1 - (1 - 1e-5) * x_0
    v_t = v_t.masked_fill(segs_mask == 0, 0)

    dt_flow = int(math.log2(denoise_steps))
    dt_base = (torch.ones(inputs.shape[0], dtype=torch.int32) * dt_flow).to(device)

    # 5. Merge flow and bootstrap samples
    bst_size = batch_size // bootstrap_every
    bst_size_data = batch_size - bst_size
    x_t = torch.cat([bst_xt, x_t[-bst_size_data:]], dim=0)
    t = torch.cat([bst_t, t[-bst_size_data:]], dim=0)
    dt_base = torch.cat([bst_dt, dt_base[-bst_size_data:]], dim=0)
    v_t = torch.cat([bst_v, v_t[-bst_size_data:]], dim=0)

    return x_t, v_t, t, dt_base


# ========================== ShortCut Model ==========================

class ShortCutModel(nn.Module):
    """
    One-step diffusion model with flow matching.

    Enables efficient inference by learning to denoise in a single step
    while being trained on multi-step flow matching targets.
    """

    def __init__(self, model, infer_steps, seq_length, bootstrap_every=8):
        super().__init__()
        self.model = model
        self.infer_steps = infer_steps
        self.seq_length = seq_length
        self.bootstrap_every = bootstrap_every

    def forward(self, x_t, v_t, t, dt_base, cond, x_1, segs_mask):
        """
        Training forward pass.

        Args:
            x_t: Noisy input
            v_t: Target velocity
            t: Timestep
            dt_base: Step size base
            cond: Trajectory condition
            x_1: Ground truth one-hot
            segs_mask: Candidate mask

        Returns:
            Combined MSE and BCE loss
        """
        v_pred = self.model(x_t, t, dt_base, cond, segs_mask)
        x_pred = x_t + v_pred

        mse_loss = F.mse_loss(v_pred, v_t)
        bce_loss = F.binary_cross_entropy(
            F.softmax(x_pred.masked_fill(segs_mask == 0, -1e9), dim=-1),
            x_1,
            reduction='mean'
        )

        loss = mse_loss + bce_loss
        return loss

    @torch.no_grad()
    def inference(self, batch_size, cond, segs_mask):
        """
        Single-step or multi-step inference.

        Args:
            batch_size: Number of samples
            cond: Trajectory conditions
            segs_mask: Candidate segment masks

        Returns:
            Predicted probabilities over road segments
        """
        device = cond.device
        eps = torch.randn((batch_size, 1, self.seq_length), device=device)

        delta_t = 1.0 / self.infer_steps
        x = eps.masked_fill(segs_mask == 0, 0)

        for ti in range(self.infer_steps):
            t = ti / self.infer_steps

            t_vector = torch.full((eps.shape[0],), t).to(device)
            dt_base = torch.ones_like(t_vector).to(device) * math.log2(self.infer_steps)

            v = self.model(x, t_vector, dt_base, cond, segs_mask)
            x = x + v * delta_t

        x = F.softmax(x.masked_fill(segs_mask == 0, -1e9), dim=-1)

        return x


# ========================== Main DiffMM Model ==========================

class DiffMM(AbstractModel):
    """
    DiffMM: One-Step Diffusion Map Matching Model for LibCity.

    This model performs map matching by learning to match GPS trajectories
    to road segments using a one-step diffusion approach with DiT blocks.

    Key Features:
    1. TrajEncoder: Encodes GPS trajectories with candidate road segment attention
    2. DiT: Diffusion Transformer with adaptive layer normalization
    3. ShortCut: One-step inference with flow matching training
    4. Bootstrap training for self-consistency

    Args:
        config (dict): Configuration dictionary containing model hyperparameters
        data_feature (dict): Data features including vocab sizes, etc.

    Required config parameters:
        - hid_dim: Hidden dimension (default: 256)
        - num_units: Denoising network units (default: 512)
        - transformer_layers: Number of Transformer layers (default: 2)
        - depth: Number of DiT blocks (default: 2)
        - timesteps: Diffusion timesteps for training (default: 2)
        - samplingsteps: Inference sampling steps (default: 1)
        - dropout: Dropout rate (default: 0.1)
        - bootstrap_every: Bootstrap sample ratio (default: 8)

    Required data_feature:
        - id_size / loc_size / trg_seg_vocab_size: Number of road segments
    """

    def __init__(self, config, data_feature):
        super(DiffMM, self).__init__(config, data_feature)

        self.device = config.get('device', 'cpu')

        # Data dimensions from data_feature - support multiple naming conventions
        self.id_size = data_feature.get('id_size',
                       data_feature.get('loc_size',
                       data_feature.get('trg_seg_vocab_size', 10000)))

        # Model hyperparameters from config
        self.hid_dim = config.get('hid_dim', 256)
        self.num_units = config.get('num_units', 512)
        self.transformer_layers = config.get('transformer_layers', 2)
        self.depth = config.get('depth', 2)
        self.timesteps = config.get('timesteps', 2)
        self.samplingsteps = config.get('samplingsteps', 1)
        self.dropout = config.get('dropout', 0.1)
        self.bootstrap_every = config.get('bootstrap_every', 8)

        # Condition dimension (2 * hid_dim from TrajEncoder output)
        self.cond_dim = 2 * self.hid_dim

        # Build model components
        self._build_model()

    def _build_model(self):
        """Build all model components."""

        # Trajectory encoder
        self.encoder = TrajEncoder(
            id_size=self.id_size,
            hid_dim=self.hid_dim,
            transformer_layers=self.transformer_layers,
            dropout=self.dropout
        )

        # DiT model
        dit = DiT(
            out_dim=self.id_size - 1,  # One less for segment indexing
            hid_dim=self.num_units,
            depth=self.depth,
            cond_dim=self.cond_dim,
            num_heads=4,
            dropout=self.dropout
        )

        # ShortCut wrapper
        self.shortcut = ShortCutModel(
            model=dit,
            infer_steps=self.samplingsteps,
            seq_length=self.id_size - 1,
            bootstrap_every=self.bootstrap_every
        )

    def _init_weights(self):
        """Initialize weights following Keras convention."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def _batch2model(self, batch):
        """
        Transform LibCity batch to model input format.

        Uses direct dictionary access pattern for map_matching task compatibility.

        Args:
            batch: LibCity Batch object (dict-like)

        Returns:
            Tuple of (traj_cond, trg_rid, trg_onehot, lengths, src_segs_id, src_segs_mask, diff_mask)
        """
        # Extract data from batch using direct access with fallback
        # Primary keys for map_matching, fallback to trajectory_loc_pred keys
        try:
            norm_gps_seq = batch['norm_gps_seq']
        except (KeyError, TypeError):
            try:
                norm_gps_seq = batch['X']
            except (KeyError, TypeError):
                norm_gps_seq = None

        try:
            lengths = batch['lengths']
        except (KeyError, TypeError):
            try:
                lengths = batch['current_len']
            except (KeyError, TypeError):
                lengths = None

        try:
            trg_rid = batch['trg_rid']
        except (KeyError, TypeError):
            try:
                trg_rid = batch['target']
            except (KeyError, TypeError):
                trg_rid = None

        try:
            trg_onehot = batch['trg_onehot']
        except (KeyError, TypeError):
            trg_onehot = None

        try:
            segs_id = batch['segs_id']
        except (KeyError, TypeError):
            try:
                segs_id = batch['candidate_segs']
            except (KeyError, TypeError):
                segs_id = None

        try:
            segs_feat = batch['segs_feat']
        except (KeyError, TypeError):
            try:
                segs_feat = batch['candidate_features']
            except (KeyError, TypeError):
                segs_feat = None

        try:
            segs_mask = batch['segs_mask']
        except (KeyError, TypeError):
            try:
                segs_mask = batch['candidate_mask']
            except (KeyError, TypeError):
                segs_mask = None

        # Convert to tensors if needed
        if norm_gps_seq is not None and not isinstance(norm_gps_seq, torch.Tensor):
            norm_gps_seq = torch.FloatTensor(norm_gps_seq)
        if lengths is not None and not isinstance(lengths, torch.Tensor):
            lengths = torch.LongTensor(lengths)
        if trg_rid is not None and not isinstance(trg_rid, torch.Tensor):
            trg_rid = torch.LongTensor(trg_rid)
        if segs_id is not None and not isinstance(segs_id, torch.Tensor):
            segs_id = torch.LongTensor(segs_id)
        if segs_feat is not None and not isinstance(segs_feat, torch.Tensor):
            segs_feat = torch.FloatTensor(segs_feat)
        if segs_mask is not None and not isinstance(segs_mask, torch.Tensor):
            segs_mask = torch.FloatTensor(segs_mask)

        # Move to device
        norm_gps_seq = norm_gps_seq.to(self.device)
        lengths = lengths.to(self.device)
        trg_rid = trg_rid.to(self.device) if trg_rid is not None else None
        segs_id = segs_id.to(self.device)
        segs_feat = segs_feat.to(self.device)
        segs_mask = segs_mask.to(self.device)

        # Generate one-hot if not provided
        if trg_onehot is None and trg_rid is not None:
            batch_size, seq_len = trg_rid.shape[:2]
            trg_onehot = torch.zeros((batch_size, seq_len, self.id_size - 1), device=self.device)
            for i in range(batch_size):
                for j in range(seq_len):
                    if trg_rid[i, j] < self.id_size - 1:
                        trg_onehot[i, j, trg_rid[i, j]] = 1
        elif trg_onehot is not None and not isinstance(trg_onehot, torch.Tensor):
            trg_onehot = torch.FloatTensor(trg_onehot).to(self.device)
        else:
            trg_onehot = trg_onehot.to(self.device) if trg_onehot is not None else None

        # Encode trajectory
        enc_out = self.encoder(norm_gps_seq, lengths.tolist(), segs_id, segs_feat, segs_mask)

        # Flatten by GPS points
        traj_cond = []
        trg_rid_diff = []
        trg_onehot_diff = []
        src_segs_id = []
        src_segs_mask = []

        for index in range(enc_out.shape[0]):
            length = lengths[index].item()
            if length > 0:
                traj_cond += [i.unsqueeze(0) for i in enc_out[index][:length]]
                if trg_rid is not None:
                    trg_rid_diff += [i.unsqueeze(0) for i in trg_rid[index][:length]]
                if trg_onehot is not None:
                    trg_onehot_diff += [i.unsqueeze(0) for i in trg_onehot[index][:length]]
                src_segs_id += [i.unsqueeze(0) for i in segs_id[index][:length]]
                src_segs_mask += [i.unsqueeze(0) for i in segs_mask[index][:length]]

        traj_cond = torch.cat(traj_cond, dim=0) if traj_cond else None
        trg_rid_diff = torch.cat(trg_rid_diff, dim=0) if trg_rid_diff else None
        trg_onehot_diff = torch.cat(trg_onehot_diff, dim=0) if trg_onehot_diff else None
        src_segs_id = torch.cat(src_segs_id, dim=0) if src_segs_id else None
        src_segs_mask = torch.cat(src_segs_mask, dim=0) if src_segs_mask else None

        if traj_cond is not None:
            # Reshape for diffusion model
            trg_rid_diff = trg_rid_diff.reshape(-1, 1, 1) if trg_rid_diff is not None else None
            trg_onehot_diff = trg_onehot_diff.reshape(traj_cond.shape[0], 1, -1) if trg_onehot_diff is not None else None
            traj_cond = traj_cond.reshape(traj_cond.shape[0], 1, -1)
            src_segs_id = src_segs_id.reshape(traj_cond.shape[0], 1, -1)
            src_segs_mask_reshaped = src_segs_mask.reshape(traj_cond.shape[0], 1, -1)

            # Create diffusion mask from candidate segments
            diff_mask = torch.zeros((traj_cond.shape[0], 1, self.id_size - 1), device=self.device)
            for i, src_segs in enumerate(src_segs_id):
                seg_num = int(src_segs_mask_reshaped[i, 0].sum().item())
                valid_segs = src_segs[0, :seg_num] - 1
                valid_segs = valid_segs[valid_segs >= 0]
                valid_segs = valid_segs[valid_segs < self.id_size - 1]
                diff_mask[i, 0, valid_segs.long()] = 1

        return traj_cond, trg_rid_diff, trg_onehot_diff, lengths, src_segs_id, src_segs_mask, diff_mask

    def forward(self, batch):
        """
        Forward pass.

        Args:
            batch: LibCity Batch object

        Returns:
            Encoded trajectory conditions
        """
        traj_cond, _, _, lengths, _, _, _ = self._batch2model(batch)
        return traj_cond

    def predict(self, batch):
        """
        Prediction method for LibCity.

        Performs inference using the one-step diffusion model to predict
        road segment probabilities for each GPS point.

        Args:
            batch: LibCity Batch object

        Returns:
            Predicted road segment indices for each GPS point
        """
        traj_cond, trg_rid, _, lengths, _, _, diff_mask = self._batch2model(batch)

        if traj_cond is None:
            return torch.tensor([])

        # Run inference
        sampled_seq = self.shortcut.inference(
            batch_size=traj_cond.shape[0],
            cond=traj_cond,
            segs_mask=diff_mask
        )

        # Extract predictions per trajectory
        predictions = []
        cur_len = 0
        for length in lengths.tolist():
            traj_preds = []
            for i in range(length):
                pred = torch.argmax(sampled_seq[cur_len + i, 0]).item()
                traj_preds.append(pred)
            predictions.append(traj_preds)
            cur_len += length

        return predictions

    def calculate_loss(self, batch):
        """
        Calculate training loss.

        Uses flow matching with bootstrap targets for training.

        Args:
            batch: LibCity Batch object

        Returns:
            Combined MSE and BCE loss
        """
        traj_cond, trg_rid, trg_onehot, lengths, src_segs_id, src_segs_mask, diff_mask = self._batch2model(batch)

        if traj_cond is None or trg_onehot is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Get bootstrap and flow-matching targets
        x_t, v_t, t, dt_base = get_targets(
            self.shortcut.model,
            trg_onehot,
            traj_cond,
            self.timesteps,
            self.device,
            diff_mask,
            self.bootstrap_every
        )

        # Calculate loss
        loss = self.shortcut(x_t, v_t, t, dt_base, traj_cond, trg_onehot, diff_mask)

        return loss

    def get_predictions_with_probs(self, batch):
        """
        Get predictions with probability scores.

        Args:
            batch: LibCity Batch object

        Returns:
            Dict containing:
                - 'predictions': Predicted road segment indices
                - 'probabilities': Probability scores for each prediction
                - 'full_probs': Full probability distribution over segments
        """
        traj_cond, _, _, lengths, _, _, diff_mask = self._batch2model(batch)

        if traj_cond is None:
            return {'predictions': [], 'probabilities': [], 'full_probs': []}

        # Run inference
        sampled_seq = self.shortcut.inference(
            batch_size=traj_cond.shape[0],
            cond=traj_cond,
            segs_mask=diff_mask
        )

        # Extract predictions and probabilities
        predictions = []
        probabilities = []
        full_probs = []

        cur_len = 0
        for length in lengths.tolist():
            traj_preds = []
            traj_probs = []
            traj_full_probs = []

            for i in range(length):
                probs = sampled_seq[cur_len + i, 0]
                pred = torch.argmax(probs).item()
                prob = probs[pred].item()

                traj_preds.append(pred)
                traj_probs.append(prob)
                traj_full_probs.append(probs.cpu().numpy())

            predictions.append(traj_preds)
            probabilities.append(traj_probs)
            full_probs.append(traj_full_probs)
            cur_len += length

        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'full_probs': full_probs
        }
