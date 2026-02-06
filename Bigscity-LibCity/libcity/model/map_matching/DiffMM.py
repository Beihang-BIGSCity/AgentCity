# coding=utf-8
"""
Adapted DiffMM model for LibCity framework.

Original paper: DiffMM: One-Step Diffusion-based Map Matching
Original authors: [DiffMM Authors]
Adapted for LibCity by: Model Adaptation Agent

Key adaptations:
1. Inherits from AbstractModel instead of standalone nn.Module
2. Constructor signature changed to (config, data_feature) following LibCity conventions
3. Added predict() and calculate_loss() methods as required by LibCity
4. Adapted data format handling to work with LibCity batch format
5. Combined TrajEncoder, ShortCut, and DiT components into a single file

The model uses a one-step diffusion approach for map matching:
- TrajEncoder: Encodes GPS trajectories with attention to candidate road segments
- DiT: Diffusion Transformer backbone for denoising
- ShortCut: One-step diffusion model that accelerates inference

Expected batch format from DeepMapMatchingExecutor:
- src_gps: Normalized GPS sequences [batch, seq_len, 3] (lat, lng, time)
- src_lens: Sequence lengths [batch]
- src_cands: Candidate road segment IDs [batch, seq_len, num_cands]
- src_cands_feat: Candidate features [batch, seq_len, num_cands, feat_dim]
- src_cands_mask: Candidate mask [batch, seq_len, num_cands]
- tgt_roads: Target road segment IDs [batch, seq_len]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging

from libcity.model.abstract_model import AbstractModel

# Set up logger
_logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

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


def modulate(x, shift, scale):
    """Modulation function for DiT blocks."""
    return x * (1 + scale) + shift


def init_weights(module):
    """
    Initialize weights following Keras default initialization.
    Reference: https://github.com/vonfeng/DeepMove/blob/master/codes/model.py
    """
    for name, param in module.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)


# ============================================================================
# Layer Components
# ============================================================================

class Norm(nn.Module):
    """Layer normalization."""
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
    """Positional encoding for transformer."""
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


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer."""
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


class FeedForward(nn.Module):
    """Feed-forward network."""
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


class TransformerEncoder(nn.Module):
    """Transformer encoder."""
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


class PointEncoder(nn.Module):
    """Encodes GPS point sequences using transformer."""
    def __init__(self, hid_dim, transformer_layers, num_heads=4):
        super().__init__()
        self.hid_dim = hid_dim
        input_dim = 3  # (lat, lng, time)
        self.fc_point = nn.Linear(input_dim, hid_dim)
        self.transformer = TransformerEncoder(hid_dim, transformer_layers, heads=num_heads)

    def forward(self, src, src_len):
        max_src_len = src.size(1)
        batch_size = src.size(0)

        if not isinstance(src_len, torch.Tensor):
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


class Attention(nn.Module):
    """Attention layer for trajectory-road segment alignment."""
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


# ============================================================================
# TrajEncoder: Encodes GPS trajectories with road segment attention
# ============================================================================

class TrajEncoder(nn.Module):
    """
    Trajectory encoder that combines GPS point encoding with
    attention-based road segment matching.
    """
    def __init__(self, id_size, hid_dim, transformer_layers, num_heads=4):
        super().__init__()
        self.id_size = id_size
        self.hid_dim = hid_dim
        self.id_emb_dim = hid_dim

        # Road segment ID embedding
        self.emb_id = nn.Parameter(torch.rand(self.id_size, self.id_emb_dim))

        # Road embedding: combines ID embedding with features (9 features)
        road_emb_input_dim = self.id_emb_dim + 9
        self.road_emb = nn.Sequential(
            nn.Linear(road_emb_input_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            Norm(self.hid_dim)
        )

        # Point encoder (transformer-based)
        self.point_encoder = PointEncoder(hid_dim, transformer_layers, num_heads)

        # Attention for aligning points to road segments
        self.attn = Attention(self.hid_dim)

        # Output projection
        self.output = nn.Linear(2 * hid_dim, hid_dim)

    def forward(self, src, src_len, src_segs, segs_feat, segs_mask):
        """
        Args:
            src: GPS sequence [batch, seq_len, 3] (normalized lat, lng, time)
            src_len: Sequence lengths [batch]
            src_segs: Candidate road segment IDs [batch, seq_len, num_cands]
            segs_feat: Candidate features [batch, seq_len, num_cands, 9]
            segs_mask: Candidate mask [batch, seq_len, num_cands]

        Returns:
            outputs: Encoded trajectory [batch, seq_len, 2*hid_dim]
        """
        # Get road segment embeddings
        src_id_emb = self.emb_id[src_segs]  # [batch, seq_len, num_cands, hid_dim]
        src_road_emb = torch.cat((src_id_emb, segs_feat), dim=-1)
        road_emb = self.road_emb(src_road_emb)

        # Encode GPS points
        point_encoder_output = self.point_encoder(src, src_len)

        # Attention: align points to road segments
        _, attention = self.attn(point_encoder_output, road_emb, road_emb, segs_mask)

        # Concatenate point encoding and road attention
        outputs = torch.cat((point_encoder_output, attention), dim=-1)

        return outputs


# ============================================================================
# DiT: Diffusion Transformer
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""
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
    """Diffusion Transformer block with adaptive layer norm."""
    def __init__(self, hid_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.cond_linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hid_dim, 6 * hid_dim)
        )

        self.norm1 = Norm(hid_dim)
        self.norm2 = Norm(hid_dim)

        self.attn = MultiHeadAttention(num_heads, hid_dim, dropout)
        self.ff = FeedForward(hid_dim, d_ff=hid_dim * 2)

    def forward(self, x, c):
        cond = self.cond_linear(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(cond, 6, dim=-1)

        x_norm1 = self.norm1(x)
        x_modulated = modulate(x_norm1, shift_msa, scale_msa)
        attn_x = self.attn(x_modulated, x_modulated, x_modulated)
        x = x + (gate_msa * attn_x)

        x_norm2 = self.norm2(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_x = self.ff(x_modulated2)
        x = x + (gate_mlp * mlp_x)

        return x


class OutputLayer(nn.Module):
    """Final output layer for DiT."""
    def __init__(self, hid_dim, out_dim):
        super().__init__()
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
    """
    Diffusion Transformer for map matching.

    Takes noisy input and conditioning to predict velocity field.
    """
    def __init__(self, out_dim, hid_dim, depth, cond_dim, num_heads=4):
        super().__init__()
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.depth = depth

        sinu_pos_emb = SinusoidalPosEmb(hid_dim)
        time_dim = hid_dim

        self.pe = PositionalEncoder(hid_dim, max_seq_len=2000)

        # Time embedding
        self.time_embedder = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(hid_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Timestep embedding
        self.timestep_embedder = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(hid_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Condition projection
        self.cond_linear = nn.Linear(cond_dim, hid_dim)

        # DiT blocks
        self.DiTBlocks = nn.ModuleList([
            DiTBlock(hid_dim, num_heads) for _ in range(depth)
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
        Args:
            x: Noisy input [batch, 1, out_dim]
            t: Diffusion timesteps [batch]
            dt: Diffusion step sizes [batch]
            cond: Conditioning from trajectory encoder [batch, 1, cond_dim]
            segs_mask: Segment mask [batch, 1, out_dim]

        Returns:
            x: Predicted velocity [batch, 1, out_dim]
        """
        x = self.noise_linear(x)
        x = self.pe(x)

        c = self.cond_linear(cond)
        te = self.time_embedder(t)
        dte = self.timestep_embedder(dt)

        c = c + te[:, None] + dte[:, None]

        for i in range(self.depth):
            x = self.DiTBlocks[i](x, c)

        x = self.output(x, cond)
        x = x.masked_fill(segs_mask == 0, 0)

        return x


# ============================================================================
# ShortCut: One-step diffusion model
# ============================================================================

class ShortCut(nn.Module):
    """
    One-step diffusion model for accelerated inference.

    Uses shortcut learning to reduce diffusion steps while maintaining quality.
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
            x_t: Noisy input at time t
            v_t: Target velocity
            t: Diffusion timesteps
            dt_base: Base timestep sizes
            cond: Conditioning
            x_1: Target output
            segs_mask: Segment mask

        Returns:
            loss: Combined MSE and BCE loss
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
        Inference using one-step diffusion.

        Args:
            batch_size: Number of samples
            cond: Conditioning from trajectory encoder
            segs_mask: Segment mask

        Returns:
            x: Predicted probabilities over road segments
        """
        device = cond.device
        eps = torch.randn((batch_size, 1, self.seq_length), device=device)

        delta_t = 1.0 / self.infer_steps
        x = eps.masked_fill(segs_mask == 0, 0)

        for ti in range(self.infer_steps):
            t = ti / self.infer_steps
            t_vector = torch.full((eps.shape[0],), t, device=device)
            dt_base = torch.ones_like(t_vector, device=device) * math.log2(self.infer_steps)

            v = self.model(x, t_vector, dt_base, cond, segs_mask)
            x = x + v * delta_t

        x = F.softmax(x.masked_fill(segs_mask == 0, -1e9), dim=-1)
        return x


# ============================================================================
# Target Generation for Training
# ============================================================================

def get_targets(model, inputs, cond, denoise_steps, device, segs_mask, bootstrap_every=8, force_t=-1, force_dt=-1):
    """
    Generate training targets using bootstrap and flow-matching.

    This function creates training pairs (x_t, v_t) for the shortcut model
    by combining bootstrap targets and flow-matching targets.
    """
    model.eval()
    batch_size = inputs.shape[0]

    # 1. Sample dt for bootstrap
    bootstrap_batchsize = batch_size // bootstrap_every
    if bootstrap_batchsize == 0:
        bootstrap_batchsize = 1

    log2_sections = int(math.log2(denoise_steps)) if denoise_steps > 1 else 1

    dt_base = torch.repeat_interleave(
        log2_sections - 1 - torch.arange(log2_sections),
        max(bootstrap_batchsize // log2_sections, 1)
    )
    dt_base = torch.cat([dt_base, torch.zeros(bootstrap_batchsize - dt_base.shape[0],)])

    force_dt_vec = torch.ones(bootstrap_batchsize) * force_dt
    dt_base = torch.where(force_dt_vec != -1, force_dt_vec, dt_base).to(device)
    dt = 1 / (2 ** (dt_base))
    dt_base_bootstrap = dt_base + 1
    dt_bootstrap = dt / 2

    # 2. Sample t for bootstrap
    dt_sections = 2 ** dt_base
    t = torch.cat([
        torch.randint(low=0, high=max(int(val.item()), 1), size=(1,)).float()
        for val in dt_sections
    ]).to(device)
    t = t / torch.clamp(dt_sections, min=1)
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

    dt_flow = int(math.log2(denoise_steps)) if denoise_steps > 1 else 1
    dt_base = (torch.ones(inputs.shape[0], dtype=torch.int32) * dt_flow).to(device)

    # 5. Merge bootstrap and flow-matching
    bst_size = min(bootstrap_batchsize, batch_size)
    bst_size_data = batch_size - bst_size

    if bst_size_data > 0:
        x_t = torch.cat([bst_xt, x_t[-bst_size_data:]], dim=0)
        t = torch.cat([bst_t, t[-bst_size_data:]], dim=0)
        dt_base = torch.cat([bst_dt, dt_base[-bst_size_data:]], dim=0)
        v_t = torch.cat([bst_v, v_t[-bst_size_data:]], dim=0)
    else:
        x_t = bst_xt
        t = bst_t
        dt_base = bst_dt
        v_t = bst_v

    return x_t, v_t, t, dt_base


# ============================================================================
# DiffMM: Main Model Class for LibCity
# ============================================================================

class DiffMM(AbstractModel):
    """
    DiffMM: One-Step Diffusion-based Map Matching

    This model uses a one-step diffusion approach with shortcut learning
    for efficient and accurate map matching of GPS trajectories to road networks.

    Architecture:
    1. TrajEncoder: Encodes GPS trajectories with attention to candidate road segments
    2. DiT: Diffusion Transformer backbone
    3. ShortCut: One-step diffusion model for fast inference

    Args:
        config: LibCity configuration dictionary
        data_feature: Data feature dictionary containing:
            - id_size: Number of road segments + 1
            - num_cands: Maximum number of candidates per point
            - feat_dim: Feature dimension for candidates
    """

    def __init__(self, config, data_feature):
        super(DiffMM, self).__init__(config, data_feature)

        # Device configuration
        self.device = config.get('device', 'cpu')

        # Model hyperparameters from config
        self.hid_dim = config.get('hid_dim', 256)
        self.num_units = config.get('num_units', 512)
        self.transformer_layers = config.get('transformer_layers', 2)
        self.depth = config.get('depth', 2)
        self.num_heads = config.get('num_heads', 4)
        self.dropout = config.get('dropout', 0.1)
        self.timesteps = config.get('timesteps', 2)
        self.samplingsteps = config.get('samplingsteps', 1)
        self.bootstrap_every = config.get('bootstrap_every', 8)

        # Data features
        self.id_size = data_feature.get('id_size', data_feature.get('num_roads', 1000))
        self.num_cands = data_feature.get('num_cands', config.get('num_cands', 10))
        self.feat_dim = data_feature.get('feat_dim', 9)

        # Store data_feature for later use
        self.data_feature = data_feature

        _logger.info(f"DiffMM: Initializing with id_size={self.id_size}, "
                     f"hid_dim={self.hid_dim}, num_units={self.num_units}")

        # Build model components
        self._build_model()

        # Initialize weights
        self.apply(init_weights)

    def _build_model(self):
        """Build all model components."""
        # Trajectory encoder
        self.encoder = TrajEncoder(
            id_size=self.id_size,
            hid_dim=self.hid_dim,
            transformer_layers=self.transformer_layers,
            num_heads=self.num_heads
        )

        # DiT (Diffusion Transformer)
        # Output dimension is id_size - 1 (excluding padding)
        out_dim = self.id_size - 1 if self.id_size > 1 else self.id_size
        self.dit = DiT(
            out_dim=out_dim,
            hid_dim=self.num_units,
            depth=self.depth,
            cond_dim=2 * self.hid_dim,
            num_heads=self.num_heads
        )

        # ShortCut diffusion model
        self.shortcut = ShortCut(
            model=self.dit,
            infer_steps=self.samplingsteps,
            seq_length=out_dim,
            bootstrap_every=self.bootstrap_every
        )

    def _prepare_batch(self, batch):
        """
        Convert LibCity batch format to DiffMM expected format.

        Expected batch keys from DeepMapMatchingDataset:
        - src_gps: [batch, seq_len, 3] normalized GPS (lat, lng, time)
        - src_lens: [batch] sequence lengths
        - src_cands: [batch, seq_len, num_cands] candidate road IDs
        - src_cands_feat: [batch, seq_len, num_cands, feat_dim] candidate features
        - src_cands_mask: [batch, seq_len, num_cands] candidate mask
        - tgt_roads: [batch, seq_len] target road segment IDs

        Returns processed tensors for the model.
        """
        # Get source GPS sequences
        src_gps = batch.get('src_gps', batch.get('X', batch.get('input_gps', batch.get('norm_gps_seq'))))
        if src_gps is None:
            raise KeyError("Batch must contain 'src_gps', 'X', or 'input_gps'")

        # Get sequence lengths
        src_lens = batch.get('src_lens', batch.get('lengths'))
        if src_lens is None:
            # Infer lengths from GPS data
            src_lens = torch.full((src_gps.shape[0],), src_gps.shape[1], device=src_gps.device)

        # Get candidate road segments
        src_cands = batch.get('src_cands', batch.get('cands', batch.get('candidate_ids', batch.get('segs_id'))))
        if src_cands is None:
            raise KeyError("Batch must contain 'src_cands', 'cands', or 'candidate_ids'")

        # Get candidate features
        src_cands_feat = batch.get('src_cands_feat', batch.get('cands_feat', batch.get('candidate_feats', batch.get('segs_feat'))))
        if src_cands_feat is None:
            # Create zero features if not provided
            src_cands_feat = torch.zeros(
                (*src_cands.shape, self.feat_dim),
                device=src_cands.device
            )

        # Get candidate mask
        src_cands_mask = batch.get('src_cands_mask', batch.get('cands_mask', batch.get('candidate_mask', batch.get('segs_mask'))))
        if src_cands_mask is None:
            # Create mask from candidate IDs (non-zero = valid)
            src_cands_mask = (src_cands > 0).float()

        # Get target roads
        tgt_roads = batch.get('tgt_roads', batch.get('y', batch.get('target', batch.get('output_trg', batch.get('trg_rid')))))

        return {
            'src_gps': src_gps,
            'src_lens': src_lens,
            'src_cands': src_cands,
            'src_cands_feat': src_cands_feat,
            'src_cands_mask': src_cands_mask,
            'tgt_roads': tgt_roads
        }

    def _batch2model(self, batch):
        """
        Process batch through encoder and prepare for diffusion.

        This mirrors the batch2model function from the original code.
        """
        processed = self._prepare_batch(batch)

        src_gps = processed['src_gps']
        src_lens = processed['src_lens']
        src_cands = processed['src_cands']
        src_cands_feat = processed['src_cands_feat']
        src_cands_mask = processed['src_cands_mask']
        tgt_roads = processed['tgt_roads']

        # Encode trajectory
        enc_out = self.encoder(src_gps, src_lens, src_cands, src_cands_feat, src_cands_mask)

        # Flatten per-point for diffusion processing
        traj_cond = []
        trg_rid_diff = []
        trg_onehot_diff = []
        src_segs_id = []
        src_segs_mask = []

        batch_size = enc_out.shape[0]
        out_dim = self.id_size - 1 if self.id_size > 1 else self.id_size

        for idx in range(batch_size):
            length = int(src_lens[idx].item()) if torch.is_tensor(src_lens[idx]) else int(src_lens[idx])
            if length > 0:
                for i in range(length):
                    traj_cond.append(enc_out[idx, i:i+1])

                    if tgt_roads is not None:
                        rid = tgt_roads[idx, i]
                        trg_rid_diff.append(rid.unsqueeze(0) if torch.is_tensor(rid) else torch.tensor([rid], device=self.device))
                        # Create one-hot encoding
                        onehot = torch.zeros(out_dim, device=self.device)
                        if rid >= 0 and rid < out_dim:
                            onehot[int(rid)] = 1
                        trg_onehot_diff.append(onehot.unsqueeze(0))

                    src_segs_id.append(src_cands[idx, i:i+1])
                    src_segs_mask.append(src_cands_mask[idx, i:i+1])

        if len(traj_cond) == 0:
            return None  # Empty batch

        traj_cond = torch.cat(traj_cond, dim=0)  # [total_points, 2*hid_dim]
        src_segs_id = torch.cat(src_segs_id, dim=0)
        src_segs_mask = torch.cat(src_segs_mask, dim=0)

        if tgt_roads is not None and len(trg_rid_diff) > 0:
            trg_rid_diff = torch.cat(trg_rid_diff, dim=0)
            trg_onehot_diff = torch.cat(trg_onehot_diff, dim=0)
        else:
            trg_rid_diff = None
            trg_onehot_diff = None

        # Reshape for diffusion
        total_points = traj_cond.shape[0]
        traj_cond = traj_cond.reshape(total_points, 1, -1)  # [N, 1, 2*hid_dim]
        src_segs_id = src_segs_id.reshape(total_points, 1, -1)
        src_segs_mask = src_segs_mask.reshape(total_points, 1, -1)

        if trg_rid_diff is not None:
            trg_rid_diff = trg_rid_diff.reshape(total_points, 1, 1)
            trg_onehot_diff = trg_onehot_diff.reshape(total_points, 1, -1)

        # Create diffusion mask from candidate segments
        diff_mask = torch.zeros((total_points, 1, out_dim), device=self.device)
        for i in range(total_points):
            seg_num = int(src_segs_mask[i, 0].sum().item())
            if seg_num > 0:
                segs = src_segs_id[i, 0, :seg_num].long()
                valid_segs = segs[(segs > 0) & (segs <= out_dim)]
                if len(valid_segs) > 0:
                    diff_mask[i, 0, valid_segs - 1] = 1

        return {
            'traj_cond': traj_cond,
            'trg_rid': trg_rid_diff,
            'trg_onehot': trg_onehot_diff,
            'lengths': src_lens,
            'src_segs_id': src_segs_id,
            'src_segs_mask': src_segs_mask,
            'diff_mask': diff_mask
        }

    def forward(self, batch):
        """
        Forward pass through the model.

        This is primarily used during training.
        """
        model_input = self._batch2model(batch)
        if model_input is None:
            return None

        return model_input

    def predict(self, batch):
        """
        Predict road segments for given batch.

        Args:
            batch: Dictionary containing trajectory data

        Returns:
            predictions: Predicted road segment IDs [batch, seq_len]
        """
        model_input = self._batch2model(batch)
        if model_input is None:
            return torch.tensor([], device=self.device)

        traj_cond = model_input['traj_cond']
        diff_mask = model_input['diff_mask']
        lengths = model_input['lengths']

        # Run inference through shortcut model
        sampled_seq = self.shortcut.inference(
            batch_size=traj_cond.shape[0],
            cond=traj_cond,
            segs_mask=diff_mask
        )

        # Convert to predicted road IDs
        pred_ids = torch.argmax(sampled_seq.squeeze(1), dim=-1)  # [total_points]

        # Reshape back to batch format
        if lengths is not None:
            batch_size = len(lengths)
            max_len = max(int(l.item()) if torch.is_tensor(l) else int(l) for l in lengths)
            predictions = torch.full((batch_size, max_len), -1, dtype=torch.long, device=self.device)

            idx = 0
            for b in range(batch_size):
                length = int(lengths[b].item()) if torch.is_tensor(lengths[b]) else int(lengths[b])
                for i in range(length):
                    if idx < len(pred_ids):
                        predictions[b, i] = pred_ids[idx]
                        idx += 1
        else:
            predictions = pred_ids.unsqueeze(0)

        return predictions

    def calculate_loss(self, batch):
        """
        Calculate training loss.

        Uses the shortcut model's combined MSE and BCE loss.

        Args:
            batch: Dictionary containing trajectory data and targets

        Returns:
            loss: Combined training loss
        """
        model_input = self._batch2model(batch)
        if model_input is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        traj_cond = model_input['traj_cond']
        trg_onehot = model_input['trg_onehot']
        diff_mask = model_input['diff_mask']

        if trg_onehot is None:
            _logger.warning("No target roads in batch, returning zero loss")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Generate training targets using bootstrap + flow matching
        x_t, v_t, t, dt_base = get_targets(
            self.shortcut.model,
            trg_onehot,
            traj_cond,
            self.timesteps,
            self.device,
            diff_mask,
            self.bootstrap_every
        )

        # Set model to training mode
        self.train()

        # Calculate loss through shortcut model
        loss = self.shortcut(x_t, v_t, t, dt_base, traj_cond, trg_onehot, diff_mask)

        return loss
