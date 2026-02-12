"""
DiffMM: Diffusion-based Map Matching Model

Original Repository: repos/DiffMM
Adapted for LibCity framework by integrating:
- TrajEncoder (trajectory encoder with point transformer and attention)
- DiT (Diffusion Transformer with Adaptive Layer Normalization)
- ShortCut (Flow Matching with Bootstrap Training)

This model uses flow matching instead of traditional diffusion for fast inference (1-2 steps).

Task: Map Matching
This model matches GPS trajectories to road segments using diffusion-based flow matching.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from libcity.model.abstract_model import AbstractModel


# ==================== Utility Functions ====================

def modulate(x, shift, scale):
    """Apply adaptive layer normalization modulation."""
    return x * (1 + scale) + shift


# ==================== Layer Components ====================

class Norm(nn.Module):
    """Layer normalization module."""
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
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_seq_len=2000):
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
    """Multi-head attention mechanism."""
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
            scores = dropout(scores)

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


class EncoderLayer(nn.Module):
    """Transformer encoder layer."""
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForward(d_model, d_ff=d_model * 2, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x, mask):
        residual = x
        x = self.dropout_1(self.attn(x, x, x, mask))
        x2 = self.norm_1(residual + x)
        x = self.ff(x2)
        return x


class TransformerEncoder(nn.Module):
    """Multi-layer transformer encoder."""
    def __init__(self, d_model, N, heads, dropout=0.1):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, heads, dropout) for _ in range(N)
        ])
        self.norm = Norm(d_model)

    def forward(self, src, mask3d=None):
        x = self.pe(src)
        for i in range(self.N):
            x = self.layers[i](x, mask3d)
        return self.norm(x)


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


class PointEncoder(nn.Module):
    """Encode GPS points using transformer."""
    def __init__(self, hid_dim, transformer_layers=2, dropout=0.1):
        super().__init__()
        self.hid_dim = hid_dim

        input_dim = 3  # lat, lng, time
        self.fc_point = nn.Linear(input_dim, hid_dim)
        self.transformer = TransformerEncoder(hid_dim, transformer_layers, heads=4, dropout=dropout)

    def forward(self, src, src_len):
        max_src_len = src.size(1)
        batch_size = src.size(0)

        src_len = torch.tensor(src_len, device=src.device) if not isinstance(src_len, torch.Tensor) else src_len

        mask3d = torch.ones(batch_size, max_src_len, max_src_len, device=src.device)
        mask2d = torch.ones(batch_size, max_src_len, device=src.device)

        mask3d = sequence_mask3d(mask3d, src_len, src_len)
        mask2d = sequence_mask(mask2d, src_len).unsqueeze(-1).repeat(1, 1, self.hid_dim)

        src = self.fc_point(src)
        outputs = self.transformer(src, mask3d)

        outputs = outputs * mask2d

        return outputs


class Attention(nn.Module):
    """Attention mechanism for candidate road segments."""
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


# ==================== Trajectory Encoder ====================

class TrajEncoder(nn.Module):
    """Encode trajectory with GPS points and candidate road segments."""
    def __init__(self, id_size, hid_dim, transformer_layers=2, dropout=0.1):
        super().__init__()
        self.id_size = id_size
        self.hid_dim = hid_dim
        self.id_emb_dim = hid_dim

        self.emb_id = nn.Parameter(torch.rand(self.id_size, self.id_emb_dim))

        road_emb_input_dim = self.id_emb_dim + 9  # 9 road features
        self.road_emb = nn.Sequential(
            nn.Linear(road_emb_input_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            Norm(self.hid_dim)
        )
        self.point_encoder = PointEncoder(hid_dim, transformer_layers, dropout)
        self.attn = Attention(self.hid_dim)

        self.output = nn.Linear(2 * hid_dim, hid_dim)

    def forward(self, src, src_len, src_segs, segs_feat, segs_mask):
        """
        Args:
            src: [batch, seq_len, 3] GPS coordinates
            src_len: [batch] actual lengths
            src_segs: [batch, seq_len, max_candidates] candidate segment IDs
            segs_feat: [batch, seq_len, max_candidates, 9] segment features
            segs_mask: [batch, seq_len, max_candidates] mask
        """
        bs = src.size(0)

        src_id_emb = self.emb_id[src_segs]
        src_road_emb = torch.cat((src_id_emb, segs_feat), dim=-1)
        road_emb = self.road_emb(src_road_emb)

        point_encoder_output = self.point_encoder(src, src_len)
        _, attention = self.attn(point_encoder_output, road_emb, road_emb, segs_mask)

        outputs = torch.cat((point_encoder_output, attention), dim=-1)

        return outputs


# ==================== Diffusion Transformer (DiT) ====================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for time steps."""
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
    """Diffusion Transformer block with Adaptive Layer Normalization."""
    def __init__(self, hid_dim, num_heads=4, dropout=0.1):
        super(DiTBlock, self).__init__()
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
        self.ff = FeedForward(hid_dim, d_ff=hid_dim * 2, dropout=dropout)

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
    """Output layer with adaptive normalization."""
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
    """Diffusion Transformer for flow matching."""
    def __init__(self, out_dim, hid_dim, depth, cond_dim, dropout=0.1):
        super(DiT, self).__init__()
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.depth = depth

        sinu_pos_emb = SinusoidalPosEmb(hid_dim)
        fourier_dim = hid_dim
        time_dim = hid_dim

        self.pe = PositionalEncoder(hid_dim, max_seq_len=2000)

        self.time_embedder = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.timestep_embedder = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.cond_linear = nn.Linear(cond_dim, hid_dim)

        self.DiTBlocks = nn.ModuleList([
            DiTBlock(hid_dim, dropout=dropout) for _ in range(depth)
        ])

        self.noise_linear = nn.Sequential(
            nn.Linear(out_dim, hid_dim),
            nn.ReLU()
        )

        self.output = OutputLayer(hid_dim, out_dim)

    def forward(self, x, t, dt, cond, segs_mask):
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


# ==================== Flow Matching (ShortCut) ====================

def get_targets(model, inputs, cond, denoise_steps, device, segs_mask, bootstrap_every=8):
    """Generate flow matching targets with bootstrapping."""
    model.eval()

    batch_size = inputs.shape[0]

    # Sample dt (timestep granularity)
    bootstrap_batchsize = batch_size // bootstrap_every
    log2_sections = int(math.log2(denoise_steps))

    dt_base = torch.repeat_interleave(
        log2_sections - 1 - torch.arange(log2_sections),
        bootstrap_batchsize // log2_sections
    )
    dt_base = torch.cat([dt_base, torch.zeros(bootstrap_batchsize - dt_base.shape[0],)])
    dt_base = dt_base.to(device)
    dt = 1 / (2 ** dt_base)
    dt_base_bootstrap = dt_base + 1
    dt_bootstrap = dt / 2

    # Sample t (time)
    dt_sections = 2 ** dt_base
    t = torch.cat([
        torch.randint(low=0, high=int(val.item()), size=(1,)).float() for val in dt_sections
    ]).to(device)
    t = t / dt_sections
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
    t = t.to(device)
    t_full = t[:, None, None]

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


class ShortCut(nn.Module):
    """Flow matching model with fast inference."""
    def __init__(self, model, infer_steps, seq_length, bootstrap_every=8):
        super().__init__()

        self.model = model
        self.infer_steps = infer_steps
        self.seq_length = seq_length
        self.bootstrap_every = bootstrap_every

    def forward(self, x_t, v_t, t, dt_base, cond, x_1, segs_mask):
        """Training forward pass."""
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
        """Fast inference with flow matching."""
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


# ==================== Main DiffMM Model ====================

class DiffMM(AbstractModel):
    """
    DiffMM: Diffusion-based Map Matching Model.

    This model uses flow matching with bootstrap training for fast and accurate
    GPS trajectory map matching to road segments. It processes GPS trajectories
    and predicts road segments for each point.

    Task: Map Matching
    Base Class: AbstractModel (for neural map matching models)
    """

    def __init__(self, config, data_feature):
        super(DiffMM, self).__init__(config, data_feature)

        # Configuration
        self.device = config.get('device', torch.device('cpu'))
        self.hid_dim = config.get('hid_dim', 256)
        self.denoise_units = config.get('denoise_units', 512)
        self.transformer_layers = config.get('transformer_layers', 2)
        self.depth = config.get('depth', 2)
        self.timesteps = config.get('timesteps', 2)
        self.sampling_steps = config.get('sampling_steps', 1)
        self.bootstrap_every = config.get('bootstrap_every', 8)
        self.dropout = config.get('dropout', 0.1)

        # Data features
        self.id_size = data_feature.get('id_size')
        if self.id_size is None:
            raise ValueError("data_feature must contain 'id_size' (number of road segments)")

        # Initialize encoder
        self.encoder = TrajEncoder(
            id_size=self.id_size,
            hid_dim=self.hid_dim,
            transformer_layers=self.transformer_layers,
            dropout=self.dropout
        )

        # Initialize DiT (Diffusion Transformer)
        dit = DiT(
            out_dim=self.id_size - 1,  # Exclude padding ID
            hid_dim=self.denoise_units,
            depth=self.depth,
            cond_dim=2 * self.hid_dim,
            dropout=self.dropout
        )

        # Initialize ShortCut (Flow Matching)
        self.shortcut = ShortCut(
            model=dit,
            infer_steps=self.sampling_steps,
            seq_length=self.id_size - 1,
            bootstrap_every=self.bootstrap_every
        )

        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def batch2model(self, batch):
        """
        Transform LibCity batch format to DiffMM format.

        Expected batch format:
        - current_loc: [batch, seq_len, 3] GPS coordinates (lat, lng, time)
        - target: [batch, seq_len] ground truth segment IDs
        - candidate_segs: [batch, seq_len, max_candidates] candidate segment IDs
        - candidate_feats: [batch, seq_len, max_candidates, 9] candidate features
        - candidate_mask: [batch, seq_len, max_candidates] candidate validity mask
        """
        norm_gps_seq = batch['current_loc']
        trg_rid = batch.get('target', None)
        segs_id = batch['candidate_segs']
        segs_feat = batch['candidate_feats']
        segs_mask = batch['candidate_mask']

        lengths = batch.get('current_loc_len', [norm_gps_seq.size(1)] * norm_gps_seq.size(0))

        # Encode trajectory
        enc_out = self.encoder(norm_gps_seq, lengths, segs_id, segs_feat, segs_mask)

        # Flatten: process each point independently
        traj_cond = []
        trg_rid_diff = []
        trg_onehot_diff = []
        src_segs_id = []
        src_segs_mask = []

        for index in range(enc_out.shape[0]):
            length = lengths[index] if isinstance(lengths, list) else lengths[index].item()
            if length > 0:
                traj_cond += [enc_out[index][i].unsqueeze(0) for i in range(length)]
                if trg_rid is not None:
                    trg_rid_diff += [trg_rid[index][i].unsqueeze(0) for i in range(length)]
                src_segs_id += [segs_id[index][i].unsqueeze(0) for i in range(length)]
                src_segs_mask += [segs_mask[index][i].unsqueeze(0) for i in range(length)]

        if len(traj_cond) == 0:
            return None

        traj_cond = torch.cat(traj_cond, dim=0)
        src_segs_id = torch.cat(src_segs_id, dim=0)
        src_segs_mask = torch.cat(src_segs_mask, dim=0)

        # Prepare one-hot targets if available
        if trg_rid is not None and len(trg_rid_diff) > 0:
            trg_rid_diff = torch.cat(trg_rid_diff, dim=0)
            trg_onehot_diff = torch.zeros((trg_rid_diff.shape[0], self.id_size - 1), device=self.device)
            for i, rid in enumerate(trg_rid_diff):
                rid_idx = rid.item() if rid.item() > 0 else 0
                if 0 < rid_idx < self.id_size:
                    trg_onehot_diff[i, rid_idx - 1] = 1
        else:
            trg_rid_diff = None
            trg_onehot_diff = None

        # Reshape for processing
        if traj_cond.dim() == 2:
            traj_cond = traj_cond.reshape(-1, 1, traj_cond.size(-1))
        if trg_onehot_diff is not None:
            trg_onehot_diff = trg_onehot_diff.reshape(-1, 1, trg_onehot_diff.size(-1))
        if src_segs_id.dim() == 2:
            src_segs_id = src_segs_id.reshape(-1, 1, src_segs_id.size(-1))
        if src_segs_mask.dim() == 2:
            src_segs_mask = src_segs_mask.reshape(-1, 1, src_segs_mask.size(-1))

        # Create diffusion mask
        diff_mask = torch.zeros((traj_cond.shape[0], 1, self.id_size - 1), device=self.device)
        for i, src_segs in enumerate(src_segs_id):
            seg_num = src_segs_mask[i, 0].sum().item()
            valid_segs = src_segs[0, :int(seg_num)] - 1
            valid_segs = valid_segs[valid_segs >= 0]
            valid_segs = valid_segs[valid_segs < self.id_size - 1]
            if len(valid_segs) > 0:
                diff_mask[i, 0, valid_segs.long()] = 1

        return traj_cond, trg_rid_diff, trg_onehot_diff, lengths, src_segs_id, src_segs_mask, diff_mask

    def forward(self, batch):
        """Forward pass for training."""
        result = self.batch2model(batch)
        if result is None:
            return None

        traj_cond, trg_rid, trg_onehot, lengths, segs_id, segs_mask, mask = result

        if trg_onehot is None:
            return None

        # Generate flow matching targets
        x_t, v_t, t, dt_base = get_targets(
            self.shortcut.model,
            trg_onehot,
            traj_cond,
            self.timesteps,
            self.device,
            mask,
            self.bootstrap_every
        )

        # Calculate loss
        loss = self.shortcut(x_t, v_t, t, dt_base, traj_cond, trg_onehot, mask)

        return loss

    def predict(self, batch):
        """Prediction for inference."""
        result = self.batch2model(batch)
        if result is None:
            return None

        traj_cond, trg_rid, trg_onehot, lengths, segs_id, segs_mask, mask = result

        # Generate predictions
        sampled_seq = self.shortcut.inference(
            batch_size=traj_cond.shape[0],
            cond=traj_cond,
            segs_mask=mask
        )

        return sampled_seq

    def calculate_loss(self, batch):
        """Calculate training loss."""
        loss = self.forward(batch)
        if loss is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return loss
