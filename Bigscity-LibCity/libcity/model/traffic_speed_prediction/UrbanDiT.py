"""
UrbanDiT Model Adaptation for LibCity Framework
================================================

Original Paper: "UrbanDiT: A Foundation Model for Open-World Urban Spatio-Temporal Learning"
Original Repository: https://github.com/YuanYuan98/UrbanDiT

This adaptation ports the UrbanDiT model to LibCity conventions for traffic speed prediction.
The model is a Diffusion Transformer designed for urban spatio-temporal prediction using
Flow Matching diffusion process.

Key Adaptations:
- Inherits from AbstractTrafficStateModel
- Converts LibCity batch format (B, T, N, F) to model format (B, T, C, H, W) for grid data
  or handles graph data directly
- Implements Flow Matching training and inference
- Simplifies for forecasting task (can extend to multi-task later)
- Includes all necessary embedding and diffusion components

Author: Model Adaptation Agent
Date: 2026-01-30
"""

import math
import copy
import numpy as np
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


# =============================================================================
# Utility Functions
# =============================================================================

def get_2d_sincos_pos_embed(embed_dim, grid_size1, grid_size2):
    """
    Generate 2D sinusoidal positional embeddings.

    Args:
        embed_dim: Embedding dimension
        grid_size1: Height of grid
        grid_size2: Width of grid

    Returns:
        pos_embed: [grid_size1*grid_size2, embed_dim]
    """
    grid_h = np.arange(grid_size1, dtype=np.float32)
    grid_w = np.arange(grid_size2, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size1, grid_size2])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generate 1D sinusoidal positional embeddings.

    Args:
        embed_dim: Output dimension for each position
        pos: Positions to encode

    Returns:
        emb: [M, embed_dim]
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def modulate(x, shift, scale, T):
    """Apply adaptive layer normalization modulation."""
    N, M = x.shape[-2], x.shape[-1]
    B = scale.shape[0]
    if T == 1:
        B = x.shape[0]
    x = rearrange(x, '(b t) n m-> b (t n) m', b=B, t=T, n=N, m=M)
    x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    x = rearrange(x, 'b (t n) m-> (b t) n m', b=B, t=T, n=N, m=M)
    return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


# =============================================================================
# Embedding Modules
# =============================================================================

class TokenEmbedding_S(nn.Module):
    """Spatial token embedding using 2D convolution."""

    def __init__(self, c_in, d_model, patch_size):
        super(TokenEmbedding_S, self).__init__()
        kernel_size = [patch_size, patch_size]
        self.tokenConv = nn.Conv2d(
            in_channels=c_in, out_channels=d_model,
            kernel_size=kernel_size, stride=kernel_size
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x)
        x = x.flatten(2)
        x = torch.einsum("ncs->nsc", x)
        return x


class TokenEmbedding_ST(nn.Module):
    """Spatio-temporal token embedding using 3D convolution."""

    def __init__(self, c_in, d_model, patch_size, t_patch_len, stride):
        super(TokenEmbedding_ST, self).__init__()
        kernel_size = [t_patch_len, patch_size, patch_size]
        stride_size = [stride, patch_size, patch_size]
        self.tokenConv = nn.Conv3d(
            in_channels=c_in, out_channels=d_model,
            kernel_size=kernel_size, stride=stride_size
        )
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x)
        x = x.flatten(3)
        x = torch.einsum("ncts->ntsc", x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x


class TimeEmbedding(nn.Module):
    """Temporal embedding with weekday and hour-of-day encodings."""

    def __init__(self, d_model, t_patch_len, stride):
        super(TimeEmbedding, self).__init__()
        self.weekday_embed = nn.Embedding(7, d_model)
        self.hour_embed_24 = nn.Embedding(24, d_model)
        self.hour_embed_48 = nn.Embedding(48, d_model)
        self.hour_embed_96 = nn.Embedding(96, d_model)
        self.hour_embed_288 = nn.Embedding(288, d_model)
        self.padding_patch_layer = nn.ReplicationPad1d((0, t_patch_len - stride))
        self.timeconv = nn.Conv1d(
            in_channels=d_model, out_channels=d_model,
            kernel_size=t_patch_len, stride=stride
        )

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x_mark, hour_num='288'):
        if '24' in hour_num:
            TimeEmb = self.hour_embed_24(x_mark[:, :, 1]) + self.weekday_embed(x_mark[:, :, 0])
        elif '48' in hour_num:
            TimeEmb = self.hour_embed_48(x_mark[:, :, 1]) + self.weekday_embed(x_mark[:, :, 0])
        elif '288' in hour_num:
            TimeEmb = self.hour_embed_288(x_mark[:, :, 1]) + self.weekday_embed(x_mark[:, :, 0])
        elif '96' in hour_num:
            TimeEmb = self.hour_embed_96(x_mark[:, :, 1]) + self.weekday_embed(x_mark[:, :, 0])
        else:
            TimeEmb = self.hour_embed_288(x_mark[:, :, 1] % 288) + self.weekday_embed(x_mark[:, :, 0] % 7)
        TimeEmb = self.padding_patch_layer(TimeEmb.transpose(1, 2))
        TimeEmb = self.timeconv(TimeEmb).transpose(1, 2)
        return TimeEmb


class TimestepEmbedder(nn.Module):
    """Embeds scalar diffusion timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Memory(nn.Module):
    """Memory prompt module for task-aware learning."""

    def __init__(self, num_memory, memory_dim):
        super().__init__()
        self.num_memory = num_memory
        self.memory_dim = memory_dim

        self.memMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))
        self.keyMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))
        self.x_proj = nn.Linear(memory_dim, memory_dim)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.memMatrix, std=0.02)
        torch.nn.init.trunc_normal_(self.keyMatrix, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        assert x.shape[-1] == self.memMatrix.shape[-1] == self.keyMatrix.shape[-1]
        x_query = torch.tanh(self.x_proj(x))
        att_weight = F.linear(input=x_query, weight=self.keyMatrix)
        att_weight = F.softmax(att_weight, dim=-1)
        out = F.linear(att_weight, self.memMatrix.permute(1, 0))
        return dict(out=out, att_weight=att_weight)


# =============================================================================
# Attention Modules (from timm)
# =============================================================================

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# =============================================================================
# UrbanDiT Block
# =============================================================================

class UrbanDiTBlock(nn.Module):
    """A UrbanDiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, num_frames=24, is_spatial=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.num_frames = num_frames
        self.is_spatial = is_spatial

        # Temporal attention
        self.temporal_norm1 = nn.LayerNorm(hidden_size)
        self.temporal_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.temporal_fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, c, num_frames=None, pt=None, ps=None, pf=None, pms=None, pmt=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        T = self.num_frames if num_frames is None else num_frames
        K, N, M = x.shape
        B = K // T

        if num_frames is None:
            x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T, n=N, m=M)

            if pt is not None:
                x = torch.cat((pt.reshape(-1, 1, pt.shape[-1]), x), dim=1)
            if pf is not None:
                x = torch.cat((pf.reshape(-1, 1, pf.shape[-1]), x), dim=1)
            if pmt is not None:
                x = torch.cat((pmt, x), dim=1)

            res_temporal = self.temporal_attn(self.temporal_norm1(x))

            if pt is not None:
                res_temporal = res_temporal[:, 1:]
                x = x[:, 1:]
            if pf is not None:
                res_temporal = res_temporal[:, 1:]
                x = x[:, 1:]
            if pmt is not None:
                res_temporal = res_temporal[:, 1:]
                x = x[:, 1:]

            res_temporal = rearrange(res_temporal, '(b n) t m -> (b t) n m', b=B, t=T, n=N, m=M)
            res_temporal = self.temporal_fc(res_temporal)

            x = rearrange(x, '(b n) t m -> (b t) n m', b=B, t=T, n=N, m=M)
            x = x + res_temporal

        if ps is not None:
            x = torch.cat((ps.reshape(-1, 1, ps.shape[-1]), x), dim=1)
        if pms is not None:
            x = torch.cat((pms, x), dim=1)

        if self.is_spatial == 1:
            attn = self.attn(modulate(self.norm1(x), shift_msa, scale_msa, self.num_frames if num_frames is None else num_frames))

            if ps is not None:
                attn = attn[:, 1:]
                x = x[:, 1:]
            if pms is not None:
                attn = attn[:, 1:]
                x = x[:, 1:]

            attn = rearrange(attn, '(b t) n m-> b (t n) m', b=B, t=T, n=N, m=M)
            attn = gate_msa.unsqueeze(1) * attn
            attn = rearrange(attn, 'b (t n) m-> (b t) n m', b=B, t=T, n=N, m=M)
            x = x + attn

            mlp = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp, self.num_frames if num_frames is None else num_frames))
            mlp = rearrange(mlp, '(b t) n m-> b (t n) m', b=B, t=T, n=N, m=M)
            mlp = gate_mlp.unsqueeze(1) * mlp
            mlp = rearrange(mlp, 'b (t n) m-> (b t) n m', b=B, t=T, n=N, m=M)
            x = x + mlp

        return x


class FinalLayer(nn.Module):
    """The final layer of UrbanDiT."""

    def __init__(self, hidden_size, patch_size, out_channels, num_frames, stride, dim=1):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels * stride * dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.num_frames = num_frames

    def forward(self, x, c, num_frames=None):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale, self.num_frames if num_frames is None else num_frames)
        x = self.linear(x)
        return x


# =============================================================================
# Flow Matching Diffusion
# =============================================================================

class FlowMatchingScheduler:
    """Simple Flow Matching scheduler for training and inference."""

    def __init__(self, num_train_timesteps=1000, shift=3.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

        # Create timesteps
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps)
        self.timesteps = torch.from_numpy(timesteps).long()

        # Compute sigmas for flow matching
        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.sigmas = torch.from_numpy(sigmas).float()

    def add_noise(self, original_samples, noise, timesteps):
        """Add noise according to flow matching: x_t = (1 - sigma) * x + sigma * noise."""
        device = original_samples.device
        sigmas = self.sigmas.to(device)[timesteps]
        while len(sigmas.shape) < len(original_samples.shape):
            sigmas = sigmas.unsqueeze(-1)
        noisy_samples = (1.0 - sigmas) * original_samples + sigmas * noise
        return noisy_samples

    def get_velocity(self, sample, noise):
        """Get velocity target for flow matching: v = noise - sample."""
        return noise - sample

    def step(self, model_output, timestep, sample, generator=None):
        """Single denoising step."""
        device = sample.device
        sigma = self.sigmas.to(device)[timestep]
        if len(sigma.shape) == 0:
            sigma = sigma.unsqueeze(0)
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        # Flow matching step: x_{t-1} = x_t - dt * v
        dt = 1.0 / self.num_train_timesteps
        prev_sample = sample - dt * model_output
        return prev_sample


# =============================================================================
# Video Mask Generator
# =============================================================================

class VideoMaskGenerator:
    """Generates masks for different prediction tasks."""

    def __init__(self, input_size, pred_len=12, his_len=12):
        self.length, self.height, self.width = input_size
        self.pred_len = pred_len
        self.his_len = his_len

    def temporal_mask(self, idx=0):
        """Generate temporal mask for prediction task."""
        mask = np.zeros((self.length, self.height, self.width))
        if idx == 0:  # Prediction: mask future
            mask[self.his_len:] = 1
        elif idx == 1:  # Backward: mask past
            mask[:-self.his_len] = 1
        return mask

    def __call__(self, batch_size=1, device=None, idx=0):
        if idx < 4:
            mask = self.temporal_mask(idx)
        else:
            mask = np.random.sample((self.length, self.height, self.width))
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
        return torch.tensor(mask, device=device).unsqueeze(0).repeat(batch_size, 1, 1, 1).int()


# =============================================================================
# Core UrbanDiT Model
# =============================================================================

class UrbanDiTCore(nn.Module):
    """Core Diffusion Transformer model for urban spatio-temporal prediction."""

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=1,
        hidden_size=384,
        depth=6,
        num_heads=6,
        mlp_ratio=2,
        pred_len=12,
        his_len=12,
        t_patch_len=2,
        stride=2,
        is_spatial=1,
        is_prompt=0,
        prompt_content='psptpfpm',
        num_memory=512,
        fft=0,
        fft_thred=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.pred_len = pred_len
        self.his_len = his_len
        self.t_patch_len = t_patch_len
        self.stride = stride
        self.is_spatial = is_spatial
        self.is_prompt = is_prompt
        self.prompt_content = prompt_content
        self.fft = fft
        self.fft_thred = fft_thred

        # Token embeddings
        self.x_embedder = TokenEmbedding_ST(in_channels, hidden_size, patch_size, t_patch_len, stride)
        self.mask_embedder = TokenEmbedding_ST(in_channels, hidden_size, patch_size, t_patch_len, stride)

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.time_embedder = TimeEmbedding(hidden_size, t_patch_len, stride)

        # Number of frames after patching
        self.num_frames = (pred_len + his_len) // stride
        self.time_embed = nn.Parameter(
            torch.zeros(1, pred_len + his_len, hidden_size), requires_grad=False
        )
        self.time_drop = nn.Dropout(p=0)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            UrbanDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                         num_frames=self.num_frames, is_spatial=is_spatial)
            for _ in range(depth)
        ])

        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, self.num_frames, stride)

        self.W_P = nn.Linear(t_patch_len, 1)
        self.padding_patch_layer = nn.ReplicationPad1d((0, t_patch_len - stride))

        # Memory modules for prompting
        self.enc_memory_temporal = Memory(num_memory=num_memory, memory_dim=hidden_size)
        self.enc_memory_spatial = Memory(num_memory=num_memory, memory_dim=hidden_size)
        self.enc_memory_freq = Memory(num_memory=num_memory, memory_dim=hidden_size)

        # Encoder attention for prompts
        self.encoder_t = Attention(hidden_size, num_heads=4, qkv_bias=True)
        self.encoder_s = Attention(hidden_size, num_heads=4, qkv_bias=True)

        # FFT encoder
        seq_len = pred_len + his_len
        if fft == 0:
            self.encoder_f = TokenEmbedding_S(seq_len + 2, hidden_size, patch_size)
            self.linear_f = nn.Linear(seq_len + 2, hidden_size)
        else:
            self.encoder_f = TokenEmbedding_S(seq_len, hidden_size, patch_size)
            self.linear_f = nn.Linear(seq_len, hidden_size)

        self.mask_t = Attention(hidden_size, num_heads=1, qkv_bias=True)
        self.mask_s = Attention(hidden_size, num_heads=1, qkv_bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def get_prompt(self, raw_x, B, T, x_f):
        _, M, D = raw_x.shape
        x_t = raw_x.reshape(B, T, M, D).permute(0, 2, 1, 3).reshape(-1, T, D)
        x_t = self.encoder_t(x_t)[:, -1]
        x_s = self.encoder_s(raw_x)[:, -1]

        prompt_f = self.enc_memory_freq(x_f.reshape(-1, x_f.shape[-1]))['out'].reshape(B, M, D)
        prompt_t = self.enc_memory_temporal(x_t)['out'].reshape(B, M, D)
        prompt_s = self.enc_memory_spatial(x_s)['out'].reshape(B, T, D)

        return dict(pt=prompt_t, ps=prompt_s, pf=prompt_f)

    def get_pos_emb(self, size1, size2):
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, size1, size2)
        pos_emb = nn.Parameter(torch.zeros(pos_embed.shape).unsqueeze(dim=0), requires_grad=False)
        pos_emb.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        return pos_emb

    def get_time_emb(self):
        grid_num_frames = np.arange(self.num_frames, dtype=np.float32)
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, grid_num_frames)
        pos_emb = nn.Parameter(torch.zeros(pos_embed.shape).unsqueeze(dim=0), requires_grad=False)
        pos_emb.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        return pos_emb

    def unpatchify(self, x, h, w, p=None):
        """Reverse the patching operation."""
        c = self.out_channels
        if p is None:
            p = self.patch_size

        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.stride, c))
        x = torch.einsum('nhwpqsc->nschpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.stride, c, h * p, w * p))
        return imgs

    def forward(self, x, t, timestamps, hour_num='288', mask=None):
        """
        Forward pass of UrbanDiT.

        Args:
            x: (B, T, C, H, W) tensor of spatial inputs
            t: (B,) tensor of diffusion timesteps
            timestamps: (B, T, 2) tensor of temporal info
            hour_num: String indicating time granularity
            mask: (B, T, H, W) tensor of masks

        Returns:
            Output predictions
        """
        B, T, C, H, W = x.shape
        device = x.device

        # FFT processing
        if self.fft == 0:
            x_f = torch.fft.rfft(x.permute(0, 2, 3, 4, 1), n=T, norm="ortho", dim=-1).squeeze(dim=1)
            x_f = torch.cat((x_f.real, x_f.imag), dim=-1).permute(0, 3, 1, 2)
        else:
            x_f = torch.fft.rfft(x.permute(0, 2, 3, 4, 1), n=T, norm="ortho", dim=-1).squeeze(dim=1)
            amplitude = torch.abs(x_f)
            threshold = amplitude.mean()
            mask_temp = amplitude > threshold
            x_f = torch.fft.irfft(x_f * mask_temp, n=T, norm="ortho", dim=-1).permute(0, 3, 1, 2)

        x_f = self.encoder_f(x_f)
        pos_embed = self.get_pos_emb(x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size)

        time_pos_embed = self.get_time_emb().unsqueeze(dim=2).repeat(B, 1, pos_embed.shape[-2], 1)
        time_pos_embed = time_pos_embed.reshape(-1, pos_embed.shape[-2], pos_embed.shape[-1])

        D = pos_embed.shape[-1]

        # Token embedding
        T_patched = T // self.stride
        x = x.contiguous().permute(0, 2, 1, 3, 4)
        x = self.x_embedder(x).reshape(B, T_patched, -1, D)

        if mask is not None:
            mask_emb = self.mask_embedder(mask.unsqueeze(dim=1).float()).reshape(B * T_patched, -1, D)

        time_embed = self.time_embedder(timestamps, hour_num).unsqueeze(dim=-2).repeat(1, 1, x.shape[2], 1)
        time_embed = time_embed.view(B * T_patched, -1, D)

        x = x.reshape(B * T_patched, -1, D)

        # Prompts
        prompt = dict(pt=None, ps=None, pf=None, pmt=None, pms=None)
        if self.is_prompt == 1:
            if 'pt' in self.prompt_content or 'ps' in self.prompt_content or 'pf' in self.prompt_content:
                prompt = self.get_prompt(x, B, T_patched, x_f)
                if 'pt' not in self.prompt_content:
                    prompt['pt'] = None
                if 'ps' not in self.prompt_content:
                    prompt['ps'] = None
                if 'pf' not in self.prompt_content:
                    prompt['pf'] = None
            if 'pm' in self.prompt_content and mask is not None:
                prompt['pms'] = self.mask_s(mask_emb)[:, -1:]
                prompt['pmt'] = self.mask_t(
                    rearrange(mask_emb, '(b t) n m -> (b n) t m', b=B, t=T_patched,
                             n=mask_emb.shape[1], m=mask_emb.shape[2])
                )[:, -1:]

        x = x + time_embed + pos_embed.to(device) + time_pos_embed.to(device)

        # Temporal processing
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T_patched)
        x = self.time_drop(x)
        x = rearrange(x, '(b n) t m -> (b t) n m', b=B, t=T_patched)

        # Timestep embedding
        t_emb = self.t_embedder(t)
        c = t_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, num_frames=None, pt=prompt['pt'], ps=prompt['ps'],
                     pf=prompt['pf'], pms=prompt['pms'], pmt=prompt['pmt'])

        # Final layer
        x = self.final_layer(x, c, num_frames=None)

        # Unpatchify
        x = self.unpatchify(x, H // self.patch_size, W // self.patch_size)
        x = x.view(B, T_patched, x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1])
        x = x.reshape(B, -1, x.shape[-3], x.shape[-2], x.shape[-1])

        return x


# =============================================================================
# LibCity Wrapper
# =============================================================================

class UrbanDiT(AbstractTrafficStateModel):
    """
    UrbanDiT: Diffusion Transformer for Urban Spatio-Temporal Prediction.

    This model uses Flow Matching diffusion for traffic prediction. It can handle
    both grid-based and graph-based traffic data.

    Args:
        config: LibCity configuration dictionary
        data_feature: Data feature dictionary containing:
            - adj_mx: Adjacency matrix
            - num_nodes: Number of nodes
            - feature_dim: Input feature dimension
            - output_dim: Output feature dimension
            - scaler: Data scaler for inverse transform
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))

        # Data features
        self._scaler = self.data_feature.get('scaler')
        self.adj_mx = self.data_feature.get('adj_mx')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)

        # Model hyperparameters
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.hidden_size = config.get('hidden_size', 384)
        self.depth = config.get('depth', 6)
        self.num_heads = config.get('num_heads', 6)
        self.mlp_ratio = config.get('mlp_ratio', 2)
        self.patch_size = config.get('patch_size', 2)
        self.t_patch_len = config.get('t_patch_len', 2)
        self.stride = config.get('stride', 2)
        self.is_spatial = config.get('is_spatial', 1)
        self.is_prompt = config.get('is_prompt', 0)
        self.prompt_content = config.get('prompt_content', 'psptpfpm')
        self.num_memory = config.get('num_memory', 512)

        # Diffusion parameters
        self.diffusion_steps = config.get('diffusion_steps', 200)
        self.num_inference_steps = config.get('num_inference_steps', 20)
        self.weighting_scheme = config.get('weighting_scheme', 'logit_normal')
        self.precondition_outputs = config.get('precondition_outputs', 1)

        # Grid size for data reshaping
        self.grid_height = config.get('grid_height', None)
        self.grid_width = config.get('grid_width', None)

        # Auto-detect grid size if not provided
        if self.grid_height is None or self.grid_width is None:
            grid_size = int(math.sqrt(self.num_nodes))
            if grid_size * grid_size == self.num_nodes:
                self.grid_height = grid_size
                self.grid_width = grid_size
            else:
                # Find closest rectangular shape
                for h in range(int(math.sqrt(self.num_nodes)), 0, -1):
                    if self.num_nodes % h == 0:
                        self.grid_height = h
                        self.grid_width = self.num_nodes // h
                        break

        self._logger.info(f'Using grid size: {self.grid_height} x {self.grid_width}')

        # Compute padded dimensions (must be divisible by patch_size)
        pad_h = (self.patch_size - self.grid_height % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - self.grid_width % self.patch_size) % self.patch_size
        self.padded_height = self.grid_height + pad_h
        self.padded_width = self.grid_width + pad_w
        if pad_h > 0 or pad_w > 0:
            self._logger.info(f'Padding grid from {self.grid_height}x{self.grid_width} to {self.padded_height}x{self.padded_width}')

        # Build core model (use padded dimensions)
        total_len = self.input_window + self.output_window
        self.model = UrbanDiTCore(
            input_size=max(self.padded_height, self.padded_width),
            patch_size=self.patch_size,
            in_channels=self.output_dim,
            hidden_size=self.hidden_size,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            pred_len=self.output_window,
            his_len=self.input_window,
            t_patch_len=self.t_patch_len,
            stride=self.stride,
            is_spatial=self.is_spatial,
            is_prompt=self.is_prompt,
            prompt_content=self.prompt_content,
            num_memory=self.num_memory,
        )

        # Flow matching scheduler
        self.noise_scheduler = FlowMatchingScheduler(
            num_train_timesteps=self.diffusion_steps,
            shift=3.0
        )

        # Mask generator for prediction task (use padded dimensions)
        self.mask_generator = VideoMaskGenerator(
            (total_len, self.padded_height, self.padded_width),
            pred_len=self.output_window,
            his_len=self.input_window
        )

        self._logger.info(f'UrbanDiT initialized with {sum(p.numel() for p in self.parameters()):,} parameters')

    def _reshape_to_grid(self, x):
        """
        Reshape input from (B, T, N, F) to (B, T, F, H, W).
        Pads grid dimensions to be divisible by patch_size.

        Args:
            x: Input tensor of shape (B, T, N, F)

        Returns:
            Reshaped tensor of shape (B, T, F, H_padded, W_padded)
        """
        B, T, N, F_dim = x.shape
        assert N == self.grid_height * self.grid_width, f"Number of nodes {N} doesn't match grid size {self.grid_height}x{self.grid_width}"
        x = x.reshape(B, T, self.grid_height, self.grid_width, F_dim)
        x = x.permute(0, 1, 4, 2, 3)  # (B, T, F, H, W)

        # Pad grid dimensions to be divisible by patch_size
        pad_h = (self.patch_size - self.grid_height % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - self.grid_width % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            # F.pad expects (left, right, top, bottom) for last 2 dims
            x = F.pad(x, (0, pad_w, 0, pad_h))

        return x

    def _reshape_from_grid(self, x):
        """
        Reshape output from (B, T, F, H, W) to (B, T, N, F).
        Crops back to original grid dimensions.

        Args:
            x: Input tensor of shape (B, T, F, H_padded, W_padded)

        Returns:
            Reshaped tensor of shape (B, T, N, F)
        """
        B, T, F_dim, H, W = x.shape

        # Crop back to original grid dimensions (remove padding)
        x = x[:, :, :, :self.grid_height, :self.grid_width]

        x = x.permute(0, 1, 3, 4, 2)  # (B, T, H, W, F)
        x = x.reshape(B, T, self.grid_height * self.grid_width, F_dim)
        return x

    def _create_timestamps(self, batch, total_len):
        """Create timestamp tensor from batch."""
        x = batch['X']
        B = x.shape[0]
        device = x.device

        # Try to extract temporal features
        if x.shape[-1] > self.output_dim:
            # Temporal features available
            x_te = x[:, :, 0, self.output_dim:self.output_dim + 2]  # (B, T_in, 2)
            if 'y' in batch and batch['y'].shape[-1] > self.output_dim:
                y_te = batch['y'][:, :, 0, self.output_dim:self.output_dim + 2]  # (B, T_out, 2)
                timestamps = torch.cat([x_te, y_te], dim=1)
            else:
                # Extrapolate timestamps for output window
                last_te = x_te[:, -1:, :]
                future_te = last_te.repeat(1, self.output_window, 1)
                timestamps = torch.cat([x_te, future_te], dim=1)
        else:
            # Create default timestamps
            timestamps = torch.zeros(B, total_len, 2, device=device, dtype=torch.long)
            # Day of week (0-6) and time of day
            for i in range(total_len):
                timestamps[:, i, 0] = i // 288 % 7  # day of week
                timestamps[:, i, 1] = i % 288  # time of day

        return timestamps.long()

    def _sample_timesteps(self, batch_size, device):
        """Sample random timesteps for training."""
        u = torch.rand(batch_size, device=device)
        indices = (u * self.diffusion_steps).long()
        return indices

    def forward(self, batch):
        """
        Forward pass for training.

        Args:
            batch: Dictionary with 'X' and 'y' tensors

        Returns:
            Model predictions
        """
        x = batch['X']  # (B, T_in, N, F)
        y = batch['y']  # (B, T_out, N, F)

        B = x.shape[0]
        device = x.device

        # Prepare input: concatenate history and future
        x_traffic = x[:, :, :, :self.output_dim]
        y_traffic = y[:, :, :, :self.output_dim]
        full_seq = torch.cat([x_traffic, y_traffic], dim=1)  # (B, T_total, N, F)

        # Reshape to grid format
        full_seq_grid = self._reshape_to_grid(full_seq)  # (B, T, F, H, W)

        # Generate prediction mask
        mask = self.mask_generator(B, device, idx=0)  # Prediction mask

        # Create timestamps
        total_len = self.input_window + self.output_window
        timestamps = self._create_timestamps(batch, total_len)

        # Sample random timesteps
        t = self._sample_timesteps(B, device)

        # Add noise
        noise = torch.randn_like(full_seq_grid)
        noisy_input = self.noise_scheduler.add_noise(full_seq_grid, noise, t)

        # Apply mask: keep history, noise only future
        noisy_input = full_seq_grid * (1 - mask.unsqueeze(2)) + noisy_input * mask.unsqueeze(2)

        # Forward through model
        model_output = self.model(noisy_input, t, timestamps, mask=mask)

        return model_output

    def predict(self, batch):
        """
        Generate predictions using iterative denoising.

        Args:
            batch: Dictionary with 'X' tensor

        Returns:
            Prediction tensor of shape (B, T_out, N, output_dim)
        """
        x = batch['X']  # (B, T_in, N, F)
        B = x.shape[0]
        device = x.device

        # Prepare history
        x_traffic = x[:, :, :, :self.output_dim]

        # Reshape to grid (this will also pad to be divisible by patch_size)
        x_grid = self._reshape_to_grid(x_traffic)  # (B, T_in, F, padded_height, padded_width)

        # Initialize future with noise (use padded dimensions)
        future_shape = (B, self.output_window, self.output_dim, self.padded_height, self.padded_width)
        future_noisy = torch.randn(future_shape, device=device)

        # Concatenate history and noisy future
        full_seq = torch.cat([x_grid, future_noisy], dim=1)

        # Generate prediction mask (use stored mask_generator with padded dimensions)
        mask = self.mask_generator(B, device, idx=0)

        # Create timestamps
        total_len = self.input_window + self.output_window
        timestamps = self._create_timestamps(batch, total_len)

        # Iterative denoising
        timesteps = list(range(self.diffusion_steps - 1, -1, -self.diffusion_steps // self.num_inference_steps))

        with torch.no_grad():
            for t_idx in timesteps:
                t = torch.tensor([t_idx] * B, device=device)

                # Forward pass
                model_output = self.model(full_seq, t, timestamps, mask=mask)

                # Denoising step
                if self.precondition_outputs:
                    # Model predicts x_0 directly
                    sigma = self.noise_scheduler.sigmas.to(device)[t_idx]
                    pred_x0 = model_output * (-sigma) + full_seq
                else:
                    # Model predicts velocity
                    pred_x0 = self.noise_scheduler.step(model_output, t_idx, full_seq)

                # Update sequence: keep original history, use predicted future
                full_seq = torch.cat([
                    x_grid,  # Keep original history [B, input_window, ...]
                    pred_x0[:, self.input_window:, :, :, :]  # Use predicted future only
                ], dim=1)

        # Extract prediction (future part)
        pred_grid = full_seq[:, self.input_window:, :, :, :]

        # Reshape back to (B, T, N, F)
        pred = self._reshape_from_grid(pred_grid)

        return pred

    def calculate_loss(self, batch):
        """
        Calculate training loss using flow matching objective.

        Args:
            batch: Dictionary with 'X' and 'y' tensors

        Returns:
            Loss tensor
        """
        x = batch['X']
        y = batch['y']

        B = x.shape[0]
        device = x.device

        # Prepare input
        x_traffic = x[:, :, :, :self.output_dim]
        y_traffic = y[:, :, :, :self.output_dim]
        full_seq = torch.cat([x_traffic, y_traffic], dim=1)

        # Reshape to grid
        full_seq_grid = self._reshape_to_grid(full_seq)

        # Generate mask
        mask = self.mask_generator(B, device, idx=0)

        # Create timestamps
        total_len = self.input_window + self.output_window
        timestamps = self._create_timestamps(batch, total_len)

        # Sample noise and timesteps
        noise = torch.randn_like(full_seq_grid)
        t = self._sample_timesteps(B, device)

        # Create noisy input
        noisy_input = self.noise_scheduler.add_noise(full_seq_grid, noise, t)
        noisy_input = full_seq_grid * (1 - mask.unsqueeze(2)) + noisy_input * mask.unsqueeze(2)

        # Forward pass
        model_output = self.model(noisy_input, t, timestamps, mask=mask)

        # Compute target
        if self.precondition_outputs:
            # Target is the clean input
            target = full_seq_grid
        else:
            # Target is velocity: noise - clean
            target = noise - full_seq_grid

        # Compute loss only on masked (future) region
        loss_tensor = ((model_output - target) * mask.unsqueeze(2)) ** 2
        loss_val = loss_tensor.mean()

        return loss_val


# Model variants
def UrbanDiT_S_1(config, data_feature):
    """Small variant 1 of UrbanDiT."""
    config['depth'] = config.get('depth', 4)
    config['hidden_size'] = config.get('hidden_size', 256)
    config['patch_size'] = config.get('patch_size', 2)
    config['num_heads'] = config.get('num_heads', 4)
    return UrbanDiT(config, data_feature)


def UrbanDiT_S_2(config, data_feature):
    """Small variant 2 of UrbanDiT."""
    config['depth'] = config.get('depth', 6)
    config['hidden_size'] = config.get('hidden_size', 384)
    config['patch_size'] = config.get('patch_size', 2)
    config['num_heads'] = config.get('num_heads', 6)
    return UrbanDiT(config, data_feature)


def UrbanDiT_S_3(config, data_feature):
    """Small variant 3 of UrbanDiT."""
    config['depth'] = config.get('depth', 12)
    config['hidden_size'] = config.get('hidden_size', 384)
    config['patch_size'] = config.get('patch_size', 2)
    config['num_heads'] = config.get('num_heads', 12)
    return UrbanDiT(config, data_feature)
