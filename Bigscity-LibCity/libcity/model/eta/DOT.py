"""
DOT: Diffusion-based Origin-Destination Travel Time Estimation

This model is adapted from the original DOT implementation for the LibCity framework.
Original paper: Diffusion-based Origin-Destination Travel Time Estimation

Key Components:
1. Unet denoiser for generating Pixelated Trajectory (PiT) representations
2. TransformerPredictor for travel time estimation from PiT
3. DiffusionProcess for forward/backward diffusion sampling
4. Two-stage architecture: diffusion (PiT generation) + prediction (ETA)

The model supports three conditional modes:
- 'odt': Origin-Destination-Time conditioning (5D: o_lng, o_lat, d_lng, d_lat, time)
- 'od': Origin-Destination conditioning only (4D)
- 't': Time conditioning only (1D)

Adapted for LibCity framework by inheriting from AbstractTrafficStateModel.

Original Repository: DOT
"""

import math
from inspect import isfunction
from functools import partial

import numpy as np
import torch
from torch import nn
from torch import einsum
import torch.nn.functional as F

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

try:
    from einops import rearrange, repeat
    HAS_EINOPS = True
except ImportError:
    HAS_EINOPS = False
    print("Warning: einops library not installed. DOT requires einops. "
          "Install it with: pip install einops")


# ============================================================================
# Utility Functions
# ============================================================================

def exists(x):
    """Check whether the input exists (is not None)."""
    return x is not None


def default(val, d):
    """Return val if it exists, otherwise return d (calling it if it's a function)."""
    if exists(val):
        return val
    return d() if isfunction(d) else d


# ============================================================================
# Diffusion Schedule Functions
# ============================================================================

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    """Linear beta schedule."""
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    """Quadratic beta schedule."""
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    """Sigmoid beta schedule."""
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


# ============================================================================
# Encoding Layers
# ============================================================================

class ContinuousEncoding(nn.Module):
    """
    Trigonometric encoding for continuous values into distance-sensitive vectors.
    Used for encoding spatial/temporal continuous features.
    """
    def __init__(self, embed_size):
        super().__init__()
        self.omega = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size))).float(),
            requires_grad=True
        )
        self.bias = nn.Parameter(torch.zeros(embed_size).float(), requires_grad=True)
        self.div_term = math.sqrt(1. / embed_size)

    def forward(self, x):
        """
        Args:
            x: input sequence for encoding, shape (batch_size) or (batch_size, seq_len)
        Returns:
            encoded sequence, shape (batch_size, embed_size) or (batch_size, seq_len, embed_size)
        """
        if x.dim() == 1:
            encode = x.unsqueeze(-1) * self.omega.reshape(1, -1) + self.bias.reshape(1, -1)
        else:
            encode = x.unsqueeze(-1) * self.omega.reshape(1, 1, -1) + self.bias.reshape(1, 1, -1)
        encode = torch.cos(encode)
        return self.div_term * encode


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal-based function for encoding diffusion timestamps.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for sequence positions.
    """
    def __init__(self, embed_size, max_len):
        super().__init__()
        pe = torch.zeros(max_len, embed_size).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


# ============================================================================
# U-Net Building Blocks
# ============================================================================

class Residual(nn.Module):
    """Adds the input to the output of a function (residual connection)."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    """2D transposed convolution for upsampling."""
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    """2D convolution for downsampling."""
    return nn.Conv2d(dim, dim, 4, 2, 1)


class Block(nn.Module):
    """Basic building block of U-Net with GroupNorm and SiLU activation."""
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """
    ResNet block with optional time embedding conditioning.
    https://arxiv.org/abs/1512.03385
    """
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)


class ConvNextBlock(nn.Module):
    """
    ConvNeXT block for the U-Net architecture.
    https://arxiv.org/abs/2201.03545
    """
    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)


class PreNorm(nn.Module):
    """Apply GroupNorm before a function (typically attention)."""
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Attention(nn.Module):
    """Multi-head self-attention for 2D feature maps."""
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    """
    Linear attention with O(n) complexity.
    More efficient than standard attention for long sequences.
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class TimeEmbed(nn.Module):
    """Time embedding layer using sinusoidal encoding followed by MLP."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, time):
        return self.time_mlp(time)


# ============================================================================
# U-Net Denoiser
# ============================================================================

class Unet(nn.Module):
    """
    U-Net denoiser for diffusion-based PiT generation.

    Architecture:
    - Initial convolution on noisy images
    - Position embeddings for noise levels
    - Downsampling stages: 2 blocks + attention + downsample
    - Middle: ResNet/ConvNeXT blocks with attention
    - Upsampling stages: 2 blocks + attention + upsample
    - Final convolution

    Args:
        dim: Base dimension for the U-Net
        init_dim: Initial dimension after first conv (default: dim // 3 * 2)
        out_dim: Output dimension (default: same as input channels)
        dim_mults: Tuple of dimension multipliers for each resolution
        channels: Number of input/output channels
        with_time_emb: Whether to use time embeddings
        resnet_block_groups: Number of groups for GroupNorm in ResNet blocks
        use_convnext: Whether to use ConvNeXT blocks instead of ResNet
        convnext_mult: Multiplier for ConvNeXT hidden dimension
        condition: Conditioning mode ('odt', 'od', 't', or None)
    """
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            with_time_emb=True,
            resnet_block_groups=8,
            use_convnext=True,
            convnext_mult=2,
            condition='odt',
    ):
        super().__init__()

        if not HAS_EINOPS:
            raise ImportError("DOT requires the einops library. "
                            "Install it with: pip install einops")

        self.channels = channels
        self.condition = condition

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # Time embeddings and conditioning
        self.y_linear = None
        if with_time_emb:
            time_dim = dim * 4
            self.time_embed = TimeEmbed(dim, time_dim)

            if condition == 'odt':
                y_dim = 5  # o_lng, o_lat, d_lng, d_lat, time
            elif condition == 'od':
                y_dim = 4  # o_lng, o_lat, d_lng, d_lat
            elif condition == 't':
                y_dim = 1  # time only
            else:
                y_dim = None

            if y_dim is not None:
                self.y_linear = nn.Linear(y_dim, time_dim)
        else:
            time_dim = None
            self.time_embed = None

        # Downsampling layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList([
                    block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                    block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Downsample(dim_out) if not is_last else nn.Identity(),
                ])
            )

        # Middle layers
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Upsampling layers
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList([
                    block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Upsample(dim_in) if not is_last else nn.Identity(),
                ])
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

        self.name = 'unet'
        self.num_layers = len(dim_mults)

    def forward(self, x, time=None, y=None):
        """
        Forward pass of U-Net denoiser.

        Args:
            x: Noisy input images, shape (B, C, H, W)
            time: Diffusion timesteps, shape (B,)
            y: Conditioning information (OD-time), shape (B, y_dim)

        Returns:
            Predicted noise, shape (B, C, H, W)
        """
        x = self.init_conv(x)

        t = self.time_embed(time) if exists(self.time_embed) else None
        if y is not None and self.y_linear is not None:
            if self.condition == 'od':
                y = y[:, :4]
            if self.condition == 't':
                y = y[:, -1:]
            y_latent = self.y_linear(y)
            t = y_latent + t

        h = []

        # Downsample path
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Upsample path
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


# ============================================================================
# Diffusion Process
# ============================================================================

class DiffusionProcess:
    """
    Utilities for the diffusion process (forward and backward sampling).

    Args:
        T: Maximum timesteps for diffusion
        schedule_name: Beta schedule type ('cosine', 'linear', 'quadratic', 'sigmoid')
    """
    def __init__(self, T, schedule_name='linear'):
        if schedule_name == 'cosine':
            self.schedule_func = cosine_beta_schedule
        elif schedule_name == 'quadratic':
            self.schedule_func = quadratic_beta_schedule
        elif schedule_name == 'sigmoid':
            self.schedule_func = sigmoid_beta_schedule
        elif schedule_name == 'linear':
            self.schedule_func = linear_beta_schedule
        else:
            raise NotImplementedError(f'Diffusion schedule {schedule_name} not implemented.')

        self.T = T

        # Define beta schedule
        self.betas = self.schedule_func(T)

        # Define alphas
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def extract(self, a, t, x_shape):
        """Extract values from a at indices t and reshape for broadcasting."""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: calculate noisy data q(x_t|x_0).

        Args:
            x_start: Original data x_0
            t: Diffusion timesteps
            noise: Optional pre-generated noise

        Returns:
            Noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, t, y=None, noise=None, loss_type="l1"):
        """
        Calculate backward diffusion loss.

        Args:
            denoise_model: U-Net denoiser model
            x_start: Original data x_0
            t: Diffusion timesteps
            y: Conditioning information
            noise: Optional pre-generated noise
            loss_type: Loss type ('l1', 'l2', 'huber')

        Returns:
            Diffusion loss
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t, y)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError(f'Loss type {loss_type} not implemented.')

        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, y=None):
        """
        Single step of backward diffusion p(x_{t-1}|x_t).

        Args:
            model: Denoiser model
            x: Noisy data at timestep t
            t: Current timestep tensor
            t_index: Current timestep index
            y: Conditioning information

        Returns:
            Denoised data at timestep t-1
        """
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t, y) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, y=None):
        """
        Full backward diffusion process from x_T to x_0.

        Args:
            model: Denoiser model
            shape: Output shape (B, C, H, W)
            y: Conditioning information

        Returns:
            Generated samples at x_0
        """
        device = next(model.parameters()).device

        b = shape[0]
        # Start from pure noise
        img = torch.randn(shape, device=device)

        for i in reversed(range(0, self.T)):
            img = self.p_sample(
                model, img,
                torch.full((b,), i, device=device, dtype=torch.long),
                i, y=y
            )

        return img


# ============================================================================
# Transformer Predictor
# ============================================================================

class TransformerPredictor(nn.Module):
    """
    Transformer-based ETA predictor from PiT representations.

    Args:
        input_dim: Number of input channels in PiT
        d_model: Transformer hidden dimension
        num_head: Number of attention heads
        num_layers: Number of transformer encoder layers
        num_grid: Total number of grid cells (split * split)
        dropout: Dropout rate
        use_grid: Whether to use grid embeddings
        use_st: Whether to use spatio-temporal features
    """
    def __init__(self, input_dim, d_model, num_head, num_layers, num_grid, dropout,
                 use_grid=True, use_st=True):
        super().__init__()

        if not HAS_EINOPS:
            raise ImportError("DOT requires the einops library. "
                            "Install it with: pip install einops")

        self.num_grid = num_grid
        self.use_grid = use_grid
        self.use_st = use_st

        trans_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_head,
            dim_feedforward=d_model, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(trans_layer, num_layers=num_layers)

        self.input_linear = nn.Linear(input_dim - 1, d_model)
        self.grid_embed = nn.Embedding(num_grid, d_model)
        self.pos_encode = PositionalEncoding(d_model, num_grid)
        self.out_linear = nn.Linear(d_model, 1)

        self.num_layers = num_layers
        self.d_model = d_model

        self.name = 'trans'

    def forward(self, x, odt=None):
        """
        Forward pass for ETA prediction.

        Args:
            x: PiT representation, shape (B, C, H, W) or (B, C, S) for flat mode
            odt: OD-time conditioning (not directly used in predictor)

        Returns:
            Predicted travel time, shape (B,)
        """
        if len(x.shape) > 3:
            x = rearrange(x, 'b c h w -> b (h w) c')  # (B, num_grid, C)
        else:
            x = rearrange(x, 'b c s -> b s c')

        # Create key padding mask (negative values indicate padding)
        mask = x[:, :, 0] < 0  # (B, num_grid)
        pos = self.pos_encode(x)  # (1, num_grid, d_model)

        x = self.input_linear(x[:, :, 1:])  # (B, num_grid, d_model)
        grid = torch.arange(0, x.size(1)).long().to(x.device)
        grid = repeat(grid, 'g -> b g', b=x.size(0))  # (B, num_grid)
        grid = self.grid_embed(grid)  # (B, num_grid, d_model)

        if self.use_st:
            pos = pos + x
        if self.use_grid:
            pos = pos + grid

        out = self.transformer(pos, src_key_padding_mask=mask)  # (B, num_grid, d_model)
        out = self.out_linear(out).mean(1).squeeze(-1)  # (B,)
        out = torch.nan_to_num(out, nan=0.0)
        return out


# ============================================================================
# DOT Model (LibCity Adapted)
# ============================================================================

class DOT(AbstractTrafficStateModel):
    """
    DOT: Diffusion-based Origin-Destination Travel Time Estimation.

    This model uses a two-stage architecture:
    1. Diffusion stage: Generate Pixelated Trajectory (PiT) from OD-time conditions
    2. Prediction stage: Estimate travel time from generated PiT

    The model can be trained in two modes:
    - Joint training: Both diffusion and prediction losses
    - Prediction only: Using pre-generated or real PiT images

    Args:
        config: Configuration dictionary containing model hyperparameters
        data_feature: Dictionary containing data-specific features
    """

    def __init__(self, config, data_feature):
        super(DOT, self).__init__(config, data_feature)

        if not HAS_EINOPS:
            raise ImportError("DOT requires the einops library. "
                            "Install it with: pip install einops")

        self.config = config
        self.data_feature = data_feature
        self.device = config.get('device', torch.device('cpu'))

        # Grid/Split configuration
        self.split = config.get('split', 20)
        self.num_grid = self.split * self.split

        # Number of channels in PiT (mask, daytime, offset)
        self.num_channel = config.get('num_channel', 3)

        # Diffusion configuration
        self.timesteps = config.get('timesteps', 1000)
        self.schedule_name = config.get('schedule_name', 'linear')
        self.diffusion_loss_type = config.get('diffusion_loss_type', 'huber')
        self.condition = config.get('condition', 'odt')  # 'odt', 'od', or 't'

        # U-Net denoiser configuration
        self.unet_dim = config.get('unet_dim', self.split)
        self.unet_init_dim = config.get('unet_init_dim', 4)
        self.unet_dim_mults = tuple(config.get('unet_dim_mults', [1, 2, 4]))
        self.use_convnext = config.get('use_convnext', True)
        self.convnext_mult = config.get('convnext_mult', 2)

        # Predictor configuration
        self.predictor_type = config.get('predictor_type', 'trans')  # 'trans' or 'unet'
        self.d_model = config.get('d_model', 128)
        self.num_head = config.get('num_head', 8)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        self.use_st = config.get('use_st', True)
        self.use_grid = config.get('use_grid', False)

        # Training configuration
        self.alpha = config.get('alpha', 0.5)  # Weight for diffusion vs prediction loss
        self.train_diffusion = config.get('train_diffusion', True)
        self.train_prediction = config.get('train_prediction', True)
        self.use_generated_pit = config.get('use_generated_pit', False)

        # Flat mode for sequence representation
        self.flat = config.get('flat', False)

        # Time normalization (in minutes)
        self.time_mean = data_feature.get('time_mean', 0.0)
        self.time_std = data_feature.get('time_std', 1.0)

        # Build diffusion process
        self.diffusion = DiffusionProcess(T=self.timesteps, schedule_name=self.schedule_name)

        # Build U-Net denoiser
        self.denoiser = Unet(
            dim=self.unet_dim,
            channels=self.num_channel,
            init_dim=self.unet_init_dim,
            dim_mults=self.unet_dim_mults,
            condition=self.condition,
            use_convnext=self.use_convnext,
            convnext_mult=self.convnext_mult,
        )

        # Build predictor
        if self.predictor_type == 'trans':
            self.predictor = TransformerPredictor(
                input_dim=self.num_channel,
                d_model=self.d_model,
                num_head=self.num_head,
                num_layers=self.num_layers,
                num_grid=self.num_grid,
                dropout=self.dropout,
                use_grid=self.use_grid,
                use_st=self.use_st,
            )
        else:
            raise NotImplementedError(f'Predictor type {self.predictor_type} not implemented.')

        # Loss function for prediction
        self.loss_func = nn.MSELoss()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for linear layers."""
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name and param.dim() >= 2 and 'norm' not in name.lower():
                nn.init.xavier_uniform_(param)

    def _normalize(self, data):
        """Normalize travel time data."""
        return (data - self.time_mean) / (self.time_std + 1e-8)

    def _inverse_normalize(self, data):
        """Inverse normalize travel time data."""
        return data * self.time_std + self.time_mean

    def _stack_batch_tensor(self, batch, key, fallback_key=None):
        """
        Helper to extract tensor from BatchPAD batch.

        BatchPAD with 'no_pad_float' type returns a list of tensors that need stacking.
        This helper handles both list and tensor cases.

        Args:
            batch: BatchPAD batch dictionary
            key: Primary key to look up
            fallback_key: Optional fallback key if primary is not found

        Returns:
            Stacked tensor or None if key not found
        """
        data = None
        if key in batch.data:
            data = batch[key]
        elif fallback_key is not None and fallback_key in batch.data:
            data = batch[fallback_key]

        if data is None:
            return None

        # If it's a list (from no_pad_float type), stack into batch tensor
        if isinstance(data, list):
            return torch.stack(data, dim=0)
        return data

    def forward(self, batch):
        """
        Forward pass of DOT model.

        During training:
            - If train_diffusion: compute diffusion loss on real PiT images
            - If train_prediction: compute prediction from real or generated PiT

        During inference:
            - Generate PiT from OD-time conditions using diffusion
            - Predict travel time from generated PiT

        Args:
            batch: Dictionary containing:
                - 'images': PiT images, shape (B, C, H, W) or (B, C, S) for flat
                - 'odt': OD-time features, shape (B, 5) [o_lng, o_lat, d_lng, d_lat, time]
                - 'time' or 'arrive_time': Ground truth travel time (for training)

        Returns:
            If training: (prediction, diffusion_loss)
            If inference: prediction
        """
        # Get conditioning information - stack list of tensors into batch tensor
        odt = self._stack_batch_tensor(batch, 'odt')
        if odt is None:
            # Try to construct from individual features
            o_lng = self._stack_batch_tensor(batch, 'o_lng', 'origin_lng')
            o_lat = self._stack_batch_tensor(batch, 'o_lat', 'origin_lat')
            d_lng = self._stack_batch_tensor(batch, 'd_lng', 'dest_lng')
            d_lat = self._stack_batch_tensor(batch, 'd_lat', 'dest_lat')
            depart_time = self._stack_batch_tensor(batch, 'depart_time', 'daytime')

            if all(x is not None for x in [o_lng, o_lat, d_lng, d_lat, depart_time]):
                odt = torch.stack([o_lng, o_lat, d_lng, d_lat, depart_time], dim=-1).float()

        if self.training:
            # Get real PiT images for training - stack list of tensors
            images = self._stack_batch_tensor(batch, 'images', 'X')
            if images is None:
                raise ValueError("Training requires 'images' or 'X' in batch")

            diffusion_loss = None

            # Compute diffusion loss if enabled
            if self.train_diffusion:
                t = torch.randint(0, self.timesteps, (images.size(0),), device=images.device).long()
                diffusion_loss = self.diffusion.p_losses(
                    self.denoiser, images, t, y=odt, loss_type=self.diffusion_loss_type
                )

            # Compute prediction
            if self.train_prediction:
                if self.use_generated_pit:
                    # Generate PiT from conditions
                    with torch.no_grad():
                        generated_images = self.diffusion.p_sample_loop(
                            self.denoiser, shape=images.shape, y=odt
                        )
                    prediction = self.predictor(generated_images, odt)
                else:
                    # Use real PiT images
                    prediction = self.predictor(images, odt)
            else:
                prediction = None

            return prediction, diffusion_loss

        else:
            # Inference mode: generate PiT and predict - stack list of tensors
            images = self._stack_batch_tensor(batch, 'images', 'X')

            if images is not None:
                # Use provided images
                pit = images
            else:
                # Generate PiT from conditions
                batch_size = odt.size(0) if odt is not None else 1
                shape = (batch_size, self.num_channel, self.split, self.split)
                pit = self.diffusion.p_sample_loop(self.denoiser, shape=shape, y=odt)

            prediction = self.predictor(pit, odt)
            return prediction

    def predict(self, batch):
        """
        Predict travel times for a batch.

        Args:
            batch: Input batch dictionary

        Returns:
            Predicted travel times, shape (B,)
        """
        self.eval()
        with torch.no_grad():
            if self.training:
                prediction, _ = self.forward(batch)
            else:
                prediction = self.forward(batch)

            if prediction is not None:
                # Inverse normalize if normalization was applied
                prediction = self._inverse_normalize(prediction)

            # Ensure output has shape (B, 1) to match LibCity ETA evaluation format
            if prediction is not None and prediction.dim() == 1:
                prediction = prediction.unsqueeze(-1)
            return prediction

    def calculate_loss(self, batch):
        """
        Calculate the combined training loss.
        Forces training mode temporarily to get both diffusion and prediction losses.

        Total loss = alpha * diffusion_loss + (1 - alpha) * prediction_loss

        Args:
            batch: Input batch dictionary with 'time' or 'arrive_time' as ground truth

        Returns:
            Combined loss tensor
        """
        # Save current mode and temporarily set to training mode
        was_training = self.training
        self.train()

        prediction, diffusion_loss = self.forward(batch)

        # Restore original mode
        if not was_training:
            self.eval()

        # Get ground truth travel time - stack list of tensors
        truth = self._stack_batch_tensor(batch, 'time')
        if truth is None:
            truth = self._stack_batch_tensor(batch, 'arrive_time')
        if truth is None:
            truth = self._stack_batch_tensor(batch, 'y')
        if truth is None:
            raise ValueError("Training requires 'time', 'arrive_time', or 'y' in batch")

        # Ensure truth is 1D
        if truth.dim() > 1:
            truth = truth.squeeze()

        total_loss = 0.0

        # Diffusion loss
        if diffusion_loss is not None and self.train_diffusion:
            total_loss = total_loss + self.alpha * diffusion_loss

        # Prediction loss
        if prediction is not None and self.train_prediction:
            # Inverse normalize prediction for loss computation
            prediction = self._inverse_normalize(prediction)
            prediction_loss = self.loss_func(prediction.squeeze(), truth.float())
            total_loss = total_loss + (1 - self.alpha) * prediction_loss

        return total_loss

    def generate_pit(self, batch):
        """
        Generate PiT images from OD-time conditions.

        Args:
            batch: Dictionary with conditioning information

        Returns:
            Generated PiT images, shape (B, C, H, W)
        """
        self.eval()
        with torch.no_grad():
            odt = self._stack_batch_tensor(batch, 'odt')
            if odt is None:
                o_lng = self._stack_batch_tensor(batch, 'o_lng', 'origin_lng')
                o_lat = self._stack_batch_tensor(batch, 'o_lat', 'origin_lat')
                d_lng = self._stack_batch_tensor(batch, 'd_lng', 'dest_lng')
                d_lat = self._stack_batch_tensor(batch, 'd_lat', 'dest_lat')
                depart_time = self._stack_batch_tensor(batch, 'depart_time', 'daytime')

                if all(x is not None for x in [o_lng, o_lat, d_lng, d_lat, depart_time]):
                    odt = torch.stack([o_lng, o_lat, d_lng, d_lat, depart_time], dim=-1).float()

            batch_size = odt.size(0) if odt is not None else 1
            shape = (batch_size, self.num_channel, self.split, self.split)

            return self.diffusion.p_sample_loop(self.denoiser, shape=shape, y=odt)
