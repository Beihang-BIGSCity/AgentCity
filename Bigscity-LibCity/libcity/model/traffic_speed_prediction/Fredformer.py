"""
Fredformer Model for LibCity Traffic Speed Prediction

Adapted from: https://github.com/chenzRG/Fredformer
Paper: "Fredformer: Frequency Debiased Transformer for Time Series Forecasting"

Original Files:
- models/Fredformer.py
- layers/Fredformer_backbone.py
- layers/cross_Transformer.py
- layers/cross_Transformer_nys.py
- layers/RevIN.py

Key Adaptations:
1. Inherits from AbstractTrafficStateModel
2. Reshapes LibCity's [batch, seq_len, num_nodes, features] to [batch, seq_len, num_nodes * features]
3. Implements predict() and calculate_loss() methods
4. Uses LibCity's config and data_feature pattern
5. All layers consolidated into single file

Hyperparameters (from config):
- input_window: Input sequence length (default: 12)
- output_window: Prediction length (default: 12)
- cf_dim: Transformer feature dimension (default: 48)
- cf_depth: Number of transformer layers (default: 2)
- cf_heads: Number of attention heads (default: 6)
- cf_mlp: MLP dimension in transformer (default: 256)
- cf_head_dim: Attention head dimension (default: 16)
- cf_drop: Dropout in transformer (default: 0.3)
- d_model: Model dimension (default: 128)
- patch_len: Frequency patch length (default: 16)
- stride: Patching stride (default: 8)
- use_nys: Use Nystrom approximation (default: 0)
- revin: Use RevIN normalization (default: 1)
- affine: RevIN affine transformation (default: 1)
- subtract_last: RevIN subtract last (default: 0)
- individual: Individual heads per channel (default: 0)
- head_dropout: Head dropout rate (default: 0.0)
- ablation: Ablation mode (default: 0)
- mlp_hidden: MLP hidden dimension (default: 256)
- mlp_drop: MLP dropout (default: 0.1)
"""

from logging import getLogger
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, einsum
from einops import rearrange, repeat, reduce
from math import ceil

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


# ====================== RevIN Layer ======================
class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time series
    From: https://github.com/ts-kim/RevIN
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


# ====================== Cross Transformer Components ======================
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class c_Attention(nn.Module):
    """Channel-wise attention for standard transformer"""
    def __init__(self, dim, heads, dim_head, dropout=0.8):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.d_k = math.sqrt(self.dim_head)
        inner_dim = dim_head * heads
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) / self.d_k

        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), attn


class c_Transformer(nn.Module):
    """Standard channel-wise transformer"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.8):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, c_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x_n, attn_weights = attn(x)
            x = x_n + x
            x = ff(x) + x
        return x, attn_weights


class Trans_C(nn.Module):
    """Transformer for Cross-frequency processing"""
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head, dropout, patch_dim, horizon, d_model):
        super().__init__()

        self.dim = dim
        self.patch_dim = patch_dim
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.transformer = c_Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Linear(dim, d_model)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        x, attn = self.transformer(x)
        x = self.dropout(x)
        x = self.mlp_head(x).squeeze()
        return x


# ====================== Nystrom Attention Components ======================
def exists(val):
    return val is not None


def moore_penrose_iter_pinv(x, iters=6):
    """Moore-Penrose iterative pseudo-inverse"""
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device=device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z


class NystromAttention(nn.Module):
    """Nystrom-based efficient attention"""
    def __init__(self, dim, heads, dim_head, num_landmarks, pinv_iterations, eps, dropout=0.5):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.eps = eps
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value=0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value=False)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        q = q * self.scale

        # Generate landmarks by sum reduction
        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l=l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l=l)

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l=l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        q_landmarks /= divisor
        k_landmarks /= divisor

        # Compute similarities
        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out


class Nystromformer(nn.Module):
    """Nystrom-based efficient transformer"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_landmarks, pinv_iterations, eps, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, NystromAttention(dim, heads=heads, dim_head=dim_head, num_landmarks=num_landmarks,
                                              pinv_iterations=pinv_iterations, eps=eps, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x


class Trans_C_nys(nn.Module):
    """Transformer with Nystrom attention for Cross-frequency processing"""
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head, dropout, patch_dim, horizon, d_model):
        super().__init__()

        self.dim = dim
        self.patch_dim = patch_dim
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.transformer = Nystromformer(dim, depth, heads, dim_head, mlp_dim,
                                          num_landmarks=5, pinv_iterations=6, eps=1e-8, dropout=dropout)

        self.mlp_head = nn.Linear(dim, d_model)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        x = self.transformer(x)
        x = self.mlp_head(x).squeeze()
        return x


# ====================== Flatten Head ======================
class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears1 = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears1.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear1 = nn.Linear(nf, nf)
            self.linear2 = nn.Linear(nf, nf)
            self.linear3 = nn.Linear(nf, nf)
            self.linear4 = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears1[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = F.relu(self.linear1(x)) + x
            x = F.relu(self.linear2(x)) + x
            x = F.relu(self.linear3(x)) + x
            x = self.linear4(x)
        return x


# ====================== Fredformer Backbone ======================
class Fredformer_backbone(nn.Module):
    """
    Fredformer backbone with FFT-based frequency domain processing
    """
    def __init__(self, ablation: int, mlp_drop: float, use_nys: int, output: int, mlp_hidden: int,
                 cf_dim: int, cf_depth: int, cf_heads: int, cf_mlp: int, cf_head_dim: int, cf_drop: float,
                 c_in: int, context_window: int, target_window: int, patch_len: int, stride: int, d_model: int,
                 head_dropout=0, padding_patch=None, individual=False, revin=True, affine=True,
                 subtract_last=False, **kwargs):

        super().__init__()
        self.use_nys = use_nys
        self.ablation = ablation

        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        self.output = output

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.targetwindow = target_window
        self.horizon = self.targetwindow
        patch_num = int((context_window - patch_len) / stride + 1)
        self.norm = nn.LayerNorm(patch_len)

        # Backbone - Frequency Transformer
        self.re_attn = True
        if self.use_nys == 0:
            self.fre_transformer = Trans_C(dim=cf_dim, depth=cf_depth, heads=cf_heads, mlp_dim=cf_mlp,
                                            dim_head=cf_head_dim, dropout=cf_drop, patch_dim=patch_len * 2,
                                            horizon=self.horizon * 2, d_model=d_model * 2)
        else:
            self.fre_transformer = Trans_C_nys(dim=cf_dim, depth=cf_depth, heads=cf_heads, mlp_dim=cf_mlp,
                                                dim_head=cf_head_dim, dropout=cf_drop, patch_dim=patch_len * 2,
                                                horizon=self.horizon * 2, d_model=d_model * 2)

        # Head
        self.head_nf_f = d_model * 2 * patch_num
        self.n_vars = c_in
        self.individual = individual
        self.head_f1 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, target_window,
                                     head_dropout=head_dropout)
        self.head_f2 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, target_window,
                                     head_dropout=head_dropout)

        self.ircom = nn.Linear(self.targetwindow * 2, self.targetwindow)
        self.rfftlayer = nn.Linear(self.targetwindow * 2 - 2, self.targetwindow)
        self.final = nn.Linear(self.targetwindow * 2, self.targetwindow)

        # Break up R&I
        self.get_r = nn.Linear(d_model * 2, d_model * 2)
        self.get_i = nn.Linear(d_model * 2, d_model * 2)
        self.output1 = nn.Linear(target_window, target_window)

        # Ablation
        self.input = nn.Linear(c_in, patch_len * 2)
        self.outpt = nn.Linear(d_model * 2, c_in)
        self.abfinal = nn.Linear(patch_len * patch_num, target_window)

    def forward(self, z):  # z: [bs x nvars x seq_len]
        # RevIN normalization
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        # FFT to frequency domain
        z = torch.fft.fft(z)
        z1 = z.real
        z2 = z.imag

        # Patching in frequency domain
        z1 = z1.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [bs x nvars x patch_num x patch_len]
        z2 = z2.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Channel-wise processing
        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)

        # Model shape
        batch_size = z1.shape[0]
        patch_num = z1.shape[1]
        c_in = z1.shape[2]
        patch_len = z1.shape[3]

        # Reshape for transformer
        z1 = torch.reshape(z1, (batch_size * patch_num, c_in, z1.shape[-1]))  # [bs * patch_num, nvars, patch_len]
        z2 = torch.reshape(z2, (batch_size * patch_num, c_in, z2.shape[-1]))

        # Process through frequency transformer
        z = self.fre_transformer(torch.cat((z1, z2), -1))
        z1 = self.get_r(z)
        z2 = self.get_i(z)

        # Reshape back
        z1 = torch.reshape(z1, (batch_size, patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size, patch_num, c_in, z2.shape[-1]))

        z1 = z1.permute(0, 2, 1, 3)  # [bs, nvars, patch_num, horizon]
        z2 = z2.permute(0, 2, 1, 3)

        # Head processing
        z1 = self.head_f1(z1)  # [bs x nvars x target_window]
        z2 = self.head_f2(z2)

        # IFFT back to time domain
        z = torch.fft.ifft(torch.complex(z1, z2))
        zr = z.real
        zi = z.imag
        z = self.ircom(torch.cat((zr, zi), -1))

        # RevIN denormalization
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)

        return z


# ====================== Main Fredformer Model for LibCity ======================
class Fredformer(AbstractTrafficStateModel):
    """
    Fredformer model adapted for LibCity traffic speed prediction.

    Data format transformation:
    - LibCity input: [batch, seq_len, num_nodes, features]
    - Fredformer expects: [batch, seq_len, channels] where channels = num_nodes * features
    - Output: [batch, pred_len, num_nodes, output_dim]

    Key Features:
    - FFT-based frequency domain processing
    - Patching mechanism in frequency domain
    - Cross-frequency transformer attention
    - Optional Nystrom approximation for efficient attention
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))

        # Data features
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self._scaler = self.data_feature.get('scaler')

        # Model parameters from config
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.input_dim = self.data_feature.get('feature_dim', config.get('input_dim', 1))
        self.output_dim = config.get('output_dim', 1)

        # Fredformer specific parameters
        self.cf_dim = config.get('cf_dim', 48)
        self.cf_depth = config.get('cf_depth', 2)
        self.cf_heads = config.get('cf_heads', 6)
        self.cf_mlp = config.get('cf_mlp', 256)
        self.cf_head_dim = config.get('cf_head_dim', 16)
        self.cf_drop = config.get('cf_drop', 0.3)

        self.d_model = config.get('d_model', 128)
        self.patch_len = config.get('patch_len', 16)
        self.stride = config.get('stride', 8)
        self.dropout = config.get('dropout', 0.2)
        self.head_dropout = config.get('head_dropout', 0.0)

        # RevIN parameters
        self.revin = config.get('revin', 1)
        self.affine = config.get('affine', 1)
        self.subtract_last = config.get('subtract_last', 0)

        # Other parameters
        self.use_nys = config.get('use_nys', 0)
        self.individual = config.get('individual', 0)
        self.ablation = config.get('ablation', 0)
        self.mlp_hidden = config.get('mlp_hidden', 256)
        self.mlp_drop = config.get('mlp_drop', 0.1)

        # Calculate total channels (num_nodes * input_dim for traffic data)
        self.c_in = self.num_nodes * self.input_dim

        # Build model
        self._build_model()

        self._logger.info(f"Fredformer initialized with: num_nodes={self.num_nodes}, "
                          f"input_window={self.input_window}, output_window={self.output_window}, "
                          f"c_in={self.c_in}, d_model={self.d_model}, cf_dim={self.cf_dim}, "
                          f"cf_depth={self.cf_depth}, cf_heads={self.cf_heads}, "
                          f"patch_len={self.patch_len}, stride={self.stride}, use_nys={self.use_nys}")

    def _build_model(self):
        """Build the Fredformer model components"""
        self.model = Fredformer_backbone(
            ablation=self.ablation,
            mlp_drop=self.mlp_drop,
            use_nys=self.use_nys,
            output=0,
            mlp_hidden=self.mlp_hidden,
            cf_dim=self.cf_dim,
            cf_depth=self.cf_depth,
            cf_heads=self.cf_heads,
            cf_mlp=self.cf_mlp,
            cf_head_dim=self.cf_head_dim,
            cf_drop=self.cf_drop,
            c_in=self.c_in,
            context_window=self.input_window,
            target_window=self.output_window,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            head_dropout=self.head_dropout,
            padding_patch=None,
            individual=self.individual,
            revin=self.revin,
            affine=self.affine,
            subtract_last=self.subtract_last
        )

    def forward(self, batch):
        """
        Forward pass

        Args:
            batch: dict with 'X' key containing input tensor
                   X shape: [batch, input_window, num_nodes, input_dim]

        Returns:
            output: [batch, output_window, num_nodes, output_dim]
        """
        x = batch['X']  # [batch, input_window, num_nodes, input_dim]
        batch_size = x.shape[0]

        # Extract only the feature dimensions needed (first input_dim features)
        x = x[..., :self.input_dim]  # [batch, input_window, num_nodes, input_dim]

        # Reshape from [batch, seq_len, num_nodes, features] to [batch, seq_len, num_nodes * features]
        x = x.reshape(batch_size, self.input_window, -1)  # [batch, input_window, num_nodes * input_dim]

        # Fredformer expects [batch, channel, seq_len]
        x = x.permute(0, 2, 1)  # [batch, num_nodes * input_dim, input_window]

        # Apply model
        x = self.model(x)  # [batch, num_nodes * input_dim, output_window]

        # Permute back
        x = x.permute(0, 2, 1)  # [batch, output_window, num_nodes * input_dim]

        # Reshape back from [batch, output_window, num_nodes * input_dim] to [batch, output_window, num_nodes, input_dim]
        x = x.reshape(batch_size, self.output_window, self.num_nodes, self.input_dim)

        # Select only output_dim features
        x = x[..., :self.output_dim]

        return x  # [batch, output_window, num_nodes, output_dim]

    def predict(self, batch):
        """
        Predict for a batch

        Args:
            batch: dict with 'X' key

        Returns:
            predictions: [batch, output_window, num_nodes, output_dim]
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate training loss

        Args:
            batch: dict with 'X' and 'y' keys

        Returns:
            loss: scalar tensor
        """
        y_true = batch['y']  # [batch, output_window, num_nodes, features]
        y_predicted = self.predict(batch)

        # Apply inverse scaler for proper loss calculation
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        return loss.masked_mae_torch(y_predicted, y_true, null_val=0.0)
