"""
LightPath: Masked Autoencoder with Vision Transformer for Trajectory Representation Learning

This model is adapted from the original LightPath implementation.
Original paper: LightPath: Lightweight and Scalable Path Representation Learning

Key Components:
1. Masked Autoencoder (MAE) architecture for trajectory representation
2. Pre-trained node2vec embeddings for road segment representation
3. Pre-trained time2vec embeddings for temporal representation
4. Vision Transformer blocks for encoding/decoding
5. Dual objectives: Reconstruction + Relational Reasoning
6. ETA prediction head for downstream travel time estimation

Adapted for LibCity framework by inheriting from AbstractTrafficStateModel.
"""

import os
import math
import pickle
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

try:
    from timm.models.vision_transformer import Block
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("Warning: timm library not installed. LightPath requires timm>=0.3.2.")


# ============================================================================
# Weight Initialization Utilities
# ============================================================================

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """
    Truncated normal initialization.
    Cut & paste from PyTorch official master until it's in a few official releases.
    Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """
    Fills the input Tensor with values drawn from a truncated normal distribution.
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ============================================================================
# Patch Embedding Components
# ============================================================================

class PatchEmbed(nn.Module):
    """
    Patch Embedding for trajectory sequences.

    Uses pre-trained node2vec embeddings for road segment representation
    and pre-trained time2vec embeddings for temporal representation.
    """

    def __init__(self, node2vec, time2vec, embed_dim=128, dropout=0.1):
        """
        Args:
            node2vec: Pre-trained node2vec embeddings tensor
            time2vec: Pre-trained time2vec embeddings tensor
            embed_dim: Embedding dimension (unused, kept for compatibility)
            dropout: Dropout rate (unused in current implementation)
        """
        super().__init__()
        self.token = nn.Embedding.from_pretrained(node2vec)
        self.time = nn.Embedding.from_pretrained(time2vec)

    def forward(self, seq, ts):
        """
        Args:
            seq: Road segment indices [batch, seq_len]
            ts: Time indices [batch, seq_len]

        Returns:
            x: Road segment embeddings [batch, seq_len, embed_dim]
            ts_: Time embeddings [batch, seq_len, embed_dim]
        """
        x = self.token(seq)
        ts_ = self.time(ts)
        return x, ts_


class PositionEmbed(nn.Module):
    """
    Sinusoidal Position Embedding.
    """
    def __init__(self, num_patches=100, d_model=128, num_tokens=0):
        super().__init__()

        self.num_tokens = num_tokens
        assert self.num_tokens >= 0, "num_tokens must be >= 0"

        pe = torch.zeros(num_patches + self.num_tokens, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, num_patches + self.num_tokens).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def __call__(self):
        return self.pe


class LearnablePatchEmbed(nn.Module):
    """
    Learnable Patch Embedding for cases where pre-trained embeddings are not available.
    Uses standard learnable embeddings instead of pre-trained ones.
    """

    def __init__(self, vocab_size, time_size, embed_dim=128, dropout=0.1):
        """
        Args:
            vocab_size: Size of road segment vocabulary
            time_size: Size of time vocabulary
            embed_dim: Embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.token = nn.Embedding(vocab_size, embed_dim)
        self.time = nn.Embedding(time_size, embed_dim)

    def forward(self, seq, ts):
        x = self.token(seq)
        ts_ = self.time(ts)
        return x, ts_


# ============================================================================
# Position Embedding Utilities
# ============================================================================

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Generate 2D sine-cosine position embedding.

    Args:
        embed_dim: Output dimension for each position
        grid_size: int of the grid height and width
        cls_token: Whether to prepend a class token embedding

    Returns:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generate 1D sine-cosine position embedding from grid.
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


# ============================================================================
# Fallback Block Implementation (when timm is not available)
# ============================================================================

class FallbackMLP(nn.Module):
    """MLP block for transformer."""
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


class FallbackAttention(nn.Module):
    """Multi-head self attention."""
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


class FallbackBlock(nn.Module):
    """Transformer block with attention and MLP."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FallbackAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FallbackMLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# LightPath Model (LibCity Adapted)
# ============================================================================

class LightPath(AbstractTrafficStateModel):
    """
    LightPath: Masked Autoencoder with Vision Transformer for Trajectory Representation Learning.

    This model uses a masked autoencoder architecture with:
    - Pre-trained node2vec embeddings for road segments
    - Pre-trained time2vec embeddings for temporal information
    - Vision Transformer encoder for trajectory encoding
    - Vision Transformer decoder for reconstruction
    - Relational reasoning head for contrastive learning
    - ETA prediction head for travel time estimation

    The model supports two modes:
    1. Pre-training mode: Uses reconstruction + relational reasoning losses
    2. Fine-tuning mode: Uses ETA prediction loss

    Args:
        config: Configuration dictionary containing model hyperparameters
        data_feature: Dictionary containing data-specific features
    """

    def __init__(self, config, data_feature):
        super(LightPath, self).__init__(config, data_feature)

        self.config = config
        self.data_feature = data_feature
        self.device = config.get('device', torch.device('cpu'))

        # Model hyperparameters
        self.embed_dim = config.get('embed_dim', 128)
        self.num_patches = config.get('num_patches', 100)
        self.depth = config.get('depth', 12)
        self.num_heads = config.get('num_heads', 8)
        self.decoder_embed_dim = config.get('decoder_embed_dim', 128)
        self.decoder_depth = config.get('decoder_depth', 1)
        self.decoder_num_heads = config.get('decoder_num_heads', 8)
        self.mlp_ratio = config.get('mlp_ratio', 4.)
        self.norm_pix_loss = config.get('norm_pix_loss', False)

        # Masking ratios for training
        self.mask_ratio1 = config.get('mask_ratio1', 0.7)
        self.mask_ratio2 = config.get('mask_ratio2', 0.8)
        self.mask_ratio_eval = config.get('mask_ratio_eval', 0.0)

        # Loss weights
        self.rec_weight = config.get('rec_weight', 1.0)
        self.rr_weight = config.get('rr_weight', 1.0)
        self.eta_weight = config.get('eta_weight', 1.0)

        # Training mode: 'pretrain' or 'finetune'
        self.train_mode = config.get('train_mode', 'finetune')

        # Pre-trained embedding paths (configurable)
        self.node2vec_path = config.get('node2vec_path', None)
        self.time2vec_path = config.get('time2vec_path', None)

        # Vocabulary sizes for learnable embeddings (fallback)
        self.vocab_size = config.get('vocab_size', data_feature.get('vocab_size', 90000))
        self.time_size = config.get('time_size', data_feature.get('time_size', 10000))

        # Use pre-trained embeddings flag
        self.use_pretrained_embeddings = config.get('use_pretrained_embeddings', False)

        # Time normalization parameters
        self.time_mean = data_feature.get('time_mean', 0.0)
        self.time_std = data_feature.get('time_std', 1.0)

        # Determine which Block implementation to use
        if HAS_TIMM:
            self.BlockClass = Block
        else:
            self.BlockClass = FallbackBlock

        # Build model components
        self._build_patch_embed()
        self._build_encoder()
        self._build_decoder()
        self._build_relation_head()
        self._build_eta_head()

        # Initialize weights
        self.apply(self._init_vit_weights)

        # Loss functions
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_cls = nn.CrossEntropyLoss()

    def _load_pretrained_embeddings(self, path):
        """Load pre-trained embeddings from pickle file."""
        if path is None or not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as f:
                embeddings = pickle.load(f)
            if isinstance(embeddings, np.ndarray):
                return torch.from_numpy(embeddings).float()
            elif isinstance(embeddings, torch.Tensor):
                return embeddings.float()
            elif isinstance(embeddings, dict):
                # Handle dict format (e.g., from gensim)
                if 'vectors' in embeddings:
                    return torch.from_numpy(embeddings['vectors']).float()
                else:
                    # Assume it's a mapping of id -> vector
                    vectors = [embeddings[i] for i in sorted(embeddings.keys())]
                    return torch.from_numpy(np.array(vectors)).float()
            else:
                print(f"Warning: Unknown embedding format in {path}")
                return None
        except Exception as e:
            print(f"Warning: Failed to load embeddings from {path}: {e}")
            return None

    def _build_patch_embed(self):
        """Build patch embedding layer."""
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if self.use_pretrained_embeddings:
            # Load pre-trained embeddings
            node2vec = self._load_pretrained_embeddings(self.node2vec_path)
            time2vec = self._load_pretrained_embeddings(self.time2vec_path)

            if node2vec is not None and time2vec is not None:
                self.patch_embed = PatchEmbed(
                    node2vec=node2vec,
                    time2vec=time2vec,
                    embed_dim=self.embed_dim,
                    dropout=0.
                )
                # Update embed_dim based on actual embedding size
                self.embed_dim = node2vec.shape[1]
                print(f"Loaded pre-trained embeddings with dim={self.embed_dim}")
            else:
                print("Warning: Pre-trained embeddings not found, using learnable embeddings")
                self.patch_embed = LearnablePatchEmbed(
                    vocab_size=self.vocab_size,
                    time_size=self.time_size,
                    embed_dim=self.embed_dim,
                    dropout=0.
                )
        else:
            # Use learnable embeddings
            self.patch_embed = LearnablePatchEmbed(
                vocab_size=self.vocab_size,
                time_size=self.time_size,
                embed_dim=self.embed_dim,
                dropout=0.
            )

    def _build_encoder(self):
        """Build the MAE encoder."""
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Position embeddings (fixed sin-cos)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.embed_dim),
            requires_grad=False
        )

        # Transformer encoder blocks
        if HAS_TIMM:
            self.blocks = nn.ModuleList([
                self.BlockClass(
                    self.embed_dim, self.num_heads, self.mlp_ratio,
                    qkv_bias=True, qk_scale=None, norm_layer=norm_layer
                )
                for _ in range(self.depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                self.BlockClass(
                    self.embed_dim, self.num_heads, self.mlp_ratio,
                    qkv_bias=True, qk_scale=None, norm_layer=norm_layer
                )
                for _ in range(self.depth)
            ])

        self.norm = norm_layer(self.embed_dim)

    def _build_decoder(self):
        """Build the MAE decoder."""
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Decoder embedding projection
        self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim, bias=True)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))

        # Decoder position embeddings (fixed sin-cos)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.decoder_embed_dim),
            requires_grad=False
        )

        # Transformer decoder blocks
        if HAS_TIMM:
            self.decoder_blocks = nn.ModuleList([
                self.BlockClass(
                    self.decoder_embed_dim, self.decoder_num_heads, self.mlp_ratio,
                    qkv_bias=True, qk_scale=None, norm_layer=norm_layer
                )
                for _ in range(self.decoder_depth)
            ])
        else:
            self.decoder_blocks = nn.ModuleList([
                self.BlockClass(
                    self.decoder_embed_dim, self.decoder_num_heads, self.mlp_ratio,
                    qkv_bias=True, qk_scale=None, norm_layer=norm_layer
                )
                for _ in range(self.decoder_depth)
            ])

        self.decoder_norm = norm_layer(self.decoder_embed_dim)

    def _build_relation_head(self):
        """Build the relational reasoning head."""
        self.relation_head = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(self.embed_dim, 1)
        )

        # Classification token projection (for peak/off-peak classification)
        self.num_classes = 3
        self.clstoken = nn.Linear(self.embed_dim * 2, self.num_classes)
        self.acf = nn.ReLU()

    def _build_eta_head(self):
        """Build the ETA prediction head for travel time estimation."""
        hidden_dim = self.config.get('eta_hidden_dim', 128)

        self.eta_head = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

    def _init_vit_weights(self, module):
        """Initialize ViT weights."""
        if isinstance(module, nn.Linear):
            if hasattr(module, 'out_features') and module.out_features == self.num_classes:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.

        Args:
            x: Input sequence [N, L, D]
            mask_ratio: Ratio of tokens to mask

        Returns:
            x_masked: Masked sequence [N, L', D] where L' = L * (1 - mask_ratio)
            mask: Binary mask [N, L], 0 is keep, 1 is remove
            ids_restore: Indices to restore original order [N, L]
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, ts, mask_ratio):
        """
        Encode trajectory with masking.

        Args:
            x: Road segment indices [batch, seq_len]
            ts: Time indices [batch, seq_len]
            mask_ratio: Ratio of tokens to mask

        Returns:
            latent: Encoded representation [batch, L' + 1, embed_dim]
            mask: Binary mask [batch, L]
            ids_restore: Indices to restore order [batch, L]
            xx: Original embedded sequence [batch, L, embed_dim]
            ts_: Time embeddings [batch, L, embed_dim]
        """
        # Embed patches
        x, ts_ = self.patch_embed(x, ts)
        xx = x  # Save original embeddings for reconstruction target

        # Pad or truncate to num_patches
        batch_size, seq_len, embed_dim = x.shape
        if seq_len < self.num_patches:
            # Pad with zeros
            padding = torch.zeros(batch_size, self.num_patches - seq_len, embed_dim, device=x.device)
            x = torch.cat([x, padding], dim=1)
            ts_pad = torch.zeros(batch_size, self.num_patches - seq_len, ts_.shape[-1], device=x.device)
            ts_ = torch.cat([ts_, ts_pad], dim=1)
            xx = torch.cat([xx, padding], dim=1)
        elif seq_len > self.num_patches:
            # Truncate
            x = x[:, :self.num_patches, :]
            ts_ = ts_[:, :self.num_patches, :]
            xx = xx[:, :self.num_patches, :]

        # Add position embedding (without cls token)
        x = x + self.pos_embed[:, 1:, :]

        # Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, xx, ts_

    def forward_decoder(self, x, ids_restore):
        """
        Decode masked tokens for reconstruction.

        Args:
            x: Encoded representation [batch, L' + 1, embed_dim]
            ids_restore: Indices to restore order [batch, L]

        Returns:
            x: Reconstructed sequence [batch, L, decoder_embed_dim]
        """
        # Embed tokens
        x = self.decoder_embed(x)

        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # Add position embedding
        x = x + self.decoder_pos_embed

        # Apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, x, pred, mask):
        """
        Compute reconstruction loss on masked patches.

        Args:
            x: Original embeddings [N, L, D]
            pred: Predicted embeddings [N, L, D]
            mask: Binary mask [N, L], 0 is keep, 1 is remove

        Returns:
            loss: Mean squared error on masked positions
        """
        loss = (pred - x) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / (mask.sum() + 1e-8)  # mean loss on removed patches
        return loss

    def aggregate(self, features, K=2):
        """
        Aggregate features for relational reasoning.
        Creates positive pairs (same trajectory, different masks) and negative pairs.

        Args:
            features: Concatenated CLS tokens from two views [2*batch, embed_dim]
            K: Number of views (default 2)

        Returns:
            relation_pairs: Concatenated feature pairs
            targets: Binary labels (1 for positive, 0 for negative)
            scores: Predicted scores from relation head
        """
        relation_pairs_list = []
        targets_list = []
        size = int(features.shape[0] / K)
        shifts_counter = 1

        for index_1 in range(0, size * K, size):
            for index_2 in range(index_1 + size, size * K, size):
                # Positive pair: same trajectory, different masks
                pos_pair = torch.cat([
                    features[index_1:index_1 + size],
                    features[index_2:index_2 + size]
                ], 1)

                # Negative pair: different trajectories
                neg_pair = torch.cat([
                    features[index_1:index_1 + size],
                    torch.roll(features[index_2:index_2 + size], shifts=shifts_counter, dims=0)
                ], 1)

                relation_pairs_list.append(pos_pair)
                relation_pairs_list.append(neg_pair)
                targets_list.append(torch.ones(size, dtype=torch.float32))
                targets_list.append(torch.zeros(size, dtype=torch.float32))

                shifts_counter += 1
                if shifts_counter >= size:
                    shifts_counter = 1  # avoid identity pairs

        relation_pairs = torch.cat(relation_pairs_list, 0)
        targets = torch.cat(targets_list, 0)

        return relation_pairs, targets, self.relation_head(relation_pairs)

    def embed(self, inputs, ts, mask_ratio=0.0):
        """
        Get trajectory embedding (CLS token) without masking.

        Args:
            inputs: Road segment indices [batch, seq_len]
            ts: Time indices [batch, seq_len]
            mask_ratio: Masking ratio (default 0 for inference)

        Returns:
            cls_token: Trajectory representation [batch, embed_dim]
        """
        latent, _, _, _, _ = self.forward_encoder(inputs, ts, mask_ratio)
        return latent[:, 0, :]

    def forward(self, batch):
        """
        Forward pass of LightPath model.

        In training mode, returns predictions and intermediate results for loss computation.
        In evaluation mode, returns only ETA predictions.

        Args:
            batch: Dictionary containing:
                - 'road_segments' or 'X': Road segment indices [batch, seq_len] or [batch, seq_len, features]
                - 'timestamps' or 'ts': Time indices [batch, seq_len]
                - 'time' (optional): Ground truth travel time for ETA task

        Returns:
            If training and train_mode == 'pretrain':
                dict with 'loss_rec', 'loss_rr', 'pred1', 'mask1', 'pred2', 'mask2'
            If training and train_mode == 'finetune':
                eta_pred: Predicted travel time [batch, 1]
            If not training:
                eta_pred: Predicted travel time [batch, 1]
        """
        # Extract inputs from batch
        # Use batch.data for key checks to avoid KeyError with LibCity's Batch class
        if 'road_segments' in batch.data:
            inputs = batch['road_segments']
        elif 'X' in batch.data:
            X = batch['X']
            if X.dim() == 4:  # [batch, time, nodes, features]
                inputs = X[:, :, 0, 0].long()  # Take first node, first feature as road segment
            elif X.dim() == 3:  # [batch, seq_len, features]
                inputs = X[:, :, 0].long()
            else:
                inputs = X.long()
        else:
            raise KeyError("Batch must contain 'road_segments' or 'X'")

        if 'timestamps' in batch.data:
            ts = batch['timestamps']
        elif 'ts' in batch.data:
            ts = batch['ts']
        elif 'X' in batch.data:
            X = batch['X']
            if X.dim() == 4:
                ts = X[:, :, 0, 1].long() if X.shape[-1] > 1 else torch.zeros_like(inputs)
            elif X.dim() == 3:
                ts = X[:, :, 1].long() if X.shape[-1] > 1 else torch.zeros_like(inputs)
            else:
                ts = torch.zeros_like(inputs)
        else:
            ts = torch.zeros_like(inputs)

        if self.training and self.train_mode == 'pretrain':
            # Pre-training mode: reconstruction + relational reasoning
            latent1, mask1, ids_restore1, xx1, ts1 = self.forward_encoder(inputs, ts, self.mask_ratio1)
            latent2, mask2, ids_restore2, xx2, ts2 = self.forward_encoder(inputs, ts, self.mask_ratio2)

            pred1 = self.forward_decoder(latent1, ids_restore1)
            pred2 = self.forward_decoder(latent2, ids_restore2)

            cls_token_1 = latent1[:, 0, :]
            cls_token_2 = latent2[:, 0, :]

            # Reconstruction loss
            loss_rec1 = self.forward_loss(xx1, pred1, mask1)
            loss_rec2 = self.forward_loss(xx2, pred2, mask2)
            loss_rec = 0.5 * (loss_rec1 + loss_rec2)

            # Relational reasoning loss
            cls_feature_rr = torch.cat([cls_token_1, cls_token_2], 0)
            relation_pairs, targets_rr, scores = self.aggregate(cls_feature_rr, K=2)

            scores = scores.to(self.device)
            targets_rr = targets_rr.to(self.device)
            loss_rr = self.criterion_bce(torch.squeeze(scores, 1), targets_rr)

            return {
                'loss_rec': loss_rec,
                'loss_rr': loss_rr,
                'pred1': pred1,
                'mask1': mask1,
                'pred2': pred2,
                'mask2': mask2,
                'cls_token': cls_token_1
            }
        else:
            # Fine-tuning or evaluation mode: ETA prediction
            cls_token = self.embed(inputs, ts, mask_ratio=self.mask_ratio_eval)
            eta_pred = self.eta_head(cls_token)
            return eta_pred

    def predict(self, batch):
        """
        Predict travel times for a batch.

        Args:
            batch: Input batch dictionary

        Returns:
            Predicted travel times [batch] (unnormalized)
        """
        self.eval()
        with torch.no_grad():
            eta_pred = self.forward(batch)

            # Unnormalize if needed
            if self.time_std != 1.0 or self.time_mean != 0.0:
                eta_pred = eta_pred * self.time_std + self.time_mean

            return eta_pred  # Keep shape (batch_size, 1) to match y_true

    def calculate_loss(self, batch):
        """
        Calculate the training loss.

        In pre-training mode:
            loss = rec_weight * loss_rec + rr_weight * loss_rr

        In fine-tuning mode:
            loss = MSE(predicted_time, actual_time)

        Args:
            batch: Input batch dictionary with 'time' as ground truth

        Returns:
            Combined loss tensor
        """
        if self.train_mode == 'pretrain':
            output = self.forward(batch)
            loss = self.rec_weight * output['loss_rec'] + self.rr_weight * output['loss_rr']
            return loss
        else:
            # Fine-tuning mode: ETA prediction
            eta_pred = self.forward(batch)

            # Get ground truth travel time
            if 'time' in batch.data:
                truth_data = batch['time']
            elif 'y' in batch.data:
                y = batch['y']
                if y.dim() == 4:  # [batch, time, nodes, features]
                    truth_data = y.squeeze()
                else:
                    truth_data = y
            else:
                raise KeyError("Batch must contain 'time' or 'y' for loss calculation")

            if truth_data.dim() == 0:
                truth_data = truth_data.unsqueeze(0)
            if truth_data.dim() == 1:
                truth_data = truth_data.unsqueeze(1)

            # Normalize ground truth for loss computation
            if self.time_std != 1.0 or self.time_mean != 0.0:
                truth_normalized = (truth_data - self.time_mean) / self.time_std
                pred_normalized = (eta_pred - self.time_mean) / self.time_std
                loss = F.mse_loss(pred_normalized.squeeze(), truth_normalized.squeeze())
            else:
                loss = F.mse_loss(eta_pred.squeeze(), truth_data.squeeze())

            return loss
