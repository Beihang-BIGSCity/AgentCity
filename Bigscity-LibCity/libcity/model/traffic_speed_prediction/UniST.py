"""
UniST: Universal Spatio-Temporal Model for Traffic Prediction

Adapted from: https://github.com/tsinghua-fib-lab/UniST
Paper: "UniST: A Prompt-Enhanced Universal Model for Urban Spatio-Temporal Prediction" (KDD 2024)

This implementation adapts the UniST model to the LibCity framework.
The model uses a Vision Transformer architecture with masked autoencoder and prompt tuning
for traffic speed prediction.

Key Features:
- Vision Transformer backbone with masked autoencoding
- Spatial-temporal prompt network for domain adaptation
- Two-stage training: pre-training + prompt tuning
- This implementation focuses on the prompt tuning stage for LibCity integration
"""

from functools import partial
from logging import getLogger
import copy

import torch
import torch.nn as nn
import math
import numpy as np

from timm.models.layers import to_2tuple
from timm.models.vision_transformer import DropPath, Mlp

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss

# Import UniST supporting modules
from libcity.model.traffic_speed_prediction.unist_modules.Embed import (
    DataEmbedding, TokenEmbedding, SpatialPatchEmb,
    get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
)
from libcity.model.traffic_speed_prediction.unist_modules.Prompt_network import Prompt_ST
from libcity.model.traffic_speed_prediction.unist_modules.mask_strategy import (
    random_masking, tube_masking, tube_block_masking, causal_masking,
    random_masking_evaluate, tube_masking_evaluate, tube_block_masking_evaluate,
    random_restore, tube_restore, causal_restore
)


class Attention(nn.Module):
    """Multi-head Self-Attention module."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size

    def forward(self, x):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x


class Block(nn.Module):
    """Transformer Block with specified Attention function."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class UniSTCore(nn.Module):
    """Core UniST Masked Autoencoder with VisionTransformer backbone."""

    def __init__(self, patch_size=1, in_chans=1,
                 embed_dim=512, decoder_embed_dim=512, depth=12, decoder_depth=8,
                 num_heads=8, decoder_num_heads=4, mlp_ratio=2, norm_layer=nn.LayerNorm,
                 t_patch_size=1, no_qkv_bias=False, pos_emb='trivial',
                 prompt_ST=1, num_memory_spatial=128, num_memory_temporal=128,
                 conv_num=3, his_len=12, pred_len=12, prompt_content='s_p_c',
                 hour_size=48, weekday_size=7):
        super().__init__()

        self.pos_emb = pos_emb
        self.prompt_ST = prompt_ST
        self.his_len = his_len
        self.pred_len = pred_len
        self.prompt_content = prompt_content

        # Embedding layers
        self.Embedding = DataEmbedding(1, embed_dim, t_patch_size=t_patch_size,
                                        patch_size=patch_size, size1=hour_size, size2=weekday_size)
        self.Embedding_24 = DataEmbedding(1, embed_dim, t_patch_size=t_patch_size,
                                           patch_size=patch_size, size1=24, size2=7)

        # Prompt network for spatial-temporal patterns
        if prompt_ST != 0:
            self.st_prompt = Prompt_ST(num_memory_spatial, num_memory_temporal, embed_dim,
                                       his_len, conv_num)
            self.spatial_patch = SpatialPatchEmb(embed_dim, embed_dim, patch_size)

        # Model configuration
        self.t_patch_size = t_patch_size
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.no_qkv_bias = no_qkv_bias

        # Position embeddings
        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, 1024, embed_dim))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, 50, embed_dim))
        self.decoder_pos_embed_spatial = nn.Parameter(torch.zeros(1, 1024, decoder_embed_dim))
        self.decoder_pos_embed_temporal = nn.Parameter(torch.zeros(1, 50, decoder_embed_dim))

        # Encoder blocks
        self.blocks = nn.ModuleList([
            Block(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=not no_qkv_bias,
                qk_scale=None,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.norm = norm_layer(embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=not no_qkv_bias)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(
                decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio,
                qkv_bias=not no_qkv_bias,
                qk_scale=None,
                norm_layer=norm_layer,
            )
            for i in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # Prediction head
        self.decoder_pred = nn.Sequential(*[
            nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=not no_qkv_bias),
            nn.GELU(),
            nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=not no_qkv_bias),
            nn.GELU(),
            nn.Linear(decoder_embed_dim, t_patch_size * patch_size**2 * in_chans, bias=not no_qkv_bias)
        ])

        self.initialize_weights_trivial()

    def initialize_weights_trivial(self):
        torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)
        torch.nn.init.trunc_normal_(self.Embedding.temporal_embedding.hour_embed.weight.data, std=0.02)
        torch.nn.init.trunc_normal_(self.Embedding.temporal_embedding.weekday_embed.weight.data, std=0.02)

        w = self.Embedding.value_embedding.tokenConv.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_weights_sincos(self, num_t_patch, num_patch_1, num_patch_2):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed_spatial.shape[-1],
            grid_size1=num_patch_1,
            grid_size2=num_patch_2
        )

        pos_embed_spatial = nn.Parameter(
            torch.zeros(1, num_patch_1 * num_patch_2, self.embed_dim)
        )
        pos_embed_temporal = nn.Parameter(
            torch.zeros(1, num_t_patch, self.embed_dim)
        )

        pos_embed_spatial.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        pos_temporal_emb = get_1d_sincos_pos_embed_from_grid(
            pos_embed_temporal.shape[-1], np.arange(num_t_patch, dtype=np.float32)
        )
        pos_embed_temporal.data.copy_(torch.from_numpy(pos_temporal_emb).float().unsqueeze(0))

        pos_embed_spatial.requires_grad = False
        pos_embed_temporal.requires_grad = False

        return pos_embed_spatial, pos_embed_temporal, copy.deepcopy(pos_embed_spatial), copy.deepcopy(pos_embed_temporal)

    def patchify(self, imgs):
        """
        imgs: (N, 1, T, H, W)
        x: (N, L, patch_size**2 *1)
        """
        N, _, T, H, W = imgs.shape
        p = self.patch_size
        u = self.t_patch_size
        assert H % p == 0 and W % p == 0 and T % u == 0
        h = H // p
        w = W // p
        t = T // u
        x = imgs.reshape(shape=(N, 1, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * 1))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x

    def unpatchify(self, imgs):
        """
        imgs: (N, L, patch_size**2 *1)
        x: (N, 1, T, H, W)
        """
        N, T, H, W, p, u, t, h, w = self.patch_info
        imgs = imgs.reshape(shape=(N, t, h, w, u, p, p))
        imgs = torch.einsum("nthwupq->ntuhpwq", imgs)
        imgs = imgs.reshape(shape=(N, T, H, W))
        return imgs

    def pos_embed_enc(self, ids_keep, batch, input_size):
        if self.pos_emb == 'trivial':
            pos_embed = self.pos_embed_spatial[:, :input_size[1]*input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal[:, :input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )
        elif self.pos_emb == 'SinCos':
            pos_embed_spatial, pos_embed_temporal, _, _ = self.get_weights_sincos(
                input_size[0], input_size[1], input_size[2]
            )
            pos_embed = pos_embed_spatial[:, :input_size[1]*input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                pos_embed_temporal[:, :input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )
        pos_embed = pos_embed.to(ids_keep.device)
        pos_embed = pos_embed.expand(batch, -1, -1)

        pos_embed_sort = torch.gather(
            pos_embed,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
        )
        return pos_embed_sort

    def pos_embed_dec(self, ids_keep, batch, input_size):
        if self.pos_emb == 'trivial':
            decoder_pos_embed = self.decoder_pos_embed_spatial[:, :input_size[1]*input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal[:, :input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )
        elif self.pos_emb == 'SinCos':
            _, _, decoder_pos_embed_spatial, decoder_pos_embed_temporal = self.get_weights_sincos(
                input_size[0], input_size[1], input_size[2]
            )
            decoder_pos_embed = decoder_pos_embed_spatial[:, :input_size[1]*input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                decoder_pos_embed_temporal[:, :input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )
        decoder_pos_embed = decoder_pos_embed.to(ids_keep.device)
        decoder_pos_embed = decoder_pos_embed.expand(batch, -1, -1)
        return decoder_pos_embed

    def prompt_generate(self, shape, x_period, x_closeness, x, pos):
        P = x_period.shape[1]
        HW = x_closeness.shape[2]

        x_period = x_period.unsqueeze(2).reshape(-1, 1, x_period.shape[-3], x_period.shape[-2], x_period.shape[-1])
        x_closeness = x_closeness.permute(0, 2, 1, 3).reshape(-1, x_closeness.shape[1], x_closeness.shape[-1])

        # Use standard embedding
        x_period = self.Embedding.value_embedding(x_period).reshape(shape[0], P, -1, self.embed_dim)
        x_period = x_period.permute(0, 2, 1, 3).reshape(-1, x_period.shape[1], x_period.shape[-1])

        prompt_t = self.st_prompt.temporal_prompt(x_closeness, x_period)

        prompt_c = prompt_t['hc'].reshape(shape[0], -1, prompt_t['hc'].shape[-1])
        prompt_p = prompt_t['hp'].reshape(shape[0], -1, prompt_t['hp'].shape[-1])

        prompt_c = prompt_c.unsqueeze(dim=1).repeat(1, self.pred_len // self.t_patch_size, 1, 1)

        pos_t = pos.reshape(shape[0], (self.his_len + self.pred_len) // self.t_patch_size, HW, self.embed_dim)[:, -self.pred_len // self.t_patch_size:]

        assert prompt_c.shape == pos_t.shape
        prompt_c = (prompt_c + pos_t).reshape(shape[0], -1, self.embed_dim)

        t_loss = prompt_t['loss']

        out_s = self.st_prompt.spatial_prompt(x)
        out_s, s_loss = out_s['out'], out_s['loss']
        out_s = [self.spatial_patch(i).unsqueeze(dim=1).repeat(
            1, self.pred_len // self.t_patch_size, 1, 1
        ).reshape(i.shape[0], -1, self.embed_dim).unsqueeze(dim=0) for i in out_s]

        out_s = torch.mean(torch.cat(out_s, dim=0), dim=0)

        return dict(tc=prompt_c, tp=prompt_p, s=out_s, loss=t_loss + s_loss)

    def forward_encoder(self, x, x_mark, mask_ratio, mask_strategy, mode='backward', use_24h_emb=False):
        N, _, T, H, W = x.shape

        if use_24h_emb:
            x, TimeEmb = self.Embedding_24(x, x_mark, is_time=True)
        else:
            x, TimeEmb = self.Embedding(x, x_mark, is_time=True)

        _, L, C = x.shape
        T = T // self.t_patch_size

        assert mode in ['backward', 'forward']

        if mode == 'backward':
            if mask_strategy == 'random':
                x, mask, ids_restore, ids_keep = random_masking(x, mask_ratio)
            elif mask_strategy == 'tube':
                x, mask, ids_restore, ids_keep = tube_masking(x, mask_ratio, T=T)
            elif mask_strategy == 'block':
                x, mask, ids_restore, ids_keep = tube_block_masking(x, mask_ratio, T=T)
            elif mask_strategy in ['frame', 'temporal']:
                x, mask, ids_restore, ids_keep = causal_masking(x, mask_ratio, T=T, mask_strategy=mask_strategy)
        elif mode == 'forward':
            if mask_strategy == 'random':
                x, mask, ids_restore, ids_keep = random_masking_evaluate(x, mask_ratio)
            elif mask_strategy == 'tube':
                x, mask, ids_restore, ids_keep = tube_masking_evaluate(x, mask_ratio, T=T)
            elif mask_strategy == 'block':
                x, mask, ids_restore, ids_keep = tube_block_masking_evaluate(x, mask_ratio, T=T)
            elif mask_strategy in ['frame', 'temporal']:
                x, mask, ids_restore, ids_keep = causal_masking(x, mask_ratio, T=T, mask_strategy=mask_strategy)

        input_size = (T, H // self.patch_size, W // self.patch_size)
        pos_embed_sort = self.pos_embed_enc(ids_keep, N, input_size)

        assert x.shape == pos_embed_sort.shape
        x_attn = x + pos_embed_sort

        for index, blk in enumerate(self.blocks):
            x_attn = blk(x_attn)

        x_attn = self.norm(x_attn)

        return x_attn, mask, ids_restore, input_size, TimeEmb

    def forward_decoder(self, x, x_period, x_origin, ids_restore, mask_strategy, TimeEmb,
                        input_size=None):
        N = x.shape[0]
        T, H, W = input_size

        x = self.decoder_embed(x)
        C = x.shape[-1]

        if mask_strategy == 'random':
            x = random_restore(x, ids_restore, N, T, H, W, C, self.mask_token)
        elif mask_strategy in ['tube', 'block']:
            x = tube_restore(x, ids_restore, N, T, H, W, C, self.mask_token)
        elif mask_strategy in ['frame', 'temporal']:
            x = causal_restore(x, ids_restore, N, T, H, W, C, self.mask_token)

        decoder_pos_embed = self.pos_embed_dec(ids_restore, N, input_size)

        assert x.shape == decoder_pos_embed.shape == TimeEmb.shape
        x_attn = x + decoder_pos_embed + TimeEmb

        if self.prompt_ST == 1:
            prompt = self.prompt_generate(
                x_attn.shape, x_period,
                x_attn.reshape(N, T, H*W, x_attn.shape[-1]),
                x_origin, pos=decoder_pos_embed + TimeEmb
            )

            if self.prompt_content == 's_p':
                token_prompt = prompt['tp'] + prompt['s']
            elif self.prompt_content == 'p_c':
                token_prompt = prompt['tp'] + prompt['tc']
            elif self.prompt_content == 's_c':
                token_prompt = prompt['s'] + prompt['tc']
            elif self.prompt_content == 's':
                token_prompt = prompt['s']
            elif self.prompt_content == 'p':
                token_prompt = prompt['tp']
            elif self.prompt_content == 'c':
                token_prompt = prompt['tc']
            elif self.prompt_content == 's_p_c':
                token_prompt = prompt['tc'] + prompt['s'] + prompt['tp']

            x_attn[:, -self.pred_len // self.t_patch_size * H * W:] += token_prompt

        for index, blk in enumerate(self.decoder_blocks):
            x_attn = blk(x_attn)
        x_attn = self.decoder_norm(x_attn)

        return x_attn

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, T, H, W]
        pred: [N, t*h*w, u*p*p*1]
        mask: [N*t, h*w], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        assert pred.shape == target.shape

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)

        loss1 = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        loss2 = (loss * (1-mask)).sum() / (1-mask).sum()
        return loss1, loss2, target

    def forward(self, imgs, imgs_mark, imgs_period=None, mask_ratio=0.5, mask_strategy='temporal',
                mode='backward', use_24h_emb=False):
        """
        Forward pass of UniST core.

        Args:
            imgs: Input tensor (N, 1, T, H, W)
            imgs_mark: Time markers (N, T, 2) with weekday and hour information
            imgs_period: Period data for prompt generation
            mask_ratio: Ratio of tokens to mask
            mask_strategy: Masking strategy ('random', 'tube', 'block', 'frame', 'temporal')
            mode: 'backward' for training, 'forward' for evaluation
            use_24h_emb: Use 24-hour embedding (for specific datasets)

        Returns:
            loss1, loss2, pred, target, mask
        """
        if imgs_period is not None:
            imgs_period = imgs_period[:, :, self.his_len:]

        T, H, W = imgs.shape[2:]
        latent, mask, ids_restore, input_size, TimeEmb = self.forward_encoder(
            imgs, imgs_mark, mask_ratio, mask_strategy, mode=mode, use_24h_emb=use_24h_emb
        )

        pred = self.forward_decoder(
            latent, imgs_period, imgs[:, :, :self.his_len].squeeze(dim=1).clone(),
            ids_restore, mask_strategy, TimeEmb, input_size=input_size
        )
        L = pred.shape[1]

        pred = self.decoder_pred(pred)
        loss1, loss2, target = self.forward_loss(imgs, pred, mask)

        return loss1, loss2, pred, target, mask


class UniST(AbstractTrafficStateModel):
    """
    UniST model adapted for LibCity traffic speed prediction.

    This model uses a Vision Transformer architecture with masked autoencoder
    and prompt tuning for traffic speed prediction. It is designed to work with
    grid-based traffic data.

    Paper: "UniST: A Prompt-Enhanced Universal Model for Urban Spatio-Temporal Prediction" (KDD 2024)
    Original Implementation: https://github.com/tsinghua-fib-lab/UniST
    """

    def __init__(self, config, data_feature):
        """
        Initialize UniST model.

        Args:
            config (dict): Configuration dictionary containing model parameters
            data_feature (dict): Dataset features including:
                - num_nodes: Number of traffic nodes (will be mapped to grid)
                - feature_dim: Input feature dimension
                - output_dim: Output feature dimension
                - scaler: Data scaler object
        """
        super().__init__(config, data_feature)

        # Section 1: Extract data features
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self._logger = getLogger()

        # Section 2: Model configuration parameters
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.device = config.get('device', torch.device('cpu'))

        # Grid configuration - UniST expects grid data
        self.grid_height = config.get('grid_height', 16)
        self.grid_width = config.get('grid_width', 16)

        # If num_nodes is provided but grid dimensions are not, try to infer grid size
        if self.num_nodes > 1 and self.grid_height * self.grid_width != self.num_nodes:
            # Try to find the closest square grid
            sqrt_nodes = int(np.sqrt(self.num_nodes))
            if sqrt_nodes * sqrt_nodes == self.num_nodes:
                self.grid_height = sqrt_nodes
                self.grid_width = sqrt_nodes
            else:
                self._logger.warning(
                    f"num_nodes ({self.num_nodes}) does not match grid_height * grid_width "
                    f"({self.grid_height} * {self.grid_width}). Using configured grid dimensions."
                )

        # UniST specific parameters
        self.model_size = config.get('model_size', 'middle')  # Model size preset
        self.embed_dim = config.get('embed_dim', 128)
        self.depth = config.get('depth', 6)
        self.decoder_embed_dim = config.get('decoder_embed_dim', 128)
        self.decoder_depth = config.get('decoder_depth', 4)
        self.num_heads = config.get('num_heads', 8)
        self.decoder_num_heads = config.get('decoder_num_heads', 4)
        self.mlp_ratio = config.get('mlp_ratio', 2)
        self.patch_size = config.get('patch_size', 1)
        self.t_patch_size = config.get('t_patch_size', 1)
        self.pos_emb = config.get('pos_emb', 'trivial')
        self.no_qkv_bias = config.get('no_qkv_bias', False)

        # Prompt tuning parameters
        self.prompt_ST = config.get('prompt_ST', 1)  # Enable prompt tuning
        self.num_memory_spatial = config.get('num_memory_spatial', 128)
        self.num_memory_temporal = config.get('num_memory_temporal', 128)
        self.conv_num = config.get('conv_num', 3)
        self.prompt_content = config.get('prompt_content', 's_p_c')

        # Masking parameters
        self.mask_ratio = config.get('mask_ratio', 0.5)
        self.mask_strategy = config.get('mask_strategy', 'temporal')

        # Temporal embedding sizes
        self.hour_size = config.get('hour_size', 48)  # 48 for 30-min, 24 for 1-hour
        self.weekday_size = config.get('weekday_size', 7)
        self.use_24h_emb = config.get('use_24h_emb', False)

        # Pre-trained weights path (optional)
        self.pretrained_weights = config.get('pretrained_weights', None)

        # Apply model size presets if specified
        self._apply_model_size_preset()

        # Section 3: Build the model
        self.unist_core = UniSTCore(
            patch_size=self.patch_size,
            in_chans=1,
            embed_dim=self.embed_dim,
            decoder_embed_dim=self.decoder_embed_dim,
            depth=self.depth,
            decoder_depth=self.decoder_depth,
            num_heads=self.num_heads,
            decoder_num_heads=self.decoder_num_heads,
            mlp_ratio=self.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            t_patch_size=self.t_patch_size,
            no_qkv_bias=self.no_qkv_bias,
            pos_emb=self.pos_emb,
            prompt_ST=self.prompt_ST,
            num_memory_spatial=self.num_memory_spatial,
            num_memory_temporal=self.num_memory_temporal,
            conv_num=self.conv_num,
            his_len=self.input_window,
            pred_len=self.output_window,
            prompt_content=self.prompt_content,
            hour_size=self.hour_size,
            weekday_size=self.weekday_size
        )

        # Load pre-trained weights if specified
        if self.pretrained_weights is not None:
            self._load_pretrained_weights()

        self._logger.info(f"UniST model initialized with {sum(p.numel() for p in self.parameters() if p.requires_grad)} trainable parameters")

    def _apply_model_size_preset(self):
        """Apply model size presets to configure dimensions and layers."""
        size_presets = {
            '1': {'embed_dim': 64, 'depth': 2, 'decoder_embed_dim': 64, 'decoder_depth': 2,
                  'num_heads': 4, 'decoder_num_heads': 2},
            '2': {'embed_dim': 64, 'depth': 6, 'decoder_embed_dim': 64, 'decoder_depth': 4,
                  'num_heads': 4, 'decoder_num_heads': 2},
            '3': {'embed_dim': 128, 'depth': 4, 'decoder_embed_dim': 128, 'decoder_depth': 3,
                  'num_heads': 8, 'decoder_num_heads': 4},
            '4': {'embed_dim': 128, 'depth': 8, 'decoder_embed_dim': 128, 'decoder_depth': 8,
                  'num_heads': 8, 'decoder_num_heads': 4},
            '5': {'embed_dim': 256, 'depth': 4, 'decoder_embed_dim': 256, 'decoder_depth': 4,
                  'num_heads': 8, 'decoder_num_heads': 8},
            '6': {'embed_dim': 256, 'depth': 8, 'decoder_embed_dim': 256, 'decoder_depth': 6,
                  'num_heads': 16, 'decoder_num_heads': 8},
            '7': {'embed_dim': 256, 'depth': 12, 'decoder_embed_dim': 256, 'decoder_depth': 10,
                  'num_heads': 16, 'decoder_num_heads': 16},
            'middle': {'embed_dim': 128, 'depth': 6, 'decoder_embed_dim': 128, 'decoder_depth': 4,
                       'num_heads': 8, 'decoder_num_heads': 4},
            'large': {'embed_dim': 384, 'depth': 6, 'decoder_embed_dim': 384, 'decoder_depth': 6,
                      'num_heads': 8, 'decoder_num_heads': 8},
        }

        if self.model_size in size_presets:
            preset = size_presets[self.model_size]
            for key, value in preset.items():
                if not hasattr(self, key) or getattr(self, key) is None:
                    setattr(self, key, value)
            self._logger.info(f"Applied model size preset: {self.model_size}")

    def _load_pretrained_weights(self):
        """Load pre-trained weights from file."""
        try:
            checkpoint = torch.load(self.pretrained_weights, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Filter out incompatible keys
            model_dict = self.unist_core.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.unist_core.load_state_dict(model_dict, strict=False)

            self._logger.info(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters from {self.pretrained_weights}")
        except Exception as e:
            self._logger.warning(f"Failed to load pre-trained weights: {e}")

    def _prepare_input(self, batch):
        """
        Prepare input data for UniST model.

        Converts LibCity batch format to UniST expected format:
        - LibCity: batch['X'] shape (batch_size, input_window, num_nodes, feature_dim)
        - UniST: (N, 1, T, H, W) + time markers (N, T, 2)

        Args:
            batch: LibCity batch dictionary

        Returns:
            imgs: Tensor of shape (N, 1, T, H, W)
            imgs_mark: Tensor of shape (N, T, 2) with weekday and hour information
            imgs_period: Tensor for period information (optional)
        """
        x = batch['X'].to(self.device)  # (B, T, N, F)
        batch_size, seq_len, num_nodes, feature_dim = x.shape

        # Extract the main traffic feature (first channel)
        x_main = x[..., 0]  # (B, T, N)

        # Reshape to grid format
        H, W = self.grid_height, self.grid_width
        if num_nodes != H * W:
            # Pad or truncate if necessary
            if num_nodes < H * W:
                padding = torch.zeros(batch_size, seq_len, H * W - num_nodes, device=x.device)
                x_main = torch.cat([x_main, padding], dim=2)
            else:
                x_main = x_main[:, :, :H * W]

        # Reshape to (B, 1, T, H, W)
        imgs = x_main.reshape(batch_size, seq_len, H, W)
        imgs = imgs.unsqueeze(1)  # Add channel dimension: (B, 1, T, H, W)

        # Prepare time markers
        # Try to extract time features from input data
        if feature_dim > 1:
            # Assume time features are in the input
            # time_of_day is typically the second feature (index 1)
            time_of_day = x[:, :, 0, 1] if feature_dim > 1 else torch.zeros(batch_size, seq_len, device=x.device)
            # day_of_week is typically encoded in features 2-8 (one-hot)
            if feature_dim > 2:
                day_of_week = torch.argmax(x[:, :, 0, 2:9], dim=-1) if feature_dim > 8 else torch.zeros(batch_size, seq_len, device=x.device, dtype=torch.long)
            else:
                day_of_week = torch.zeros(batch_size, seq_len, device=x.device, dtype=torch.long)
        else:
            # Generate default time markers
            time_of_day = torch.zeros(batch_size, seq_len, device=x.device)
            day_of_week = torch.zeros(batch_size, seq_len, device=x.device, dtype=torch.long)

        # Convert time_of_day to hour indices
        if time_of_day.max() <= 1.0:  # Normalized
            hour_indices = (time_of_day * self.hour_size).long()
        else:
            hour_indices = time_of_day.long() % self.hour_size

        # Create time markers: (N, T, 2) with [weekday, hour]
        imgs_mark = torch.stack([day_of_week.long(), hour_indices.long()], dim=-1)

        # Create period data (simplified - use historical data as period reference)
        imgs_period = None
        if self.prompt_ST:
            # Use the historical data reshaped for period prompt
            imgs_period = imgs.clone()

        return imgs, imgs_mark, imgs_period

    def _prepare_full_sequence(self, batch):
        """
        Prepare full sequence (input + target) for training with masked prediction.

        Args:
            batch: LibCity batch dictionary with 'X' and 'y'

        Returns:
            Combined sequence for masked prediction training
        """
        x = batch['X'].to(self.device)  # (B, T_in, N, F)
        y = batch['y'].to(self.device)  # (B, T_out, N, F)

        batch_size = x.shape[0]
        H, W = self.grid_height, self.grid_width

        # Extract main feature and reshape to grid
        x_main = x[..., 0]  # (B, T_in, N)
        y_main = y[..., 0]  # (B, T_out, N)

        # Handle node count mismatch
        num_nodes = x_main.shape[2]
        if num_nodes != H * W:
            if num_nodes < H * W:
                x_padding = torch.zeros(batch_size, x_main.shape[1], H * W - num_nodes, device=x.device)
                y_padding = torch.zeros(batch_size, y_main.shape[1], H * W - num_nodes, device=y.device)
                x_main = torch.cat([x_main, x_padding], dim=2)
                y_main = torch.cat([y_main, y_padding], dim=2)
            else:
                x_main = x_main[:, :, :H * W]
                y_main = y_main[:, :, :H * W]

        # Combine input and target for full sequence
        full_seq = torch.cat([x_main, y_main], dim=1)  # (B, T_in + T_out, H*W)
        full_seq = full_seq.reshape(batch_size, self.input_window + self.output_window, H, W)
        imgs = full_seq.unsqueeze(1)  # (B, 1, T_in + T_out, H, W)

        # Prepare time markers for full sequence
        feature_dim = x.shape[-1]
        if feature_dim > 1:
            time_of_day_x = x[:, :, 0, 1] if feature_dim > 1 else torch.zeros(batch_size, self.input_window, device=x.device)
            time_of_day_y = y[:, :, 0, 1] if feature_dim > 1 else torch.zeros(batch_size, self.output_window, device=y.device)
            time_of_day = torch.cat([time_of_day_x, time_of_day_y], dim=1)

            if feature_dim > 8:
                dow_x = torch.argmax(x[:, :, 0, 2:9], dim=-1)
                dow_y = torch.argmax(y[:, :, 0, 2:9], dim=-1)
                day_of_week = torch.cat([dow_x, dow_y], dim=1)
            else:
                day_of_week = torch.zeros(batch_size, self.input_window + self.output_window, device=x.device, dtype=torch.long)
        else:
            time_of_day = torch.zeros(batch_size, self.input_window + self.output_window, device=x.device)
            day_of_week = torch.zeros(batch_size, self.input_window + self.output_window, device=x.device, dtype=torch.long)

        if time_of_day.max() <= 1.0:
            hour_indices = (time_of_day * self.hour_size).long()
        else:
            hour_indices = time_of_day.long() % self.hour_size

        imgs_mark = torch.stack([day_of_week.long(), hour_indices.long()], dim=-1)

        # Period data
        imgs_period = imgs.clone()

        return imgs, imgs_mark, imgs_period

    def forward(self, batch):
        """
        Forward pass of the UniST model.

        Args:
            batch (dict): Dictionary containing:
                - X: Input tensor of shape (batch_size, input_window, num_nodes, feature_dim)
                - y: Target tensor of shape (batch_size, output_window, num_nodes, output_dim)

        Returns:
            torch.Tensor: Predictions of shape (batch_size, output_window, num_nodes, output_dim)
        """
        # Prepare full sequence for masked prediction
        imgs, imgs_mark, imgs_period = self._prepare_full_sequence(batch)

        # Forward through UniST core in evaluation mode
        mode = 'forward' if not self.training else 'backward'
        loss1, loss2, pred, target, mask = self.unist_core.forward(
            imgs, imgs_mark, imgs_period,
            mask_ratio=self.mask_ratio,
            mask_strategy=self.mask_strategy,
            mode=mode,
            use_24h_emb=self.use_24h_emb
        )

        # Unpatchify predictions
        pred_unpatch = self.unist_core.unpatchify(pred)  # (N, T, H, W)

        # Extract only the prediction window
        pred_output = pred_unpatch[:, self.input_window:, :, :]  # (N, T_out, H, W)

        # Reshape back to node format
        batch_size = pred_output.shape[0]
        H, W = self.grid_height, self.grid_width
        pred_output = pred_output.reshape(batch_size, self.output_window, H * W)

        # Truncate or pad to match original num_nodes
        if H * W > self.num_nodes:
            pred_output = pred_output[:, :, :self.num_nodes]
        elif H * W < self.num_nodes:
            padding = torch.zeros(batch_size, self.output_window, self.num_nodes - H * W, device=pred_output.device)
            pred_output = torch.cat([pred_output, padding], dim=2)

        # Add output dimension
        pred_output = pred_output.unsqueeze(-1)  # (N, T_out, N_nodes, 1)

        return pred_output

    def predict(self, batch):
        """
        Make predictions for a batch of data.

        Args:
            batch (dict): Input batch dictionary

        Returns:
            torch.Tensor: Predictions of shape (batch_size, output_window, num_nodes, output_dim)
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate the training loss.

        Args:
            batch (dict): Batch containing input X and target y

        Returns:
            torch.Tensor: Scalar loss value
        """
        y_true = batch['y'].to(self.device)
        y_pred = self.predict(batch)

        # Use inverse transform if scaler is available
        if self._scaler is not None:
            y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
            y_pred = self._scaler.inverse_transform(y_pred[..., :self.output_dim])

        # Use masked MAE loss
        return loss.masked_mae_torch(y_pred, y_true, null_val=0.0)
