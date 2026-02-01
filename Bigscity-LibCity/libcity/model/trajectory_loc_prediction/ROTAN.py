# coding: utf-8
"""
ROTAN: Rotation-based Temporal Attention Network for Next POI Recommendation

This module adapts the ROTAN model for LibCity framework.

Key Innovations:
1. Dual transformer architecture with rotation-based temporal attention
2. Multi-modal embeddings: User, POI, GPS, Time, Category
3. Rotation operations for temporal modeling (inspired by RotatE knowledge graph embeddings)
4. Quadkey-based GPS encoding for spatial representation

Original Architecture:
- Two parallel transformer encoders:
  - Encoder1: User + POI embeddings
  - Encoder2: POI + GPS embeddings
- Temporal rotation mechanism applied to both branches
- Final prediction: 0.7 * out1 + 0.3 * out2 weighting

Required data_feature keys:
- loc_size: Number of POI locations
- uid_size: Number of users
- cat_size: Number of categories (optional)
- quadkey_size: Number of unique quadkeys for GPS encoding (optional)

Required config parameters:
- poi_embed_dim: POI embedding dimension (default: 128)
- user_embed_dim: User embedding dimension (default: 128)
- gps_embed_dim: GPS embedding dimension (default: 64)
- time_embed_dim: Time embedding dimension (default: 32)
- cat_embed_dim: Category embedding dimension (default: 32)
- transformer_nhid: Hidden dim in TransformerEncoder (default: 1024)
- transformer_nlayers: Number of TransformerEncoderLayers (default: 2)
- transformer_nhead: Number of attention heads (default: 2)
- transformer_dropout: Dropout rate for transformer (default: 0.4)
- quadkey_n: Quadkey encoding level for GPS (default: 6)
- quadkey_len: Length of quadkey string (default: 25)
- hour_weight: Weight for hourly temporal rotation (default: 0.7)
- day_weight: Weight for daily temporal rotation (default: 0.3)
- encoder1_weight: Weight for encoder1 output (default: 0.7)
- encoder2_weight: Weight for encoder2 output (default: 0.3)
"""

import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, TransformerEncoder, TransformerEncoderLayer
from itertools import product
from libcity.model.abstract_model import AbstractModel


# ==============================================================================
# Quadkey Encoding Utilities
# ==============================================================================

EarthRadius = 6378137
MinLatitude = -85.05112878
MaxLatitude = 85.05112878
MinLongitude = -180
MaxLongitude = 180


def clip(n, minValue, maxValue):
    """Clip a value between min and max."""
    return min(max(n, minValue), maxValue)


def map_size(levelOfDetail):
    """Calculate map size at a given level of detail."""
    return 256 << levelOfDetail


def latlng2pxy(latitude, longitude, levelOfDetail):
    """Convert latitude/longitude to pixel coordinates."""
    latitude = clip(latitude, MinLatitude, MaxLatitude)
    longitude = clip(longitude, MinLongitude, MaxLongitude)

    x = (longitude + 180) / 360
    sinLatitude = math.sin(latitude * math.pi / 180)
    y = 0.5 - math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * math.pi)

    mapSize = map_size(levelOfDetail)
    pixelX = int(clip(x * mapSize + 0.5, 0, mapSize - 1))
    pixelY = int(clip(y * mapSize + 0.5, 0, mapSize - 1))
    return pixelX, pixelY


def pxy2txy(pixelX, pixelY):
    """Convert pixel coordinates to tile coordinates."""
    tileX = pixelX // 256
    tileY = pixelY // 256
    return tileX, tileY


def txy2quadkey(tileX, tileY, levelOfDetail):
    """Convert tile coordinates to quadkey string."""
    quadKey = []
    for i in range(levelOfDetail, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tileX & mask) != 0:
            digit += 1
        if (tileY & mask) != 0:
            digit += 2
        quadKey.append(str(digit))
    return ''.join(quadKey)


def latlng2quadkey(lat, lng, level):
    """Convert latitude/longitude to quadkey string."""
    pixelX, pixelY = latlng2pxy(lat, lng, level)
    tileX, tileY = pxy2txy(pixelX, pixelY)
    return txy2quadkey(tileX, tileY, level)


def get_all_permutations_dict(length):
    """Generate all possible n-gram permutations for quadkey encoding."""
    characters = ['0', '1', '2', '3']
    all_permutations = [''.join(p) for p in product(characters, repeat=length)]
    permutation_dict = dict(zip(all_permutations, range(len(all_permutations))))
    return permutation_dict


def get_ngrams_of_quadkey(quadkey, n, permutations_dict):
    """Extract n-grams from quadkey string."""
    from nltk import ngrams as nltk_ngrams
    region_quadkey_bigram = ' '.join([''.join(x) for x in nltk_ngrams(quadkey, n)])
    region_quadkey_bigram = region_quadkey_bigram.split()
    region_quadkey_bigram = [permutations_dict.get(each, 0) for each in region_quadkey_bigram]
    return region_quadkey_bigram


# ==============================================================================
# Utility Functions for Rotation Operations
# ==============================================================================

def rotate(head, relation, hidden, device):
    """
    Rotation-based transformation for temporal attention (non-batch version).

    Args:
        head: Input tensor of shape (seq_len, embed_dim)
        relation: Time embedding tensor of shape (seq_len, hidden)
        hidden: Hidden dimension for rotation
        device: Device to run the computation on

    Returns:
        Rotated embedding tensor of shape (seq_len, embed_dim)
    """
    pi = 3.14159265358979323846

    # Split head into real and imaginary parts
    re_head, im_head = torch.chunk(head, 2, dim=1)

    # Embedding range parameter for phase normalization
    embedding_range = nn.Parameter(
        torch.Tensor([(24.0 + 2.0) / hidden]),
        requires_grad=False
    ).to(device)

    # Convert relation to phase
    phase_relation = relation / (embedding_range / pi)

    # Compute rotation components
    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)

    # Apply rotation in complex space
    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation

    # Concatenate real and imaginary parts
    score = torch.cat([re_score, im_score], dim=1)
    return score


def rotate_batch(head, relation, hidden, device):
    """
    Rotation-based transformation for temporal attention (batch version).

    This function applies a rotation operation inspired by RotatE knowledge graph embeddings.
    The head embedding is rotated by the relation (time) embedding in the complex space.

    Args:
        head: Input tensor of shape (batch, seq_len, embed_dim)
        relation: Time embedding tensor of shape (batch, seq_len, hidden)
        hidden: Hidden dimension for rotation
        device: Device to run the computation on

    Returns:
        Rotated embedding tensor of shape (batch, seq_len, embed_dim)
    """
    pi = 3.14159265358979323846

    # Split head into real and imaginary parts
    re_head, im_head = torch.chunk(head, 2, dim=2)

    # Embedding range parameter for phase normalization
    embedding_range = nn.Parameter(
        torch.Tensor([(24.0 + 2.0) / hidden]),
        requires_grad=False
    ).to(device)

    # Convert relation to phase
    phase_relation = relation / (embedding_range / pi)

    # Compute rotation components
    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)

    # Apply rotation in complex space
    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation

    # Concatenate real and imaginary parts
    score = torch.cat([re_score, im_score], dim=2)
    return score


# ==============================================================================
# Embedding Modules
# ==============================================================================

class UserEmbeddings(nn.Module):
    """User embedding layer."""
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()
        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )

    def forward(self, user_idx):
        return self.user_embedding(user_idx)


class PoiEmbeddings(nn.Module):
    """POI embedding layer."""
    def __init__(self, num_pois, embedding_dim):
        super(PoiEmbeddings, self).__init__()
        self.poi_embedding = nn.Embedding(
            num_embeddings=num_pois,
            embedding_dim=embedding_dim
        )

    def forward(self, poi_idx):
        return self.poi_embedding(poi_idx)


class CategoryEmbeddings(nn.Module):
    """Category embedding layer."""
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()
        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
        )

    def forward(self, cat_idx):
        return self.cat_embedding(cat_idx)


class GPSEmbeddings(nn.Module):
    """GPS quadkey embedding layer."""
    def __init__(self, num_gps, embedding_dim):
        super(GPSEmbeddings, self).__init__()
        self.gps_embedding = nn.Embedding(
            num_embeddings=num_gps,
            embedding_dim=embedding_dim
        )

    def forward(self, gps_idx):
        return self.gps_embedding(gps_idx)


class TimeEmbeddings(nn.Module):
    """Time slot embedding layer."""
    def __init__(self, num_times, embedding_dim):
        super(TimeEmbeddings, self).__init__()
        self.time_embedding = nn.Embedding(
            num_embeddings=num_times,
            embedding_dim=embedding_dim
        )

    def forward(self, time_idx):
        return self.time_embedding(time_idx)


class FuseEmbeddings(nn.Module):
    """Fuse two embeddings with a linear layer."""
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, poi_embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), dim=-1))
        x = self.leaky_relu(x)
        return x


# ==============================================================================
# Time Encoding Modules
# ==============================================================================

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    """Time2Vec transformation function."""
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)


class SineActivation(nn.Module):
    """Sine activation for Time2Vec."""
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    """Cosine activation for Time2Vec."""
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class OriginTime2Vec(nn.Module):
    """
    Original Time2Vec implementation from the ROTAN paper.
    Uses sine or cosine activation to encode continuous time values.
    """
    def __init__(self, activation, out_dim):
        super(OriginTime2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        """
        Args:
            x: Time tensor of shape (seq_len,) or (batch, seq_len)

        Returns:
            Time encoding of shape (seq_len, out_dim) or (batch * seq_len, out_dim)
        """
        if x.dim() == 1:
            fea = x.reshape(-1, 1)  # Changed from .view() to .reshape() for non-contiguous tensors
            return self.l1(fea)
        elif x.dim() == 2:
            batch_size, seq_len = x.shape
            fea = x.reshape(-1, 1)  # Changed from .view() to .reshape() for non-contiguous tensors
            return self.l1(fea)
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}")


class CatTime2Vec(nn.Module):
    """Category-aware Time2Vec encoding."""
    def __init__(self, cat_num, out_dim):
        super(CatTime2Vec, self).__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(cat_num, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(cat_num, 1))
        self.w = nn.parameter.Parameter(torch.randn(cat_num, out_dim - 1))
        self.b = nn.parameter.Parameter(torch.randn(cat_num, out_dim - 1))

    def forward(self, cat_idx, norm_time):
        """
        Args:
            cat_idx: Category indices of shape (seq_len,)
            norm_time: Normalized time values of shape (seq_len,)

        Returns:
            Category-aware time encoding of shape (seq_len, out_dim)
        """
        w = self.w[cat_idx]
        b = self.b[cat_idx]
        w0 = self.w0[cat_idx]
        b0 = self.b0[cat_idx]

        norm_time_ = norm_time.view(-1, 1)
        v1 = torch.sin(norm_time_ * w + b)
        v2 = norm_time_ * w0 + b0
        return torch.cat((v1, v2), dim=-1)


class Time2Vec(nn.Module):
    """Time2Vec encoding with cosine basis."""
    def __init__(self, out_dim):
        super(Time2Vec, self).__init__()
        self.w = nn.parameter.Parameter(torch.randn(1, out_dim))
        self.b = nn.parameter.Parameter(torch.randn(1, out_dim))
        self.f = torch.cos

    def forward(self, time):
        """
        Args:
            time: 1D tensor of shape (seq_len,) or 2D tensor of shape (batch, seq_len)

        Returns:
            Time embeddings of shape (seq_len, time_dim) or (batch, seq_len, time_dim)
        """
        if time.dim() == 1:
            vec_time = time.view(-1, 1)
            out = torch.matmul(vec_time, self.w) + self.b
            v1 = out[:, 0].view(-1, 1)
            v2 = out[:, 1:]
            v2 = self.f(v2)
            return torch.cat((v1, v2), dim=-1)
        else:
            # Handle batched input
            batch_size, seq_len = time.shape
            vec_time = time.view(-1, 1)
            out = torch.matmul(vec_time, self.w) + self.b
            v1 = out[:, 0].view(-1, 1)
            v2 = out[:, 1:]
            v2 = self.f(v2)
            result = torch.cat((v1, v2), dim=-1)
            return result.view(batch_size, seq_len, -1)


class TimeEncoder(nn.Module):
    """
    Trainable encoder to map continuous time value into a low-dimension time vector.

    Reference: https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs
    """
    def __init__(self, embedding_dim):
        super(TimeEncoder, self).__init__()
        self.time_dim = embedding_dim
        self.expand_dim = self.time_dim
        self.use_linear_trans = True

        self.basis_freq = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float()
        )
        self.phase = nn.Parameter(torch.zeros(self.time_dim).float())
        if self.use_linear_trans:
            self.dense = nn.Linear(self.time_dim, self.expand_dim, bias=False)
            nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts):
        """
        Args:
            ts: Time tensor of shape (seq_len,) or (batch, seq_len)

        Returns:
            Time encoding of shape (seq_len, embed_dim) or (batch, seq_len, embed_dim)
        """
        original_shape = ts.shape
        if ts.dim() == 1:
            edge_len = ts.size().numel()
            ts = ts.view(edge_len, 1)
        else:
            batch_size, seq_len = ts.shape
            ts = ts.view(-1, 1)

        map_ts = ts * self.basis_freq.view(1, -1)
        map_ts += self.phase.view(1, -1)
        harmonic = torch.cos(map_ts)
        if self.use_linear_trans:
            harmonic = harmonic.type(self.dense.weight.dtype)
            harmonic = self.dense(harmonic)

        if len(original_shape) == 1:
            return harmonic
        else:
            return harmonic.view(batch_size, seq_len, -1)


# ==============================================================================
# GPS Encoder
# ==============================================================================

class GPSEncoder(nn.Module):
    """Transformer-based GPS encoder for quadkey sequences."""
    def __init__(self, embed_size, nhead, nhid, nlayers, dropout):
        super(GPSEncoder, self).__init__()
        encoder_layers = TransformerEncoderLayer(
            embed_size, nhead, nhid, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, src):
        """
        Args:
            src: GPS embeddings of shape (batch, seq_len, quadkey_len, embed_dim)
                 or (seq_len, quadkey_len, embed_dim)

        Returns:
            Aggregated GPS embeddings of shape (batch, seq_len, embed_dim)
            or (seq_len, embed_dim)
        """
        src = src * math.sqrt(self.embed_size)
        x = self.transformer_encoder(src)
        x = torch.mean(x, -2)
        return self.norm(x)


# ==============================================================================
# Positional Encoding
# ==============================================================================

class RightPositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    def __init__(self, d_model, dropout, max_len=600):
        super(RightPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)

        Returns:
            Position-encoded tensor of same shape
        """
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


# ==============================================================================
# MLP Module
# ==============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron."""
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, input_size)
        self.layer2 = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


# ==============================================================================
# Core Transformer Model
# ==============================================================================

class TransformerModel(nn.Module):
    """
    Dual Transformer architecture with rotation-based temporal attention.

    This model uses two parallel transformer encoders:
    1. Encoder1: Processes user + POI embeddings
    2. Encoder2: Processes POI + GPS embeddings

    Both encoders apply rotation-based temporal attention using hour and day time embeddings.
    Final prediction combines outputs with configurable weights.
    """
    def __init__(self, num_poi, embed_size, nhead, nhid, nlayers,
                 user_embed_dim, poi_embed_dim, gps_embed_dim, time_embed_dim,
                 device, dropout=0.4):
        super(TransformerModel, self).__init__()

        self.user_embed_dim = user_embed_dim
        self.poi_embed_dim = poi_embed_dim
        self.gps_embed_dim = gps_embed_dim
        self.time_embed_dim = time_embed_dim
        self.device = device

        # Compute user_time_dim for rotation operations
        self.user_time_dim = int(0.5 * (user_embed_dim + poi_embed_dim))

        # Positional encoders for both branches
        self.pos_encoder1 = RightPositionalEncoding(
            user_embed_dim + poi_embed_dim, dropout
        )
        self.pos_encoder2 = RightPositionalEncoding(
            poi_embed_dim + gps_embed_dim, dropout
        )

        # Transformer encoder 1: User + POI
        encoder_layers1 = TransformerEncoderLayer(
            user_embed_dim + poi_embed_dim, nhead, nhid, dropout, batch_first=True
        )
        self.transformer_encoder1 = TransformerEncoder(encoder_layers1, nlayers)

        # Transformer encoder 2: POI + GPS
        encoder_layers2 = TransformerEncoderLayer(
            poi_embed_dim + gps_embed_dim, nhead, nhid, dropout, batch_first=True
        )
        self.transformer_encoder2 = TransformerEncoder(encoder_layers2, nlayers)

        # Decoder layers for POI prediction
        # Decoder1: user_embed + poi_embed + poi_embed (from concatenation after rotation)
        self.decoder_poi1 = nn.Linear(
            user_embed_dim + 2 * poi_embed_dim, num_poi
        )
        # Decoder2: poi_embed + gps_embed + gps_embed (from concatenation after rotation)
        self.decoder_poi2 = nn.Linear(
            poi_embed_dim + 2 * gps_embed_dim, num_poi
        )

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        """Generate causal attention mask."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi1.bias.data.zero_()
        self.decoder_poi1.weight.data.uniform_(-initrange, initrange)
        self.decoder_poi2.bias.data.zero_()
        self.decoder_poi2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src1, src2, src_mask, target_hour, target_day,
                poi_embeds, gps_embeds, hour_weight=0.7, day_weight=0.3,
                encoder1_weight=0.7, encoder2_weight=0.3):
        """
        Forward pass through dual transformer with rotation-based temporal attention.

        Args:
            src1: User + POI embeddings (batch, seq_len, user_embed_dim + poi_embed_dim)
            src2: POI + GPS embeddings (batch, seq_len, poi_embed_dim + gps_embed_dim)
            src_mask: Attention mask (seq_len, seq_len)
            target_hour: Hour-based time embeddings (batch, seq_len, time_dim)
            target_day: Day-based time embeddings (batch, seq_len, time_dim)
            poi_embeds: POI embeddings for concatenation (batch, seq_len, poi_embed_dim)
            gps_embeds: GPS embeddings for concatenation (batch, seq_len, gps_embed_dim)
            hour_weight: Weight for hourly rotation (default: 0.7)
            day_weight: Weight for daily rotation (default: 0.3)
            encoder1_weight: Weight for encoder1 output (default: 0.7)
            encoder2_weight: Weight for encoder2 output (default: 0.3)

        Returns:
            POI prediction logits of shape (batch, seq_len, num_poi)
        """
        # Branch 1: User + POI transformer
        src1 = src1 * math.sqrt(self.user_embed_dim + self.poi_embed_dim)
        src1 = self.pos_encoder1(src1)
        src1 = self.transformer_encoder1(src1, src_mask)

        # Apply rotation-based temporal attention for branch 1
        src1_hour = rotate_batch(
            src1, target_hour[:, :, :self.user_time_dim],
            self.user_time_dim, self.device
        )
        src1_day = rotate_batch(
            src1, target_day[:, :, :self.user_time_dim],
            self.user_time_dim, self.device
        )

        # Combine hour and day rotations
        src1 = hour_weight * src1_hour + day_weight * src1_day
        src1 = torch.cat((src1, poi_embeds), dim=-1)

        out_poi_prob1 = self.decoder_poi1(src1)

        # Branch 2: POI + GPS transformer
        src2 = src2 * math.sqrt(self.poi_embed_dim + self.gps_embed_dim)
        src2 = self.pos_encoder2(src2)
        src2 = self.transformer_encoder2(src2, src_mask)

        # Apply rotation-based temporal attention for branch 2
        src2_hour = rotate_batch(
            src2, target_hour[:, :, self.user_time_dim:],
            2 * self.time_embed_dim, self.device
        )
        src2_day = rotate_batch(
            src2, target_day[:, :, self.user_time_dim:],
            2 * self.time_embed_dim, self.device
        )

        # Combine hour and day rotations
        src2 = hour_weight * src2_hour + day_weight * src2_day
        src2 = torch.cat((src2, gps_embeds), dim=-1)

        out_poi_prob2 = self.decoder_poi2(src2)

        # Combine both branches
        out_poi_prob = encoder1_weight * out_poi_prob1 + encoder2_weight * out_poi_prob2

        return out_poi_prob


# ==============================================================================
# Main ROTAN Model (LibCity Adapter)
# ==============================================================================

class ROTAN(AbstractModel):
    """
    ROTAN: Rotation-based Temporal Attention Network for Next POI Recommendation

    This model implements a dual transformer architecture with rotation-based
    temporal attention for next POI prediction. The core innovation is using
    rotation operations (inspired by RotatE) to model temporal patterns.

    Architecture:
    1. Multi-modal embeddings: User, POI, GPS (quadkey), Time
    2. Dual transformers: User+POI branch and POI+GPS branch
    3. Rotation-based temporal attention using hour and day encodings
    4. Weighted combination of both branches for final prediction

    The model uses separate time encoders for:
    - User-level hour encoding
    - User-level day encoding
    - POI-level hour encoding
    - POI-level day encoding
    - Target time encodings for rotation operations
    """

    def __init__(self, config, data_feature):
        super(ROTAN, self).__init__(config, data_feature)

        self.device = config.get('device', 'cpu')

        # Data dimensions from data_feature
        self.num_poi = data_feature.get('loc_size', 1000)
        self.num_users = data_feature.get('uid_size', 100)
        self.num_cats = data_feature.get('cat_size', 50)
        self.quadkey_size = data_feature.get('quadkey_size', 4096)

        # POI to category mapping (optional)
        self.poi_idx2cat_idx = data_feature.get('poi_idx2cat_idx', None)
        if self.poi_idx2cat_idx is None:
            self.poi_idx2cat_idx = {i: i % self.num_cats for i in range(self.num_poi)}

        # Model hyperparameters from config
        self.poi_embed_dim = config.get('poi_embed_dim', 128)
        self.user_embed_dim = config.get('user_embed_dim', 128)
        self.gps_embed_dim = config.get('gps_embed_dim', 128)
        self.time_embed_dim = config.get('time_embed_dim', 32)
        self.cat_embed_dim = config.get('cat_embed_dim', 32)

        # Transformer hyperparameters
        self.transformer_nhid = config.get('transformer_nhid', 1024)
        self.transformer_nlayers = config.get('transformer_nlayers', 2)
        self.transformer_nhead = config.get('transformer_nhead', 2)
        self.transformer_dropout = config.get('transformer_dropout', 0.4)

        # Quadkey parameters for GPS encoding
        self.quadkey_n = config.get('quadkey_n', 6)
        self.quadkey_len = config.get('quadkey_len', 25)

        # Weighting parameters
        self.hour_weight = config.get('hour_weight', 0.7)
        self.day_weight = config.get('day_weight', 0.3)
        self.encoder1_weight = config.get('encoder1_weight', 0.7)
        self.encoder2_weight = config.get('encoder2_weight', 0.3)

        # Pre-trained KG embeddings path (optional)
        self.kg_embed_path = config.get('kg_embed_path', None)
        self.use_pretrained_embeddings = config.get('use_pretrained_embeddings', False)

        # Initialize quadkey permutations dictionary
        self.permutations_dict = get_all_permutations_dict(self.quadkey_n)

        # Build all sub-models
        self._build_models()

        # Load pre-trained embeddings if available
        if self.kg_embed_path is not None and self.use_pretrained_embeddings:
            self._load_pretrained_embeddings()

    def _build_models(self):
        """Build all sub-models."""
        # Embedding layers
        self.user_embed_model = UserEmbeddings(self.num_users, self.user_embed_dim)
        self.poi_embed_model = PoiEmbeddings(self.num_poi, self.poi_embed_dim)
        self.cat_embed_model = CategoryEmbeddings(self.num_cats, self.cat_embed_dim)
        self.gps_embed_model = GPSEmbeddings(self.quadkey_size, self.gps_embed_dim)

        # Compute user_time_dim for rotation operations
        user_time_dim = int(0.5 * (self.user_embed_dim + self.poi_embed_dim))

        # Time encoders for multi-granularity time encoding (following original implementation)
        # User-level time encoders
        self.time_embed_model_user = OriginTime2Vec('sin', user_time_dim)
        self.time_embed_model_user_tgt = OriginTime2Vec('sin', user_time_dim)
        self.time_embed_model_user_day = OriginTime2Vec('sin', user_time_dim)
        self.time_embed_model_user_day_tgt = OriginTime2Vec('sin', user_time_dim)

        # POI-level time encoders
        self.time_embed_model_poi = OriginTime2Vec('sin', 2 * self.time_embed_dim)
        self.time_embed_model_poi_tgt = OriginTime2Vec('sin', 2 * self.time_embed_dim)
        self.time_embed_model_poi_day = OriginTime2Vec('sin', 2 * self.time_embed_dim)
        self.time_embed_model_poi_day_tgt = OriginTime2Vec('sin', 2 * self.time_embed_dim)

        # GPS encoder (Transformer-based for processing quadkey sequences)
        self.gps_encoder = GPSEncoder(
            embed_size=self.gps_embed_dim,
            nhead=1,
            nhid=2 * self.gps_embed_dim,
            nlayers=2,
            dropout=0.3
        )

        # Core transformer model
        self.transformer_model = TransformerModel(
            num_poi=self.num_poi,
            embed_size=self.poi_embed_dim,
            nhead=self.transformer_nhead,
            nhid=self.transformer_nhid,
            nlayers=self.transformer_nlayers,
            user_embed_dim=self.user_embed_dim,
            poi_embed_dim=self.poi_embed_dim,
            gps_embed_dim=self.gps_embed_dim,
            time_embed_dim=self.time_embed_dim,
            device=self.device,
            dropout=self.transformer_dropout
        )

        # Optional: Pre-trained POI embeddings (can be loaded from KG training)
        self.poi_pre_embeddings = None

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def _load_pretrained_embeddings(self):
        """Load pre-trained KG embeddings if available."""
        try:
            import os
            if os.path.exists(self.kg_embed_path):
                kg_embeds = torch.load(self.kg_embed_path, map_location=self.device)
                if 'poi_embeddings' in kg_embeds:
                    self.poi_embed_model.poi_embedding.weight.data = kg_embeds['poi_embeddings']
                    print("Loaded pre-trained POI embeddings")
                if 'user_embeddings' in kg_embeds:
                    self.user_embed_model.user_embedding.weight.data = kg_embeds['user_embeddings']
                    print("Loaded pre-trained user embeddings")
                # Handle numpy array format
                if isinstance(kg_embeds, np.ndarray):
                    self.poi_pre_embeddings = nn.Parameter(
                        torch.tensor(kg_embeds, dtype=torch.float, requires_grad=True).to(self.device)
                    )
                    print("Loaded pre-trained POI embeddings from numpy array")
        except Exception as e:
            print(f"Warning: Could not load pre-trained embeddings: {e}")

    def _create_embeddings(self, batch):
        """
        Create all required embeddings from batch data.

        Expected batch keys:
        - 'current_loc': (batch_size, seq_len) - POI indices
        - 'current_tim': (batch_size, seq_len) - normalized time values
        - 'uid': (batch_size,) or (batch_size, 1) - user indices
        - 'current_hour': (batch_size, seq_len) - hour time slots (optional)
        - 'current_day': (batch_size, seq_len) - day time slots (optional)
        - 'current_gps': (batch_size, seq_len, quadkey_ngram_len) - GPS quadkey indices (optional)
        - 'target_tim': (batch_size, seq_len) - target normalized time (optional)
        - 'target_day': (batch_size, seq_len) - target day time (optional)

        Returns:
            Dictionary containing all computed embeddings
        """
        current_loc = batch['current_loc']
        current_tim = batch['current_tim']
        batch_size, seq_len = current_loc.shape

        # Handle user index
        if 'uid' in batch.data:
            uid = batch['uid']
            if uid.dim() == 1:
                uid = uid.unsqueeze(1)
            uid = uid.expand(-1, seq_len)
        else:
            uid = torch.zeros(batch_size, seq_len, dtype=torch.long, device=current_loc.device)

        # Clamp indices to valid ranges
        current_loc_clamped = torch.clamp(current_loc, 0, self.num_poi - 1)
        uid_clamped = torch.clamp(uid, 0, self.num_users - 1)

        # Get POI embeddings (use pre-trained if available)
        if self.poi_pre_embeddings is not None:
            poi_embeds = torch.index_select(self.poi_pre_embeddings, 0, current_loc_clamped.view(-1))
            poi_embeds = poi_embeds.view(batch_size, seq_len, -1)
        else:
            poi_embeds = self.poi_embed_model(current_loc_clamped)

        # Get user embeddings and expand to sequence length
        user_embeds_base = self.user_embed_model(uid_clamped[:, 0])  # (batch_size, user_embed_dim)
        user_embeds = user_embeds_base.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, user_embed_dim)

        # Get GPS embeddings
        if 'current_gps' in batch.data:
            gps_idx = batch['current_gps']
            if gps_idx.dim() == 3:
                # (batch_size, seq_len, quadkey_ngram_len)
                gps_idx_clamped = torch.clamp(gps_idx, 0, self.quadkey_size - 1)
                gps_embeds_raw = self.gps_embed_model(gps_idx_clamped)  # (batch, seq, ngram, embed)
                gps_embeds = self.gps_encoder(gps_embeds_raw)  # (batch, seq, embed)
            else:
                # (batch_size, seq_len) - single quadkey index per position
                gps_idx_clamped = torch.clamp(gps_idx, 0, self.quadkey_size - 1)
                gps_embeds = self.gps_embed_model(gps_idx_clamped)
        else:
            # Use zero embeddings if GPS data not available
            gps_embeds = torch.zeros(
                batch_size, seq_len, self.gps_embed_dim,
                device=current_loc.device
            )

        # Get time embeddings (normalized to [0, 1])
        hour_time = current_tim.float()  # Assuming current_tim is normalized hour
        if 'current_day' in batch.data:
            day_time = batch['current_day'].float()
        else:
            day_time = torch.zeros_like(hour_time)

        # Get target time embeddings (for rotation)
        if 'target_tim' in batch.data:
            target_hour_time = batch['target_tim'].float()
            # Handle scalar target_tim: expand to sequence length
            if target_hour_time.dim() == 1:
                target_hour_time = target_hour_time.unsqueeze(1).expand(-1, seq_len).contiguous()
        else:
            target_hour_time = hour_time

        if 'target_day' in batch.data:
            target_day_time = batch['target_day'].float()
            # Handle scalar target_day: expand to sequence length
            if target_day_time.dim() == 1:
                target_day_time = target_day_time.unsqueeze(1).expand(-1, seq_len).contiguous()
        else:
            target_day_time = day_time

        # Compute user_time_dim
        user_time_dim = int(0.5 * (self.user_embed_dim + self.poi_embed_dim))

        # Encode user times
        user_times = self.time_embed_model_user(hour_time)  # (batch*seq, user_time_dim)
        user_times = user_times.view(batch_size, seq_len, -1)

        user_day_times = self.time_embed_model_user_day(day_time)
        user_day_times = user_day_times.view(batch_size, seq_len, -1)

        user_next_times = self.time_embed_model_user_tgt(target_hour_time)
        user_next_times = user_next_times.view(batch_size, seq_len, -1)

        user_next_day_times = self.time_embed_model_user_day_tgt(target_day_time)
        user_next_day_times = user_next_day_times.view(batch_size, seq_len, -1)

        # Encode POI times
        poi_times = self.time_embed_model_poi(hour_time)
        poi_times = poi_times.view(batch_size, seq_len, -1)

        poi_day_times = self.time_embed_model_poi_day(day_time)
        poi_day_times = poi_day_times.view(batch_size, seq_len, -1)

        poi_next_times = self.time_embed_model_poi_tgt(target_hour_time)
        poi_next_times = poi_next_times.view(batch_size, seq_len, -1)

        poi_next_day_times = self.time_embed_model_poi_day_tgt(target_day_time)
        poi_next_day_times = poi_next_day_times.view(batch_size, seq_len, -1)

        # Apply rotation to user embeddings (user + poi)
        user_poi_embeds = torch.cat([user_embeds, poi_embeds], dim=-1)

        user_rotate_hour = rotate_batch(
            user_poi_embeds, user_times, user_time_dim, self.device
        )
        user_rotate_day = rotate_batch(
            user_poi_embeds, user_day_times, user_time_dim, self.device
        )
        user_rotate = self.hour_weight * user_rotate_hour + self.day_weight * user_rotate_day

        # Apply rotation to POI embeddings (poi + gps)
        poi_gps_embeds = torch.cat([poi_embeds, gps_embeds], dim=-1)

        poi_rotate_hour = rotate_batch(
            poi_gps_embeds, poi_times, 2 * self.time_embed_dim, self.device
        )
        poi_rotate_day = rotate_batch(
            poi_gps_embeds, poi_day_times, 2 * self.time_embed_dim, self.device
        )
        poi_rotate = self.hour_weight * poi_rotate_hour + self.day_weight * poi_rotate_day

        # Target time embeddings for transformer
        target_hour = torch.cat([user_next_times, poi_next_times], dim=-1)
        target_day = torch.cat([user_next_day_times, poi_next_day_times], dim=-1)

        return {
            'src1': user_rotate,
            'src2': poi_rotate,
            'target_hour': target_hour,
            'target_day': target_day,
            'poi_embeds': poi_embeds,
            'gps_embeds': gps_embeds,
            'batch_size': batch_size,
            'seq_len': seq_len
        }

    def forward(self, batch):
        """
        Forward pass through the ROTAN model.

        Args:
            batch: Dictionary containing:
                - 'current_loc': (batch_size, seq_len) - POI indices
                - 'current_tim': (batch_size, seq_len) - normalized time values
                - 'uid': (batch_size,) - user indices
                - Other optional fields (current_hour, current_day, current_gps)

        Returns:
            POI prediction logits of shape (batch_size, seq_len, num_poi)
        """
        # Create all embeddings
        embeds = self._create_embeddings(batch)

        # Generate attention mask
        src_mask = self.transformer_model.generate_square_subsequent_mask(
            embeds['seq_len']
        ).to(self.device)

        # Forward through transformer
        out_poi = self.transformer_model(
            src1=embeds['src1'],
            src2=embeds['src2'],
            src_mask=src_mask,
            target_hour=embeds['target_hour'],
            target_day=embeds['target_day'],
            poi_embeds=embeds['poi_embeds'],
            gps_embeds=embeds['gps_embeds'],
            hour_weight=self.hour_weight,
            day_weight=self.day_weight,
            encoder1_weight=self.encoder1_weight,
            encoder2_weight=self.encoder2_weight
        )

        return out_poi

    def predict(self, batch):
        """
        Predict next POI for each position in the sequence.

        Args:
            batch: Dictionary containing input data

        Returns:
            torch.Tensor: POI prediction scores (batch_size, num_poi)
                          Returns predictions for the last position of each sequence.
        """
        out_poi = self.forward(batch)

        # Get predictions for the last position in each sequence
        last_pred = out_poi[:, -1, :]  # (batch_size, num_poi)

        return last_pred

    def calculate_loss(self, batch):
        """
        Calculate cross-entropy loss for POI prediction.

        Args:
            batch: Dictionary containing:
                - 'current_loc': (batch_size, seq_len) - input POI indices
                - 'current_tim': (batch_size, seq_len) - input time values
                - 'uid': (batch_size,) - user indices
                - 'target': (batch_size,) or (batch_size, seq_len) - target POI indices

        Returns:
            torch.Tensor: Cross-entropy loss (scalar)
        """
        out_poi = self.forward(batch)
        batch_size, seq_len, _ = out_poi.shape

        # Get target labels
        target = batch['target']

        # Handle single target (next POI prediction)
        if target.dim() == 1:
            # Only use last position prediction
            out_poi_last = out_poi[:, -1, :]  # (batch_size, num_poi)
            loss = self.criterion(out_poi_last, target)
        else:
            # Sequence target (for sequence-to-sequence prediction)
            # out_poi: (batch_size, seq_len, num_poi)
            # target: (batch_size, seq_len)
            loss = self.criterion(out_poi.transpose(1, 2), target)

        return loss
