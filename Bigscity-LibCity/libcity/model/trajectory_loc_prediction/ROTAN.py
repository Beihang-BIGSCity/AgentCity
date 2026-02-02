"""
ROTAN: Rotation-based Temporal Attention Network for Next POI Recommendation

This model is adapted from the original ROTAN implementation.

Key Components:
1. Dual-stream Transformer architecture (user-POI stream + POI-GPS stream)
2. Rotation-based temporal encoding using complex-valued operations
3. Multiple time granularity encoding (hour, day)
4. Quadkey-based GPS embedding for spatial encoding

Adaptations for LibCity:
- Unified all sub-models into a single ROTAN class
- Adapted batch input format to LibCity's trajectory batch dictionary
- Implemented predict() and calculate_loss() methods following LibCity conventions
- Extracted hyperparameters from config dict
- Extracted data features from data_feature dict

Original files:
- repos/ROTAN/old_model.py (TransformerModel, embedding modules)
- repos/ROTAN/utils.py (rotate, rotate_batch functions)
- repos/ROTAN/quad_key_encoder.py (spatial encoding utilities)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np

from libcity.model.abstract_model import AbstractModel


# ========================== Rotation Functions ==========================

def rotate(head, relation, hidden, device):
    """
    Rotation operation for temporal encoding (single sample).

    Uses complex-valued rotation inspired by RotatE knowledge graph embedding.
    Treats the head embedding as a complex number and rotates it by the phase
    derived from the relation (time) embedding.

    Args:
        head: Input embedding tensor of shape (seq_len, hidden_dim)
        relation: Relation/time embedding tensor of shape (seq_len, hidden_dim/2)
        hidden: Hidden dimension size (hidden_dim)
        device: Computing device

    Returns:
        Rotated embedding tensor of shape (seq_len, hidden_dim)
    """
    pi = 3.14159265358979323846

    # Split head into real and imaginary parts
    re_head, im_head = torch.chunk(head, 2, dim=1)

    # Compute embedding range for phase normalization
    embedding_range = nn.Parameter(
        torch.Tensor([(24.0 + 2.0) / hidden]),
        requires_grad=False
    ).to(device)

    # Convert relation to phase
    phase_relation = relation / (embedding_range / pi)

    # Compute rotation matrices
    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)

    # Apply rotation: (re_head + i*im_head) * (re_relation + i*im_relation)
    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation

    # Concatenate real and imaginary parts
    score = torch.cat([re_score, im_score], dim=1)
    return score


def rotate_batch(head, relation, hidden, device):
    """
    Batch rotation operation for temporal encoding.

    Uses complex-valued rotation inspired by RotatE knowledge graph embedding.
    Treats the head embedding as a complex number and rotates it by the phase
    derived from the relation (time) embedding.

    Args:
        head: Input embedding tensor of shape (batch_size, seq_len, hidden_dim)
        relation: Relation/time embedding tensor of shape (batch_size, seq_len, hidden_dim/2)
        hidden: Hidden dimension size (hidden_dim)
        device: Computing device

    Returns:
        Rotated embedding tensor of shape (batch_size, seq_len, hidden_dim)
    """
    pi = 3.14159265358979323846

    # Split head into real and imaginary parts along the last dimension
    re_head, im_head = torch.chunk(head, 2, dim=2)

    # Compute embedding range for phase normalization
    embedding_range = nn.Parameter(
        torch.Tensor([(24.0 + 2.0) / hidden]),
        requires_grad=False
    ).to(device)

    # Convert relation to phase
    phase_relation = relation / (embedding_range / pi)

    # Compute rotation matrices
    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)

    # Apply rotation: (re_head + i*im_head) * (re_relation + i*im_relation)
    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation

    # Concatenate real and imaginary parts
    score = torch.cat([re_score, im_score], dim=2)
    return score


# ========================== Quadkey Encoding ==========================

class QuadKeyEncoder:
    """
    Quadkey encoder for converting GPS coordinates to spatial tokens.

    Uses Microsoft's Bing Maps tile system for hierarchical spatial encoding.
    """

    EARTH_RADIUS = 6378137
    MIN_LATITUDE = -85.05112878
    MAX_LATITUDE = 85.05112878
    MIN_LONGITUDE = -180
    MAX_LONGITUDE = 180

    @staticmethod
    def clip(n, min_value, max_value):
        """Clip value to range."""
        return min(max(n, min_value), max_value)

    @staticmethod
    def map_size(level_of_detail):
        """Get map size at given level of detail."""
        return 256 << level_of_detail

    @staticmethod
    def latlng_to_pixel(latitude, longitude, level_of_detail):
        """Convert lat/lng to pixel coordinates."""
        latitude = QuadKeyEncoder.clip(latitude, QuadKeyEncoder.MIN_LATITUDE, QuadKeyEncoder.MAX_LATITUDE)
        longitude = QuadKeyEncoder.clip(longitude, QuadKeyEncoder.MIN_LONGITUDE, QuadKeyEncoder.MAX_LONGITUDE)

        x = (longitude + 180) / 360
        sin_latitude = math.sin(latitude * math.pi / 180)
        y = 0.5 - math.log((1 + sin_latitude) / (1 - sin_latitude)) / (4 * math.pi)

        map_size = QuadKeyEncoder.map_size(level_of_detail)
        pixel_x = int(QuadKeyEncoder.clip(x * map_size + 0.5, 0, map_size - 1))
        pixel_y = int(QuadKeyEncoder.clip(y * map_size + 0.5, 0, map_size - 1))
        return pixel_x, pixel_y

    @staticmethod
    def pixel_to_tile(pixel_x, pixel_y):
        """Convert pixel coordinates to tile coordinates."""
        tile_x = pixel_x // 256
        tile_y = pixel_y // 256
        return tile_x, tile_y

    @staticmethod
    def tile_to_quadkey(tile_x, tile_y, level_of_detail):
        """Convert tile coordinates to quadkey string."""
        quad_key = []
        for i in range(level_of_detail, 0, -1):
            digit = 0
            mask = 1 << (i - 1)
            if (tile_x & mask) != 0:
                digit += 1
            if (tile_y & mask) != 0:
                digit += 2
            quad_key.append(str(digit))
        return ''.join(quad_key)

    @staticmethod
    def latlng_to_quadkey(lat, lng, level):
        """Convert lat/lng to quadkey."""
        pixel_x, pixel_y = QuadKeyEncoder.latlng_to_pixel(lat, lng, level)
        tile_x, tile_y = QuadKeyEncoder.pixel_to_tile(pixel_x, pixel_y)
        return QuadKeyEncoder.tile_to_quadkey(tile_x, tile_y, level)


# ========================== Embedding Modules ==========================

class UserEmbeddings(nn.Module):
    """User embedding layer."""

    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()
        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )

    def forward(self, user_idx):
        embed = self.user_embedding(user_idx)
        return embed


class PoiEmbeddings(nn.Module):
    """POI (location) embedding layer."""

    def __init__(self, num_pois, embedding_dim):
        super(PoiEmbeddings, self).__init__()
        self.poi_embedding = nn.Embedding(
            num_embeddings=num_pois,
            embedding_dim=embedding_dim
        )

    def forward(self, poi_idx):
        embed = self.poi_embedding(poi_idx)
        return embed


class GPSEmbeddings(nn.Module):
    """GPS/Quadkey spatial embedding layer."""

    def __init__(self, num_gps, embedding_dim):
        super(GPSEmbeddings, self).__init__()
        self.gps_embedding = nn.Embedding(
            num_embeddings=num_gps,
            embedding_dim=embedding_dim
        )

    def forward(self, gps_idx):
        embed = self.gps_embedding(gps_idx)
        return embed


class CategoryEmbeddings(nn.Module):
    """Category embedding layer."""

    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()
        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
        )

    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed


class FuseEmbeddings(nn.Module):
    """Fuse two embeddings (e.g., user+POI or POI+GPS)."""

    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, poi_embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), dim=-1))
        x = self.leaky_relu(x)
        return x


# ========================== Time Encoding Modules ==========================

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    """Time2Vec transformation function."""
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)


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
    """Original Time2Vec encoding without user bias."""

    def __init__(self, activation, out_dim):
        super(OriginTime2Vec, self).__init__()
        if activation in ("sin", "sine"):
            self.l1 = SineActivation(1, out_dim)
        elif activation in ("cos", "cosine"):
            self.l1 = CosineActivation(1, out_dim)
        else:
            raise ValueError(f"Unknown activation: {activation}. Expected 'sin', 'sine', 'cos', or 'cosine'.")

    def forward(self, x):
        fea = x.view(-1, 1)
        return self.l1(fea)


class Time2Vec(nn.Module):
    """Time2Vec encoding module."""

    def __init__(self, out_dim):
        super(Time2Vec, self).__init__()
        self.w = nn.parameter.Parameter(torch.randn(1, out_dim))
        self.b = nn.parameter.Parameter(torch.randn(1, out_dim))
        self.f = torch.cos

    def forward(self, time):
        """
        Args:
            time (1d tensor): shape is seq_len

        Returns:
            time embeddings (2d tensor): shape is seq_len * time_dim
        """
        vec_time = time.view(-1, 1)
        out = torch.matmul(vec_time, self.w) + self.b
        v1 = out[:, 0].view(-1, 1)
        v2 = out[:, 1:]
        v2 = self.f(v2)
        return torch.cat((v1, v2), dim=-1)


class CatTime2Vec(nn.Module):
    """Category-aware Time2Vec encoding."""

    def __init__(self, cat_num, out_dim):
        super(CatTime2Vec, self).__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(cat_num, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(cat_num, 1))
        self.w = nn.parameter.Parameter(torch.randn(cat_num, out_dim - 1))
        self.b = nn.parameter.Parameter(torch.randn(cat_num, out_dim - 1))

    def forward(self, cat_idx, norm_time):
        w = self.w[cat_idx]
        b = self.b[cat_idx]
        w0 = self.w0[cat_idx]
        b0 = self.b0[cat_idx]

        norm_time_ = norm_time.view(-1, 1)
        v1 = torch.sin(norm_time_ * w + b)
        v2 = norm_time_ * w0 + b0
        return torch.cat((v1, v2), dim=-1)


class TimeEncoder(nn.Module):
    """
    Trainable encoder to map continuous time value into a low-dimension time vector.
    Reference: Inductive representation learning on temporal graphs
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
        if ts.dim() == 1:
            dim = 1
            edge_len = ts.size().numel()
        else:
            edge_len, dim = ts.size()
        ts = ts.view(edge_len, dim)
        map_ts = ts * self.basis_freq.view(1, -1)
        map_ts += self.phase.view(1, -1)
        harmonic = torch.cos(map_ts)
        if self.use_linear_trans:
            harmonic = harmonic.type(self.dense.weight.dtype)
            harmonic = self.dense(harmonic)
        return harmonic


# ========================== Positional Encoding ==========================

class RightPositionalEncoding(nn.Module):
    """Standard positional encoding for Transformer."""

    def __init__(self, d_model, dropout, max_len=600):
        super(RightPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Initialize positional encoding
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
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
        """
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


# ========================== Dual-Stream Transformer ==========================

class DualStreamTransformer(nn.Module):
    """
    Dual-stream Transformer for ROTAN.

    Stream 1: User + POI embeddings
    Stream 2: POI + GPS embeddings

    Both streams use rotation-based temporal encoding with hour and day granularities.
    """

    def __init__(self, num_poi, user_embed_dim, poi_embed_dim, gps_embed_dim,
                 time_embed_dim, nhead, nhid, nlayers, device, dropout=0.5):
        super(DualStreamTransformer, self).__init__()

        self.device = device
        self.user_embed_dim = user_embed_dim
        self.poi_embed_dim = poi_embed_dim
        self.gps_embed_dim = gps_embed_dim
        self.time_embed_dim = time_embed_dim

        # Calculate stream dimensions
        self.stream1_dim = user_embed_dim + poi_embed_dim
        self.stream2_dim = poi_embed_dim + gps_embed_dim

        # User time dimension (for rotation)
        self.user_time_dim = int(0.5 * (user_embed_dim + poi_embed_dim))

        # Positional encoding for both streams
        self.pos_encoder1 = RightPositionalEncoding(self.stream1_dim, dropout)
        self.pos_encoder2 = RightPositionalEncoding(self.stream2_dim, dropout)

        # Transformer encoders for both streams
        encoder_layers1 = TransformerEncoderLayer(
            self.stream1_dim, nhead, nhid, dropout, batch_first=True
        )
        self.transformer_encoder1 = TransformerEncoder(encoder_layers1, nlayers)

        encoder_layers2 = TransformerEncoderLayer(
            self.stream2_dim, nhead, nhid, dropout, batch_first=True
        )
        self.transformer_encoder2 = TransformerEncoder(encoder_layers2, nlayers)

        # Decoder layers
        # Stream 1: includes rotated output + poi_embeds
        self.decoder_poi1 = nn.Linear(self.stream1_dim + poi_embed_dim, num_poi)
        # Stream 2: includes rotated output + gps_embeds
        self.decoder_poi2 = nn.Linear(self.stream2_dim + gps_embed_dim, num_poi)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for self-attention."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi1.bias.data.zero_()
        self.decoder_poi1.weight.data.uniform_(-initrange, initrange)
        self.decoder_poi2.bias.data.zero_()
        self.decoder_poi2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src1, src2, src_mask, target_hour, target_day, poi_embeds, gps_embeds):
        """
        Forward pass through dual-stream transformer.

        Args:
            src1: Stream 1 input (user + POI) of shape (batch, seq_len, stream1_dim)
            src2: Stream 2 input (POI + GPS) of shape (batch, seq_len, stream2_dim)
            src_mask: Causal attention mask
            target_hour: Hour-level time embeddings (batch, seq_len, time_dim)
            target_day: Day-level time embeddings (batch, seq_len, time_dim)
            poi_embeds: POI embeddings for concatenation (batch, seq_len, poi_embed_dim)
            gps_embeds: GPS embeddings for concatenation (batch, seq_len, gps_embed_dim)

        Returns:
            out_poi_prob: Combined POI prediction logits (batch, seq_len, num_poi)
        """
        # Stream 1: User + POI
        src1 = src1 * math.sqrt(self.stream1_dim)
        src1 = self.pos_encoder1(src1)
        src1 = self.transformer_encoder1(src1, src_mask)

        # Apply rotation-based temporal encoding for stream 1
        src1_hour = rotate_batch(
            src1, target_hour[:, :, :self.user_time_dim],
            self.user_time_dim, self.device
        )
        src1_day = rotate_batch(
            src1, target_day[:, :, :self.user_time_dim],
            self.user_time_dim, self.device
        )

        # Combine hour and day with 0.7/0.3 weighting
        src1 = 0.7 * src1_hour + 0.3 * src1_day
        src1 = torch.cat((src1, poi_embeds), dim=-1)

        out_poi_prob1 = self.decoder_poi1(src1)

        # Stream 2: POI + GPS
        src2 = src2 * math.sqrt(self.stream2_dim)
        src2 = self.pos_encoder2(src2)
        src2 = self.transformer_encoder2(src2, src_mask)

        # Apply rotation-based temporal encoding for stream 2
        src2_hour = rotate_batch(
            src2, target_hour[:, :, self.user_time_dim:],
            2 * self.time_embed_dim, self.device
        )
        src2_day = rotate_batch(
            src2, target_day[:, :, self.user_time_dim:],
            2 * self.time_embed_dim, self.device
        )

        # Combine hour and day with 0.7/0.3 weighting
        src2 = 0.7 * src2_hour + 0.3 * src2_day
        src2 = torch.cat((src2, gps_embeds), dim=-1)

        out_poi_prob2 = self.decoder_poi2(src2)

        # Combine predictions from both streams
        out_poi_prob = 0.7 * out_poi_prob1 + 0.3 * out_poi_prob2

        return out_poi_prob


# ========================== Main ROTAN Model ==========================

class ROTAN(AbstractModel):
    """
    ROTAN: Rotation-based Temporal Attention Network for Next POI Recommendation.

    This model uses a dual-stream Transformer architecture with rotation-based
    temporal encoding for trajectory-based POI prediction.

    Key Features:
    1. Dual-stream processing: User-POI stream and POI-GPS stream
    2. Rotation-based temporal encoding using complex-valued operations
    3. Multiple time granularities (hour and day)
    4. Quadkey-based spatial encoding for GPS coordinates

    Args:
        config (dict): Configuration dictionary containing model hyperparameters
        data_feature (dict): Data features including vocab sizes, etc.

    Required config parameters:
        - user_embed_dim: User embedding dimension (default: 128)
        - poi_embed_dim: POI embedding dimension (default: 128)
        - gps_embed_dim: GPS embedding dimension (default: 64)
        - time_embed_dim: Time embedding dimension (default: 64)
        - transformer_nhid: Hidden dim in TransformerEncoder (default: 1024)
        - transformer_nlayers: Number of TransformerEncoderLayers (default: 2)
        - transformer_nhead: Number of attention heads (default: 2)
        - transformer_dropout: Dropout rate for transformer (default: 0.3)
        - time_activation: Time2Vec activation function (default: "cos")
        - quadkey_level: Level of detail for quadkey encoding (default: 17)
        - stream1_weight: Weight for stream 1 predictions (default: 0.7)
        - stream2_weight: Weight for stream 2 predictions (default: 0.3)

    Required data_feature:
        - loc_size: Number of POI locations
        - uid_size: Number of users
        - num_categories: Number of POI categories (optional)
        - num_gps: Number of GPS quadkey tokens (optional, default: 4^quadkey_level)
    """

    def __init__(self, config, data_feature):
        super(ROTAN, self).__init__(config, data_feature)

        self.device = config.get('device', 'cpu')

        # Data dimensions from data_feature
        self.num_pois = data_feature.get('loc_size', 1000)
        self.num_users = data_feature.get('uid_size', 100)
        self.num_categories = data_feature.get('num_categories',
                                               data_feature.get('tim_size', 48))

        # Model hyperparameters from config
        self.user_embed_dim = config.get('user_embed_dim', 128)
        self.poi_embed_dim = config.get('poi_embed_dim', 128)
        self.gps_embed_dim = config.get('gps_embed_dim', 64)
        self.time_embed_dim = config.get('time_embed_dim', 64)
        self.transformer_nhid = config.get('transformer_nhid', 1024)
        self.transformer_nlayers = config.get('transformer_nlayers', 2)
        self.transformer_nhead = config.get('transformer_nhead', 2)
        self.transformer_dropout = config.get('transformer_dropout', 0.3)
        self.time_activation = config.get('time_activation', 'cos')
        self.quadkey_level = config.get('quadkey_level', 17)
        self.stream1_weight = config.get('stream1_weight', 0.7)
        self.stream2_weight = config.get('stream2_weight', 0.3)

        # GPS vocabulary size (4^level for quadkey)
        self.num_gps = data_feature.get('num_gps', 4 ** min(self.quadkey_level, 6))

        # Calculate time dimension for rotation
        self.user_time_dim = int(0.5 * (self.user_embed_dim + self.poi_embed_dim))
        self.total_time_dim = self.user_time_dim + 2 * self.time_embed_dim

        # POI to category mapping
        self.poi_to_cat = data_feature.get('poi_to_cat', None)
        if self.poi_to_cat is None:
            self.poi_to_cat = {i: 0 for i in range(self.num_pois)}

        # Build model components
        self._build_model()

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def _build_model(self):
        """Build all model components."""

        # Embedding layers
        self.user_embed = UserEmbeddings(self.num_users, self.user_embed_dim)
        self.poi_embed = PoiEmbeddings(self.num_pois, self.poi_embed_dim)
        self.gps_embed = GPSEmbeddings(self.num_gps, self.gps_embed_dim)
        self.cat_embed = CategoryEmbeddings(self.num_categories, self.poi_embed_dim)

        # Fusion layers
        self.fuse_user_poi = FuseEmbeddings(self.user_embed_dim, self.poi_embed_dim)
        self.fuse_poi_gps = FuseEmbeddings(self.poi_embed_dim, self.gps_embed_dim)

        # Time encoders for hour and day granularities
        self.time_hour_encoder = OriginTime2Vec(self.time_activation, self.total_time_dim)
        self.time_day_encoder = OriginTime2Vec(self.time_activation, self.total_time_dim)

        # Alternative Time2Vec for simpler encoding
        self.time_encoder = TimeEncoder(self.time_embed_dim)

        # Dual-stream Transformer
        self.transformer = DualStreamTransformer(
            num_poi=self.num_pois,
            user_embed_dim=self.user_embed_dim,
            poi_embed_dim=self.poi_embed_dim,
            gps_embed_dim=self.gps_embed_dim,
            time_embed_dim=self.time_embed_dim,
            nhead=self.transformer_nhead,
            nhid=self.transformer_nhid,
            nlayers=self.transformer_nlayers,
            device=self.device,
            dropout=self.transformer_dropout
        )

    def _get_category_for_poi(self, poi_idx):
        """Get category index for a POI."""
        if isinstance(poi_idx, torch.Tensor):
            poi_idx = poi_idx.item() if poi_idx.dim() == 0 else poi_idx.tolist()
        if isinstance(poi_idx, list):
            return [self.poi_to_cat.get(p % self.num_pois, 0) for p in poi_idx]
        return self.poi_to_cat.get(poi_idx % self.num_pois, 0)

    def _compute_gps_idx(self, lat, lng):
        """Compute GPS quadkey index from coordinates."""
        try:
            quadkey = QuadKeyEncoder.latlng_to_quadkey(lat, lng, self.quadkey_level)
            # Convert quadkey to index (simple hash)
            idx = 0
            for char in quadkey:
                idx = idx * 4 + int(char)
            return idx % self.num_gps
        except Exception:
            return 0

    def _normalize_hour(self, hour):
        """Normalize hour to [0, 1]."""
        return hour / 24.0

    def _normalize_day(self, day_of_week):
        """Normalize day of week to [0, 1]."""
        return day_of_week / 7.0

    def forward(self, batch):
        """
        Forward pass.

        Args:
            batch: LibCity Batch object containing trajectory data.
                Required keys:
                    - 'current_loc': Location/POI sequences (batch_size, seq_len)
                Optional keys:
                    - 'uid': User IDs (batch_size,)
                    - 'current_tim': Time sequences (batch_size, seq_len)
                    - 'current_coord': GPS coordinates (batch_size, seq_len, 2)

        Returns:
            y_pred: POI prediction logits (batch_size, seq_len, num_pois)
            seq_lens: Original sequence lengths before padding
        """
        # Extract data from batch - use direct key access for LibCity Batch
        loc_seq = batch['current_loc']  # (batch_size, seq_len)

        # Get time sequence if available
        if 'current_tim' in batch.data:
            time_seq = batch['current_tim']  # (batch_size, seq_len)
        else:
            time_seq = None

        # Get user IDs if available
        if 'uid' in batch.data:
            user_ids = batch['uid']  # (batch_size,)
        else:
            # If no user IDs, create dummy user 0 for all samples
            batch_size = loc_seq.size(0)
            user_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Get sequence lengths using LibCity's get_origin_len method
        seq_lens = batch.get_origin_len('current_loc')

        # Get coordinates if available
        if 'current_coord' in batch.data:
            coords = batch['current_coord']  # (batch_size, seq_len, 2) [lat, lng]
        else:
            coords = None

        # Convert to tensors if needed and move to device
        if not isinstance(loc_seq, torch.Tensor):
            loc_seq = torch.LongTensor(loc_seq)
        loc_seq = loc_seq.to(self.device)

        if time_seq is not None:
            if not isinstance(time_seq, torch.Tensor):
                time_seq = torch.FloatTensor(time_seq)
            time_seq = time_seq.to(self.device)

        if not isinstance(user_ids, torch.Tensor):
            user_ids = torch.LongTensor(user_ids)
        user_ids = user_ids.to(self.device)

        if coords is not None:
            if not isinstance(coords, torch.Tensor):
                coords = torch.FloatTensor(coords)
            coords = coords.to(self.device)

        batch_size = loc_seq.shape[0]
        seq_len = loc_seq.shape[1]

        # Clamp location indices to valid range
        loc_seq = torch.clamp(loc_seq, 0, self.num_pois - 1)

        # Get POI embeddings
        poi_embeds = self.poi_embed(loc_seq)  # (batch, seq_len, poi_embed_dim)

        # Get user embeddings (expanded to seq_len)
        if user_ids is not None:
            user_ids = torch.clamp(user_ids, 0, self.num_users - 1)
            user_embeds = self.user_embed(user_ids)  # (batch, user_embed_dim)
            user_embeds = user_embeds.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, user_embed_dim)
        else:
            user_embeds = torch.zeros(batch_size, seq_len, self.user_embed_dim, device=self.device)

        # Get GPS embeddings
        if coords is not None:
            # Compute GPS indices from coordinates
            gps_indices = []
            for b in range(batch_size):
                batch_gps = []
                for t in range(seq_len):
                    lat = coords[b, t, 0].item()
                    lng = coords[b, t, 1].item()
                    gps_idx = self._compute_gps_idx(lat, lng)
                    batch_gps.append(gps_idx)
                gps_indices.append(batch_gps)
            gps_indices = torch.LongTensor(gps_indices).to(self.device)
        else:
            # Default GPS indices if no coordinates available
            gps_indices = torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device)

        gps_indices = torch.clamp(gps_indices, 0, self.num_gps - 1)
        gps_embeds = self.gps_embed(gps_indices)  # (batch, seq_len, gps_embed_dim)

        # Compute time embeddings
        if time_seq is not None:
            # Normalize time to get hour and day information
            # Assuming time_seq contains normalized time or hour values
            norm_hour = time_seq % 1.0 if time_seq.max() <= 1.0 else (time_seq / 24.0) % 1.0
            norm_day = (time_seq / 7.0) % 1.0 if time_seq.max() > 1.0 else time_seq
        else:
            norm_hour = torch.zeros(batch_size, seq_len, device=self.device)
            norm_day = torch.zeros(batch_size, seq_len, device=self.device)

        # Encode hour and day for rotation
        # Flatten for encoding
        norm_hour_flat = norm_hour.view(-1)
        norm_day_flat = norm_day.view(-1)

        target_hour = self.time_hour_encoder(norm_hour_flat)  # (batch*seq_len, total_time_dim)
        target_day = self.time_day_encoder(norm_day_flat)  # (batch*seq_len, total_time_dim)

        # Reshape back to batch form
        target_hour = target_hour.view(batch_size, seq_len, -1)  # (batch, seq_len, total_time_dim)
        target_day = target_day.view(batch_size, seq_len, -1)  # (batch, seq_len, total_time_dim)

        # Fuse embeddings for Stream 1: User + POI
        src1 = self.fuse_user_poi(user_embeds, poi_embeds)  # (batch, seq_len, poi_embed_dim)
        # Concatenate user embedding to match expected dimension
        src1 = torch.cat([user_embeds, poi_embeds], dim=-1)  # (batch, seq_len, user+poi)

        # Fuse embeddings for Stream 2: POI + GPS
        src2 = torch.cat([poi_embeds, gps_embeds], dim=-1)  # (batch, seq_len, poi+gps)

        # Generate attention mask
        max_len = seq_len
        src_mask = self.transformer.generate_square_subsequent_mask(max_len).to(self.device)

        # Forward through dual-stream transformer
        y_pred = self.transformer(
            src1, src2, src_mask, target_hour, target_day, poi_embeds, gps_embeds
        )

        return y_pred, seq_lens

    def predict(self, batch):
        """
        Prediction method for LibCity.

        Args:
            batch: Input batch dictionary

        Returns:
            POI prediction scores for the last timestep of each sequence
        """
        y_pred, seq_lens = self.forward(batch)

        # Get predictions for the last valid timestep of each sequence
        batch_size = y_pred.size(0)
        predictions = []

        for i in range(batch_size):
            last_idx = seq_lens[i] - 1
            last_idx = max(0, min(last_idx, y_pred.size(1) - 1))
            predictions.append(y_pred[i, last_idx, :])

        return torch.stack(predictions)  # (batch_size, num_pois)

    def calculate_loss(self, batch):
        """
        Calculate training loss.

        Args:
            batch: LibCity Batch object containing:
                - trajectory data for forward pass
                - 'target': Target POI indices (batch_size,)

        Returns:
            loss: Cross-entropy loss tensor
        """
        y_pred, seq_lens = self.forward(batch)

        # Get target from batch - use direct key access for LibCity Batch
        target = batch['target']  # (batch_size,)

        if not isinstance(target, torch.Tensor):
            target = torch.LongTensor(target)
        target = target.to(self.device)

        # Clamp target to valid range
        target = torch.clamp(target, 0, self.num_pois - 1)

        batch_size = y_pred.size(0)

        # Calculate loss using last timestep prediction for each sequence
        losses = []
        for i in range(batch_size):
            last_idx = seq_lens[i] - 1
            last_idx = max(0, min(last_idx, y_pred.size(1) - 1))
            pred = y_pred[i, last_idx, :].unsqueeze(0)
            tgt = target[i].unsqueeze(0)
            losses.append(self.criterion(pred, tgt))

        loss = torch.stack(losses).mean()
        return loss
