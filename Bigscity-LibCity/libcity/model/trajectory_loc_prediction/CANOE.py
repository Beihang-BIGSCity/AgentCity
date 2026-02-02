# coding: utf-8
"""
CANOE: Chaotic Neural Oscillator Enhanced Location Prediction

This module adapts the CANOE (Chaotic Attentive Neural Oscillator for Enhanced Location Prediction)
model from the original paper implementation.

Original repository: repos/CANOE/
Original files:
    - model/cnolp.py (main model classes)
    - model/cnolp_module.py (model components)
    - model/dataloader.py (data loading and LDA preprocessing)
    - model/tools.py (utilities)

Key innovations preserved:
1. Chaotic Oscillator - Neural oscillator with chaotic dynamics for attention enhancement
2. Tri-Pair Interaction Encoders:
   - UserLocationPair (LDA-based topic modeling for user-location preferences)
   - TimeUserPair (attention-based temporal user preference modeling)
   - LocationTimePair (transformer-based spatio-temporal encoding)
3. CrossContextAttentiveDecoder - Multi-head attention integrating pairwise features
4. Multi-task learning (location + time + ranking predictions)

Key adaptations for LibCity:
- Inherits from AbstractModel
- Implements predict() and calculate_loss() methods
- Adapts batch format to work with LibCity's trajectory data loaders
- LDA preprocessing is handled during initialization when topic data is available

Required data_feature keys:
- loc_size: Number of locations
- uid_size: Number of users
- tim_size: Number of time slots (should be 24 for hour-based)
- user_topic_loc: (optional) Pre-computed LDA topic distributions for users

Required config parameters:
- dim: Base embedding dimension (default: 16 for TC, 8 for MP)
- topic_num: Number of LDA topics (default: 350 for TC, 150 for MP)
- bandwidth: Gaussian kernel bandwidth for time smoothing (default: 1.0)
- model_type: Model variant 'tc' or 'mp' (default: 'tc')
- encoder: Encoder type 'trans' for transformer (default: 'trans')
- at: Attention type 'osc' for oscillator (default: 'osc')
- lambda_loc: Weight for location loss (default: 0.9)
- lambda_time: Weight for time loss (default: 0.4)
- lambda_rank: Weight for ranking loss (default: 0.6)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from libcity.model.abstract_model import AbstractModel


class MultimodalContextualEmbedding(nn.Module):
    """
    Multimodal Contextual Embedding Module.

    Generates embeddings for users, locations, and time slots with Gaussian kernel
    smoothing for temporal features.
    """
    def __init__(self, dim, num_locations, num_users, bandwidth=1.0, max_seq_length=64):
        super(MultimodalContextualEmbedding, self).__init__()
        self.num_locations = num_locations
        self.base_dim = dim
        self.num_users = num_users
        self.bandwidth = bandwidth
        self.max_seq_length = max_seq_length

        self.user_embedding = nn.Embedding(self.num_users, self.base_dim)
        self.location_embedding = nn.Embedding(self.num_locations, self.base_dim)
        self.timeslot_embedding = nn.Embedding(24, self.base_dim)

    def gaussian_kernel(self, timestamps, tn, device):
        """
        Compute Gaussian kernel weights considering periodicity.

        Args:
            timestamps: Time indices
            tn: Target time index
            device: Device for tensor operations

        Returns:
            Gaussian kernel weights
        """
        timestamps = timestamps.float()
        tn = float(tn)
        # Consider periodicity of 24-hour cycle
        dist = torch.min(torch.abs(timestamps - tn), 24 - torch.abs(timestamps - tn))
        return torch.exp(-0.5 * (dist / self.bandwidth) ** 2)

    def forward(self, location_x, device):
        """
        Generate embeddings.

        Args:
            location_x: Location indices tensor [batch_size, seq_len]
            device: Device for tensor operations

        Returns:
            Tuple of (loc_embedded, timeslot_embedded, smoothed_timeslot_embedded, user_embedded)
        """
        loc_embedded = self.location_embedding(location_x)
        user_embedded = self.user_embedding(torch.arange(end=self.num_users, dtype=torch.int, device=device))
        timeslot_embedded = self.timeslot_embedding(torch.arange(end=24, dtype=torch.int, device=device))

        # Compute Gaussian-smoothed time embeddings for all 24 time slots
        smoothed_list = []
        for tn in range(24):
            kernel_weights = self.gaussian_kernel(torch.arange(24, device=device), tn, device).view(24, 1)
            smoothed = torch.sum(kernel_weights * timeslot_embedded, dim=0)
            smoothed_list.append(smoothed)

        smoothed_timeslot_embedded = torch.stack(smoothed_list, dim=0)  # [24, base_dim]

        return loc_embedded, timeslot_embedded, smoothed_timeslot_embedded, user_embedded


class UserLocationPair(nn.Module):
    """
    User-Location Pair Interaction Encoder.

    Encodes user-location preferences using LDA topic distributions.
    The topic vectors are pre-computed using gensim LDA on user visit histories.
    """
    def __init__(self, input_dim, output_dim):
        super(UserLocationPair, self).__init__()
        self.topic_num = input_dim
        self.output_dim = output_dim
        self.block = nn.Sequential(
            nn.Linear(self.topic_num, self.topic_num * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.topic_num * 2, self.topic_num)
        )
        self.final = nn.Sequential(
            nn.LayerNorm(self.topic_num),
            nn.Linear(self.topic_num, self.output_dim)
        )

    def forward(self, topic_vec):
        """
        Encode topic vector with residual connection.

        Args:
            topic_vec: LDA topic distribution [batch_size, topic_num]

        Returns:
            Encoded topic features [batch_size, output_dim]
        """
        x = topic_vec
        topic_vec = self.block(topic_vec)
        topic_vec = x + topic_vec
        return self.final(topic_vec)


class Oscillator(nn.Module):
    """
    Chaotic Neural Oscillator.

    Core innovation of CANOE - uses chaotic dynamics to enhance attention mechanisms.
    The oscillator produces excitatory (u) and inhibitory (v) signals that modulate
    the attention scores through chaotic dynamics.

    Parameters:
        a1, a2: Excitatory connection weights
        b1, b2: Inhibitory connection weights
        k: Exponential decay factor
        n: Number of oscillation iterations
    """
    def __init__(self, model_type='tc', device='cpu'):
        super(Oscillator, self).__init__()
        self.device = device
        self.uthreshold = 0
        self.vthreshold = 0

        # Different parameters for TC (Traffic Camera) and MP (Mobile Phone) datasets
        if model_type == 'tc':
            self.u = 0  # excitatory state
            self.v = 0  # inhibitory state
            self.a1 = torch.tensor([5.0])
            self.a2 = torch.tensor([5.0])
            self.b1 = torch.tensor([5.0])
            self.b2 = torch.tensor([-5.0])
        else:  # 'mp'
            self.u = 0
            self.v = 0
            self.a1 = torch.tensor([5.0])
            self.a2 = torch.tensor([5.0])
            self.b1 = torch.tensor([-5.0])
            self.b2 = torch.tensor([-5.0])

        self.k = torch.tensor([-500.0])
        self.n = 50  # Number of oscillation iterations

    def Calculatez(self, I):
        """
        Calculate output signal z from input I.

        Uses the formula: z = (u - v) * exp(k * I^2) + ReLU(I)
        """
        uv = torch.sub(self.u, self.v)
        kI = torch.mul(self.k.to(I.device), I)
        kI2 = torch.mul(kI, I)
        z = torch.add(torch.mul(uv, torch.exp(kI2)), F.relu(I))
        return z

    def forward(self, I):
        """
        Apply chaotic oscillator dynamics to input.

        Args:
            I: Input attention scores

        Returns:
            Modulated attention scores through chaotic dynamics
        """
        device = I.device
        self.u = torch.zeros(I.shape, device=device)
        self.v = torch.zeros(I.shape, device=device)
        self.uthreshold = torch.tensor(0, device=device)
        self.vthreshold = torch.tensor(0, device=device)

        z = self.Calculatez(I)

        # Iterate oscillator dynamics
        for i in range(self.n):
            self.u = F.relu(torch.add(
                torch.add(
                    torch.mul(self.a1.to(device), self.u),
                    torch.mul(self.a2.to(device), self.v)
                ),
                torch.sub(I, self.uthreshold)
            ))
            self.v = F.relu(torch.sub(
                torch.sub(
                    torch.mul(self.b1.to(device), self.u),
                    torch.mul(self.b2.to(device), self.v)
                ),
                self.vthreshold
            ))

        z = self.Calculatez(I)
        return z


class TimeUserPair(nn.Module):
    """
    Time-User Pair Interaction Encoder.

    Models user temporal preferences using multi-head attention with oscillator enhancement.
    Captures how users prefer different locations at different times.
    """
    def __init__(self, dim, num_users, at_type='osc', model_type='tc', device='cpu'):
        super(TimeUserPair, self).__init__()
        self.base_dim = dim
        self.num_heads = 4
        self.device = device
        assert self.base_dim % self.num_heads == 0, "base_dim must be divisible by num_heads"
        self.head_dim = self.base_dim // self.num_heads
        self.num_users = num_users
        self.timeslot_num = 24
        self.at_type = at_type

        if at_type == 'osc':
            self.user_preference = nn.Embedding(self.num_users, self.base_dim)
            self.w_q = nn.ModuleList(
                [nn.Linear(self.base_dim * 2, self.head_dim) for _ in range(self.num_heads)])
            self.w_k = nn.ModuleList(
                [nn.Linear(self.base_dim, self.head_dim) for _ in range(self.num_heads)])
            self.w_v = nn.ModuleList(
                [nn.Linear(self.base_dim, self.head_dim) for _ in range(self.num_heads)])

            self.unify_heads = nn.Linear(self.base_dim, self.base_dim)

        self.Oscillator = Oscillator(model_type=model_type, device=device)
        self.time_head = nn.Linear(self.base_dim, self.timeslot_num)

    def forward(self, timeslot_embedded, smoothed_timeslot_embedded, user_embedded,
                user_x, hour_x, hour_mask):
        """
        Encode time-user pair interactions.

        Args:
            timeslot_embedded: Time slot embeddings [24, dim]
            smoothed_timeslot_embedded: Gaussian smoothed time embeddings [24, dim]
            user_embedded: User embeddings [num_users, dim]
            user_x: User indices [batch_size]
            hour_x: Hour indices [batch_size, seq_len]
            hour_mask: Mask for invalid hours [batch_size * seq_len, 24]

        Returns:
            Tuple of (at_emb, time_logits)
        """
        batch_size, sequence_length = hour_x.shape
        total_sequences = batch_size * sequence_length

        if self.at_type == 'osc':
            hour_x_flat = hour_x.view(batch_size * sequence_length)

            head_outputs = []
            user_preference = self.user_preference(user_x).unsqueeze(1).repeat(1, sequence_length, 1)
            user_feature = user_preference.view(batch_size * sequence_length, -1)
            time_feature = timeslot_embedded[hour_x_flat]

            query = torch.cat([user_feature, time_feature], dim=-1)
            key = smoothed_timeslot_embedded

            for i in range(self.num_heads):
                query_i = self.w_q[i](query)
                key_i = self.w_k[i](key)
                value_i = self.w_v[i](key)
                attn_scores_i = torch.matmul(query_i, key_i.T)
                scale = 1.0 / (key_i.size(-1) ** 0.5)
                attn_scores_i = attn_scores_i * scale
                attn_scores_i = attn_scores_i.masked_fill(hour_mask == 1, float('-inf'))
                attn_scores_i = self.Oscillator(attn_scores_i)
                attn_scores_i = torch.softmax(attn_scores_i, dim=-1)
                weighted_values_i = torch.matmul(attn_scores_i, value_i)
                head_outputs.append(weighted_values_i)

            head_outputs = torch.cat(head_outputs, dim=-1)
            head_outputs = head_outputs.view(batch_size, sequence_length, -1)
            at_emb = self.unify_heads(head_outputs)
            time_logits = self.time_head(head_outputs)
        else:
            # Fallback for non-oscillator attention
            at_emb = torch.zeros(batch_size, sequence_length, self.base_dim, device=hour_x.device)
            time_logits = torch.zeros(batch_size, sequence_length, self.timeslot_num, device=hour_x.device)

        return at_emb, time_logits


class LocationTimePair(nn.Module):
    """
    Location-Time Pair Interaction Encoder.

    Uses Transformer encoder to model sequential location-time patterns.
    """
    def __init__(self, input_dim, nhead=4, num_layers=3, dropout=0.1):
        super(LocationTimePair, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            activation='gelu',
            batch_first=True,
            dim_feedforward=input_dim,
            nhead=nhead,
            dropout=dropout
        )
        encoder_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=encoder_norm
        )
        self.initialize_parameters()

    def forward(self, embedded_out, src_mask):
        """
        Apply transformer encoding.

        Args:
            embedded_out: Input embeddings [batch_size, seq_len, dim]
            src_mask: Causal attention mask [seq_len, seq_len]

        Returns:
            Encoded output [batch_size, seq_len, dim]
        """
        out = self.encoder(embedded_out, mask=src_mask)
        return out

    def initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)


class CrossContextAttentiveDecoder(nn.Module):
    """
    Cross Context Attentive Decoder.

    Combines information from different pair encoders using multi-head attention
    with oscillator enhancement.
    """
    def __init__(self, query_dim, kv_dim, embed_dim, output_dim, model_type='tc', num_heads=4, device='cpu'):
        super(CrossContextAttentiveDecoder, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(kv_dim, embed_dim) if kv_dim != embed_dim else nn.Identity()
        self.v_proj = nn.Linear(kv_dim, embed_dim) if kv_dim != embed_dim else nn.Identity()
        self.Oscillator = Oscillator(model_type=model_type, device=device)
        self.out_fc = nn.Linear(embed_dim, output_dim)

    def forward(self, query, key, value):
        """
        Apply cross-context attention.

        Args:
            query: Query tensor [B, L_q, query_dim]
            key: Key tensor [B, L_k, kv_dim]
            value: Value tensor [B, L_k, kv_dim]

        Returns:
            Output tensor [B, L_q, output_dim]
        """
        B, L_q, _ = query.size()
        B, L_k, _ = key.size()

        Q = self.query_proj(query)   # [B, L_q, embed_dim]
        K = self.k_proj(key)         # [B, L_k, embed_dim]
        V = self.v_proj(value)       # [B, L_k, embed_dim]

        Q = Q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)   # [B, nh, L_q, hd]
        K = K.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)   # [B, nh, L_k, hd]
        V = V.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)   # [B, nh, L_k, hd]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = self.Oscillator(scores)
        attn_weights = torch.softmax(scores, dim=-1)

        head_outputs = torch.matmul(attn_weights, V)
        head_outputs = head_outputs.transpose(1, 2).contiguous().view(B, L_q, self.embed_dim)

        out = self.out_fc(head_outputs)
        return out


class NextLocationPrediction(nn.Module):
    """
    Next Location Prediction Head.

    Final prediction layer with residual connection and normalization.
    """
    def __init__(self, input_dim, output_dim):
        super(NextLocationPrediction, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim),
            nn.Dropout(0.1),
        )
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.drop = nn.Dropout(0.1)
        self.linear_class = nn.Linear(input_dim, output_dim)

    def forward(self, out):
        """
        Apply prediction head.

        Args:
            out: Input features [batch_size * seq_len, input_dim]

        Returns:
            Location predictions [batch_size * seq_len, num_locations]
        """
        x = out
        out = self.block(out)
        out = out + x
        out = self.batch_norm(out)
        out = self.drop(out)
        return self.linear_class(out)


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer.

    Adds sinusoidal positional embeddings to input sequences.
    """
    def __init__(self, emb_dim, max_len=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.emb_dim = emb_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, emb_dim))

        pos_encoding = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * -(math.log(10000.0) / emb_dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, out):
        out = out + self.pos_encoding[:, :out.size(1)].detach()
        out = self.dropout(out)
        return out


class CANOE(AbstractModel):
    """
    CANOE: Chaotic Attentive Neural Oscillator for Enhanced Location Prediction.

    A multi-task learning model for next location prediction that leverages:
    1. Tri-pair interaction encoding (user-location, time-user, location-time)
    2. Chaotic neural oscillator for attention enhancement
    3. Cross-context attentive decoder
    4. Multi-task learning (location, time, ranking prediction)

    This model is adapted for LibCity's trajectory location prediction task.
    """

    def __init__(self, config, data_feature):
        super(CANOE, self).__init__(config, data_feature)

        # Device configuration
        self.device = config.get('device', 'cpu')

        # Data dimensions from data_feature
        self.num_locations = data_feature.get('loc_size', 1000)
        self.num_users = data_feature.get('uid_size', 100)
        self.tim_size = data_feature.get('tim_size', 24)

        # Model hyperparameters from config
        self.base_dim = config.get('dim', 16)
        self.topic_num = config.get('topic_num', 350)
        self.bandwidth = config.get('bandwidth', 1.0)
        self.model_type = config.get('model_type', 'tc')
        self.encoder_type = config.get('encoder', 'trans')
        self.at_type = config.get('at', 'osc')
        self.max_seq_length = config.get('max_seq_length', 64)

        # Loss weights for multi-task learning
        self.lambda_loc = config.get('lambda_loc', 0.9)
        self.lambda_time = config.get('lambda_time', 0.4)
        self.lambda_rank = config.get('lambda_rank', 0.6)

        # Sequence length configuration
        self.seq_len = config.get('seq_len', 20)

        # User topic location data (pre-computed LDA distributions)
        self.user_topic_loc = data_feature.get('user_topic_loc', None)
        if self.user_topic_loc is not None and not isinstance(self.user_topic_loc, torch.Tensor):
            self.user_topic_loc = torch.tensor(self.user_topic_loc, dtype=torch.float32)

        # Build model components
        self._build_model()

    def _build_model(self):
        """Build all model components."""
        # Multimodal Contextual Embedding
        self.embedding_layer = MultimodalContextualEmbedding(
            dim=self.base_dim,
            num_locations=self.num_locations,
            num_users=self.num_users,
            bandwidth=self.bandwidth,
            max_seq_length=self.max_seq_length
        )

        # Determine model architecture based on type
        # Calculate the actual combined dimension based on available components
        # Components in 'combined' tensor:
        # - encoder_out: base_dim (16)
        # - lt_embedded: base_dim (16)
        # - at_embedded (if at_type != 'none'): base_dim (16)
        # - user_embedded_seq: base_dim (16)
        # - pre_embedded (if topic_num > 0): base_dim (16)
        # Total for tc with topic_num > 0: base_dim * 5 = 80
        # Total for tc with topic_num = 0: base_dim * 4 = 64
        #
        # Note: combined_emb_no_pre (used as key/value for CNOA) always has 4 components:
        # encoder_out + lt_embedded + at_embedded + user_embedded_seq = base_dim * 4 = 64

        # CNOA key/value dimension is always base_dim * 4 (combined_emb_no_pre)
        # because it doesn't include pre_embedded (which is used as query)
        cnoa_kv_dim = self.base_dim * 4  # 64: encoder_out + lt_embedded + at_embedded + user_embedded_seq

        if self.model_type == 'tc':
            # Traffic Camera dataset configuration
            if self.topic_num > 0:
                combined_dim = self.base_dim * 5  # 80: all 5 components
            else:
                combined_dim = self.base_dim * 4  # 64: only 4 components (no LDA)
            self.fc_mapping = nn.Linear(self.num_locations, combined_dim)
        else:
            # Mobile Phone dataset configuration
            if self.topic_num > 0:
                combined_dim = 64
                # mp1 transforms the combined tensor (with pre_embedded) from 80 to 64
                self.mp1 = nn.Linear(self.base_dim * 5, combined_dim)
            else:
                combined_dim = self.base_dim * 4  # 64: only 4 components (no LDA)
                # mp1 transforms the combined tensor (without pre_embedded) from 64 to 64
                self.mp1 = nn.Linear(self.base_dim * 4, combined_dim)
            self.fc_mapping = nn.Linear(self.num_locations, combined_dim)

        # Location-Time Pair Encoder (Transformer)
        if self.encoder_type == 'trans':
            self.positional_encoding = PositionalEncoding(emb_dim=self.base_dim)
            self.encoder = LocationTimePair(input_dim=self.base_dim)

        # Time-User Pair Encoder
        if self.at_type != 'none':
            self.at_net = TimeUserPair(
                dim=self.base_dim,
                num_users=self.num_users,
                at_type=self.at_type,
                model_type=self.model_type,
                device=self.device
            )

        # User-Location Pair Encoder (LDA-based)
        if self.topic_num > 0:
            self.user_net = UserLocationPair(
                input_dim=self.topic_num,
                output_dim=self.base_dim
            )

        # Cross Context Attentive Decoder
        self.cnoa = CrossContextAttentiveDecoder(
            query_dim=self.base_dim,
            kv_dim=cnoa_kv_dim,
            embed_dim=cnoa_kv_dim,
            output_dim=self.num_locations,
            model_type=self.model_type,
            num_heads=4,
            device=self.device
        )

        # Prediction heads
        # Calculate fc_input_dim based on actual components that will be concatenated:
        # - encoder_out: base_dim (16)
        # - lt_embedded: base_dim (16)
        # - at_embedded (if at_type != 'none'): base_dim (16)
        # - user_embedded_seq: base_dim (16)
        # - pre_embedded (if topic_num > 0): base_dim (16)
        # Total for tc with topic_num > 0: base_dim * 5 = 80
        # Total for tc with topic_num = 0: base_dim * 4 = 64
        if self.model_type == 'tc':
            if self.topic_num > 0:
                fc_input_dim = self.base_dim * 5  # 80: all 5 components
            else:
                fc_input_dim = self.base_dim * 4  # 64: only 4 components (no LDA)
        else:  # model_type == 'mp'
            if self.topic_num > 0:
                fc_input_dim = 64
            else:
                fc_input_dim = self.base_dim * 4  # 64: adjust for mp as well

        # Store fc_input_dim for use in forward method
        self.fc_input_dim = fc_input_dim

        self.fc_layer = NextLocationPrediction(
            input_dim=fc_input_dim,
            output_dim=self.num_locations
        )
        self.rank_head = NextLocationPrediction(
            input_dim=fc_input_dim,
            output_dim=self.num_locations
        )

        self.out_dropout = nn.Dropout(0.1)

        # Loss functions
        self.criterion_loc = nn.CrossEntropyLoss()
        self.criterion_time = nn.CrossEntropyLoss()
        self.criterion_rank = nn.MarginRankingLoss(margin=1.0)

    def _prepare_batch(self, batch):
        """
        Prepare batch data from LibCity format.

        LibCity batch format for trajectory location prediction typically includes:
        - 'current_loc': Current location sequence [batch_size, seq_len]
        - 'current_tim': Current time sequence [batch_size, seq_len]
        - 'target': Target location [batch_size]
        - 'uid': User ID [batch_size]

        This method adapts to CANOE's expected format.
        """
        # Extract data from batch
        if hasattr(batch, 'data'):
            # LibCity Batch object
            loc_x = batch['current_loc']
            uid = batch['uid'] if 'uid' in batch.data else torch.zeros(loc_x.size(0), dtype=torch.long, device=self.device)

            # Handle time data
            if 'current_tim' in batch.data:
                tim = batch['current_tim']
                # Convert to hour if needed (assuming time slots)
                if tim.max() >= 24:
                    hour_x = tim % 24
                else:
                    hour_x = tim
            else:
                # Default to zeros if no time data
                hour_x = torch.zeros_like(loc_x, dtype=torch.long)

            target = batch['target'] if 'target' in batch.data else None

            # Get target time if available
            if 'target_tim' in batch.data:
                target_tim = batch['target_tim']
                if target_tim.max() >= 24:
                    target_tim = target_tim % 24
            else:
                target_tim = None
        else:
            # Dictionary-like batch
            loc_x = batch.get('current_loc', batch.get('location_x'))
            uid = batch.get('uid', batch.get('user', torch.zeros(loc_x.size(0), dtype=torch.long, device=self.device)))
            hour_x = batch.get('current_tim', batch.get('hour', torch.zeros_like(loc_x, dtype=torch.long)))
            if hour_x.max() >= 24:
                hour_x = hour_x % 24
            target = batch.get('target', batch.get('location_y'))
            target_tim = batch.get('target_tim', batch.get('timeslot_y'))

        # Handle user topic location data
        if self.topic_num > 0:
            if 'user_topic_loc' in (batch.data if hasattr(batch, 'data') else batch):
                user_topic_loc = batch['user_topic_loc']
            elif self.user_topic_loc is not None:
                # Index from stored topic data
                user_topic_loc = self.user_topic_loc[uid].to(self.device)
            else:
                # Default to zeros
                user_topic_loc = torch.zeros(loc_x.size(0), self.topic_num, device=self.device)
        else:
            user_topic_loc = None

        # Create hour mask (mask out hours where user never appears)
        batch_size, seq_len = loc_x.shape
        hour_mask = torch.zeros(batch_size * seq_len, 24, dtype=torch.int32, device=self.device)

        return {
            'location_x': loc_x.to(self.device),
            'user': uid.to(self.device),
            'hour': hour_x.to(self.device),
            'hour_mask': hour_mask,
            'user_topic_loc': user_topic_loc,
            'location_y': target.to(self.device) if target is not None else None,
            'timeslot_y': target_tim.to(self.device) if target_tim is not None else None
        }

    def forward(self, batch):
        """
        Forward pass.

        Args:
            batch: LibCity batch or dictionary containing input data

        Returns:
            Tuple of (location_output, time_logits, rank_output)
        """
        # Prepare batch data
        batch_data = self._prepare_batch(batch)

        user_x = batch_data['user']
        loc_x = batch_data['location_x']
        hour_x = batch_data['hour']
        hour_mask = batch_data['hour_mask']

        batch_size, sequence_length = loc_x.shape

        # Get embeddings
        loc_embedded, timeslot_embedded, smoothed_timeslot_embedded, user_embedded = \
            self.embedding_layer(loc_x, self.device)

        # Get time embeddings
        time_embedded = timeslot_embedded[hour_x]
        smoothed_time_embedded = smoothed_timeslot_embedded[hour_x]

        # Combine location and time embeddings
        lt_embedded = loc_embedded + time_embedded

        # Apply transformer encoder
        if hasattr(self, 'encoder'):
            future_mask = torch.triu(
                torch.ones(sequence_length, sequence_length, device=self.device),
                diagonal=1
            )
            future_mask = future_mask.masked_fill(future_mask == 1, float('-inf')).bool()
            encoder_out = self.encoder(
                self.positional_encoding(lt_embedded * math.sqrt(self.base_dim)),
                src_mask=future_mask
            )
        else:
            encoder_out = lt_embedded

        # Combine encoder output with location-time embedding
        combined = torch.cat([encoder_out, lt_embedded], dim=-1)

        # Get user embedding for each sample
        user_embedded_batch = user_embedded[user_x]

        # Apply Time-User Pair encoder if enabled
        if hasattr(self, 'at_net') and self.at_type != 'none':
            at_embedded, time_logits = self.at_net(
                timeslot_embedded,
                smoothed_timeslot_embedded,
                user_embedded_batch,
                user_x,
                hour_x,
                hour_mask
            )
            combined = torch.cat([combined, at_embedded], dim=-1)
            # combined_emb should be the same as combined at this point
            combined_emb = combined
        else:
            time_logits = torch.zeros(batch_size, sequence_length, 24, device=self.device)
            combined_emb = combined

        # Expand user embedding for sequence
        user_embedded_seq = user_embedded_batch.unsqueeze(1).repeat(1, sequence_length, 1)
        combined = torch.cat([combined, user_embedded_seq], dim=-1)
        combined_emb_no_pre = torch.cat([combined_emb, user_embedded_seq], dim=-1)

        # Apply User-Location Pair encoder if LDA topics available
        if self.topic_num > 0 and batch_data['user_topic_loc'] is not None:
            pre_embedded = self.user_net(batch_data['user_topic_loc']).unsqueeze(1).repeat(1, sequence_length, 1)
            combined = torch.cat([combined, pre_embedded], dim=-1)
        else:
            pre_embedded = user_embedded_seq

        # Apply Cross Context Attentive Decoder
        final_output = self.cnoa(pre_embedded, combined_emb_no_pre, combined_emb_no_pre)

        # Apply final mapping
        final_output = self.fc_mapping(final_output)

        # Residual connection
        if self.model_type == 'mp' and hasattr(self, 'mp1'):
            combined = self.mp1(combined)

        residual_output = final_output + combined

        # Reshape for prediction heads
        residual_flat = residual_output.view(batch_size * sequence_length, residual_output.shape[2])

        # Location prediction
        out = self.fc_layer(residual_flat)

        # Ranking prediction
        rank = self.rank_head(residual_flat)

        return out, time_logits, rank

    def predict(self, batch):
        """
        Predict next location.

        Args:
            batch: LibCity batch containing input data

        Returns:
            torch.Tensor: Location prediction scores [batch_size, num_locations]
        """
        self.eval()
        with torch.no_grad():
            loc_output, _, _ = self.forward(batch)

            # Get batch size and sequence length
            batch_data = self._prepare_batch(batch)
            batch_size = batch_data['location_x'].size(0)
            seq_len = batch_data['location_x'].size(1)

            # Reshape output to [batch_size, seq_len, num_locations]
            loc_output = loc_output.view(batch_size, seq_len, -1)

            # Return predictions for the last position
            last_pred = loc_output[:, -1, :]

            # Apply log softmax for compatibility with NLLLoss evaluation
            return F.log_softmax(last_pred, dim=-1)

    def calculate_loss(self, batch):
        """
        Calculate multi-task loss.

        The loss combines:
        1. Location prediction loss (CrossEntropy)
        2. Time prediction loss (CrossEntropy)
        3. Ranking loss (MarginRanking between positive and negative samples)

        Args:
            batch: LibCity batch containing input data and targets

        Returns:
            torch.Tensor: Combined loss (scalar)
        """
        # Forward pass
        loc_output, time_logits, rank_output = self.forward(batch)

        # Prepare batch data
        batch_data = self._prepare_batch(batch)
        batch_size = batch_data['location_x'].size(0)
        seq_len = batch_data['location_x'].size(1)

        # Reshape outputs
        loc_output = loc_output.view(batch_size, seq_len, -1)
        rank_output = rank_output.view(batch_size, seq_len, -1)

        # Get target
        if batch_data['location_y'] is not None:
            target = batch_data['location_y']

            # Handle different target formats
            if target.dim() == 1:
                # Single target per sequence - use last position
                loc_pred = loc_output[:, -1, :]
                rank_pred = rank_output[:, -1, :]
                time_pred = time_logits[:, -1, :]
            else:
                # Target for each position
                loc_pred = loc_output.view(-1, self.num_locations)
                rank_pred = rank_output.view(-1, self.num_locations)
                time_pred = time_logits.view(-1, 24)
                target = target.view(-1)
        else:
            # Fallback
            loc_pred = loc_output[:, -1, :]
            rank_pred = rank_output[:, -1, :]
            time_pred = time_logits[:, -1, :]
            target = batch['target']

        # Location loss
        loss_loc = self.criterion_loc(loc_pred, target)

        # Time prediction loss
        if batch_data['timeslot_y'] is not None:
            target_time = batch_data['timeslot_y']
            if target_time.dim() > 1:
                target_time = target_time.view(-1)
            loss_time = self.criterion_time(time_pred, target_time)
        else:
            loss_time = torch.tensor(0.0, device=self.device)

        # Ranking loss (positive vs negative samples)
        # Get scores for positive (target) locations
        positive_scores = torch.gather(rank_pred, 1, target.unsqueeze(1)).squeeze(1)

        # Sample negative locations
        neg_indices = torch.randint(0, self.num_locations, target.shape, device=self.device)
        # Make sure negatives are different from targets
        neg_indices = torch.where(neg_indices == target, (neg_indices + 1) % self.num_locations, neg_indices)
        negative_scores = torch.gather(rank_pred, 1, neg_indices.unsqueeze(1)).squeeze(1)

        # Margin ranking loss (positive should be higher than negative)
        labels = torch.ones_like(positive_scores)
        loss_rank = self.criterion_rank(positive_scores, negative_scores, labels)

        # Combined loss with weights
        total_loss = (self.lambda_loc * loss_loc +
                     self.lambda_time * loss_time +
                     self.lambda_rank * loss_rank)

        return total_loss
