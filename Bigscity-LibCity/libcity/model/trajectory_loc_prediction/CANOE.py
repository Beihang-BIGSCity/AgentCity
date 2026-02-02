"""
CANOE: Chaotic Neural Oscillator for Next Location Prediction
Adapted for LibCity Framework

Original source: repos/CANOE/model/cnolp.py and cnolp_module.py
Key components preserved:
- MultimodalContextualEmbedding
- Oscillator (CNOA mechanism - core innovation)
- CrossContextAttentiveDecoder
- LocationTimePair, TimeUserPair, UserLocationPair
- NextLocationPrediction

Key adaptations:
- Inherited from AbstractModel
- Adapted forward() signature to match LibCity's Batch format
- Implemented predict() and calculate_loss() methods
- Handled data_feature parameters from LibCity dataloaders
- Preserved multi-task loss function (location + time + ranking)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from libcity.model.abstract_model import AbstractModel


# ============================================================================
# Module Components (from cnolp_module.py)
# ============================================================================

class MultimodalContextualEmbedding(nn.Module):
    """Multimodal Contextual Embedding Module for location, time, and user embeddings."""

    def __init__(self, base_dim, num_locations, num_users, bandwidth=0.5):
        super(MultimodalContextualEmbedding, self).__init__()
        self.num_locations = num_locations
        self.base_dim = base_dim
        self.num_users = num_users

        self.user_embedding = nn.Embedding(self.num_users, self.base_dim)
        self.location_embedding = nn.Embedding(self.num_locations, self.base_dim)
        self.timeslot_embedding = nn.Embedding(24, self.base_dim)

        self.bandwidth = bandwidth
        self.max_seq_length = 64

    def gaussian_kernel(self, timestamps, tn, device):
        """Compute gaussian kernel weights considering periodicity."""
        dist = torch.min(torch.abs(timestamps - tn), 24 - torch.abs(timestamps - tn))
        return torch.exp(-0.5 * (dist / self.bandwidth) ** 2)

    def forward(self, location_x, device):
        """
        Args:
            location_x: Location sequence tensor [batch_size, seq_len]
            device: Device to place tensors on
        Returns:
            loc_embedded: Location embeddings
            timeslot_embedded: Raw timeslot embeddings for all 24 hours
            smoothed_timeslot_embedded: Gaussian-smoothed timeslot embeddings
            user_embedded: User embeddings for all users
        """
        loc_embedded = self.location_embedding(location_x)
        user_embedded = self.user_embedding(
            torch.arange(end=self.num_users, dtype=torch.int, device=device)
        )
        timeslot_embedded = self.timeslot_embedding(
            torch.arange(end=24, dtype=torch.int, device=device)
        )  # Shape: [24, base_dim]

        smoothed_list = []
        for tn in range(24):
            kernel_weights = self.gaussian_kernel(
                torch.arange(24, device=device), tn, device
            ).view(24, 1)
            smoothed = torch.sum(kernel_weights * timeslot_embedded, dim=0)  # Shape: [base_dim]
            smoothed_list.append(smoothed)

        smoothed_timeslot_embedded = torch.stack(smoothed_list, dim=0)  # [24, base_dim]

        return loc_embedded, timeslot_embedded, smoothed_timeslot_embedded, user_embedded


class UserLocationPair(nn.Module):
    """User-Location Pair encoder using LDA topic vectors."""

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
        x = topic_vec
        topic_vec = self.block(topic_vec)
        topic_vec = x + topic_vec
        return self.final(topic_vec)


class Oscillator(nn.Module):
    """
    Chaotic Neural Oscillator (CNOA) - Core innovation of CANOE.
    Implements oscillatory dynamics for attention modulation.
    """

    def __init__(self, oscillator_type='cnoa_tc', device=None):
        super(Oscillator, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.uthreshold = 0
        self.vthreshold = 0

        if oscillator_type == 'cnoa_tc':
            self.u = 0  # excitatory
            self.v = 0  # inhibitory
            self.a1 = torch.tensor([5.0])
            self.a2 = torch.tensor([5.0])
            self.b1 = torch.tensor([5.0])
            self.b2 = torch.tensor([-5.0])
        elif oscillator_type == 'cnoa_mp':
            self.u = 0  # excitatory
            self.v = 0  # inhibitory
            self.a1 = torch.tensor([5.0])
            self.a2 = torch.tensor([5.0])
            self.b1 = torch.tensor([-5.0])
            self.b2 = torch.tensor([-5.0])
        else:
            # Default to cnoa_tc
            self.u = 0
            self.v = 0
            self.a1 = torch.tensor([5.0])
            self.a2 = torch.tensor([5.0])
            self.b1 = torch.tensor([5.0])
            self.b2 = torch.tensor([-5.0])

        self.k = torch.tensor([-500.0])
        self.n = 50

    def _ensure_device(self, I):
        """Ensure oscillator parameters are on the same device as input."""
        device = I.device
        if self.a1.device != device:
            self.a1 = self.a1.to(device)
            self.a2 = self.a2.to(device)
            self.b1 = self.b1.to(device)
            self.b2 = self.b2.to(device)
            self.k = self.k.to(device)

    def Calculatez(self, I):
        device = I.device
        self.u = torch.randn(I.shape, device=device) * 0.01
        self.v = torch.randn(I.shape, device=device) * 0.01
        uv = torch.sub(self.u, self.v)
        kI = torch.mul(self.k, I)
        kI2 = torch.mul(kI, I)
        z = torch.add(torch.mul(uv, torch.exp(kI2)), F.relu(I))
        return z

    def forward(self, I):
        self._ensure_device(I)
        device = I.device
        self.u = torch.zeros(I.shape, device=device)
        self.v = torch.zeros(I.shape, device=device)
        self.uthreshold = torch.tensor(0, device=device)
        self.vthreshold = torch.tensor(0, device=device)
        z = self.Calculatez(I)

        for i in range(self.n):
            self.u = F.relu(torch.add(
                torch.add(torch.mul(self.a1, self.u), torch.mul(self.a2, self.v)),
                torch.sub(I, self.uthreshold)
            ))
            self.v = F.relu(torch.sub(
                torch.sub(torch.mul(self.b1, self.u), torch.mul(self.b2, self.v)),
                self.vthreshold
            ))
        z = self.Calculatez(I)

        return z


class TimeUserPair(nn.Module):
    """Time-User Pair encoder with Chaotic Neural Oscillator Attention."""

    def __init__(self, base_dim, num_users, at_type='osc', oscillator_type='cnoa_tc', device=None):
        super(TimeUserPair, self).__init__()
        self.base_dim = base_dim
        self.num_heads = 4
        assert self.base_dim % self.num_heads == 0, "base_dim must be divisible by num_heads"
        self.head_dim = self.base_dim // self.num_heads
        self.num_users = num_users
        self.timeslot_num = 24
        self.at_type = at_type

        if at_type == 'osc':
            self.user_preference = nn.Embedding(self.num_users, self.base_dim)
            self.w_q = nn.ModuleList(
                [nn.Linear(self.base_dim * 2, self.head_dim) for _ in range(self.num_heads)]
            )
            self.w_k = nn.ModuleList(
                [nn.Linear(self.base_dim, self.head_dim) for _ in range(self.num_heads)]
            )
            self.w_v = nn.ModuleList(
                [nn.Linear(self.base_dim, self.head_dim) for _ in range(self.num_heads)]
            )

            self.unify_heads = nn.Linear(self.base_dim, self.base_dim)
            self.key_trans = nn.Linear(in_features=5120, out_features=24, bias=True)
            self.query_trans = nn.Linear(in_features=24, out_features=5120, bias=True)

        self.Oscillator = Oscillator(oscillator_type=oscillator_type, device=device)
        self.time_head = nn.Linear(self.base_dim, self.timeslot_num)

    def forward(self, timeslot_embedded, smoothed_timeslot_embedded, user_embedded,
                user_x, hour_x, hour_mask, return_attn=False):
        """
        Args:
            timeslot_embedded: [24, base_dim]
            smoothed_timeslot_embedded: [24, base_dim]
            user_embedded: [num_users, base_dim]
            user_x: [batch_size]
            hour_x: [batch_size, seq_len]
            hour_mask: [batch_size, seq_len, 24]
        """
        attn_probs = None
        batch_size, sequence_length = hour_x.shape
        total_sequences = batch_size * sequence_length
        hour_mask_flat = hour_mask.view(batch_size * sequence_length, -1)

        if self.at_type == 'osc':
            hour_x_flat = hour_x.view(batch_size * sequence_length)
            head_outputs = []
            user_preference = self.user_preference(user_x).unsqueeze(1).repeat(1, sequence_length, 1)
            user_feature = user_preference.view(batch_size * sequence_length, -1)
            time_feature = timeslot_embedded[hour_x_flat]

            query = torch.cat([user_feature, time_feature], dim=-1)
            key = smoothed_timeslot_embedded

            head_outputs = []
            for i in range(self.num_heads):
                query_i = self.w_q[i](query)
                key_i = self.w_k[i](key)
                value_i = self.w_v[i](key)
                attn_scores_i = torch.matmul(query_i, key_i.T)
                scale = 1.0 / (key_i.size(-1) ** 0.5)
                attn_scores_i = attn_scores_i * scale
                attn_scores_i = attn_scores_i.masked_fill(hour_mask_flat == 1, float('-inf'))
                attn_scores_i = self.Oscillator(attn_scores_i)
                attn_scores_i = torch.softmax(attn_scores_i, dim=-1)
                weighted_values_i = torch.matmul(attn_scores_i, value_i)
                head_outputs.append(weighted_values_i)
            head_outputs = torch.cat(head_outputs, dim=-1)
            head_outputs = head_outputs.view(batch_size, sequence_length, -1)
            at_emb = self.unify_heads(head_outputs)
            time_logits = self.time_head(head_outputs)
        else:
            # Fallback if at_type is not 'osc'
            at_emb = torch.zeros(batch_size, sequence_length, self.base_dim, device=hour_x.device)
            time_logits = torch.zeros(batch_size, sequence_length, self.timeslot_num, device=hour_x.device)

        return at_emb, time_logits


class LocationTimePair(nn.Module):
    """Location-Time Pair encoder using Transformer."""

    def __init__(self, input_dim):
        super(LocationTimePair, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            activation='gelu',
            batch_first=True,
            dim_feedforward=input_dim,
            nhead=4,
            dropout=0.1
        )
        encoder_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=3,
            norm=encoder_norm
        )
        self.initialize_parameters()

    def forward(self, embedded_out, src_mask):
        out = self.encoder(embedded_out, mask=src_mask)
        return out

    def initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)


class CrossContextAttentiveDecoder(nn.Module):
    """Cross Context Attentive Decoder with Chaotic Neural Oscillator."""

    def __init__(self, query_dim, kv_dim, embed_dim, output_dim, oscillator_type='cnoa_tc',
                 num_heads=4, device=None):
        super(CrossContextAttentiveDecoder, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(kv_dim, embed_dim) if kv_dim != embed_dim else nn.Identity()
        self.v_proj = nn.Linear(kv_dim, embed_dim) if kv_dim != embed_dim else nn.Identity()
        self.Oscillator = Oscillator(oscillator_type=oscillator_type, device=device)
        self.out_fc = nn.Linear(embed_dim, output_dim)

    def forward(self, query, key, value):
        B, L_q, _ = query.size()
        B, L_k, _ = key.size()
        Q = self.query_proj(query)   # [B, L_q, embed_dim]
        K = self.k_proj(key)         # [B, L_k, embed_dim]
        V = self.v_proj(value)       # [B, L_k, embed_dim]
        Q = Q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)   # [B, nh, L_q, hd]
        K = K.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)   # [B, nh, L_k, hd]
        V = V.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)   # [B, nh, L_k, hd]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores_flat = scores.contiguous().view(B * self.num_heads, L_q, L_k)
        scores = scores_flat.view(B, self.num_heads, L_q, L_k)
        scores = self.Oscillator(scores)
        attn_weights = torch.softmax(scores, dim=-1)
        head_outputs = torch.matmul(attn_weights, V)
        head_outputs = head_outputs.transpose(1, 2).contiguous().view(B, L_q, self.embed_dim)
        out = self.out_fc(head_outputs)
        return out


class NextLocationPrediction(nn.Module):
    """Next Location Prediction head with residual connections."""

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

        num_locations = output_dim
        self.linear_class = nn.Linear(input_dim, num_locations)

    def forward(self, out):
        x = out
        out = self.block(out)
        out = out + x
        out = self.batch_norm(out)
        out = self.drop(out)

        return self.linear_class(out)


class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer encoder."""

    def __init__(self, emb_dim, max_len=512):
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
        self.dropout = nn.Dropout(0.1)

    def forward(self, out):
        out = out + self.pos_encoding[:, :out.size(1)].detach()
        out = self.dropout(out)
        return out


# ============================================================================
# Main CANOE Model adapted for LibCity
# ============================================================================

class CANOE(AbstractModel):
    """
    CANOE: Chaotic Neural Oscillator for Next Location Prediction

    This model uses chaotic neural oscillators to capture complex temporal dynamics
    in human mobility patterns for next location prediction.

    Based on CNOLP_tc variant from the original implementation.

    Required config parameters:
        - dim: Base embedding dimension (default: 16, must be multiple of 4)
        - bandwidth: Gaussian kernel bandwidth (default: 0.5)
        - topic_num: Number of LDA topics (default: 0, set to >0 to enable LDA features)
        - oscillator_type: Type of oscillator ('cnoa_tc' or 'cnoa_mp', default: 'cnoa_tc')
        - at_type: Attention type ('osc' for oscillator attention, default: 'osc')
        - encoder_type: Encoder type ('trans' for transformer, default: 'trans')
        - lambda_loc: Weight for location loss (default: 0.9)
        - lambda_time: Weight for time loss (default: 0.4)
        - lambda_rank: Weight for ranking loss (default: 0.6)

    Required data_feature parameters:
        - loc_size: Number of unique locations
        - uid_size: Number of unique users
        - topic_num: Number of LDA topics (optional, can be 0)
    """

    def __init__(self, config, data_feature):
        super(CANOE, self).__init__(config, data_feature)

        # Get parameters from config
        self.device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.base_dim = config.get('dim', 16)
        self.bandwidth = config.get('bandwidth', 0.5)
        self.oscillator_type = config.get('oscillator_type', 'cnoa_tc')
        self.at_type = config.get('at_type', 'osc')
        self.encoder_type = config.get('encoder_type', 'trans')

        # Loss weights
        self.lambda_loc = config.get('lambda_loc', 0.9)
        self.lambda_time = config.get('lambda_time', 0.4)
        self.lambda_rank = config.get('lambda_rank', 0.6)

        # Get parameters from data_feature
        self.num_locations = data_feature.get('loc_size', data_feature.get('num_locations', 100))
        self.num_users = data_feature.get('uid_size', data_feature.get('num_users', 10))
        self.topic_num = config.get('topic_num', data_feature.get('topic_num', 0))

        # Build model components
        self.embedding_layer = MultimodalContextualEmbedding(
            base_dim=self.base_dim,
            num_locations=self.num_locations,
            num_users=self.num_users,
            bandwidth=self.bandwidth
        )

        self.fc_mapping = nn.Linear(self.num_locations, 80)

        if self.encoder_type == 'trans':
            emb_dim = self.base_dim
            self.positional_encoding = PositionalEncoding(emb_dim=emb_dim)
            self.encoder = LocationTimePair(input_dim=self.base_dim)

        fc_input_dim = self.base_dim + self.base_dim

        if self.at_type != 'none':
            self.at_net = TimeUserPair(
                base_dim=self.base_dim,
                num_users=self.num_users,
                at_type=self.at_type,
                oscillator_type=self.oscillator_type,
                device=self.device
            )
            fc_input_dim += self.base_dim

        if self.topic_num > 0:
            self.user_net = UserLocationPair(
                input_dim=self.topic_num,
                output_dim=self.base_dim
            )
            fc_input_dim += self.base_dim

        self.cnoa = CrossContextAttentiveDecoder(
            query_dim=self.base_dim,
            kv_dim=80,
            embed_dim=80,
            output_dim=self.num_locations,
            oscillator_type=self.oscillator_type,
            num_heads=4,
            device=self.device
        )

        self.fc_layer = NextLocationPrediction(
            input_dim=self.base_dim * 5,
            output_dim=self.num_locations
        )
        self.rank_head = NextLocationPrediction(
            input_dim=self.base_dim * 5,
            output_dim=self.num_locations
        )
        self.out_dropout = nn.Dropout(0.1)

    def forward(self, batch):
        """
        Forward pass of CANOE model.

        Args:
            batch: LibCity Batch object containing:
                - 'current_loc': Location sequence [batch_size, seq_len]
                - 'current_tim': Time sequence [batch_size, seq_len] (0-47 encoding)
                - 'uid': User IDs [batch_size]
                - 'hour_mask': Mask for valid hours [batch_size, seq_len, 24] (optional)
                - 'user_topic_loc': LDA topic vectors [batch_size, topic_num] (optional)

        Returns:
            loc_logits: Location prediction logits [batch_size, num_locations]
            time_logits: Time prediction logits [batch_size, 24]
            rank_logits: Ranking head logits [batch_size, num_locations]
        """
        # Use standard LibCity batch keys directly
        loc_x = batch['current_loc']
        # Convert time encoding: LibCity uses 0-47 (0-23 weekdays, 24-47 weekends)
        # CANOE expects 0-23 hours
        hour_x = batch['current_tim'] % 24
        user_x = batch['uid']

        # Get pre-computed LDA topic vectors if available
        pre_embedded = None
        if self.topic_num > 0:
            # Check batch.data dictionary for optional keys
            if 'user_topic_loc' in batch.data:
                pre_embedded = batch['user_topic_loc']
            else:
                # Create zero placeholder if not available
                pre_embedded = torch.zeros(
                    loc_x.size(0), self.topic_num,
                    device=loc_x.device, dtype=torch.float32
                )

        batch_size, sequence_length = loc_x.shape

        # Get embeddings
        loc_embedded, timeslot_embedded, smoothed_timeslot_embedded, user_embedded = \
            self.embedding_layer(loc_x, loc_x.device)

        time_embedded = timeslot_embedded[hour_x]
        smoothed_time_embedded = smoothed_timeslot_embedded[hour_x]

        lt_embedded = loc_embedded + time_embedded

        # Transformer encoder
        if hasattr(self, 'encoder'):
            future_mask = torch.triu(
                torch.ones(sequence_length, sequence_length, device=lt_embedded.device),
                diagonal=1
            )
            future_mask = future_mask.masked_fill(future_mask == 1, float('-inf')).bool()
            encoder_out = self.encoder(
                self.positional_encoding(lt_embedded * math.sqrt(self.base_dim)),
                src_mask=future_mask
            )
        else:
            encoder_out = lt_embedded

        combined = torch.cat([encoder_out, lt_embedded], dim=-1)
        encoder_out_flat = encoder_out.view(-1, encoder_out.size(-1))

        user_embedded_selected = user_embedded[user_x]

        # Time-User attention
        if hasattr(self, 'at_net'):
            # Check batch.data dictionary for optional hour_mask key
            if 'hour_mask' in batch.data:
                hour_mask = batch['hour_mask']
            else:
                hour_mask = torch.zeros(
                    batch_size, sequence_length, 24,
                    device=loc_x.device, dtype=torch.long
                )

            at_embedded, time_logits = self.at_net(
                timeslot_embedded,
                smoothed_timeslot_embedded,
                user_embedded_selected,
                user_x,
                hour_x,
                hour_mask
            )

            combined = torch.cat([combined, at_embedded], dim=-1)
            combined_emb = torch.cat([combined, at_embedded], dim=-1)
        else:
            time_logits = torch.zeros(
                batch_size, sequence_length, 24,
                device=loc_x.device
            )
            combined_emb = combined

        user_embedded_expanded = user_embedded_selected.unsqueeze(1).repeat(1, sequence_length, 1)
        combined = torch.cat([combined, user_embedded_expanded], dim=-1)
        combined_emb_no_pre = torch.cat([combined_emb, user_embedded_expanded], dim=-1)

        # User-Location (LDA) processing
        if self.topic_num > 0 and pre_embedded is not None:
            pre_embedded_proj = self.user_net(pre_embedded).unsqueeze(1).repeat(1, sequence_length, 1)
            combined = torch.cat([combined, pre_embedded_proj], dim=-1)
        else:
            # Create placeholder for CNOA query
            pre_embedded_proj = torch.zeros(
                batch_size, sequence_length, self.base_dim,
                device=loc_x.device
            )
            # Concatenate placeholder to combined to maintain 80 dims (base_dim * 5)
            combined = torch.cat([combined, pre_embedded_proj], dim=-1)

        # Cross-Context Attentive Decoder
        final_output = self.cnoa(pre_embedded_proj, combined_emb_no_pre, combined_emb_no_pre)

        final_output = self.fc_mapping(final_output)
        residual_output = final_output + combined

        # Prediction heads - extract last sequence position for prediction
        # LibCity expects one prediction per trajectory, not per sequence position
        last_output = residual_output[:, -1, :]  # [batch_size, hidden_dim]
        out = self.fc_layer(last_output)  # [batch_size, num_locations]
        rank = self.rank_head(last_output)  # [batch_size, num_locations]

        # Extract last time logits as well
        time_out = time_logits[:, -1, :]  # [batch_size, 24]

        return out, time_out, rank

    def predict(self, batch):
        """
        Predict next locations for the given batch.

        Args:
            batch: LibCity Batch object

        Returns:
            Location prediction scores [batch_size, num_locations]
        """
        loc_logits, time_logits, rank_logits = self.forward(batch)
        return loc_logits

    def calculate_loss(self, batch):
        """
        Calculate the multi-task loss (location + time + ranking).

        Args:
            batch: LibCity Batch object containing:
                - 'target': Target locations [batch_size]
                - 'target_tim': Target time slots (optional) [batch_size]

        Returns:
            Combined loss tensor
        """
        loc_logits, time_logits, rank_logits = self.forward(batch)

        # Get target locations using standard LibCity key
        # Target is already [batch_size] - one location per trajectory
        loc_y = batch['target'].to(self.device)  # [batch_size]

        # Location loss
        loc_loss = F.cross_entropy(loc_logits, loc_y, reduction='mean')

        # Ranking loss
        rank_loss = F.cross_entropy(rank_logits, loc_y, reduction='mean')

        # Time loss (if time targets available)
        # Check batch.data dictionary for optional target_tim key
        if 'target_tim' in batch.data:
            # Convert time encoding: LibCity uses 0-47, CANOE expects 0-23
            ts_y = (batch['target_tim'] % 24).to(self.device)  # [batch_size]
            time_loss = F.cross_entropy(time_logits, ts_y, reduction='mean')
        else:
            time_loss = torch.tensor(0.0, device=self.device)

        # Combined loss
        total_loss = (self.lambda_loc * loc_loss +
                      self.lambda_time * time_loss +
                      self.lambda_rank * rank_loss)

        return total_loss
