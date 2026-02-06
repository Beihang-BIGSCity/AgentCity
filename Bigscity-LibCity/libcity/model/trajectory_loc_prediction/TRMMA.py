"""
TRMMA: Trajectory Recovery Model with Multi-level Matching and Attention

This model is adapted from the original TRMMA implementation for trajectory
recovery/map matching tasks. The model uses a dual transformer encoder
(GPS + Route) and an attention-based GRU decoder for trajectory recovery.

Original repository: https://github.com/xxx/TRMMA

Key Components:
1. PositionalEncoder - Sinusoidal positional encoding for sequences
2. MultiHeadAttention - Multi-head self-attention mechanism
3. GPSFormer - GPS sequence transformer encoder
4. GRFormer - Route-GPS dual transformer with cross-attention
5. GPSEncoder - Encodes GPS trajectory with temporal features
6. GREncoder - Encodes both GPS and route information
7. DecoderMulti - GRU-based decoder with route attention
8. TRMMA - Main trajectory recovery model

Adaptations for LibCity:
- Inherits from AbstractModel for LibCity compatibility
- Adapted batch input format to LibCity's trajectory batch dictionary
- Implemented predict() and calculate_loss() methods following LibCity conventions
- Uses config.get() and data_feature.get() for parameter extraction
- Added proper device handling for tensor operations
- Removed external dependencies (spatial_func, trajectory_func, etc.)
- Simplified route planning (DA planner) to use pre-computed routes from batch

Limitations compared to original:
- Requires pre-computed routes in the batch data
- Simplified road network feature handling
- No dynamic route planning during inference

The adapted model performs trajectory location prediction with route-aware
attention mechanism for improved accuracy.
"""

import random
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from libcity.model.abstract_model import AbstractModel


def sequence_mask(X, valid_len, value=0.):
    """Mask irrelevant entries in sequences.

    Args:
        X: Input tensor (batch, seq_len)
        valid_len: Valid lengths for each sequence (batch,)
        value: Value to fill in masked positions

    Returns:
        Masked tensor
    """
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def sequence_mask3d(X, valid_len, valid_len2, value=0.):
    """Mask irrelevant entries in 3D sequences.

    Args:
        X: Input tensor (batch, seq_len1, seq_len2)
        valid_len: Valid lengths for dimension 1 (batch,)
        valid_len2: Valid lengths for dimension 2 (batch,)
        value: Value to fill in masked positions

    Returns:
        Masked tensor
    """
    maxlen = X.size(1)
    maxlen2 = X.size(2)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    mask2 = torch.arange((maxlen2), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len2[:, None]
    mask_fin = torch.bmm(mask.float().unsqueeze(-1), mask2.float().unsqueeze(-2)).bool()
    X[~mask_fin] = value
    return X


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding for transformer.

    Args:
        d_model: Model dimension
        max_seq_len: Maximum sequence length
    """

    def __init__(self, d_model, max_seq_len=500):
        super().__init__()
        self.d_model = d_model

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Apply positional encoding.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).to(x.device)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.

    Args:
        heads: Number of attention heads
        d_model: Model dimension
        dropout: Dropout probability
    """

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

    def forward(self, q, k, v, mask=None):
        """Forward pass.

        Args:
            q: Query tensor (batch, seq_len, d_model)
            k: Key tensor (batch, seq_len, d_model)
            v: Value tensor (batch, seq_len, d_model)
            mask: Attention mask (batch, seq_len, seq_len)

        Returns:
            Attended output (batch, seq_len, d_model)
        """
        bs = q.size(0)

        # Linear projections and split into heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # Transpose for attention: (bs, heads, seq_len, d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # Concatenate heads
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        """Scaled dot-product attention."""
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = self.dropout(scores)

        output = torch.matmul(scores, v)
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    Args:
        d_model: Model dimension
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
    """

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


class Norm(nn.Module):
    """Layer normalization.

    Args:
        d_model: Model dimension
        eps: Epsilon for numerical stability
    """

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


class GPSLayer(nn.Module):
    """GPS transformer layer with self-attention.

    Args:
        d_model: Model dimension
        heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff=d_model * 2)
        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x, mask):
        residual = x
        x = self.dropout_1(self.attn(x, x, x, mask))
        x2 = self.norm_1(residual + x)
        x = self.ff(x2)
        return x


class GPSFormer(nn.Module):
    """GPS sequence transformer encoder.

    Args:
        d_model: Model dimension
        N: Number of layers
        heads: Number of attention heads
    """

    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        self.layers = nn.ModuleList([
            GPSLayer(d_model, heads) for _ in range(N)
        ])

    def forward(self, src, mask3d=None):
        x = src
        for i in range(self.N):
            x = self.layers[i](x, mask3d)
        return x


class RouteLayer(nn.Module):
    """Route transformer layer with self-attention and cross-attention to GPS.

    Args:
        d_model: Model dimension
        heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.slf_attn = MultiHeadAttention(heads, d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff=d_model * 2)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, route, gps, route_mask, inter_mask):
        route1 = self.dropout_1(self.slf_attn(route, route, route, route_mask))
        route_out = self.norm_1(route + route1)

        route2 = self.dropout_2(self.attn(route_out, gps, gps, inter_mask))
        route_out2 = self.norm_2(route_out + route2)

        x = self.ff(route_out2)
        return x


class GRLayer(nn.Module):
    """GPS-Route dual layer with GPS encoding and route cross-attention.

    Args:
        d_model: Model dimension
        heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.gps_enc = GPSLayer(d_model, heads, dropout)
        self.route_enc = RouteLayer(d_model, heads, dropout)

    def forward(self, route, route_mask, gps, gps_mask, inter_mask):
        gps_emb = self.gps_enc(gps, gps_mask)
        route_emb = self.route_enc(route, gps_emb, route_mask, inter_mask)
        return route_emb, gps_emb


class GRFormer(nn.Module):
    """GPS-Route dual transformer for encoding GPS and route sequences.

    Args:
        d_model: Model dimension
        N: Number of layers
        heads: Number of attention heads
    """

    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        self.layers = nn.ModuleList([
            GRLayer(d_model, heads) for _ in range(N)
        ])

    def forward(self, src, route, mask3d, route_mask3d, inter_mask):
        x = route
        y = src
        for i in range(self.N):
            x, y = self.layers[i](x, route_mask3d, y, mask3d, inter_mask)
        return x


class Attention(nn.Module):
    """Bahdanau-style attention for decoder.

    Args:
        hid_dim: Hidden dimension
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.attn = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.v = nn.Linear(self.hid_dim, 1, bias=False)

    def forward(self, query, key, value, attn_mask):
        """Calculate attention weights.

        Args:
            query: Query tensor (batch, src_len, hid_dim)
            key: Key tensor (batch, src_len, candi_num, hid_dim)
            value: Value tensor (batch, src_len, candi_num, hid_dim)
            attn_mask: Attention mask

        Returns:
            scores: Attention weights
            weighted: Weighted sum of values
        """
        bs, src_len = query.shape[0], query.shape[1]
        candi_num = key.shape[-2]

        # Repeat query for each candidate
        query = query.unsqueeze(-2).repeat(1, 1, candi_num, 1)

        energy = torch.tanh(self.attn(torch.cat((query, key), dim=-1)))
        attention = self.v(energy).squeeze(-1)
        attention = attention.masked_fill(attn_mask == 0, -1e10)

        scores = F.softmax(attention, dim=-1)
        weighted = torch.bmm(
            scores.reshape(bs * src_len, candi_num).unsqueeze(-2),
            value.reshape(bs * src_len, candi_num, -1)
        ).squeeze(-2)
        weighted = weighted.reshape(bs, src_len, -1)

        return scores, weighted


class GPSEncoder(nn.Module):
    """GPS sequence encoder with transformer and temporal features.

    Args:
        hid_dim: Hidden dimension
        transformer_layers: Number of transformer layers
        heads: Number of attention heads
        pro_features_flag: Whether to use temporal features
        pro_input_dim: Temporal feature input dimension
        pro_output_dim: Temporal feature output dimension
    """

    def __init__(self, hid_dim, transformer_layers, heads,
                 pro_features_flag=True, pro_input_dim=48, pro_output_dim=64):
        super().__init__()
        self.pro_features_flag = pro_features_flag
        self.hid_dim = hid_dim

        self.transformer = GPSFormer(hid_dim, transformer_layers, heads=heads)

        if self.pro_features_flag:
            self.temporal = nn.Embedding(pro_input_dim, pro_output_dim)
            self.fc_hid = nn.Linear(hid_dim + pro_output_dim, hid_dim)

    def forward(self, src, src_len, pro_features):
        """Encode GPS sequence.

        Args:
            src: GPS sequence (seq_len, batch, input_dim)
            src_len: Sequence lengths (batch,)
            pro_features: Temporal features (batch,)

        Returns:
            outputs: Encoder outputs (seq_len, batch, hid_dim)
            hidden: Final hidden state (1, batch, hid_dim)
        """
        bs = src.size(1)
        max_src_len = src.size(0)

        mask3d = torch.ones(bs, max_src_len, max_src_len, device=src.device)
        mask2d = torch.ones(bs, max_src_len, device=src.device)

        mask3d = sequence_mask3d(mask3d, src_len, src_len)
        mask2d = sequence_mask(mask2d, src_len).transpose(0, 1).unsqueeze(-1).repeat(1, 1, self.hid_dim)

        src = src.transpose(0, 1)
        outputs = self.transformer(src, mask3d)
        outputs = outputs.transpose(0, 1)

        outputs = outputs * mask2d
        hidden = torch.sum(outputs, dim=0) / src_len.unsqueeze(-1).repeat(1, self.hid_dim)
        hidden = hidden.unsqueeze(0)

        if self.pro_features_flag:
            extra_emb = self.temporal(pro_features)
            extra_emb = extra_emb.unsqueeze(0)
            hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=-1)))

        return outputs, hidden


class GREncoder(nn.Module):
    """GPS-Route dual encoder with transformer.

    Args:
        hid_dim: Hidden dimension
        transformer_layers: Number of transformer layers
        heads: Number of attention heads
        pro_features_flag: Whether to use temporal features
        pro_input_dim: Temporal feature input dimension
        pro_output_dim: Temporal feature output dimension
    """

    def __init__(self, hid_dim, transformer_layers, heads,
                 pro_features_flag=True, pro_input_dim=48, pro_output_dim=64):
        super().__init__()
        self.hid_dim = hid_dim
        self.pro_features_flag = pro_features_flag

        self.transformer = GRFormer(hid_dim, transformer_layers, heads=heads)

        if self.pro_features_flag:
            self.temporal = nn.Embedding(pro_input_dim, pro_output_dim)
            self.fc_hid = nn.Linear(hid_dim + pro_output_dim, hid_dim)

    def forward(self, src, src_len, route, route_len, pro_features):
        """Encode GPS and route sequences.

        Args:
            src: GPS sequence (seq_len, batch, input_dim)
            src_len: GPS sequence lengths (batch,)
            route: Route sequence (route_len, batch, input_dim)
            route_len: Route sequence lengths (batch,)
            pro_features: Temporal features (batch,)

        Returns:
            outputs: Route encoder outputs (route_len, batch, hid_dim)
            hidden: Final hidden state (1, batch, hid_dim)
        """
        bs = src.size(1)
        src_max_len = src.size(0)
        route_max_len = route.size(0)

        mask3d = torch.ones(bs, src_max_len, src_max_len, device=src.device)
        route_mask3d = torch.ones(bs, route_max_len, route_max_len, device=src.device)
        route_mask2d = torch.ones(bs, route_max_len, device=src.device)
        inter_mask = torch.ones(bs, route_max_len, src_max_len, device=src.device)

        mask3d = sequence_mask3d(mask3d, src_len, src_len)
        route_mask3d = sequence_mask3d(route_mask3d, route_len, route_len)
        route_mask2d = sequence_mask(route_mask2d, route_len).transpose(0, 1).unsqueeze(-1).repeat(1, 1, self.hid_dim)
        inter_mask = sequence_mask3d(inter_mask, route_len, src_len)

        src = src.transpose(0, 1)
        route = route.transpose(0, 1)
        outputs = self.transformer(src, route, mask3d, route_mask3d, inter_mask)
        outputs = outputs.transpose(0, 1)

        outputs = outputs * route_mask2d
        hidden = torch.sum(outputs, dim=0) / route_len.unsqueeze(-1).repeat(1, self.hid_dim)
        hidden = hidden.unsqueeze(0)

        if self.pro_features_flag:
            extra_emb = self.temporal(pro_features)
            extra_emb = extra_emb.unsqueeze(0)
            hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=-1)))

        return outputs, hidden


class DecoderMulti(nn.Module):
    """GRU-based decoder with route attention for trajectory recovery.

    Args:
        hid_dim: Hidden dimension
        id_size: Vocabulary size for road segment IDs
        dest_type: Destination embedding type (0, 1, or 2)
        rate_flag: Whether to predict position rate
        prog_flag: Whether to use progressive decoding
        rid_feats_flag: Whether to use road segment features
        rid_fea_dim: Road segment feature dimension
    """

    def __init__(self, hid_dim, id_size, dest_type=1, rate_flag=False,
                 prog_flag=False, rid_feats_flag=False, rid_fea_dim=8):
        super().__init__()

        self.id_size = id_size
        self.emb_id = None  # Updated from main model
        self.dest_type = dest_type
        self.rate_flag = rate_flag
        self.prog_flag = prog_flag
        self.rid_feats_flag = rid_feats_flag
        self.rid_fea_dim = rid_fea_dim
        self.hid_dim = hid_dim

        rnn_input_dim = hid_dim
        if self.rid_feats_flag:
            rnn_input_dim += rid_fea_dim
        if self.rate_flag:
            rnn_input_dim += 1
        if self.dest_type in [1, 2]:
            rnn_input_dim += hid_dim
            if self.rid_feats_flag:
                rnn_input_dim += rid_fea_dim
            if self.rate_flag:
                rnn_input_dim += 1

        self.rnn = nn.GRU(rnn_input_dim, hid_dim)
        self.attn_route = Attention(hid_dim)

        if self.rate_flag:
            fc_rate_out_input_dim = hid_dim + hid_dim
            self.fc_rate_out = nn.Sequential(
                nn.Linear(fc_rate_out_input_dim, hid_dim * 2),
                nn.ReLU(),
                nn.Linear(hid_dim * 2, 1),
                nn.Sigmoid()
            )

    def decoding_step(self, input_id, input_rate, hidden, route_outputs,
                      route_attn_mask, d_rids, d_rates, rid_features_dict,
                      dt=None, observed_emb=None, observed_mask=None):
        """Single decoding step.

        Args:
            input_id: Input road segment IDs (batch,)
            input_rate: Input position rates (batch, 1)
            hidden: Hidden state (1, batch, hid_dim)
            route_outputs: Route encoder outputs (route_len, batch, hid_dim)
            route_attn_mask: Route attention mask (batch, route_len)
            d_rids: Destination road segment IDs (batch,)
            d_rates: Destination position rates (batch, 1)
            rid_features_dict: Road segment features tensor

        Returns:
            prediction_id: Road segment prediction scores (batch, route_len)
            prediction_rate: Position rate predictions (batch, 1)
            hidden: Updated hidden state (1, batch, hid_dim)
        """
        rnn_input = self.emb_id[input_id]
        if self.rid_feats_flag:
            if rid_features_dict is not None:
                input_feats = rid_features_dict[input_id]
            else:
                # Pad with zeros when rid_features are expected but not available
                input_feats = torch.zeros(
                    input_id.size(0), self.rid_fea_dim,
                    device=rnn_input.device, dtype=rnn_input.dtype
                )
            rnn_input = torch.cat([rnn_input, input_feats], dim=-1)
        if self.rate_flag:
            rnn_input = torch.cat((rnn_input, input_rate), dim=-1)
        if self.dest_type in [1, 2]:
            embed_drids = self.emb_id[d_rids]
            rnn_input = torch.cat((rnn_input, embed_drids), dim=-1)
            if self.rid_feats_flag:
                if rid_features_dict is not None:
                    dest_feats = rid_features_dict[d_rids]
                else:
                    # Pad with zeros when rid_features are expected but not available
                    dest_feats = torch.zeros(
                        d_rids.size(0), self.rid_fea_dim,
                        device=rnn_input.device, dtype=rnn_input.dtype
                    )
                rnn_input = torch.cat([rnn_input, dest_feats], dim=-1)
            if self.rate_flag:
                rnn_input = torch.cat((rnn_input, d_rates), dim=-1)

        rnn_input = rnn_input.unsqueeze(0)
        output, hidden = self.rnn(rnn_input, hidden)

        query = hidden.permute(1, 0, 2)
        key = route_outputs.permute(1, 0, 2).unsqueeze(1)
        scores, weighted = self.attn_route(query, key, key, route_attn_mask.unsqueeze(1))
        prediction_id = scores.squeeze(1).masked_fill(route_attn_mask == 0, 0)
        weighted = weighted.permute(1, 0, 2)

        if self.rate_flag:
            rate_input = torch.cat((hidden, weighted), dim=-1).squeeze(0)
            prediction_rate = self.fc_rate_out(rate_input)
        else:
            prediction_rate = torch.ones(
                (prediction_id.shape[0], 1),
                dtype=torch.float32,
                device=hidden.device
            ) / 2

        return prediction_id, prediction_rate, hidden

    def forward(self, max_trg_len, batch_size, trg_id, trg_rate, trg_len,
                hidden, rid_features_dict, routes, route_outputs,
                route_attn_mask, d_rids, d_rates, teacher_forcing_ratio):
        """Forward pass for decoding.

        Args:
            max_trg_len: Maximum target length
            batch_size: Batch size
            trg_id: Target road segment IDs (seq_len, batch)
            trg_rate: Target position rates (seq_len, batch, 1)
            trg_len: Target sequence lengths
            hidden: Initial hidden state (1, batch, hid_dim)
            rid_features_dict: Road segment features tensor
            routes: Route sequences (route_len, batch)
            route_outputs: Route encoder outputs (route_len, batch, hid_dim)
            route_attn_mask: Route attention mask (batch, route_len)
            d_rids: Destination road segment IDs (batch,)
            d_rates: Destination position rates (batch, 1)
            teacher_forcing_ratio: Teacher forcing probability

        Returns:
            outputs_id: Road segment prediction scores (seq_len, batch, route_len)
            outputs_rate: Position rate predictions (seq_len, batch, 1)
        """
        routes = routes.permute(1, 0)  # (batch, route_len)
        outputs_id = torch.zeros([max_trg_len, batch_size, routes.shape[1]], device=hidden.device)
        rate_out_dim = 1
        outputs_rate = torch.zeros([max_trg_len, batch_size, rate_out_dim], device=hidden.device)

        # First input
        input_id = trg_id[0, :]
        input_rate = trg_rate[0, :]

        for t in range(1, max_trg_len):
            teacher_force = random.random() < teacher_forcing_ratio

            prediction_id, prediction_rate, hidden = self.decoding_step(
                input_id, input_rate, hidden, route_outputs, route_attn_mask,
                d_rids, d_rates, rid_features_dict
            )

            if teacher_forcing_ratio == -1 and self.prog_flag:
                for i in range(batch_size):
                    if t < trg_len[i]:
                        prev_idx_matches = (input_id[i] == routes[i]).nonzero(as_tuple=True)[0]
                        if len(prev_idx_matches) > 0:
                            prev_idx = prev_idx_matches[0]
                            tmp_flag = True
                            while tmp_flag:
                                cur_idx = prediction_id[i].argmax()
                                if cur_idx < prev_idx:
                                    prediction_id[i, cur_idx] = 1e-6
                                else:
                                    tmp_flag = False

            outputs_id[t] = prediction_id
            outputs_rate[t] = prediction_rate

            if teacher_force:
                input_id = trg_id[t]
                input_rate = trg_rate[t]
            else:
                input_id = (F.one_hot(prediction_id.argmax(dim=1), routes.shape[1]) * routes).sum(-1)
                input_rate = prediction_rate

        mask_trg = torch.ones([batch_size, max_trg_len], device=outputs_id.device)
        mask_trg = sequence_mask(mask_trg, torch.tensor(trg_len, device=outputs_id.device))
        outputs_rate = outputs_rate.permute(1, 0, 2)  # (batch, seq_len, 1)
        outputs_rate = outputs_rate.masked_fill(mask_trg.unsqueeze(-1) == 0, 0)
        outputs_rate = outputs_rate.permute(1, 0, 2)

        return outputs_id, outputs_rate


class TrajRecoveryModule(nn.Module):
    """Core trajectory recovery module with dual encoder and decoder.

    This is the internal module that handles the actual model computation.

    Args:
        id_size: Vocabulary size for road segment IDs
        hid_dim: Hidden dimension
        id_emb_dim: Road segment embedding dimension
        transformer_layers: Number of transformer layers
        heads: Number of attention heads
        learn_pos: Whether to use learned positional embeddings
        da_route_flag: Whether to use route encoder
        srcseg_flag: Whether to use source segment features
        rid_feats_flag: Whether to use road segment features
        rid_fea_dim: Road segment feature dimension
        dest_type: Destination embedding type
        rate_flag: Whether to predict position rate
        prog_flag: Whether to use progressive decoding
        pro_features_flag: Whether to use temporal features
        pro_input_dim: Temporal feature input dimension
        pro_output_dim: Temporal feature output dimension
    """

    def __init__(self, id_size, hid_dim, id_emb_dim, transformer_layers, heads,
                 learn_pos=True, da_route_flag=True, srcseg_flag=False,
                 rid_feats_flag=False, rid_fea_dim=8, dest_type=1,
                 rate_flag=False, prog_flag=False, pro_features_flag=True,
                 pro_input_dim=48, pro_output_dim=64):
        super().__init__()
        self.srcseg_flag = srcseg_flag
        self.hid_dim = hid_dim
        self.da_route_flag = da_route_flag
        self.learn_pos = learn_pos
        self.rid_feats_flag = rid_feats_flag
        self.rid_fea_dim = rid_fea_dim

        # Road segment embedding
        self.emb_id = nn.Parameter(torch.rand(id_size, id_emb_dim))

        if self.learn_pos:
            max_input_length = 500
            self.pos_embedding_gps = nn.Embedding(max_input_length, hid_dim)
            self.pos_embedding_route = nn.Embedding(max_input_length, hid_dim)

        # GPS input projection
        input_dim_gps = 3
        if self.learn_pos:
            input_dim_gps += hid_dim
        if self.srcseg_flag:
            input_dim_gps += hid_dim + 1
        self.fc_in_gps = nn.Linear(input_dim_gps, hid_dim)

        # Route input projection
        input_dim_route = hid_dim
        if self.learn_pos:
            input_dim_route += hid_dim
        if self.rid_feats_flag:
            input_dim_route += rid_fea_dim
        self.fc_in_route = nn.Linear(input_dim_route, hid_dim)

        # Encoder
        if self.da_route_flag:
            self.encoder = GREncoder(
                hid_dim, transformer_layers, heads,
                pro_features_flag, pro_input_dim, pro_output_dim
            )
        else:
            self.encoder = GPSEncoder(
                hid_dim, transformer_layers, heads,
                pro_features_flag, pro_input_dim, pro_output_dim
            )

        # Decoder
        self.decoder = DecoderMulti(
            hid_dim, id_size, dest_type, rate_flag, prog_flag,
            rid_feats_flag, rid_fea_dim
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following Keras conventions."""
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, src, src_len, trg_id, trg_rate, trg_len, pro_features,
                rid_features_dict, da_routes, da_lengths, da_pos,
                src_seg_seqs, src_seg_feats, d_rids, d_rates, teacher_forcing_ratio):
        """Forward pass.

        Args:
            src: GPS sequence (seq_len, batch, 3) - lat, lng, time
            src_len: GPS sequence lengths
            trg_id: Target road segment IDs (seq_len, batch)
            trg_rate: Target position rates (seq_len, batch, 1)
            trg_len: Target sequence lengths
            pro_features: Temporal features (batch,)
            rid_features_dict: Road segment features tensor
            da_routes: Route sequences (route_len, batch)
            da_lengths: Route sequence lengths
            da_pos: Route position indices (route_len, batch)
            src_seg_seqs: Source segment sequences
            src_seg_feats: Source segment features
            d_rids: Destination road segment IDs (batch,)
            d_rates: Destination position rates (batch, 1)
            teacher_forcing_ratio: Teacher forcing probability

        Returns:
            final_outputs_id: Road segment predictions (seq_len-2, batch, route_len)
            final_outputs_rate: Position rate predictions (seq_len-2, batch, 1)
        """
        max_trg_len = trg_id.size(0)
        batch_size = trg_id.size(1)

        # Share embedding with decoder
        self.decoder.emb_id = self.emb_id

        # GPS embedding
        gps_emb = src.float()
        if self.learn_pos:
            gps_pos = src[:, :, -1].long()
            gps_pos_emb = self.pos_embedding_gps(gps_pos)
            gps_emb = torch.cat([gps_emb, gps_pos_emb], dim=-1)
        if self.srcseg_flag:
            seg_emb = self.emb_id[src_seg_seqs]
            gps_emb = torch.cat((gps_emb, seg_emb, src_seg_feats), dim=-1)
        gps_in = self.fc_in_gps(gps_emb)
        gps_in_lens = torch.tensor(src_len, device=src.device)

        # Encoding
        if self.da_route_flag:
            route_emb = self.emb_id[da_routes]
            if self.learn_pos:
                route_pos_emb = self.pos_embedding_route(da_pos)
                route_emb = torch.cat([route_emb, route_pos_emb], dim=-1)
            if self.rid_feats_flag:
                if rid_features_dict is not None:
                    route_feats = rid_features_dict[da_routes]
                else:
                    # Pad with zeros when rid_features are expected but not available
                    route_feats = torch.zeros(
                        route_emb.size(0), route_emb.size(1), self.rid_fea_dim,
                        device=route_emb.device, dtype=route_emb.dtype
                    )
                route_emb = torch.cat([route_emb, route_feats], dim=-1)
            route_in = self.fc_in_route(route_emb)
            route_in_lens = torch.tensor(da_lengths, device=src.device)
            route_outputs, hiddens = self.encoder(gps_in, gps_in_lens, route_in, route_in_lens, pro_features)
        else:
            _, hiddens = self.encoder(gps_in, gps_in_lens, pro_features)
            route_in_lens = torch.tensor(da_lengths, device=src.device)
            route_outputs = self.emb_id[da_routes]

        route_attn_mask = torch.ones(batch_size, max(da_lengths), device=src.device)
        route_attn_mask = sequence_mask(route_attn_mask, route_in_lens)

        # Decoding
        outputs_id, outputs_rate = self.decoder(
            max_trg_len, batch_size, trg_id, trg_rate, trg_len, hiddens,
            rid_features_dict, da_routes, route_outputs, route_attn_mask,
            d_rids, d_rates, teacher_forcing_ratio
        )

        # Remove first and last elements
        final_outputs_id = outputs_id[1:-1]
        final_outputs_rate = outputs_rate[1:-1]

        return final_outputs_id, final_outputs_rate


class TRMMA(AbstractModel):
    """
    TRMMA: Trajectory Recovery Model with Multi-level Matching and Attention

    This is a LibCity-compatible trajectory recovery model that uses a dual
    transformer encoder (GPS + Route) and an attention-based GRU decoder.

    The model takes sparse GPS observations and predicts the complete trajectory
    on the road network, including road segment IDs and optional position ratios.

    Args:
        config (dict): Configuration dictionary containing model hyperparameters
        data_feature (dict): Data features including vocabulary sizes

    Required config parameters:
        - hid_dim: Hidden dimension (default: 256)
        - id_emb_dim: Road segment embedding dimension (default: 128)
        - transformer_layers: Number of transformer layers (default: 2)
        - heads: Number of attention heads (default: 4)
        - dropout: Dropout probability (default: 0.1)
        - pro_features_flag: Whether to use temporal features (default: True)
        - pro_input_dim: Temporal feature input dimension (default: 48)
        - pro_output_dim: Temporal feature output dimension (default: 64)
        - learn_pos: Whether to use learned positional embeddings (default: True)
        - da_route_flag: Whether to use route encoder (default: True)
        - srcseg_flag: Whether to use source segment features (default: False)
        - rid_feats_flag: Whether to use road segment features (default: False)
        - rid_fea_dim: Road segment feature dimension (default: 8)
        - dest_type: Destination embedding type (default: 1)
        - rate_flag: Whether to predict position rate (default: False)
        - prog_flag: Whether to use progressive decoding (default: False)
        - lambda1: Loss weight for ID prediction (default: 1.0)
        - lambda2: Loss weight for rate prediction (default: 0.0)
        - teacher_forcing_ratio: Teacher forcing ratio for training (default: 0.5)

    Required data_feature:
        - loc_size/id_size: Number of road segment IDs in vocabulary
        - loc_pad/id_pad: Padding index
    """

    def __init__(self, config, data_feature):
        super(TRMMA, self).__init__(config, data_feature)

        self._logger = logging.getLogger(__name__)
        self.device = config.get('device', 'cpu')

        # Data dimensions
        self.id_size = data_feature.get('loc_size', data_feature.get('id_size', 10000))
        self.id_pad = data_feature.get('loc_pad', data_feature.get('id_pad', 0))

        # Model hyperparameters
        self.hid_dim = config.get('hid_dim', 256)
        self.id_emb_dim = config.get('id_emb_dim', 128)
        self.transformer_layers = config.get('transformer_layers', 2)
        self.heads = config.get('heads', 4)
        self.dropout = config.get('dropout', 0.1)

        # Feature flags
        self.pro_features_flag = config.get('pro_features_flag', True)
        self.pro_input_dim = config.get('pro_input_dim', 48)
        self.pro_output_dim = config.get('pro_output_dim', 64)
        self.learn_pos = config.get('learn_pos', True)
        self.da_route_flag = config.get('da_route_flag', True)
        self.srcseg_flag = config.get('srcseg_flag', False)
        self.rid_feats_flag = config.get('rid_feats_flag', False)
        self.rid_fea_dim = config.get('rid_fea_dim', 8)

        # Decoder settings
        self.dest_type = config.get('dest_type', 1)
        self.rate_flag = config.get('rate_flag', False)
        self.prog_flag = config.get('prog_flag', False)

        # Loss weights
        self.lambda1 = config.get('lambda1', 1.0)
        self.lambda2 = config.get('lambda2', 0.0)

        # Training settings
        self.teacher_forcing_ratio = config.get('teacher_forcing_ratio', 0.5)
        self.evaluate_method = config.get('evaluate_method', 'popularity')

        # Standard mode flag - will be set based on data format detected
        # True: Standard LibCity POI prediction format (current_loc + target)
        # False: Specialized trajectory recovery format (src + trg_id + da_route)
        self.standard_mode = True  # Default to standard mode, will adapt if specialized data is found

        # Build model
        self._build_model()

        self._logger.info(
            f"TRMMA model initialized with id_size={self.id_size}, "
            f"hid_dim={self.hid_dim}, transformer_layers={self.transformer_layers}"
        )

    def _build_model(self):
        """Build the trajectory recovery module."""
        self.model = TrajRecoveryModule(
            id_size=self.id_size,
            hid_dim=self.hid_dim,
            id_emb_dim=self.id_emb_dim,
            transformer_layers=self.transformer_layers,
            heads=self.heads,
            learn_pos=self.learn_pos,
            da_route_flag=self.da_route_flag,
            srcseg_flag=self.srcseg_flag,
            rid_feats_flag=self.rid_feats_flag,
            rid_fea_dim=self.rid_fea_dim,
            dest_type=self.dest_type,
            rate_flag=self.rate_flag,
            prog_flag=self.prog_flag,
            pro_features_flag=self.pro_features_flag,
            pro_input_dim=self.pro_input_dim,
            pro_output_dim=self.pro_output_dim
        )

    def _extract_batch_data(self, batch):
        """Extract and preprocess data from LibCity batch.

        Args:
            batch: LibCity Batch object

        Returns:
            Dictionary containing preprocessed tensors
        """
        data = {}

        # GPS source sequence - expected shape: (batch, seq_len, 3) or (batch, seq_len) for IDs
        # Contains: lat, lng, normalized_time OR location IDs
        if 'current_loc' in batch.data:
            src = batch['current_loc']
            if not isinstance(src, torch.Tensor):
                src = torch.tensor(src)
            src = src.to(self.device)

            # Check if it's location IDs (2D) or GPS coordinates (3D)
            if src.dim() == 2:
                # It's location IDs - need to convert to pseudo-GPS format
                batch_size, seq_len = src.shape
                # Create pseudo-GPS: [lat, lng, time]
                # Use zeros for lat/lng (will be learned via embeddings)
                # Use sequential time steps
                src_gps = torch.zeros(batch_size, seq_len, 3, device=self.device, dtype=torch.float)
                src_gps[:, :, 2] = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1).float() / max(seq_len, 1)
                src = src_gps

            # Transpose to (seq_len, batch, 3)
            src = src.transpose(0, 1)
            data['src'] = src
            data['src_len'] = [src.size(0)] * src.size(1)
        elif 'src_grid_seq' in batch.data:
            src = batch['src_grid_seq']
            if not isinstance(src, torch.Tensor):
                src = torch.tensor(src, dtype=torch.float32)
            src = src.to(self.device)
            src = src.transpose(0, 1)
            data['src'] = src
            data['src_len'] = [src.size(0)] * src.size(1)

        # Target road segment IDs - expected shape: (batch, seq_len)
        if 'target_loc' in batch.data:
            trg_id = batch['target_loc']
            if not isinstance(trg_id, torch.Tensor):
                trg_id = torch.tensor(trg_id, dtype=torch.long)
            trg_id = trg_id.to(self.device)
            trg_id = trg_id.transpose(0, 1)
            data['trg_id'] = trg_id
            data['trg_len'] = [trg_id.size(0)] * trg_id.size(1)
        elif 'trg_rid' in batch.data:
            trg_id = batch['trg_rid']
            if not isinstance(trg_id, torch.Tensor):
                trg_id = torch.tensor(trg_id, dtype=torch.long)
            trg_id = trg_id.to(self.device)
            trg_id = trg_id.transpose(0, 1)
            data['trg_id'] = trg_id
            data['trg_len'] = [trg_id.size(0)] * trg_id.size(1)
        elif 'target' in batch.data:
            # Fallback for standard LibCity format (single next-location prediction)
            target = batch['target']
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target, dtype=torch.long)
            target = target.to(self.device)

            # Get last current_loc as start of target sequence
            src_loc = batch['current_loc']
            if not isinstance(src_loc, torch.Tensor):
                src_loc = torch.tensor(src_loc, dtype=torch.long)
            src_loc = src_loc.to(self.device)

            # Handle both 2D (batch, seq) and 3D (batch, seq, feat) current_loc
            if src_loc.dim() == 3:
                # If 3D, assume first feature is location ID or use zeros
                last_loc = torch.zeros(src_loc.size(0), dtype=torch.long, device=self.device)
                second_last_loc = torch.zeros(src_loc.size(0), dtype=torch.long, device=self.device)
            else:
                last_loc = src_loc[:, -1]
                # Get second to last location if available
                if src_loc.size(1) > 1:
                    second_last_loc = src_loc[:, -2]
                else:
                    second_last_loc = last_loc

            # Create target sequence: [second_last, last_loc, target] - shape (3, batch)
            # This ensures after removing first and last, we have 1 element to predict
            trg_id = torch.stack([second_last_loc, last_loc, target], dim=0)
            data['trg_id'] = trg_id
            data['trg_len'] = [3] * trg_id.size(1)

            # Create dummy target rates (all 0.5 = middle of segment)
            data['trg_rate'] = torch.full_like(trg_id.unsqueeze(-1), 0.5, dtype=torch.float)

        # Target position rates - expected shape: (batch, seq_len, 1)
        # Check if trg_rate was already set (e.g., by 'target' fallback branch)
        if 'trg_rate' not in data:
            if 'trg_rate' in batch.data:
                trg_rate = batch['trg_rate']
                if not isinstance(trg_rate, torch.Tensor):
                    trg_rate = torch.tensor(trg_rate, dtype=torch.float32)
                trg_rate = trg_rate.to(self.device)
                if trg_rate.dim() == 2:
                    trg_rate = trg_rate.unsqueeze(-1)
                trg_rate = trg_rate.transpose(0, 1)
                data['trg_rate'] = trg_rate
            elif 'trg_id' in data:
                # Default: no rate information
                data['trg_rate'] = torch.zeros(
                    data['trg_id'].size(0), data['trg_id'].size(1), 1,
                    device=self.device
                )

        # Temporal features (hour of day)
        if 'pro_features' in batch.data:
            pro_features = batch['pro_features']
            if not isinstance(pro_features, torch.Tensor):
                pro_features = torch.tensor(pro_features, dtype=torch.long)
            data['pro_features'] = pro_features.to(self.device)
        elif 'current_tim' in batch.data:
            tim = batch['current_tim']
            if not isinstance(tim, torch.Tensor):
                tim = torch.tensor(tim, dtype=torch.long)
            # Use first time value as temporal feature
            data['pro_features'] = tim[:, 0].to(self.device) % self.pro_input_dim
        else:
            # Default temporal feature
            data['pro_features'] = torch.zeros(
                data['src'].size(1), dtype=torch.long, device=self.device
            )

        # DA Routes - expected shape: (batch, route_len)
        if 'da_route' in batch.data:
            da_routes = batch['da_route']
            if not isinstance(da_routes, torch.Tensor):
                da_routes = torch.tensor(da_routes, dtype=torch.long)
            da_routes = da_routes.to(self.device)
            da_routes = da_routes.transpose(0, 1)
            data['da_routes'] = da_routes
            data['da_lengths'] = [da_routes.size(0)] * da_routes.size(1)
            # Position indices for routes
            data['da_pos'] = torch.arange(
                da_routes.size(0), device=self.device
            ).unsqueeze(1).expand(-1, da_routes.size(1))
        elif 'trg_id' in data:
            # Use target as route (simplified)
            data['da_routes'] = data['trg_id'].clone()
            data['da_lengths'] = data['trg_len']
            data['da_pos'] = torch.arange(
                data['da_routes'].size(0), device=self.device
            ).unsqueeze(1).expand(-1, data['da_routes'].size(1))

        # Fallback: use current_loc sequence as route if da_routes still not set
        if 'da_routes' not in data and 'current_loc' in batch.data:
            route_data = batch['current_loc']
            if not isinstance(route_data, torch.Tensor):
                route_data = torch.tensor(route_data, dtype=torch.long)
            route_data = route_data.to(self.device)

            # If 2D (location IDs), use directly
            if route_data.dim() == 2:
                # (batch, seq_len) -> (seq_len, batch)
                da_routes = route_data.transpose(0, 1)
            else:
                # If 3D (GPS coordinates), create dummy route IDs
                batch_size = route_data.size(0)
                seq_len = route_data.size(1)
                da_routes = torch.zeros(seq_len, batch_size, dtype=torch.long, device=self.device)

            data['da_routes'] = da_routes
            data['da_lengths'] = [da_routes.size(0)] * da_routes.size(1)
            data['da_pos'] = torch.arange(
                da_routes.size(0), device=self.device
            ).unsqueeze(1).expand(-1, da_routes.size(1))

        # Source segment sequences (optional)
        if 'src_seg_seq' in batch.data:
            src_seg_seqs = batch['src_seg_seq']
            if not isinstance(src_seg_seqs, torch.Tensor):
                src_seg_seqs = torch.tensor(src_seg_seqs, dtype=torch.long)
            data['src_seg_seqs'] = src_seg_seqs.to(self.device).transpose(0, 1)
        else:
            data['src_seg_seqs'] = None

        # Source segment features (optional)
        if 'src_seg_feat' in batch.data:
            src_seg_feats = batch['src_seg_feat']
            if not isinstance(src_seg_feats, torch.Tensor):
                src_seg_feats = torch.tensor(src_seg_feats, dtype=torch.float32)
            data['src_seg_feats'] = src_seg_feats.to(self.device).transpose(0, 1)
        else:
            data['src_seg_feats'] = None

        # Destination road segment ID and rate
        if 'trg_id' in data:
            data['d_rids'] = data['trg_id'][-1, :]
            data['d_rates'] = data['trg_rate'][-1, :]
        else:
            batch_size = data['src'].size(1)
            data['d_rids'] = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            data['d_rates'] = torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device)

        # Road segment features (optional)
        if 'rid_features' in batch.data:
            rid_features_dict = batch['rid_features']
            if not isinstance(rid_features_dict, torch.Tensor):
                rid_features_dict = torch.tensor(rid_features_dict, dtype=torch.float32)
            data['rid_features_dict'] = rid_features_dict.to(self.device)
        else:
            data['rid_features_dict'] = None

        # Labels for attention (optional)
        if 'label' in batch.data:
            label = batch['label']
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label, dtype=torch.float32)
            data['label'] = label.to(self.device)
        else:
            data['label'] = None

        return data

    def forward(self, batch, teacher_forcing_ratio=None):
        """Forward pass for trajectory recovery.

        Args:
            batch: LibCity Batch object containing trajectory data
            teacher_forcing_ratio: Override teacher forcing ratio

        Returns:
            outputs_id: Road segment prediction scores
            outputs_rate: Position rate predictions
        """
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = self.teacher_forcing_ratio if self.training else 0.0

        # Extract data from batch
        data = self._extract_batch_data(batch)

        # Forward through model
        outputs_id, outputs_rate = self.model(
            src=data['src'],
            src_len=data['src_len'],
            trg_id=data['trg_id'],
            trg_rate=data['trg_rate'],
            trg_len=data['trg_len'],
            pro_features=data['pro_features'],
            rid_features_dict=data['rid_features_dict'],
            da_routes=data['da_routes'],
            da_lengths=data['da_lengths'],
            da_pos=data['da_pos'],
            src_seg_seqs=data['src_seg_seqs'],
            src_seg_feats=data['src_seg_feats'],
            d_rids=data['d_rids'],
            d_rates=data['d_rates'],
            teacher_forcing_ratio=teacher_forcing_ratio
        )

        return outputs_id, outputs_rate

    def predict(self, batch):
        """Prediction method for LibCity evaluation.

        Args:
            batch: Input batch dictionary

        Returns:
            POI prediction scores (batch, loc_size) for next location
        """
        self.eval()
        with torch.no_grad():
            outputs_id, outputs_rate = self.forward(batch, teacher_forcing_ratio=0.0)

            # Extract data for route mapping
            data = self._extract_batch_data(batch)
            da_routes = data['da_routes'].permute(1, 0)  # (batch, route_len)

            batch_size = outputs_id.size(1)

            # Convert route-level predictions to full vocabulary predictions
            # Get prediction at first timestep (index 0 after removing first/last)
            if outputs_id.size(0) > 0:
                pred_scores = outputs_id[0]  # (batch, route_len)
            else:
                pred_scores = torch.zeros(batch_size, da_routes.size(1), device=self.device)

            # Create full vocabulary scores
            full_scores = torch.zeros(batch_size, self.id_size, device=self.device)
            full_scores.fill_(-float('inf'))

            # Scatter route scores to full vocabulary
            for i in range(batch_size):
                route_ids = da_routes[i]
                route_scores = pred_scores[i]
                for j, rid in enumerate(route_ids):
                    if rid != self.id_pad:
                        full_scores[i, rid] = route_scores[j]

            # Apply log softmax
            scores = F.log_softmax(full_scores, dim=-1)

            if self.evaluate_method == 'sample':
                if 'neg_loc' in batch.data:
                    pos_neg_index = torch.cat(
                        (batch['target'].unsqueeze(1), batch['neg_loc']), dim=1
                    )
                    scores = torch.gather(scores, 1, pos_neg_index)

            return scores

    def calculate_loss(self, batch):
        """Calculate training loss.

        Args:
            batch: LibCity Batch object containing trajectory data

        Returns:
            Combined loss (ID prediction + rate prediction)
        """
        # Forward pass with teacher forcing
        outputs_id, outputs_rate = self.forward(batch)

        # Extract label and target data
        data = self._extract_batch_data(batch)

        # ID prediction loss
        if data['label'] is not None:
            # Use pre-computed label for attention supervision
            label = data['label']
            # label shape: (batch, seq_len-2, route_len)
            # outputs_id shape: (seq_len-2, batch, route_len)
            label = label.permute(1, 0, 2)  # (seq_len-2, batch, route_len)

            # Binary cross entropy for attention scores
            loss_id = F.binary_cross_entropy(
                outputs_id.clamp(1e-7, 1-1e-7),
                label.clamp(0, 1),
                reduction='mean'
            )
        else:
            # Use target IDs for cross-entropy loss
            trg_id = data['trg_id']
            da_routes = data['da_routes']

            # Create target indices within route
            batch_size = trg_id.size(1)
            seq_len = outputs_id.size(0)
            target_indices = torch.zeros(seq_len, batch_size, dtype=torch.long, device=self.device)

            for t in range(seq_len):
                for i in range(batch_size):
                    trg_val = trg_id[t + 1, i]  # +1 because we removed first element
                    route = da_routes[:, i]
                    matches = (route == trg_val).nonzero(as_tuple=True)[0]
                    if len(matches) > 0:
                        target_indices[t, i] = matches[0]

            # Flatten for cross-entropy
            pred_flat = outputs_id.permute(1, 0, 2).contiguous().view(-1, outputs_id.size(2))
            target_flat = target_indices.permute(1, 0).contiguous().view(-1)

            loss_id = F.cross_entropy(pred_flat, target_flat, reduction='mean')

        # Rate prediction loss
        if self.rate_flag and self.lambda2 > 0:
            trg_rate = data['trg_rate']
            # Exclude first and last
            trg_rate_target = trg_rate[1:-1]
            loss_rate = F.mse_loss(outputs_rate, trg_rate_target, reduction='mean')
        else:
            loss_rate = torch.tensor(0.0, device=self.device)

        # Combined loss
        total_loss = self.lambda1 * loss_id + self.lambda2 * loss_rate

        return total_loss
