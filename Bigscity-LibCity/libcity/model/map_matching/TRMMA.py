"""
TRMMA: Trajectory Recovery with Multi-Modal Alignment

This model is adapted from the original TRMMA (TrajRecovery) implementation for
trajectory recovery/map matching tasks. The model uses a GPS-Route dual encoder
with transformer attention and a multi-task decoder for segment prediction and
position ratio estimation.

Original Repository: https://github.com/xxx/TRMMA

Key Components:
1. PositionalEncoder - Sinusoidal positional encoding
2. MultiHeadAttention - Multi-head self-attention mechanism
3. GPSFormer - Transformer encoder for GPS sequences
4. GRFormer - GPS-Route joint encoder with cross-attention
5. DecoderMulti - Multi-task decoder with segment and rate prediction
6. TRMMA - Main model for trajectory recovery

Adaptations for LibCity:
- Inherits from AbstractModel base class
- Implements forward(), predict(), calculate_loss() methods
- Removed DAPlanner dependency (simplified route handling)
- Adapted batch input format to LibCity's trajectory batch dictionary
- Configuration parameters extracted from config dict
- Uses LibCity's logger instead of custom logging

Key Features:
- Dual encoder architecture for GPS and route information
- Multi-task learning: segment classification + position regression
- Teacher forcing support for training
- Attention-based candidate selection from route candidates

Limitations compared to original:
- Simplified route candidate generation (no external DAPlanner)
- No segment-level road network features by default
- Simplified candidate path handling
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from libcity.model.abstract_model import AbstractModel


# ============================================================================
# Utility Functions
# ============================================================================

def sequence_mask(X, valid_len, value=0.):
    """Mask irrelevant entries in sequences.

    Args:
        X: Tensor of shape (batch, seq_len)
        valid_len: Tensor of shape (batch,) with valid lengths
        value: Fill value for masked positions

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
        X: Tensor of shape (batch, seq_len1, seq_len2)
        valid_len: Tensor of shape (batch,) for first dimension
        valid_len2: Tensor of shape (batch,) for second dimension
        value: Fill value for masked positions

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


# ============================================================================
# Attention Modules
# ============================================================================

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
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        output = torch.matmul(scores, v)

        # Concatenate heads
        concat = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        return self.out(concat)


class FeedForward(nn.Module):
    """Position-wise feed-forward network with residual connection.

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
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.linear_2(F.relu(self.linear_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.norm(x)
        return x


class LayerNorm(nn.Module):
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
        """Calculate attention.

        Args:
            query: Query tensor (batch, src_len, hid_dim)
            key: Key tensor (batch, src_len, candi_num, hid_dim)
            value: Value tensor (batch, src_len, candi_num, hid_dim)
            attn_mask: Attention mask

        Returns:
            scores: Attention scores
            weighted: Weighted sum of values
        """
        bs, src_len = query.shape[0], query.shape[1]
        candi_num = key.shape[-2]

        # Expand query for candidate dimension
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


# ============================================================================
# Encoder Layers
# ============================================================================

class GPSLayer(nn.Module):
    """GPS encoder layer with self-attention.

    Args:
        d_model: Model dimension
        heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = LayerNorm(d_model)
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
    """Route encoder layer with self-attention and cross-attention to GPS.

    Args:
        d_model: Model dimension
        heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(heads, d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff=d_model * 2)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, route, gps, route_mask, inter_mask):
        # Self-attention on route
        route1 = self.dropout_1(self.slf_attn(route, route, route, route_mask))
        route_out = self.norm_1(route + route1)

        # Cross-attention to GPS
        route2 = self.dropout_2(self.attn(route_out, gps, gps, inter_mask))
        route_out2 = self.norm_2(route_out + route2)

        x = self.ff(route_out2)
        return x


class GRLayer(nn.Module):
    """GPS-Route joint encoder layer.

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
    """GPS-Route joint transformer encoder.

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


# ============================================================================
# Encoder Components
# ============================================================================

class GPSEncoder(nn.Module):
    """GPS sequence encoder with transformer.

    Args:
        hid_dim: Hidden dimension
        transformer_layers: Number of transformer layers
        heads: Number of attention heads
        pro_features_flag: Whether to use temporal features
        pro_input_dim: Input dimension for temporal embedding
        pro_output_dim: Output dimension for temporal embedding
    """

    def __init__(self, hid_dim, transformer_layers, heads,
                 pro_features_flag=True, pro_input_dim=48, pro_output_dim=8):
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
            src: Source GPS sequence (src_len, batch, features)
            src_len: Sequence lengths (batch,)
            pro_features: Temporal features (batch,)

        Returns:
            outputs: Encoder outputs (src_len, batch, hid_dim)
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
        outputs = outputs.transpose(0, 1)  # (src_len, bs, hid_dim)

        outputs = outputs * mask2d
        hidden = torch.sum(outputs, dim=0) / src_len.unsqueeze(-1).repeat(1, self.hid_dim)
        hidden = hidden.unsqueeze(0)

        if self.pro_features_flag:
            extra_emb = self.temporal(pro_features)
            extra_emb = extra_emb.unsqueeze(0)
            hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=-1)))

        return outputs, hidden


class GREncoder(nn.Module):
    """GPS-Route joint encoder.

    Args:
        hid_dim: Hidden dimension
        transformer_layers: Number of transformer layers
        heads: Number of attention heads
        pro_features_flag: Whether to use temporal features
        pro_input_dim: Input dimension for temporal embedding
        pro_output_dim: Output dimension for temporal embedding
    """

    def __init__(self, hid_dim, transformer_layers, heads,
                 pro_features_flag=True, pro_input_dim=48, pro_output_dim=8):
        super().__init__()
        self.hid_dim = hid_dim
        self.pro_features_flag = pro_features_flag

        self.transformer = GRFormer(hid_dim, transformer_layers, heads=heads)

        if self.pro_features_flag:
            self.temporal = nn.Embedding(pro_input_dim, pro_output_dim)
            self.fc_hid = nn.Linear(hid_dim + pro_output_dim, hid_dim)

    def forward(self, src, src_len, route, route_len, pro_features):
        """Encode GPS and route sequences jointly.

        Args:
            src: Source GPS sequence (src_len, batch, features)
            src_len: GPS sequence lengths (batch,)
            route: Route sequence (route_len, batch, features)
            route_len: Route sequence lengths (batch,)
            pro_features: Temporal features (batch,)

        Returns:
            outputs: Encoder outputs for routes (route_len, batch, hid_dim)
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
        outputs = outputs.transpose(0, 1)  # (route_len, bs, hid_dim)

        outputs = outputs * route_mask2d
        hidden = torch.sum(outputs, dim=0) / route_len.unsqueeze(-1).repeat(1, self.hid_dim)
        hidden = hidden.unsqueeze(0)

        if self.pro_features_flag:
            extra_emb = self.temporal(pro_features)
            extra_emb = extra_emb.unsqueeze(0)
            hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=-1)))

        return outputs, hidden


# ============================================================================
# Decoder Component
# ============================================================================

class DecoderMulti(nn.Module):
    """Multi-task decoder for segment prediction and rate estimation.

    Args:
        id_size: Size of segment vocabulary
        hid_dim: Hidden dimension
        dest_type: Destination encoding type (0, 1, or 2)
        rate_flag: Whether to predict position rates
        prog_flag: Whether to enforce progressive decoding
        rid_feats_flag: Whether to use road features
        rid_fea_dim: Road feature dimension
    """

    def __init__(self, id_size, hid_dim, dest_type=1, rate_flag=True,
                 prog_flag=False, rid_feats_flag=False, rid_fea_dim=8):
        super().__init__()

        self.id_size = id_size
        self.emb_id = None  # Updated from main model
        self.dest_type = dest_type
        self.rate_flag = rate_flag
        self.prog_flag = prog_flag
        self.rid_feats_flag = rid_feats_flag

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
                      route_attn_mask, d_rids, d_rates, rid_features_dict):
        """Single decoding step.

        Args:
            input_id: Input segment IDs (batch,)
            input_rate: Input position rates (batch, 1)
            hidden: Hidden state (1, batch, hid_dim)
            route_outputs: Route encoder outputs (route_len, batch, hid_dim)
            route_attn_mask: Attention mask (batch, route_len)
            d_rids: Destination segment IDs (batch,)
            d_rates: Destination rates (batch, 1)
            rid_features_dict: Road feature dictionary

        Returns:
            prediction_id: Segment prediction scores (batch, route_len)
            prediction_rate: Position rate prediction (batch, 1)
            hidden: Updated hidden state (1, batch, hid_dim)
        """
        rnn_input = self.emb_id[input_id]

        if self.rid_feats_flag and rid_features_dict is not None:
            rnn_input = torch.cat([rnn_input, rid_features_dict[input_id]], dim=-1)
        if self.rate_flag:
            rnn_input = torch.cat((rnn_input, input_rate), dim=-1)
        if self.dest_type in [1, 2]:
            embed_drids = self.emb_id[d_rids]
            rnn_input = torch.cat((rnn_input, embed_drids), dim=-1)
            if self.rid_feats_flag and rid_features_dict is not None:
                rnn_input = torch.cat([rnn_input, rid_features_dict[d_rids]], dim=-1)
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
            prediction_rate = torch.ones((prediction_id.shape[0], 1), dtype=torch.float32, device=hidden.device) / 2

        return prediction_id, prediction_rate, hidden

    def forward(self, max_trg_len, batch_size, trg_id, trg_rate, trg_len, hidden,
                rid_features_dict, routes, route_outputs, route_attn_mask,
                d_rids, d_rates, teacher_forcing_ratio):
        """Full decoding pass.

        Args:
            max_trg_len: Maximum target length
            batch_size: Batch size
            trg_id: Target segment IDs (max_trg_len, batch)
            trg_rate: Target rates (max_trg_len, batch, 1)
            trg_len: Target lengths (batch,)
            hidden: Initial hidden state (1, batch, hid_dim)
            rid_features_dict: Road feature dictionary
            routes: Route candidates (route_len, batch)
            route_outputs: Route encoder outputs (route_len, batch, hid_dim)
            route_attn_mask: Route attention mask (batch, route_len)
            d_rids: Destination segment IDs (batch,)
            d_rates: Destination rates (batch, 1)
            teacher_forcing_ratio: Teacher forcing probability

        Returns:
            outputs_id: Segment predictions (max_trg_len, batch, route_len)
            outputs_rate: Rate predictions (max_trg_len, batch, 1)
        """
        routes = routes.permute(1, 0)  # (batch, route_len)
        outputs_id = torch.zeros([max_trg_len, batch_size, routes.shape[1]], device=hidden.device)
        outputs_rate = torch.zeros([max_trg_len, batch_size, 1], device=hidden.device)

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
                        prev_idx = (input_id[i] == routes[i]).nonzero(as_tuple=True)[0][0]
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

        # Apply target length mask
        mask_trg = torch.ones([batch_size, max_trg_len], device=outputs_id.device)
        mask_trg = sequence_mask(mask_trg, torch.tensor(trg_len, device=outputs_id.device))
        outputs_rate = outputs_rate.permute(1, 0, 2)  # (batch, max_trg_len, 1)
        outputs_rate = outputs_rate.masked_fill(mask_trg.unsqueeze(-1) == 0, 0)
        outputs_rate = outputs_rate.permute(1, 0, 2)

        return outputs_id, outputs_rate


# ============================================================================
# Main TRMMA Model
# ============================================================================

class TRMMA(AbstractModel):
    """TRMMA: Trajectory Recovery with Multi-Modal Alignment

    This is an adaptation of the TrajRecovery model for LibCity framework.
    The model uses dual GPS-Route encoders with transformer attention and
    a multi-task decoder for segment classification and position regression.

    Args:
        config (dict): Configuration dictionary containing model hyperparameters
        data_feature (dict): Data features including vocabulary sizes

    Required config parameters:
        - hid_dim: Hidden dimension (default: 128)
        - id_emb_dim: Segment embedding dimension (default: 128)
        - transformer_layers: Number of transformer layers (default: 2)
        - heads: Number of attention heads (default: 4)
        - dropout: Dropout probability (default: 0.1)
        - learn_pos: Whether to use learnable position embeddings (default: False)
        - da_route_flag: Whether to use GPS-Route joint encoder (default: True)
        - srcseg_flag: Whether to include source segment info (default: True)
        - rid_feats_flag: Whether to use road features (default: False)
        - rate_flag: Whether to predict position rates (default: True)
        - dest_type: Destination encoding type (default: 1)
        - prog_flag: Whether to enforce progressive decoding (default: False)
        - pro_features_flag: Whether to use temporal features (default: True)
        - pro_input_dim: Temporal feature vocabulary size (default: 48)
        - pro_output_dim: Temporal feature embedding dimension (default: 8)
        - tf_ratio: Teacher forcing ratio for training (default: 0.5)
        - lambda1: Weight for segment classification loss (default: 1.0)
        - lambda2: Weight for rate regression loss (default: 0.5)

    Required data_feature:
        - loc_size / id_size: Number of segment tokens in vocabulary
        - rid_fea_dim: Road feature dimension (if rid_feats_flag)
    """

    def __init__(self, config, data_feature):
        super(TRMMA, self).__init__(config, data_feature)

        self.config = config
        self.data_feature = data_feature
        self.device = config.get('device', 'cpu')

        # Data dimensions from data_feature
        self.id_size = data_feature.get('id_size', data_feature.get('loc_size', 5000))
        self.rid_fea_dim = data_feature.get('rid_fea_dim', 8)

        # Model architecture parameters
        self.hid_dim = config.get('hid_dim', 128)
        self.id_emb_dim = config.get('id_emb_dim', 128)
        self.transformer_layers = config.get('transformer_layers', 2)
        self.heads = config.get('heads', 4)
        self.dropout = config.get('dropout', 0.1)

        # Feature flags
        self.learn_pos = config.get('learn_pos', False)
        self.da_route_flag = config.get('da_route_flag', True)
        self.srcseg_flag = config.get('srcseg_flag', True)
        self.rid_feats_flag = config.get('rid_feats_flag', False)
        self.rate_flag = config.get('rate_flag', True)
        self.dest_type = config.get('dest_type', 1)
        self.prog_flag = config.get('prog_flag', False)

        # Temporal features
        self.pro_features_flag = config.get('pro_features_flag', True)
        self.pro_input_dim = config.get('pro_input_dim', 48)
        self.pro_output_dim = config.get('pro_output_dim', 8)

        # Training parameters
        self.tf_ratio = config.get('tf_ratio', 0.5)
        self.lambda1 = config.get('lambda1', 1.0)
        self.lambda2 = config.get('lambda2', 0.5)

        # Evaluation
        self.evaluate_method = config.get('evaluate_method', 'all')

        # Build model
        self._build_model()
        self.init_weights()

    def _build_model(self):
        """Build model components."""

        # Segment embedding
        self.emb_id = nn.Parameter(torch.rand(self.id_size, self.id_emb_dim))

        # Learnable positional embeddings (optional)
        if self.learn_pos:
            max_input_length = 500
            self.pos_embedding_gps = nn.Embedding(max_input_length, self.hid_dim)
            self.pos_embedding_route = nn.Embedding(max_input_length, self.hid_dim)

        # Input projection for GPS
        input_dim_gps = 3  # (x, y, t)
        if self.learn_pos:
            input_dim_gps += self.hid_dim
        if self.srcseg_flag:
            input_dim_gps += self.hid_dim + 1  # segment embedding + rate
        self.fc_in_gps = nn.Linear(input_dim_gps, self.hid_dim)

        # Input projection for route
        input_dim_route = self.hid_dim
        if self.learn_pos:
            input_dim_route += self.hid_dim
        if self.rid_feats_flag:
            input_dim_route += self.rid_fea_dim
        self.fc_in_route = nn.Linear(input_dim_route, self.hid_dim)

        # Encoder
        if self.da_route_flag:
            self.encoder = GREncoder(
                hid_dim=self.hid_dim,
                transformer_layers=self.transformer_layers,
                heads=self.heads,
                pro_features_flag=self.pro_features_flag,
                pro_input_dim=self.pro_input_dim,
                pro_output_dim=self.pro_output_dim
            )
        else:
            self.encoder = GPSEncoder(
                hid_dim=self.hid_dim,
                transformer_layers=self.transformer_layers,
                heads=self.heads,
                pro_features_flag=self.pro_features_flag,
                pro_input_dim=self.pro_input_dim,
                pro_output_dim=self.pro_output_dim
            )

        # Decoder
        self.decoder = DecoderMulti(
            id_size=self.id_size,
            hid_dim=self.hid_dim,
            dest_type=self.dest_type,
            rate_flag=self.rate_flag,
            prog_flag=self.prog_flag,
            rid_feats_flag=self.rid_feats_flag,
            rid_fea_dim=self.rid_fea_dim
        )

    def init_weights(self):
        """Initialize model weights."""
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, batch, teacher_forcing_ratio=None):
        """Forward pass for trajectory recovery.

        Args:
            batch: LibCity Batch object containing trajectory data.
                Expected keys:
                - 'src_grid': GPS grid features (batch, src_len, 3)
                - 'src_len': Source lengths (batch,)
                - 'trg_id': Target segment IDs (batch, trg_len)
                - 'trg_rate': Target position rates (batch, trg_len, 1)
                - 'trg_len': Target lengths (batch,)
                - 'pro_features': Temporal features (batch,)
                - 'routes': Route candidates (batch, route_len)
                - 'route_len': Route lengths (batch,)
                - 'd_rid': Destination segment ID (batch,)
                - 'd_rate': Destination rate (batch, 1)
                Optional:
                - 'src_seg': Source segment IDs (batch, src_len)
                - 'src_seg_feat': Source segment features (batch, src_len, feat_dim)
                - 'route_pos': Route position indices (batch, route_len)
                - 'rid_features': Road feature dictionary

        Returns:
            outputs_id: Segment predictions (trg_len-2, batch, route_len)
            outputs_rate: Rate predictions (trg_len-2, batch, 1)
        """
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = self.tf_ratio if self.training else 0.0

        # Extract data from batch
        src = batch['src_grid']  # (batch, src_len, 3)
        src_len = batch['src_len']
        trg_id = batch['trg_id']  # (batch, trg_len)
        trg_rate = batch['trg_rate']  # (batch, trg_len, 1)
        trg_len = batch['trg_len']
        pro_features = batch.get('pro_features', torch.zeros(src.size(0), dtype=torch.long, device=self.device))

        routes = batch['routes']  # (batch, route_len)
        route_len = batch['route_len']
        d_rids = batch['d_rid']
        d_rates = batch['d_rate']

        # Move to device
        if not isinstance(src, torch.Tensor):
            src = torch.FloatTensor(src)
        src = src.to(self.device)

        if not isinstance(trg_id, torch.Tensor):
            trg_id = torch.LongTensor(trg_id)
        trg_id = trg_id.to(self.device)

        if not isinstance(trg_rate, torch.Tensor):
            trg_rate = torch.FloatTensor(trg_rate)
        trg_rate = trg_rate.to(self.device)

        if not isinstance(routes, torch.Tensor):
            routes = torch.LongTensor(routes)
        routes = routes.to(self.device)

        if not isinstance(pro_features, torch.Tensor):
            pro_features = torch.LongTensor(pro_features)
        pro_features = pro_features.to(self.device)

        if not isinstance(d_rids, torch.Tensor):
            d_rids = torch.LongTensor(d_rids)
        d_rids = d_rids.to(self.device)

        if not isinstance(d_rates, torch.Tensor):
            d_rates = torch.FloatTensor(d_rates)
        d_rates = d_rates.to(self.device)

        batch_size = src.size(0)
        max_trg_len = trg_id.size(1)

        # Transpose for sequence-first format
        src = src.transpose(0, 1)  # (src_len, batch, 3)
        trg_id = trg_id.transpose(0, 1)  # (trg_len, batch)
        trg_rate = trg_rate.transpose(0, 1)  # (trg_len, batch, 1)
        routes = routes.transpose(0, 1)  # (route_len, batch)

        # Handle sequence lengths
        if isinstance(src_len, torch.Tensor):
            src_len_tensor = src_len.to(self.device)
            src_len_list = src_len.tolist()
        else:
            src_len_list = list(src_len)
            src_len_tensor = torch.tensor(src_len_list, device=self.device)

        if isinstance(route_len, torch.Tensor):
            route_len_tensor = route_len.to(self.device)
            route_len_list = route_len.tolist()
        else:
            route_len_list = list(route_len)
            route_len_tensor = torch.tensor(route_len_list, device=self.device)

        if isinstance(trg_len, torch.Tensor):
            trg_len_list = trg_len.tolist()
        else:
            trg_len_list = list(trg_len)

        # Share embedding with decoder
        self.decoder.emb_id = self.emb_id

        # Prepare GPS embedding
        gps_emb = src.float()

        if self.learn_pos:
            gps_pos = src[:, :, -1].long()
            gps_pos_emb = self.pos_embedding_gps(gps_pos)
            gps_emb = torch.cat([gps_emb, gps_pos_emb], dim=-1)

        if self.srcseg_flag:
            if 'src_seg' in batch:
                src_seg_seqs = batch['src_seg']
                if not isinstance(src_seg_seqs, torch.Tensor):
                    src_seg_seqs = torch.LongTensor(src_seg_seqs)
                src_seg_seqs = src_seg_seqs.transpose(0, 1).to(self.device)
                seg_emb = self.emb_id[src_seg_seqs]

                if 'src_seg_feat' in batch:
                    src_seg_feats = batch['src_seg_feat']
                    if not isinstance(src_seg_feats, torch.Tensor):
                        src_seg_feats = torch.FloatTensor(src_seg_feats)
                    src_seg_feats = src_seg_feats.transpose(0, 1).to(self.device)
                else:
                    src_seg_feats = torch.zeros(src_seg_seqs.size(0), src_seg_seqs.size(1), 1, device=self.device)

                gps_emb = torch.cat((gps_emb, seg_emb, src_seg_feats), dim=-1)
            else:
                # If no source segments, pad with zeros
                pad_size = self.hid_dim + 1
                gps_emb = torch.cat([gps_emb, torch.zeros(*gps_emb.shape[:2], pad_size, device=self.device)], dim=-1)

        gps_in = self.fc_in_gps(gps_emb)
        gps_in_lens = src_len_tensor

        # Get road features if available
        rid_features_dict = batch.get('rid_features', None)

        # Encode
        if self.da_route_flag:
            # Prepare route embedding
            route_emb = self.emb_id[routes]

            if self.learn_pos:
                if 'route_pos' in batch:
                    route_pos = batch['route_pos']
                    if not isinstance(route_pos, torch.Tensor):
                        route_pos = torch.LongTensor(route_pos)
                    route_pos = route_pos.transpose(0, 1).to(self.device)
                else:
                    route_pos = torch.arange(routes.size(0), device=self.device).unsqueeze(1).expand_as(routes)
                route_pos_emb = self.pos_embedding_route(route_pos)
                route_emb = torch.cat([route_emb, route_pos_emb], dim=-1)

            if self.rid_feats_flag and rid_features_dict is not None:
                route_feats = rid_features_dict[routes]
                route_emb = torch.cat([route_emb, route_feats], dim=-1)

            route_in = self.fc_in_route(route_emb)
            route_in_lens = route_len_tensor

            route_outputs, hiddens = self.encoder(
                gps_in, gps_in_lens, route_in, route_in_lens, pro_features
            )
        else:
            _, hiddens = self.encoder(gps_in, gps_in_lens, pro_features)
            route_in_lens = route_len_tensor
            route_outputs = self.emb_id[routes]

        # Prepare route attention mask
        route_attn_mask = torch.ones(batch_size, max(route_len_list), device=self.device)
        route_attn_mask = sequence_mask(route_attn_mask, route_in_lens)

        # Decode
        outputs_id, outputs_rate = self.decoder(
            max_trg_len, batch_size, trg_id, trg_rate, trg_len_list, hiddens,
            rid_features_dict, routes, route_outputs, route_attn_mask,
            d_rids, d_rates, teacher_forcing_ratio
        )

        # Return outputs excluding first and last positions (start/end tokens)
        final_outputs_id = outputs_id[1:-1]
        final_outputs_rate = outputs_rate[1:-1]

        return final_outputs_id, final_outputs_rate

    def predict(self, batch):
        """Prediction method for LibCity evaluation.

        Args:
            batch: Input batch dictionary

        Returns:
            Dictionary with 'seg_pred' and 'rate_pred' tensors
        """
        self.eval()
        with torch.no_grad():
            outputs_id, outputs_rate = self.forward(batch, teacher_forcing_ratio=0.0)

            # Get predicted segment indices
            seg_pred = outputs_id.argmax(dim=-1)  # (trg_len-2, batch)
            seg_pred = seg_pred.transpose(0, 1)  # (batch, trg_len-2)

            rate_pred = outputs_rate.squeeze(-1).transpose(0, 1)  # (batch, trg_len-2)

            return {
                'seg_pred': seg_pred,
                'rate_pred': rate_pred,
                'seg_scores': outputs_id.transpose(0, 1)  # (batch, trg_len-2, route_len)
            }

    def calculate_loss(self, batch):
        """Calculate multi-task training loss.

        The loss combines:
        1. Binary cross-entropy for segment classification
        2. L1 loss for position rate regression

        Args:
            batch: LibCity Batch object containing trajectory data and labels.
                Expected additional keys:
                - 'labels': Binary labels for segment selection (batch, trg_len-2, route_len)

        Returns:
            Total loss (weighted sum of segment and rate losses)
        """
        # Forward pass with teacher forcing
        outputs_id, outputs_rate = self.forward(batch)

        # Get labels
        labels = batch['labels']  # (batch, trg_len-2, route_len)
        if not isinstance(labels, torch.Tensor):
            labels = torch.FloatTensor(labels)
        labels = labels.to(self.device)

        # Get target rates (for rate loss)
        trg_rate = batch['trg_rate']  # (batch, trg_len, 1)
        if not isinstance(trg_rate, torch.Tensor):
            trg_rate = torch.FloatTensor(trg_rate)
        trg_rate = trg_rate.to(self.device)
        trg_rate = trg_rate[:, 1:-1, :]  # Exclude start/end

        # Get target lengths for masking
        trg_len = batch['trg_len']
        if isinstance(trg_len, torch.Tensor):
            trg_len_list = trg_len.tolist()
        else:
            trg_len_list = list(trg_len)

        batch_size = labels.size(0)
        max_trg_len = labels.size(1)

        # Transpose outputs to (batch, seq, dim)
        outputs_id = outputs_id.transpose(0, 1)  # (batch, trg_len-2, route_len)
        outputs_rate = outputs_rate.transpose(0, 1)  # (batch, trg_len-2, 1)

        # Create mask for valid positions
        mask = torch.zeros(batch_size, max_trg_len, device=self.device)
        for i, length in enumerate(trg_len_list):
            valid_len = min(length - 2, max_trg_len)  # Exclude start/end tokens
            if valid_len > 0:
                mask[i, :valid_len] = 1

        # Segment classification loss (BCE)
        # Apply softmax and compute BCE with labels
        outputs_id_flat = outputs_id.contiguous().view(-1, outputs_id.size(-1))
        labels_flat = labels.contiguous().view(-1, labels.size(-1))
        mask_flat = mask.contiguous().view(-1)

        # Only compute loss on valid positions
        valid_indices = mask_flat > 0
        if valid_indices.sum() > 0:
            valid_outputs = outputs_id_flat[valid_indices]
            valid_labels = labels_flat[valid_indices]

            # BCE loss
            loss_seg = F.binary_cross_entropy(
                valid_outputs.clamp(min=1e-6, max=1-1e-6),
                valid_labels,
                reduction='mean'
            )
        else:
            loss_seg = torch.tensor(0.0, device=self.device)

        # Rate regression loss (L1)
        if self.rate_flag:
            trg_rate_valid = trg_rate[:, :max_trg_len, :]
            rate_diff = torch.abs(outputs_rate - trg_rate_valid)
            rate_diff = rate_diff.squeeze(-1) * mask

            if mask.sum() > 0:
                loss_rate = rate_diff.sum() / mask.sum()
            else:
                loss_rate = torch.tensor(0.0, device=self.device)
        else:
            loss_rate = torch.tensor(0.0, device=self.device)

        # Combined loss
        total_loss = self.lambda1 * loss_seg + self.lambda2 * loss_rate

        return total_loss

    def recover_trajectory(self, batch, greedy=True):
        """Recover full trajectory from sparse observations.

        Args:
            batch: Input batch with sparse trajectory observations
            greedy: Whether to use greedy decoding

        Returns:
            recovered_segs: Recovered segment indices (batch, trg_len-2)
            recovered_rates: Recovered position rates (batch, trg_len-2)
        """
        result = self.predict(batch)
        return result['seg_pred'], result['rate_pred']
