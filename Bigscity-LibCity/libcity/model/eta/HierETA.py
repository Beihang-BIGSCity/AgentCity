"""
HierETA Model adapted for LibCity framework.

Original source: repos/HierETA/models/
Paper: Hierarchical ETA for Travel Time Estimation

Key adaptations:
1. Inherits from AbstractTrafficStateModel
2. Uses config dict instead of FLAGS argparse object
3. Device-agnostic tensor operations
4. LibCity batch format handling
5. All submodules consolidated in one file

Author: Model Adaptation Agent
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


# ==============================================================================
# Attribute Feature Extractor (adapted from attrs.py)
# ==============================================================================

class HierETAAttr(nn.Module):
    """
    Attribute Feature Extractor for HierETA.
    Handles categorical and continuous attributes for external, segment, and link features.

    Adaptations:
    - Made embedding dimensions configurable via config dict
    - Removed hardcoded vocabulary sizes; use config/data_feature
    """

    def __init__(self, config, data_feature, device):
        super(HierETAAttr, self).__init__()
        self.device = device
        self.batch_size = config.get('batch_size', 64)

        # External categorical attributes: (name, dim_in, dim_out)
        # These are configured based on the dataset
        week_vocab = config.get('week_vocab_size', 8)
        time_vocab = config.get('time_vocab_size', 289)
        driver_vocab = data_feature.get('driver_vocab_size', config.get('driver_vocab_size', 200141))

        self.ext_cates = [
            ("weekID", week_vocab, config.get('week_emb_dim', 3)),
            ("timeID", time_vocab, config.get('time_emb_dim', 5)),
            ("driverID", driver_vocab, config.get('driver_emb_dim', 16))
        ]
        self.ext_conts = []

        # Segment categorical attributes
        seg_vocab = data_feature.get('seg_vocab_size', config.get('seg_vocab_size', 1376567))
        func_level_vocab = config.get('func_level_vocab_size', 9)
        road_state_vocab = config.get('road_state_vocab_size', 6)
        lane_num_vocab = config.get('lane_num_vocab_size', 7)
        road_level_vocab = config.get('road_level_vocab_size', 8)

        self.seg_cates = [
            ("segID", seg_vocab, config.get('seg_emb_dim', 16)),
            ("segment_functional_level", func_level_vocab, config.get('func_level_emb_dim', 2)),
            ("roadState", road_state_vocab, config.get('road_state_emb_dim', 2)),
            ("laneNum", lane_num_vocab, config.get('lane_num_emb_dim', 2)),
            ("roadLevel", road_level_vocab, config.get('road_level_emb_dim', 2))
        ]
        self.seg_conts = ["wid", "speedLimit", "time", "len"]

        # Link (crossing) categorical attributes
        cross_vocab = data_feature.get('cross_vocab_size', config.get('cross_vocab_size', 101009))

        self.link_cates = [
            ("crossID", cross_vocab, config.get('cross_emb_dim', 15))
        ]
        self.link_conts = ["delayTime"]

        # Create embedding layers for all categorical attributes
        for name, dim_in, dim_out in self.ext_cates + self.seg_cates + self.link_cates:
            self.add_module("attr-" + name, nn.Embedding(dim_in, dim_out))

    def forward(self, attrs):
        """
        Extract attribute features.

        Args:
            attrs: dict containing attribute tensors

        Returns:
            ext: External features tensor
            seg: Segment features tensor
            link: Link features tensor
        """
        ext = self._emb_helper(attrs, "ext")
        seg = self._emb_helper(attrs, "seg")
        link = self._emb_helper(attrs, "link")
        return ext, seg, link

    def _emb_helper(self, attrs, attr_type):
        """Helper to embed categorical and concat continuous attributes."""
        cates, conts = self._type_helper(attr_type)
        emb_list = []

        # Infer batch_size dynamically from the first available attribute
        inferred_batch_size = None

        for name, dim_in, dim_out in cates:
            embed = getattr(self, "attr-" + name)
            if name in attrs.data:
                # Infer batch_size from the first categorical attribute
                if inferred_batch_size is None:
                    inferred_batch_size = attrs[name].shape[0]
                batch_size = attrs[name].shape[0]
                attr_t = attrs[name].view(batch_size, -1)
                # Only squeeze the last dimension if it's size 1, preserve batch dimension
                attr_t = embed(attr_t)
                if attr_t.dim() > 2 and attr_t.shape[-1] == 1:
                    attr_t = attr_t.squeeze(-1)
                emb_list.append(attr_t)

        for name in conts:
            if name in attrs.data:
                attr_t = attrs[name].float()
                # Infer batch_size from continuous attribute if not yet set
                if inferred_batch_size is None:
                    inferred_batch_size = attr_t.shape[0]
                emb_list.append(attr_t.unsqueeze(-1))

        if len(emb_list) > 0:
            out = torch.cat(emb_list, -1)
            # For external features (route-level scalars), squeeze the sequence dimension
            # External features have shape [batch, 1, emb_dim] but should be [batch, emb_dim]
            if attr_type == "ext" and out.dim() == 3 and out.shape[1] == 1:
                out = out.squeeze(1)  # [batch, 1, total_emb] -> [batch, total_emb]
        else:
            # Return empty tensor with correct batch size (fallback to configured batch_size)
            batch_size = inferred_batch_size if inferred_batch_size is not None else self.batch_size
            out = torch.zeros(batch_size, 0, device=self.device)
        return out

    def _type_helper(self, types):
        """Get categorical and continuous attribute lists by type."""
        if types == "ext":
            return self.ext_cates, self.ext_conts
        elif types == "seg":
            return self.seg_cates, self.seg_conts
        elif types == "link":
            return self.link_cates, self.link_conts
        else:
            raise ValueError("must choose from ext, seg and link!")

    def out_size(self, attr_type):
        """Calculate output dimension for a given attribute type."""
        cates, conts = self._type_helper(attr_type)
        size = 0
        for name, dim_in, dim_out in cates:
            size += dim_out
        size += len(conts)
        return size


# ==============================================================================
# Segment-Level Self-Attention (adapted from segment_encoder.py)
# ==============================================================================

class SegSelfAtt(nn.Module):
    """
    Segment-level self-attention with local and global views.
    Uses windowed attention for local context and full attention for global.
    """

    def __init__(self, d_s, seq_len, win_size, seg_tag=False, device=None):
        super(SegSelfAtt, self).__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.scaling = d_s ** 0.5

        self.W_Q = nn.Linear(d_s, d_s, bias=False)
        self.W_K = nn.Linear(d_s, d_s, bias=False)
        self.W_V = nn.Linear(d_s, d_s, bias=False)

        # Gating mechanism
        self.W_h = nn.Linear(d_s, 1, bias=True)
        self.W_g = nn.Linear(d_s, 1, bias=False)
        self.W_l = nn.Linear(d_s, 1, bias=False)

        # Pre-compute local attention mask
        self.register_buffer('feat_simi', self._get_mask(seq_len, win_size, seg_tag))
        self.seg_layer_norm = nn.LayerNorm(d_s, eps=1e-6)

    def forward(self, H_s, seg_mask):
        """
        Apply segment-level self-attention.

        Args:
            H_s: Hidden states (batch, links, segments, features)
            seg_mask: Mask for valid segments

        Returns:
            att_out: Attention output with residual connection
        """
        Q = self.W_Q(H_s)
        K = self.W_K(H_s)
        V = self.W_V(H_s)

        # Global attention
        GP = torch.matmul(Q / self.scaling, K.transpose(-2, -1))
        GP = GP.masked_fill(~seg_mask, -1e10)
        G_Att = torch.matmul(F.softmax(GP, dim=-1), V)

        # Local (windowed) attention
        LP = GP.masked_fill(~self.feat_simi, -1e10)
        L_Att = torch.matmul(F.softmax(LP, dim=-1), V)

        # Gating fusion
        gate = torch.sigmoid(self.W_h(H_s) + self.W_g(G_Att) + self.W_l(L_Att))
        Fusion = gate * L_Att + (1.0 - gate) * G_Att

        att_out = self.seg_layer_norm(Fusion + H_s)
        return att_out

    def _get_mask(self, seq_len, win_size, seg=False):
        """Create windowed attention mask."""
        single_sided = math.floor(win_size / 2)
        mask = np.zeros((seq_len, seq_len))
        for i in range(-single_sided, single_sided + 1):
            mask += np.eye(seq_len, k=i)
        mask = torch.from_numpy(mask).bool().unsqueeze(0)
        if not seg:
            mask = mask.unsqueeze(0)
        return mask


class SegmentEncoder(nn.Module):
    """
    Segment Encoder with bidirectional LSTM and self-attention.
    Processes segment sequences within each link.
    """

    def __init__(self, config, attr_module, device):
        super(SegmentEncoder, self).__init__()
        self.device = device

        self.emb_dim = attr_module.out_size("seg") + attr_module.out_size("ext")
        self.hidden_dim = config.get('seg_hidden_dim', 128)
        self.link_num = config.get('link_num', 31)
        self.segment_num = config.get('segment_num', 50)
        self.batch_size = config.get('batch_size', 64)

        self.attr_mapping = nn.Linear(attr_module.out_size("seg"), 128)
        self.relu = nn.ReLU()

        self.seg_lstm = nn.LSTM(
            self.emb_dim,
            self.hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        self.linear_hidden = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.linear_cell = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        win_size = config.get('win_size', 3)
        self.seg_self_att = SegSelfAtt(
            d_s=256,
            seq_len=self.segment_num,
            win_size=win_size,
            seg_tag=True,
            device=device
        )

    def forward(self, route, ext, seg):
        """
        Encode segment features.

        Args:
            route: Dict with route information
            ext: External features
            seg: Segment features

        Returns:
            seg_context_feat: Contextualized segment features
        """
        # Infer batch_size from input tensor
        batch_size = seg.shape[0]

        # Expand external features to match segment dimensions
        ext = torch.unsqueeze(ext, dim=1)
        expand_ext = ext.expand(seg.size()[:2] + (ext.size()[-1],))
        segment_input = torch.cat((seg, expand_ext), dim=2)
        segment_input = torch.reshape(
            segment_input,
            (batch_size, self.link_num, self.segment_num, -1)
        )

        link_seg_lens = route["link_seg_lens"]
        link_segment_mask = torch.reshape(
            route["road_segment_mask"],
            (batch_size, self.link_num, self.segment_num)
        ).bool()

        # Process each link's segments
        emb_enc_inputs = []
        for i in range(segment_input.shape[1]):
            emb_enc_inputs.append(torch.squeeze(segment_input[:, i, :, :], dim=1))

        hidden = None
        seg_lstm_outs = []

        for i in range(len(emb_enc_inputs)):
            # Clamp to avoid zero-length sequences
            seq_lens_i = torch.clamp_min(link_seg_lens[:, i], min=1)
            enc_input = nn.utils.rnn.pack_padded_sequence(
                emb_enc_inputs[i],
                seq_lens_i.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            enc_output, (hidden_h, hidden_c) = self.seg_lstm(enc_input, hidden)

            # Handle state propagation for empty segments
            if i > 0:
                (hidden_h_pre, hidden_c_pre) = hidden
                real_seq_lens = (link_seg_lens[:, i] != 0).float().view(1, -1, 1).to(self.device)
                hidden_h = real_seq_lens * hidden_h + (1.0 - real_seq_lens) * hidden_h_pre
                hidden_c = real_seq_lens * hidden_c + (1.0 - real_seq_lens) * hidden_c_pre

            hidden = (hidden_h, hidden_c)
            enc_output, _ = nn.utils.rnn.pad_packed_sequence(enc_output, batch_first=True)
            enc_output = F.pad(
                input=enc_output,
                pad=[0, 0, 0, self.segment_num - enc_output.shape[1], 0, 0],
                mode='constant',
                value=0
            )
            seg_lstm_outs.append(enc_output)

        seg_lstm_outs = torch.stack(seg_lstm_outs, dim=1)
        seg_context_feat = self.seg_self_att(seg_lstm_outs, link_segment_mask.unsqueeze(2))
        return seg_context_feat


# ==============================================================================
# Link-Level Encoder (adapted from link_encoder.py)
# ==============================================================================

class VanilaAttention(nn.Module):
    """Simple attention mechanism for aggregating segment features."""

    def __init__(self, hidden_size):
        super(VanilaAttention, self).__init__()
        self.W = nn.Linear(hidden_size, 1)

    def forward(self, x):
        f = F.softmax(self.W(x), dim=-2)
        out = torch.sum(f * x, dim=-2)
        return out


class LinkSelfAtt(nn.Module):
    """Link-level self-attention module."""

    def __init__(self, d_l):
        super(LinkSelfAtt, self).__init__()
        self.temperature = d_l ** 0.5

        self.W_Q = nn.Linear(d_l, d_l, bias=False)
        self.W_K = nn.Linear(d_l, d_l, bias=False)
        self.W_V = nn.Linear(d_l, d_l, bias=False)

        self.link_layer_norm = nn.LayerNorm(d_l, eps=1e-6)

    def forward(self, H_hat_l, link_mask):
        """
        Apply link-level self-attention.

        Args:
            H_hat_l: Link hidden states
            link_mask: Mask for valid links

        Returns:
            att_out: Attention output
        """
        Q = self.W_Q(H_hat_l)
        K = self.W_K(H_hat_l)
        V = self.W_V(H_hat_l)

        attn = torch.matmul(Q / self.temperature, K.transpose(-2, -1))
        attn = attn.masked_fill(~link_mask, -1e10)

        attention_score = F.softmax(attn, dim=-1)
        attention_score = torch.matmul(attention_score, V)

        att_out = self.link_layer_norm(attention_score + H_hat_l)
        return att_out


class LinkEncoder(nn.Module):
    """
    Link Encoder that aggregates segment features and encodes link sequences.
    """

    def __init__(self, config, attr_module, device):
        super(LinkEncoder, self).__init__()
        self.device = device
        self.link_num = config.get('link_num', 31)
        self.batch_size = config.get('batch_size', 64)

        self.link_inp_dim = 256
        self.link_hidden_dim = config.get('link_hidden_dim', 192)
        self.cross_inp_dim = attr_module.out_size("link")
        self.cross_hidden_dim = config.get('cross_hidden_dim', 64)

        self.vanila_att = VanilaAttention(256)

        self.rnn_link = nn.LSTM(
            self.link_inp_dim,
            self.link_hidden_dim,
            batch_first=True
        )
        self.rnn_cross = nn.LSTM(
            self.cross_inp_dim,
            self.cross_hidden_dim,
            batch_first=True
        )

        self.link_self_att = LinkSelfAtt(d_l=256)

    def forward(self, route, seg_context_feat, cross):
        """
        Encode link features.

        Args:
            route: Dict with route information
            seg_context_feat: Contextualized segment features
            cross: Crossing (intersection) features

        Returns:
            link_context_feat: Contextualized link features
        """
        # Infer batch_size from input tensor
        batch_size = seg_context_feat.shape[0]

        # Aggregate segment features to link level
        link_feat = self.vanila_att(seg_context_feat)

        link_lens = route["link_lens"].squeeze(-1)  # Convert [batch, 1] to [batch]
        road_link_mask = torch.reshape(
            route["road_link_mask"],
            (batch_size, self.link_num)
        ).bool().unsqueeze(1)

        # Process link sequence
        link_lstm_enc = nn.utils.rnn.pack_padded_sequence(
            link_feat,
            link_lens.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        link_lstm_enc, _ = self.rnn_link(link_lstm_enc)
        link_lstm_enc, _ = nn.utils.rnn.pad_packed_sequence(link_lstm_enc, batch_first=True)
        link_lstm_enc = F.pad(
            input=link_lstm_enc,
            pad=[0, 0, 0, self.link_num - link_lstm_enc.shape[1], 0, 0],
            mode='constant',
            value=0
        )

        # Process crossing sequence
        cross_lstm_enc = nn.utils.rnn.pack_padded_sequence(
            cross,
            link_lens.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        cross_lstm_enc, _ = self.rnn_cross(cross_lstm_enc)
        cross_lstm_enc, _ = nn.utils.rnn.pad_packed_sequence(cross_lstm_enc, batch_first=True)
        cross_lstm_enc = F.pad(
            input=cross_lstm_enc,
            pad=[0, 0, 0, self.link_num - cross_lstm_enc.shape[1], 0, 0],
            mode='constant',
            value=0
        )

        # Concatenate and apply self-attention
        H_hat_l = torch.cat((link_lstm_enc, cross_lstm_enc), -1)
        link_context_feat = self.link_self_att(H_hat_l, road_link_mask)
        return link_context_feat


# ==============================================================================
# Hierarchy-Aware Attention Decoder (adapted from decoder.py)
# ==============================================================================

class SegAtt(nn.Module):
    """Segment-level attention for decoder."""

    def __init__(self, ext_size):
        super(SegAtt, self).__init__()
        self.w1 = nn.Linear(256, 256)
        self.w2 = nn.Linear(ext_size, 256)
        self.v_seg = nn.Linear(256, 1, bias=False)
        self.softmax_seg = nn.Softmax(1)

    def forward(self, seg_context_feat, ext):
        e_seg = torch.sum(
            self.v_seg(torch.tanh(self.w1(seg_context_feat)) + self.w2(ext.unsqueeze(-2))),
            dim=(2)
        )
        att_dist_seg = self.softmax_seg(e_seg)
        return att_dist_seg


class LinkAtt(nn.Module):
    """Link-level attention for decoder."""

    def __init__(self, ext_size):
        super(LinkAtt, self).__init__()
        self.w1 = nn.Linear(256, 256)
        self.w2 = nn.Linear(ext_size, 256)
        self.v_link = nn.Linear(256, 1, bias=False)
        self.softmax_link = nn.Softmax(1)

    def forward(self, link_context_feat, ext):
        e_link = torch.sum(
            self.v_link(torch.tanh(self.w1(link_context_feat) + self.w2(ext.unsqueeze(-2)))),
            dim=2
        )
        att_dist_link = self.softmax_link(e_link)
        return att_dist_link


class AttentionDecoder(nn.Module):
    """
    Hierarchy-Aware Attention Decoder.
    Combines segment and link level information for final prediction.
    """

    def __init__(self, config, attr_module, device):
        super(AttentionDecoder, self).__init__()
        self.device = device
        self.batch_size = config.get('batch_size', 64)
        self.link_num = config.get('link_num', 31)
        self.segment_num = config.get('segment_num', 50)
        self.Lambda = config.get('Lambda', 0.4)

        ext_size = attr_module.out_size("ext")
        self.seg_level_att = SegAtt(ext_size)
        self.link_level_att = LinkAtt(ext_size)

        self.softmax = nn.Softmax(1)
        self.linear = nn.Linear(256, 1)

    def forward(self, route, seg_context_feat, link_context_feat, ext):
        """
        Decode hierarchical features to ETA prediction.

        Args:
            route: Dict with route information
            seg_context_feat: Segment context features
            link_context_feat: Link context features
            ext: External features

        Returns:
            output: Predicted travel time
        """
        # Infer batch_size from input tensor
        batch_size = seg_context_feat.shape[0]

        seg_context_feat_reshaped = torch.reshape(
            seg_context_feat,
            (batch_size, self.link_num * self.segment_num, -1)
        )

        att_dist_seg = self.seg_level_att(seg_context_feat_reshaped, ext)
        att_dist_link = self.link_level_att(link_context_feat, ext)

        # Hierarchical attention guidance
        att_dist_guide = torch.mul(
            torch.reshape(att_dist_seg, (batch_size, self.link_num, self.segment_num)),
            att_dist_link.unsqueeze(dim=-1)
        )

        segment_padding_mask = route["road_segment_mask"]

        masked_dist_seg = self.softmax(torch.reshape(
            att_dist_guide * segment_padding_mask.reshape(
                batch_size, self.link_num, self.segment_num
            ),
            [batch_size, -1]
        ))

        att_seg = torch.sum(
            torch.reshape(
                masked_dist_seg,
                [batch_size, self.link_num, self.segment_num, 1]
            ) * seg_context_feat,
            [1, 2]
        )

        att_link = torch.sum(
            torch.reshape(att_dist_link, [batch_size, self.link_num, 1]) * link_context_feat,
            [1]
        )

        # Fusion with Lambda weighting
        R = torch.reshape(
            (1.0 - self.Lambda) * att_seg + self.Lambda * att_link,
            [batch_size, -1]
        )
        output = self.linear(R)
        return output


# ==============================================================================
# Main HierETA Model (adapted for LibCity)
# ==============================================================================

class HierETA(AbstractTrafficStateModel):
    """
    Hierarchical ETA Model for Travel Time Estimation.

    Architecture:
    1. Attribute Feature Extractor - embeds categorical and continuous features
    2. Segment Encoder - processes segments within each link using BiLSTM + self-attention
    3. Link Encoder - aggregates segment info and processes link sequence
    4. Attention Decoder - hierarchical attention for final ETA prediction

    LibCity Adaptations:
    - Inherits from AbstractTrafficStateModel
    - Uses config dict for hyperparameters
    - Device-agnostic operations
    - Implements predict() and calculate_loss() methods
    """

    def __init__(self, config, data_feature):
        super(HierETA, self).__init__(config, data_feature)

        self.config = config
        self.data_feature = data_feature
        self.device = config.get('device', torch.device('cpu'))
        self.batch_size = config.get('batch_size', 64)

        # Store normalization statistics
        self.time_mean = data_feature.get('train_gt_eta_time_mean', 0.0)
        self.time_std = data_feature.get('train_gt_eta_time_std', 1.0)

        # Attribute Feature Extractor
        self.attr_net = HierETAAttr(config, data_feature, self.device)

        # Hierarchical Self-Attention Network
        self.seg_enc = SegmentEncoder(config, self.attr_net, self.device)
        self.link_enc = LinkEncoder(config, self.attr_net, self.device)

        # Hierarchy-Aware Attention Decoder
        self.decoder = AttentionDecoder(config, self.attr_net, self.device)

        self._init_weight()

    def _init_weight(self):
        """Initialize model weights using Xavier initialization."""
        for name, params in self.named_parameters():
            if "norm" in name.lower():
                continue
            if name.find('.bias') != -1:
                params.data.fill_(0)
            elif name.find('.weight') != -1:
                nn.init.xavier_uniform_(params.data)

    def forward(self, batch):
        """
        Forward pass of HierETA.

        Args:
            batch: Dict containing route information and features
                Expected keys:
                - weekID, timeID, driverID: External categorical features
                - segID, segment_functional_level, roadState, laneNum, roadLevel: Segment categoricals
                - wid, speedLimit, time, len: Segment continuous features
                - crossID: Link categorical features
                - delayTime: Link continuous features
                - link_seg_lens: Length of segments in each link
                - road_segment_mask: Mask for valid segments
                - link_lens: Length of links
                - road_link_mask: Mask for valid links

        Returns:
            pred: Predicted travel time (unnormalized)
        """
        # Extract features
        ext, seg, cross = self.attr_net(batch)

        # Hierarchical encoding
        seg_context_feat = self.seg_enc(batch, ext, seg)
        link_context_feat = self.link_enc(batch, seg_context_feat, cross)

        # Decode to prediction
        pred = self.decoder(batch, seg_context_feat, link_context_feat, ext)

        # Unnormalize prediction
        pred = pred * self.time_std + self.time_mean

        return pred

    def predict(self, batch):
        """
        Make predictions on a batch.

        Args:
            batch: Input batch dictionary

        Returns:
            pred: Predicted travel times (batch_size, 1)
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate Mean Absolute Error loss.

        Args:
            batch: Input batch with 'gt_eta_time' as ground truth

        Returns:
            loss: MAE loss tensor
        """
        pred = self.forward(batch)

        # Get ground truth and unnormalize (use -1 to infer batch size dynamically)
        label = batch['gt_eta_time'].view(-1, 1).to(self.device)
        label = label * self.time_std + self.time_mean

        # MAE loss
        loss = torch.mean(torch.abs(pred - label))

        return loss
