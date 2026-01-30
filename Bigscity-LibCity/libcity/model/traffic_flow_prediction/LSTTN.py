"""
LSTTN (Long Short-Term Transformer Network) for traffic flow prediction.

Adapted from: https://github.com/changjiangyu/LSTTN
Original paper: "LSTTN: Long Short-Term Transformer-based Network for
                Traffic Flow Forecasting"

Architecture overview:
  - Combines long-term trends, periodic/seasonal features, and short-term
    patterns for multi-step traffic forecasting.
  - Uses StackedDilatedConv for extracting long-term trend features.
  - Uses DynamicGraphConv for extracting weekly and daily seasonality features.
  - Uses a GraphWaveNet backbone for short-term spatiotemporal features.
  - A frozen pretrained Transformer encodes long historical sequences into
    patch-level representations consumed by the trend and seasonality branches.
  - The final prediction fuses trend, seasonality, and short-term features
    through an MLP.

Key adaptations for LibCity:
  - Inherits from AbstractTrafficStateModel.
  - Reads adjacency matrix from data_feature['adj_mx'] and constructs
    doubletransition supports internally.
  - All hyperparameters are read from LibCity's config system.
  - batch['X'] has shape (batch, input_window, num_nodes, feature_dim).
    The model slices short_x (last short_seq_len steps) and long_x (all
    input_window steps) from this tensor.  The caller must set input_window
    equal to long_seq_len (default 4032) so that the full long history is
    available.
  - batch['y'] has shape (batch, output_window, num_nodes, output_dim).
  - Normalization/denormalization is handled via self._scaler.
  - The pretrained Transformer path is configurable via 'pretrained_transformer_path'.
    If left empty the Transformer encoder is trained from scratch jointly with the
    rest of the model (weights are not frozen).
"""

import os
import math
import random
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.init import trunc_normal_

from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


# ---------------------------------------------------------------------------
# Adjacency matrix utilities (doubletransition)
# ---------------------------------------------------------------------------

def _asym_adj(adj):
    """Row-normalize an adjacency matrix (transition matrix)."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def _build_doubletransition_supports(adj_mx):
    """
    Build forward and backward transition matrices from a raw adjacency
    matrix, matching the original LSTTN 'doubletransition' processing.
    Returns a list of two numpy arrays.
    """
    return [_asym_adj(adj_mx), _asym_adj(np.transpose(adj_mx))]


# ---------------------------------------------------------------------------
# Transformer sub-components
# ---------------------------------------------------------------------------

class InputEmbedding(nn.Module):
    """Patch-based input embedding using a 2-D convolution."""
    def __init__(self, patch_size, input_channel, output_channel):
        super().__init__()
        self.output_channel = output_channel
        self.patch_size = patch_size
        self.input_channel = input_channel
        self.input_embedding = nn.Conv2d(
            input_channel, output_channel,
            kernel_size=(self.patch_size, 1),
            stride=(self.patch_size, 1)
        )

    def forward(self, x):
        # x: (batch, num_nodes, num_channels, long_seq_len)
        batch_size, num_nodes, num_channels, long_seq_len = x.size()
        x = x.unsqueeze(-1)
        x = x.reshape(batch_size * num_nodes, num_channels, long_seq_len, 1)
        out = self.input_embedding(x)
        out = out.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)
        return out


class LearnableTemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(max_len, d_model), requires_grad=True)
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x, indices):
        if indices is None:
            pe = self.pe[:x.size(1), :].unsqueeze(0)
        else:
            pe = self.pe[indices].unsqueeze(0)
        x = x + pe
        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.tem_pe = LearnableTemporalPositionalEncoding(hidden_dim, dropout)

    def forward(self, x, indices=None):
        batch_size, num_nodes, num_subseq, out_channels = x.size()
        x = self.tem_pe(
            x.view(batch_size * num_nodes, num_subseq, out_channels),
            indices=indices
        )
        x = x.view(batch_size, num_nodes, num_subseq, out_channels)
        return x


class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(
            hidden_dim, num_heads, hidden_dim * 4, dropout,
            batch_first=False
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, x):
        batch_size, num_nodes, num_subseq, out_channels = x.size()
        x = x * math.sqrt(self.d_model)
        x = x.view(batch_size * num_nodes, num_subseq, out_channels)
        x = x.transpose(0, 1)
        out = self.transformer_encoder(x, mask=None)
        out = out.transpose(0, 1).view(batch_size, num_nodes, num_subseq, out_channels)
        return out


class MaskGenerator(nn.Module):
    """Generates random mask indices for masked pre-training."""
    def __init__(self, mask_size, mask_ratio):
        super().__init__()
        self.mask_size = int(mask_size)
        self.mask_ratio = mask_ratio
        self.sort = True

    def uniform_rand(self):
        mask = list(range(self.mask_size))
        random.shuffle(mask)
        mask_len = int(self.mask_size * self.mask_ratio)
        masked_tokens = mask[:mask_len]
        unmasked_tokens = mask[mask_len:]
        if self.sort:
            masked_tokens = sorted(masked_tokens)
            unmasked_tokens = sorted(unmasked_tokens)
        return unmasked_tokens, masked_tokens

    def forward(self):
        unmasked_tokens, masked_tokens = self.uniform_rand()
        return unmasked_tokens, masked_tokens


class LSTTNTransformer(nn.Module):
    """
    Transformer encoder used to produce patch-level representations from
    long historical sequences.  Supports both pretrain and inference modes.
    In LibCity integration the model is used in inference mode only.
    """
    def __init__(
        self, patch_size, in_channel, out_channel, dropout,
        mask_size, mask_ratio, num_encoder_layers=6, mode="inference"
    ):
        super().__init__()
        self.patch_size = patch_size
        self.selected_feature = 0
        self.mode = mode
        self.patch = InputEmbedding(patch_size, in_channel, out_channel)
        self.pe = PositionalEncoding(out_channel, dropout=dropout)
        self.mask = MaskGenerator(mask_size, mask_ratio)
        self.encoder = TransformerLayers(out_channel, num_encoder_layers)
        self.encoder_2_decoder = nn.Linear(out_channel, out_channel)
        self.decoder = TransformerLayers(out_channel, 1)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, out_channel))
        trunc_normal_(self.mask_token, std=0.02)
        self.output_layer = nn.Linear(out_channel, patch_size)

    def _forward_pretrain(self, x):
        batch_size, num_nodes, num_features, long_seq_len = x.size()
        patches = self.patch(x)
        patches = patches.transpose(-1, -2)
        patches = self.pe(patches)

        indices_not_masked, indices_masked = self.mask()
        repr_not_masked = patches[:, :, indices_not_masked, :]
        hidden_not_masked = self.encoder(repr_not_masked)
        hidden_not_masked = self.encoder_2_decoder(hidden_not_masked)
        hidden_masked = self.pe(
            self.mask_token.expand(
                batch_size, num_nodes, len(indices_masked),
                hidden_not_masked.size(-1)
            ),
            indices=indices_masked
        )
        hidden = torch.cat([hidden_not_masked, hidden_masked], dim=-2)
        hidden = self.decoder(hidden)

        output = self.output_layer(hidden)
        output_masked = output[:, :, len(indices_not_masked):, :]
        output_masked = output_masked.view(batch_size, num_nodes, -1).transpose(1, 2)

        labels = (
            x.permute(0, 3, 1, 2)
            .unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :]
            .transpose(1, 2)
        )
        labels_masked = labels[:, :, indices_masked, :].contiguous()
        labels_masked = labels_masked.view(batch_size, num_nodes, -1).transpose(1, 2)
        return output_masked, labels_masked

    def _forward_backend(self, x):
        patches = self.patch(x)
        patches = patches.transpose(-1, -2)
        patches = self.pe(patches)
        hidden = self.encoder(patches)
        return hidden

    def forward(self, input_data):
        if self.mode == "pretrain":
            return self._forward_pretrain(input_data)
        else:
            return self._forward_backend(input_data)


# ---------------------------------------------------------------------------
# GraphWaveNet backbone (short-term spatiotemporal feature extractor)
# ---------------------------------------------------------------------------

class _NConv(nn.Module):
    """Neighbourhood convolution (graph convolution via einsum)."""
    def __init__(self):
        super().__init__()

    def forward(self, x, A):
        A = A.to(x.device)
        if len(A.shape) == 3:
            x = torch.einsum("ncvl,nvw->ncwl", (x, A))
        else:
            x = torch.einsum("ncvl,vw->ncwl", (x, A))
        return x.contiguous()


class _Linear(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.mlp = nn.Conv2d(
            c_in, c_out,
            kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True
        )

    def forward(self, x):
        return self.mlp(x)


class _GCN(nn.Module):
    """Graph convolution used inside GraphWaveNet."""
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        self.nconv = _NConv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = _Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            a = a.to(x.device)
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class _GraphWaveNet(nn.Module):
    """
    GraphWaveNet backbone for short-term spatiotemporal feature extraction.
    This is an internal helper class -- not registered as a standalone
    LibCity model.
    """
    def __init__(
        self, num_nodes, supports,
        dropout=0.3, gcn_bool=True, addaptadj=True, aptinit=None,
        in_dim=2, out_dim=12,
        residual_channels=32, dilation_channels=32,
        skip_channels=256, end_channels=512,
        kernel_size=2, blocks=4, layers=2,
        **kwargs
    ):
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels,
            kernel_size=(1, 1)
        )
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(
                    torch.randn(num_nodes, 10), requires_grad=True
                )
                self.nodevec2 = nn.Parameter(
                    torch.randn(10, num_nodes), requires_grad=True
                )
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.residual_convs.append(
                    nn.Conv1d(
                        in_channels=dilation_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(
                        _GCN(
                            dilation_channels, residual_channels,
                            dropout, support_len=self.supports_len
                        )
                    )

        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels, out_channels=end_channels,
            kernel_size=(1, 1), bias=True
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels, out_channels=out_dim,
            kernel_size=(1, 1), bias=True
        )
        self.receptive_field = receptive_field

    def forward(self, x):
        # x: (batch, num_features, num_nodes, seq_len)
        x = F.pad(x, (1, 0, 0, 0))
        x = x[:, :2, :, :]
        in_len = x.size(3)
        if in_len < self.receptive_field:
            x = F.pad(x, (self.receptive_field - in_len, 0, 0, 0))
        x = self.start_conv(x)
        skip = 0

        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(
                F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1
            )
            new_supports = self.supports + [adp]

        for i in range(self.blocks * self.layers):
            residual = x
            filt = self.filter_convs[i](residual)
            filt = torch.tanh(filt)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filt * gate

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except Exception:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


# ---------------------------------------------------------------------------
# LSTTN-specific sub-modules
# ---------------------------------------------------------------------------

class StackedDilatedConv(nn.Module):
    """
    Extracts long-term trend features from the Transformer-encoded long
    historical representation via stacked dilated 1-D convolutions.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=2, dilation=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, dilation=2, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, dilation=4, padding=4)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, dilation=8, padding=8)
        self.pool4 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        h = self.conv1(x)
        h = F.gelu(h)
        h = self.pool1(h)
        h = self.conv2(h)
        h = F.gelu(h)
        h = self.pool2(h)
        h = self.conv3(h)
        h = F.gelu(h)
        h = self.pool3(h)
        h = self.conv4(h)
        h = F.gelu(h)
        h = self.pool4(h)
        return h


class DynamicGraphConv(nn.Module):
    """
    Graph convolution with an adaptive adjacency matrix learned from
    node embeddings.  Used for weekly and daily seasonality extraction.
    """
    def __init__(self, num_nodes, input_dim, output_dim, dropout,
                 support_len=3, order=2):
        super().__init__()
        self.node_vec1 = nn.Parameter(torch.randn(num_nodes, 10))
        self.node_vec2 = nn.Parameter(torch.randn(10, num_nodes))
        input_dim_expanded = (order * support_len + 1) * input_dim
        self.linear = nn.Linear(input_dim_expanded, output_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.order = order

    def conv(self, x, adj_mx):
        return torch.einsum("bnh,nm->bnh", (x, adj_mx)).contiguous()

    def forward(self, x, supports):
        outputs = [x]
        new_supports = list(supports)
        adaptive = torch.softmax(
            torch.relu(torch.mm(self.node_vec1, self.node_vec2)), dim=1
        )
        new_supports.append(adaptive)
        for adj_mx in new_supports:
            adj_mx = adj_mx.to(x.device)
            x1 = self.conv(x, adj_mx)
            outputs.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.conv(x1, adj_mx)
                outputs.append(x2)
                x1 = x2
        outputs = torch.cat(outputs, dim=2)
        outputs = self.linear(outputs)
        outputs = self.dropout(outputs)
        return outputs


# ===========================================================================
# Main LSTTN model adapted for LibCity
# ===========================================================================

class LSTTN(AbstractTrafficStateModel):
    """
    LSTTN: Long Short-Term Transformer-based Network for Traffic Flow
    Forecasting, adapted for the LibCity framework.

    Config parameters
    -----------------
    input_window : int
        Total number of historical time steps available in batch['X'].
        Should be equal to long_seq_len (default 4032) so that the full
        long history is present.  The last ``short_seq_len`` steps of this
        window serve as the short-term input, and the entire window serves
        as the long-term input.
    output_window : int
        Number of steps to predict (default 12).
    short_seq_len : int
        Length of the short-term input subsequence (default 12).
    long_seq_len : int
        Length of the long-term input subsequence (default 4032 = 288*7*2).
    transformer_hidden_dim : int
        Hidden dimension of the Transformer encoder (default 64).
    transformer_dropout : float
        Dropout in the Transformer encoder (default 0.1).
    transformer_mask_ratio : float
        Mask ratio used during pre-training (default 0.75).
    num_transformer_encoder_layers : int
        Number of Transformer encoder layers (default 4).
    long_trend_hidden_dim : int
        Hidden dim of the StackedDilatedConv trend extractor (default 4).
    seasonality_hidden_dim : int
        Hidden dim of the DynamicGraphConv seasonality extractors (default 4).
    mlp_hidden_dim : int
        Hidden dim of the fusion MLP (default 128).
    gwnet_dropout : float
        Dropout in GraphWaveNet (default 0.3).
    gwnet_gcn_bool : bool
        Whether to use GCN in GraphWaveNet (default True).
    gwnet_addaptadj : bool
        Whether to use adaptive adjacency in GraphWaveNet (default True).
    gwnet_in_dim : int
        Input feature dimension for GraphWaveNet (default 2).
    gwnet_out_dim : int
        Output dimension of GraphWaveNet (default 128).
    gwnet_residual_channels : int (default 32)
    gwnet_dilation_channels : int (default 32)
    gwnet_skip_channels : int (default 256)
    gwnet_end_channels : int (default 512)
    gwnet_kernel_size : int (default 2)
    gwnet_blocks : int (default 4)
    gwnet_layers : int (default 2)
    model_dropout : float
        Dropout in DynamicGraphConv modules (default 0.3).
    adjtype : str
        Adjacency type -- currently only 'doubletransition' (default).
    pretrained_transformer_path : str
        Path to a pretrained Transformer checkpoint.  If empty or
        'none', the Transformer is trained end-to-end (weights NOT
        frozen).
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))

        # ----- data features -----
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)

        # ----- sequence dimensions -----
        self.input_window = config.get('input_window', 4032)
        self.output_window = config.get('output_window', 12)
        self.short_seq_len = config.get('short_seq_len', 12)
        self.long_seq_len = config.get('long_seq_len', 4032)
        self.pre_len = self.output_window

        # ----- Transformer config -----
        self.transformer_hidden_dim = config.get('transformer_hidden_dim', 64)
        self.transformer_dropout = config.get('transformer_dropout', 0.1)
        self.transformer_mask_ratio = config.get('transformer_mask_ratio', 0.75)
        self.num_transformer_encoder_layers = config.get(
            'num_transformer_encoder_layers', 4
        )
        self.patch_size = self.short_seq_len
        self.mask_size = self.long_seq_len / self.patch_size

        # ----- LSTTN branch config -----
        self.long_trend_hidden_dim = config.get('long_trend_hidden_dim', 4)
        self.seasonality_hidden_dim = config.get('seasonality_hidden_dim', 4)
        self.mlp_hidden_dim = config.get('mlp_hidden_dim', 128)
        self.model_dropout = config.get('model_dropout', 0.3)

        # ----- GraphWaveNet config -----
        self.gwnet_dropout = config.get('gwnet_dropout', 0.3)
        self.gwnet_gcn_bool = config.get('gwnet_gcn_bool', True)
        self.gwnet_addaptadj = config.get('gwnet_addaptadj', True)
        self.gwnet_in_dim = config.get('gwnet_in_dim', 2)
        self.gwnet_out_dim = config.get('gwnet_out_dim', 128)
        self.gwnet_residual_channels = config.get('gwnet_residual_channels', 32)
        self.gwnet_dilation_channels = config.get('gwnet_dilation_channels', 32)
        self.gwnet_skip_channels = config.get('gwnet_skip_channels', 256)
        self.gwnet_end_channels = config.get('gwnet_end_channels', 512)
        self.gwnet_kernel_size = config.get('gwnet_kernel_size', 2)
        self.gwnet_blocks = config.get('gwnet_blocks', 4)
        self.gwnet_layers = config.get('gwnet_layers', 2)

        # ----- adjacency / supports -----
        adj_mx_raw = self.data_feature.get('adj_mx')
        self.adjtype = config.get('adjtype', 'doubletransition')
        adj_list = _build_doubletransition_supports(adj_mx_raw)
        self.supports = [
            torch.tensor(a, dtype=torch.float32).to(self.device)
            for a in adj_list
        ]

        # ----- build sub-modules -----
        # Transformer (inference mode: encoder only, no masking)
        self.transformer = LSTTNTransformer(
            patch_size=self.patch_size,
            in_channel=1,
            out_channel=self.transformer_hidden_dim,
            dropout=self.transformer_dropout,
            mask_size=self.mask_size,
            mask_ratio=self.transformer_mask_ratio,
            num_encoder_layers=self.num_transformer_encoder_layers,
            mode="inference",
        )

        # Optionally load pretrained Transformer weights and freeze them
        pretrained_path = config.get('pretrained_transformer_path', '')
        if pretrained_path and pretrained_path.lower() != 'none':
            self._load_pretrained_transformer(pretrained_path)

        # Long-term trend extractor
        self.long_term_trend_extractor = StackedDilatedConv(
            self.transformer_hidden_dim, self.long_trend_hidden_dim
        )

        # Seasonality extractors
        self.weekly_seasonality_extractor = DynamicGraphConv(
            self.num_nodes, self.transformer_hidden_dim,
            self.seasonality_hidden_dim, self.model_dropout
        )
        self.daily_seasonality_extractor = DynamicGraphConv(
            self.num_nodes, self.transformer_hidden_dim,
            self.seasonality_hidden_dim, self.model_dropout
        )

        # Short-term extractor (GraphWaveNet)
        self.short_term_trend_extractor = _GraphWaveNet(
            num_nodes=self.num_nodes,
            supports=self.supports,
            dropout=self.gwnet_dropout,
            gcn_bool=self.gwnet_gcn_bool,
            addaptadj=self.gwnet_addaptadj,
            aptinit=None,
            in_dim=self.gwnet_in_dim,
            out_dim=self.gwnet_out_dim,
            residual_channels=self.gwnet_residual_channels,
            dilation_channels=self.gwnet_dilation_channels,
            skip_channels=self.gwnet_skip_channels,
            end_channels=self.gwnet_end_channels,
            kernel_size=self.gwnet_kernel_size,
            blocks=self.gwnet_blocks,
            layers=self.gwnet_layers,
        )

        # Fusion MLPs
        self.trend_seasonality_mlp = nn.Sequential(
            nn.Linear(
                self.long_trend_hidden_dim + self.seasonality_hidden_dim * 2,
                self.mlp_hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
        )
        self.mlp = nn.Sequential(
            nn.Linear(
                self.mlp_hidden_dim + self.gwnet_out_dim,
                self.mlp_hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, self.pre_len),
        )

    def _load_pretrained_transformer(self, path):
        """Load pretrained Transformer weights and freeze parameters."""
        if not os.path.isfile(path):
            self._logger.warning(
                'Pretrained Transformer checkpoint not found at %s. '
                'The Transformer will be trained from scratch.', path
            )
            return
        self._logger.info(
            'Loading pretrained Transformer from %s', path
        )
        state_dict = torch.load(path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.transformer.load_state_dict(state_dict)
        for param in self.transformer.parameters():
            param.requires_grad = False
        self._logger.info('Pretrained Transformer loaded and frozen.')

    def forward(self, batch):
        """
        Forward pass.

        Args:
            batch: dict with at least keys 'X' and 'y'.
                batch['X'] shape: (batch_size, input_window, num_nodes, feature_dim)
                batch['y'] shape: (batch_size, output_window, num_nodes, output_dim)

        Returns:
            output: (batch_size, output_window, num_nodes, 1)
        """
        x = batch['X']  # (B, T, N, F)
        batch_size = x.size(0)

        # --- split short and long sequences ---
        # short_x: last short_seq_len steps, shape (B, short_seq_len, N, F)
        short_x = x[:, -self.short_seq_len:, :, :]

        # long_x: last long_seq_len steps (or all available), shape (B, L, N, F)
        if x.size(1) >= self.long_seq_len:
            long_x = x[:, -self.long_seq_len:, :, :]
        else:
            long_x = x

        # =====================================================================
        # Long-term branch: Transformer + trend + seasonality
        # =====================================================================
        # Use only the first feature (traffic flow/speed) for the Transformer
        # Original: long_x[..., [0]] then permute to (B, N, 1, L)
        long_x_tsf = long_x[..., [0]]
        long_x_tsf = long_x_tsf.permute(0, 2, 3, 1)
        # long_x_tsf: (B, N, 1, L)

        long_repr = self.transformer(long_x_tsf)
        # long_repr: (B, N, num_subseq, transformer_hidden_dim)
        num_subseq = long_repr.size(2)

        # --- Long-term trend ---
        # Reshape for 1-D convolution: (B*N, transformer_hidden_dim, num_subseq)
        long_repr_flat = long_repr.reshape(
            -1, num_subseq, self.transformer_hidden_dim
        ).transpose(1, 2)
        long_trend_hidden = self.long_term_trend_extractor(long_repr_flat)
        # Take the last time step: (B*N, long_trend_hidden_dim)
        long_trend_hidden = long_trend_hidden[:, :, -1]
        # Reshape back: (B, N, long_trend_hidden_dim)
        long_trend_hidden = long_trend_hidden.reshape(
            batch_size, self.num_nodes, -1
        )

        # --- Weekly and daily seasonality ---
        # last_week_repr: repr at index -(7*24+1) = position of 1 week ago
        last_week_repr = long_repr[:, :, -7 * 24 - 1, :]
        # last_day_repr: repr at index -25 = position of 1 day ago
        last_day_repr = long_repr[:, :, -25, :]

        weekly_hidden = self.weekly_seasonality_extractor(
            last_week_repr, self.supports
        )
        daily_hidden = self.daily_seasonality_extractor(
            last_day_repr, self.supports
        )

        # Fuse trend + seasonality
        trend_seasonality_hidden = torch.cat(
            (long_trend_hidden, weekly_hidden, daily_hidden), dim=-1
        )
        trend_seasonality_hidden = self.trend_seasonality_mlp(
            trend_seasonality_hidden
        )

        # =====================================================================
        # Short-term branch: GraphWaveNet
        # =====================================================================
        # Original expects (B, F, N, T)
        short_x_gwnet = short_x.permute(0, 3, 1, 2)
        # short_x_gwnet: (B, F, short_seq_len, N) -- but GWNet expects
        # (B, F, N, T); original code does transpose(1,3) on (B,T,N,F)
        # which yields (B,F,N,T).
        short_x_gwnet = short_x.transpose(1, 3)
        # short_x_gwnet: (B, F, N, T)

        short_term_hidden = self.short_term_trend_extractor(short_x_gwnet)
        # (B, gwnet_out_dim, N, 1)  -- squeeze last dim and transpose
        short_term_hidden = short_term_hidden.squeeze(-1).transpose(1, 2)
        # (B, N, gwnet_out_dim)

        # =====================================================================
        # Fusion and prediction
        # =====================================================================
        hidden = torch.cat(
            (short_term_hidden, trend_seasonality_hidden), dim=-1
        )
        # (B, N, mlp_hidden_dim + gwnet_out_dim) -> (B, N, pre_len)
        output = self.mlp(hidden)

        # Reshape to LibCity convention: (B, output_window, N, 1)
        output = output.transpose(1, 2).unsqueeze(-1)
        return output

    def predict(self, batch):
        """
        Return predictions for evaluation.  Shape: (B, output_window, N, output_dim).
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Compute training loss (masked MAE on inverse-transformed values).
        """
        y_true = batch['y']  # (B, output_window, N, output_dim)
        y_predicted = self.predict(batch)

        # Inverse-transform both prediction and ground truth
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        return loss.masked_mae_torch(y_predicted, y_true, 0)
