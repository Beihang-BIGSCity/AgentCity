"""
DST-Mamba: Deep Spatio-Temporal Mamba for Traffic Speed Prediction

This module adapts the DST-Mamba model to the LibCity framework.

Original Source:
- Repository: repos/DST-Mamba
- Main Model: baselines/DSTMamba/DSTMamba_Arch.py
- Components: MambaEnc.py, SeriesDec.py, SeriesMix.py, Embed.py, RevIN.py

Key Adaptations:
1. Changed inheritance from nn.Module to AbstractTrafficStateModel
2. Modified forward() to accept LibCity batch dictionary format
3. Mapped configuration parameters to LibCity conventions:
   - history_seq_len -> input_window
   - future_seq_len -> output_window
   - num_channels -> num_nodes
4. Added predict() and calculate_loss() methods required by LibCity
5. Integrated with LibCity's scaler system for data normalization

Dependencies:
- mamba-ssm: For Mamba state space model layers
- einops: For tensor reshaping operations

Configuration Parameters:
- input_window: Length of input sequence (default: 12)
- output_window: Length of output/prediction sequence (default: 12)
- d_model: Model dimension (default: 128)
- use_norm: Whether to use RevIN normalization (default: True)
- emb_dropout: Dropout rate for embeddings (default: 0.1)
- decom_type: Decomposition type, 'STD' for standard (default: 'STD')
- std_kernel: Kernel size for temporal decomposition (default: 25)
- rank: Rank for adapter (default: 8)
- node_dim: Dimension for node embeddings (default: 32)
- e_layers: Number of encoder layers (default: 2)
- d_state: State dimension for Mamba (default: 16)
- d_conv: Convolution dimension for Mamba (default: 4)
- expand: Expansion factor for Mamba (default: 2)
- d_ff: Feed-forward dimension (default: 256)
- ffn_dropout: FFN dropout rate (default: 0.1)
- ffn_activation: Activation function ('relu' or 'gelu') (default: 'gelu')
- ds_type: Downsampling type ('max', 'avg', 'conv') (default: 'avg')
- ds_layers: Number of downsampling layers (default: 2)
- ds_window: Downsampling window size (default: 2)
- initial_tre_w: Initial trend weight (default: 0.5)

Limitations:
- RevIN normalization operates independently of LibCity's scaler (by design)
- Original model uses channel-independent processing
"""

from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import repeat
from mamba_ssm import Mamba

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


# ==============================================================================
# Component: RevIN (Reversible Instance Normalization)
# Original: baselines/DSTMamba/RevIN.py
# ==============================================================================
class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time series.
    Normalizes input data and can reverse the normalization on output.
    """
    def __init__(self,
                 num_features: int,
                 eps=1e-5,
                 affine=True,
                 subtract_last=False):
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


# ==============================================================================
# Component: Temporal Decomposition (Moving Average)
# Original: baselines/DSTMamba/SeriesDec.py
# ==============================================================================
class MovingAvg(nn.Module):
    """
    Moving average layer for temporal decomposition.
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Pad input to handle boundary effects
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class TemporalDecomposition(nn.Module):
    """
    Decomposes time series into seasonal and trend components.
    """
    def __init__(self, kernel_size):
        super(TemporalDecomposition, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# ==============================================================================
# Component: Multi-Scale Trend Mixing
# Original: baselines/DSTMamba/SeriesMix.py
# ==============================================================================
class MultiScaleTrendMixing(nn.Module):
    """
    Multi-scale trend mixing module for capturing trends at different temporal scales.
    """
    def __init__(self,
                 history_seq_len,
                 future_seq_len,
                 num_channels,
                 ds_layers,
                 ds_window):
        super(MultiScaleTrendMixing, self).__init__()

        self.history_seq_len = history_seq_len
        self.future_seq_len = future_seq_len
        self.num_channels = num_channels
        self.ds_layers = ds_layers
        self.ds_window = ds_window

        # Length Alignment via upsampling
        self.up_sampling = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        self.history_seq_len // (self.ds_window ** (layer + 1)),
                        self.history_seq_len // (self.ds_window ** layer),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        self.history_seq_len // (self.ds_window ** layer),
                        self.history_seq_len // (self.ds_window ** layer),
                    )
                ) for layer in reversed(range(self.ds_layers))
            ]
        )

    def forward(self, ms_trend_list):
        length_list = []
        trend_list = []
        for x in ms_trend_list:
            _, t, _ = x.size()
            length_list.append(t)
            trend_list.append(x.permute(0, 2, 1))  # [B, N, t]

        # Trend mixing from coarse to fine
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()

        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]

        out_trend_list = [out_low.permute(0, 2, 1)]
        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()

        out_list = []
        for out_trend, length in zip(out_trend_list, length_list):
            out_list.append(out_trend[:, :length, :])  # list of [B, t, C]

        return out_list


# ==============================================================================
# Component: Data Embedding
# Original: baselines/DSTMamba/Embed.py
# ==============================================================================
class DataEmbedding(nn.Module):
    """
    Embedding layer for time series data.
    """
    def __init__(self, history_seq_len, d_model, dropout):
        super(DataEmbedding, self).__init__()
        self.ValueEmb = nn.Linear(history_seq_len, d_model)
        self.Dropout = nn.Dropout(p=dropout)

    def forward(self, x_in):
        # x_in: (batch_size, history_seq_len, num_channels)
        x_in = x_in.permute(0, 2, 1)  # -> (batch_size, num_channels, history_seq_len)
        x_emb = self.ValueEmb(x_in)   # -> (batch_size, num_channels, d_model)
        x_emb = self.Dropout(x_emb)
        return x_emb


# ==============================================================================
# Component: Encoder (Mamba-based)
# Original: baselines/DSTMamba/MambaEnc.py
# ==============================================================================
class Encoder(nn.Module):
    """
    Encoder composed of multiple Mamba-based encoder layers.
    """
    def __init__(self, ssm_layers, norm_layer):
        super(Encoder, self).__init__()
        self.ssm_layers = nn.ModuleList(ssm_layers)
        self.norm_layer = norm_layer

    def forward(self, x_emb):
        # x_emb: [batch_size, num_channels, d_model]
        x_enc = x_emb
        for ssm_layer in self.ssm_layers:
            x_enc = ssm_layer(x_enc)
        if self.norm_layer is not None:
            enc_out = self.norm_layer(x_enc)
        return enc_out


class EncoderLayer(nn.Module):
    """
    Single encoder layer with bidirectional Mamba and feed-forward network.
    """
    def __init__(self, ssm, ssm_r, d_model, d_ff, dropout, activation):
        super(EncoderLayer, self).__init__()

        self.ssm = ssm
        self.ssm_r = ssm_r  # Reverse direction Mamba for bidirectional processing

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x_enc):
        # Bidirectional Mamba processing
        if self.ssm_r is not None:
            ssm_out = self.ssm(x_enc) + self.ssm_r(x_enc.flip(dims=[1])).flip(dims=[1])
        else:
            ssm_out = self.ssm(x_enc)

        out = x_enc = self.norm1(ssm_out)
        out = self.dropout(self.activation(self.conv1(out.transpose(-1, 1))))
        out = self.dropout(self.conv2(out).transpose(-1, 1))

        return self.norm2(out + x_enc)


# ==============================================================================
# Main Model: DSTMamba (LibCity-adapted)
# Original: baselines/DSTMamba/DSTMamba_Arch.py
# ==============================================================================
class DSTMamba(AbstractTrafficStateModel):
    """
    DST-Mamba: Deep Spatio-Temporal Mamba for Traffic Speed Prediction.

    This model uses Mamba (State Space Model) for temporal modeling combined
    with multi-scale trend decomposition and mixing for traffic prediction.

    Adapted from the original implementation to work with LibCity framework.
    """

    def __init__(self, config, data_feature):
        super(DSTMamba, self).__init__(config, data_feature)

        self._logger = getLogger()

        # Data feature extraction (LibCity convention)
        self.num_nodes = data_feature.get('num_nodes')
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)
        self._scaler = data_feature.get('scaler')

        # Sequence length configuration (LibCity mapping)
        # history_seq_len -> input_window, future_seq_len -> output_window
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)

        # For internal model, map to original naming
        self.history_seq_len = self.input_window
        self.future_seq_len = self.output_window
        self.num_channels = self.num_nodes  # In DST-Mamba, channels = nodes

        # Model hyperparameters
        self.d_model = config.get('d_model', 128)
        self.use_norm = config.get('use_norm', True)
        self.emb_dropout = config.get('emb_dropout', 0.1)
        self.decom_type = config.get('decom_type', 'STD')
        self.std_kernel = config.get('std_kernel', 25)

        # Adapter parameters for spatial embeddings
        self.rank = config.get('rank', 8)
        self.node_dim = config.get('node_dim', 32)

        # Mamba encoder parameters
        self.e_layers = config.get('e_layers', 2)
        self.d_state = config.get('d_state', 16)
        self.d_conv = config.get('d_conv', 4)
        self.expand = config.get('expand', 2)

        # Feed-forward network parameters
        self.d_ff = config.get('d_ff', 256)
        self.ffn_dropout = config.get('ffn_dropout', 0.1)
        self.ffn_activation = config.get('ffn_activation', 'gelu')

        # Downsampling parameters for multi-scale processing
        self.ds_type = config.get('ds_type', 'avg')
        assert self.ds_type in ['max', 'avg', 'conv'], "ds_type must be 'max', 'avg', or 'conv'"
        self.ds_layers = config.get('ds_layers', 2)
        self.ds_window = config.get('ds_window', 2)

        # Trend weight initialization
        self.initial_tre_w = config.get('initial_tre_w', 0.5)

        # Device configuration
        self.device = config.get('device', torch.device('cpu'))

        # Build model components
        self._build()

        self._logger.info("DSTMamba model initialized with {} nodes, input_window={}, output_window={}".format(
            self.num_nodes, self.input_window, self.output_window))

    def _build(self):
        """Build all model components."""
        # RevIN normalization layer
        self.revin_layer = RevIN(num_features=self.num_channels)

        # Temporal decomposition
        if self.decom_type == 'STD':
            self.decom = TemporalDecomposition(self.std_kernel)

        # Embedding layer
        embed_dim = self.d_model - self.node_dim
        self.embedding = DataEmbedding(self.history_seq_len, embed_dim, self.emb_dropout)

        # Low-rank adapter for spatial embeddings
        self.adapter = nn.Parameter(torch.empty(self.num_channels, embed_dim, self.rank))  # [N, E, r]
        nn.init.xavier_uniform_(self.adapter)
        self.lora = nn.Linear(self.rank, self.node_dim, bias=False)

        # Mamba encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ssm=Mamba(self.d_model, self.d_state, self.d_conv, self.expand),
                    ssm_r=Mamba(self.d_model, self.d_state, self.d_conv, self.expand),
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    dropout=self.ffn_dropout,
                    activation=self.ffn_activation
                ) for _ in range(self.e_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model)
        )

        # Output projection
        self.projector = nn.Linear(self.d_model, self.future_seq_len, bias=True)

        # Downsampling pooling layer
        if self.ds_type == 'max':
            self.down_pool = nn.MaxPool1d(self.ds_window, return_indices=False)
        elif self.ds_type == 'avg':
            self.down_pool = nn.AvgPool1d(self.ds_window)
        elif self.ds_type == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            self.down_pool = nn.Conv1d(
                in_channels=self.num_channels,
                out_channels=self.num_channels,
                kernel_size=3,
                padding=padding,
                stride=self.ds_window,
                padding_mode='circular',
                bias=False
            )

        # Multi-scale trend mixing
        self.ms_mixing = MultiScaleTrendMixing(
            self.history_seq_len,
            self.future_seq_len,
            self.num_channels,
            self.ds_layers,
            self.ds_window
        )

        # Linear mappings for each scale
        self.linear_mappings = nn.ModuleList(
            [
                nn.Linear(
                    self.history_seq_len // (self.ds_window ** layer),
                    self.future_seq_len
                ) for layer in range(self.ds_layers + 1)
            ]
        )

        # Learnable trend weight per channel
        self.tre_w = nn.Parameter(
            torch.FloatTensor([self.initial_tre_w] * self.num_channels),
            requires_grad=True
        )

    def forward(self, batch):
        """
        Forward pass of DSTMamba model.

        Args:
            batch (dict): LibCity batch dictionary containing:
                - 'X': Input tensor of shape [B, T_in, N, F]
                - 'y': Target tensor of shape [B, T_out, N, F] (used in calculate_loss)

        Returns:
            torch.Tensor: Predictions of shape [B, T_out, N, F]
        """
        # Extract input from LibCity batch format
        # LibCity format: [B, T, N, F] where F includes features
        x = batch['X']  # [B, T_in, N, F]

        # Extract only the first feature (traffic speed) for processing
        # Original model expects [B, T, N] format
        x_in = x[..., 0]  # [B, T_in, N]
        B, _, _ = x_in.shape

        # Apply RevIN normalization if enabled
        if self.use_norm:
            x_in = self.revin_layer(x_in, mode='norm')

        # Temporal decomposition into seasonal and trend components
        x_sea, _ = self.decom(x_in)

        # ==================== Seasonal Processing ====================
        # Embedding: [B, T, N] -> [B, N, E-D]
        x_emb = self.embedding(x_sea)

        # Add spatial embeddings via low-rank adaptation
        adaptation = []
        adapter = F.relu(self.lora(self.adapter))  # [N, E-D, r] -> [N, E-D, D]
        adapter = adapter.permute(1, 2, 0)  # [E-D, D, N]
        adapter = repeat(adapter, 'D d n -> repeat D d n', repeat=B)  # [B, E-D, D, N]
        x_emb = x_emb.transpose(1, 2)  # (B, N, E-D) -> (B, E-D, N)
        adaptation.append(torch.einsum('bDn,bDdn->bdn', [x_emb, adapter]))  # [B, D, N]
        x_emb = torch.cat([x_emb] + adaptation, dim=1)  # [B, E, N]
        x_emb = x_emb.transpose(1, 2)  # [B, E, N] -> [B, N, E]

        # Encoder: [B, N, E] -> [B, N, E]
        enc_out = self.encoder(x_emb)

        # Project to output sequence length
        x_sea_out = self.projector(enc_out).permute(0, 2, 1)  # [B, N, T_out] -> [B, T_out, N]

        # ==================== Trend Processing ====================
        # Multi-scale processing
        ms_list = [x_in]  # Start with original input [B, T, N]

        x_ms = x_in.permute(0, 2, 1)  # [B, N, T]
        for _ in range(self.ds_layers):
            x_sampling = self.down_pool(x_ms)  # [B, N, t_downsampled]
            ms_list.append(x_sampling.permute(0, 2, 1))
            x_ms = x_sampling

        # Extract trend from each scale
        ms_trend_list = []
        for x_scale in ms_list:
            _, x_tre = self.decom(x_scale)
            ms_trend_list.append(x_tre)

        # Multi-scale trend mixing
        ms_trend_list = self.ms_mixing(ms_trend_list)

        # Project each scale's trend to output length
        out_trend_list = []
        for i, trend in enumerate(ms_trend_list):
            trend_out = self.linear_mappings[i](trend.permute(0, 2, 1)).permute(0, 2, 1)
            out_trend_list.append(trend_out)

        # Sum trends from all scales
        x_tre_out = torch.stack(out_trend_list, dim=-1).sum(-1)

        # ==================== Combine and Denormalize ====================
        # Weighted sum of seasonal and trend components
        prediction = self.revin_layer(x_sea_out + self.tre_w * x_tre_out, mode='denorm')

        # Add feature dimension to match LibCity output format [B, T, N, F]
        prediction = prediction.unsqueeze(-1)

        return prediction

    def predict(self, batch):
        """
        Generate predictions for a batch.

        Args:
            batch (dict): LibCity batch dictionary

        Returns:
            torch.Tensor: Predictions of shape [B, T_out, N, F]
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate training loss.

        Args:
            batch (dict): LibCity batch dictionary containing 'X' and 'y'

        Returns:
            torch.Tensor: Scalar loss value
        """
        y_true = batch['y']  # [B, T_out, N, F]
        y_predicted = self.predict(batch)

        # Apply inverse scaling for loss calculation
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        # Use masked MAE loss (standard in traffic prediction)
        return loss.masked_mae_torch(y_predicted, y_true, 0)
