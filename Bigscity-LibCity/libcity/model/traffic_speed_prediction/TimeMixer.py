"""
TimeMixer++ model adapted for LibCity framework.

Original Paper: TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis
Original Implementation: https://github.com/kwuking/TimeMixer

Key Adaptations:
1. Inherits from AbstractTrafficStateModel for LibCity compatibility
2. Accepts (config, data_feature) parameters in __init__
3. Implements forward(), predict(), calculate_loss() methods
4. Handles LibCity batch format with 'X' and 'y' keys
5. Data shape: (batch, time_in, num_nodes, features) -> (batch, time, channels)
6. Uses masked_mae_torch loss function (L1 loss for PEMS datasets)

Configuration Parameters (set in config file):
- input_window: Input sequence length (seq_len)
- output_window: Prediction horizon (pred_len)
- d_model: Model dimension (default: 32)
- d_ff: Feed-forward dimension (default: 64)
- e_layers: Number of encoder layers (default: 2)
- down_sampling_layers: Number of down-sampling levels (default: 2)
- down_sampling_window: Down-sampling window size (default: 2)
- down_sampling_method: 'avg', 'max', or 'conv' (default: 'avg')
- decomp_method: 'moving_avg' or 'dft_decomp' (default: 'moving_avg')
- moving_avg: Moving average kernel size (default: 25)
- top_k: Top-k for DFT decomposition (default: 5)
- channel_independence: 0 for channel-dependent, 1 for channel-independent (default: 1)
- dropout: Dropout rate (default: 0.1)
- use_norm: Whether to use normalization (default: 1)
"""

from logging import getLogger
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


# ============================================================================
# Embedding Layers (from layers/Embed.py)
# ============================================================================

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model,
            kernel_size=3, padding=padding, padding_mode='circular', bias=False
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'ms': 7,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map.get(freq, 4)
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding_wo_pos(nn.Module):
    """Data embedding without positional encoding (for TimeMixer)."""

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x is None and x_mark is not None:
            return self.temporal_embedding(x_mark)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


# ============================================================================
# Normalization Layer (from layers/StandardNorm.py)
# ============================================================================

class Normalize(nn.Module):
    """Reversible instance normalization for time series."""

    def __init__(self, num_features, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
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
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


# ============================================================================
# Moving Average Decomposition (from layers/Autoformer_EncDec.py)
# ============================================================================

class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series."""

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition block using moving average."""

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# ============================================================================
# DFT-based Decomposition
# ============================================================================

class DFT_series_decomp(nn.Module):
    """Series decomposition using Discrete Fourier Transform."""

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


# ============================================================================
# Multi-Scale Mixing Modules
# ============================================================================

class MultiScaleSeasonMixing(nn.Module):
    """Bottom-up mixing of seasonal patterns across scales."""

    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super(MultiScaleSeasonMixing, self).__init__()
        self.down_sampling_layers = torch.nn.ModuleList([
            nn.Sequential(
                torch.nn.Linear(
                    seq_len // (down_sampling_window ** i),
                    seq_len // (down_sampling_window ** (i + 1)),
                ),
                nn.GELU(),
                torch.nn.Linear(
                    seq_len // (down_sampling_window ** (i + 1)),
                    seq_len // (down_sampling_window ** (i + 1)),
                ),
            )
            for i in range(down_sampling_layers)
        ])

    def forward(self, season_list):
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """Top-down mixing of trend patterns across scales."""

    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super(MultiScaleTrendMixing, self).__init__()
        self.up_sampling_layers = torch.nn.ModuleList([
            nn.Sequential(
                torch.nn.Linear(
                    seq_len // (down_sampling_window ** (i + 1)),
                    seq_len // (down_sampling_window ** i),
                ),
                nn.GELU(),
                torch.nn.Linear(
                    seq_len // (down_sampling_window ** i),
                    seq_len // (down_sampling_window ** i),
                ),
            )
            for i in reversed(range(down_sampling_layers))
        ])

    def forward(self, trend_list):
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


# ============================================================================
# Past Decomposable Mixing Block
# ============================================================================

class PastDecomposableMixing(nn.Module):
    """PDM block: combines seasonal and trend mixing with decomposition."""

    def __init__(self, seq_len, pred_len, down_sampling_window, down_sampling_layers,
                 d_model, d_ff, dropout, channel_independence, decomp_method,
                 moving_avg_kernel, top_k):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.down_sampling_window = down_sampling_window

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.channel_independence = channel_independence

        if decomp_method == 'moving_avg':
            self.decompsition = series_decomp(moving_avg_kernel)
        elif decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(top_k)
        else:
            raise ValueError('decomposition method not supported')

        if channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_ff),
                nn.GELU(),
                nn.Linear(in_features=d_ff, out_features=d_model),
            )

        self.mixing_multi_scale_season = MultiScaleSeasonMixing(
            seq_len, down_sampling_window, down_sampling_layers
        )
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(
            seq_len, down_sampling_window, down_sampling_layers
        )

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.GELU(),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # Bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # Top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(
            x_list, out_season_list, out_trend_list, length_list
        ):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


# ============================================================================
# TimeMixer Model for LibCity
# ============================================================================

class TimeMixer(AbstractTrafficStateModel):
    """
    TimeMixer++ adapted for LibCity traffic speed prediction.

    This model performs multi-scale time series forecasting by:
    1. Multi-scale decomposition of input sequences
    2. Past Decomposable Mixing (PDM) for encoding
    3. Future Multi-Predictor Mixing for decoding

    Reference:
        TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()

        # Device
        self.device = config.get('device', torch.device('cpu'))

        # Data features from LibCity
        self.num_nodes = data_feature.get('num_nodes', 1)
        self._scaler = data_feature.get('scaler')
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)

        # Sequence lengths
        self.seq_len = config.get('input_window', 12)
        self.pred_len = config.get('output_window', 12)
        self.label_len = config.get('label_len', 0)

        # Model dimensions
        self.d_model = config.get('d_model', 32)
        self.d_ff = config.get('d_ff', 64)
        self.e_layers = config.get('e_layers', 2)
        self.dropout = config.get('dropout', 0.1)

        # Multi-scale settings
        self.down_sampling_layers = config.get('down_sampling_layers', 2)
        self.down_sampling_window = config.get('down_sampling_window', 2)
        self.down_sampling_method = config.get('down_sampling_method', 'avg')

        # Decomposition settings
        self.decomp_method = config.get('decomp_method', 'moving_avg')
        self.moving_avg = config.get('moving_avg', 25)
        self.top_k = config.get('top_k', 5)

        # Channel settings: for traffic data, treat each node as a channel
        # channel_independence=1 means each node is processed independently
        self.channel_independence = config.get('channel_independence', 1)
        self.use_norm = config.get('use_norm', 1)

        # For traffic prediction: enc_in = num_nodes * output_dim (channels)
        # In channel-independent mode, we process each channel separately
        self.enc_in = self.num_nodes * self.output_dim
        self.c_out = self.num_nodes * self.output_dim

        # Embedding settings
        self.embed = config.get('embed', 'timeF')
        self.freq = config.get('freq', 'h')
        self.use_future_temporal_feature = config.get('use_future_temporal_feature', False)

        # Build model components
        self._build_model()

        self._logger.info(
            f"TimeMixer initialized: seq_len={self.seq_len}, pred_len={self.pred_len}, "
            f"num_nodes={self.num_nodes}, d_model={self.d_model}, e_layers={self.e_layers}, "
            f"down_sampling_layers={self.down_sampling_layers}"
        )

    def _build_model(self):
        """Build all model components."""
        # PDM blocks (encoder)
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                down_sampling_window=self.down_sampling_window,
                down_sampling_layers=self.down_sampling_layers,
                d_model=self.d_model,
                d_ff=self.d_ff,
                dropout=self.dropout,
                channel_independence=self.channel_independence,
                decomp_method=self.decomp_method,
                moving_avg_kernel=self.moving_avg,
                top_k=self.top_k
            )
            for _ in range(self.e_layers)
        ])

        # Preprocessing decomposition
        self.preprocess = series_decomp(self.moving_avg)

        # Embedding layer
        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(
                1, self.d_model, self.embed, self.freq, self.dropout
            )
        else:
            self.enc_embedding = DataEmbedding_wo_pos(
                self.enc_in, self.d_model, self.embed, self.freq, self.dropout
            )

        # Normalization layers for each scale
        self.normalize_layers = torch.nn.ModuleList([
            Normalize(
                self.enc_in,
                affine=True,
                non_norm=True if self.use_norm == 0 else False
            )
            for _ in range(self.down_sampling_layers + 1)
        ])

        # Prediction layers for each scale
        self.predict_layers = torch.nn.ModuleList([
            torch.nn.Linear(
                self.seq_len // (self.down_sampling_window ** i),
                self.pred_len,
            )
            for i in range(self.down_sampling_layers + 1)
        ])

        # Projection layer
        if self.channel_independence == 1:
            self.projection_layer = nn.Linear(self.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(self.d_model, self.c_out, bias=True)
            self.out_res_layers = torch.nn.ModuleList([
                torch.nn.Linear(
                    self.seq_len // (self.down_sampling_window ** i),
                    self.seq_len // (self.down_sampling_window ** i),
                )
                for i in range(self.down_sampling_layers + 1)
            ])
            self.regression_layers = torch.nn.ModuleList([
                torch.nn.Linear(
                    self.seq_len // (self.down_sampling_window ** i),
                    self.pred_len,
                )
                for i in range(self.down_sampling_layers + 1)
            ])

    def _multi_scale_process_inputs(self, x_enc):
        """Generate multi-scale representations of input."""
        if self.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(
                in_channels=self.enc_in, out_channels=self.enc_in,
                kernel_size=3, padding=padding,
                stride=self.down_sampling_window,
                padding_mode='circular',
                bias=False
            ).to(x_enc.device)
        else:
            return [x_enc]

        # B,T,C -> B,C,T for pooling
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc

        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))

        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

        return x_enc_sampling_list

    def pre_enc(self, x_list):
        """Preprocess before encoding."""
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def out_projection(self, dec_out, i, out_res):
        """Output projection for channel-dependent mode."""
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        """Future Multi-Predictor Mixing for decoding."""
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)
        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def forecast(self, x_enc):
        """
        Forecast future values.

        Args:
            x_enc: Input tensor of shape (B, T, N) where
                   B = batch size, T = seq_len, N = num_channels

        Returns:
            Prediction tensor of shape (B, pred_len, N)
        """
        # Multi-scale processing
        x_enc_list = self._multi_scale_process_inputs(x_enc)

        x_list = []
        for i, x in enumerate(x_enc_list):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence == 1:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # Embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        for i, x in enumerate(x_list[0]):
            enc_out = self.enc_embedding(x, None)
            enc_out_list.append(enc_out)

        # PDM blocks (encoder)
        for i in range(self.e_layers):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multi-Predictor Mixing (decoder)
        B = x_enc.shape[0]
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        # Aggregate predictions from all scales
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')

        return dec_out

    def forward(self, batch):
        """
        Forward pass for LibCity batch format.

        Args:
            batch: Dictionary with 'X' key containing input tensor
                   X shape: (batch, time_in, num_nodes, features)

        Returns:
            Prediction tensor of shape (batch, time_out, num_nodes, output_dim)
        """
        # Get input: (batch, time_in, num_nodes, features)
        x = batch['X']

        # Use only the target feature (usually speed, first channel)
        # x shape: (B, T, N, F) -> (B, T, N)
        x = x[..., :self.output_dim]

        batch_size, seq_len, num_nodes, output_dim = x.shape

        # Reshape for TimeMixer: (B, T, N*output_dim)
        # TimeMixer treats each (node, feature) pair as a channel
        x = x.reshape(batch_size, seq_len, num_nodes * output_dim)

        # Forward through TimeMixer
        # Output shape: (B, pred_len, N*output_dim)
        dec_out = self.forecast(x)

        # Reshape back to LibCity format: (B, pred_len, N, output_dim)
        dec_out = dec_out.reshape(batch_size, self.pred_len, num_nodes, output_dim)

        return dec_out

    def predict(self, batch):
        """
        Prediction method for LibCity.

        Args:
            batch: Dictionary with 'X' key

        Returns:
            Prediction tensor
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate loss for LibCity training.

        Uses masked MAE (L1) loss which is standard for PEMS datasets.

        Args:
            batch: Dictionary with 'X' and 'y' keys

        Returns:
            Loss tensor
        """
        y_true = batch['y']
        y_predicted = self.predict(batch)

        # Inverse transform to original scale
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        # Use masked MAE loss (L1 loss with masking for zero values)
        return loss.masked_mae_torch(y_predicted, y_true, null_val=0.0)
