"""
TimeMixerPP: TimeMixer++ adapted for LibCity framework.

Original Paper: TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis
Original Repository: https://github.com/kwuking/TimeMixer

Key Adaptations:
1. Inherits from AbstractTrafficStateModel for LibCity compatibility
2. Handles LibCity's batch dict format (batch['X'], batch['y'])
3. Input shape: [batch_size, input_window, num_nodes, feature_dim]
4. Output shape: [batch_size, output_window, num_nodes, output_dim]
5. Treats each node's features as independent channels (channel_independence=1)
6. Integrates with LibCity's scaler for normalization/denormalization

Required Config Parameters:
- d_model: Model dimension (default: 64)
- d_ff: Feed-forward dimension (default: 128)
- e_layers: Number of encoder layers (default: 2)
- down_sampling_layers: Number of downsampling layers (default: 2)
- down_sampling_window: Downsampling window size (default: 2)
- down_sampling_method: Method for downsampling ('avg', 'max', 'conv') (default: 'avg')
- decomp_method: Decomposition method ('moving_avg', 'dft_decomp') (default: 'moving_avg')
- moving_avg: Moving average kernel size (default: 25)
- top_k: Top-k for DFT decomposition (default: 5)
- channel_independence: Whether channels are independent (default: 1)
- use_norm: Whether to use normalization (default: 1)
- dropout: Dropout rate (default: 0.1)
"""

import math
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


# ==============================================================================
# Layer Components (adapted from TimeMixer layers)
# ==============================================================================

class Normalize(nn.Module):
    """RevIN-style normalization layer."""

    def __init__(self, num_features: int, eps=1e-5, affine=True, non_norm=False):
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.non_norm = non_norm
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

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
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
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
        x = x + self.mean
        return x


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series."""

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Padding on both ends of time series
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


class DFT_series_decomp(nn.Module):
    """Series decomposition block using DFT."""

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, min(self.top_k, freq.shape[-1]))
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class TokenEmbedding(nn.Module):
    """Token embedding using 1D convolution."""

    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode='circular',
            bias=False
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding_wo_pos(nn.Module):
    """Data embedding without positional encoding."""

    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        if x is None:
            return None
        x = self.value_embedding(x)
        return self.dropout(x)


class MultiScaleSeasonMixing(nn.Module):
    """Bottom-up mixing for seasonal patterns."""

    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super(MultiScaleSeasonMixing, self).__init__()
        self.down_sampling_layers = torch.nn.ModuleList(
            [
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
            ]
        )

    def forward(self, season_list):
        # Mixing high -> low
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
    """Top-down mixing for trend patterns."""

    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super(MultiScaleTrendMixing, self).__init__()
        self.up_sampling_layers = torch.nn.ModuleList(
            [
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
            ]
        )

    def forward(self, trend_list):
        # Mixing low -> high
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


class PastDecomposableMixing(nn.Module):
    """Past Decomposable Mixing block - core component of TimeMixer."""

    def __init__(self, seq_len, pred_len, down_sampling_window, down_sampling_layers,
                 d_model, d_ff, dropout, decomp_method, moving_avg_kernel, top_k,
                 channel_independence):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.down_sampling_window = down_sampling_window
        self.channel_independence = channel_independence

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if decomp_method == 'moving_avg':
            self.decomposition = series_decomp(moving_avg_kernel)
        elif decomp_method == "dft_decomp":
            self.decomposition = DFT_series_decomp(top_k)
        else:
            raise ValueError(f'Unknown decomposition method: {decomp_method}')

        if channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_ff),
                nn.GELU(),
                nn.Linear(in_features=d_ff, out_features=d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(
            seq_len, down_sampling_window, down_sampling_layers
        )

        # Mixing trend
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
            season, trend = self.decomposition(x)
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


# ==============================================================================
# TimeMixerPP Model for LibCity
# ==============================================================================

class TimeMixerPP(AbstractTrafficStateModel):
    """
    TimeMixer++ adapted for LibCity traffic speed prediction.

    This model uses multi-scale temporal decomposition with MLP-based
    seasonal/trend mixing. It is a purely temporal model that processes
    each node independently (channel_independence=1 by default).

    The input traffic data is reshaped from [B, T, N, C] to [B*N, T, C]
    for processing, then reshaped back for output.
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._logger = getLogger()

        # Data features from LibCity
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)
        self._scaler = data_feature.get('scaler')

        # Sequence configuration
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.seq_len = self.input_window
        self.pred_len = self.output_window

        # Model hyperparameters
        self.d_model = config.get('d_model', 64)
        self.d_ff = config.get('d_ff', 128)
        self.e_layers = config.get('e_layers', 2)
        self.down_sampling_layers = config.get('down_sampling_layers', 2)
        self.down_sampling_window = config.get('down_sampling_window', 2)
        self.down_sampling_method = config.get('down_sampling_method', 'avg')
        self.decomp_method = config.get('decomp_method', 'moving_avg')
        self.moving_avg = config.get('moving_avg', 25)
        self.top_k = config.get('top_k', 5)
        self.channel_independence = config.get('channel_independence', 1)
        self.use_norm = config.get('use_norm', 1)
        self.dropout = config.get('dropout', 0.1)

        self.device = config.get('device', torch.device('cpu'))

        # Input channels: for traffic prediction, we use output_dim (typically speed)
        # For channel_independence=1, each channel is processed separately
        self.enc_in = self.output_dim
        self.c_out = self.output_dim

        self._logger.info(f"TimeMixerPP Config: seq_len={self.seq_len}, pred_len={self.pred_len}, "
                         f"d_model={self.d_model}, e_layers={self.e_layers}, "
                         f"down_sampling_layers={self.down_sampling_layers}")

        # Validate sequence length for downsampling
        min_seq = self.down_sampling_window ** self.down_sampling_layers
        if self.seq_len < min_seq:
            self._logger.warning(f"seq_len ({self.seq_len}) < min required ({min_seq}), "
                               f"reducing down_sampling_layers")
            self.down_sampling_layers = max(0, int(math.log(self.seq_len) / math.log(self.down_sampling_window)))

        # Build model components
        self._build_model()

    def _build_model(self):
        """Build all model components."""

        # Preprocessing decomposition
        self.preprocess = series_decomp(self.moving_avg)

        # Embedding layer
        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, self.d_model, self.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(self.enc_in, self.d_model, self.dropout)

        # PDM blocks (Past Decomposable Mixing)
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                down_sampling_window=self.down_sampling_window,
                down_sampling_layers=self.down_sampling_layers,
                d_model=self.d_model,
                d_ff=self.d_ff,
                dropout=self.dropout,
                decomp_method=self.decomp_method,
                moving_avg_kernel=self.moving_avg,
                top_k=self.top_k,
                channel_independence=self.channel_independence
            )
            for _ in range(self.e_layers)
        ])

        # Normalization layers for each scale
        self.normalize_layers = torch.nn.ModuleList([
            Normalize(
                self.enc_in,
                affine=True,
                non_norm=(self.use_norm == 0)
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

        # Final projection
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
        """
        Perform multi-scale downsampling of input.

        Args:
            x_enc: Input tensor [B, T, C]

        Returns:
            x_enc_list: List of downsampled tensors at different scales
        """
        if self.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(
                in_channels=self.enc_in,
                out_channels=self.enc_in,
                kernel_size=3,
                padding=padding,
                stride=self.down_sampling_window,
                padding_mode='circular',
                bias=False
            ).to(x_enc.device)
        else:
            return [x_enc]

        # B, T, C -> B, C, T
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
        """Preprocessing before encoding."""
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
        """Future multi-predictor mixing."""
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
        Perform forecasting.

        Args:
            x_enc: Input tensor [B, T, C] or [B*N, T, 1] for channel_independence

        Returns:
            dec_out: Output tensor [B, pred_len, C]
        """
        # Multi-scale input processing
        x_enc_list = self._multi_scale_process_inputs(x_enc)

        # Normalize and prepare inputs
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

        # Past Decomposable Mixing (encoder)
        for i in range(self.e_layers):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing (decoder)
        B = x_enc.size(0)
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        # Aggregate predictions from all scales
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')

        return dec_out

    def forward(self, batch):
        """
        Forward pass for LibCity batch format.

        Args:
            batch: Dictionary with keys 'X' and 'y'
                   batch['X'] shape: [batch_size, input_window, num_nodes, feature_dim]

        Returns:
            output: Predictions with shape [batch_size, output_window, num_nodes, output_dim]
        """
        # Get input data: [B, T, N, C]
        x = batch['X']
        batch_size, seq_len, num_nodes, feature_dim = x.shape

        # Use only the output features (typically the first output_dim features)
        x = x[..., :self.output_dim]  # [B, T, N, output_dim]

        # Reshape for TimeMixer: treat each node as a separate batch
        # [B, T, N, C] -> [B*N, T, C]
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, N, T, C]
        x = x.reshape(batch_size * num_nodes, seq_len, self.output_dim)  # [B*N, T, C]

        # Perform forecasting
        output = self.forecast(x)  # [B*N, pred_len, C]

        # Reshape back to LibCity format: [B*N, pred_len, C] -> [B, pred_len, N, C]
        output = output.reshape(batch_size, num_nodes, self.pred_len, self.output_dim)
        output = output.permute(0, 2, 1, 3).contiguous()  # [B, pred_len, N, C]

        return output

    def predict(self, batch):
        """
        Prediction interface for LibCity.

        Args:
            batch: Dictionary with keys 'X' and 'y'

        Returns:
            output: Predictions with shape [batch_size, output_window, num_nodes, output_dim]
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate training loss using LibCity's scaler.

        Args:
            batch: Dictionary with keys 'X' and 'y'
                   batch['y'] shape: [batch_size, output_window, num_nodes, feature_dim]

        Returns:
            loss_value: Scalar loss tensor
        """
        y_true = batch['y']
        y_predicted = self.predict(batch)

        # Apply inverse transform for proper loss calculation
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        return loss.masked_mae_torch(y_predicted, y_true, 0)
