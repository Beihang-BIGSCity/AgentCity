"""
EasyST: An Easy and Efficient Spatio-Temporal Learning Framework for Traffic Forecasting

This is a LibCity adaptation of the DMLP_Stu_IB (Dynamic MLP with Information Bottleneck) model.

Original Paper: EasyST: A Simple Framework for Spatio-Temporal Prediction
Original Repository: https://github.com/HKUDS/EasyST

Key Components:
- Information Bottleneck (IB): Variational IB with reparameterization (z = mu + eps * sigma)
- Dynamic Node Encoding (DNE): Time-varying node representations via temporal indices
- Temporal Embeddings: Time-of-day and day-of-week embeddings
- Spatial Embeddings: Learnable node embeddings

Adaptation Notes:
- Converted from standalone script to LibCity AbstractTrafficStateModel
- MultiLayerPerceptron (MLP) module included inline (was missing dependency)
- Temporal features extracted from LibCity batch format
- IB loss incorporated in calculate_loss method
- n_sample parameter for stochastic inference during prediction

Configuration Parameters:
- input_window: Input sequence length (lag)
- output_window: Output sequence length (horizon)
- embed_dim: Time series embedding dimension
- node_dim: Spatial embedding dimension
- temp_dim_tid: Time-of-day embedding dimension
- temp_dim_diw: Day-of-week embedding dimension
- num_layer: Number of MLP encoder layers
- if_T_i_D: Whether to use time-of-day embeddings
- if_D_i_W: Whether to use day-of-week embeddings
- if_spatial: Whether to use spatial embeddings
- if_dne: Whether to use Dynamic Node Encoding
- mid_dim: DNE intermediate dimension
- dne_act: DNE activation function (softplus, relu, leakyrelu, sigmoid, softmax, none)
- beta_ib: IB loss weight
- n_sample: Number of samples for stochastic inference
"""

from logging import getLogger
from numbers import Number

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def none_act(inputs):
    """Identity activation function."""
    return inputs


def cuda(tensor, is_cuda):
    """Move tensor to CUDA if needed."""
    if is_cuda:
        return tensor.cuda()
    else:
        return tensor


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual connection.

    This is a simple MLP block used in the encoder.
    Uses Conv2d for efficient batched computation over nodes.
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.15):
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            input_data: Input tensor with shape [B, D, N, 1]

        Returns:
            Output tensor with shape [B, D, N, 1]
        """
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        hidden = hidden + input_data  # residual connection
        return hidden


class EasyST(AbstractTrafficStateModel):
    """
    EasyST: Easy and Efficient Spatio-Temporal Learning Framework

    This model implements a Dynamic MLP with Information Bottleneck (DMLP-IB)
    for traffic state prediction. The key features include:

    1. Time Series Embedding: Embeds input sequences using Conv2d
    2. Spatial Embedding: Learnable node embeddings
    3. Temporal Embedding: Time-of-day and day-of-week embeddings
    4. Dynamic Node Encoding (DNE): Time-varying node representations
    5. Information Bottleneck: Variational IB for robust representations
    6. Stochastic Inference: Multiple samples for prediction averaging
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        # Data features from LibCity
        self.num_nodes = data_feature.get('num_nodes')
        self._scaler = data_feature.get('scaler')
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)

        # Window sizes
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)

        # Embedding dimensions
        self.embed_dim = config.get('embed_dim', 32)
        self.node_dim = config.get('node_dim', 32)
        self.temp_dim_tid = config.get('temp_dim_tid', 32)
        self.temp_dim_diw = config.get('temp_dim_diw', 32)

        # Feature flags
        self.if_time_in_day = config.get('if_T_i_D', True)
        self.if_day_in_week = config.get('if_D_i_W', True)
        self.if_spatial = config.get('if_spatial', True)
        self.if_dne = config.get('if_dne', True)

        # Model configuration
        self.num_layer = config.get('num_layer', 3)
        self.dropout = config.get('dropout', 0.15)
        self.mid_dim = config.get('mid_dim', 32)
        self.dne_act_name = config.get('dne_act', 'softplus')

        # IB configuration
        self.beta_ib = config.get('beta_ib', 0.001)
        self.n_sample_train = config.get('n_sample_train', 1)
        self.n_sample_predict = config.get('n_sample_predict', 12)

        # Time intervals for computing time-of-day size
        self.time_intervals = config.get('time_intervals', 300)  # default 5 minutes

        # Compute time-of-day and day-of-week sizes
        assert (24 * 60 * 60) % self.time_intervals == 0, "time_of_day_size should be Int"
        self.time_of_day_size = int((24 * 60 * 60) / self.time_intervals)
        self.day_of_week_size = 7

        # Device
        self.device = config.get('device', torch.device('cpu'))

        # Logger
        self._logger = getLogger()

        # Initialize spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)

        # Initialize temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid)
            )
            nn.init.xavier_uniform_(self.time_in_day_emb)

        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw)
            )
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # Time series embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.output_dim * self.input_window,
            out_channels=self.embed_dim,
            kernel_size=(1, 1),
            bias=True
        )

        # Compute hidden dimension based on enabled features
        hidden_dims = []
        hidden_dims.append(self.embed_dim)
        if self.if_time_in_day:
            hidden_dims.append(self.temp_dim_tid)
        if self.if_day_in_week:
            hidden_dims.append(self.temp_dim_diw)
        if self.if_spatial:
            hidden_dims.append(self.node_dim)
        if self.if_dne:
            hidden_dims.append(self.node_dim)

        self.hidden_dim = sum(hidden_dims)

        # IB latent dimension (half of hidden dim)
        self.K = int(self.hidden_dim // 2)

        # MLP Encoder
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim, self.dropout)
              for _ in range(self.num_layer)]
        )

        # Regression layer
        self.regression_layer = nn.Conv2d(
            in_channels=self.K,
            out_channels=self.output_window,
            kernel_size=(1, 1),
            bias=True
        )

        # Dynamic Node Encoding (DNE) parameters
        if self.if_dne:
            self.nodevec_p1 = nn.Parameter(
                torch.randn(self.time_of_day_size, self.mid_dim)
            )
            self.nodevec_p2 = nn.Parameter(
                torch.randn(self.num_nodes, self.mid_dim)
            )
            self.nodevec_pk = nn.Parameter(
                torch.randn(self.mid_dim, self.mid_dim, self.node_dim)
            )
            self.dne_emb_layer = nn.Conv2d(
                in_channels=self.input_window,
                out_channels=1,
                kernel_size=(1, 1),
                bias=True
            )

            # DNE activation function
            self.dne_act = {
                'softplus': F.softplus,
                'leakyrelu': nn.LeakyReLU(negative_slope=0.01, inplace=False),
                'relu': torch.nn.ReLU(inplace=False),
                'sigmoid': nn.Sigmoid(),
                'softmax': nn.Softmax(dim=2),
                'none': none_act
            }[self.dne_act_name]

        self._logger.info(f'EasyST initialized with hidden_dim={self.hidden_dim}, K={self.K}')

    def construct_dne(self, te):
        """Construct Dynamic Node Encoding from temporal indices.

        Args:
            te: Temporal indices with shape [B, T] (time-of-day indices)

        Returns:
            DNE tensor with shape [B, D, N, 1]
        """
        assert len(te.shape) == 2, "'te' should be (B, T)"

        # Clamp indices to valid range
        te = te.clamp(0, self.time_of_day_size - 1)

        # Compute DNE via einsum operations
        # nodevec_p1[te]: [B, T, mid_dim]
        # nodevec_pk: [mid_dim, mid_dim, node_dim]
        dne = torch.einsum('bai,ijk->bajk', self.nodevec_p1[te], self.nodevec_pk)
        # dne: [B, T, mid_dim, node_dim]

        # nodevec_p2: [N, mid_dim]
        dne = torch.einsum('bj,cajk->cabk', self.nodevec_p2, dne)
        # dne: [B, T, N, node_dim]

        # Aggregate over time dimension
        dne = self.dne_emb_layer(dne).transpose(3, 1)  # [B, node_dim, N, 1]

        dne = self.dne_act(dne)

        return dne

    def reparametrize_n(self, mu, std, n=1):
        """Reparameterization trick for variational inference.

        Args:
            mu: Mean tensor
            std: Standard deviation tensor
            n: Number of samples

        Returns:
            Sampled latent representation
        """
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1:
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std

    def _extract_temporal_indices(self, batch):
        """Extract temporal indices from LibCity batch format.

        LibCity batch format:
        - X shape: [B, T, N, C] where C includes time features
        - Feature channels typically: [data, time_in_day, day_of_week_onehot...]

        Returns:
            t_i_d_data: Time-of-day data [B, T, N]
            d_i_w_data: Day-of-week data [B, T, N]
            te: Temporal embedding indices [B, T]
        """
        X = batch['X']  # [B, T, N, C]

        # Check feature dimension
        if X.shape[-1] >= 2:
            # Time-in-day is typically in channel 1 (normalized to [0, 1])
            t_i_d_data = X[..., 1]  # [B, T, N]

            # Convert normalized time to discrete index
            t_i_d_idx = (t_i_d_data * self.time_of_day_size).long()
            t_i_d_idx = t_i_d_idx.clamp(0, self.time_of_day_size - 1)
        else:
            t_i_d_idx = torch.zeros(X.shape[0], X.shape[1], X.shape[2],
                                     dtype=torch.long, device=X.device)

        if X.shape[-1] >= 9:
            # Day-of-week is typically one-hot encoded in channels 2-8
            d_i_w_data = torch.argmax(X[..., 2:9], dim=-1)  # [B, T, N]
        elif X.shape[-1] >= 3:
            # Fallback: use channel 2 if exists
            d_i_w_data = (X[..., 2] * 7).long().clamp(0, 6)
        else:
            d_i_w_data = torch.zeros(X.shape[0], X.shape[1], X.shape[2],
                                     dtype=torch.long, device=X.device)

        # For DNE: use time indices from first node (they should be same across nodes)
        te = t_i_d_idx[:, :, 0]  # [B, T]

        return t_i_d_idx, d_i_w_data, te

    def forward(self, batch, n_sample=1):
        """Forward pass.

        Args:
            batch: LibCity batch dict with 'X' key
            n_sample: Number of samples for stochastic inference

        Returns:
            prediction: Predicted values [B, T_out, N, 1]
            hidden_out: Latent representation
            ib_params: (mu, std) for IB loss computation
        """
        X = batch['X']  # [B, T, N, C]

        # Extract only the traffic data (first channel)
        traffic_data = X[..., :self.output_dim]  # [B, T, N, output_dim]

        # Extract temporal indices
        t_i_d_idx, d_i_w_data, te = self._extract_temporal_indices(batch)

        # Get embeddings for last time step
        B, L, N, _ = traffic_data.shape

        # Time-of-day embedding
        if self.if_time_in_day:
            # Use last time step indices
            tid_last = t_i_d_idx[:, -1, :].to(self.device)  # [B, N]
            T_i_D_emb = self.time_in_day_emb[tid_last]  # [B, N, temp_dim_tid]
        else:
            T_i_D_emb = None

        # Day-of-week embedding
        if self.if_day_in_week:
            diw_last = d_i_w_data[:, -1, :].to(self.device)  # [B, N]
            D_i_W_emb = self.day_in_week_emb[diw_last]  # [B, N, temp_dim_diw]
        else:
            D_i_W_emb = None

        # Time series embedding
        # Reshape: [B, T, N, C] -> [B, N, T*C] -> [B, T*C, N, 1]
        time_series = traffic_data.transpose(1, 2).contiguous()  # [B, N, T, C]
        time_series = time_series.view(B, N, -1)  # [B, N, T*C]
        time_series = time_series.transpose(1, 2).unsqueeze(-1)  # [B, T*C, N, 1]
        time_series_emb = self.time_series_emb_layer(time_series)  # [B, embed_dim, N, 1]

        # Collect node embeddings
        node_emb = []
        if self.if_spatial:
            # Expand node embedding for batch
            node_emb.append(
                self.node_emb.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1)
            )  # [B, node_dim, N, 1]

        if self.if_dne:
            te = te.to(self.device)
            dne = self.construct_dne(te)  # [B, node_dim, N, 1]
            node_emb.append(dne)

        # Collect temporal embeddings
        tem_emb = []
        if T_i_D_emb is not None:
            tem_emb.append(T_i_D_emb.transpose(1, 2).unsqueeze(-1))  # [B, temp_dim_tid, N, 1]
        if D_i_W_emb is not None:
            tem_emb.append(D_i_W_emb.transpose(1, 2).unsqueeze(-1))  # [B, temp_dim_diw, N, 1]

        # Concatenate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)  # [B, hidden_dim, N, 1]

        # Encode
        hidden = self.encoder(hidden)  # [B, hidden_dim, N, 1]

        # Information Bottleneck: split into mu and std
        mu = hidden[:, :self.K, :, :]  # [B, K, N, 1]
        std = F.softplus(hidden[:, self.K:, :, :])  # [B, K, N, 1]

        # Reparameterization
        encoding = self.reparametrize_n(mu, std, n=n_sample)

        if n_sample == 1:
            pass
        elif n_sample > 1:
            # Average over samples
            encoding = encoding.mean(0)

        hidden_out = encoding

        # Regression
        prediction = self.regression_layer(encoding)  # [B, T_out, N, 1]

        return prediction, hidden_out, (mu, std)

    def predict(self, batch):
        """Prediction method for inference.

        Uses multiple samples and averages for robust prediction.

        Args:
            batch: LibCity batch dict

        Returns:
            Predicted values [B, T_out, N, 1]
        """
        prediction, _, _ = self.forward(batch, n_sample=self.n_sample_predict)
        return prediction

    def calculate_loss(self, batch):
        """Calculate training loss including IB regularization.

        The loss consists of:
        1. Prediction loss (MAE)
        2. Information Bottleneck loss (KL divergence from prior)

        Args:
            batch: LibCity batch dict with 'X' and 'y'

        Returns:
            Total loss tensor
        """
        y_true = batch['y']  # [B, T_out, N, C]

        # Forward pass with single sample during training
        prediction, _, (mu, std) = self.forward(batch, n_sample=self.n_sample_train)

        # Inverse transform for loss computation
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(prediction[..., :self.output_dim])

        # Prediction loss (masked MAE)
        pred_loss = loss.masked_mae_torch(y_predicted, y_true, 0)

        # Information Bottleneck loss (KL divergence)
        # KL(q(z|x) || p(z)) where p(z) = N(0, 1)
        ib_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean()
        ib_loss = ib_loss / math.log(2)  # Convert to bits

        # Total loss
        total_loss = pred_loss + self.beta_ib * ib_loss

        return total_loss
