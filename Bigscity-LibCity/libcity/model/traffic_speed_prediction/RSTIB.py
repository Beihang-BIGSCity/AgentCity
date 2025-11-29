from logging import getLogger
from numbers import Number

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
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N, 1]

        Returns:
            torch.Tensor: latent representation with residual connection
        """
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))  # MLP
        hidden = hidden + input_data  # residual
        return hidden


class RSTIB(AbstractTrafficStateModel):
    """
    Paper: Information Bottleneck-guided MLPs for Robust Spatial-temporal Forecasting
    Link: ICML 2025
    Repo: https://github.com/mchen644/RSTIB

    Robust Spatial-Temporal Information Bottleneck (RSTIB) model using MLPs
    for efficient and robust spatial-temporal forecasting.
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        # Data feature extraction
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)
        self._scaler = self.data_feature.get('scaler')
        self._logger = getLogger()

        # Model configuration
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.device = config.get('device', torch.device('cpu'))

        # RSTIB-specific parameters
        self.input_dim = config.get('input_dim', 1)  # Traffic feature dimension (e.g., speed)
        self.embed_dim = config.get('embed_dim', 64)
        self.num_layer = config.get('num_layer', 3)
        self.node_dim = config.get('node_dim', 64)
        self.temp_dim_tid = config.get('temp_dim_tid', 64)
        self.temp_dim_diw = config.get('temp_dim_diw', 64)

        # Feature flags
        self.if_spatial = config.get('if_spatial', True)
        self.if_time_in_day = config.get('if_time_in_day', True)
        self.if_day_in_week = config.get('if_day_in_week', True)
        self.if_dne = config.get('if_dne', False)  # Dynamic Node Embeddings

        # Information Bottleneck parameters
        self.info_beta = config.get('info_beta', 0.001)
        self.n_sample = config.get('n_sample', 1)  # Number of sampling for reparameterization

        # Time embeddings calculation
        time_intervals = config.get('time_intervals', 300)  # 5 minutes = 300 seconds
        assert (24 * 60 * 60) % time_intervals == 0, "time_of_day_size should be Int"
        self.time_of_day_size = int((24 * 60 * 60) / time_intervals)
        self.day_of_week_size = 7

        # Spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)

        # Temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # Embedding layer for time series
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_window,
            out_channels=self.embed_dim,
            kernel_size=(1, 1),
            bias=True)

        # Calculate hidden dimensions
        hidden_dims = []
        hidden_dims.append(self.embed_dim +
                          self.temp_dim_tid * int(self.if_time_in_day) +
                          self.temp_dim_diw * int(self.if_day_in_week))
        hidden_dims.append(self.node_dim * int(self.if_spatial))
        hidden_dims.append(self.node_dim * int(self.if_dne))

        self.hidden_dim = sum(hidden_dims)
        self.K = int(self.hidden_dim // 2)  # Information Bottleneck dimension

        # Encoder layers
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # Information Bottleneck layers
        if self.if_dne is False:
            self.reduce_dimen_mu = nn.Conv2d(self.K, self.output_window, 1)
            self.reduce_dimen_std = nn.Conv2d(self.K, self.output_window, 1)
            self.increase_dimen_mu = nn.Conv2d(in_channels=self.K, out_channels=self.hidden_dim, kernel_size=1)
            self.increase_dimen_std = nn.Conv2d(in_channels=self.K, out_channels=self.hidden_dim, kernel_size=1)
        else:
            # Adjust dimensions if using dynamic node embeddings
            adjusted_k = self.K
            self.reduce_dimen_mu = nn.Conv2d(adjusted_k, self.output_window, 1)
            self.reduce_dimen_std = nn.Conv2d(adjusted_k, self.output_window, 1)
            self.increase_dimen_mu = nn.Conv2d(in_channels=adjusted_k, out_channels=self.hidden_dim, kernel_size=1)
            self.increase_dimen_std = nn.Conv2d(in_channels=adjusted_k, out_channels=self.hidden_dim, kernel_size=1)

        # Regression layer (output prediction)
        self.regression_layer = nn.Conv2d(
            in_channels=self.K,
            out_channels=self.output_window * self.output_dim,
            kernel_size=(1, 1),
            bias=True)

        # Dynamic Node Embeddings (DNE)
        if self.if_dne:
            mid_dim = config.get('mid_dim', 64)
            dne_act_name = config.get('dne_act', 'softmax')
            self.nodevec_p1 = nn.Parameter(
                torch.randn(self.time_of_day_size, mid_dim).to(self.device),
                requires_grad=True)
            self.nodevec_p2 = nn.Parameter(
                torch.randn(self.num_nodes, mid_dim).to(self.device),
                requires_grad=True)
            self.nodevec_pk = nn.Parameter(
                torch.randn(mid_dim, mid_dim, self.node_dim).to(self.device),
                requires_grad=True)
            self.dne_emb_layer = nn.Conv2d(
                in_channels=self.input_window,
                out_channels=1,
                kernel_size=(1, 1),
                bias=True)

            self.dne_act = {
                'softplus': F.softplus,
                'leakyrelu': nn.LeakyReLU(negative_slope=0.01, inplace=False),
                'relu': torch.nn.ReLU(inplace=False),
                'sigmoid': nn.Sigmoid(),
                'softmax': nn.Softmax(dim=2),
                'none': none_act
            }[dne_act_name]

    def construct_dne(self, te):
        """
        Construct Dynamic Node Embeddings based on time encoding.

        Args:
            te: Time encoding tensor of shape [B, T]

        Returns:
            torch.Tensor: Dynamic node embeddings [B, D, N, 1]
        """
        assert len(te.shape) == 2, "'te' should be (B, T)"
        dne = torch.einsum('bai, ijk->bajk', self.nodevec_p1[te], self.nodevec_pk)
        dne = torch.einsum('bj, cajk->cabk', self.nodevec_p2, dne)
        # B, T, N, D
        dne = self.dne_emb_layer(dne).transpose(3, 1)  # B, D, N, 1
        dne = self.dne_act(dne)
        return dne

    def reparametrize_n(self, mu, std, n=1):
        """
        Reparameterization trick for variational inference.

        Args:
            mu: Mean of the distribution
            std: Standard deviation of the distribution
            n: Number of samples

        Returns:
            torch.Tensor: Sampled latent representation
        """
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1:
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda)).to(self.device)
        return mu + eps * std

    def forward(self, batch):
        """
        Forward pass of RSTIB model.

        Args:
            batch: Dictionary containing input data with keys:
                - 'X': Input tensor [B, L, N, C] where C includes [traffic_feature, time_in_day, day_in_week]

        Returns:
            torch.Tensor: Predictions [B, T, N, output_dim]
        """
        history_data = batch['X']  # [B, L, N, C]

        # Extract traffic features (first input_dim features)
        X = history_data[..., :self.input_dim]  # [B, L, N, input_dim]

        # Extract temporal features
        if self.if_time_in_day and history_data.shape[-1] > self.input_dim:
            t_i_d_data = history_data[..., self.input_dim]  # Time in day (normalized 0-1)
            # Clamp to valid range and convert to indices
            time_indices = (t_i_d_data[:, -1, :] * self.time_of_day_size).clamp(0, self.time_of_day_size - 1).long()
            T_i_D_emb = self.time_in_day_emb[time_indices]
        else:
            T_i_D_emb = None

        if self.if_day_in_week and history_data.shape[-1] > self.input_dim + 1:
            d_i_w_data = history_data[..., self.input_dim + 1]  # Day in week (0-6 or normalized)
            # Clamp to valid range
            day_indices = d_i_w_data[:, -1, :].clamp(0, self.day_of_week_size - 1).long()
            D_i_W_emb = self.day_in_week_emb[day_indices]
        else:
            D_i_W_emb = None

        B, L, N, _ = X.shape

        # Reshape for Conv2d: [B, N, L*input_dim] -> [B, L*input_dim, N, 1]
        X = X.transpose(1, 2).contiguous()  # [B, N, L, input_dim]
        X = X.view(B, N, -1).transpose(1, 2).unsqueeze(-1)  # [B, L*input_dim, N, 1]

        # Time series embedding
        time_series_emb = self.time_series_emb_layer(X)  # [B, embed_dim, N, 1]

        # Node embeddings
        node_emb = []
        if self.if_spatial:
            node_emb.append(
                self.node_emb.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1))  # [B, node_dim, N, 1]

        if self.if_dne and 'te' in batch:
            dne = self.construct_dne(batch['te'].type(torch.LongTensor))
            node_emb.append(dne)

        # Temporal embeddings
        tem_emb = []
        if T_i_D_emb is not None:
            tem_emb.append(T_i_D_emb.transpose(1, 2).unsqueeze(-1))  # [B, temp_dim_tid, N, 1]
        if D_i_W_emb is not None:
            tem_emb.append(D_i_W_emb.transpose(1, 2).unsqueeze(-1))  # [B, temp_dim_diw, N, 1]

        # Concatenate all embeddings
        hidden_input = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)  # [B, hidden_dim, N, 1]

        # Encoding with Information Bottleneck
        hidden = self.encoder(hidden_input)  # [B, hidden_dim, N, 1]

        # Split into mean and std for variational inference
        mu = hidden[:, :self.K, :, :]
        std = F.softplus(hidden[:, self.K:, :, :])
        std = std.clamp(1e-2, 1 - 1e-2)

        # Information bottleneck transformations
        mu_y = self.reduce_dimen_mu(mu)
        std_y = self.reduce_dimen_std(std)
        std_y = std_y.clamp(1e-2, 1 - 1e-2)

        mu_x = self.increase_dimen_mu(mu)
        std_x = self.increase_dimen_std(std)
        std_x = std_x.clamp(1e-2, 1 - 1e-2)

        # Reparameterization
        noise_y = self.reparametrize_n(mu_y, std_y, n=self.n_sample)
        noise_x = self.reparametrize_n(mu_x, std_x, n=self.n_sample)

        if self.n_sample == 1:
            pass
        elif self.n_sample > 1:
            noise_x = noise_x.mean(0)
            noise_y = noise_y.mean(0)

        # Add noise to input and re-encode
        hidden_input_clean = hidden_input + noise_x
        hidden_clean = self.encoder(hidden_input_clean)

        mu_clean = hidden_clean[:, :self.K, :, :]
        std_clean = F.softplus(hidden_clean[:, self.K:, :, :])
        std_clean = std_clean.clamp(1e-2, 1 - 1e-2)

        # Final encoding
        encoding = self.reparametrize_n(mu_clean, std_clean, n=self.n_sample)

        if self.n_sample == 1:
            pass
        elif self.n_sample > 1:
            encoding = encoding.mean(0)

        # Prediction
        prediction = self.regression_layer(encoding)  # [B, output_window*output_dim, N, 1]

        # Reshape to [B, output_window, N, output_dim]
        prediction = prediction.squeeze(-1).transpose(1, 2)  # [B, N, output_window*output_dim]
        prediction = prediction.view(B, N, self.output_window, self.output_dim)  # [B, N, output_window, output_dim]
        prediction = prediction.transpose(1, 2)  # [B, output_window, N, output_dim]

        return prediction

    def calculate_loss(self, batch):
        """
        Calculate loss for the model.

        Args:
            batch: Dictionary containing 'X' and 'y'

        Returns:
            torch.Tensor: Loss value
        """
        y_true = batch['y']  # [B, output_window, N, output_dim]
        y_predicted = self.predict(batch)  # [B, output_window, N, output_dim]

        # Inverse transform predictions
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        """
        Multi-step prediction.

        Args:
            batch: Dictionary containing input data

        Returns:
            torch.Tensor: Predictions [B, output_window, N, output_dim]
        """
        return self.forward(batch)
