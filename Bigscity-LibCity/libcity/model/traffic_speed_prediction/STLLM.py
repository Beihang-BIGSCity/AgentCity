"""
ST-LLM: Spatial-Temporal Large Language Model for Traffic Prediction

Adapted from: https://github.com/ChenxiLiu-HNU/ST-LLM
Original Paper: "Spatial-Temporal Large Language Model for Traffic Flow Prediction"

This module adapts the ST-LLM model to the LibCity framework conventions.
The model leverages GPT-2 as a backbone with partial fine-tuning (PFA) for
spatial-temporal traffic prediction.

Key Adaptations:
- Inherits from AbstractTrafficStateModel
- Extracts parameters from LibCity's config and data_feature
- Handles LibCity batch format {'X': tensor, 'y': tensor}
- Uses LibCity's scaler for normalization
- Implements required predict() and calculate_loss() methods
"""

from logging import getLogger

import torch
import torch.nn as nn

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

try:
    from transformers.models.gpt2.modeling_gpt2 import GPT2Model
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    GPT2Model = None


class TemporalEmbedding(nn.Module):
    """
    Temporal Embedding module that creates learnable embeddings for
    time-of-day and day-of-week features.

    Args:
        time (int): Number of time slots per day (e.g., 288 for 5-min intervals)
        features (int): Embedding dimension
    """

    def __init__(self, time, features, device='cpu'):
        super(TemporalEmbedding, self).__init__()
        self.time = time
        self.device = device

        # Temporal embeddings for time of day
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        # Temporal embeddings for day of week
        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x, tid_data, diw_data):
        """
        Forward pass for temporal embedding.

        Args:
            x: Input tensor (not used directly, kept for interface compatibility)
            tid_data: Time-in-day data, shape (batch, time, nodes), normalized [0, 1]
            diw_data: Day-in-week data, shape (batch, time, nodes), values 0-6

        Returns:
            Temporal embedding tensor of shape (batch, features, nodes, 1)
        """
        # Get time of day embedding using the last time step
        # tid_data is normalized to [0, 1], multiply by time slots to get index
        time_indices = (tid_data[:, -1, :] * self.time).long().to(self.device)
        time_indices = torch.clamp(time_indices, 0, self.time - 1)
        time_day = self.time_day[time_indices]  # (batch, nodes, features)
        time_day = time_day.transpose(1, 2).unsqueeze(-1)  # (batch, features, nodes, 1)

        # Get day of week embedding using the last time step
        week_indices = diw_data[:, -1, :].long().to(self.device)
        week_indices = torch.clamp(week_indices, 0, 6)
        time_week = self.time_week[week_indices]  # (batch, nodes, features)
        time_week = time_week.transpose(1, 2).unsqueeze(-1)  # (batch, features, nodes, 1)

        # Combine temporal embeddings
        tem_emb = time_day + time_week
        return tem_emb


class PFA(nn.Module):
    """
    Partial Fine-tuning Approach (PFA) for GPT-2.

    This module loads a pre-trained GPT-2 model and applies partial fine-tuning:
    - Earlier layers: Only layer normalization and position embeddings are trainable
    - Later layers (last U layers): Attention layers are trainable, MLP is frozen

    Args:
        device (str): Device to place the model on
        gpt_layers (int): Number of GPT-2 layers to use
        U (int): Number of layers at the end with more trainable parameters
    """

    def __init__(self, device="cuda:0", gpt_layers=6, U=1):
        super(PFA, self).__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library is required for ST-LLM. "
                "Install with: pip install transformers>=4.36.2"
            )

        self.gpt2 = GPT2Model.from_pretrained(
            "gpt2", output_attentions=True, output_hidden_states=True
        )
        # Use only the first gpt_layers layers
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.U = U

        # Apply partial fine-tuning strategy
        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                if layer_index < gpt_layers - self.U:
                    # Earlier layers: only ln and wpe are trainable
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    # Later layers: mlp is frozen, attention is trainable
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

    def forward(self, x):
        """
        Forward pass through GPT-2.

        Args:
            x: Input embeddings of shape (batch, seq_len, hidden_dim)

        Returns:
            Last hidden state of shape (batch, seq_len, hidden_dim)
        """
        return self.gpt2(inputs_embeds=x).last_hidden_state


class STLLM(AbstractTrafficStateModel):
    """
    ST-LLM: Spatial-Temporal Large Language Model for Traffic Prediction

    This model combines:
    1. Temporal embeddings (time-of-day and day-of-week)
    2. Spatial node embeddings
    3. Input feature convolution
    4. Feature fusion
    5. GPT-2 backbone with partial fine-tuning
    6. Regression layer for prediction

    LibCity Config Parameters:
        input_window (int): Number of input time steps (default: 12)
        output_window (int): Number of output time steps (default: 12)
        llm_layer (int): Number of GPT-2 layers to use (default: 6)
        pfa_U (int): Number of layers with more trainable params (default: 1)
        gpt_channel (int): Intermediate channel dimension (default: 256)
        time_intervals (int): Seconds per time interval (default: 300 for 5-min)
        device: Computation device

    Data Features:
        num_nodes (int): Number of nodes in the traffic network
        feature_dim (int): Input feature dimension
        output_dim (int): Output feature dimension
        scaler: Data scaler for normalization
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._logger = getLogger()

        # Extract data features
        self.num_nodes = data_feature.get('num_nodes')
        self.feature_dim = data_feature.get('feature_dim', 3)
        self.output_dim = data_feature.get('output_dim', 1)
        self._scaler = data_feature.get('scaler')

        # Extract config parameters
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.llm_layer = config.get('llm_layer', 6)
        self.pfa_U = config.get('pfa_U', 1)
        self.gpt_channel = config.get('gpt_channel', 256)
        self.time_intervals = config.get('time_intervals', 300)
        self.device = config.get('device', torch.device('cpu'))

        # Calculate time slots per day based on time_intervals
        # For 5-minute intervals: 288 slots, for 30-minute: 48 slots
        assert (24 * 60 * 60) % self.time_intervals == 0, \
            "time_intervals must evenly divide seconds in a day"
        self.time_of_day_size = int((24 * 60 * 60) / self.time_intervals)

        # Model dimensions
        to_gpt_channel = 768  # GPT-2 hidden dimension

        # Temporal embedding layer
        self.Temb = TemporalEmbedding(
            self.time_of_day_size,
            self.gpt_channel,
            device=self.device
        )

        # Spatial node embedding
        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.gpt_channel))
        nn.init.xavier_uniform_(self.node_emb)

        # Input convolution: flatten input along time and feature dimensions
        # Input shape after permute: (batch, input_dim * input_window, nodes, 1)
        self.start_conv = nn.Conv2d(
            self.output_dim * self.input_window,
            self.gpt_channel,
            kernel_size=(1, 1)
        )

        # GPT-2 backbone with partial fine-tuning
        self.gpt = PFA(
            device=str(self.device),
            gpt_layers=self.llm_layer,
            U=self.pfa_U
        )

        # Feature fusion: combine input, temporal, and spatial embeddings
        self.feature_fusion = nn.Conv2d(
            self.gpt_channel * 3,
            to_gpt_channel,
            kernel_size=(1, 1)
        )

        # Regression layer for final prediction
        self.regression_layer = nn.Conv2d(
            to_gpt_channel,  # Changed from gpt_channel * 3 to match GPT output
            self.output_window * self.output_dim,  # Output both time steps and features
            kernel_size=(1, 1)
        )

        self._logger.info(
            f"STLLM initialized: num_nodes={self.num_nodes}, "
            f"input_window={self.input_window}, output_window={self.output_window}, "
            f"llm_layer={self.llm_layer}, time_of_day_size={self.time_of_day_size}"
        )

    def forward(self, batch):
        """
        Forward pass of ST-LLM.

        Args:
            batch (dict): LibCity batch containing:
                - 'X': Input tensor of shape (batch, input_window, num_nodes, feature_dim)
                       where features typically include [value, time_of_day, day_of_week, ...]
                - 'y': Target tensor (not used in forward, used in calculate_loss)

        Returns:
            torch.Tensor: Predictions of shape (batch, output_window, num_nodes, output_dim)
        """
        # Extract input data: (batch, time, nodes, features)
        input_data = batch['X']
        batch_size, seq_len, num_nodes, feat_dim = input_data.shape

        # Extract temporal features
        # Assuming feature order: [value, time_of_day, day_of_week, ...]
        # time_of_day is normalized to [0, 1], day_of_week is 0-6
        if feat_dim >= 3:
            tid_data = input_data[..., 1]  # (batch, time, nodes)
            # Handle day_of_week: could be one-hot encoded or integer
            if feat_dim > 9:  # Likely one-hot encoded (7 days)
                diw_data = torch.argmax(input_data[..., 2:9], dim=-1)  # (batch, time, nodes)
            else:
                diw_data = input_data[..., 2]  # (batch, time, nodes)
        elif feat_dim >= 2:
            tid_data = input_data[..., 1]
            diw_data = torch.zeros_like(tid_data)
        else:
            # No temporal features, use zeros
            tid_data = torch.zeros(batch_size, seq_len, num_nodes, device=input_data.device)
            diw_data = torch.zeros(batch_size, seq_len, num_nodes, device=input_data.device)

        # Get temporal embedding
        tem_emb = self.Temb(input_data, tid_data, diw_data)  # (batch, gpt_channel, nodes, 1)

        # Get spatial node embedding
        node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, nodes, gpt_channel)
        node_emb = node_emb.transpose(1, 2).unsqueeze(-1)  # (batch, gpt_channel, nodes, 1)

        # Process input time series
        # Use only the traffic value (first feature) for the main input
        time_series = input_data[..., :self.output_dim]  # (batch, time, nodes, output_dim)
        time_series = time_series.transpose(1, 2).contiguous()  # (batch, nodes, time, output_dim)
        time_series = time_series.reshape(batch_size, num_nodes, -1)  # (batch, nodes, time * output_dim)
        time_series = time_series.transpose(1, 2).unsqueeze(-1)  # (batch, time * output_dim, nodes, 1)

        # Input convolution
        input_emb = self.start_conv(time_series)  # (batch, gpt_channel, nodes, 1)

        # Concatenate all embeddings
        data_st = torch.cat([input_emb, tem_emb, node_emb], dim=1)  # (batch, gpt_channel * 3, nodes, 1)

        # Feature fusion
        data_st = self.feature_fusion(data_st)  # (batch, 768, nodes, 1)

        # Reshape for GPT-2: (batch, nodes, 768)
        data_st = data_st.squeeze(-1).permute(0, 2, 1)  # (batch, nodes, 768)

        # Pass through GPT-2
        data_st = self.gpt(data_st)  # (batch, nodes, 768)

        # Reshape back: (batch, 768, nodes, 1)
        data_st = data_st.permute(0, 2, 1).unsqueeze(-1)

        # Regression to get predictions
        prediction = self.regression_layer(data_st)  # (batch, output_window * output_dim, nodes, 1)
        prediction = prediction.squeeze(-1)  # (batch, output_window * output_dim, nodes)
        prediction = prediction.reshape(batch_size, self.output_window, self.output_dim, num_nodes)
        prediction = prediction.permute(0, 1, 3, 2)  # (batch, output_window, nodes, output_dim)

        return prediction

    def predict(self, batch):
        """
        Prediction wrapper that calls forward.

        Args:
            batch (dict): LibCity batch

        Returns:
            torch.Tensor: Predictions of shape (batch, output_window, num_nodes, output_dim)
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate the training loss using masked MAE.

        Args:
            batch (dict): LibCity batch containing 'X' and 'y'

        Returns:
            torch.Tensor: Scalar loss value
        """
        y_true = batch['y']  # (batch, output_window, num_nodes, output_dim)
        y_predicted = self.predict(batch)

        # Apply inverse transform to get real values
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        # Use masked MAE loss (mask out zero values)
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def param_num(self):
        """
        Get the total number of parameters in the model.

        Returns:
            int: Total parameter count
        """
        return sum([param.nelement() for param in self.parameters()])
