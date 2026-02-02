"""
FCNNBus: Fully Convolutional Neural Network for Bus Arrival Time Prediction

This model is adapted from the original TensorFlow/Keras implementation.

Original Repository: /home/wangwenrui/shk/AgentCity/repos/FCNNBus
Original Notebook: /home/wangwenrui/shk/AgentCity/repos/FCNNBus/Notebooks/Copy_of_CNN_for_bus_sample_2.ipynb

Original Keras Architecture:
    model = Sequential()
    model.add(Conv1D(32, 3, activation='relu', padding='same', input_shape=(4, 1)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 2, activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))  # Output for regression

Key Adaptations:
    1. Converted from TensorFlow/Keras to PyTorch
    2. Adapted to LibCity's AbstractTrafficStateModel interface
    3. Input format transformation: LibCity batch (batch, time, nodes, features) to Conv1D input
    4. Output format: Produces predictions compatible with LibCity's evaluation framework

Limitations:
    - The original model was designed for 4 specific features (timestamp, direction, bus line, stop)
    - The adapted version uses input_window * feature_dim as input channels
    - Loss function: MSE (as in original), but can use LibCity's masked_mse_torch

Required Config Parameters:
    - input_window: Number of input time steps (default: 12)
    - output_window: Number of output time steps (default: 12)
    - conv1_filters: Number of filters in first Conv1D layer (default: 32)
    - conv1_kernel: Kernel size for first Conv1D layer (default: 3)
    - conv2_filters: Number of filters in second Conv1D layer (default: 64)
    - conv2_kernel: Kernel size for second Conv1D layer (default: 2)
    - pool_size: MaxPool1D pool size (default: 2)
    - hidden_size: Size of hidden dense layer (default: 128)
"""

from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class FCNNBus(AbstractTrafficStateModel):
    """
    Fully Convolutional Neural Network for Bus/Traffic Prediction.

    Adapted from the FCNNBus TensorFlow/Keras model for bus arrival time prediction.
    The architecture uses 1D convolutions to capture temporal patterns in traffic data.
    """

    def __init__(self, config, data_feature):
        """
        Initialize the FCNNBus model.

        Args:
            config: Configuration dictionary containing model hyperparameters
            data_feature: Dictionary containing data-related features like scaler, num_nodes, etc.
        """
        super().__init__(config, data_feature)

        # Get data features
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)

        # Logger
        self._logger = getLogger()

        # Device configuration
        self.device = config.get('device', torch.device('cpu'))

        # Model hyperparameters from config
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)

        # CNN architecture parameters (matching original Keras model)
        self.conv1_filters = config.get('conv1_filters', 32)
        self.conv1_kernel = config.get('conv1_kernel', 3)
        self.conv2_filters = config.get('conv2_filters', 64)
        self.conv2_kernel = config.get('conv2_kernel', 2)
        self.pool_size = config.get('pool_size', 2)
        self.hidden_size = config.get('hidden_size', 128)

        # Loss function type
        self.loss_type = config.get('loss_type', 'mse')  # 'mse' or 'mae'

        # Build the model architecture
        # Note: In PyTorch Conv1d, input is (batch, channels, sequence_length)
        # In Keras Conv1D, input is (batch, steps, features)
        # We treat input_window as sequence length and feature_dim as channels

        # First Conv1D layer: input channels = feature_dim, output = conv1_filters
        # padding='same' in Keras means output has same length as input
        self.conv1 = nn.Conv1d(
            in_channels=self.feature_dim,
            out_channels=self.conv1_filters,
            kernel_size=self.conv1_kernel,
            padding='same'  # PyTorch >= 1.9 supports 'same' padding
        )

        # MaxPooling layer
        self.pool = nn.MaxPool1d(kernel_size=self.pool_size)

        # Second Conv1D layer
        self.conv2 = nn.Conv1d(
            in_channels=self.conv1_filters,
            out_channels=self.conv2_filters,
            kernel_size=self.conv2_kernel,
            padding='same'
        )

        # Calculate the flattened size after convolutions and pooling
        # After conv1 with same padding: sequence_length remains input_window
        # After pool: input_window // pool_size
        # After conv2 with same padding: sequence_length remains the same
        pooled_length = self.input_window // self.pool_size
        flatten_size = self.conv2_filters * pooled_length

        # Fully connected layers
        self.fc1 = nn.Linear(flatten_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_window * self.output_dim)

        # ReLU activation
        self.relu = nn.ReLU()

        self._logger.info(f"FCNNBus model initialized with:")
        self._logger.info(f"  - num_nodes: {self.num_nodes}")
        self._logger.info(f"  - feature_dim: {self.feature_dim}")
        self._logger.info(f"  - output_dim: {self.output_dim}")
        self._logger.info(f"  - input_window: {self.input_window}")
        self._logger.info(f"  - output_window: {self.output_window}")
        self._logger.info(f"  - conv1_filters: {self.conv1_filters}")
        self._logger.info(f"  - conv2_filters: {self.conv2_filters}")
        self._logger.info(f"  - hidden_size: {self.hidden_size}")
        self._logger.info(f"  - flatten_size: {flatten_size}")

    def forward(self, batch):
        """
        Forward pass of the FCNNBus model.

        Args:
            batch: Dictionary containing 'X' tensor with shape (batch, time_in, num_nodes, features)

        Returns:
            torch.Tensor: Predictions with shape (batch, time_out, num_nodes, output_dim)
        """
        # Get input tensor
        # LibCity format: (batch, time_in, num_nodes, features)
        x = batch['X']  # (B, T, N, F)

        batch_size = x.shape[0]

        # Reshape for per-node processing
        # (B, T, N, F) -> (B, N, T, F)
        x = x.permute(0, 2, 1, 3)

        # Combine batch and nodes for parallel processing
        # (B, N, T, F) -> (B*N, T, F)
        x = x.reshape(batch_size * self.num_nodes, self.input_window, self.feature_dim)

        # PyTorch Conv1d expects (batch, channels, sequence_length)
        # Current shape: (B*N, T, F) -> need (B*N, F, T)
        x = x.permute(0, 2, 1)  # (B*N, F, T)

        # First convolution block
        x = self.conv1(x)  # (B*N, conv1_filters, T)
        x = self.relu(x)
        x = self.pool(x)   # (B*N, conv1_filters, T//pool_size)

        # Second convolution block
        x = self.conv2(x)  # (B*N, conv2_filters, T//pool_size)
        x = self.relu(x)

        # Flatten for fully connected layers
        x = x.flatten(start_dim=1)  # (B*N, conv2_filters * T//pool_size)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # (B*N, output_window * output_dim)

        # Reshape output to LibCity format
        # (B*N, output_window * output_dim) -> (B, N, output_window, output_dim)
        x = x.reshape(batch_size, self.num_nodes, self.output_window, self.output_dim)

        # Permute to LibCity format: (B, T_out, N, output_dim)
        x = x.permute(0, 2, 1, 3)

        return x

    def predict(self, batch):
        """
        Generate predictions for a batch of input data.

        Args:
            batch: Dictionary containing 'X' tensor

        Returns:
            torch.Tensor: Predictions with shape (batch, time_out, num_nodes, output_dim)
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate the training loss.

        The original FCNNBus model uses MSE loss. This implementation
        supports both MSE and MAE (masked versions from LibCity).

        Args:
            batch: Dictionary containing 'X' and 'y' tensors

        Returns:
            torch.Tensor: Scalar loss value
        """
        y_true = batch['y']  # (B, T_out, N, F)
        y_predicted = self.predict(batch)

        # Inverse transform to original scale for loss calculation
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        # Calculate loss based on configured loss type
        if self.loss_type == 'mse':
            return loss.masked_mse_torch(y_predicted, y_true)
        elif self.loss_type == 'mae':
            return loss.masked_mae_torch(y_predicted, y_true)
        else:
            # Default to MSE as in original model
            return loss.masked_mse_torch(y_predicted, y_true)
