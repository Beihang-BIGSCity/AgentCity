"""
Gridded Transformer Neural Processes for Traffic Speed Prediction
Adapted from: https://github.com/cambridge-mlg/gridded-tnp
Paper: "Gridded Transformer Neural Processes for Spatio-Temporal Data" (ICML 2025)
"""

import torch
import torch.nn as nn
import numpy as np
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


class GriddedTNP(AbstractTrafficStateModel):
    """
    Gridded Transformer Neural Process model adapted for LibCity traffic speed prediction.

    This model uses transformer-based neural processes with grid encoders to handle
    spatio-temporal data for traffic prediction tasks.
    """

    def __init__(self, config, data_feature):
        """
        Initialize the GriddedTNP model.

        Args:
            config (dict): Configuration dictionary containing model parameters
            data_feature (dict): Dataset features including:
                - num_nodes: Number of traffic sensors/nodes
                - feature_dim: Input feature dimension
                - output_dim: Output feature dimension
                - adj_mx: Adjacency matrix (optional)
                - scaler: Data scaler object
        """
        super().__init__(config, data_feature)

        # Section 1: Extract data features
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self._logger = getLogger()

        # Section 2: Model configuration parameters
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.device = config.get('device', torch.device('cpu'))

        # GriddedTNP specific parameters
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_heads = config.get('num_heads', 4)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.1)
        self.grid_size = config.get('grid_size', 32)  # Size of the latent grid

        # Section 3: Model architecture components

        # Input embedding layer
        self.input_embedding = nn.Linear(
            self.feature_dim,
            self.hidden_dim
        )

        # Spatial embedding for nodes (learnable position encoding)
        self.spatial_embedding = nn.Parameter(
            torch.randn(self.num_nodes, self.hidden_dim)
        )

        # Temporal embedding
        self.temporal_embedding = nn.Parameter(
            torch.randn(self.input_window + self.output_window, self.hidden_dim)
        )

        # Grid encoder: Maps observations to a latent grid representation
        self.grid_encoder = GridEncoder(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            grid_size=self.grid_size,
            dropout=self.dropout
        )

        # Transformer encoder for processing gridded representations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        # Grid decoder: Maps from grid back to target locations
        self.grid_decoder = GridDecoder(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dropout=self.dropout
        )

        # Output projection
        self.output_projection = nn.Linear(self.hidden_dim, self.output_dim)

        self._logger.info("GriddedTNP model initialized with {} parameters".format(
            sum(p.numel() for p in self.parameters() if p.requires_grad)
        ))

    def forward(self, batch):
        """
        Forward pass of the model.

        Args:
            batch (dict): Dictionary containing:
                - X: Input tensor of shape (batch_size, input_window, num_nodes, feature_dim)
                - y: Target tensor of shape (batch_size, output_window, num_nodes, output_dim) [optional]

        Returns:
            torch.Tensor: Predictions of shape (batch_size, output_window, num_nodes, output_dim)
        """
        x = batch['X'].to(self.device)  # (B, T_in, N, F_in)
        batch_size = x.shape[0]

        # Step 1: Embed input features
        # Reshape for embedding: (B, T_in, N, F_in) -> (B, T_in*N, F_in)
        x_reshaped = x.reshape(batch_size, self.input_window * self.num_nodes, self.feature_dim)
        x_embedded = self.input_embedding(x_reshaped)  # (B, T_in*N, H)

        # Step 2: Add spatial and temporal embeddings
        # Create spatial embeddings repeated for each time step
        spatial_emb = self.spatial_embedding.unsqueeze(0).unsqueeze(0)  # (1, 1, N, H)
        spatial_emb = spatial_emb.repeat(batch_size, self.input_window, 1, 1)  # (B, T_in, N, H)
        spatial_emb = spatial_emb.reshape(batch_size, self.input_window * self.num_nodes, self.hidden_dim)

        # Create temporal embeddings repeated for each node
        temporal_emb = self.temporal_embedding[:self.input_window].unsqueeze(0).unsqueeze(2)  # (1, T_in, 1, H)
        temporal_emb = temporal_emb.repeat(batch_size, 1, self.num_nodes, 1)  # (B, T_in, N, H)
        temporal_emb = temporal_emb.reshape(batch_size, self.input_window * self.num_nodes, self.hidden_dim)

        # Combine embeddings
        x_embedded = x_embedded + spatial_emb + temporal_emb  # (B, T_in*N, H)

        # Step 3: Encode to grid
        grid_features = self.grid_encoder(x_embedded)  # (B, grid_size, H)

        # Step 4: Process with transformer
        grid_features = self.transformer_encoder(grid_features)  # (B, grid_size, H)

        # Step 5: Decode from grid to target locations
        # Create query locations for output window
        output_spatial_emb = self.spatial_embedding.unsqueeze(0).unsqueeze(0)  # (1, 1, N, H)
        output_spatial_emb = output_spatial_emb.repeat(batch_size, self.output_window, 1, 1)
        output_spatial_emb = output_spatial_emb.reshape(batch_size, self.output_window * self.num_nodes, self.hidden_dim)

        output_temporal_emb = self.temporal_embedding[self.input_window:self.input_window + self.output_window]
        output_temporal_emb = output_temporal_emb.unsqueeze(0).unsqueeze(2)  # (1, T_out, 1, H)
        output_temporal_emb = output_temporal_emb.repeat(batch_size, 1, self.num_nodes, 1)
        output_temporal_emb = output_temporal_emb.reshape(batch_size, self.output_window * self.num_nodes, self.hidden_dim)

        query_features = output_spatial_emb + output_temporal_emb  # (B, T_out*N, H)

        # Decode from grid
        decoded_features = self.grid_decoder(grid_features, query_features)  # (B, T_out*N, H)

        # Step 6: Project to output dimension
        predictions = self.output_projection(decoded_features)  # (B, T_out*N, output_dim)

        # Reshape to (B, T_out, N, output_dim)
        predictions = predictions.reshape(batch_size, self.output_window, self.num_nodes, self.output_dim)

        return predictions

    def predict(self, batch):
        """
        Make predictions for a batch of data.

        Args:
            batch (dict): Input batch dictionary

        Returns:
            torch.Tensor: Predictions of shape (batch_size, output_window, num_nodes, output_dim)
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate the training loss.

        Args:
            batch (dict): Batch containing input X and target y

        Returns:
            torch.Tensor: Scalar loss value
        """
        y_true = batch['y'].to(self.device)
        y_pred = self.predict(batch)

        # Use inverse transform if scaler is available
        if self._scaler is not None:
            y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
            y_pred = self._scaler.inverse_transform(y_pred[..., :self.output_dim])

        # Use masked MAE loss (common for traffic prediction)
        return loss.masked_mae_torch(y_pred, y_true, null_val=0.0)


class GridEncoder(nn.Module):
    """
    Grid Encoder: Maps point observations to a latent grid representation.
    """

    def __init__(self, input_dim, hidden_dim, grid_size, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size

        # Learnable grid points (pseudo-tokens)
        self.grid_tokens = nn.Parameter(torch.randn(grid_size, hidden_dim))

        # Attention mechanism to aggregate observations to grid
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, num_observations, hidden_dim)

        Returns:
            Grid features (batch_size, grid_size, hidden_dim)
        """
        batch_size = x.shape[0]

        # Expand grid tokens for batch
        grid = self.grid_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # (B, grid_size, H)

        # Use attention to aggregate observations onto grid
        # Grid tokens query the observations
        grid_features, _ = self.attention(grid, x, x)  # (B, grid_size, H)

        # Residual connection and normalization
        grid_features = self.norm(grid + self.dropout(grid_features))

        return grid_features


class GridDecoder(nn.Module):
    """
    Grid Decoder: Maps from grid representation back to target locations.
    """

    def __init__(self, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Attention mechanism to decode from grid to targets
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # MLP for final processing
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, grid_features, query_features):
        """
        Args:
            grid_features: Grid representation (batch_size, grid_size, hidden_dim)
            query_features: Query locations (batch_size, num_targets, hidden_dim)

        Returns:
            Decoded features (batch_size, num_targets, hidden_dim)
        """
        # Query features attend to grid
        decoded, _ = self.attention(query_features, grid_features, grid_features)

        # Residual connection and normalization
        decoded = self.norm(query_features + self.dropout(decoded))

        # MLP processing
        decoded = decoded + self.dropout(self.mlp(decoded))

        return decoded
