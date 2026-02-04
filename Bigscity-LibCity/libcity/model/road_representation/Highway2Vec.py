"""
Highway2Vec: Unsupervised Road Network Representation Learning via Autoencoder

This model adapts the original Highway2Vec implementation for the LibCity framework.
Highway2Vec learns road network embeddings using a simple autoencoder architecture
that compresses high-dimensional road feature vectors (one-hot encoded OSM attributes)
into low-dimensional dense representations.

Original Paper: "Highway2Vec: Learning representations for road network elements"
Original Repository: https://github.com/sarm/highway2vec

Key Features:
- Simple autoencoder architecture: input -> 64 -> 3 -> 64 -> output
- Unsupervised learning via reconstruction loss (MSE)
- Input features are one-hot encoded OSM road network attributes (100-200 dimensions)
- Output is a 3D embedding suitable for visualization and clustering

Key Adaptations for LibCity:
- Inherits from AbstractTrafficStateModel (following existing road_representation patterns)
- Uses LibCity's data_feature and config patterns
- Implements predict() and calculate_loss() methods
- Adapts road segment features from LibCity's batch format
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


class AutoEncoder(nn.Module):
    """
    Core autoencoder module for Highway2Vec.

    Architecture:
        Encoder: input_dim -> hidden_dim -> latent_dim
        Decoder: latent_dim -> hidden_dim -> input_dim

    The encoder extracts dense representations while the decoder
    reconstructs the original input for unsupervised training.
    """

    def __init__(self, input_dim, hidden_dim=64, latent_dim=3):
        """
        Initialize the autoencoder.

        Args:
            input_dim: Dimension of input features (one-hot encoded road attributes)
            hidden_dim: Dimension of hidden layer (default: 64)
            latent_dim: Dimension of latent/embedding space (default: 3)
        """
        super(AutoEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder: maps input to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Decoder: reconstructs input from latent representation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        """
        Encode input features to latent representation.

        Args:
            x: [N, input_dim] tensor of input features

        Returns:
            [N, latent_dim] tensor of latent embeddings
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decode latent representation back to input space.

        Args:
            z: [N, latent_dim] tensor of latent embeddings

        Returns:
            [N, input_dim] tensor of reconstructed features
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Forward pass through encoder only (for inference/embedding extraction).

        Args:
            x: [N, input_dim] tensor of input features

        Returns:
            [N, latent_dim] tensor of latent embeddings
        """
        return self.encode(x)

    def reconstruct(self, x):
        """
        Full autoencoder pass: encode then decode.

        Args:
            x: [N, input_dim] tensor of input features

        Returns:
            [N, input_dim] tensor of reconstructed features
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


class Highway2Vec(AbstractTrafficStateModel):
    """
    Highway2Vec: Road Network Representation Learning via Autoencoder

    This model learns dense embeddings for road segments by training
    an autoencoder to reconstruct one-hot encoded road attributes.
    The latent space embeddings can be used for downstream tasks
    such as clustering, visualization, or transfer learning.

    Config Parameters:
        - input_dim: Dimension of input features (default: from data_feature['feature_dim'])
        - hidden_dim: Dimension of hidden layer (default: 64)
        - output_dim/latent_dim: Dimension of latent space (default: 3)

    Data Feature Requirements:
        - feature_dim: Dimension of input road features
        - num_nodes: Number of road segments in the network
        - scaler: Optional scaler for normalization
    """

    def __init__(self, config, data_feature):
        """
        Initialize Highway2Vec model.

        Args:
            config: Configuration dictionary containing model hyperparameters
            data_feature: Dictionary containing data-related features
        """
        super(Highway2Vec, self).__init__(config, data_feature)

        self._logger = getLogger()

        # Data features
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self._scaler = data_feature.get('scaler')

        # Update config with data features (for compatibility)
        config['num_nodes'] = self.num_nodes
        config['feature_dim'] = self.feature_dim

        # Device configuration
        self.device = config.get('device', torch.device('cpu'))

        # Model metadata for saving
        self.model = config.get('model', 'Highway2Vec')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)

        # Model hyperparameters
        # input_dim can be specified in config or defaults to feature_dim
        self.input_dim = config.get('input_dim', self.feature_dim)
        self.hidden_dim = config.get('hidden_dim', 64)
        # output_dim is the latent dimension (3D for visualization by default)
        self.output_dim = config.get('output_dim', config.get('latent_dim', 3))

        # Learning rate (stored for reference, actual LR handled by executor)
        self.lr = config.get('learning_rate', 1e-3)

        # Initialize autoencoder
        self.autoencoder = AutoEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.output_dim
        )

        self._logger.info(
            f"Highway2Vec initialized: input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, output_dim={self.output_dim}, "
            f"num_nodes={self.num_nodes}"
        )

    def forward(self, batch):
        """
        Forward pass to get embeddings.

        Args:
            batch: dict containing:
                - 'node_features': [N, feature_dim] tensor of road features
                OR
                - 'X': [batch, time, nodes, features] traffic state tensor

        Returns:
            [N, output_dim] tensor of latent embeddings
        """
        # Extract node features from batch
        inputs = self._get_node_features(batch)

        # Ensure float type for the linear layers
        inputs = inputs.float()

        # Get embeddings from encoder
        embeddings = self.autoencoder.encode(inputs)

        return embeddings

    def predict(self, batch):
        """
        Get embeddings for prediction/inference and save to disk.

        This follows the pattern established by other road representation
        models (ChebConv, LINE, etc.) which save embeddings during prediction.

        Args:
            batch: dict containing node features

        Returns:
            [N, output_dim] tensor of embeddings
        """
        embeddings = self.forward(batch)

        # Save embeddings to disk (following ChebConv pattern)
        if self.exp_id is not None:
            save_path = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.npy'.format(
                self.exp_id, self.model, self.dataset, self.output_dim)
            np.save(save_path, embeddings.detach().cpu().numpy())
            self._logger.info(f'Saved embeddings to {save_path}')

        return embeddings

    def calculate_loss(self, batch):
        """
        Calculate reconstruction loss for training.

        The autoencoder is trained to minimize the MSE between
        the original input and the reconstructed output.

        Args:
            batch: dict containing:
                - 'node_features': [N, feature_dim] tensor of road features
                - 'node_labels': [N, feature_dim] reconstruction targets (optional)
                - 'mask': [N] boolean mask for valid nodes (optional)

        Returns:
            torch.Tensor: Scalar reconstruction loss
        """
        # Get input features
        inputs = self._get_node_features(batch)
        inputs = inputs.float()

        # Get reconstruction
        reconstructed = self.autoencoder.reconstruct(inputs)

        # Get target (either node_labels or same as input for autoencoder)
        if 'node_labels' in batch:
            target = batch['node_labels'].float()
            # Apply inverse transform if scaler is available
            if self._scaler is not None:
                target = self._scaler.inverse_transform(target)
                reconstructed_unscaled = self._scaler.inverse_transform(reconstructed)
            else:
                reconstructed_unscaled = reconstructed
        else:
            # For pure autoencoder, target is the input itself
            target = inputs
            reconstructed_unscaled = reconstructed

        # Apply mask if available
        if 'mask' in batch:
            mask = batch['mask']
            return loss.masked_mse_torch(reconstructed_unscaled[mask], target[mask])
        else:
            # Standard MSE loss (matching original Highway2Vec)
            return F.mse_loss(reconstructed_unscaled, target)

    def _get_node_features(self, batch):
        """
        Extract node features from batch in various formats.

        Args:
            batch: dict that may contain 'node_features', 'X', or other formats

        Returns:
            [N, feature_dim] tensor of node features
        """
        if 'node_features' in batch:
            return batch['node_features'].to(self.device)
        elif 'X' in batch:
            # Traffic state format: (batch, time, nodes, features)
            x = batch['X']
            if x.dim() == 4:
                # Take the last timestep and flatten batch dimension
                # Shape: (batch, nodes, features) -> (batch * nodes, features)
                x = x[:, -1, :, :]
                x = x.reshape(-1, x.shape[-1])
            elif x.dim() == 3:
                # Shape: (batch, nodes, features) -> (batch * nodes, features)
                x = x.reshape(-1, x.shape[-1])
            elif x.dim() == 2:
                # Already in (nodes, features) format
                pass
            return x.to(self.device)
        else:
            raise ValueError("batch must contain 'node_features' or 'X'")

    def get_embeddings(self, batch):
        """
        Get embeddings as a numpy array.

        Args:
            batch: dict containing node features

        Returns:
            numpy.ndarray: [N, output_dim] embeddings
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(batch)
        return embeddings.cpu().numpy()

    def save_embeddings(self, batch, save_path=None):
        """
        Save embeddings to a numpy file.

        Args:
            batch: dict containing node features
            save_path: Path to save the embeddings (optional)
        """
        embeddings = self.get_embeddings(batch)

        if save_path is None:
            save_path = f'./libcity/cache/{self.exp_id}/evaluate_cache/embedding_{self.model}_{self.dataset}_{self.output_dim}.npy'

        np.save(save_path, embeddings)
        self._logger.info(f"Embeddings saved to {save_path}")
