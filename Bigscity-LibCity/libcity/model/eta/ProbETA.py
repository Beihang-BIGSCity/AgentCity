"""
ProbETA Model: Probabilistic Embedding-based Travel Time Estimation

Adapted from: /home/wangwenrui/shk/AgentCity/repos/ProbETA/Model/ProbETA/model.py

This model implements a probabilistic embedding-based neural network for travel time estimation.
It constructs two separate embedding spaces for road segments and applies linear transformations
to project them into different latent feature spaces for mean and covariance prediction.

Key Features:
- Dual embedding layers for road segments
- Mean prediction network (4 FC layers with dropout)
- Covariance estimation network (3 FC layers with dropout)
- Multivariate Gaussian NLL loss

Original Authors: ProbETA team
LibCity Adaptation: Model Adaptation Agent

Key Changes from Original:
1. Inherits from AbstractTrafficStateModel
2. Extracts parameters from config and data_feature dictionaries
3. Handles LibCity batch format with 'road_segments', 'device_ids', 'time' keys
4. Implements predict() and calculate_loss() methods per LibCity conventions
5. Added padding/masking support for variable-length sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import math

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def calculate_similarity_matrices(devid):
    """
    Calculate similarity matrix based on device IDs.
    Samples from the same device get similarity 1, otherwise 0.

    Args:
        devid: Device ID tensor of shape (batch_size,)

    Returns:
        same_dev_mask: Binary similarity matrix of shape (batch_size, batch_size)
    """
    same_dev_mask = (devid.unsqueeze(0) == devid.unsqueeze(1)).int()
    return same_dev_mask


def multivariate_gaussian_nll_loss(predicted_mean, predicted_covariance, observed_values):
    """
    Compute negative log-likelihood loss for multivariate Gaussian distribution.

    Args:
        predicted_mean: Predicted mean tensor of shape (batch_size,)
        predicted_covariance: Predicted covariance matrix of shape (batch_size, batch_size)
        observed_values: Ground truth values of shape (batch_size,)

    Returns:
        Negative log-likelihood scalar
    """
    # Symmetrize covariance matrix
    predicted_covariance = (predicted_covariance + predicted_covariance.mT) / 2

    # Add small diagonal term for numerical stability
    batch_size = predicted_covariance.size(0)
    eps = 1e-4
    predicted_covariance = predicted_covariance + eps * torch.eye(
        batch_size, device=predicted_covariance.device
    )

    try:
        mvn = dist.MultivariateNormal(predicted_mean, predicted_covariance)
        likelihood = mvn.log_prob(observed_values)
        return -likelihood
    except RuntimeError:
        # Fallback to MSE loss if covariance is not positive definite
        mse_loss = F.mse_loss(predicted_mean, observed_values)
        return mse_loss


class CRPSMetric:
    """
    Continuous Ranked Probability Score (CRPS) for Gaussian distributions.
    Used for probabilistic forecast evaluation.
    """
    def __init__(self, x, loc, scale):
        self.value = x
        self.loc = loc
        self.scale = torch.sqrt(scale)

    def gaussian_pdf(self, x):
        """Probability density function of standard Gaussian."""
        _normconst = 1.0 / math.sqrt(2.0 * math.pi)
        return _normconst * torch.exp(-(x * x) / 2.0)

    def gaussian_cdf(self, x):
        """Cumulative distribution function of standard Gaussian."""
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def gaussian_crps(self):
        """Compute CRPS for Gaussian distribution."""
        sx = (self.value - self.loc) / self.scale
        pdf = self.gaussian_pdf(sx)
        cdf = self.gaussian_cdf(sx)
        pi_inv = 1.0 / math.sqrt(math.pi)
        crps = self.scale * (sx * (2 * cdf - 1) + 2 * pdf - pi_inv)
        return crps


class ProbETA(AbstractTrafficStateModel):
    """
    ProbETA Model: Probabilistic Embedding-based Travel Time Estimation.

    This model constructs two separate embedding spaces for road segments and applies
    linear transformations to predict mean travel time and covariance matrix.

    Args:
        config: Configuration dictionary containing model hyperparameters
        data_feature: Data feature dictionary containing dataset statistics

    Config Parameters:
        - embedding_dim: Dimension of road segment embeddings (default: 64)
        - dropout_mean: Dropout rate for mean prediction network (default: 0.9)
        - dropout_cov: Dropout rate for covariance network (default: 0.3)
        - hidden_mean_1: First hidden layer size for mean network (default: 72)
        - hidden_mean_2: Second hidden layer size for mean network (default: 64)
        - hidden_mean_3: Third hidden layer size for mean network (default: 32)
        - hidden_cov_1: First hidden layer size for covariance network (default: 32)
        - hidden_cov_2: Second hidden layer size for covariance network (default: 16)
        - use_device_similarity: Whether to use device ID for similarity (default: True)
        - loss_type: Loss function type ('nll' or 'mse') (default: 'nll')

    Data Feature Parameters:
        - road_num: Number of road segments in the dataset
        - time_mean: Mean of travel time for normalization (optional)
        - time_std: Standard deviation of travel time (optional)
    """

    def __init__(self, config, data_feature):
        super(ProbETA, self).__init__(config, data_feature)

        self.config = config
        self.data_feature = data_feature
        self.device = config.get('device', torch.device('cpu'))

        # Model parameters from config
        self.embedding_dim = config.get('embedding_dim', 64)
        self.dropout_mean = config.get('dropout_mean', 0.9)
        self.dropout_cov = config.get('dropout_cov', 0.3)
        self.hidden_mean_1 = config.get('hidden_mean_1', 72)
        self.hidden_mean_2 = config.get('hidden_mean_2', 64)
        self.hidden_mean_3 = config.get('hidden_mean_3', 32)
        self.hidden_cov_1 = config.get('hidden_cov_1', 32)
        self.hidden_cov_2 = config.get('hidden_cov_2', 16)
        self.use_device_similarity = config.get('use_device_similarity', True)
        self.loss_type = config.get('loss_type', 'nll')

        # Data feature parameters
        self.road_num = data_feature.get('road_num', 10000)
        self.time_mean = data_feature.get('time_mean', 0.0)
        self.time_std = data_feature.get('time_std', 1.0)

        # Define two separate embedding layers for roads
        # +1 for padding index (0)
        self.embeddings1 = nn.Embedding(self.road_num + 1, self.embedding_dim, padding_idx=0)
        self.embeddings2 = nn.Embedding(self.road_num + 1, self.embedding_dim, padding_idx=0)

        # Fully connected layers for mean estimation
        self.proj_m = nn.Linear(self.embedding_dim, self.hidden_mean_1)
        self.hidden_m = nn.Linear(self.hidden_mean_1, self.hidden_mean_2)
        self.hidden_m2 = nn.Linear(self.hidden_mean_2, self.hidden_mean_3)
        self.output_m = nn.Linear(self.hidden_mean_3, 1)

        # Fully connected layers for covariance matrix estimation
        self.proj_d = nn.Linear(self.embedding_dim, self.hidden_cov_1)
        self.hidden_d = nn.Linear(self.hidden_cov_1, self.hidden_cov_2)
        self.hidden_d2 = nn.Linear(self.hidden_cov_2, 1)

        # Dropout layers for regularization
        self.dropout = nn.Dropout(p=self.dropout_mean)
        self.dropout2 = nn.Dropout(p=self.dropout_cov)

        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        """
        Initialize embeddings with orthogonal initialization and normalization.
        This follows the original ProbETA implementation.
        """
        with torch.no_grad():
            nn.init.orthogonal_(self.embeddings1.weight)
            self.embeddings1.weight.div_(
                torch.norm(self.embeddings1.weight, dim=1, keepdim=True).clamp(min=1e-8)
            )
            nn.init.orthogonal_(self.embeddings2.weight)
            self.embeddings2.weight.div_(
                torch.norm(self.embeddings2.weight, dim=1, keepdim=True).clamp(min=1e-8)
            )
            # Zero out padding embedding
            self.embeddings1.weight[0].zero_()
            self.embeddings2.weight[0].zero_()

    def output_embeddings1(self):
        """
        Returns the first embedding matrix for all road segments.

        Returns:
            torch.Tensor: Embedding tensor of shape (road_num + 1, embedding_dim)
        """
        return self.embeddings1(torch.arange(self.road_num + 1).to(self.device))

    def output_embeddings2(self):
        """
        Returns the second embedding matrix for all road segments.

        Returns:
            torch.Tensor: Embedding tensor of shape (road_num + 1, embedding_dim)
        """
        return self.embeddings2(torch.arange(self.road_num + 1).to(self.device))

    def forward(self, batch):
        """
        Forward pass of the ProbETA model.

        Args:
            batch: Dictionary containing:
                - 'road_segments': Road segment indices (batch_size, seq_len)
                - 'device_ids': Device identifiers (batch_size,) - optional
                - 'X': Alternative key for road segments if road_segments not present

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - T_mean: Predicted mean travel time (batch_size, 1)
                - Cov: Predicted covariance matrix (batch_size, batch_size)
        """
        # Get road segment inputs from batch
        # Use try-except pattern because LibCity's Batch class doesn't support 'in' operator
        try:
            inputs = batch['road_segments']
        except KeyError:
            try:
                # LibCity default format - assume first feature channel is road segment ID
                inputs = batch['X']
                if inputs.dim() > 2:
                    # Flatten spatial/temporal dimensions if needed
                    inputs = inputs.squeeze(-1)
                    if inputs.dim() > 2:
                        inputs = inputs[:, :, 0]  # Take first feature
            except KeyError:
                raise KeyError("Batch must contain 'road_segments' or 'X' key")

        inputs = inputs.long().to(self.device)

        # Get device IDs for similarity matrix (optional)
        # Use try-except pattern because LibCity's Batch class doesn't support 'in' operator
        device_ids = None
        if self.use_device_similarity:
            try:
                device_ids = batch['device_ids'].to(self.device)
                if device_ids.dim() > 1:
                    device_ids = device_ids.squeeze(-1)  # Ensure 1D tensor [batch_size]
            except KeyError:
                pass
        if device_ids is None:
            # Create dummy device IDs (all different)
            device_ids = torch.arange(inputs.size(0), device=self.device)

        # Create mask for padding (zero indices)
        inputs_mask = torch.where(
            inputs > 0,
            torch.ones_like(inputs, dtype=torch.float),
            torch.zeros_like(inputs, dtype=torch.float)
        ).unsqueeze(2).to(self.device)

        # Compute embeddings and apply mask
        road_embeds1 = self.embeddings1(inputs) * inputs_mask
        road_embeds2 = self.embeddings2(inputs) * inputs_mask

        # Aggregate embeddings along sequence dimension
        aggregated_embeds1 = torch.sum(road_embeds1, dim=1)
        aggregated_embeds2 = torch.sum(road_embeds2, dim=1)

        # Compute first covariance matrix component (L1)
        L1Cov = torch.mm(aggregated_embeds1, aggregated_embeds1.t())

        # Compute second covariance matrix with device similarity scaling (L2)
        similarity_matrix = calculate_similarity_matrices(device_ids)
        L2Cov = torch.mm(aggregated_embeds2, aggregated_embeds2.t()) * similarity_matrix

        # Compute diagonal covariance term using neural network transformation
        D_s = torch.log(1 + torch.exp(
            self.hidden_d2(self.hidden_d(F.relu(self.proj_d(self.dropout2(aggregated_embeds2)))))
        ))

        # Compute final covariance matrix
        ST_Cov = L1Cov + torch.diag(D_s.squeeze(-1)) + L2Cov
        Cov = ST_Cov

        # Compute mean prediction using multi-layer transformation
        T_mean = self.output_m(
            self.hidden_m2(F.relu(self.hidden_m(F.relu(self.proj_m(self.dropout(aggregated_embeds1))))))
        )

        return T_mean, Cov

    def predict(self, batch):
        """
        Generate travel time predictions for a batch.

        Args:
            batch: Input batch dictionary

        Returns:
            torch.Tensor: Predicted mean travel time (batch_size, 1)
        """
        T_mean, _ = self.forward(batch)

        # Unnormalize if normalization was applied
        if self.time_std != 1.0 or self.time_mean != 0.0:
            T_mean = T_mean * self.time_std + self.time_mean

        return T_mean

    def predict_with_uncertainty(self, batch):
        """
        Generate travel time predictions with uncertainty estimates.

        Args:
            batch: Input batch dictionary

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - T_mean: Predicted mean travel time (batch_size, 1)
                - T_var: Predicted variance (batch_size,) - diagonal of covariance
        """
        T_mean, Cov = self.forward(batch)

        # Extract variance from diagonal of covariance matrix
        T_var = torch.diag(Cov)

        # Unnormalize if normalization was applied
        if self.time_std != 1.0 or self.time_mean != 0.0:
            T_mean = T_mean * self.time_std + self.time_mean
            T_var = T_var * (self.time_std ** 2)

        return T_mean, T_var

    def calculate_loss(self, batch):
        """
        Calculate training loss for the batch.

        Uses multivariate Gaussian negative log-likelihood loss by default,
        which considers both mean prediction accuracy and covariance structure.

        Args:
            batch: Input batch dictionary containing:
                - 'road_segments' or 'X': Road segment sequences
                - 'time' or 'y': Ground truth travel times
                - 'device_ids': Device identifiers (optional)

        Returns:
            torch.Tensor: Scalar loss value
        """
        T_mean, Cov = self.forward(batch)

        # Get ground truth travel times
        # Use try-except pattern because LibCity's Batch class doesn't support 'in' operator
        try:
            observed_values = batch['time'].to(self.device)
        except KeyError:
            try:
                observed_values = batch['y'].to(self.device)
            except KeyError:
                raise KeyError("Batch must contain 'time' or 'y' key for ground truth")

        # Flatten observed values if needed
        if observed_values.dim() > 1:
            observed_values = observed_values.squeeze()

        # Normalize ground truth if model uses normalization
        if self.time_std != 1.0 or self.time_mean != 0.0:
            observed_values = (observed_values - self.time_mean) / self.time_std

        # Compute loss based on loss type
        if self.loss_type == 'nll':
            # Multivariate Gaussian NLL loss
            loss = multivariate_gaussian_nll_loss(
                T_mean.squeeze(),
                Cov,
                observed_values
            )
        elif self.loss_type == 'mse':
            # Standard MSE loss (ignoring covariance)
            loss = F.mse_loss(T_mean.squeeze(), observed_values)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

    def compute_crps(self, batch):
        """
        Compute CRPS (Continuous Ranked Probability Score) for evaluation.

        Args:
            batch: Input batch dictionary

        Returns:
            torch.Tensor: CRPS score
        """
        T_mean, Cov = self.forward(batch)

        # Get ground truth
        # Use try-except pattern because LibCity's Batch class doesn't support 'in' operator
        try:
            observed_values = batch['time'].to(self.device)
        except KeyError:
            try:
                observed_values = batch['y'].to(self.device)
            except KeyError:
                raise KeyError("Batch must contain 'time' or 'y' key")

        if observed_values.dim() > 1:
            observed_values = observed_values.squeeze()

        # Get variance from diagonal
        variance = torch.diag(Cov)

        # Compute CRPS
        crps_metric = CRPSMetric(
            observed_values,
            T_mean.squeeze(),
            variance
        )

        return crps_metric.gaussian_crps().mean()
