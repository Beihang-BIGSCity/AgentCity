# coding: utf-8
"""
GNPRSID: Graph-based Next POI Recommendation with Semantic ID

This module adapts the CRQVAE (Cosine Residual Quantized VAE) model from:
    GNPR-SID: Graph-based Next POI Recommendation with Semantic ID

Original repository: https://github.com/wds1996/GNPR-SID
V2 implementation is used as recommended.

Key adaptations made for LibCity:
1. Combined CRQVAE, ResidualVectorQuantizer, CosineVectorQuantizer, and MLPLayers into a single file
2. Adapted data handling to use LibCity's batch dictionary format
3. Implemented predict() and calculate_loss() methods following LibCity conventions
4. Added POI embedding extraction from batch data
5. Semantic ID generation for POI encoding
6. Added location prediction head for next-POI recommendation

Architecture:
- Input: POI embeddings (~79 dimensions: 64 category + 3 spatial + 12 temporal)
- Encoder: MLP [input_dim -> 512 -> 256 -> 128 -> 64]
- Residual Vector Quantization: 3 layers with 64 embeddings each
- Decoder: MLP [64 -> 128 -> 256 -> 512 -> input_dim]
- Prediction Head: Linear [e_dim -> loc_size] for next-POI prediction
- Output: Location prediction scores + Reconstructed POI embedding + Semantic IDs

Required data_feature keys:
- loc_size: Number of POI locations
- poi_embed_dim: Dimension of POI embeddings (if provided externally)
- poi_embeddings: Pre-computed POI embeddings (optional)

Required config parameters:
- input_dim: Input POI embedding dimension (default: 79)
- e_dim: Latent/codebook embedding dimension (default: 64)
- num_emb_list: List of codebook sizes for each RQ layer (default: [64, 64, 64])
- encoder_layers: Hidden layer sizes for encoder MLP (default: [512, 256, 128])
- dropout_prob: Dropout probability (default: 0.0)
- use_bn: Whether to use batch normalization (default: False)
- loss_type: Reconstruction loss type, 'mse' or 'l1' (default: 'mse')
- quant_loss_weight: Weight for quantization loss (default: 0.25)
- pred_loss_weight: Weight for location prediction loss (default: 1.0)
- beta: Commitment loss weight (default: 0.25)
- kmeans_init: Whether to use K-means initialization (default: False)
- kmeans_iters: Number of K-means iterations (default: 100)
- sk_epsilons: Sinkhorn epsilon values for each RQ layer (default: [0.05, 0.05, 0.05])
- sk_iters: Sinkhorn iterations (default: 100)
- use_linear: Whether to use linear projection for codebook (default: 0)
- use_ema: Whether to use EMA for codebook updates (default: True)
- ema_decay: EMA decay rate (default: 0.95)
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from sklearn.cluster import KMeans as SklearnKMeans

from libcity.model.abstract_model import AbstractModel


# ============================================================================
# Utility Functions
# ============================================================================

def activation_layer(activation_name="relu", emb_dim=None):
    """Create activation layer by name.

    Args:
        activation_name: Name of activation function
        emb_dim: Embedding dimension (unused but kept for compatibility)

    Returns:
        nn.Module: Activation layer or None
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "none":
            activation = None
        else:
            raise NotImplementedError(
                "activation function {} is not implemented".format(activation_name)
            )
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )
    return activation


def kmeans_clustering(samples, num_clusters, num_iters=10):
    """Perform K-means clustering using sklearn.

    Args:
        samples: Input samples tensor [B, D]
        num_clusters: Number of clusters
        num_iters: Maximum iterations

    Returns:
        Tensor: Cluster centers [num_clusters, D]
    """
    device = samples.device
    x = samples.cpu().detach().numpy()

    cluster = SklearnKMeans(n_clusters=num_clusters, max_iter=num_iters, n_init=1)
    cluster.fit(x)

    centers = cluster.cluster_centers_
    tensor_centers = torch.from_numpy(centers).float().to(device)

    return tensor_centers


@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    """Sinkhorn algorithm for optimal transport.

    Args:
        distances: Distance matrix [B, K]
        epsilon: Regularization parameter
        sinkhorn_iterations: Number of iterations

    Returns:
        Tensor: Assignment matrix [B, K]
    """
    distances = torch.clamp(distances, min=-1e3, max=1e3)
    Q = torch.exp(-distances / (epsilon + 1e-8))
    Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-8)

    for _ in range(sinkhorn_iterations):
        Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-8)
        Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-8)

    return Q


# ============================================================================
# MLP Layers
# ============================================================================

class MLPLayers(nn.Module):
    """Multi-layer Perceptron with configurable layers.

    Args:
        layers: List of layer dimensions [input_dim, hidden1, ..., output_dim]
        dropout: Dropout probability
        activation: Activation function name
        bn: Whether to use batch normalization
    """

    def __init__(self, layers, dropout=0.0, activation="relu", bn=False):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Linear(input_size, output_size))

            # Add batch norm (except for last layer)
            if self.use_bn and idx != (len(self.layers) - 2):
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))

            # Add activation (except for last layer)
            if idx != len(self.layers) - 2:
                activation_func = activation_layer(self.activation, output_size)
                if activation_func is not None:
                    mlp_modules.append(activation_func)

            mlp_modules.append(nn.Dropout(p=self.dropout))

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using Xavier normal initialization."""
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        """Forward pass through MLP layers."""
        return self.mlp_layers(input_feature)


# ============================================================================
# Cosine Vector Quantizer with EMA
# ============================================================================

class CosineVectorQuantizer(nn.Module):
    """Cosine similarity-based Vector Quantizer with EMA updates.

    This quantizer uses cosine similarity for index selection and supports:
    - K-means initialization
    - Sinkhorn-Knopp algorithm for balanced assignment
    - EMA updates for codebook vectors
    - Dead code replacement

    Args:
        n_e: Number of embeddings in codebook
        e_dim: Embedding dimension
        beta: Commitment loss weight
        kmeans_init: Whether to use K-means initialization
        kmeans_iters: K-means iterations
        sk_epsilon: Sinkhorn epsilon (None to disable)
        sk_iters: Sinkhorn iterations
        use_linear: Whether to use linear projection for codebook
        use_ema: Whether to use EMA updates
        ema_decay: EMA decay rate
        ema_epsilon: EMA epsilon for numerical stability
    """

    def __init__(self, n_e, e_dim, beta=0.25, kmeans_init=False, kmeans_iters=10,
                 sk_epsilon=None, sk_iters=100, use_linear=0, use_ema=True,
                 ema_decay=0.95, ema_epsilon=1e-5):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.use_linear = use_linear

        # EMA parameters
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon

        if use_ema:
            self.register_buffer('cluster_size', torch.zeros(n_e))
            self.register_buffer('ema_w', torch.zeros(n_e, e_dim))
            if use_linear == 1:
                self.use_linear = 0

        # Initialize codebook
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

        if use_ema:
            self.embedding.weight.requires_grad_(False)

        if use_linear == 1:
            self.codebook_projection = nn.Linear(self.e_dim, self.e_dim)
            nn.init.normal_(self.codebook_projection.weight, std=self.e_dim ** -0.5)

    def get_codebook(self):
        """Get codebook embeddings, optionally projected."""
        codebook = self.embedding.weight
        if self.use_linear:
            codebook = self.codebook_projection(codebook)
        return codebook

    @torch.no_grad()
    def init_emb(self, data):
        """Initialize codebook using K-means."""
        centers = kmeans_clustering(data, self.n_e, self.kmeans_iters)
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    def forward(self, x, use_sk=True):
        """Forward pass through vector quantizer.

        Args:
            x: Input tensor [B, D]
            use_sk: Whether to use Sinkhorn-Knopp algorithm

        Returns:
            x_q: Quantized output [B, D]
            loss: Quantization loss (scalar)
            indices: Codebook indices [B]
            scalar: Projection scalars [B]
        """
        B, D = x.shape
        latent = x.view(B, D)

        # Initialize codebook with K-means if needed
        if not self.initted and self.training:
            self.init_emb(latent)

        codebook = self.get_codebook()  # [K, D]

        # Cosine similarity for index selection
        latent_norm = F.normalize(latent, dim=1)
        codebook_norm = F.normalize(codebook, dim=1)
        sim = torch.matmul(latent_norm, codebook_norm.t())  # [B, K]
        distances = 1 - sim  # Lower is closer

        # Sinkhorn-Knopp for balanced assignment
        if use_sk and self.sk_epsilon is not None and self.sk_epsilon > 0:
            d_soft = self._center_distance_for_constraint(distances)
            d_soft = d_soft.double()
            Q = sinkhorn_algorithm(d_soft, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any():
                indices = torch.argmin(distances, dim=-1)
            else:
                indices = torch.argmax(Q, dim=-1)
        else:
            indices = torch.argmin(distances, dim=-1)

        # Get codebook vectors
        codebook_vec = F.embedding(indices, codebook)  # [B, D]

        # Projection quantization: w = (x . c) / ||c||^2
        dot_product = torch.sum(latent * codebook_vec, dim=-1, keepdim=True)  # [B, 1]
        norm_sq = torch.sum(codebook_vec * codebook_vec, dim=-1, keepdim=True)
        scalar = dot_product / (norm_sq + 1e-8)  # [B, 1]
        proj_vec = scalar * codebook_vec

        # Loss calculation
        if self.use_ema:
            commitment_loss = F.cosine_similarity(proj_vec.detach(), latent, dim=-1)
            loss = self.beta * (1 - commitment_loss).mean()
        else:
            commitment_loss = F.cosine_similarity(proj_vec.detach(), latent, dim=-1)
            codebook_loss = F.cosine_similarity(proj_vec, latent.detach(), dim=-1)
            loss = (1 - codebook_loss).mean() + self.beta * (1 - commitment_loss).mean()

        # Straight-through estimator
        x_q = x + (proj_vec - x).detach()

        # EMA update (training only)
        if self.use_ema and self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.n_e).float()
                cluster_size = one_hot.sum(dim=0)
                self.cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)

                dw = torch.zeros_like(self.ema_w)
                dw.index_add_(0, indices, latent.to(self.ema_w.device))
                self.ema_w.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)

                # Update embedding weights
                n = self.cluster_size.unsqueeze(1).clamp(min=self.ema_epsilon)
                self.embedding.weight.data.copy_(self.ema_w / n)

                # Dead code replacement
                avg_usage = self.cluster_size.mean()
                dead_threshold = avg_usage * 0.1
                dead_indices = torch.where(self.cluster_size < dead_threshold)[0]
                num_dead = dead_indices.numel()
                if num_dead > 0 and B > 0:
                    if B >= num_dead:
                        sample_indices = torch.randperm(B)[:num_dead]
                    else:
                        sample_indices = torch.randint(0, B, (num_dead,), device=latent.device)
                    replace_samples = latent[sample_indices].to(self.embedding.weight.device)
                    self.embedding.weight.data[dead_indices] = replace_samples
                    self.cluster_size[dead_indices] = 1.0
                    self.ema_w[dead_indices] = replace_samples.to(self.ema_w.device)

        indices = indices.view(B)
        scalar = scalar.view(B)

        return x_q, loss, indices, scalar

    @staticmethod
    def _center_distance_for_constraint(distances):
        """Center and normalize distances for Sinkhorn."""
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        centered_distances = (distances - middle) / amplitude
        return centered_distances


# ============================================================================
# Residual Vector Quantizer
# ============================================================================

class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer with multiple quantization layers.

    This implements hierarchical quantization where each layer quantizes
    the residual from previous layers.

    Args:
        n_e_list: List of codebook sizes for each layer
        e_dim: Embedding dimension
        sk_epsilons: List of Sinkhorn epsilon values
        beta: Commitment loss weight
        kmeans_init: Whether to use K-means initialization
        kmeans_iters: K-means iterations
        sk_iters: Sinkhorn iterations
        use_linear: Whether to use linear projection
    """

    def __init__(self, n_e_list, e_dim, sk_epsilons=None, beta=0.25,
                 kmeans_init=False, kmeans_iters=100, sk_iters=100, use_linear=0):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons if sk_epsilons else [0.05] * len(n_e_list)
        self.sk_iters = sk_iters
        self.use_linear = use_linear

        self.vq_layers = nn.ModuleList([
            CosineVectorQuantizer(
                n_e, e_dim,
                beta=self.beta,
                kmeans_init=self.kmeans_init,
                kmeans_iters=self.kmeans_iters,
                sk_epsilon=sk_epsilon,
                sk_iters=sk_iters,
                use_linear=use_linear
            )
            for n_e, sk_epsilon in zip(n_e_list, self.sk_epsilons)
        ])

    def forward(self, x, use_sk=True):
        """Forward pass through residual quantizer.

        Args:
            x: Input tensor [B, D] or [B, T, D]
            use_sk: Whether to use Sinkhorn-Knopp algorithm

        Returns:
            x_q: Quantized output (same shape as input)
            mean_loss: Average quantization loss
            (all_indices, all_scalars): Tuple of indices and scalars
        """
        original_shape = x.shape
        if x.ndim == 3:
            B, T, D = x.shape
            x = x.view(-1, D)  # [B*T, D]
        elif x.ndim == 2:
            B, D = x.shape
        else:
            raise ValueError("x must be [B, D] or [B, T, D]")

        residual = x
        x_q = torch.zeros_like(x)
        all_losses = []
        all_indices = []
        all_scalars = []

        for quantizer in self.vq_layers:
            x_res, loss, indices, scalar = quantizer(residual, use_sk=use_sk)
            x_q = x_q + x_res
            residual = residual - x_res

            all_losses.append(loss)
            all_indices.append(indices)
            all_scalars.append(scalar)

        x_q = x_q.view(original_shape)

        mean_loss = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)  # [B, L]
        all_scalars = torch.stack(all_scalars, dim=-1)  # [B, L]

        if len(original_shape) == 3:
            all_indices = all_indices.view(B, T, -1)  # [B, T, L]
            all_scalars = all_scalars.view(B, T, -1)  # [B, T, L]
        else:
            all_indices = all_indices.view(B, -1)  # [B, L]
            all_scalars = all_scalars.view(B, -1)  # [B, L]

        return x_q, mean_loss, (all_indices, all_scalars)

    @torch.no_grad()
    def get_codebook(self):
        """Get all codebooks stacked."""
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)


# ============================================================================
# GNPRSID (CRQVAE) Model
# ============================================================================

class GNPRSID(AbstractModel):
    """GNPRSID: Graph-based Next POI Recommendation with Semantic ID.

    This model implements a Cosine Residual Quantized VAE (CRQVAE) for
    generating semantic IDs from POI embeddings. The semantic IDs can be
    used for efficient POI recommendation and trajectory encoding.

    The model consists of:
    1. Encoder MLP: Maps POI embeddings to latent space
    2. Residual Vector Quantizer: Generates discrete semantic IDs
    3. Decoder MLP: Reconstructs POI embeddings from quantized latents

    Args:
        config: Configuration dictionary
        data_feature: Data feature dictionary
    """

    def __init__(self, config, data_feature):
        super(GNPRSID, self).__init__(config, data_feature)

        self._logger = logging.getLogger(__name__)
        self.device = config.get('device', 'cpu')

        # Data dimensions from data_feature
        self.num_poi = data_feature.get('loc_size', 1000)

        # Model hyperparameters from config
        self.input_dim = config.get('input_dim', 79)  # POI embedding dimension
        self.e_dim = config.get('e_dim', 64)  # Latent dimension
        self.num_emb_list = config.get('num_emb_list', [64, 64, 64])  # Codebook sizes
        self.encoder_layers = config.get('encoder_layers', [512, 256, 128])  # Hidden layers
        self.dropout_prob = config.get('dropout_prob', 0.0)
        self.use_bn = config.get('use_bn', False)
        self.loss_type = config.get('loss_type', 'mse')  # 'mse' or 'l1'
        self.quant_loss_weight = config.get('quant_loss_weight', 0.25)
        self.pred_loss_weight = config.get('pred_loss_weight', 1.0)  # Location prediction loss weight
        self.beta = config.get('beta', 0.25)  # Commitment loss weight
        self.kmeans_init = config.get('kmeans_init', False)
        self.kmeans_iters = config.get('kmeans_iters', 100)
        self.sk_epsilons = config.get('sk_epsilons', [0.05, 0.05, 0.05])
        self.sk_iters = config.get('sk_iters', 100)
        self.use_linear = config.get('use_linear', 0)
        self.use_sk = config.get('use_sk', True)  # Use Sinkhorn during training

        # POI embedding options
        self.use_external_embeddings = config.get('use_external_embeddings', False)

        # Load pre-computed POI embeddings if available
        self._init_poi_embeddings(data_feature, config)

        # Build encoder
        self.encode_layer_dims = [self.input_dim] + self.encoder_layers + [self.e_dim]
        self.encoder = MLPLayers(
            layers=self.encode_layer_dims,
            dropout=self.dropout_prob,
            bn=self.use_bn
        )

        # Build residual vector quantizer
        self.rq = ResidualVectorQuantizer(
            self.num_emb_list, self.e_dim,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
            use_linear=self.use_linear
        )

        # Build decoder (reverse of encoder)
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(
            layers=self.decode_layer_dims,
            dropout=self.dropout_prob,
            bn=self.use_bn
        )

        # Build prediction head for next-POI prediction
        # Maps from quantized latent space to location scores
        self.pred_head = nn.Sequential(
            nn.Linear(self.e_dim, self.e_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(self.e_dim * 2, self.num_poi)
        )
        # Initialize prediction head
        for layer in self.pred_head:
            if isinstance(layer, nn.Linear):
                xavier_normal_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.0)

        self._logger.info(f"GNPRSID initialized with input_dim={self.input_dim}, "
                          f"e_dim={self.e_dim}, num_quantizers={len(self.num_emb_list)}, "
                          f"num_poi={self.num_poi}")

    def _init_poi_embeddings(self, data_feature, config):
        """Initialize POI embeddings from data_feature or create learnable ones.

        Args:
            data_feature: Data feature dictionary
            config: Configuration dictionary
        """
        poi_embeddings = data_feature.get('poi_embeddings', None)

        if poi_embeddings is not None:
            self._logger.info("Using pre-computed POI embeddings from data_feature")
            if isinstance(poi_embeddings, np.ndarray):
                poi_embeddings = torch.from_numpy(poi_embeddings).float()
            self.register_buffer('poi_embeddings', poi_embeddings)
            # Update input_dim based on actual embedding dimension
            self.input_dim = poi_embeddings.shape[1]
        elif self.use_external_embeddings:
            # Embeddings should be provided in batch
            self._logger.info("Will use external POI embeddings from batch data")
            self.poi_embeddings = None
        else:
            # Create learnable embeddings
            self._logger.info("Creating learnable POI embeddings")
            self.poi_embedding_layer = nn.Embedding(self.num_poi, self.input_dim)
            nn.init.xavier_normal_(self.poi_embedding_layer.weight)
            self.poi_embeddings = None

    def _get_poi_embeddings(self, batch):
        """Get POI embeddings from batch or internal storage.

        Args:
            batch: Batch dictionary

        Returns:
            Tensor: POI embeddings [batch_size, seq_len, input_dim] or [batch_size, input_dim]
        """
        # Check if embeddings are provided in batch
        # Note: Use batch.data for key checking since LibCity's Batch class
        # doesn't support the 'in' operator directly
        if 'poi_embeddings' in batch.data:
            return batch['poi_embeddings']

        # Check if we have pre-computed embeddings
        if self.poi_embeddings is not None:
            # Get POI indices from batch
            if 'current_loc' in batch.data:
                poi_indices = batch['current_loc']  # [batch_size, seq_len]
            elif 'loc' in batch.data:
                poi_indices = batch['loc']
            else:
                raise ValueError("Batch must contain 'current_loc' or 'loc' or 'poi_embeddings'")

            # Index embeddings
            poi_indices_clamped = torch.clamp(poi_indices, 0, self.poi_embeddings.shape[0] - 1)
            return self.poi_embeddings[poi_indices_clamped]

        # Use learnable embeddings
        if hasattr(self, 'poi_embedding_layer'):
            if 'current_loc' in batch.data:
                poi_indices = batch['current_loc']
            elif 'loc' in batch.data:
                poi_indices = batch['loc']
            else:
                raise ValueError("Batch must contain 'current_loc' or 'loc'")

            poi_indices_clamped = torch.clamp(poi_indices, 0, self.num_poi - 1)
            return self.poi_embedding_layer(poi_indices_clamped)

        raise ValueError("No POI embeddings available. Provide embeddings in batch or config.")

    def forward(self, batch, use_sk=None):
        """Forward pass through CRQVAE.

        Args:
            batch: Dictionary containing POI data
            use_sk: Override for Sinkhorn usage (None uses self.use_sk)

        Returns:
            out: Reconstructed POI embeddings
            rq_loss: Quantization loss
            codes: Tuple of (indices, scalars) from RQ
        """
        if use_sk is None:
            use_sk = self.use_sk and self.training

        # Get POI embeddings
        x = self._get_poi_embeddings(batch)
        original_shape = x.shape

        # Flatten if 3D (batch, seq, dim)
        if x.ndim == 3:
            batch_size, seq_len, _ = x.shape
            x = x.view(-1, x.shape[-1])

        # Encode
        z = self.encoder(x)

        # Quantize
        z_q, rq_loss, codes = self.rq(z, use_sk=use_sk)

        # Decode
        out = self.decoder(z_q)

        # Reshape if needed
        if len(original_shape) == 3:
            out = out.view(batch_size, seq_len, -1)

        return out, rq_loss, codes

    def predict(self, batch):
        """Predict next POI location scores.

        This method returns location prediction scores compatible with
        TrajLocPredExecutor's evaluation framework (Recall@k).

        Args:
            batch: Batch object containing POI data

        Returns:
            Tensor: Location scores [batch_size, num_poi] - higher score means
                    higher probability of visiting that location next
        """
        # Get POI embeddings
        x = self._get_poi_embeddings(batch)

        # Handle sequence data - use the last position for prediction
        if x.ndim == 3:
            batch_size, seq_len, _ = x.shape
            # Use the last valid position in the sequence
            x = x[:, -1, :]  # [batch_size, input_dim]

        # Encode to latent space
        z = self.encoder(x)

        # Quantize (without Sinkhorn during inference)
        z_q, _, _ = self.rq(z, use_sk=False)

        # Get location prediction scores
        loc_scores = self.pred_head(z_q)  # [batch_size, num_poi]

        # Apply log_softmax for compatibility with NLLLoss-based evaluation
        loc_scores = F.log_softmax(loc_scores, dim=-1)

        return loc_scores

    def calculate_loss(self, batch):
        """Calculate total loss for training.

        The loss consists of three components:
        1. Reconstruction loss: MSE/L1 between input and reconstructed embeddings
        2. Quantization loss: Commitment loss from residual vector quantizer
        3. Prediction loss: Cross-entropy loss for next-POI prediction

        Args:
            batch: Batch object containing POI data

        Returns:
            Tensor: Total loss (reconstruction + quantization + prediction)
        """
        # Get original POI embeddings as target
        x_original = self._get_poi_embeddings(batch)
        original_shape = x_original.shape

        # Handle sequence data
        if x_original.ndim == 3:
            batch_size, seq_len, embed_dim = x_original.shape
            # Use last position for prediction
            x_last = x_original[:, -1, :]  # [batch_size, embed_dim]
            x_flat = x_original.view(-1, embed_dim)  # [batch_size * seq_len, embed_dim]
        else:
            batch_size = x_original.shape[0]
            x_last = x_original
            x_flat = x_original

        # Forward pass for reconstruction
        out, rq_loss, codes = self.forward(batch, use_sk=self.use_sk)

        # Flatten output if needed
        if out.ndim == 3:
            out = out.view(-1, out.shape[-1])

        # Reconstruction loss
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, x_flat, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, x_flat, reduction='mean')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Location prediction loss
        # Encode the last position and predict next location
        z_last = self.encoder(x_last)
        z_q_last, _, _ = self.rq(z_last, use_sk=self.use_sk)
        loc_scores = self.pred_head(z_q_last)  # [batch_size, num_poi]
        loc_log_probs = F.log_softmax(loc_scores, dim=-1)

        # Get target locations from batch
        if 'target' in batch.data:
            target = batch['target']  # [batch_size]
            # Clamp target to valid range
            target = torch.clamp(target, 0, self.num_poi - 1)
            loss_pred = F.nll_loss(loc_log_probs, target, reduction='mean')
        else:
            # If no target available, skip prediction loss
            loss_pred = torch.tensor(0.0, device=x_original.device)

        # Total loss
        total_loss = (loss_recon +
                      self.quant_loss_weight * rq_loss +
                      self.pred_loss_weight * loss_pred)

        return total_loss

    @torch.no_grad()
    def get_semantic_ids(self, batch, use_sk=False):
        """Get semantic IDs (indices) for POIs.

        This is the main method for generating semantic IDs that can be
        used for downstream tasks.

        Args:
            batch: Dictionary containing POI data
            use_sk: Whether to use Sinkhorn for assignment

        Returns:
            x_q: Quantized embeddings
            indices: Semantic IDs [batch_size, num_quantizers]
        """
        x = self._get_poi_embeddings(batch)
        original_shape = x.shape

        # Flatten if 3D
        if x.ndim == 3:
            batch_size, seq_len, _ = x.shape
            x = x.view(-1, x.shape[-1])

        # Encode
        z = self.encoder(x)

        # Quantize
        z_q, _, (indices, scalars) = self.rq(z, use_sk=use_sk)

        # Reshape if needed
        if len(original_shape) == 3:
            z_q = z_q.view(batch_size, seq_len, -1)
            indices = indices.view(batch_size, seq_len, -1)

        return z_q, indices

    @torch.no_grad()
    def encode(self, x):
        """Encode POI embeddings to latent space.

        Args:
            x: POI embeddings [batch_size, input_dim]

        Returns:
            Tensor: Latent embeddings [batch_size, e_dim]
        """
        return self.encoder(x)

    @torch.no_grad()
    def decode(self, z):
        """Decode latent embeddings to POI embeddings.

        Args:
            z: Latent embeddings [batch_size, e_dim]

        Returns:
            Tensor: Reconstructed POI embeddings [batch_size, input_dim]
        """
        return self.decoder(z)

    @torch.no_grad()
    def get_codebooks(self):
        """Get all codebook embeddings.

        Returns:
            Tensor: Codebooks [num_quantizers, codebook_size, e_dim]
        """
        return self.rq.get_codebook()
