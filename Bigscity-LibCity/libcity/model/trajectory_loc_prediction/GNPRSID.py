# coding: utf-8
"""
GNPRSID: Graph-based Next POI Recommendation using Semantic ID with Residual Quantization

This module adapts the CRQVAE model from the GNPRSID repository for LibCity.

Original repository: repos/GNPRSID/V2/SID/CRQVAE/
Original files used:
- crqvae.py: Main CRQVAE model
- rq.py: ResidualVectorQuantizer
- cvq_ema.py: CosineVectorQuantizer with EMA
- mlp.py: MLPLayers and utility functions

Key adaptations made for LibCity:
1. Combined all components (MLP, CosineVQ, ResidualVQ, CRQVAE) into a single file
2. Added POI/User embeddings for trajectory representation
3. Added prediction head for next POI prediction (original model only learns embeddings)
4. Implemented predict() and calculate_loss() following LibCity conventions
5. Adapted data handling to use LibCity's batch dictionary format

Required data_feature keys:
- loc_size: Number of POI locations
- uid_size: Number of users
- loc_pad: Padding index for location (optional)

Required config parameters:
- loc_emb_size: POI embedding dimension (default: 128)
- uid_emb_size: User embedding dimension (default: 64)
- encoder_layers: Hidden layer sizes for encoder (default: [512, 256, 128])
- e_dim: Embedding dimension for quantized codes (default: 64)
- num_codebooks: Number of codebook entries per layer (default: 64)
- num_rq_layers: Number of residual quantization layers (default: 3)
- dropout_prob: Dropout probability (default: 0.1)
- use_bn: Whether to use batch normalization (default: False)
- loss_type: Reconstruction loss type, 'mse' or 'l1' (default: 'mse')
- quant_loss_weight: Weight for quantization loss (default: 0.25)
- beta: Commitment loss coefficient (default: 0.25)
- kmeans_init: Whether to use kmeans initialization (default: False)
- sk_epsilon: Sinkhorn epsilon (default: 0.05)
- sk_iters: Sinkhorn iterations (default: 100)
- use_ema: Whether to use EMA for codebook updates (default: True)
- ema_decay: EMA decay rate (default: 0.95)
- pred_loss_weight: Weight for prediction loss (default: 1.0)
- recon_loss_weight: Weight for reconstruction loss (default: 0.1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_

from libcity.model.abstract_model import AbstractModel


def activation_layer(activation_name="relu", emb_dim=None):
    """Create activation layer by name."""
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


def kmeans(samples, num_clusters, num_iters=10):
    """K-means clustering for codebook initialization."""
    try:
        from sklearn.cluster import KMeans
        B, dim = samples.shape[0], samples.shape[-1]
        dtype, device = samples.dtype, samples.device
        x = samples.cpu().detach().numpy()

        cluster = KMeans(n_clusters=num_clusters, max_iter=num_iters, n_init=10).fit(x)
        centers = cluster.cluster_centers_
        tensor_centers = torch.from_numpy(centers).to(device).to(dtype)
        return tensor_centers
    except ImportError:
        # Fallback: random initialization
        device = samples.device
        indices = torch.randperm(samples.shape[0])[:num_clusters]
        return samples[indices].clone()


@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    """Sinkhorn algorithm for optimal transport."""
    distances = torch.clamp(distances, min=-1e3, max=1e3)
    Q = torch.exp(-distances / (epsilon + 1e-8))
    Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-8)
    for _ in range(sinkhorn_iterations):
        Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-8)
        Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-8)
    return Q


class MLPLayers(nn.Module):
    """Multi-layer perceptron with optional batch normalization and dropout."""

    def __init__(self, layers, dropout=0.0, activation="relu", bn=False):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Linear(input_size, output_size))

            if self.use_bn and idx != (len(self.layers) - 2):
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            if idx != len(self.layers) - 2:
                activation_func = activation_layer(self.activation, output_size)
                if activation_func is not None:
                    mlp_modules.append(activation_func)

            mlp_modules.append(nn.Dropout(p=self.dropout))

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)

    def init_weights(self, module):
        """Initialize weights with Xavier normal."""
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


class CosineVectorQuantizer(nn.Module):
    """
    Cosine similarity-based vector quantizer with EMA updates.

    Uses cosine similarity for codebook matching and optional projection quantization.
    Supports EMA (Exponential Moving Average) updates for codebook.
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
        """Get codebook, optionally with projection."""
        codebook = self.embedding.weight
        if self.use_linear:
            codebook = self.codebook_projection(codebook)
        return codebook

    @torch.no_grad()
    def init_emb(self, data):
        """Initialize codebook with k-means clustering."""
        centers = kmeans(data, self.n_e, self.kmeans_iters)
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    def forward(self, x, use_sk=True):
        B, D = x.shape
        latent = x.view(B, D)

        if not self.initted and self.training:
            self.init_emb(latent)

        codebook = self.get_codebook()  # [K, D]

        # Cosine similarity for index selection
        latent_norm = F.normalize(latent, dim=1)
        codebook_norm = F.normalize(codebook, dim=1)
        sim = torch.matmul(latent_norm, codebook_norm.t())  # [B, K]
        distances = 1 - sim  # smaller = closer

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

        # EMA update during training
        if self.use_ema and self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.n_e).float()
                cluster_size = one_hot.sum(dim=0)
                self.cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)

                dw = torch.zeros_like(self.ema_w)
                dw.index_add_(0, indices, latent.to(self.ema_w.device))
                self.ema_w.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)

                # Update embedding weight
                n = self.cluster_size.unsqueeze(1).clamp(min=self.ema_epsilon)
                self.embedding.weight.data.copy_(self.ema_w / n)

                # Dead code reset
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
        """Center distances for Sinkhorn algorithm."""
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        centered_distances = (distances - middle) / amplitude
        return centered_distances


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer with multiple quantization layers.

    Each layer quantizes the residual from the previous layer,
    allowing for progressive refinement of the representation.
    """

    def __init__(self, n_e_list, e_dim, sk_epsilons=None, beta=0.25,
                 kmeans_init=False, kmeans_iters=100, sk_iters=100, use_linear=0,
                 use_ema=True, ema_decay=0.95):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons if sk_epsilons is not None else [0.05] * len(n_e_list)
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
                use_linear=use_linear,
                use_ema=use_ema,
                ema_decay=ema_decay
            )
            for n_e, sk_epsilon in zip(n_e_list, self.sk_epsilons)
        ])

    def forward(self, x, use_sk=True):
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


class GNPRSID(AbstractModel):
    """
    GNPRSID: Graph-based Next POI Recommendation using Semantic ID

    This model uses a CRQVAE (Cosine Residual Quantized Variational AutoEncoder)
    architecture to learn semantic IDs for POIs and users, then predicts the
    next location based on the quantized representations.

    Architecture:
    1. Embedding Layer: Maps POI and user indices to dense vectors
    2. Encoder: MLP that compresses embeddings to latent space
    3. Residual VQ: Multi-layer vector quantization for semantic IDs
    4. Decoder: MLP that reconstructs the input (for auxiliary loss)
    5. Prediction Head: Linear layer for next POI prediction

    The model combines:
    - Reconstruction loss (MSE/L1): Ensures good representation learning
    - Quantization loss (cosine-based): Ensures codebook utilization
    - Prediction loss (CrossEntropy): Ensures good prediction performance
    """

    def __init__(self, config, data_feature):
        super(GNPRSID, self).__init__(config, data_feature)

        # Device
        self.device = config.get('device', 'cpu')

        # Data dimensions from data_feature
        self.loc_size = data_feature.get('loc_size', 1000)
        self.uid_size = data_feature.get('uid_size', 100)
        self.loc_pad = data_feature.get('loc_pad', 0)

        # Model hyperparameters from config
        self.loc_emb_size = config.get('loc_emb_size', 128)
        self.uid_emb_size = config.get('uid_emb_size', 64)
        self.encoder_layers = config.get('encoder_layers', [512, 256, 128])
        self.e_dim = config.get('e_dim', 64)
        self.num_codebooks = config.get('num_codebooks', 64)
        self.num_rq_layers = config.get('num_rq_layers', 3)
        self.dropout_prob = config.get('dropout_prob', 0.1)
        self.use_bn = config.get('use_bn', False)
        self.loss_type = config.get('loss_type', 'mse')
        self.quant_loss_weight = config.get('quant_loss_weight', 0.25)
        self.beta = config.get('beta', 0.25)
        self.kmeans_init = config.get('kmeans_init', False)
        self.kmeans_iters = config.get('kmeans_iters', 100)
        self.sk_epsilon = config.get('sk_epsilon', 0.05)
        self.sk_iters = config.get('sk_iters', 100)
        self.use_ema = config.get('use_ema', True)
        self.ema_decay = config.get('ema_decay', 0.95)
        self.use_linear = config.get('use_linear', 0)

        # Loss weights
        self.pred_loss_weight = config.get('pred_loss_weight', 1.0)
        self.recon_loss_weight = config.get('recon_loss_weight', 0.1)

        # Evaluation method
        self.evaluate_method = config.get('evaluate_method', 'popularity')

        # Calculate input dimension for encoder
        self.in_dim = self.loc_emb_size + self.uid_emb_size

        # Build model components
        self._build_model()

    def _build_model(self):
        """Build all model components."""
        # Embedding layers
        self.loc_embedding = nn.Embedding(
            self.loc_size, self.loc_emb_size, padding_idx=self.loc_pad
        )
        self.uid_embedding = nn.Embedding(
            self.uid_size, self.uid_emb_size
        )

        # Encoder: maps input to latent space
        self.encode_layer_dims = [self.in_dim] + self.encoder_layers + [self.e_dim]
        self.encoder = MLPLayers(
            layers=self.encode_layer_dims,
            dropout=self.dropout_prob,
            bn=self.use_bn
        )

        # Residual Vector Quantizer
        num_emb_list = [self.num_codebooks] * self.num_rq_layers
        sk_epsilons = [self.sk_epsilon] * self.num_rq_layers
        self.rq = ResidualVectorQuantizer(
            num_emb_list, self.e_dim,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=sk_epsilons,
            sk_iters=self.sk_iters,
            use_linear=self.use_linear,
            use_ema=self.use_ema,
            ema_decay=self.ema_decay
        )

        # Decoder: reconstructs input from quantized representation
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(
            layers=self.decode_layer_dims,
            dropout=self.dropout_prob,
            bn=self.use_bn
        )

        # Prediction head: predicts next location from quantized codes
        self.prediction_head = nn.Sequential(
            nn.Linear(self.e_dim, self.e_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.e_dim * 2, self.loc_size)
        )

        # Loss criterion for prediction
        self.criterion = nn.CrossEntropyLoss()

    def _create_input_embedding(self, batch):
        """
        Create input embeddings from batch data.

        Args:
            batch: Dictionary containing:
                - 'current_loc': (batch_size, seq_len) or (batch_size,) - POI indices
                - 'uid': (batch_size,) or (batch_size, 1) - user indices

        Returns:
            input_embed: (batch_size, in_dim) or (batch_size, seq_len, in_dim)
        """
        current_loc = batch['current_loc']

        # Handle user index
        if 'uid' in batch.data:
            uid = batch['uid']
        else:
            # Create dummy user indices if not provided
            if current_loc.dim() == 1:
                uid = torch.zeros(current_loc.size(0), dtype=torch.long, device=current_loc.device)
            else:
                uid = torch.zeros(current_loc.size(0), dtype=torch.long, device=current_loc.device)

        # Get embeddings
        if current_loc.dim() == 2:
            # Sequence input: (batch_size, seq_len)
            batch_size, seq_len = current_loc.size()

            # Location embedding: (batch_size, seq_len, loc_emb_size)
            loc_emb = self.loc_embedding(current_loc)

            # User embedding: (batch_size, uid_emb_size) -> (batch_size, seq_len, uid_emb_size)
            if uid.dim() == 1:
                uid_emb = self.uid_embedding(uid).unsqueeze(1).expand(-1, seq_len, -1)
            else:
                uid_emb = self.uid_embedding(uid.squeeze(-1)).unsqueeze(1).expand(-1, seq_len, -1)

            # Concatenate: (batch_size, seq_len, in_dim)
            input_embed = torch.cat([loc_emb, uid_emb], dim=-1)

            # Use last position for prediction
            input_embed = input_embed[:, -1, :]  # (batch_size, in_dim)
        else:
            # Single location input: (batch_size,)
            batch_size = current_loc.size(0)

            # Location embedding: (batch_size, loc_emb_size)
            loc_emb = self.loc_embedding(current_loc)

            # User embedding: (batch_size, uid_emb_size)
            if uid.dim() > 1:
                uid = uid.squeeze(-1)
            uid_emb = self.uid_embedding(uid)

            # Concatenate: (batch_size, in_dim)
            input_embed = torch.cat([loc_emb, uid_emb], dim=-1)

        return input_embed

    def forward(self, batch, use_sk=True):
        """
        Forward pass through the CRQVAE model.

        Args:
            batch: Dictionary containing input data
            use_sk: Whether to use Sinkhorn algorithm for quantization

        Returns:
            out: Reconstructed input embedding
            rq_loss: Quantization loss
            codes: Tuple of (indices, scalars) from residual quantization
            x_q: Quantized latent representation
        """
        # Create input embedding
        x = self._create_input_embedding(batch)

        # Encode
        z = self.encoder(x)

        # Quantize with residual VQ
        x_q, rq_loss, codes = self.rq(z, use_sk=use_sk)

        # Decode for reconstruction
        out = self.decoder(x_q)

        return out, rq_loss, codes, x_q

    def predict(self, batch):
        """
        Predict next POI location.

        Args:
            batch: Dictionary containing input data

        Returns:
            torch.Tensor: POI prediction scores (batch_size, loc_size)
        """
        # Forward pass
        out, rq_loss, codes, x_q = self.forward(batch, use_sk=False)

        # Predict next location from quantized representation
        logits = self.prediction_head(x_q)  # (batch_size, loc_size)

        # Apply log softmax for scoring
        score = F.log_softmax(logits, dim=-1)

        if self.evaluate_method == 'sample':
            # Build pos_neg_index for sampled evaluation
            if 'neg_loc' in batch.data:
                pos_neg_index = torch.cat((batch['target'].unsqueeze(1), batch['neg_loc']), dim=1)
                score = torch.gather(score, 1, pos_neg_index)

        return score

    def calculate_loss(self, batch):
        """
        Calculate combined loss for training.

        The total loss combines:
        1. Prediction loss: CrossEntropy for next POI prediction
        2. Reconstruction loss: MSE/L1 for input reconstruction
        3. Quantization loss: Cosine-based commitment loss

        Args:
            batch: Dictionary containing:
                - 'current_loc': Input POI indices
                - 'uid': User indices
                - 'target': Target POI index

        Returns:
            torch.Tensor: Total loss (scalar)
        """
        # Forward pass with Sinkhorn during training
        out, rq_loss, codes, x_q = self.forward(batch, use_sk=True)

        # Get input embedding for reconstruction loss
        input_embed = self._create_input_embedding(batch)

        # Reconstruction loss
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, input_embed, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, input_embed, reduction='mean')
        else:
            raise ValueError(f'Incompatible loss type: {self.loss_type}')

        # Prediction loss
        logits = self.prediction_head(x_q)  # (batch_size, loc_size)
        target = batch['target']
        loss_pred = self.criterion(logits, target)

        # Combined loss
        total_loss = (
            self.pred_loss_weight * loss_pred +
            self.recon_loss_weight * loss_recon +
            self.quant_loss_weight * rq_loss
        )

        return total_loss

    @torch.no_grad()
    def get_semantic_ids(self, batch, use_sk=False):
        """
        Get semantic IDs (quantized code indices) for input POIs.

        Args:
            batch: Dictionary containing input data
            use_sk: Whether to use Sinkhorn algorithm

        Returns:
            x_q: Quantized representation (batch_size, e_dim)
            indices: Code indices (batch_size, num_rq_layers)
        """
        x = self._create_input_embedding(batch)
        z = self.encoder(x)
        x_q, _, (indices, scalars) = self.rq(z, use_sk=use_sk)
        return x_q.cpu(), indices.cpu()
