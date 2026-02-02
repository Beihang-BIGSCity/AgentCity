"""
GNPRSID: Generative Next POI Recommendation with Semantic ID

This model is adapted from the original GNPRSID implementation:
https://github.com/... (Original repository)

The model uses a Cosine Residual Quantized Variational Autoencoder (CRQVAE) to learn
semantic IDs for POIs. The CRQVAE encodes POI embeddings into discrete codes using
residual vector quantization with cosine similarity-based clustering.

Key Components:
1. MLPLayers - Multi-layer perceptron encoder/decoder
2. CosineVectorQuantizer - Vector quantizer using cosine similarity with EMA updates
3. ResidualVectorQuantizer - Residual vector quantization using multiple codebooks
4. CRQVAE - Main autoencoder model for learning semantic IDs

Adaptations for LibCity:
- Wrapped CRQVAE in GNPRSID class inheriting from AbstractModel
- Added POI embedding layer to convert location IDs to embeddings
- Added sequence encoder (Transformer) to process trajectories
- Added prediction head for next POI prediction
- Implemented predict() and calculate_loss() methods following LibCity conventions
- Adapted batch input format to LibCity's trajectory batch dictionary

The model can operate in two modes:
1. Full mode: Uses trajectory sequences and predicts next POI
2. Embedding mode: Uses pre-computed POI embeddings for SID training
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from torch.nn.utils.rnn import pad_sequence

from libcity.model.abstract_model import AbstractModel

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def activation_layer(activation_name="relu", emb_dim=None):
    """Create activation layer based on name.

    Args:
        activation_name: Name of activation function
        emb_dim: Embedding dimension (unused, kept for compatibility)

    Returns:
        nn.Module: Activation layer
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
        elif activation_name.lower() == "gelu":
            activation = nn.GELU()
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )

    return activation


def kmeans(samples, num_clusters, num_iters=10):
    """K-means clustering for codebook initialization.

    Args:
        samples: Input samples tensor (B, D)
        num_clusters: Number of clusters
        num_iters: Maximum number of iterations

    Returns:
        torch.Tensor: Cluster centers (num_clusters, D)
    """
    if not SKLEARN_AVAILABLE:
        # Fallback to random initialization if sklearn is not available
        indices = torch.randperm(samples.shape[0])[:num_clusters]
        return samples[indices].clone()

    B, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    x = samples.cpu().detach().numpy()

    cluster = KMeans(n_clusters=num_clusters, max_iter=num_iters, n_init=1).fit(x)

    centers = cluster.cluster_centers_
    tensor_centers = torch.from_numpy(centers).to(device).to(dtype)

    return tensor_centers


@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    """Sinkhorn-Knopp algorithm for optimal transport.

    Args:
        distances: Distance matrix (B, K)
        epsilon: Temperature parameter
        sinkhorn_iterations: Number of iterations

    Returns:
        torch.Tensor: Assignment probabilities (B, K)
    """
    distances = torch.clamp(distances, min=-1e3, max=1e3)
    Q = torch.exp(-distances / (epsilon + 1e-8))
    Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-8)
    for _ in range(sinkhorn_iterations):
        Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-8)
        Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-8)
    return Q


class MLPLayers(nn.Module):
    """Multi-layer perceptron with optional batch normalization and dropout.

    Args:
        layers: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
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
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


class CosineVectorQuantizer(nn.Module):
    """Cosine similarity-based vector quantizer with EMA updates.

    This quantizer uses cosine similarity to find the nearest codebook entry
    and supports projection quantization, EMA updates, and Sinkhorn regularization.

    Args:
        n_e: Number of codebook entries
        e_dim: Dimension of codebook entries
        beta: Commitment loss weight
        kmeans_init: Whether to use kmeans initialization
        kmeans_iters: Number of kmeans iterations
        sk_epsilon: Sinkhorn epsilon (None to disable)
        sk_iters: Sinkhorn iterations
        use_linear: Whether to use linear projection for codebook
        use_ema: Whether to use EMA updates
        ema_decay: EMA decay rate
        ema_epsilon: EMA epsilon for numerical stability
    """

    def __init__(self, n_e, e_dim,
                 beta=0.25, kmeans_init=False, kmeans_iters=10,
                 sk_epsilon=None, sk_iters=100, use_linear=0,
                 use_ema=True, ema_decay=0.95, ema_epsilon=1e-5):
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
        """Get the codebook with optional linear projection."""
        codebook = self.embedding.weight
        if self.use_linear:
            codebook = self.codebook_projection(codebook)
        return codebook

    @torch.no_grad()
    def init_emb(self, data):
        """Initialize codebook using kmeans."""
        centers = kmeans(data, self.n_e, self.kmeans_iters)
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    def forward(self, x, use_sk=True):
        """Forward pass through the vector quantizer.

        Args:
            x: Input tensor (B, D)
            use_sk: Whether to use Sinkhorn regularization

        Returns:
            x_q: Quantized output (B, D)
            loss: Quantization loss
            indices: Codebook indices (B,)
            scalar: Projection scalars (B,)
        """
        B, D = x.shape
        latent = x.view(B, D)

        if not self.initted and self.training:
            self.init_emb(latent)

        codebook = self.get_codebook()  # [K, D]

        # Cosine similarity clustering
        latent_norm = F.normalize(latent, dim=1)
        codebook_norm = F.normalize(codebook, dim=1)
        sim = torch.matmul(latent_norm, codebook_norm.t())  # [B, K]
        distances = 1 - sim  # Lower is better

        if use_sk and self.sk_epsilon is not None and self.sk_epsilon > 0:
            d_soft = self.center_distance_for_constraint(distances)
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

        # Loss computation
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

        indices = indices.view(B)  # [B]
        scalar = scalar.view(B)  # [B]

        return x_q, loss, indices, scalar

    @staticmethod
    def center_distance_for_constraint(distances):
        """Center distances for Sinkhorn constraint."""
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        centered_distances = (distances - middle) / amplitude
        return centered_distances


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer using multiple codebooks.

    This applies vector quantization residually, where each quantizer
    operates on the residual from the previous quantizer.

    Args:
        n_e_list: List of codebook sizes for each level
        e_dim: Embedding dimension
        sk_epsilons: List of Sinkhorn epsilons for each level
        beta: Commitment loss weight
        kmeans_init: Whether to use kmeans initialization
        kmeans_iters: Kmeans iterations
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
        self.sk_epsilons = sk_epsilons if sk_epsilons is not None else [0.1] * len(n_e_list)
        self.sk_iters = sk_iters
        self.use_linear = use_linear

        self.vq_layers = nn.ModuleList([
            CosineVectorQuantizer(n_e, e_dim,
                                  beta=self.beta,
                                  kmeans_init=self.kmeans_init,
                                  kmeans_iters=self.kmeans_iters,
                                  sk_epsilon=sk_epsilon,
                                  sk_iters=sk_iters,
                                  use_linear=use_linear)
            for n_e, sk_epsilon in zip(n_e_list, self.sk_epsilons)
        ])

    def forward(self, x, use_sk=True):
        """Forward pass through residual quantization.

        Args:
            x: Input tensor (B, D) or (B, T, D)
            use_sk: Whether to use Sinkhorn regularization

        Returns:
            x_q: Quantized output
            mean_loss: Mean quantization loss
            (all_indices, all_scalars): Indices and scalars for each level
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


class CRQVAE(nn.Module):
    """Cosine Residual Quantized Variational Autoencoder.

    This model encodes inputs through an MLP encoder, quantizes the latent
    representation using residual vector quantization, and reconstructs
    through an MLP decoder.

    Args:
        in_dim: Input dimension
        num_emb_list: List of codebook sizes for each RQ level
        e_dim: Latent embedding dimension
        layers: List of hidden layer dimensions for encoder
        dropout_prob: Dropout probability
        bn: Whether to use batch normalization
        loss_type: Reconstruction loss type ('mse' or 'l1')
        quant_loss_weight: Weight for quantization loss
        beta: Commitment loss weight
        kmeans_init: Whether to use kmeans initialization
        kmeans_iters: Kmeans iterations
        sk_epsilons: List of Sinkhorn epsilons
        sk_iters: Sinkhorn iterations
        use_linear: Whether to use linear projection
    """

    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=0.25,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=None,
                 sk_iters=100,
                 use_linear=0):
        super(CRQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list if num_emb_list is not None else [64, 64, 64]
        self.e_dim = e_dim
        self.layers = layers if layers is not None else [512, 256, 128]
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons if sk_epsilons is not None else [0.1] * len(self.num_emb_list)
        self.sk_iters = sk_iters
        self.use_linear = use_linear

        # Encoder
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob, bn=self.bn)

        # Residual Vector Quantizer
        self.rq = ResidualVectorQuantizer(
            self.num_emb_list, e_dim,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
            use_linear=self.use_linear
        )

        # Decoder
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                 dropout=self.dropout_prob, bn=self.bn)

    def forward(self, x, use_sk=True):
        """Forward pass.

        Args:
            x: Input tensor (B, in_dim)
            use_sk: Whether to use Sinkhorn regularization

        Returns:
            out: Reconstructed output
            rq_loss: Quantization loss
            codes: (indices, scalars) from RQ
        """
        x = self.encoder(x)
        x_q, rq_loss, codes = self.rq(x, use_sk=use_sk)
        out = self.decoder(x_q)
        return out, rq_loss, codes

    def compute_loss(self, quant_loss, out, xs=None):
        """Compute total loss.

        Args:
            quant_loss: Quantization loss from RQ
            out: Reconstructed output
            xs: Original input for reconstruction loss

        Returns:
            loss_total: Total loss
            quant_loss: Quantization loss
            loss_recon: Reconstruction loss
        """
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, quant_loss, loss_recon

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        """Get quantization indices for inputs.

        Args:
            xs: Input tensor
            use_sk: Whether to use Sinkhorn

        Returns:
            x_q: Quantized embeddings
            indices: Codebook indices
        """
        x_e = self.encoder(xs)
        x_q, _, (indices, scalars) = self.rq(x_e, use_sk=use_sk)
        return x_q, indices

    def get_quantized(self, xs, use_sk=False):
        """Get quantized embeddings.

        Args:
            xs: Input tensor
            use_sk: Whether to use Sinkhorn

        Returns:
            x_q: Quantized latent embeddings
        """
        x_e = self.encoder(xs)
        x_q, _, _ = self.rq(x_e, use_sk=use_sk)
        return x_q


class PositionalEncoding(nn.Module):
    """Standard positional encoding for Transformer."""

    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerSequenceEncoder(nn.Module):
    """Transformer encoder for sequence modeling.

    Args:
        embed_size: Input embedding size
        num_pois: Number of POI locations for output
        nhead: Number of attention heads
        nhid: Hidden dimension in feedforward
        nlayers: Number of transformer layers
        dropout: Dropout probability
    """

    def __init__(self, embed_size, num_pois, nhead=4, nhid=256, nlayers=2, dropout=0.1):
        super(TransformerSequenceEncoder, self).__init__()
        from torch.nn import TransformerEncoder as TorchTransformerEncoder
        from torch.nn import TransformerEncoderLayer

        self.embed_size = embed_size
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TorchTransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(embed_size, num_pois)

    def generate_square_subsequent_mask(self, sz, device):
        """Generate causal attention mask."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask=None):
        """Forward pass.

        Args:
            src: Input sequence (batch, seq_len, embed_size)
            src_mask: Attention mask

        Returns:
            out: Output logits (batch, seq_len, num_pois)
        """
        src = src * np.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out = self.decoder(x)
        return out


class GNPRSID(AbstractModel):
    """
    GNPRSID: Generative Next POI Recommendation with Semantic ID

    This model combines a CRQVAE for learning semantic POI IDs with a
    Transformer-based sequence model for next POI prediction.

    The model works in two modes:
    1. Joint training: Learns POI embeddings and predictions together
    2. Pre-trained SID: Uses pre-trained CRQVAE for semantic IDs

    Args:
        config (dict): Configuration dictionary containing model hyperparameters
        data_feature (dict): Data features including vocab sizes

    Required config parameters:
        - loc_emb_size: Location embedding dimension (default: 128)
        - uid_emb_size: User embedding dimension (default: 64)
        - encoder_layers: List of encoder layer dimensions (default: [512, 256, 128])
        - e_dim: CRQVAE latent dimension (default: 64)
        - num_codebooks: Number of codebook entries per level (default: 64)
        - num_rq_layers: Number of residual quantization layers (default: 3)
        - dropout_prob: Dropout probability (default: 0.1)
        - use_bn: Whether to use batch normalization (default: True)
        - loss_type: Reconstruction loss type (default: 'mse')
        - quant_loss_weight: Quantization loss weight (default: 0.5)
        - beta: Commitment loss weight (default: 0.25)
        - kmeans_init: Use kmeans initialization (default: True)
        - kmeans_iters: Kmeans iterations (default: 100)
        - sk_epsilon: Sinkhorn epsilon (default: 0.1)
        - sk_iters: Sinkhorn iterations (default: 50)
        - use_ema: Use EMA updates (default: True)
        - use_linear: Use linear projection (default: 1)
        - pred_loss_weight: Prediction loss weight (default: 1.0)
        - recon_loss_weight: Reconstruction loss weight (default: 0.1)

    Required data_feature:
        - loc_size: Number of POI locations
        - uid_size: Number of users
        - loc_pad: Padding index for locations
    """

    def __init__(self, config, data_feature):
        super(GNPRSID, self).__init__(config, data_feature)

        self.device = config.get('device', 'cpu')

        # Data dimensions from data_feature
        self.num_pois = data_feature.get('loc_size', 1000)
        self.num_users = data_feature.get('uid_size', 100)
        self.loc_pad = data_feature.get('loc_pad', 0)

        # Model hyperparameters from config
        self.loc_emb_size = config.get('loc_emb_size', 128)
        self.uid_emb_size = config.get('uid_emb_size', 64)
        self.encoder_layers = config.get('encoder_layers', [512, 256, 128])
        self.e_dim = config.get('e_dim', 64)
        self.num_codebooks = config.get('num_codebooks', 64)
        self.num_rq_layers = config.get('num_rq_layers', 3)
        self.dropout_prob = config.get('dropout_prob', 0.1)
        self.use_bn = config.get('use_bn', True)
        self.loss_type = config.get('loss_type', 'mse')
        self.quant_loss_weight = config.get('quant_loss_weight', 0.5)
        self.beta = config.get('beta', 0.25)
        self.kmeans_init = config.get('kmeans_init', True)
        self.kmeans_iters = config.get('kmeans_iters', 100)
        self.sk_epsilon = config.get('sk_epsilon', 0.1)
        self.sk_iters = config.get('sk_iters', 50)
        self.use_ema = config.get('use_ema', True)
        self.ema_decay = config.get('ema_decay', 0.95)
        self.use_linear = config.get('use_linear', 1)
        self.pred_loss_weight = config.get('pred_loss_weight', 1.0)
        self.recon_loss_weight = config.get('recon_loss_weight', 0.1)

        # Transformer config
        self.transformer_nhead = config.get('transformer_nhead', 4)
        self.transformer_nhid = config.get('transformer_nhid', 256)
        self.transformer_nlayers = config.get('transformer_nlayers', 2)
        self.transformer_dropout = config.get('transformer_dropout', 0.1)

        # Evaluation method
        self.evaluate_method = config.get('evaluate_method', 'all')

        # Build model components
        self._build_model()

    def _build_model(self):
        """Build all model components."""

        # POI Embedding layer
        self.poi_embedding = nn.Embedding(
            self.num_pois, self.loc_emb_size,
            padding_idx=self.loc_pad
        )

        # User embedding layer
        self.user_embedding = nn.Embedding(
            self.num_users, self.uid_emb_size
        )

        # Build num_emb_list for RQ
        num_emb_list = [self.num_codebooks] * self.num_rq_layers
        sk_epsilons = [self.sk_epsilon] * self.num_rq_layers

        # CRQVAE for learning semantic IDs
        self.crqvae = CRQVAE(
            in_dim=self.loc_emb_size,
            num_emb_list=num_emb_list,
            e_dim=self.e_dim,
            layers=self.encoder_layers,
            dropout_prob=self.dropout_prob,
            bn=self.use_bn,
            loss_type=self.loss_type,
            quant_loss_weight=self.quant_loss_weight,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=sk_epsilons,
            sk_iters=self.sk_iters,
            use_linear=self.use_linear
        )

        # Sequence model input: quantized embedding + user embedding
        seq_input_dim = self.e_dim + self.uid_emb_size

        # Transformer sequence encoder for prediction
        self.seq_encoder = TransformerSequenceEncoder(
            embed_size=seq_input_dim,
            num_pois=self.num_pois,
            nhead=self.transformer_nhead,
            nhid=self.transformer_nhid,
            nlayers=self.transformer_nlayers,
            dropout=self.transformer_dropout
        )

        # Loss functions
        self.criterion_pred = nn.CrossEntropyLoss(ignore_index=self.loc_pad)

    def _get_poi_quantized(self, loc_indices, use_sk=False):
        """Get quantized POI embeddings.

        Args:
            loc_indices: Location indices (batch, seq_len)
            use_sk: Whether to use Sinkhorn

        Returns:
            Quantized embeddings (batch, seq_len, e_dim)
        """
        # Get raw POI embeddings
        poi_emb = self.poi_embedding(loc_indices)  # (batch, seq_len, loc_emb_size)

        batch_size, seq_len, emb_dim = poi_emb.shape

        # Flatten for CRQVAE processing
        poi_emb_flat = poi_emb.view(-1, emb_dim)  # (batch*seq_len, emb_dim)

        # Get quantized embeddings
        poi_quantized = self.crqvae.get_quantized(poi_emb_flat, use_sk=use_sk)

        # Reshape back
        poi_quantized = poi_quantized.view(batch_size, seq_len, -1)

        return poi_quantized

    def forward(self, batch):
        """
        Forward pass.

        Args:
            batch: LibCity Batch object containing trajectory data. Expected keys:
                - 'uid': User IDs (batch_size,)
                - 'current_loc': Location sequences (batch_size, seq_len)

        Returns:
            y_pred: POI prediction logits (batch_size, seq_len, num_pois)
            rq_loss: Quantization loss
            recon_loss: Reconstruction loss
            batch_seq_lens: Original sequence lengths
        """
        # Extract data from batch
        user_ids = batch['uid']
        loc_seq = batch['current_loc']

        # Get sequence lengths
        seq_lens = batch.get_origin_len('current_loc')

        # Convert to tensors if needed
        if not isinstance(loc_seq, torch.Tensor):
            loc_seq = torch.LongTensor(loc_seq)
        if not isinstance(user_ids, torch.Tensor):
            user_ids = torch.LongTensor(user_ids)

        loc_seq = loc_seq.to(self.device)
        user_ids = user_ids.to(self.device)

        if isinstance(seq_lens, torch.Tensor):
            seq_lens = seq_lens.tolist()

        batch_size = loc_seq.shape[0]
        max_seq_len = loc_seq.shape[1]

        # Get POI embeddings
        poi_emb = self.poi_embedding(loc_seq)  # (batch, seq_len, loc_emb_size)

        # Flatten for CRQVAE
        poi_emb_flat = poi_emb.view(-1, self.loc_emb_size)

        # Forward through CRQVAE
        recon, rq_loss, codes = self.crqvae(poi_emb_flat, use_sk=self.training)

        # Compute reconstruction loss
        recon_loss = F.mse_loss(recon, poi_emb_flat)

        # Get quantized embeddings
        poi_quantized_flat = self.crqvae.get_quantized(poi_emb_flat, use_sk=False)
        poi_quantized = poi_quantized_flat.view(batch_size, max_seq_len, -1)

        # Get user embeddings and expand to sequence length
        user_emb = self.user_embedding(user_ids)  # (batch, uid_emb_size)
        user_emb = user_emb.unsqueeze(1).expand(-1, max_seq_len, -1)  # (batch, seq_len, uid_emb_size)

        # Concatenate for sequence model input
        seq_input = torch.cat([poi_quantized, user_emb], dim=-1)  # (batch, seq_len, e_dim + uid_emb_size)

        # Generate attention mask
        src_mask = self.seq_encoder.generate_square_subsequent_mask(max_seq_len, self.device)

        # Forward through sequence encoder
        y_pred = self.seq_encoder(seq_input, src_mask)  # (batch, seq_len, num_pois)

        return y_pred, rq_loss, recon_loss, seq_lens

    def predict(self, batch):
        """
        Prediction method for LibCity.

        Args:
            batch: Input batch dictionary

        Returns:
            POI prediction scores for the last timestep of each sequence
        """
        y_pred, _, _, seq_lens = self.forward(batch)

        batch_size = y_pred.size(0)
        predictions = []

        for i in range(batch_size):
            # Get prediction for last valid timestep
            last_idx = seq_lens[i] - 1 if isinstance(seq_lens, list) else seq_lens[i].item() - 1
            last_idx = max(0, min(last_idx, y_pred.size(1) - 1))
            predictions.append(y_pred[i, last_idx, :])

        scores = torch.stack(predictions)  # (batch_size, num_pois)

        # Apply log softmax for evaluation
        scores = F.log_softmax(scores, dim=-1)

        if self.evaluate_method == 'sample':
            # Handle negative sampling evaluation
            if 'neg_loc' in batch:
                pos_neg_index = torch.cat(
                    (batch['target'].unsqueeze(1), batch['neg_loc']), dim=1
                )
                scores = torch.gather(scores, 1, pos_neg_index)

        return scores

    def calculate_loss(self, batch):
        """
        Calculate combined loss for training.

        Args:
            batch: LibCity Batch object containing:
                - trajectory data for forward pass
                - 'target': Target POI indices (batch_size,)

        Returns:
            Combined loss (prediction loss + weighted quantization loss + weighted reconstruction loss)
        """
        y_pred, rq_loss, recon_loss, seq_lens = self.forward(batch)

        # Get target POI
        target_poi = batch['target']

        if not isinstance(target_poi, torch.Tensor):
            target_poi = torch.LongTensor(target_poi)
        target_poi = target_poi.to(self.device)

        batch_size = y_pred.size(0)

        # Calculate prediction loss (using last timestep)
        pred_losses = []
        for i in range(batch_size):
            last_idx = seq_lens[i] - 1 if isinstance(seq_lens, list) else seq_lens[i].item() - 1
            last_idx = max(0, min(last_idx, y_pred.size(1) - 1))
            pred = y_pred[i, last_idx, :].unsqueeze(0)
            target = target_poi[i].unsqueeze(0)
            pred_losses.append(self.criterion_pred(pred, target))

        loss_pred = torch.stack(pred_losses).mean()

        # Combined loss
        total_loss = (self.pred_loss_weight * loss_pred +
                      self.quant_loss_weight * rq_loss +
                      self.recon_loss_weight * recon_loss)

        return total_loss

    @torch.no_grad()
    def get_semantic_ids(self, loc_indices):
        """Get semantic IDs for locations.

        Args:
            loc_indices: Location indices (batch,) or (batch, seq_len)

        Returns:
            Semantic ID codes (batch, num_rq_layers) or (batch, seq_len, num_rq_layers)
        """
        self.eval()

        if not isinstance(loc_indices, torch.Tensor):
            loc_indices = torch.LongTensor(loc_indices)
        loc_indices = loc_indices.to(self.device)

        # Get POI embeddings
        poi_emb = self.poi_embedding(loc_indices)

        original_shape = poi_emb.shape
        if poi_emb.ndim == 2:
            poi_emb = poi_emb.view(-1, self.loc_emb_size)
        else:
            batch_size, seq_len, _ = poi_emb.shape
            poi_emb = poi_emb.view(-1, self.loc_emb_size)

        # Get indices from CRQVAE
        _, indices = self.crqvae.get_indices(poi_emb, use_sk=False)

        if len(original_shape) == 2:
            return indices
        else:
            return indices.view(batch_size, seq_len, -1)

    def load_pretrained_crqvae(self, checkpoint_path):
        """Load pre-trained CRQVAE weights.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Filter to only CRQVAE keys
        crqvae_state = {k: v for k, v in state_dict.items()}
        self.crqvae.load_state_dict(crqvae_state, strict=False)
