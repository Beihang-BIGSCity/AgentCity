"""
PRME (Personalized Ranking Metric Embedding) Model for LibCity Framework

This is a PyTorch adaptation of the PRME model originally implemented in Cython.

Reference:
    Feng, Shanshan, et al. "Personalized Ranking Metric Embedding for Next New POI Recommendation."
    IJCAI 2015.

Original implementation: https://github.com/flaviovdf/prme (Cython-based)

Key adaptations for LibCity:
    - Converted Cython/NumPy implementation to PyTorch with nn.Embedding
    - Replaced manual SGD with PyTorch autograd
    - Adapted data format from (user, src, dest, dwell_time) to LibCity batch format
    - Implemented BPR pairwise ranking loss with negative sampling
    - Distance metric combines personalized and geographic components

Architecture:
    - XG_ok: Geographic embedding for locations (num_locs x embedding_dim)
    - XP_ok: Personalized location embedding (num_locs x embedding_dim)
    - XP_hk: User embedding (num_users x embedding_dim)

Distance formula:
    dist(h, s, d) = alpha * ||XP_ok[d] - XP_hk[h]||^2 + (1-alpha) * ||XG_ok[d] - XG_ok[s]||^2

Loss (BPR-style):
    L = -log(sigmoid(dist(neg) - dist(pos))) + regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model.abstract_model import AbstractModel


class PRME(AbstractModel):
    """
    PRME: Personalized Ranking Metric Embedding for Next POI Recommendation.

    This model uses three embedding matrices to compute personalized and geographic
    distances for ranking candidate locations. It employs BPR-style pairwise
    ranking loss with negative sampling.
    """

    def __init__(self, config, data_feature):
        """
        Initialize the PRME model.

        Args:
            config: Configuration dictionary containing model hyperparameters
            data_feature: Data features dictionary containing dataset information
        """
        super(PRME, self).__init__(config, data_feature)

        # Device configuration
        self.device = config.get('device', 'cpu')

        # Data dimensions from data_feature
        self.uid_size = data_feature.get('uid_size', 1)
        self.loc_size = data_feature.get('loc_size', 1)
        self.loc_pad = data_feature.get('loc_pad', 0)

        # Model hyperparameters from config
        self.embedding_dim = config.get('embedding_dim', 50)
        self.alpha = config.get('alpha', 0.5)  # Balance between personalized and geographic
        self.tau = config.get('tau', 3.0)  # Time threshold in hours for cold-start
        self.num_negative = config.get('num_negative', 10)  # Number of negative samples
        self.regularization = config.get('regularization', 0.03)

        # Embedding layers
        # XG_ok: Geographic embedding for locations
        self.geo_embedding = nn.Embedding(
            self.loc_size,
            self.embedding_dim,
            padding_idx=self.loc_pad
        )

        # XP_ok: Personalized location embedding
        self.loc_embedding = nn.Embedding(
            self.loc_size,
            self.embedding_dim,
            padding_idx=self.loc_pad
        )

        # XP_hk: User embedding
        self.user_embedding = nn.Embedding(
            self.uid_size,
            self.embedding_dim
        )

        # Initialize weights with small normal distribution (matching original)
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights with small normal distribution."""
        nn.init.normal_(self.geo_embedding.weight.data, mean=0.0, std=0.01)
        nn.init.normal_(self.loc_embedding.weight.data, mean=0.0, std=0.01)
        nn.init.normal_(self.user_embedding.weight.data, mean=0.0, std=0.01)

        # Zero out padding index embeddings
        if self.loc_pad is not None:
            self.geo_embedding.weight.data[self.loc_pad].zero_()
            self.loc_embedding.weight.data[self.loc_pad].zero_()

    def compute_distance(self, user_ids, src_locs, dest_locs, alpha=None):
        """
        Compute the PRME distance metric combining personalized and geographic components.

        Args:
            user_ids: User IDs tensor (batch_size,)
            src_locs: Source location IDs tensor (batch_size,)
            dest_locs: Destination location IDs tensor (batch_size,) or (batch_size, num_candidates)
            alpha: Optional alpha value (uses self.alpha if not provided)

        Returns:
            Distance tensor of shape matching dest_locs
        """
        if alpha is None:
            alpha = self.alpha

        # Get embeddings
        user_emb = self.user_embedding(user_ids)  # (batch_size, embedding_dim)
        src_geo_emb = self.geo_embedding(src_locs)  # (batch_size, embedding_dim)

        # Handle both single destination and multiple candidates
        if dest_locs.dim() == 1:
            dest_loc_emb = self.loc_embedding(dest_locs)  # (batch_size, embedding_dim)
            dest_geo_emb = self.geo_embedding(dest_locs)  # (batch_size, embedding_dim)

            # Personalized distance: ||XP_ok[d] - XP_hk[h]||^2
            personalized_dist = torch.sum((dest_loc_emb - user_emb) ** 2, dim=-1)

            # Geographic distance: ||XG_ok[d] - XG_ok[s]||^2
            geographic_dist = torch.sum((dest_geo_emb - src_geo_emb) ** 2, dim=-1)
        else:
            # dest_locs has shape (batch_size, num_candidates)
            dest_loc_emb = self.loc_embedding(dest_locs)  # (batch_size, num_candidates, embedding_dim)
            dest_geo_emb = self.geo_embedding(dest_locs)  # (batch_size, num_candidates, embedding_dim)

            # Expand user and src embeddings for broadcasting
            user_emb = user_emb.unsqueeze(1)  # (batch_size, 1, embedding_dim)
            src_geo_emb = src_geo_emb.unsqueeze(1)  # (batch_size, 1, embedding_dim)

            # Personalized distance
            personalized_dist = torch.sum((dest_loc_emb - user_emb) ** 2, dim=-1)

            # Geographic distance
            geographic_dist = torch.sum((dest_geo_emb - src_geo_emb) ** 2, dim=-1)

        # Combined distance
        distance = alpha * personalized_dist + (1 - alpha) * geographic_dist

        return distance

    def forward(self, batch):
        """
        Forward pass computing scores for all locations.

        In PRME, lower distance = higher preference, so we return negative distances
        as scores for compatibility with LibCity's evaluation (higher score = better).

        Args:
            batch: Dictionary containing 'uid', 'current_loc', and optionally 'current_tim'

        Returns:
            Scores tensor of shape (batch_size, loc_size) where higher = better
        """
        # Extract user IDs
        user_ids = batch['uid']  # (batch_size,)

        # Get the last location in the trajectory as source
        current_loc = batch['current_loc']  # (batch_size, seq_len)
        if hasattr(batch, 'get_origin_len'):
            origin_len = batch.get_origin_len('current_loc')
            last_loc_index = torch.LongTensor(origin_len) - 1
            last_loc_index = last_loc_index.to(self.device)
        else:
            # Fallback: use last non-padding position or just last position
            last_loc_index = (current_loc != self.loc_pad).sum(dim=1) - 1
            last_loc_index = last_loc_index.clamp(min=0)

        src_locs = torch.gather(current_loc, 1, last_loc_index.unsqueeze(1)).squeeze(1)

        batch_size = user_ids.shape[0]

        # Create all location candidates
        all_locs = torch.arange(self.loc_size, device=self.device)
        all_locs = all_locs.unsqueeze(0).expand(batch_size, -1)  # (batch_size, loc_size)

        # Compute distances to all locations
        distances = self.compute_distance(user_ids, src_locs, all_locs)

        # Return negative distances as scores (lower distance = higher score)
        scores = -distances

        return scores

    def predict(self, batch):
        """
        Make predictions for the batch.

        Args:
            batch: Input batch dictionary

        Returns:
            Prediction scores tensor
        """
        return self.forward(batch)

    def _sample_negatives(self, batch_size, positive_locs, num_negatives):
        """
        Sample negative locations that are different from positive locations.

        Args:
            batch_size: Number of samples in batch
            positive_locs: Positive location tensor (batch_size,)
            num_negatives: Number of negative samples per positive

        Returns:
            Negative location tensor (batch_size, num_negatives)
        """
        # Random sampling excluding positive locations
        # For efficiency, we use random sampling and rely on probability
        # that collision is rare when loc_size is large
        neg_locs = torch.randint(
            0, self.loc_size,
            (batch_size, num_negatives),
            device=self.device
        )

        # Resample any that match positive (simple rejection)
        positive_expanded = positive_locs.unsqueeze(1).expand(-1, num_negatives)
        mask = (neg_locs == positive_expanded)

        # Replace collisions with different random samples
        while mask.any():
            replacement = torch.randint(
                0, self.loc_size,
                (mask.sum().item(),),
                device=self.device
            )
            neg_locs[mask] = replacement
            mask = (neg_locs == positive_expanded)

        return neg_locs

    def calculate_loss(self, batch):
        """
        Calculate BPR pairwise ranking loss with negative sampling.

        The loss encourages the model to rank positive (actual next location)
        higher than negative samples (random locations).

        Loss = -log(sigmoid(dist_neg - dist_pos)) + regularization

        Args:
            batch: Input batch dictionary

        Returns:
            Loss tensor
        """
        # Extract user IDs
        user_ids = batch['uid']  # (batch_size,)

        # Get the last location as source
        current_loc = batch['current_loc']  # (batch_size, seq_len)
        if hasattr(batch, 'get_origin_len'):
            origin_len = batch.get_origin_len('current_loc')
            last_loc_index = torch.LongTensor(origin_len) - 1
            last_loc_index = last_loc_index.to(self.device)
        else:
            last_loc_index = (current_loc != self.loc_pad).sum(dim=1) - 1
            last_loc_index = last_loc_index.clamp(min=0)

        src_locs = torch.gather(current_loc, 1, last_loc_index.unsqueeze(1)).squeeze(1)

        # Get target (positive) locations
        target_locs = batch['target']  # (batch_size,)

        batch_size = user_ids.shape[0]

        # Sample negative locations
        neg_locs = self._sample_negatives(batch_size, target_locs, self.num_negative)

        # Compute distances for positive samples
        pos_dist = self.compute_distance(user_ids, src_locs, target_locs)  # (batch_size,)

        # Compute distances for negative samples
        neg_dist = self.compute_distance(user_ids, src_locs, neg_locs)  # (batch_size, num_negative)

        # Expand positive distances for broadcasting
        pos_dist_expanded = pos_dist.unsqueeze(1)  # (batch_size, 1)

        # BPR loss: -log(sigmoid(dist_neg - dist_pos))
        # We want dist_neg > dist_pos (negative samples should be farther)
        diff = neg_dist - pos_dist_expanded  # (batch_size, num_negative)
        bpr_loss = -F.logsigmoid(diff).mean()

        # L2 regularization on embeddings
        reg_loss = 0.0
        if self.regularization > 0:
            # Regularize used embeddings
            user_emb = self.user_embedding(user_ids)
            src_geo_emb = self.geo_embedding(src_locs)
            target_loc_emb = self.loc_embedding(target_locs)
            target_geo_emb = self.geo_embedding(target_locs)
            neg_loc_emb = self.loc_embedding(neg_locs)
            neg_geo_emb = self.geo_embedding(neg_locs)

            reg_loss = self.regularization * (
                torch.sum(user_emb ** 2) +
                torch.sum(src_geo_emb ** 2) +
                torch.sum(target_loc_emb ** 2) +
                torch.sum(target_geo_emb ** 2) +
                torch.sum(neg_loc_emb ** 2) +
                torch.sum(neg_geo_emb ** 2)
            ) / batch_size

        total_loss = bpr_loss + reg_loss

        return total_loss
