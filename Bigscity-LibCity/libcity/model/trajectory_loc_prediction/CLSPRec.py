# coding: utf-8
"""
CLSPRec: Contrastive Learning for Sequential POI Recommendation

This module adapts the CLSPRec model from the original repository for LibCity framework.

Original paper: "Contrastive Learning with Short-Long Term Preferences for Next POI Recommendation"

Key adaptations made for LibCity:
1. Inherited from AbstractModel instead of nn.Module
2. Adapted data handling to use LibCity's batch dictionary format
3. Implemented predict() and calculate_loss() methods following LibCity conventions
4. Extracted hyperparameters from config dictionary instead of settings module
5. Handled device management via config.get('device')

Required data_feature keys:
- loc_size: Number of POI locations
- uid_size: Number of users
- cat_size: Number of categories
- tim_size: Number of time slots (hours typically 24)
- day_size: Number of day types (typically 7 for weekdays)

Required config parameters:
- f_embed_size: Feature embedding dimension (default: 60)
- num_encoder_layers: Number of transformer encoder layers (default: 1)
- num_lstm_layers: Number of LSTM layers (default: 1)
- num_heads: Number of attention heads (default: 1)
- forward_expansion: Feed-forward expansion ratio (default: 4)
- dropout_p: Dropout probability (default: 0.2)
- neg_weight: Weight for contrastive loss (default: 1.0)
- mask_prop: Proportion of features to mask (default: 0.1)
- enable_ssl: Enable self-supervised contrastive learning (default: True)
- enable_random_mask: Enable random feature masking (default: True)
- neg_sample_count: Number of negative samples (default: 5)
"""

import torch
from torch import nn
import torch.nn.functional as F
from libcity.model.abstract_model import AbstractModel


class CheckInEmbedding(nn.Module):
    """
    Multi-feature embedding layer for check-in data.
    Embeds POI, category, user, hour, and day features.
    """
    def __init__(self, f_embed_size, vocab_size):
        super().__init__()
        self.embed_size = f_embed_size
        self.poi_num = vocab_size["POI"]
        self.cat_num = vocab_size["cat"]
        self.user_num = vocab_size["user"]
        self.hour_num = vocab_size["hour"]
        self.day_num = vocab_size["day"]

        self.poi_embed = nn.Embedding(self.poi_num + 1, self.embed_size, padding_idx=self.poi_num)
        self.cat_embed = nn.Embedding(self.cat_num + 1, self.embed_size, padding_idx=self.cat_num)
        self.user_embed = nn.Embedding(self.user_num + 1, self.embed_size, padding_idx=self.user_num)
        self.hour_embed = nn.Embedding(self.hour_num + 1, self.embed_size, padding_idx=self.hour_num)
        self.day_embed = nn.Embedding(self.day_num + 1, self.embed_size, padding_idx=self.day_num)

    def forward(self, x):
        """
        Args:
            x: Tuple of (poi, cat, user, hour, day) tensors, each of shape (seq_len,) or (batch, seq_len)
        Returns:
            Concatenated embeddings of shape (seq_len, 5*embed_size) or (batch, seq_len, 5*embed_size)
        """
        poi_emb = self.poi_embed(x[0])
        cat_emb = self.cat_embed(x[1])
        user_emb = self.user_embed(x[2])
        hour_emb = self.hour_embed(x[3])
        day_emb = self.day_embed(x[4])

        return torch.cat((poi_emb, cat_emb, user_emb, hour_emb, day_emb), -1)


class SelfAttention(nn.Module):
    """Multi-head self-attention layer."""
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads

        assert (
            self.head_dim * self.heads == self.embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embed_size)

    def forward(self, values, keys, query):
        """
        Args:
            values: (seq_len, embed_size) or (batch, seq_len, embed_size)
            keys: (seq_len, embed_size) or (batch, seq_len, embed_size)
            query: (seq_len, embed_size) or (batch, seq_len, embed_size)
        """
        # Handle both 2D (seq_len, embed) and 3D (batch, seq_len, embed) inputs
        if values.dim() == 2:
            value_len, key_len, query_len = values.shape[0], keys.shape[0], query.shape[0]

            values = self.values(values)
            keys = self.keys(keys)
            queries = self.queries(query)

            values = values.reshape(value_len, self.heads, self.head_dim)
            keys = keys.reshape(key_len, self.heads, self.head_dim)
            queries = queries.reshape(query_len, self.heads, self.head_dim)

            energy = torch.einsum("qhd,khd->hqk", [queries, keys])
            attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)

            out = torch.einsum("hql,lhd->qhd", [attention, values]).reshape(
                query_len, self.heads * self.head_dim
            )
        else:
            # Batched version
            batch_size, seq_len, _ = values.shape

            values = self.values(values)
            keys = self.keys(keys)
            queries = self.queries(query)

            values = values.reshape(batch_size, seq_len, self.heads, self.head_dim)
            keys = keys.reshape(batch_size, -1, self.heads, self.head_dim)
            queries = queries.reshape(batch_size, -1, self.heads, self.head_dim)

            energy = torch.einsum("bqhd,bkhd->bhqk", [queries, keys])
            attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

            out = torch.einsum("bhql,blhd->bqhd", [attention, values]).reshape(
                batch_size, -1, self.heads * self.head_dim
            )

        out = self.fc_out(out)
        return out


class EncoderBlock(nn.Module):
    """Transformer encoder block with self-attention and feed-forward network."""
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(EncoderBlock, self).__init__()
        self.embed_size = embed_size
        self.attention = SelfAttention(self.embed_size, heads)
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, forward_expansion * self.embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * self.embed_size, self.embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class TransformerEncoder(nn.Module):
    """Transformer encoder for sequence encoding."""
    def __init__(
            self,
            embedding_layer,
            embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout,
    ):
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = embedding_layer
        self.add_module('embedding', self.embedding_layer)

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, feature_seq):
        """
        Args:
            feature_seq: Tuple of feature tensors (poi, cat, user, hour, day)
        Returns:
            Encoded sequence of shape (seq_len, embed_size) or (batch, seq_len, embed_size)
        """
        embedding = self.embedding_layer(feature_seq)
        out = self.dropout(embedding)

        for layer in self.layers:
            out = layer(out, out, out)

        return out


class Attention(nn.Module):
    """Attention mechanism for query and key with different dimensions."""
    def __init__(self, qdim, kdim):
        super().__init__()
        self.expansion = nn.Linear(qdim, kdim)

    def forward(self, query, key, value):
        """
        Args:
            query: (embed_size,) or (batch, embed_size)
            key: (seq_len, embed_size) or (batch, seq_len, embed_size)
            value: (seq_len, embed_size) or (batch, seq_len, embed_size)
        """
        q = self.expansion(query)

        if q.dim() == 1:
            # Single sample processing
            temp = torch.inner(q, key)
            weight = torch.softmax(temp, dim=0)
            weight = torch.unsqueeze(weight, 1)
            temp2 = torch.mul(value, weight)
            out = torch.sum(temp2, 0)
        else:
            # Batched processing
            # q: (batch, kdim), key: (batch, seq_len, kdim)
            temp = torch.bmm(key, q.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len)
            weight = torch.softmax(temp, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
            temp2 = torch.mul(value, weight)  # (batch, seq_len, kdim)
            out = torch.sum(temp2, dim=1)  # (batch, kdim)

        return out


class CLSPRec(AbstractModel):
    """
    CLSPRec: Contrastive Learning for Sequential POI Recommendation

    This model combines:
    1. Multi-feature embeddings (POI, Category, User, Hour, Day)
    2. Transformer encoder for long-term and short-term sequences
    3. LSTM for user enhancement
    4. Contrastive learning with InfoNCE loss
    5. Attention mechanism for final prediction

    The model supports:
    - Random feature masking for data augmentation
    - Self-supervised contrastive learning between long-term and short-term preferences
    - User-enhanced representation via LSTM
    """

    def __init__(self, config, data_feature):
        super(CLSPRec, self).__init__(config, data_feature)

        self.device = config.get('device', 'cpu')

        # Data dimensions from data_feature
        self.loc_size = data_feature.get('loc_size', 1000)
        self.uid_size = data_feature.get('uid_size', 100)
        self.cat_size = data_feature.get('cat_size', 50)
        self.tim_size = data_feature.get('tim_size', 24)  # hours
        self.day_size = data_feature.get('day_size', 7)  # weekdays

        # Build vocab_size dictionary for compatibility with original architecture
        self.vocab_size = {
            "POI": self.loc_size,
            "cat": self.cat_size,
            "user": self.uid_size,
            "hour": self.tim_size,
            "day": self.day_size
        }

        # Model hyperparameters from config
        self.f_embed_size = config.get('f_embed_size', 60)
        self.num_encoder_layers = config.get('num_encoder_layers', 1)
        self.num_lstm_layers = config.get('num_lstm_layers', 1)
        self.num_heads = config.get('num_heads', 1)
        self.forward_expansion = config.get('forward_expansion', 4)
        self.dropout_p = config.get('dropout_p', 0.2)

        # Contrastive learning parameters
        self.neg_weight = config.get('neg_weight', 1.0)
        self.mask_prop = config.get('mask_prop', 0.1)
        self.enable_ssl = config.get('enable_ssl', True)
        self.enable_random_mask = config.get('enable_random_mask', True)
        self.neg_sample_count = config.get('neg_sample_count', 5)

        # Total embedding size (5 features concatenated)
        self.total_embed_size = self.f_embed_size * 5

        # Build model layers
        self._build_model()

    def _build_model(self):
        """Build all model layers."""
        # Embedding layer
        self.embedding = CheckInEmbedding(
            self.f_embed_size,
            self.vocab_size
        )

        # Transformer encoder
        self.encoder = TransformerEncoder(
            self.embedding,
            self.total_embed_size,
            self.num_encoder_layers,
            self.num_heads,
            self.forward_expansion,
            self.dropout_p,
        )

        # LSTM for user enhancement
        self.lstm = nn.LSTM(
            input_size=self.total_embed_size,
            hidden_size=self.total_embed_size,
            num_layers=self.num_lstm_layers,
            dropout=0,
            batch_first=True
        )

        # Final attention layer
        self.final_attention = Attention(
            qdim=self.f_embed_size,
            kdim=self.total_embed_size
        )

        # Output layer
        self.out_linear = nn.Sequential(
            nn.Linear(self.total_embed_size, self.total_embed_size * self.forward_expansion),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.total_embed_size * self.forward_expansion, self.loc_size)
        )

        # Loss function
        self.loss_func = nn.CrossEntropyLoss()

        # User enhancement layers
        self.tryone_line2 = nn.Linear(self.total_embed_size, self.f_embed_size)
        self.enhance_val = nn.Parameter(torch.tensor(0.5))

    def feature_mask(self, features, mask_prop):
        """
        Apply random feature masking for data augmentation.

        Args:
            features: Dictionary containing feature tensors
            mask_prop: Proportion of positions to mask
        Returns:
            Masked features dictionary
        """
        if not self.enable_random_mask or not self.training:
            return features

        masked_features = {}
        for key, val in features.items():
            masked_features[key] = val.clone()

        seq_len = masked_features['current_loc'].shape[-1]
        if seq_len > 1:
            mask_count = max(1, int(mask_prop * seq_len))
            # Generate mask indices (avoid masking the last position which is for prediction)
            masked_index = torch.randperm(seq_len - 1, device=self.device)[:mask_count]

            # Apply masking using padding indices
            masked_features['current_loc'][:, masked_index] = self.vocab_size["POI"]
            if 'current_cat' in masked_features:
                masked_features['current_cat'][:, masked_index] = self.vocab_size["cat"]
            if 'current_hour' in masked_features:
                masked_features['current_hour'][:, masked_index] = self.vocab_size["hour"]
            if 'current_day' in masked_features:
                masked_features['current_day'][:, masked_index] = self.vocab_size["day"]

        return masked_features

    def ssl_loss(self, embedding_1, embedding_2, neg_embedding):
        """
        Compute self-supervised contrastive loss using InfoNCE.

        Args:
            embedding_1: Short-term sequence embedding
            embedding_2: Long-term sequence embedding
            neg_embedding: Negative sample embedding
        Returns:
            Contrastive loss scalar
        """
        def score(x1, x2):
            return torch.mean(torch.mul(x1, x2), dim=-1)

        pos = score(embedding_1, embedding_2)
        neg1 = score(embedding_1, neg_embedding)
        neg2 = score(embedding_2, neg_embedding)
        neg = (neg1 + neg2) / 2

        one = torch.ones_like(neg)
        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg))))

        return con_loss

    def _extract_features_from_batch(self, batch):
        """
        Extract feature tensors from LibCity batch format.

        Args:
            batch: LibCity Batch object containing trajectory data
        Returns:
            Dictionary of feature tensors
        """
        features = {}

        # Get current location (POI)
        if hasattr(batch, 'data'):
            features['current_loc'] = batch['current_loc']  # (batch, seq_len)
        else:
            features['current_loc'] = batch.get('current_loc', None)

        # Get user IDs
        if 'uid' in batch.data if hasattr(batch, 'data') else 'uid' in batch:
            uid = batch['uid']
            if uid.dim() == 1:
                uid = uid.unsqueeze(1).expand(-1, features['current_loc'].shape[1])
            features['uid'] = uid
        else:
            features['uid'] = torch.zeros_like(features['current_loc'])

        # Get time features
        if 'current_tim' in (batch.data if hasattr(batch, 'data') else batch):
            features['current_hour'] = batch['current_tim']
        else:
            features['current_hour'] = torch.zeros_like(features['current_loc'])

        # Get day features
        if 'current_day' in (batch.data if hasattr(batch, 'data') else batch):
            features['current_day'] = batch['current_day']
        else:
            features['current_day'] = torch.zeros_like(features['current_loc'])

        # Get category features
        if 'current_cat' in (batch.data if hasattr(batch, 'data') else batch):
            features['current_cat'] = batch['current_cat']
        else:
            features['current_cat'] = torch.zeros_like(features['current_loc'])

        return features

    def _get_feature_tuple(self, features, idx=None):
        """
        Convert features dictionary to tuple format for embedding layer.

        Args:
            features: Dictionary of feature tensors
            idx: Optional slice indices for selecting portion of sequence
        Returns:
            Tuple of (poi, cat, user, hour, day) tensors
        """
        if idx is not None:
            return (
                features['current_loc'][:, idx],
                features['current_cat'][:, idx],
                features['uid'][:, idx] if features['uid'].dim() > 1 else features['uid'],
                features['current_hour'][:, idx],
                features['current_day'][:, idx]
            )
        else:
            return (
                features['current_loc'],
                features['current_cat'],
                features['uid'] if features['uid'].dim() > 1 else features['uid'].unsqueeze(1).expand(-1, features['current_loc'].shape[1]),
                features['current_hour'],
                features['current_day']
            )

    def _encode_sequence(self, feature_tuple):
        """
        Encode a sequence using the transformer encoder.

        Args:
            feature_tuple: Tuple of (poi, cat, user, hour, day) tensors
        Returns:
            Encoded sequence of shape (batch, seq_len, total_embed_size)
        """
        return self.encoder(feature_tuple)

    def forward(self, batch):
        """
        Forward pass for training.

        Args:
            batch: LibCity Batch object containing:
                - 'current_loc': (batch_size, seq_len) - POI indices
                - 'current_tim': (batch_size, seq_len) - hour indices (optional, defaults to zeros)
                - 'current_day': (batch_size, seq_len) - day indices (optional, defaults to zeros)
                - 'current_cat': (batch_size, seq_len) - category indices (optional, defaults to zeros)
                - 'uid': (batch_size,) - user indices (1D tensor from StandardTrajectoryEncoder)
                - 'history_loc': (batch_size, history_len) - historical POI sequence (2D tensor from StandardTrajectoryEncoder)
                - 'target': (batch_size,) - target POI indices

        Note:
            LibCity's StandardTrajectoryEncoder provides:
            - history_loc as 2D tensor (batch, history_len), NOT 3D
            - uid as 1D tensor (batch,), NOT 2D
            - No current_cat, current_day features by default

            This model handles these formats by:
            - Treating history_loc as a single concatenated sequence
            - Creating dummy zeros for missing category/day features
            - Broadcasting uid to match sequence lengths

        Returns:
            Tuple of (ssl_loss, output) where:
                - ssl_loss: Contrastive learning loss (0 if SSL disabled or no neg samples)
                - output: (batch_size, loc_size) prediction logits
        """
        features = self._extract_features_from_batch(batch)
        batch_size = features['current_loc'].shape[0]
        seq_len = features['current_loc'].shape[1]

        # Short-term features (exclude the last position which is the target)
        short_term_idx = slice(0, -1) if seq_len > 1 else slice(None)
        short_term_tuple = self._get_feature_tuple(features, short_term_idx)

        # Encode short-term sequence
        short_term_state = self._encode_sequence(short_term_tuple)  # (batch, seq_len-1, embed)

        # Process long-term history if available
        # LibCity StandardTrajectoryEncoder provides history_loc as 2D tensor (batch, history_len)
        if 'history_loc' in (batch.data if hasattr(batch, 'data') else batch):
            history_loc = batch['history_loc']  # (batch, history_len) - 2D tensor from LibCity

            # Handle both 2D (LibCity standard) and 3D (legacy) formats
            if history_loc.dim() == 2:
                # LibCity StandardTrajectoryEncoder format: (batch, history_len)
                # Treat the entire history as a single concatenated sequence
                hist_len = history_loc.shape[1]

                # Create dummy features for category and day since they are not provided
                # by StandardTrajectoryEncoder
                hist_features = {
                    'current_loc': history_loc,  # (batch, history_len)
                    'current_cat': torch.zeros_like(history_loc),  # dummy zeros
                    'current_hour': torch.zeros_like(history_loc),  # dummy zeros
                    'current_day': torch.zeros_like(history_loc),  # dummy zeros
                }

                # Handle uid - expand to match history length if needed
                if features['uid'].dim() == 1:
                    hist_features['uid'] = features['uid'].unsqueeze(1).expand(-1, hist_len)
                else:
                    # If uid is already 2D, expand or truncate to match history length
                    uid_len = features['uid'].shape[1]
                    if uid_len >= hist_len:
                        hist_features['uid'] = features['uid'][:, :hist_len]
                    else:
                        # Expand by repeating
                        hist_features['uid'] = features['uid'][:, 0:1].expand(-1, hist_len)

                # Apply masking for data augmentation during training
                if self.training:
                    hist_features = self.feature_mask(hist_features, self.mask_prop)

                hist_tuple = self._get_feature_tuple(hist_features)
                long_term_catted = self._encode_sequence(hist_tuple)  # (batch, history_len, embed)

            elif history_loc.dim() == 3:
                # Legacy 3D format: (batch, hist_sessions, seq_len)
                # Encode each historical session separately
                long_term_states = []
                for i in range(history_loc.shape[1]):
                    hist_features = {
                        'current_loc': history_loc[:, i, :],
                        'current_cat': batch.get('history_cat', torch.zeros_like(history_loc))[:, i, :] if 'history_cat' in (batch.data if hasattr(batch, 'data') else batch) else torch.zeros_like(history_loc[:, i, :]),
                        'uid': features['uid'][:, :history_loc.shape[2]] if features['uid'].dim() > 1 else features['uid'].unsqueeze(1).expand(-1, history_loc.shape[2]),
                        'current_hour': batch.get('history_tim', torch.zeros_like(history_loc))[:, i, :] if 'history_tim' in (batch.data if hasattr(batch, 'data') else batch) else torch.zeros_like(history_loc[:, i, :]),
                        'current_day': batch.get('history_day', torch.zeros_like(history_loc))[:, i, :] if 'history_day' in (batch.data if hasattr(batch, 'data') else batch) else torch.zeros_like(history_loc[:, i, :])
                    }
                    # Apply masking
                    if self.training:
                        hist_features = self.feature_mask(hist_features, self.mask_prop)

                    hist_tuple = self._get_feature_tuple(hist_features)
                    hist_encoded = self._encode_sequence(hist_tuple)  # (batch, seq_len, embed)
                    long_term_states.append(hist_encoded)

                # Concatenate all history
                long_term_catted = torch.cat(long_term_states, dim=1)  # (batch, total_hist_len, embed)
            else:
                # Fallback: use short-term as long-term
                long_term_catted = short_term_state
        else:
            # Use short-term as long-term if no history available
            long_term_catted = short_term_state

        # User enhancement using LSTM
        user_id = features['uid'][:, 0] if features['uid'].dim() > 1 else features['uid']
        user_embed_raw = self.embedding.user_embed(user_id)  # (batch, f_embed_size)

        # LSTM encoding of short-term sequence
        short_term_embedding = self.embedding(short_term_tuple)  # (batch, seq_len-1, total_embed)
        lstm_out, _ = self.lstm(short_term_embedding)  # (batch, seq_len-1, total_embed)
        short_term_enhance = lstm_out.mean(dim=1)  # (batch, total_embed)

        user_embed = self.enhance_val * user_embed_raw + (1 - self.enhance_val) * self.tryone_line2(short_term_enhance)

        # Contrastive learning loss
        ssl_loss_val = torch.tensor(0.0, device=self.device)
        if self.enable_ssl and self.training:
            if 'neg_loc' in (batch.data if hasattr(batch, 'data') else batch):
                neg_loc = batch['neg_loc']  # (batch, neg_count, seq_len)

                # Encode negative samples
                neg_states = []
                for i in range(neg_loc.shape[1]):
                    neg_features = {
                        'current_loc': neg_loc[:, i, :],
                        'current_cat': batch.get('neg_cat', torch.zeros_like(neg_loc))[:, i, :] if 'neg_cat' in (batch.data if hasattr(batch, 'data') else batch) else torch.zeros_like(neg_loc[:, i, :]),
                        'uid': torch.zeros(batch_size, neg_loc.shape[2], dtype=torch.long, device=self.device),
                        'current_hour': batch.get('neg_tim', torch.zeros_like(neg_loc))[:, i, :] if 'neg_tim' in (batch.data if hasattr(batch, 'data') else batch) else torch.zeros_like(neg_loc[:, i, :]),
                        'current_day': batch.get('neg_day', torch.zeros_like(neg_loc))[:, i, :] if 'neg_day' in (batch.data if hasattr(batch, 'data') else batch) else torch.zeros_like(neg_loc[:, i, :])
                    }
                    neg_tuple = self._get_feature_tuple(neg_features)
                    neg_encoded = self._encode_sequence(neg_tuple)  # (batch, seq_len, embed)
                    neg_states.append(neg_encoded.mean(dim=1))  # (batch, embed)

                neg_embed_mean = torch.stack(neg_states, dim=0).mean(dim=0)  # (batch, embed)
                short_embed_mean = short_term_state.mean(dim=1)  # (batch, embed)
                long_embed_mean = long_term_catted.mean(dim=1)  # (batch, embed)

                ssl_loss_val = self.ssl_loss(short_embed_mean, long_embed_mean, neg_embed_mean)

        # Final prediction
        h_all = torch.cat((short_term_state, long_term_catted), dim=1)  # (batch, total_len, embed)
        final_att = self.final_attention(user_embed, h_all, h_all)  # (batch, total_embed)
        output = self.out_linear(final_att)  # (batch, loc_size)

        return ssl_loss_val, output

    def predict(self, batch):
        """
        Predict next POI location.

        Args:
            batch: LibCity Batch object containing input data

        Returns:
            torch.Tensor: POI prediction scores of shape (batch_size, loc_size)
        """
        self.eval()
        with torch.no_grad():
            _, output = self.forward(batch)
        return output

    def calculate_loss(self, batch):
        """
        Calculate combined prediction and contrastive loss.

        Args:
            batch: LibCity Batch object containing:
                - Input features (see forward method)
                - 'target': (batch_size,) - target POI indices

        Returns:
            torch.Tensor: Total loss scalar
        """
        ssl_loss_val, output = self.forward(batch)

        # Get target
        target = batch['target']  # (batch_size,)

        # Prediction loss
        pred_loss = self.loss_func(output, target)

        # Combined loss
        total_loss = pred_loss + ssl_loss_val * self.neg_weight

        return total_loss
