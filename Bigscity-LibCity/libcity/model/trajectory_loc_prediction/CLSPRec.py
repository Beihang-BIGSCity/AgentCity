"""
CLSPRec: Contrastive Learning for Long and Short-term Preference Recommendation
Adapted for LibCity Framework

Original source: repos/CLSPRec/CLSPRec.py
Key components preserved:
- CheckInEmbedding: 5 features (POI, category, user, hour, day)
- TransformerEncoder: Custom implementation with SelfAttention and EncoderBlock
- LSTM: Short-term sequence modeling
- Attention: Custom attention for prediction
- Contrastive Learning: InfoNCE loss for long/short-term alignment

Key adaptations:
- Inherited from AbstractModel
- Adapted __init__() to use LibCity's config and data_feature parameters
- Implemented predict() method following LibCity conventions
- Implemented calculate_loss() method (cross-entropy + contrastive learning)
- Preserved dual-preference modeling (long-term + short-term sequences)
- Adapted batch format for LibCity's data loading conventions

Required config parameters:
    - f_embed_size: Feature embedding dimension (default: 60)
    - num_encoder_layers: Number of transformer encoder layers (default: 1)
    - num_lstm_layers: Number of LSTM layers (default: 1)
    - num_heads: Number of attention heads (default: 1)
    - forward_expansion: Feed-forward expansion factor (default: 4)
    - dropout_p: Dropout probability (default: 0.2)
    - neg_weight: Weight for contrastive loss (default: 1.0)
    - mask_prop: Proportion of features to mask (default: 0.1)
    - enable_ssl: Whether to enable contrastive learning (default: True)
    - enable_random_mask: Whether to enable random masking (default: True)

Required data_feature parameters:
    - loc_size: Number of unique POI locations
    - cat_size: Number of unique categories
    - uid_size: Number of unique users
    - tim_size: Number of time slots (typically 48 for LibCity)
"""

import torch
from torch import nn
import torch.nn.functional as F

from libcity.model.abstract_model import AbstractModel


class CheckInEmbedding(nn.Module):
    """
    Embedding layer for check-in data with 5 features:
    POI, category, user, hour, and day of week.
    """

    def __init__(self, f_embed_size, vocab_size):
        super().__init__()
        self.embed_size = f_embed_size
        poi_num = vocab_size["POI"]
        cat_num = vocab_size["cat"]
        user_num = vocab_size["user"]
        hour_num = vocab_size["hour"]
        day_num = vocab_size["day"]

        self.poi_embed = nn.Embedding(poi_num + 1, self.embed_size, padding_idx=poi_num)
        self.cat_embed = nn.Embedding(cat_num + 1, self.embed_size, padding_idx=cat_num)
        self.user_embed = nn.Embedding(user_num + 1, self.embed_size, padding_idx=user_num)
        self.hour_embed = nn.Embedding(hour_num + 1, self.embed_size, padding_idx=hour_num)
        self.day_embed = nn.Embedding(day_num + 1, self.embed_size, padding_idx=day_num)

    def forward(self, x):
        """
        Args:
            x: tuple of 5 tensors (poi, cat, user, hour, day), each of shape [seq_len]

        Returns:
            Concatenated embeddings of shape [seq_len, embed_size * 5]
        """
        poi_emb = self.poi_embed(x[0])
        cat_emb = self.cat_embed(x[1])
        user_emb = self.user_embed(x[2])
        hour_emb = self.hour_embed(x[3])
        day_emb = self.day_embed(x[4])

        return torch.cat((poi_emb, cat_emb, user_emb, hour_emb, day_emb), -1)


class SelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""

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

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class TransformerEncoder(nn.Module):
    """Custom Transformer encoder with embedding layer."""

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
            feature_seq: tuple of 5 tensors (poi, cat, user, hour, day)

        Returns:
            Encoded sequence of shape [seq_len, embed_size]
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
        # Resize q's dimension to k
        self.expansion = nn.Linear(qdim, kdim)

    def forward(self, query, key, value):
        q = self.expansion(query)
        temp = torch.inner(q, key)
        weight = torch.softmax(temp, dim=0)
        weight = torch.unsqueeze(weight, 1)
        temp2 = torch.mul(value, weight)
        out = torch.sum(temp2, 0)

        return out


class CLSPRec(AbstractModel):
    """
    CLSPRec: Contrastive Learning for Long and Short-term Preference Recommendation

    This model captures both long-term and short-term user preferences for
    next POI recommendation using:
    1. Transformer encoder for long-term sequence modeling
    2. LSTM for short-term sequence modeling
    3. Contrastive learning (InfoNCE) to align long and short-term representations

    The model processes check-in sequences with 5 features per check-in:
    POI location, category, user ID, hour of day, and day of week.
    """

    def __init__(self, config, data_feature):
        super(CLSPRec, self).__init__(config, data_feature)

        # Get device
        self.device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

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
        self.temperature = config.get('temperature', 0.07)

        # Vocabulary sizes from data_feature
        # LibCity provides loc_size, cat_size (if available), uid_size, tim_size
        self.vocab_size = {
            "POI": data_feature.get('loc_size', 100),
            "cat": data_feature.get('cat_size', 50),
            "user": data_feature.get('uid_size', 10),
            "hour": 24,  # Hours are 0-23
            "day": 7,    # Days are 0-6
        }

        self.total_embed_size = self.f_embed_size * 5

        # Build model layers
        self.embedding = CheckInEmbedding(
            self.f_embed_size,
            self.vocab_size
        )

        self.encoder = TransformerEncoder(
            self.embedding,
            self.total_embed_size,
            self.num_encoder_layers,
            self.num_heads,
            self.forward_expansion,
            self.dropout_p,
        )

        self.lstm = nn.LSTM(
            input_size=self.total_embed_size,
            hidden_size=self.total_embed_size,
            num_layers=self.num_lstm_layers,
            dropout=0
        )

        self.final_attention = Attention(
            qdim=self.f_embed_size,
            kdim=self.total_embed_size
        )

        self.out_linear = nn.Sequential(
            nn.Linear(self.total_embed_size, self.total_embed_size * self.forward_expansion),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.total_embed_size * self.forward_expansion, self.vocab_size["POI"])
        )

        self.loss_func = nn.CrossEntropyLoss()

        # User enhancement layer
        self.tryone_line2 = nn.Linear(self.total_embed_size, self.f_embed_size)
        self.enhance_val = nn.Parameter(torch.tensor(0.5))

    def feature_mask(self, features, mask_prop):
        """
        Apply random masking to features for data augmentation.

        Args:
            features: Feature tensor of shape [5, seq_len]
            mask_prop: Proportion of positions to mask

        Returns:
            Masked feature tensor
        """
        if not self.enable_random_mask or not self.training:
            return features

        features = features.clone()
        seq_len = features.shape[1]
        mask_count = int(torch.ceil(mask_prop * torch.tensor(seq_len)).item())

        if mask_count > 0 and seq_len > 1:
            masked_index = torch.randperm(seq_len - 1, device=features.device) + 1
            masked_index = masked_index[:mask_count]

            features[0, masked_index] = self.vocab_size["POI"]   # mask POI
            features[1, masked_index] = self.vocab_size["cat"]   # mask category
            features[3, masked_index] = self.vocab_size["hour"]  # mask hour
            features[4, masked_index] = self.vocab_size["day"]   # mask day

        return features

    def ssl_loss(self, embedding_1, embedding_2, neg_embedding):
        """
        Compute InfoNCE contrastive loss for self-supervised learning.

        Args:
            embedding_1: Long-term embedding
            embedding_2: Short-term embedding
            neg_embedding: Negative sample embedding

        Returns:
            Contrastive loss scalar
        """
        def score(x1, x2):
            return torch.mean(torch.mul(x1, x2))

        pos = score(embedding_1, embedding_2)
        neg1 = score(embedding_1, neg_embedding)
        neg2 = score(embedding_2, neg_embedding)
        neg = (neg1 + neg2) / 2

        one = torch.tensor([1.0], device=self.device)
        con_loss = torch.sum(
            -torch.log(1e-8 + torch.sigmoid(pos)) -
            torch.log(1e-8 + (one - torch.sigmoid(neg)))
        )
        return con_loss

    def _extract_features_from_batch(self, batch):
        """
        Extract and prepare features from LibCity batch format.

        LibCity batch contains:
        - current_loc: [batch_size, seq_len] - POI locations
        - current_tim: [batch_size, seq_len] - Time slots (0-47)
        - uid: [batch_size] - User IDs
        - target: [batch_size] - Target POI
        - history_loc (optional): Historical location sequences
        - category (optional): POI categories

        Returns:
            Tuple of (features_list, target, neg_features_list)
            where features_list contains feature tuples for each sample
        """
        current_loc = batch['current_loc']  # [batch_size, seq_len]
        current_tim = batch['current_tim']  # [batch_size, seq_len]
        uid = batch['uid']                  # [batch_size]
        target = batch['target']            # [batch_size]

        batch_size, seq_len = current_loc.shape

        # Convert time to hour (0-23) and day (0-6)
        # LibCity time encoding: 0-23 for weekday hours, 24-47 for weekend hours
        hour = current_tim % 24
        day = (current_tim >= 24).long()  # 0 for weekday, 1 for weekend
        # Expand day to 0-6 range if we have day information
        if 'day' in batch.data:
            day = batch['day']

        # Get category if available, otherwise use location as proxy
        if 'category' in batch.data:
            category = batch['category']
        elif 'cat' in batch.data:
            category = batch['cat']
        else:
            # Use location as category proxy (will be learned)
            category = current_loc % self.vocab_size["cat"]

        # Expand user ID to match sequence length
        user = uid.unsqueeze(1).expand(-1, seq_len)

        return current_loc, category, user, hour, day, target

    def forward(self, batch):
        """
        Forward pass for CLSPRec model.

        This method adapts the original CLSPRec forward pass to work with
        LibCity's batch format while preserving the dual-preference modeling.

        Args:
            batch: LibCity Batch object

        Returns:
            Tuple of (loss, output_logits) during training
            Output logits of shape [batch_size, num_locations] during inference
        """
        current_loc, category, user, hour, day, target = self._extract_features_from_batch(batch)

        batch_size, seq_len = current_loc.shape

        outputs = []
        losses = []

        for i in range(batch_size):
            # Extract features for this sample
            # Shape: [5, seq_len] where 5 = (poi, cat, user, hour, day)
            features = torch.stack([
                current_loc[i],
                category[i],
                user[i],
                hour[i],
                day[i]
            ], dim=0)

            # Split into input and target
            input_features = features[:, :-1]  # All but last for input
            sample_target = current_loc[i, -1]  # Last POI as target
            user_id = user[i, 0]

            # Apply masking for long-term (data augmentation)
            if self.training and self.enable_random_mask:
                masked_features = self.feature_mask(input_features, self.mask_prop)
            else:
                masked_features = input_features

            # Long-term encoding via Transformer
            # Feature tuple for embedding: (poi, cat, user, hour, day)
            feature_tuple = (
                masked_features[0],
                masked_features[1],
                masked_features[2],
                masked_features[3],
                masked_features[4]
            )
            long_term_out = self.encoder(feature_seq=feature_tuple)

            # Short-term encoding via LSTM
            short_feature_tuple = (
                input_features[0],
                input_features[1],
                input_features[2],
                input_features[3],
                input_features[4]
            )
            short_term_state = self.encoder(feature_seq=short_feature_tuple)

            # User enhancement
            user_embed = self.embedding.user_embed(user_id)
            embedding_seq = self.embedding(short_feature_tuple)
            embedding_seq = torch.unsqueeze(embedding_seq, 0)  # Add batch dim for LSTM
            output, _ = self.lstm(embedding_seq)
            short_term_enhance = torch.squeeze(output)

            user_embed = (
                self.enhance_val * user_embed +
                (1 - self.enhance_val) * self.tryone_line2(torch.mean(short_term_enhance, dim=0))
            )

            # Combine long and short term representations
            h_all = torch.cat((short_term_state, long_term_out))
            final_att = self.final_attention(user_embed, h_all, h_all)
            output = self.out_linear(final_att)

            outputs.append(output)

            # Calculate prediction loss
            if self.training:
                label = sample_target.unsqueeze(0)
                pred = output.unsqueeze(0)
                pred_loss = self.loss_func(pred, label)

                # Contrastive loss
                ssl_loss = torch.tensor(0.0, device=self.device)
                if self.enable_ssl:
                    short_embed_mean = torch.mean(short_term_state, dim=0)
                    long_embed_mean = torch.mean(long_term_out, dim=0)
                    # Use a shifted version of embeddings as negative samples
                    neg_embed_mean = torch.roll(short_embed_mean, shifts=1, dims=0)
                    ssl_loss = self.ssl_loss(short_embed_mean, long_embed_mean, neg_embed_mean)

                total_loss = pred_loss + ssl_loss * self.neg_weight
                losses.append(total_loss)

        # Stack outputs
        output_tensor = torch.stack(outputs, dim=0)  # [batch_size, num_locations]

        if self.training:
            avg_loss = torch.mean(torch.stack(losses))
            return avg_loss, output_tensor
        else:
            return output_tensor

    def predict(self, batch):
        """
        Predict next locations for the given batch.

        Args:
            batch: LibCity Batch object

        Returns:
            Location prediction scores [batch_size, num_locations]
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(batch)
            if isinstance(output, tuple):
                output = output[1]
        return output

    def calculate_loss(self, batch):
        """
        Calculate the combined loss (cross-entropy + contrastive learning).

        Args:
            batch: LibCity Batch object

        Returns:
            Combined loss tensor
        """
        self.train()
        loss, _ = self.forward(batch)
        return loss
