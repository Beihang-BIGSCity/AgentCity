# coding: utf-8
"""
PLSPL: Personalized Long-Short term Preference Learning for Next POI Recommendation

This module adapts the PLSPL model (originally named long_short) for the LibCity framework.

Model Architecture:
1. Embedding layers: user, POI, category, hour, week
2. Dual LSTM networks: one for POI sequences, one for category sequences
3. Long-term preference module: attention-based aggregation over user history
4. Personalized fusion: learned per-user weights to combine POI, category, and long-term preferences

Key Innovations:
- Dual-stream LSTM for POI and category sequences
- Personalized weighting scheme for combining short-term and long-term preferences
- Attention mechanism for aggregating user's long-term historical preferences

Original Paper Reference:
The model captures both short-term sequential patterns (via LSTMs) and long-term
user preferences (via attention-based aggregation) for next POI recommendation.

Required data_feature keys:
- loc_size: Number of unique POIs (vocab_poi)
- uid_size: Number of unique users (vocab_user)
- cat_size: Number of POI categories (vocab_cat)
- loc_pad: Padding index for locations (optional)
- long_term: Dictionary storing per-user historical data (optional, can be built dynamically)

Required config parameters:
- hidden_size: LSTM hidden size (default: 128)
- num_layers: Number of LSTM layers (default: 1)
- embed_poi: POI embedding dimension (default: 300)
- embed_cat: Category embedding dimension (default: 100)
- embed_user: User embedding dimension (default: 50)
- embed_hour: Hour embedding dimension (default: 20)
- embed_week: Week embedding dimension (default: 7)
- dropout: Dropout rate (default: 0.5)

Adapted from:
- Original file: repos/PLSPL/model_longshort.py
- Original class: long_short

Key Changes Made:
1. Renamed class from long_short to PLSPL
2. Inherits from AbstractModel instead of nn.Module
3. Updated __init__ to use (config, data_feature) signature
4. Implemented predict() and calculate_loss() methods for LibCity
5. Removed deprecated torch.autograd.Variable usage
6. Adapted forward() to work with LibCity batch format
7. Added device handling for GPU/CPU compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from libcity.model.abstract_model import AbstractModel


class PLSPL(AbstractModel):
    """
    Personalized Long-Short term Preference Learning for Next POI Recommendation.

    This model combines:
    1. Short-term preferences via dual LSTM (POI and category streams)
    2. Long-term user preferences via attention-based aggregation
    3. Personalized per-user fusion weights
    """

    def __init__(self, config, data_feature):
        """
        Initialize the PLSPL model.

        Args:
            config: Configuration dictionary containing model hyperparameters
            data_feature: Dictionary containing data-related information
        """
        super(PLSPL, self).__init__(config, data_feature)

        # Device configuration
        self.device = config.get('device', 'cpu')

        # Extract vocabulary sizes from data_feature
        self.vocab_poi = data_feature.get('loc_size', 1000)
        self.vocab_user = data_feature.get('uid_size', 100)
        self.vocab_cat = data_feature.get('cat_size', 50)
        self.vocab_hour = 24  # Fixed: 24 hours in a day
        self.vocab_week = 7   # Fixed: 7 days in a week

        # Extract hyperparameters from config with defaults
        self.hidden_size = config.get('hidden_size', 128)
        self.num_layers = config.get('num_layers', 1)
        self.embed_size_poi = config.get('embed_poi', 300)
        self.embed_size_cat = config.get('embed_cat', 100)
        self.embed_size_user = config.get('embed_user', 50)
        self.embed_size_hour = config.get('embed_hour', 20)
        self.embed_size_week = config.get('embed_week', 7)
        self.dropout = config.get('dropout', 0.5)

        # Sequence length for repeat operations (can be overridden dynamically)
        self.default_seq_len = config.get('seq_len', 19)

        # Long-term preference data storage
        # This dictionary stores per-user historical data for long-term preference computation
        # Format: {user_id: {'loc': tensor, 'hour': tensor, 'week': tensor, 'category': tensor}}
        self.long_term = data_feature.get('long_term', {})

        # POI to category mapping (optional)
        self.poi_to_cat = data_feature.get('poi_to_cat', None)

        # Total embedding size for concatenated features
        self.embed_total_size = (self.embed_size_poi + self.embed_size_cat +
                                  self.embed_size_hour + self.embed_size_week)

        # Build model layers
        self._build_model()

    def _build_model(self):
        """Build all model layers."""
        # Embedding layers
        self.embed_user = nn.Embedding(self.vocab_user, self.embed_size_user)
        self.embed_poi = nn.Embedding(self.vocab_poi, self.embed_size_poi)
        self.embed_cat = nn.Embedding(self.vocab_cat, self.embed_size_cat)
        self.embed_hour = nn.Embedding(self.vocab_hour, self.embed_size_hour)
        self.embed_week = nn.Embedding(self.vocab_week, self.embed_size_week)

        # Long-term preference attention weights
        self.weight_poi = Parameter(torch.ones(self.embed_size_poi, self.embed_size_user))
        self.weight_cat = Parameter(torch.ones(self.embed_size_cat, self.embed_size_user))
        self.weight_time = Parameter(torch.ones(
            self.embed_size_hour + self.embed_size_week, self.embed_size_user))
        self.bias = Parameter(torch.ones(self.embed_size_user))

        # Activation function
        self.activate_func = nn.ReLU()

        # Personalized per-user output weights
        # These are learnable parameters that determine how much each user
        # weights the POI stream, category stream, and long-term preferences
        self.out_w_long = Parameter(torch.Tensor([0.5]).repeat(self.vocab_user))
        self.out_w_poi = Parameter(torch.Tensor([0.25]).repeat(self.vocab_user))
        self.out_w_cat = Parameter(torch.Tensor([0.25]).repeat(self.vocab_user))

        # Hidden layer weights for potential attention mechanisms
        self.weight_hidden_poi = Parameter(torch.ones(self.hidden_size, 1))
        self.weight_hidden_cat = Parameter(torch.ones(self.hidden_size, 1))

        # LSTM input sizes
        size_poi = self.embed_size_poi + self.embed_size_user + self.embed_size_hour
        size_cat = self.embed_size_cat + self.embed_size_user + self.embed_size_hour

        # Dual LSTM networks
        self.lstm_poi = nn.LSTM(
            input_size=size_poi,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        self.lstm_cat = nn.LSTM(
            input_size=size_cat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )

        # Fully connected layers for final predictions
        self.fc_poi = nn.Linear(self.hidden_size, self.vocab_poi)
        self.fc_cat = nn.Linear(self.hidden_size, self.vocab_poi)
        self.attn_linear = nn.Linear(self.hidden_size * 2, self.vocab_poi)

        # Long-term preference projection
        self.fc_longterm = nn.Linear(self.embed_total_size, self.vocab_poi)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embedding weights
        nn.init.xavier_uniform_(self.embed_user.weight)
        nn.init.xavier_uniform_(self.embed_poi.weight)
        nn.init.xavier_uniform_(self.embed_cat.weight)
        nn.init.xavier_uniform_(self.embed_hour.weight)
        nn.init.xavier_uniform_(self.embed_week.weight)

        # Initialize LSTM weights
        for name, param in self.lstm_poi.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

        for name, param in self.lstm_cat.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def set_long_term_data(self, long_term_dict):
        """
        Set the long-term preference data for users.

        This method allows updating the long-term historical data that is used
        for computing user's long-term preferences.

        Args:
            long_term_dict: Dictionary mapping user_id to their historical data
                           Format: {user_id: {'loc': tensor, 'hour': tensor,
                                              'week': tensor, 'category': tensor}}
        """
        self.long_term = long_term_dict

    def get_output(self, inputs, inputs_user, inputs_time, embed_layer,
                   embed_user, embed_time, lstm_layer, fc_layer):
        """
        Process a sequence through embedding and LSTM layers.

        This is a shared function for both POI and category streams.

        Args:
            inputs: Input sequence tensor (batch_size, seq_len)
            inputs_user: User indices tensor (batch_size,)
            inputs_time: Time (hour) indices tensor (batch_size, seq_len)
            embed_layer: Embedding layer for inputs (POI or category)
            embed_user: User embedding layer
            embed_time: Time embedding layer
            lstm_layer: LSTM layer to process the sequence
            fc_layer: Fully connected layer for output projection

        Returns:
            Output tensor of shape (batch_size, seq_len, vocab_poi)
        """
        # Embed the input sequences
        seq_tensor = embed_layer(inputs)  # (batch, seq_len, embed_dim)

        # Embed users and expand to sequence length
        seq_user = embed_user(inputs_user)  # (batch, user_embed_dim)
        seq_user = seq_user.unsqueeze(1).repeat(1, seq_tensor.size(1), 1)

        # Embed time
        seq_time = embed_time(inputs_time)  # (batch, seq_len, time_embed_dim)

        # Concatenate all embeddings
        input_tensor = torch.cat((seq_tensor, seq_user, seq_time), dim=2)

        # Pass through LSTM
        output, _ = lstm_layer(input_tensor)

        # Project to vocabulary size
        out = fc_layer(output)

        return out

    def get_u_long(self, inputs_user):
        """
        Compute long-term user preference representations.

        This method computes an attention-weighted aggregation of each user's
        historical check-in data to form their long-term preference vector.

        Args:
            inputs_user: User indices tensor (batch_size,)

        Returns:
            Long-term preference tensor of shape (batch_size, embed_total_size)
        """
        u_long = {}

        for user in inputs_user:
            user_index = user.item() if isinstance(user, torch.Tensor) else user

            if user_index not in u_long.keys():
                # Check if we have long-term data for this user
                if user_index in self.long_term:
                    poi = self.long_term[user_index]['loc'].to(self.device)
                    hour = self.long_term[user_index]['hour'].to(self.device)
                    week = self.long_term[user_index]['week'].to(self.device)
                    cat = self.long_term[user_index]['category'].to(self.device)

                    # Embed historical data
                    seq_poi = self.embed_poi(poi)
                    seq_cat = self.embed_cat(cat)
                    seq_user = self.embed_user(user.to(self.device) if isinstance(user, torch.Tensor) else
                                               torch.tensor(user, device=self.device))
                    seq_hour = self.embed_hour(hour)
                    seq_week = self.embed_week(week)
                    seq_time = torch.cat((seq_hour, seq_week), dim=1)

                    # Compute attention-weighted representation
                    poi_mm = torch.mm(seq_poi, self.weight_poi)
                    cat_mm = torch.mm(seq_cat, self.weight_cat)
                    time_mm = torch.mm(seq_time, self.weight_time)

                    hidden_vec = poi_mm + cat_mm + time_mm + self.bias
                    hidden_vec = self.activate_func(hidden_vec)

                    # Compute attention weights
                    alpha = F.softmax(torch.mm(hidden_vec, seq_user.unsqueeze(1)), dim=0)

                    # Concatenate all embeddings
                    poi_concat = torch.cat((seq_poi, seq_cat, seq_hour, seq_week), dim=1)

                    # Attention-weighted aggregation
                    u_long[user_index] = torch.sum(torch.mul(poi_concat, alpha), dim=0)
                else:
                    # If no long-term data available, use zero vector
                    u_long[user_index] = torch.zeros(self.embed_total_size, device=self.device)

        # Stack all user representations
        batch_size = len(inputs_user)
        u_long_all = torch.zeros(batch_size, self.embed_total_size, device=self.device)

        for i in range(batch_size):
            user_index = inputs_user[i].item() if isinstance(inputs_user[i], torch.Tensor) else inputs_user[i]
            u_long_all[i, :] = u_long[user_index]

        return u_long_all

    def forward(self, batch):
        """
        Forward pass of the PLSPL model.

        Args:
            batch: Dictionary containing:
                - 'current_loc': POI sequence (batch_size, seq_len)
                - 'current_cat' or inferred from current_loc: Category sequence
                - 'uid': User indices (batch_size,)
                - 'current_tim' or 'current_hour': Hour indices (batch_size, seq_len)
                - 'current_week' (optional): Week day indices

        Returns:
            Output tensor of shape (batch_size, seq_len, vocab_poi) containing
            POI prediction scores for each position in the sequence.
        """
        # Extract inputs from batch
        inputs_poi = batch['current_loc']  # (batch_size, seq_len)
        batch_size, seq_len = inputs_poi.shape

        # Get user indices
        if 'uid' in batch.data:
            inputs_user = batch['uid']
            if inputs_user.dim() > 1:
                inputs_user = inputs_user.squeeze(-1)
        else:
            inputs_user = torch.zeros(batch_size, dtype=torch.long, device=inputs_poi.device)

        # Get category sequence
        if 'current_cat' in batch.data:
            inputs_cat = batch['current_cat']
        elif self.poi_to_cat is not None:
            # Map POI to category
            inputs_cat = torch.tensor(
                [[self.poi_to_cat.get(p.item(), 0) for p in seq] for seq in inputs_poi],
                dtype=torch.long, device=inputs_poi.device
            )
        else:
            # Use zeros if no category mapping available
            inputs_cat = torch.zeros_like(inputs_poi)

        # Get time (hour) indices
        if 'current_hour' in batch.data:
            inputs_time = batch['current_hour']
        elif 'current_tim' in batch.data:
            # Assume current_tim is hour index or convert from normalized time
            tim = batch['current_tim']
            if tim.dtype in [torch.float, torch.float32, torch.float64]:
                # Convert normalized time to hour index
                inputs_time = (tim * 24).long().clamp(0, 23)
            else:
                inputs_time = tim.clamp(0, 23)
        else:
            inputs_time = torch.zeros_like(inputs_poi)

        # Clamp indices to valid ranges
        inputs_poi = torch.clamp(inputs_poi, 0, self.vocab_poi - 1)
        inputs_cat = torch.clamp(inputs_cat, 0, self.vocab_cat - 1)
        inputs_user = torch.clamp(inputs_user, 0, self.vocab_user - 1)
        inputs_time = torch.clamp(inputs_time, 0, self.vocab_hour - 1)

        # Process POI stream through LSTM
        out_poi = self.get_output(
            inputs_poi, inputs_user, inputs_time,
            self.embed_poi, self.embed_user, self.embed_hour,
            self.lstm_poi, self.fc_poi
        )

        # Process category stream through LSTM
        out_cat = self.get_output(
            inputs_cat, inputs_user, inputs_time,
            self.embed_cat, self.embed_user, self.embed_hour,
            self.lstm_cat, self.fc_cat
        )

        # Compute long-term preference
        u_long = self.get_u_long(inputs_user)

        # Project long-term preference to vocabulary size and expand to sequence length
        out_long = self.fc_longterm(u_long).unsqueeze(1).repeat(1, seq_len, 1)

        # Get personalized weights for this batch of users
        weight_poi = self.out_w_poi[inputs_user]  # (batch_size,)
        weight_cat = self.out_w_cat[inputs_user]  # (batch_size,)
        weight_long = 1 - weight_poi - weight_cat  # (batch_size,)

        # Expand weights for broadcasting: (batch_size, seq_len, 1)
        weight_poi = weight_poi.unsqueeze(1).unsqueeze(2).repeat(1, seq_len, 1)
        weight_cat = weight_cat.unsqueeze(1).unsqueeze(2).repeat(1, seq_len, 1)
        weight_long = weight_long.unsqueeze(1).unsqueeze(2).repeat(1, seq_len, 1)

        # Weighted combination of all three streams
        out = out_poi * weight_poi + out_cat * weight_cat + out_long * weight_long

        return out

    def predict(self, batch):
        """
        Predict the next POI for each sequence in the batch.

        Args:
            batch: Dictionary containing input data

        Returns:
            Prediction scores tensor of shape (batch_size, vocab_poi)
            Returns the prediction for the last position in each sequence.
        """
        out = self.forward(batch)

        # Return predictions for the last position
        last_pred = out[:, -1, :]  # (batch_size, vocab_poi)

        return last_pred

    def calculate_loss(self, batch):
        """
        Calculate the cross-entropy loss for next POI prediction.

        Args:
            batch: Dictionary containing:
                - Input data for forward pass
                - 'target': Target POI indices (batch_size,) or (batch_size, seq_len)

        Returns:
            Loss tensor (scalar)
        """
        out = self.forward(batch)
        target = batch['target']

        # Handle single target (next POI prediction)
        if target.dim() == 1:
            # Use prediction from last position
            out_last = out[:, -1, :]  # (batch_size, vocab_poi)
            loss = self.criterion(out_last, target)
        else:
            # Sequence-to-sequence prediction
            # out: (batch_size, seq_len, vocab_poi)
            # target: (batch_size, seq_len)
            batch_size, seq_len, vocab_size = out.shape
            out_flat = out.view(-1, vocab_size)
            target_flat = target.view(-1)
            loss = self.criterion(out_flat, target_flat)

        return loss
