"""
DutyTTE: Mixture of Experts with Uncertainty Quantification for Travel Time Estimation

This model is adapted from the MoEUQ_network in the DutyTTE repository.
Original source: /home/wangwenrui/shk/AgentCity/repos/DutyTTE/uncertainty_quantification/MoEUQ.py

Key Components:
1. SparseMoE module (Mixture of Experts with Sparse Gating)
2. NoisyTopkRouter (expert selection with noise for exploration)
3. Expert networks (MLPs with 4x expansion)
4. Multi-branch Regressor (mean, lower, upper bound predictions)
5. MIS loss function (Mean Interval Score) for uncertainty quantification

The model provides:
- Point prediction (travel time estimate)
- Uncertainty quantification via prediction intervals (lower/upper bounds)
- Load balancing across experts

Adapted for LibCity framework by inheriting from AbstractTrafficStateModel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


# ============================================================================
# MLP Component
# ============================================================================

class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer perceptron with batch normalization and ReLU activation.

    Args:
        input_dim: Input feature dimension
        embed_dims: Tuple of hidden layer dimensions
        dropout: Dropout rate (default: 0)
        output_layer: Whether to add a final linear layer to output dim 1
    """

    def __init__(self, input_dim, embed_dims, dropout=0, output_layer=False):
        super(MultiLayerPerceptron, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Float tensor of size (batch_size, input_dim) or (batch_size, seq_len, input_dim)

        Returns:
            Output tensor
        """
        return self.mlp(x)


# ============================================================================
# Regressor Component
# ============================================================================

class Regressor(nn.Module):
    """
    Multi-branch regressor that combines deep and recurrent features.

    This component fuses features from the deep pathway (e.g., origin-destination
    embeddings) with features from the recurrent pathway (LSTM output).

    Args:
        input_dim: Input dimension for both branches
        output_dim: Output dimension (before final output layer)
    """

    def __init__(self, input_dim, output_dim):
        super(Regressor, self).__init__()
        self.linear_wide = nn.Linear(input_dim, output_dim, bias=False)
        self.linear_deep = nn.Linear(input_dim, output_dim, bias=False)
        self.linear_recurrent = nn.Linear(input_dim, output_dim, bias=False)
        self.out_layer = MultiLayerPerceptron(output_dim, (output_dim,), output_layer=True)

    def forward(self, deep, recurrent):
        """
        Combine deep and recurrent features.

        Args:
            deep: Deep pathway features (batch_size, input_dim)
            recurrent: Recurrent pathway features (batch_size, input_dim)

        Returns:
            Output prediction (batch_size, 1)
        """
        fuse = self.linear_deep(deep) + self.linear_recurrent(recurrent)
        return self.out_layer(fuse)


# ============================================================================
# Noisy Top-K Router for MoE
# ============================================================================

class NoisyTopkRouter(nn.Module):
    """
    Noisy Top-K Router for Mixture of Experts.

    Selects top-k experts for each input with added noise for exploration.
    The noise helps prevent the router from always selecting the same experts.

    Args:
        n_embed: Input embedding dimension
        num_experts: Number of experts
        top_k: Number of experts to select for each input
    """

    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        """
        Route input to top-k experts.

        Args:
            mh_output: Input tensor (batch_size, seq_len, n_embed) or (batch_size, n_embed)

        Returns:
            router_output: Sparse gating weights (same shape as input but last dim is num_experts)
            indices: Indices of selected experts
            softmax_gating: Full softmax gating (for load balancing loss)
        """
        logits = self.topkroute_linear(mh_output)

        # Add noise for exploration
        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        # Select top-k experts
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)

        return router_output, indices, F.softmax(logits, dim=-1)


# ============================================================================
# Expert Network
# ============================================================================

class Expert(nn.Module):
    """
    Expert network: An MLP with 4x expansion.

    Each expert is a simple feedforward network with ReLU activation.

    Args:
        n_embd: Input/output embedding dimension
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Sparse Mixture of Experts
# ============================================================================

class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts layer.

    Only activates top-k experts for each input, making it computationally
    efficient while maintaining model capacity.

    Args:
        n_embed: Input embedding dimension
        num_experts: Total number of experts
        top_k: Number of experts to activate per input
        dropout: Dropout rate for experts (default: 0.1)
    """

    def __init__(self, n_embed, num_experts, top_k, dropout=0.1):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed, dropout) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        """
        Process input through sparse mixture of experts.

        Args:
            x: Input tensor (batch_size, seq_len, n_embed)

        Returns:
            final_output: Weighted combination of expert outputs
            softmax_gating_output: Full softmax gating (for load balancing)
        """
        gating_output, indices, softmax_gating_output = self.router(x)
        final_output = torch.zeros_like(x)

        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.experts):
            # Find which samples selected this expert
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Weight by gating score
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output, softmax_gating_output


# ============================================================================
# DutyTTE Model (LibCity Adapted)
# ============================================================================

class DutyTTE(AbstractTrafficStateModel):
    """
    DutyTTE: Travel Time Estimation with Mixture of Experts and Uncertainty Quantification.

    This model provides:
    1. Point prediction of travel time
    2. Prediction intervals (lower and upper bounds) for uncertainty quantification
    3. Sparse Mixture of Experts for efficient and expressive modeling

    The model uses:
    - Segment embeddings for road segment representation
    - Node embeddings for origin-destination representation
    - Time slice embeddings for temporal context
    - LSTM for sequential processing of trajectory
    - SparseMoE for capturing diverse patterns
    - Three regressors for mean, lower, and upper bound predictions

    Loss function: Mean Interval Score (MIS) for proper uncertainty calibration

    Args:
        config: Configuration dictionary containing model hyperparameters
        data_feature: Dictionary containing data-specific features

    Config Parameters:
        - segment_dims: Number of road segments (default from data_feature or 12693)
        - node_dims: Number of nodes/intersections (default from data_feature or 4601)
        - id_embed_dim: Embedding dimension for IDs (default: 20)
        - slice_dims: Number of time slices (default: 145)
        - slice_embed_dim: Embedding dimension for time slices (default: 20)
        - hidden_size: LSTM hidden size and regressor dimension (default: 128)
        - num_experts: Number of experts in MoE (default: 8)
        - top_k: Number of experts to activate (default: 2)
        - n_embed: Embedding dimension for distribution (default: 128)
        - m: Number of distribution parameters (default: 5)
        - alpha: Weight for interval loss (default: 0.1)
        - load_balance_weight: Weight for load balancing loss (default: 0.01)
        - dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, config, data_feature):
        super(DutyTTE, self).__init__(config, data_feature)

        self.config = config
        self.data_feature = data_feature
        self.device = config.get('device', torch.device('cpu'))

        # Dimension parameters - can be overridden by data_feature
        # Default values from original implementation
        segment_dims = data_feature.get('segment_dims', config.get('segment_dims', 12693))
        node_dims = data_feature.get('node_dims', config.get('node_dims', 4601))

        # Embedding dimensions
        id_embed_dim = config.get('id_embed_dim', 20)
        slice_dims = config.get('slice_dims', 145)
        slice_embed_dim = config.get('slice_embed_dim', 20)

        # Model dimensions
        hidden_size = config.get('hidden_size', 128)
        mlp_out_dim = hidden_size
        lstm_hidden_size = hidden_size
        reg_input_dim = hidden_size
        reg_output_dim = hidden_size
        deep_mlp_dims = (hidden_size,)

        # MoE parameters
        num_experts = config.get('num_experts', 8)
        top_k = config.get('top_k', 2)
        n_embed = config.get('n_embed', 128)
        dropout = config.get('dropout', 0.1)

        # Distribution embedding parameter
        # m is the number of distribution parameters (e.g., histogram bins)
        m = config.get('m', 5)

        # Loss weights
        self.alpha = config.get('alpha', 0.1)  # Weight for interval loss
        self.load_balance_weight = config.get('load_balance_weight', 0.01)

        # Time normalization parameters from data features
        self.time_mean = data_feature.get('time_mean', 0.0)
        self.time_std = data_feature.get('time_std', 1.0)

        # Distribution embedding: maps segment travel time features to n_embed
        # Input: m * 2 + 1 (e.g., histogram + cumulative + count)
        self.distribution_embed = nn.Linear(m * 2 + 1, n_embed)

        # Sparse Mixture of Experts
        self.MoEUQ = SparseMoE(reg_input_dim, num_experts, top_k, dropout)

        # ID embeddings
        self.segment_embedding = nn.Embedding(segment_dims, id_embed_dim)
        self.node_embedding = nn.Embedding(node_dims, id_embed_dim)

        # Time slice embedding
        self.slice_embedding = nn.Embedding(slice_dims, slice_embed_dim)

        # Feature fusion MLP
        # Input: id_embed + slice_embed + n_embed
        self.all_mlp = nn.Sequential(
            nn.Linear(id_embed_dim + slice_embed_dim + n_embed, mlp_out_dim),
            nn.ReLU()
        )

        # Sequence encoder
        self.lstm = nn.LSTM(
            input_size=mlp_out_dim,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True
        )

        # Multi-branch regressors for uncertainty quantification
        self.regressor = Regressor(reg_input_dim, reg_output_dim)  # Mean prediction
        self.regressor_lower = Regressor(reg_input_dim, reg_output_dim)  # Lower bound
        self.regressor_upper = Regressor(reg_input_dim, reg_output_dim)  # Upper bound

        # Deep pathway MLP for origin-destination features
        # Input: 1 (time) + 2 * id_embed (origin + destination)
        self.deep_mlp = MultiLayerPerceptron(1 + id_embed_dim * 2, deep_mlp_dims)

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)

    def forward(self, batch):
        """
        Forward pass of DutyTTE model.

        Args:
            batch: Dictionary containing:
                - 'segments': Segment IDs [batch_size, seq_len]
                - 'segment_travel_time': Segment travel time features [batch_size, seq_len, m*2+1]
                - 'num_segments': Number of segments per trajectory [batch_size] or [batch_size, 1]
                - 'start_time': Start time index (10-min buckets) [batch_size] or [batch_size, 1]
                - 'origin': Origin node IDs [batch_size]
                - 'destination': Destination node IDs [batch_size]

        Returns:
            hat_y: Mean travel time prediction [batch_size, 1]
            bias_lower: Lower bound offset [batch_size, 1]
            bias_upper: Upper bound offset [batch_size, 1]
            load_balancing_loss: Load balancing loss for MoE
        """
        # Extract inputs from batch
        xs = batch['segments']  # [batch_size, seq_len]
        segment_travel_time = batch['segment_travel_time']  # [batch_size, seq_len, features]
        number_of_roadsegments = batch['num_segments']  # [batch_size] or [batch_size, 1]
        start_ts_10min = batch['start_time']  # [batch_size] or [batch_size, 1]

        # Handle origin-destination as separate keys or combined
        # Use try-except instead of 'in' operator since LibCity's Batch class
        # does not implement __contains__ method
        try:
            od = batch['od']  # [batch_size, 2]
            origin = od[:, 0]
            destination = od[:, 1]
        except KeyError:
            origin = batch['origin']
            destination = batch['destination']

        # Ensure correct dimensions for all scalar inputs
        if number_of_roadsegments.dim() == 2:
            number_of_roadsegments = number_of_roadsegments.squeeze(-1)
        if start_ts_10min.dim() == 2:
            start_ts_10min = start_ts_10min.squeeze(-1)
        if origin.dim() == 2:
            origin = origin.squeeze(-1)
        if destination.dim() == 2:
            destination = destination.squeeze(-1)

        device = xs.device

        # Origin-destination embeddings - origin/destination are now 1D [batch_size]
        o_embed = self.node_embedding(origin.long().to(device))  # [batch, embed_dim]
        d_embed = self.node_embedding(destination.long().to(device))  # [batch, embed_dim]

        # Deep pathway: combine start time with OD embeddings
        deep_output = self.deep_mlp(torch.cat([
            start_ts_10min.float().unsqueeze(-1).to(device),  # [batch, 1]
            o_embed.to(device),  # [batch, embed_dim]
            d_embed.to(device)   # [batch, embed_dim]
        ], dim=-1))  # Result: [batch, 1 + embed_dim + embed_dim]

        # Segment embeddings
        all_id_embedding = self.segment_embedding(xs.long().to(device))  # [batch, seq, embed]

        # Time slice embeddings - expand to sequence length
        all_slice_embedding = self.slice_embedding(
            start_ts_10min.unsqueeze(1).expand(-1, xs.shape[1]).long().to(device)
        )  # [batch, seq, embed]

        # Distribution embeddings for segment travel times
        all_real = self.distribution_embed(segment_travel_time.to(device))  # [batch, seq, n_embed]

        # Concatenate all features
        all_input = torch.cat([all_id_embedding, all_slice_embedding, all_real], dim=2)

        # Feature fusion
        recurrent_input = self.all_mlp(all_input)

        # Pack for LSTM (handle variable length sequences)
        packed_all_input = pack_padded_sequence(
            recurrent_input,
            number_of_roadsegments.cpu().long(),
            enforce_sorted=False,
            batch_first=True
        )

        seq_out, _ = self.lstm(packed_all_input)
        seq_out, _ = pad_packed_sequence(seq_out, batch_first=True)

        B, N_valid = seq_out.shape[0], seq_out.shape[1]

        # Apply Mixture of Experts
        seq_out, softmax_gating_output = self.MoEUQ(seq_out)

        # Mask for valid sequence positions
        mask_indices = torch.arange(N_valid, device=device).unsqueeze(0).expand(B, -1)
        mask = (mask_indices < number_of_roadsegments.unsqueeze(1).to(device)).unsqueeze(-1).float()

        # Sum pooling over valid positions
        seq_out = torch.sum(seq_out * mask, dim=1)

        # Multi-branch prediction
        hat_y = self.regressor(deep_output, seq_out)
        bias_upper = self.regressor_upper(deep_output, seq_out)
        bias_lower = self.regressor_lower(deep_output, seq_out)

        # Calculate load balancing loss
        valid_gating_output = softmax_gating_output * mask
        expert_load = torch.sum(valid_gating_output, dim=(0, 1))
        total_load = torch.sum(expert_load)
        normalized_load = expert_load / (total_load + 1e-9)
        num_experts = softmax_gating_output.shape[-1]
        ideal_load = 1.0 / num_experts
        load_balancing_loss = torch.sum(
            normalized_load * torch.log(normalized_load / ideal_load + 1e-9)
        )

        return hat_y, bias_lower, bias_upper, load_balancing_loss

    def predict(self, batch):
        """
        Predict travel times for a batch.

        Returns only the mean prediction for evaluation.

        Args:
            batch: Input batch dictionary

        Returns:
            Predicted travel times [batch_size, 1]
        """
        hat_y, bias_lower, bias_upper, _ = self.forward(batch)
        return hat_y  # Return [batch_size, 1] to match ground truth shape

    def predict_with_uncertainty(self, batch):
        """
        Predict travel times with uncertainty bounds.

        Args:
            batch: Input batch dictionary

        Returns:
            dict with keys:
                - 'prediction': Mean prediction [batch_size]
                - 'lower_bound': Lower bound of prediction interval [batch_size]
                - 'upper_bound': Upper bound of prediction interval [batch_size]
        """
        hat_y, bias_lower, bias_upper, _ = self.forward(batch)

        # Convert offsets to absolute bounds
        # Lower bound: prediction - abs(bias_lower)
        # Upper bound: prediction + abs(bias_upper)
        lower_bound = hat_y - torch.abs(bias_lower)
        upper_bound = hat_y + torch.abs(bias_upper)

        return {
            'prediction': hat_y.squeeze(-1),
            'lower_bound': lower_bound.squeeze(-1),
            'upper_bound': upper_bound.squeeze(-1)
        }

    def calculate_loss(self, batch):
        """
        Calculate the combined loss for training.

        The loss consists of:
        1. Mean Interval Score (MIS) loss for uncertainty quantification
        2. Load balancing loss for MoE

        MIS loss penalizes:
        - Wide prediction intervals
        - Observations falling outside the prediction interval

        Args:
            batch: Input batch dictionary with 'time' as ground truth

        Returns:
            Combined loss tensor
        """
        hat_y, bias_lower, bias_upper, load_balancing_loss = self.forward(batch)

        # Get ground truth travel time
        y_true = batch['time']
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(1)
        y_true = y_true.to(hat_y.device)

        # Calculate prediction bounds
        # Ensure bounds are positive offsets
        bias_lower = torch.abs(bias_lower)
        bias_upper = torch.abs(bias_upper)

        lower_bound = hat_y - bias_lower
        upper_bound = hat_y + bias_upper

        # Mean Interval Score (MIS) loss
        # Penalizes both interval width and coverage violations
        interval_width = upper_bound - lower_bound

        # Coverage penalties
        lower_violation = torch.clamp(lower_bound - y_true, min=0)
        upper_violation = torch.clamp(y_true - upper_bound, min=0)

        # MIS loss = interval_width + 2/alpha * (lower_violation + upper_violation)
        # alpha is the desired coverage level (e.g., 0.1 for 90% coverage)
        mis_loss = torch.mean(
            interval_width + (2.0 / self.alpha) * (lower_violation + upper_violation)
        )

        # Point prediction loss (MAE)
        point_loss = F.l1_loss(hat_y, y_true)

        # Combined loss
        total_loss = point_loss + mis_loss + self.load_balance_weight * load_balancing_loss

        return total_loss
