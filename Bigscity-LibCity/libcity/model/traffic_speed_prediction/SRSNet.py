"""
SRSNet: Enhancing Time Series Forecasting through Selective Representation Spaces

Paper: Enhancing Time Series Forecasting through Selective Representation Spaces: A Patch Perspective
Source: https://github.com/decisionintelligence/SRSNet
Venue: NeurIPS 2025 (Spotlight)

Adapted for LibCity by integrating the upstream repository code.
"""

import math
import torch
import torch.nn as nn
from logging import getLogger
from einops import rearrange

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


class PositionalEmbedding(nn.Module):
    """Positional encoding for patch sequences."""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class SRS(nn.Module):
    """Selective Representation Space module."""

    def __init__(self, d_model, patch_len, stride, seq_len, dropout, hidden_size, alpha=2.0, pos=True):
        super(SRS, self).__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.seq_len = seq_len

        self.patch_num = math.ceil((self.seq_len - self.patch_len) / self.stride) + 1
        self.padding = self.patch_len + (self.patch_num - 1) * self.stride - self.seq_len
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.padding))

        # Selective patching scorer
        self.scorer_select = nn.Sequential(
            nn.Linear(self.patch_len, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.patch_num)
        )

        # Dynamic reassembly scorer
        self.scorer_shuffle = nn.Sequential(
            nn.Linear(self.patch_len, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Patch embedding layers
        self.value_embedding_org = nn.Linear(patch_len, d_model, bias=False)
        self.value_embedding_rec = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        if pos:
            self.position_embedding = PositionalEmbedding(d_model)

        self.pos = pos
        self.dropout = nn.Dropout(dropout)

        # Adaptive fusion weight
        self.alpha = nn.Parameter(torch.ones(self.patch_num, d_model) * alpha)

    def _origin_view(self, x):
        """Original view: standard patching."""
        # [batch_size, n_vars, patch_num, patch_size]
        x_origin = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # [batch_size * n_vars, patch_num, patch_size]
        origin_patches = rearrange(x_origin, 'b c n p -> (b c) n p')
        return origin_patches

    def _rec_view(self, x):
        """Reconstruction view: selective patching + dynamic reassembly."""
        # [batch_size, n_vars, seq_len - patch_size + 1, patch_size]
        x_rec = x.unfold(dimension=-1, size=self.patch_len, step=1)
        # Selective patching
        selected_patches = self._select(x_rec)
        # Dynamic reassembly
        shuffled_patches = self._shuffle(selected_patches)
        # [batch_size * n_vars, patch_num, patch_size]
        rec_patches = rearrange(shuffled_patches, 'b c n p -> (b c) n p')
        return rec_patches

    def _select(self, x_rec):
        """Selective patching: select most informative patches."""
        # [batch_size, n_vars, seq_len - patch_size + 1, select_num]
        scores = self.scorer_select(x_rec)
        # [batch_size, n_vars, 1, select_num]
        indices = torch.argmax(scores, dim=-2, keepdim=True)
        # [batch_size, n_vars, 1, select_num]
        max_scores = torch.gather(input=scores, dim=-2, index=indices)
        non_zero_mask = max_scores != 0
        inv = (1 / max_scores[non_zero_mask]).detach()

        # [batch_size, n_vars, select_num, patch_size]
        x_rec_indices = indices.repeat(1, 1, self.patch_len, 1).permute(0, 1, 3, 2)
        # [batch_size, n_vars, select_num, patch_size]
        selected_patches = torch.gather(input=x_rec, index=x_rec_indices, dim=-2)

        max_scores[non_zero_mask] *= inv
        # [batch_size, n_vars, select_num, patch_size]
        selected_patches = max_scores.permute(0, 1, 3, 2) * selected_patches

        return selected_patches

    def _shuffle(self, selected_patches):
        """Dynamic reassembly: reorder patches based on importance."""
        # [batch_size, n_vars, patch_num, 1]
        shuffle_scores = self.scorer_shuffle(selected_patches)
        # [batch_size, n_vars, patch_num, 1]
        shuffle_indices = torch.argsort(input=shuffle_scores, dim=-2, descending=True)
        # [batch_size, n_vars, patch_num, 1]
        shuffled_scores = torch.gather(input=shuffle_scores, index=shuffle_indices, dim=-2)
        non_zero_mask = shuffled_scores != 0
        inv = (1 / shuffled_scores[non_zero_mask]).detach()

        # [batch_size, n_vars, patch_num, patch_size]
        shuffle_patch_indices = shuffle_indices.repeat(1, 1, 1, self.patch_len)
        # [batch_size, n_vars, patch_num, patch_size]
        shuffled_patches = torch.gather(input=selected_patches, index=shuffle_patch_indices, dim=-2)
        shuffled_scores[non_zero_mask] *= inv
        # [batch_size, n_vars, patch_num, patch_size]
        shuffled_patches = shuffled_scores * shuffled_patches

        return shuffled_patches

    def forward(self, x):
        """
        Args:
            x: [batch_size, n_vars, seq_len]

        Returns:
            embedding: [batch_size * n_vars, patch_num, d_model]
            n_vars: number of variables
        """
        n_vars = x.shape[1]
        # Padding for patching
        x = self.padding_patch_layer(x)

        # Get two representation spaces
        rec_repr_space = self._rec_view(x)
        original_repr_space = self._origin_view(x)

        # Adaptive fusion
        weight = torch.sigmoid(self.alpha)
        embedding = weight * self.value_embedding_org(original_repr_space) \
                    + (1 - weight) * self.value_embedding_rec(rec_repr_space)

        if self.pos:
            position_embedding = self.position_embedding(original_repr_space)
            embedding = embedding + position_embedding

        return self.dropout(embedding), n_vars


class RevIN(nn.Module):
    """Reversible Instance Normalization."""

    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class FlattenHead(nn.Module):
    """Prediction head."""

    def __init__(self, n_vars, nf, target_window, head_dropout=0, mode='linear'):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        if mode == 'linear':
            self.head = nn.Linear(nf, target_window)
        else:
            self.head = nn.Sequential(
                nn.Linear(nf, nf // 2),
                nn.SiLU(),
                nn.Linear(nf // 2, target_window)
            )
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.head(x)
        x = self.dropout(x)
        return x


class SRSNet(AbstractTrafficStateModel):
    """
    SRSNet: Enhancing Time Series Forecasting through Selective Representation Spaces

    Paper: Enhancing Time Series Forecasting through Selective Representation Spaces: A Patch Perspective
    Source: https://github.com/decisionintelligence/SRSNet
    Venue: NeurIPS 2025 (Spotlight)

    This model uses selective patching and dynamic reassembly to enhance patch-based forecasting.
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        # Get data features
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self._logger = getLogger()

        # Get model config
        self.input_window = config.get('input_window', 96)
        self.output_window = config.get('output_window', 96)
        self.device = config.get('device', torch.device('cpu'))

        # SRSNet specific parameters
        self.patch_len = config.get('patch_len', 16)
        self.stride = config.get('stride', 8)
        self.d_model = config.get('d_model', 128)
        self.hidden_size = config.get('hidden_size', 256)
        self.dropout = config.get('dropout', 0.1)
        self.alpha = config.get('alpha', 2.0)
        self.pos = config.get('pos', True)
        self.head_mode = config.get('head_mode', 'linear')
        self.affine = config.get('affine', True)
        self.subtract_last = config.get('subtract_last', False)

        # Calculate number of variables (nodes in traffic forecasting)
        self.enc_in = self.num_nodes

        # Selective Representation Space module
        self.patch_embedding = SRS(
            self.d_model, self.patch_len, self.stride, self.input_window,
            self.dropout, self.hidden_size, self.alpha, self.pos
        )

        # Prediction Head
        self.head_nf = self.d_model * (math.ceil((self.input_window - self.patch_len) / self.stride) + 1)
        self.head = FlattenHead(
            self.enc_in,
            self.head_nf,
            self.output_window,
            head_dropout=self.dropout,
            mode=self.head_mode
        )

        # Reversible Instance Normalization
        self.revin = RevIN(
            num_features=self.enc_in,
            affine=self.affine,
            subtract_last=self.subtract_last
        )

        self._logger.info(f"SRSNet initialized with {self.num_nodes} nodes, "
                         f"input_window={self.input_window}, output_window={self.output_window}")
        self._logger.info(f"Patch config: patch_len={self.patch_len}, stride={self.stride}, d_model={self.d_model}")

    def forward(self, batch):
        """
        Forward pass of SRSNet.

        Args:
            batch: dict with key 'X': [batch_size, input_window, num_nodes, feature_dim]

        Returns:
            predictions: [batch_size, output_window, num_nodes, output_dim]
        """
        # Extract input from batch
        x = batch['X']  # [batch_size, input_window, num_nodes, feature_dim]
        batch_size, seq_len, num_nodes, feature_dim = x.shape

        # Use only the traffic values (first feature)
        x_enc = x[..., 0]  # [batch_size, input_window, num_nodes]

        # Apply RevIN normalization
        x_enc = self.revin(x_enc, 'norm')

        # Reshape for patch embedding: [batch_size, num_nodes, input_window]
        x_enc = x_enc.permute(0, 2, 1)

        # Apply SRS module: [batch_size * num_nodes, patch_num, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Reshape back: [batch_size, num_nodes, patch_num, d_model]
        enc_out = torch.reshape(
            enc_out, (batch_size, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )

        # Transpose for head: [batch_size, num_nodes, d_model, patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Prediction head: [batch_size, num_nodes, output_window]
        dec_out = self.head(enc_out)

        # Transpose to match expected output format: [batch_size, output_window, num_nodes]
        dec_out = dec_out.permute(0, 2, 1)

        # De-normalization
        dec_out = self.revin(dec_out, 'denorm')

        # Add feature dimension: [batch_size, output_window, num_nodes, 1]
        dec_out = dec_out.unsqueeze(-1)

        return dec_out

    def calculate_loss(self, batch):
        """
        Calculate loss between predictions and ground truth.

        Args:
            batch: dict with keys 'X' (input) and 'y' (ground truth)

        Returns:
            loss: scalar tensor
        """
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        """
        Make predictions.

        Args:
            batch: dict with key 'X' (input)

        Returns:
            predictions: [batch_size, output_window, num_nodes, output_dim]
        """
        return self.forward(batch)
