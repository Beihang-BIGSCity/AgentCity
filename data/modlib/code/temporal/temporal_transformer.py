"""Stacked temporal self-attention transformer with positional encoding.

Category: temporal
Interface: ``forward(x: (B, T, N, d)) -> (B, T', N, d)`` with ``T' == T``.

Principle: each node becomes a sequence of length ``T``; sinusoidal positional
encodings are added so the permutation-invariant attention keeps temporal order,
then ``num_layers`` pre-norm transformer encoder layers model the temporal
context with multi-head self-attention. Attention runs over time only, so the
node axis is treated as an independent batch dimension.

Applicable when long-range or bursty temporal structure matters (STAEformer /
PDFormer temporal encoder). Cost is ``O(B * N * T^2 * d)`` memory and time, which
grows quadratically in the input window.

Reference: ``LibCity traffic_speed_prediction/STAEformer.py``
(``SelfAttentionLayer``: attention + feed-forward with layer norm and residual);
``LibCity traffic_flow_prediction/PDFormer.py`` uses the
same temporal-attention recipe.
"""

import math

import torch
import torch.nn as nn

__all__ = ["TemporalTransformer"]


class TemporalTransformer(nn.Module):
    """Transformer encoder over the time axis with ``T' == T``."""

    def __init__(
        self,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        dim_feedforward=None,
        input_dim=None,
        max_len=512,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = max(1, int(num_layers))
        self.num_heads = int(num_heads) if int(num_heads) > 0 and self.hidden_dim % int(num_heads) == 0 else 1
        self.max_len = max(1, int(max_len))
        in_dim = self.hidden_dim if input_dim is None else int(input_dim)
        self.in_proj = nn.Identity() if in_dim == self.hidden_dim else nn.Linear(in_dim, self.hidden_dim)
        feedforward = int(dim_feedforward) if dim_feedforward else 4 * self.hidden_dim
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=feedforward,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.register_buffer("position", self._sinusoidal(self.max_len, self.hidden_dim), persistent=False)

    @staticmethod
    def _sinusoidal(length, width):
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
        encoding = torch.zeros(length, width)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        return encoding

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.in_proj(x)
        batch, steps, nodes, channels = x.shape
        sequence = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, channels)
        if steps > self.position.shape[0]:
            self.position = self._sinusoidal(steps, self.hidden_dim).to(device=x.device, dtype=x.dtype)
        out = self.encoder(sequence + self.position[:steps].to(device=x.device, dtype=x.dtype))
        return out.reshape(batch, nodes, steps, channels).permute(0, 2, 1, 3)
