"""Gated dilated temporal convolution with exponentially growing dilation.

Category: temporal
Interface: ``forward(x: (B, T, N, d)) -> (B, T', N, d)`` with ``T' == T``.

Principle: the WaveNet-style block of GWNET. Each layer applies two dilated
convolutions along time with dilation ``base * 2^i`` and combines them
multiplicatively (``tanh(filter) * sigmoid(gate)``), which gives the receptive
field an exponential growth in the number of layers. The convolution is causal:
the sequence is left-padded by ``dilation * (kernel_size - 1)`` and the
resulting overshoot is trimmed, therefore the output length equals the input
length ``T' == T`` (dilated convolutions may legitimately shorten the sequence;
this implementation keeps it, and the trimming is the only length rule).

Applicable when a bounded, parallelisable receptive field is enough; cost is
``O(L * d^2 * k * T * N)`` with ``O(L * d^2 * k)`` parameters.

Reference: ``LibCity traffic_speed_prediction/GWNET.py``
(the dilated ``filter_convs`` / ``gate_convs`` pair inside the WaveNet loop).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DilatedTCN"]


class DilatedTCN(nn.Module):
    """Causal, gated, exponentially dilated temporal convolution stack."""

    def __init__(self, hidden_dim=64, num_layers=3, kernel_size=2, dropout=0.0, input_dim=None, base_dilation=1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = max(1, int(num_layers))
        self.kernel_size = max(2, int(kernel_size))
        self.dropout = float(dropout)
        self.dilations = [max(1, int(base_dilation)) * (2 ** i) for i in range(self.num_layers)]
        in_dim = self.hidden_dim if input_dim is None else int(input_dim)
        self.in_proj = nn.Identity() if in_dim == self.hidden_dim else nn.Linear(in_dim, self.hidden_dim)
        self.filter_convs = nn.ModuleList(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, (1, self.kernel_size), dilation=(1, d))
            for d in self.dilations
        )
        self.gate_convs = nn.ModuleList(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, (1, self.kernel_size), dilation=(1, d))
            for d in self.dilations
        )
        self.skip_convs = nn.ModuleList(nn.Conv2d(self.hidden_dim, self.hidden_dim, (1, 1)) for _ in self.dilations)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        hidden = self.in_proj(x).permute(0, 3, 2, 1)
        for index, dilation in enumerate(self.dilations):
            residual = hidden
            pad = dilation * (self.kernel_size - 1)
            if pad > 0:
                hidden = F.pad(hidden, (pad, 0))
            hidden = torch.tanh(self.filter_convs[index](hidden)) * torch.sigmoid(self.gate_convs[index](hidden))
            hidden = self.skip_convs[index](hidden)
            hidden = self.dropout_layer(hidden) + residual[:, :, :, -hidden.shape[-1]:]
        return hidden.permute(0, 3, 2, 1)
