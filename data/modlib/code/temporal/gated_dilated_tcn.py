"""GLU-gated dilated causal convolution (STSGCN / STFGNN temporal branch).

Category: temporal
Interface: ``forward(x: (B, T, N, d)) -> (B, T', N, d)`` with ``T' == T``.

Principle: one convolution emits ``2 * d`` channels; the result is split in half
and combined as ``a * sigmoid(b)`` (a gated linear unit), which lets the layer
gate its own activations without a second parameterised branch. Dilations grow
as ``base * 2^i`` so a few layers cover a long window. Padding is causal (left
only) and the overshoot is trimmed, hence ``T' == T``.

Applicable where the temporal branch is stacked alternately with graph blocks
inside a joint spatio-temporal block; cost ``O(L * d^2 * k * T * N)`` with
``O(2 * L * d^2 * k)`` parameters.

Reference: ``LibCity traffic_flow_prediction/STSGCN.py``
(``STSGCL``: temporal convolution followed by ``GLU``, i.e. ``a * sigmoid(b)``);
the STFGNN ``STSGCL`` temporal branch follows the same recipe.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GatedDilatedTCN"]


class GatedDilatedTCN(nn.Module):
    """GLU-gated dilated causal temporal convolution stack."""

    def __init__(self, hidden_dim=64, num_layers=2, kernel_size=2, dropout=0.0, input_dim=None, base_dilation=1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = max(1, int(num_layers))
        self.kernel_size = max(2, int(kernel_size))
        self.dropout = float(dropout)
        self.dilations = [max(1, int(base_dilation)) * (2 ** i) for i in range(self.num_layers)]
        in_dim = self.hidden_dim if input_dim is None else int(input_dim)
        self.in_proj = nn.Identity() if in_dim == self.hidden_dim else nn.Linear(in_dim, self.hidden_dim)
        self.convs = nn.ModuleList(
            nn.Conv2d(self.hidden_dim, 2 * self.hidden_dim, (1, self.kernel_size), dilation=(1, d))
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
            content, gate = torch.chunk(self.convs[index](hidden), 2, dim=1)
            hidden = self.skip_convs[index](content * torch.sigmoid(gate))
            hidden = self.dropout_layer(hidden) + residual[:, :, :, -hidden.shape[-1]:]
        return hidden.permute(0, 3, 2, 1)
