"""Linear feature embedding layer (input projection + ReLU).

Category: encoder
Interface: ``forward(x: (B, T, N, C)) -> (B, T, N, d)`` with ``d = hidden_dim``.

Principle: a single ``Linear(C -> hidden_dim)`` followed by ReLU lifts the raw
feature channels into the hidden width used by the spatio-temporal backbone.
This is the ``first_layer_embedding`` of STSGCN / STFGNN.

Applicable as the very first layer of any graph or attention model; cost is
``O(B*T*N*C*d)`` with ``C*d`` parameters, so it is negligible next to the
backbone.

Reference: ``LibCity traffic_flow_prediction/STSGCN.py``
(``STSGCN.first_layer_embedding = nn.Linear(num_of_features, 64)``; the
reference layer is linear only, this module additionally applies the ReLU
non-linearity requested by the library interface).
"""

import torch
import torch.nn as nn

__all__ = ["FeatureEmbeddingEncoder"]


class FeatureEmbeddingEncoder(nn.Module):
    """Project ``C`` input channels to ``hidden_dim`` hidden channels."""

    def __init__(self, input_dim=3, hidden_dim=64, dropout=0.0, bias=True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.proj = nn.Linear(self.input_dim, self.hidden_dim, bias=bias)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if self.proj.in_features != x.shape[-1]:
            self.proj = nn.Linear(x.shape[-1], self.hidden_dim).to(device=x.device, dtype=x.dtype)
        return self.dropout(self.act(self.proj(x)))
