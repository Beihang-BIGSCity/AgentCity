"""Element-wise addition of the spatial and temporal representations.

Category: fusion
Interface: ``forward(h_sp: (B, T, N, d), h_tm: (B, T, N, d)) -> (B, T, N, d)``.

Principle: the two branch tensors are simply summed. When the two last
dimensions differ (or differ from ``hidden_dim``) each branch is first projected
to ``hidden_dim`` with its own ``Linear`` so the shapes always line up.

Applicable when both branches already live in a comparable scale and a
parameter-free fusion is preferred (it is the cheapest possible combination);
with matching widths the module carries no parameters at all, otherwise two
``d x hidden_dim`` projections.

Reference: ``LibCity traffic_flow_prediction/ASTGCN.py``
(``FusionLayer``: the attention branches are combined by a matrix-shaped
trainable weighting before being summed / reshaped).
"""

import torch
import torch.nn as nn

__all__ = ["AddFusion"]


class AddFusion(nn.Module):
    """Sum two ``(B, T, N, d)`` representations."""

    def __init__(self, hidden_dim=64, dropout=0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.proj_sp = None
        self.proj_tm = None

    def _project(self, h, spatial):
        if h.shape[-1] == self.hidden_dim:
            return h
        proj = self.proj_sp if spatial else self.proj_tm
        if proj is None or proj.in_features != h.shape[-1]:
            proj = nn.Linear(h.shape[-1], self.hidden_dim).to(device=h.device, dtype=h.dtype)
            if spatial:
                self.proj_sp = proj
            else:
                self.proj_tm = proj
        return proj(h)

    def _align(self, h_sp, h_tm):
        if h_sp.shape[-1] == h_tm.shape[-1] == self.hidden_dim:
            return h_sp, h_tm
        return self._project(h_sp, True), self._project(h_tm, False)

    def forward(self, h_sp, h_tm):
        h_sp, h_tm = self._align(h_sp, h_tm)
        return self.dropout(h_sp + h_tm)
