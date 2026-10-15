"""Concatenation fusion with a linear projection back to ``hidden_dim``.

Category: fusion
Interface: ``forward(h_sp: (B, T, N, d), h_tm: (B, T, N, d)) -> (B, T, N, d)``.

Principle: the two branch tensors are concatenated along the feature axis and
mapped back to ``hidden_dim`` by a single ``Linear(2 * hidden_dim, hidden_dim)``.
Nothing is discarded, and the linear layer learns which combination of the two
branches matters.

Applicable when the spatial and temporal representations are complementary and a
little extra capacity is affordable; parameters ``2 * hidden_dim^2``. Mismatched
branch widths are projected to ``hidden_dim`` first, so the module never fails
on shape disagreement.

Reference: ``LibCity traffic_speed_prediction/GWNET.py``
(``GWNET`` stacks residual / skip features along the channel axis and fuses them
through ``end_conv`` linear layers; ``OutputLayer`` in
``traffic_flow_prediction/STSGCN.py`` uses the same flatten-and-project idea).
"""

import torch
import torch.nn as nn

__all__ = ["ConcatFusion"]


class ConcatFusion(nn.Module):
    """Concatenate two representations and project them back."""

    def __init__(self, hidden_dim=64, dropout=0.0, bias=True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.proj = nn.Linear(2 * self.hidden_dim, self.hidden_dim, bias=bias)
        self.proj_sp = None
        self.proj_tm = None

    def _project(self, h, spatial):
        if h.shape[-1] == self.hidden_dim:
            return h
        branch = self.proj_sp if spatial else self.proj_tm
        if branch is None or branch.in_features != h.shape[-1]:
            branch = nn.Linear(h.shape[-1], self.hidden_dim).to(device=h.device, dtype=h.dtype)
            if spatial:
                self.proj_sp = branch
            else:
                self.proj_tm = branch
        return branch(h)

    def _align(self, h_sp, h_tm):
        if h_sp.shape[-1] == h_tm.shape[-1] == self.hidden_dim:
            return h_sp, h_tm
        return self._project(h_sp, True), self._project(h_tm, False)

    def forward(self, h_sp, h_tm):
        h_sp, h_tm = self._align(h_sp, h_tm)
        return self.dropout(self.proj(torch.cat([h_sp, h_tm], dim=-1)))
