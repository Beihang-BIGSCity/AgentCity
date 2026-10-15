"""Learned scalar weights over the two branches (gated fusion).

Category: fusion
Interface: ``forward(h_sp: (B, T, N, d), h_tm: (B, T, N, d)) -> (B, T, N, d)``.

Principle: two learnable scalars are normalised with a softmax and used as a
convex combination of the branches, ``w0 * h_sp + w1 * h_tm``. The model can
therefore decide how much of the spatial versus temporal branch to use without
any per-position cost.

Applicable when a single global mixing ratio is enough; parameters are 2 scalars
(plus optional width projections). A per-position gate (GMAN's ``GatedFusion``)
is the richer alternative, this module stays at the cheap end of the spectrum.

Reference: ``LibCity traffic_speed_prediction/GMAN.py``
(``GatedFusion``: a sigmoid gate mixes the spatial ``HS`` and temporal ``HT``
representations; this module uses the softmax normalised form of the same idea).
"""

import torch
import torch.nn as nn

__all__ = ["WeightedFusion"]


class WeightedFusion(nn.Module):
    """Softmax-weighted convex combination of two representations."""

    def __init__(self, hidden_dim=64, dropout=0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.branch_weight = nn.Parameter(torch.zeros(2))
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
        weights = torch.softmax(self.branch_weight, dim=0)
        return self.dropout(weights[0] * h_sp + weights[1] * h_tm)
