"""Cross-attention fusion between the spatial and temporal branches.

Category: fusion
Interface: ``forward(h_sp: (B, T, N, d), h_tm: (B, T, N, d)) -> (B, T, N, d)``.

Principle: the temporal branch supplies the queries while the spatial branch
supplies keys and values (both folded to ``B*N`` independent sequences over
time), so every time step of the temporal representation can attend to the
spatial context it needs. Multi-head attention keeps several interaction
patterns in parallel, followed by an output projection back to ``hidden_dim``.

Applicable when the two branches must interact explicitly rather than being
averaged; cost ``O(B * N * T^2 * d)`` with ``4 * d^2`` projection parameters.
Mismatched branch widths are projected to ``hidden_dim`` first.

Reference: ``LibCity traffic_speed_prediction/GMAN.py``
(``TransformAttention``: separate query/key/value projections then scaled
dot-product attention across representations) and
``LibCity traffic_flow_prediction/DSTAGNN.py``
(``MultiHeadAttention``).
"""

import torch
import torch.nn as nn

__all__ = ["CrossAttentionFusion"]


class CrossAttentionFusion(nn.Module):
    """Multi-head cross-attention of the temporal onto the spatial branch."""

    def __init__(self, hidden_dim=64, num_heads=4, dropout=0.0, bias=True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads) if int(num_heads) > 0 and self.hidden_dim % int(num_heads) == 0 else 1
        self.attention = nn.MultiheadAttention(
            self.hidden_dim, self.num_heads, dropout=float(dropout), bias=bias, batch_first=True
        )
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)
        self.dropout = nn.Dropout(float(dropout))
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

    @staticmethod
    def _flatten(h):
        batch, steps, nodes, channels = h.shape
        return h.permute(0, 2, 1, 3).reshape(batch * nodes, steps, channels)

    def forward(self, h_sp, h_tm):
        h_sp, h_tm = self._align(h_sp, h_tm)
        batch, steps, nodes, channels = h_tm.shape
        query = self._flatten(h_tm)
        key_value = self._flatten(h_sp)
        out, _ = self.attention(query, key_value, key_value)
        out = self.out_proj(self.dropout(out))
        return out.reshape(batch, nodes, steps, channels).permute(0, 2, 1, 3)
