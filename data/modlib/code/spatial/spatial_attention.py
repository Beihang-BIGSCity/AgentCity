"""Multi-head attention across nodes (DSTAGNN / GMAN style).

Category: spatial
Interface: ``forward(x: (B, T, N, d), adj: (N, N) | None) -> (B, T, N, d)``.

Principle: for every time step the ``N`` node vectors are projected to queries,
keys and values per head and compared pairwise, giving an ``N x N`` spatial
attention map. When ``adj`` is supplied it is used as a mask (self loops stay
enabled, out-of-graph pairs are suppressed with a large negative bias) so the
attention respects the physical network; without ``adj`` the attention is fully
connected and learns the graph on its own.

Applicable when long-range or non-local spatial correlations matter more than a
fixed local topology. Attention scores cost ``O(B * T * heads * N^2)`` memory and
compute; parameters are the four ``d x d`` projections.

Reference: ``LibCity traffic_flow_prediction/ASTGCN.py``
(``SpatialAttentionLayer``, adjacency-free node attention) together with
``LibCity traffic_flow_prediction/DSTAGNN.py`` /
``LibCity traffic_speed_prediction/GMAN.py``
(multi-head node attention ``SMultiHeadAttention`` / ``SpatialAttention``).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SpatialAttention"]


class SpatialAttention(nn.Module):
    """Multi-head self-attention over the node axis, optionally masked by adj."""

    def __init__(self, hidden_dim=64, num_nodes=207, num_heads=4, dropout=0.0, bias=True, mask_value=-1e4):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_nodes = int(num_nodes)
        self.num_heads = int(num_heads) if int(num_heads) > 0 and self.hidden_dim % int(num_heads) == 0 else 1
        self.dropout = float(dropout)
        self.mask_value = float(mask_value)
        self.in_proj = None
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)

    def _align(self, x):
        if x.shape[-1] == self.hidden_dim:
            return x
        if self.in_proj is None or self.in_proj.in_features != x.shape[-1]:
            self.in_proj = nn.Linear(x.shape[-1], self.hidden_dim).to(device=x.device, dtype=x.dtype)
        return self.in_proj(x)

    def _split_heads(self, tensor, batch, steps, nodes):
        head_dim = self.hidden_dim // self.num_heads
        return tensor.view(batch, steps, nodes, self.num_heads, head_dim).permute(0, 1, 3, 2, 4)

    def forward(self, x, adj=None):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self._align(x)
        batch, steps, nodes, _ = x.shape
        query = self._split_heads(self.query(x), batch, steps, nodes)
        key = self._split_heads(self.key(x), batch, steps, nodes)
        value = self._split_heads(self.value(x), batch, steps, nodes)
        head_dim = self.hidden_dim // self.num_heads
        scores = query @ key.transpose(-1, -2) / math.sqrt(head_dim)
        if adj is not None:
            adjacency = torch.nan_to_num(adj.to(device=x.device, dtype=x.dtype), nan=0.0)
            allowed = (adjacency != 0) | torch.eye(nodes, device=x.device, dtype=torch.bool)
            scores = scores.masked_fill(~allowed, self.mask_value)
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        out = (weights @ value).permute(0, 1, 3, 2, 4).reshape(batch, steps, nodes, self.hidden_dim)
        return self.out_proj(out)
