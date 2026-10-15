"""Bidirectional random-walk diffusion convolution (DCRNN style).

Category: spatial
Interface: ``forward(x: (B, T, N, d), adj: (N, N) | None) -> (B, T, N, d)``.

Principle: build the forward and backward transition matrices
``P_f = D^-1 A`` and ``P_b = D^-1 A^T``, diffuse the node signal ``K`` hops along
both (``x, P x, P^2 x, ...``), concatenate the ``1 + 2K`` blocks and fuse them
with a single linear projection. Because the two directions are modelled
separately, the layer captures upstream/downstream asymmetry, which is what
traffic propagation looks like on a directed road graph.

Applicable to directed networks with an explicit adjacency. Parameters
``(1 + 2K) * d * d``; diffusion itself costs ``O(K * N^2 * d)`` per time step.
When ``adj`` is ``None`` the two supports become the identity matrix, so the
layer degrades to a learned linear map plus a two-hop self diffusion instead of
raising an error.

Reference: ``LibCity traffic_speed_prediction/DCRNN.py``
(``GCONV`` with ``filter_type='dual_random_walk'``); the random-walk helpers
live in ``LibCity model/utils.py``
(``calculate_random_walk_matrix``).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DiffusionGCN"]


class DiffusionGCN(nn.Module):
    """Random-walk diffusion convolution with two propagation directions."""

    def __init__(self, hidden_dim=64, num_nodes=207, num_diffusion_steps=2, dropout=0.0, bias=True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_nodes = int(num_nodes)
        self.num_diffusion_steps = max(1, int(num_diffusion_steps))
        self.num_directions = 2
        self.dropout = float(dropout)
        self.in_proj = None
        self.blocks = 1 + self.num_directions * self.num_diffusion_steps
        self.proj = nn.Linear(self.blocks * self.hidden_dim, self.hidden_dim, bias=bias)

    def _align(self, x):
        if x.shape[-1] == self.hidden_dim:
            return x
        if self.in_proj is None or self.in_proj.in_features != x.shape[-1]:
            self.in_proj = nn.Linear(x.shape[-1], self.hidden_dim).to(device=x.device, dtype=x.dtype)
        return self.in_proj(x)

    def _supports(self, adj, num_nodes, device, dtype):
        eye = torch.eye(num_nodes, device=device, dtype=dtype)
        if adj is None:
            return [eye, eye]
        adj = torch.nan_to_num(adj.to(device=device, dtype=dtype), nan=0.0).clamp(min=0.0)

        def random_walk(matrix):
            degree = matrix.sum(dim=-1, keepdim=True)
            inverse = torch.where(degree > 0, 1.0 / degree.clamp(min=1e-6), torch.zeros_like(degree))
            return inverse * matrix

        return [random_walk(adj), random_walk(adj.transpose(0, 1))]

    def forward(self, x, adj=None):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self._align(x)
        supports = self._supports(adj, x.shape[2], x.device, x.dtype)
        blocks = [x]
        for support in supports:
            propagated = x
            for _ in range(self.num_diffusion_steps):
                propagated = torch.einsum("nm,btmc->btnc", support, propagated)
                blocks.append(propagated)
        out = self.proj(torch.cat(blocks, dim=-1))
        return F.dropout(out, p=self.dropout, training=self.training)
