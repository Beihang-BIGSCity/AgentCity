"""Chebyshev polynomial graph convolution over a fixed adjacency matrix.

Category: spatial
Interface: ``forward(x: (B, T, N, d), adj: (N, N) | None) -> (B, T, N, d)``.

Principle: approximate spectral graph convolution with the K-term Chebyshev
expansion ``T_0 = I``, ``T_1 = L~``, ``T_k = 2L~ T_{k-1} - T_{k-2}``, where
``L~ = 2L / lambda_max - I`` is the scaled symmetric Laplacian. Every order gets
its own ``d x d`` weight, so the layer is a K-order polynomial filter of the
graph instead of a dense ``N x N`` operator.

Applicable when the road network topology is known and fixed (ASTGCN-style).
Parameters ``K * d * d``; the polynomial terms cost ``O(K * N^2 * d)`` per time
step. When ``adj`` is ``None`` the polynomial degenerates to ``T_0 = I`` (a
learned linear map on every node) so the module never crashes.

Reference: ``LibCity traffic_flow_prediction/ASTGCN.py``
(``ChebConvWithSAt``). Note: the reference path in the task brief
(``traffic_speed_prediction/ASTGCN.py``) does not exist in this checkout --
LibCity keeps ASTGCN under ``traffic_flow_prediction/``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ChebGCN"]


class ChebGCN(nn.Module):
    """K-order Chebyshev graph convolution."""

    def __init__(self, hidden_dim=64, num_nodes=207, ks=2, dropout=0.0, bias=True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_nodes = int(num_nodes)
        self.ks = max(1, int(ks))
        self.dropout = float(dropout)
        self.in_proj = None
        self.weight = nn.Parameter(torch.empty(self.ks, self.hidden_dim, self.hidden_dim))
        self.bias = nn.Parameter(torch.zeros(self.hidden_dim)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def _align(self, x):
        if x.shape[-1] == self.hidden_dim:
            return x
        if self.in_proj is None or self.in_proj.in_features != x.shape[-1]:
            self.in_proj = nn.Linear(x.shape[-1], self.hidden_dim).to(device=x.device, dtype=x.dtype)
        return self.in_proj(x)

    def _supports(self, adj, num_nodes, device, dtype):
        eye = torch.eye(num_nodes, device=device, dtype=dtype)
        if adj is None:
            zero = torch.zeros_like(eye)
            return [eye] + [zero for _ in range(self.ks - 1)]
        adj = torch.nan_to_num(adj.to(device=device, dtype=dtype), nan=0.0).clamp(min=0.0)
        adj = 0.5 * (adj + adj.transpose(0, 1)) + eye
        degree = adj.sum(dim=-1)
        scale = torch.where(degree > 0, degree.clamp(min=1e-6).rsqrt(), torch.zeros_like(degree))
        norm = scale.unsqueeze(-1) * adj * scale.unsqueeze(0)
        lap = eye - norm
        lambda_max = torch.clamp(torch.linalg.eigvalsh(lap.detach()).max(), min=1e-5)
        scaled = 2.0 * lap / lambda_max - eye
        supports = [eye, scaled]
        for _ in range(2, self.ks):
            supports.append(2.0 * scaled @ supports[-1] - supports[-2])
        return supports[: self.ks]

    def forward(self, x, adj=None):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self._align(x)
        supports = self._supports(adj, x.shape[2], x.device, x.dtype)
        terms = torch.stack([torch.einsum("nm,btmc->btnc", s, x) for s in supports], dim=0)
        out = torch.einsum("kbtnc,kco->btno", terms, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return F.dropout(out, p=self.dropout, training=self.training)
