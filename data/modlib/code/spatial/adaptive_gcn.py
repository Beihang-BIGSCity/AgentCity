"""Adaptive graph convolution from learned node embeddings (AGCRN style).

Category: spatial
Interface: ``forward(x: (B, T, N, d), adj: (N, N) | None) -> (B, T, N, d)``.

Principle: each node owns a trainable embedding ``E``; the adjacency is derived
as ``S = softmax(ReLU(E E^T))`` so the graph is learned end-to-end instead of
being read from a file. The layer then runs a Neumann power series over the
adaptive graph (``I, S, S^2, ...`` up to ``ks`` terms) with one ``d x d``
weight per order.

Applicable when the adjacency matrix is missing, noisy or incomplete and the
model should discover latent structure. Parameters ``N * embed_dim + ks * d * d``;
the power series costs ``O(ks * N^2 * d)`` per time step. Precedence: an explicit
``adj`` argument wins, otherwise the learned embedding graph is used, otherwise
(no adjacency and a node count that does not match the embedding table) the
series degenerates to ``I`` so the module never crashes.

Reference: ``LibCity traffic_flow_prediction/AGCRN.py``
(``AVWGCN`` derives ``softmax(relu(E E^T))`` and applies a K-order polynomial;
here the polynomial is the plain Neumann power series and the per-node weight
pool is replaced by shared per-order weights so the layer works for any ``N``).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AdaptiveGCN"]


class AdaptiveGCN(nn.Module):
    """Graph convolution over a node-embedding-derived adaptive adjacency."""

    def __init__(self, hidden_dim=64, num_nodes=207, embed_dim=10, ks=2, dropout=0.0, bias=True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_nodes = int(num_nodes)
        self.embed_dim = int(embed_dim)
        self.ks = max(1, int(ks))
        self.dropout = float(dropout)
        self.in_proj = None
        self.node_embeddings = nn.Parameter(0.1 * torch.randn(self.num_nodes, self.embed_dim))
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
        if adj is not None:
            adj = torch.nan_to_num(adj.to(device=device, dtype=dtype), nan=0.0).clamp(min=0.0)
            degree = adj.sum(dim=-1, keepdim=True)
            base = torch.where(degree > 0, adj / degree.clamp(min=1e-6), torch.zeros_like(adj))
        elif num_nodes == self.num_nodes:
            similarity = self.node_embeddings @ self.node_embeddings.transpose(0, 1)
            base = F.softmax(F.relu(similarity), dim=1).to(device=device, dtype=dtype)
        else:
            zero = torch.zeros_like(eye)
            return [eye] + [zero for _ in range(self.ks - 1)]
        supports = [eye, base]
        for _ in range(2, self.ks):
            supports.append(base @ supports[-1])
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
