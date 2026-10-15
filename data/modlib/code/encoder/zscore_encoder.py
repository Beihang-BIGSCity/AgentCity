"""Per-node z-score normalisation with running statistics.

Category: encoder
Interface: ``forward(x: (B, T, N, C)) -> (B, T, N, C)`` -- the channel count is
preserved (``d == C``), the sequence and node axes are never resized.

Principle: keep a per-node / per-channel mean and variance as running buffers
(exponential moving average) and standardise the input with them. During
training the statistics of the current batch are used and the buffers are
updated; in eval mode the running buffers are reused, which is what makes the
transform safe to apply at inference time.

Applicable when raw traffic features have very different scales per sensor;
cost is ``O(B*T*N*C)`` time and ``O(N*C)`` memory. The module holds no trainable
parameters, only two buffers. Non-finite entries are ignored by the statistics
and written back as ``0`` so the encoder never emits ``NaN``; filling missing
observations with a real value is left to ``MissingFillEncoder``.

Reference: ``LibCity traffic_flow_prediction/STNorm.py``
(``STNorm`` keeps in-model running mean/std statistics for z-score
normalisation; here the same idea is exposed as a standalone encoder).
"""

import torch
import torch.nn as nn

__all__ = ["ZScoreEncoder"]


class ZScoreEncoder(nn.Module):
    """Standardise ``(B, T, N, C)`` traffic features per node and channel."""

    def __init__(self, num_nodes=207, input_dim=3, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.register_buffer("running_mean", torch.zeros(self.num_nodes, self.input_dim))
        self.register_buffer("running_var", torch.ones(self.num_nodes, self.input_dim))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        valid = ~torch.isnan(x)
        count = valid.to(x.dtype).sum(dim=(0, 1)).clamp_min(1.0)
        filled = torch.where(valid, x, torch.zeros_like(x))
        batch_mean = filled.sum(dim=(0, 1)) / count
        centered = torch.where(valid, x - batch_mean, torch.zeros_like(x))
        batch_var = centered.square().sum(dim=(0, 1)) / count
        aligned = x.shape[2] == self.num_nodes and x.shape[3] == self.input_dim
        if self.training:
            if aligned:
                with torch.no_grad():
                    self.running_mean.mul_(1.0 - self.momentum).add_(self.momentum * batch_mean.detach())
                    self.running_var.mul_(1.0 - self.momentum).add_(self.momentum * batch_var.detach())
            mean, var = batch_mean, batch_var
        elif aligned:
            mean, var = self.running_mean, self.running_var
        else:
            mean, var = batch_mean, batch_var
        normalised = (x - mean) / torch.sqrt(var + self.eps)
        return torch.nan_to_num(normalised, nan=0.0)
