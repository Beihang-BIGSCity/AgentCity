"""Single linear multi-step prediction head (STFGNN output head).

Category: head
Interface: ``forward(x: (B, T', N, d)) -> (B, H, N, output_dim)``.

Principle: the ``(T', d)`` history of every node is flattened and mapped by one
``Linear(T' * d, H * output_dim)`` straight to all horizon steps at once, then
reshaped to ``(B, H, N, output_dim)``. Direct multi-step regression, no
autoregression.

Applicable when the horizon is fixed and the simplest possible head is wanted;
parameters ``T' * d * H * output_dim``. ``loss`` implements masked MAE / MSE /
MAPE and the class default is ``masked_mae``.

Reference: ``LibCity traffic_speed_prediction/STFGNN.py``
(``OutputLayer``: ``nn.Linear(num_of_filters, predict_length * output_dim)`` after
flattening ``(B, N, T, C)``); the masked loss functions mirror
``LibCity model/loss.py`` (``masked_mae_torch`` and friends).
"""

import torch
import torch.nn as nn

__all__ = ["LinearHead"]


class LinearHead(nn.Module):
    """Flatten-then-linear head producing ``(B, H, N, output_dim)``."""

    DEFAULT_LOSS = "masked_mae"

    def __init__(self, hidden_dim=64, input_len=12, horizon=12, output_dim=1, dropout=0.0, bias=True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input_len = int(input_len)
        self.horizon = int(horizon)
        self.output_dim = int(output_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.fc = nn.Linear(self.input_len * self.hidden_dim, self.horizon * self.output_dim, bias=bias)

    def _ensure_layer(self, in_features, device, dtype):
        if self.fc.in_features != in_features:
            self.fc = nn.Linear(in_features, self.horizon * self.output_dim).to(device=device, dtype=dtype)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        batch, steps, nodes, channels = x.shape
        self._ensure_layer(steps * channels, x.device, x.dtype)
        flat = x.permute(0, 2, 1, 3).reshape(batch, nodes, steps * channels)
        out = self.fc(self.dropout(flat))
        return out.view(batch, nodes, self.horizon, self.output_dim).permute(0, 2, 1, 3)

    def loss(self, pred, target, null_val=0.0, loss_type=None):
        loss_type = (loss_type or self.DEFAULT_LOSS).lower()
        target = target.to(pred.dtype)
        mask = (target != null_val).to(pred.dtype)
        diff = pred - target
        if loss_type in ("masked_mae", "mae"):
            error = diff.abs()
        elif loss_type in ("masked_mse", "mse"):
            error = diff * diff
        elif loss_type in ("masked_mape", "mape"):
            error = diff.abs() / target.abs().clamp_min(1e-5)
        else:
            raise ValueError("unsupported loss_type: {}".format(loss_type))
        masked = error * mask
        return masked.sum() / mask.sum().clamp_min(1.0)
