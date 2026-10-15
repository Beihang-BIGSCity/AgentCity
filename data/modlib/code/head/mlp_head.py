"""Two-layer MLP multi-step prediction head.

Category: head
Interface: ``forward(x: (B, T', N, d)) -> (B, H, N, output_dim)``.

Principle: the flattened ``(T', d)`` history first goes through a hidden
``Linear + ReLU`` (default width ``2 * hidden_dim``) and then through the output
layer that emits all horizon steps. The extra non-linearity lets the head model
interactions that a single linear map cannot express.

Applicable when the mapping from history to horizon is clearly non-linear;
parameters ``T' * d * mlp_hidden + mlp_hidden * H * output_dim``. ``loss``
implements masked MAE / MSE / MAPE and the class default is ``masked_mse``.

Reference: ``LibCity traffic_speed_prediction/GMAN.py``
and ``traffic_flow_prediction/AGCRN.py`` both terminate their stacks with a
multi-layer ``Linear`` output block (here exposed as a standalone head); the
masked losses follow ``LibCity model/loss.py``.
"""

import torch
import torch.nn as nn

__all__ = ["MLPHead"]


class MLPHead(nn.Module):
    """Hidden-layer MLP head producing ``(B, H, N, output_dim)``."""

    DEFAULT_LOSS = "masked_mse"

    def __init__(
        self,
        hidden_dim=64,
        input_len=12,
        horizon=12,
        output_dim=1,
        mlp_hidden=None,
        dropout=0.0,
        bias=True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input_len = int(input_len)
        self.horizon = int(horizon)
        self.output_dim = int(output_dim)
        self.mlp_hidden = int(mlp_hidden) if mlp_hidden else 2 * self.hidden_dim
        self.dropout = nn.Dropout(float(dropout))
        self.fc1 = nn.Linear(self.input_len * self.hidden_dim, self.mlp_hidden, bias=bias)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(self.mlp_hidden, self.horizon * self.output_dim, bias=bias)

    def _ensure_layers(self, in_features, device, dtype):
        if self.fc1.in_features != in_features:
            self.fc1 = nn.Linear(in_features, self.mlp_hidden).to(device=device, dtype=dtype)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        batch, steps, nodes, channels = x.shape
        self._ensure_layers(steps * channels, x.device, x.dtype)
        flat = x.permute(0, 2, 1, 3).reshape(batch, nodes, steps * channels)
        hidden = self.dropout(self.act(self.fc1(flat)))
        out = self.fc2(hidden)
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
