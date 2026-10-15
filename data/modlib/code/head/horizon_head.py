"""Direct multi-output head with one linear layer per horizon step.

Category: head
Interface: ``forward(x: (B, T', N, d)) -> (B, H, N, output_dim)``.

Principle: instead of sharing one output matrix across the horizon, every step
``h`` owns its own ``Linear(T' * d, output_dim)``. The predictions are stacked
along a new horizon axis, which implements a direct (non-autoregressive)
multi-output strategy where the parameters of different steps cannot interfere.

Applicable when the pattern of each forecast step differs noticeably (peaks at
some steps, plateaus at others); parameters ``H * T' * d * output_dim``, i.e.
the same count as a single wide linear layer but organised per step. ``loss``
implements masked MAE / MSE / MAPE and the class default is ``masked_mape``.

Reference: ``LibCity traffic_speed_prediction/STFGNN.py``
and ``traffic_flow_prediction/STSGCN.py``, where ``self.outputs`` is a
``nn.ModuleList`` holding one ``OutputLayer`` per prediction step and the
results are concatenated over the horizon; the masked losses follow
``LibCity model/loss.py`` (``masked_mape_torch``).
"""

import torch
import torch.nn as nn

__all__ = ["HorizonHead"]


class HorizonHead(nn.Module):
    """Per-horizon-step linear head producing ``(B, H, N, output_dim)``."""

    DEFAULT_LOSS = "masked_mape"

    def __init__(self, hidden_dim=64, input_len=12, horizon=12, output_dim=1, dropout=0.0, bias=True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input_len = int(input_len)
        self.horizon = int(horizon)
        self.output_dim = int(output_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.heads = nn.ModuleList(
            nn.Linear(self.input_len * self.hidden_dim, self.output_dim, bias=bias) for _ in range(self.horizon)
        )

    def _ensure_heads(self, in_features, device, dtype):
        if self.heads[0].in_features != in_features:
            self.heads = nn.ModuleList(
                nn.Linear(in_features, self.output_dim).to(device=device, dtype=dtype)
                for _ in range(self.horizon)
            )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        batch, steps, nodes, channels = x.shape
        self._ensure_heads(steps * channels, x.device, x.dtype)
        flat = x.permute(0, 2, 1, 3).reshape(batch, nodes, steps * channels)
        flat = self.dropout(flat)
        stacked = torch.stack([head(flat) for head in self.heads], dim=1)
        return stacked.permute(0, 1, 2, 3)

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
