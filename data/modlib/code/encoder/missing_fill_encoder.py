"""Missing-value imputation by forward fill along the time axis.

Category: encoder
Interface: ``forward(x: (B, T, N, C)) -> (B, T, N, C)`` -- the channel count is
preserved (``d == C``).

Principle: both ``0.0`` and ``NaN`` are treated as missing. For every
``(batch, node, channel)`` series the last valid observation is carried
forward; when no valid observation has been seen yet the value falls back to
``0``. This is the cheap, parameter-free imputation used before feeding
partially failed sensors into a spatio-temporal backbone.

Cost: one ``O(B*T*N*C)`` scan over time, no trainable parameters.

Reference: ``LibCity data/dataset/dataset_subclass/dstagnn_dataset.py``
(LibCity's convention of normalising missing values with
``nan_to_num(..., nan=0.0)``; the library ships no dedicated imputation
module, so this encoder makes the preprocessing step explicit).
"""

import torch
import torch.nn as nn

__all__ = ["MissingFillEncoder"]


class MissingFillEncoder(nn.Module):
    """Forward-fill ``0.0`` / ``NaN`` entries along the time axis."""

    def __init__(self, input_dim=3, missing_value=0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.missing_value = float(missing_value)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        valid = (~torch.isnan(x)) & (x != self.missing_value)
        value = torch.nan_to_num(x, nan=0.0)
        last = torch.zeros_like(value[:, 0])
        seen = torch.zeros_like(valid[:, 0])
        steps = []
        for step in range(value.shape[1]):
            current_valid = valid[:, step]
            last = torch.where(current_valid, value[:, step], last)
            seen = seen | current_valid
            steps.append(torch.where(seen, last, torch.zeros_like(last)))
        return torch.stack(steps, dim=1)
