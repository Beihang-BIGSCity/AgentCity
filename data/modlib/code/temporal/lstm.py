"""LSTM encoder over the time axis.

Category: temporal
Interface: ``forward(x: (B, T, N, d)) -> (B, T', N, d)`` with ``T' == T``.

Principle: identical folding strategy to the GRU encoder, but with an LSTM cell
that keeps a separate cell state, so gate activations are less exposed to
vanishing gradients over longer windows. Input and hidden width are both
``hidden_dim`` (projected on the way in when needed).

Applicable when the temporal dependency spans many steps; parameters
``O(4 * num_layers * d^2)`` and sequential cost ``O(T)``.

Reference: ``LibCity traffic_speed_prediction/RNN.py``
(the ``rnn_type='LSTM'`` branch builds ``nn.LSTM`` over flattened
``num_nodes * feature_dim`` inputs). The brief pointed at a standalone
``traffic_speed_prediction/LSTM.py``, which does not exist in this checkout --
LibCity ships RNN/LSTM/GRU behind the single ``RNN.py`` wrapper.
"""

import torch
import torch.nn as nn

__all__ = ["LSTMEncoder"]


class LSTMEncoder(nn.Module):
    """Long short-term memory encoder with ``T' == T``."""

    def __init__(self, hidden_dim=64, num_layers=1, dropout=0.0, bidirectional=False, input_dim=None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = max(1, int(num_layers))
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1
        in_dim = self.hidden_dim if input_dim is None else int(input_dim)
        self.in_proj = nn.Identity() if in_dim == self.hidden_dim else nn.Linear(in_dim, self.hidden_dim)
        self.rnn = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        self.out_proj = (
            nn.Identity()
            if self.num_directions == 1
            else nn.Linear(self.num_directions * self.hidden_dim, self.hidden_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.in_proj(x)
        batch, steps, nodes, channels = x.shape
        sequence = x.permute(1, 0, 2, 3).reshape(steps, batch * nodes, channels)
        out, _ = self.rnn(sequence)
        out = out.reshape(steps, batch, nodes, out.shape[-1]).permute(1, 0, 2, 3)
        return self.out_proj(out)
