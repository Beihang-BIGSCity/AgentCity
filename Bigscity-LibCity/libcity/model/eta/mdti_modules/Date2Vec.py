"""
Date2Vec: Temporal embedding module for encoding date/time information.

This module implements learnable temporal embeddings that encode
date and time information into fixed-dimensional vectors.

Original source: repos/MDTI/dataset/Model.py and repos/MDTI/dataset/Date2Vec.py
"""

import torch
from torch import nn


class Date2Vec(nn.Module):
    """
    Date2Vec: Learnable temporal embedding model.

    This model encodes date/time features (hour, minute, second, year, month, day)
    into a k-dimensional embedding using a combination of linear and periodic
    (sin/cos) transformations.

    Args:
        k: Dimension of the output embedding (default: 32)
        act: Activation function - 'sin' or 'cos' (default: 'sin')
    """

    def __init__(self, k=32, act="sin"):
        super(Date2Vec, self).__init__()

        if k % 2 == 0:
            k1 = k // 2
            k2 = k // 2
        else:
            k1 = k // 2
            k2 = k // 2 + 1

        self.fc1 = nn.Linear(6, k1)
        self.fc2 = nn.Linear(6, k2)
        self.d2 = nn.Dropout(0.3)

        if act == 'sin':
            self.activation = torch.sin
        else:
            self.activation = torch.cos

        self.fc3 = nn.Linear(k, k // 2)
        self.d3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(k // 2, 6)
        self.fc5 = nn.Linear(6, 6)

    def forward(self, x):
        """
        Forward pass for training/reconstruction.

        Args:
            x: Input tensor of shape (batch, 6) containing
               [hour, minute, second, year, month, day]

        Returns:
            Reconstructed temporal features of shape (batch, 6)
        """
        out1 = self.fc1(x)
        out2 = self.d2(self.activation(self.fc2(x)))
        out = torch.cat([out1, out2], 1)
        out = self.d3(self.fc3(out))
        out = self.fc4(out)
        out = self.fc5(out)
        return out

    def encode(self, x):
        """
        Encode temporal features to embedding.

        Args:
            x: Input tensor of shape (batch, 6) containing
               [hour, minute, second, year, month, day]

        Returns:
            Temporal embedding of shape (batch, k)
        """
        out1 = self.fc1(x)
        out2 = self.activation(self.fc2(x))
        out = torch.cat([out1, out2], 1)
        return out


class Date2VecConvert:
    """
    Wrapper class for Date2Vec model for inference.

    This class loads a pre-trained Date2Vec model and provides
    a simple interface for encoding temporal features.

    Args:
        dim: Dimension of the temporal embedding
        model_path: Path to the pre-trained model weights
    """

    def __init__(self, dim, model_path=None):
        self.model = Date2Vec(k=dim)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = self.model.eval()

    def __call__(self, x):
        """
        Encode temporal features.

        Args:
            x: Input tensor of shape (6,) or (batch, 6)

        Returns:
            Temporal embedding
        """
        with torch.no_grad():
            if x.dim() == 1:
                return self.model.encode(x.unsqueeze(0)).squeeze(0)
            return self.model.encode(x)
