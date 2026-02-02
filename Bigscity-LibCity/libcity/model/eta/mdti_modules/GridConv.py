"""
GridConv: Convolutional Neural Network for Grid Image Processing.

This module implements a simple CNN for extracting features from grid images.

Original source: repos/MDTI/model/GridConv.py
"""

import torch.nn as nn


class GridConv(nn.Module):
    """
    Convolutional Neural Network for processing grid images.

    This module applies two convolutional layers with batch normalization
    and ReLU activation to extract spatial features from grid images.

    Args:
        in_channels: Number of input channels (e.g., 3 for RGB)
        out_channels: Number of output channels
        stride: Stride for convolution (default: 1)
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Forward pass of GridConv.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Feature tensor of shape (B, out_channels, H, W)
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out
