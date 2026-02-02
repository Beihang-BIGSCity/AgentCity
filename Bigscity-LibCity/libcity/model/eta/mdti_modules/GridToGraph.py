"""
GridToGraph: Converts grid images to graph structure for GAT processing.

This module implements the conversion of grid-based spatial representations
to graph structures with 8-connectivity for Graph Attention Network processing.

Original source: repos/MDTI/model/GridToGraph.py
"""

import torch
import torch.nn as nn

try:
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("Warning: torch_geometric not installed. GridToGraph requires torch_geometric.")


class GridToGraph(nn.Module):
    """
    Converts batch of grid images to PyTorch Geometric graph batches.

    Each pixel in the grid becomes a node, and edges are created based on
    8-connectivity (including self-loops).

    Args:
        in_channels: Number of input channels (e.g., 3 for RGB)
        grid_h: Height of the grid
        grid_w: Width of the grid
    """

    def __init__(self, in_channels: int, grid_h: int, grid_w: int):
        super().__init__()

        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("GridToGraph requires torch_geometric. "
                            "Install it with: pip install torch_geometric")

        self.in_channels = in_channels
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_nodes = grid_h * grid_w

    def forward(self, grid_image: torch.Tensor):
        """
        Convert batch of grid images to graph structure.

        Args:
            grid_image: Batch of grid images. Shape: (B, C, H, W)

        Returns:
            torch_geometric.data.Batch: Batch of graph data containing
                node features (x) and edge indices (edge_index).
        """
        B, C, H, W = grid_image.shape
        if H != self.grid_h or W != self.grid_w or C != self.in_channels:
            raise ValueError(f"Input grid_image shape {grid_image.shape} does not match "
                           f"initialized dimensions (B, {self.in_channels}, {self.grid_h}, {self.grid_w})")

        data_list = []
        for i in range(B):
            single_grid = grid_image[i]  # (C, H, W)

            # Node features: reshape (C, H, W) to (num_nodes, C)
            x = single_grid.permute(1, 2, 0).reshape(self.num_nodes, C)

            edge_indices = []
            for r in range(H):
                for c in range(W):
                    node_idx = r * W + c  # Calculate current pixel's node index
                    # Define 8-connectivity neighbors (including self-loop)
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W:  # Check if within grid
                                neighbor_idx = nr * W + nc
                                edge_indices.append((node_idx, neighbor_idx))

            # Edge index: convert to (2, num_edges) format
            edge_index = torch.tensor(edge_indices, dtype=torch.long, device=grid_image.device).t().contiguous()
            data_list.append(Data(x=x, edge_index=edge_index))

        # PyTorch Geometric's Batch object can handle multiple graphs
        return Batch.from_data_list(data_list)
