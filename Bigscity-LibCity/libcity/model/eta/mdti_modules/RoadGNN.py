"""
RoadGNN: Graph Neural Network for Road Network Encoding.

This module implements a GAT-based encoder for road network features.

Original source: repos/MDTI/model/RoadGNN.py
"""

import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("Warning: torch_geometric not installed. RoadGNN requires torch_geometric.")


class RoadGNN(nn.Module):
    """
    Road Network GNN encoder using Graph Attention Networks (GAT).

    This module encodes road segment features by propagating information
    through the road network graph structure using GAT layers.

    Args:
        fea_size: Input feature size for each road segment
        g_dim_per_layer: List of hidden dimensions for each layer
        g_heads_per_layer: List of attention heads for each layer
        num_layers: Number of GAT layers
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, fea_size, g_dim_per_layer, g_heads_per_layer, num_layers, dropout=0.1):
        super().__init__()

        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("RoadGNN requires torch_geometric. "
                            "Install it with: pip install torch_geometric")

        self.linear = nn.Linear(fea_size, g_dim_per_layer[0])
        self.gat_net = nn.ModuleList([
            GATConv(in_channels=g_dim_per_layer[i],
                    out_channels=g_dim_per_layer[i] // g_heads_per_layer[i],
                    heads=g_heads_per_layer[i],
                    dropout=dropout
                    )
            for i in range(num_layers)
        ])

    def forward(self, x, edge_index):
        """
        Forward pass of Road GNN.

        Args:
            x: Node feature tensor of shape (num_nodes, fea_size)
            edge_index: Edge index tensor of shape (2, num_edges)

        Returns:
            Encoded node features of shape (num_nodes, g_dim_per_layer[-1])
        """
        x = self.linear(x)
        for layer in self.gat_net[:-1]:
            x = F.relu(x + layer(x, edge_index), inplace=True)
        x = self.gat_net[-1](x, edge_index)
        return x
