"""
CCA-SSG: Canonical Correlation Analysis for Self-Supervised Learning on Graphs

This model adapts the original CCA-SSG implementation for the LibCity framework.
CCA-SSG learns node embeddings using self-supervised learning via graph augmentation
and Canonical Correlation Analysis (CCA) based loss.

Original Paper: "From Canonical Correlation Analysis to Self-supervised Graph Neural Networks"
Original Repository: https://github.com/hengruizhang98/CCA-SSG

Key Features:
- Two augmented graph views (edge dropping + feature dropping)
- CCA-based loss: maximizes cross-view agreement while decorrelating features
- GCN backbone for graph convolution
- Optional MLP backbone for non-graph settings

Key Adaptations for LibCity:
- Inherits from AbstractTrafficStateModel
- Uses LibCity's data_feature and config patterns
- Implements predict() and calculate_loss() methods
- Adapts DGL operations to work with LibCity's adjacency matrix format
- Handles graph augmentation within the model

Configuration Parameters:
- hid_dim: Hidden dimension (default: 512)
- out_dim: Output embedding dimension (default: 512)
- n_layers: Number of GNN layers (default: 2)
- lambd: Decorrelation loss weight (default: 1e-3)
- dfr: Drop feature ratio for augmentation (default: 0.2)
- der: Drop edge ratio for augmentation (default: 0.2)
- use_mlp: Use MLP backbone instead of GCN (default: False)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from logging import getLogger

try:
    import dgl
    from dgl.nn import GraphConv
    HAS_DGL = True
except ImportError:
    HAS_DGL = False

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


# =====================================================
# Graph Augmentation Utilities
# =====================================================
def random_aug(graph, x, feat_drop_rate, edge_mask_rate, device):
    """
    Random augmentation for graph and features.

    Args:
        graph: DGL graph
        x: Node features tensor
        feat_drop_rate: Probability of dropping features
        edge_mask_rate: Probability of masking edges
        device: Target device

    Returns:
        ng: Augmented graph
        feat: Augmented features
    """
    n_node = graph.number_of_nodes()

    # Edge masking
    edge_mask = mask_edge(graph, edge_mask_rate)

    # Feature dropping
    feat = drop_feature(x, feat_drop_rate)

    # Create new graph with masked edges
    ng = dgl.graph([])
    ng = ng.to(device)
    ng.add_nodes(n_node)

    src = graph.edges()[0]
    dst = graph.edges()[1]

    if len(edge_mask) > 0:
        nsrc = src[edge_mask]
        ndst = dst[edge_mask]
        ng.add_edges(nsrc.to(device), ndst.to(device))

    return ng, feat


def drop_feature(x, drop_prob):
    """
    Drop features randomly by zeroing out columns.

    Args:
        x: Feature tensor of shape (N, F)
        drop_prob: Probability of dropping each feature

    Returns:
        Augmented feature tensor
    """
    if drop_prob == 0:
        return x

    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device
    ).uniform_(0, 1) < drop_prob

    x = x.clone()
    x[:, drop_mask] = 0

    return x


def mask_edge(graph, mask_prob):
    """
    Create edge mask for graph augmentation.

    Args:
        graph: DGL graph
        mask_prob: Probability of masking each edge

    Returns:
        Tensor of indices for edges to keep
    """
    E = graph.number_of_edges()

    if E == 0 or mask_prob == 0:
        return torch.arange(E)

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)

    return mask_idx


# =====================================================
# MLP Backbone
# =====================================================
class MLP(nn.Module):
    """
    Simple MLP backbone as alternative to GCN.
    Used when graph structure is not available or not needed.
    """
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, graph, x):
        """Forward pass ignoring graph structure."""
        x = self.layer1(x)
        if self.use_bn and x.size(0) > 1:
            x = self.bn(x)
        x = self.act_fn(x)
        x = self.layer2(x)
        return x


# =====================================================
# GCN Backbone (using DGL)
# =====================================================
class GCN(nn.Module):
    """
    Graph Convolutional Network backbone using DGL's GraphConv.
    """
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_dim, hid_dim, norm='both'))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim, norm='both'))
            self.convs.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x):
        """
        Forward pass through GCN layers.

        Args:
            graph: DGL graph
            x: Node features (N, in_dim)

        Returns:
            Node embeddings (N, out_dim)
        """
        for i in range(self.n_layers - 1):
            x = F.relu(self.convs[i](graph, x))
        x = self.convs[-1](graph, x)
        return x


# =====================================================
# Fallback GCN using native PyTorch (when DGL not available)
# =====================================================
class GCNFallback(nn.Module):
    """
    Fallback GCN implementation using native PyTorch.
    Uses sparse matrix multiplication for graph convolution.
    """
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, num_nodes=None, adj_mx=None, device=None):
        super().__init__()

        self.n_layers = n_layers
        self.device = device
        self.layers = nn.ModuleList()

        # Build layers
        self.layers.append(nn.Linear(in_dim, hid_dim))
        if n_layers > 1:
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hid_dim, hid_dim))
            self.layers.append(nn.Linear(hid_dim, out_dim))

        # Store adjacency matrix for convolution
        self.adj_mx = adj_mx
        self._logger = getLogger()

    def forward(self, graph_or_adj, x):
        """
        Forward pass through GCN layers.

        Args:
            graph_or_adj: Adjacency matrix (dense or sparse) - ignored if self.adj_mx is set
            x: Node features (N, in_dim)

        Returns:
            Node embeddings (N, out_dim)
        """
        # Use stored adjacency or create identity
        if self.adj_mx is not None:
            adj = self.adj_mx
        elif isinstance(graph_or_adj, torch.Tensor):
            adj = graph_or_adj
        else:
            # No graph structure available, use identity
            adj = torch.eye(x.size(0), device=x.device)

        # Normalize adjacency (D^-0.5 * A * D^-0.5)
        if adj.is_sparse:
            adj = adj.to_dense()
        adj = adj + torch.eye(adj.size(0), device=adj.device)  # Add self-loops
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        adj_norm = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)

        for i in range(self.n_layers - 1):
            x = torch.mm(adj_norm, x)
            x = self.layers[i](x)
            x = F.relu(x)

        x = torch.mm(adj_norm, x)
        x = self.layers[-1](x)

        return x


# =====================================================
# CCA-SSG Core Model
# =====================================================
class CCA_SSG_Core(nn.Module):
    """
    Core CCA-SSG model that performs self-supervised learning.

    Args:
        in_dim: Input feature dimension
        hid_dim: Hidden dimension
        out_dim: Output embedding dimension
        n_layers: Number of GNN layers
        use_mlp: Use MLP backbone instead of GCN
        use_dgl: Use DGL-based GCN (if available)
        num_nodes: Number of nodes (for fallback GCN)
        adj_mx: Adjacency matrix (for fallback GCN)
        device: Target device
    """
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, use_mlp=False,
                 use_dgl=True, num_nodes=None, adj_mx=None, device=None):
        super().__init__()

        self.use_dgl = use_dgl and HAS_DGL

        if use_mlp:
            self.backbone = MLP(in_dim, hid_dim, out_dim)
        elif self.use_dgl:
            self.backbone = GCN(in_dim, hid_dim, out_dim, n_layers)
        else:
            self.backbone = GCNFallback(in_dim, hid_dim, out_dim, n_layers,
                                        num_nodes=num_nodes, adj_mx=adj_mx, device=device)

    def get_embedding(self, graph, feat):
        """
        Get node embeddings without gradient.

        Args:
            graph: DGL graph or adjacency matrix
            feat: Node features

        Returns:
            Detached node embeddings
        """
        out = self.backbone(graph, feat)
        return out.detach()

    def forward(self, graph1, feat1, graph2, feat2):
        """
        Forward pass for contrastive learning.

        Args:
            graph1: First augmented graph
            feat1: First augmented features
            graph2: Second augmented graph
            feat2: Second augmented features

        Returns:
            z1, z2: Standardized embeddings from both views
        """
        h1 = self.backbone(graph1, feat1)
        h2 = self.backbone(graph2, feat2)

        # Standardize embeddings (zero mean, unit variance)
        z1 = (h1 - h1.mean(0)) / (h1.std(0) + 1e-8)
        z2 = (h2 - h2.mean(0)) / (h2.std(0) + 1e-8)

        return z1, z2


# =====================================================
# Linear Evaluation Module
# =====================================================
class LogReg(nn.Module):
    """
    Logistic regression for linear evaluation of embeddings.
    """
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


# =====================================================
# Main CCASSG Model for LibCity
# =====================================================
class CCASSG(AbstractTrafficStateModel):
    """
    CCA-SSG: Canonical Correlation Analysis for Self-Supervised Learning on Graphs

    This model learns node embeddings using self-supervised learning with:
    - Random graph augmentation (edge dropping + feature dropping)
    - CCA-based loss for cross-view agreement and feature decorrelation
    - GCN or MLP backbone

    Config Parameters:
        - hid_dim: Hidden dimension (default: 512)
        - out_dim: Output embedding dimension (default: 512)
        - n_layers: Number of GNN layers (default: 2)
        - lambd: Decorrelation loss weight (default: 1e-3)
        - dfr: Drop feature ratio for augmentation (default: 0.2)
        - der: Drop edge ratio for augmentation (default: 0.2)
        - use_mlp: Use MLP backbone instead of GCN (default: False)

    Data Feature Requirements:
        - adj_mx: Adjacency matrix
        - num_nodes: Number of nodes in the graph
        - feature_dim: Input feature dimension
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._logger = getLogger()

        # Data features
        self.adj_mx = data_feature.get('adj_mx')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self._scaler = data_feature.get('scaler')

        # Device configuration
        self.device = config.get('device', torch.device('cpu'))
        self.model_name = config.get('model', 'CCASSG')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)

        # Model hyperparameters
        self.hid_dim = config.get('hid_dim', 512)
        self.out_dim = config.get('out_dim', 512)
        self.n_layers = config.get('n_layers', 2)
        self.use_mlp = config.get('use_mlp', False)

        # CCA loss hyperparameters
        self.lambd = config.get('lambd', 1e-3)

        # Augmentation hyperparameters
        self.dfr = config.get('dfr', 0.2)  # Drop feature ratio
        self.der = config.get('der', 0.2)  # Drop edge ratio

        # Build DGL graph from adjacency matrix
        self.dgl_graph = None
        self.adj_tensor = None
        if HAS_DGL:
            self._build_dgl_graph()
        else:
            self._build_adj_tensor()
            self._logger.warning("DGL not available. Using fallback GCN implementation.")

        # Initialize the core CCA-SSG model
        self.cca_ssg = CCA_SSG_Core(
            in_dim=self.feature_dim,
            hid_dim=self.hid_dim,
            out_dim=self.out_dim,
            n_layers=self.n_layers,
            use_mlp=self.use_mlp,
            use_dgl=HAS_DGL,
            num_nodes=self.num_nodes,
            adj_mx=self.adj_tensor,
            device=self.device
        )

        self._logger.info(
            f"CCASSG initialized with hid_dim={self.hid_dim}, out_dim={self.out_dim}, "
            f"n_layers={self.n_layers}, lambd={self.lambd}, dfr={self.dfr}, der={self.der}, "
            f"use_mlp={self.use_mlp}, use_dgl={HAS_DGL}, num_nodes={self.num_nodes}"
        )

    def _build_dgl_graph(self):
        """Build DGL graph from adjacency matrix."""
        if self.adj_mx is None:
            # Create empty graph
            self.dgl_graph = dgl.graph([])
            self.dgl_graph.add_nodes(self.num_nodes)
            self.dgl_graph = self.dgl_graph.to(self.device)
            return

        # Handle different adjacency matrix formats
        if hasattr(self.adj_mx, 'tocoo'):
            # Sparse scipy matrix
            coo = self.adj_mx.tocoo()
            src = torch.tensor(coo.row, dtype=torch.long)
            dst = torch.tensor(coo.col, dtype=torch.long)
        elif hasattr(self.adj_mx, 'row') and hasattr(self.adj_mx, 'col'):
            # COO format
            src = torch.tensor(self.adj_mx.row, dtype=torch.long)
            dst = torch.tensor(self.adj_mx.col, dtype=torch.long)
        else:
            # Dense numpy array or tensor
            adj = np.array(self.adj_mx) if not isinstance(self.adj_mx, np.ndarray) else self.adj_mx
            rows, cols = np.where(adj > 0)
            src = torch.tensor(rows, dtype=torch.long)
            dst = torch.tensor(cols, dtype=torch.long)

        # Create DGL graph
        self.dgl_graph = dgl.graph((src, dst), num_nodes=self.num_nodes)
        self.dgl_graph = self.dgl_graph.to(self.device)

        # Add self-loops
        self.dgl_graph = dgl.remove_self_loop(self.dgl_graph)
        self.dgl_graph = dgl.add_self_loop(self.dgl_graph)

    def _build_adj_tensor(self):
        """Build adjacency tensor for fallback GCN."""
        if self.adj_mx is None:
            self.adj_tensor = torch.eye(self.num_nodes, device=self.device)
            return

        # Convert to dense tensor
        if hasattr(self.adj_mx, 'toarray'):
            adj = self.adj_mx.toarray()
        elif hasattr(self.adj_mx, 'tocoo'):
            adj = self.adj_mx.tocoo().toarray()
        else:
            adj = np.array(self.adj_mx)

        self.adj_tensor = torch.tensor(adj, dtype=torch.float32, device=self.device)

    def forward(self, batch):
        """
        Forward pass for the CCASSG model.

        For road representation, this returns node embeddings.

        Args:
            batch: dict containing:
                - 'node_features': [N, feature_dim] tensor of node features
                - Or 'X': traffic state tensor that will be reshaped

        Returns:
            [N, out_dim] node embeddings
        """
        # Get node features from batch
        if 'node_features' in batch:
            node_features = batch['node_features']
        elif 'X' in batch:
            x = batch['X']
            if x.dim() == 4:
                # Traffic state format: (batch, time, nodes, features)
                node_features = x[:, -1, :, :].reshape(-1, x.shape[-1])
            elif x.dim() == 3:
                # (time, nodes, features)
                node_features = x[-1, :, :]
            else:
                node_features = x
        else:
            raise ValueError("batch must contain 'node_features' or 'X'")

        # Move to device
        node_features = node_features.to(self.device).float()

        # Get embeddings
        if HAS_DGL and self.dgl_graph is not None:
            graph = self.dgl_graph.to(self.device)
            embeddings = self.cca_ssg.get_embedding(graph, node_features)
        else:
            embeddings = self.cca_ssg.get_embedding(self.adj_tensor, node_features)

        return embeddings

    def predict(self, batch):
        """
        Get embeddings for prediction/inference and save to disk.

        Args:
            batch: dict containing node features

        Returns:
            [N, out_dim] node embeddings
        """
        embeddings = self.forward(batch)

        # Save embeddings to disk (following ChebConv/GeomGCN pattern)
        try:
            save_path = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.npy'.format(
                self.exp_id, self.model_name, self.dataset, self.out_dim)
            np.save(save_path, embeddings.detach().cpu().numpy())
            self._logger.info('Saved embeddings to {}'.format(save_path))
        except Exception as e:
            self._logger.warning(f'Failed to save embeddings: {e}')

        return embeddings

    def calculate_loss(self, batch):
        """
        Calculate CCA-based self-supervised loss.

        The CCA-SSG loss consists of:
        1. Invariance loss: Maximizes agreement between two augmented views
        2. Decorrelation loss: Ensures feature dimensions are uncorrelated

        Args:
            batch: dict containing:
                - 'node_features': [N, feature_dim] or
                - 'X': traffic state tensor

        Returns:
            loss tensor
        """
        # Get node features from batch
        if 'node_features' in batch:
            node_features = batch['node_features']
        elif 'X' in batch:
            x = batch['X']
            if x.dim() == 4:
                node_features = x[:, -1, :, :].reshape(-1, x.shape[-1])
            elif x.dim() == 3:
                node_features = x[-1, :, :]
            else:
                node_features = x
        else:
            raise ValueError("batch must contain 'node_features' or 'X'")

        # Move to device
        node_features = node_features.to(self.device).float()
        N = node_features.size(0)

        if HAS_DGL and self.dgl_graph is not None:
            # Create two augmented views using DGL
            graph = self.dgl_graph.to(self.device)

            graph1, feat1 = random_aug(graph, node_features, self.dfr, self.der, self.device)
            graph2, feat2 = random_aug(graph, node_features, self.dfr, self.der, self.device)

            # Add self-loops to augmented graphs
            graph1 = dgl.add_self_loop(graph1)
            graph2 = dgl.add_self_loop(graph2)

            # Forward pass through model
            z1, z2 = self.cca_ssg(graph1, feat1, graph2, feat2)
        else:
            # Fallback: use feature dropping only (no edge augmentation)
            feat1 = drop_feature(node_features, self.dfr)
            feat2 = drop_feature(node_features, self.dfr)

            # Forward pass using adjacency tensor
            z1, z2 = self.cca_ssg(self.adj_tensor, feat1, self.adj_tensor, feat2)

        # Compute CCA loss
        # Cross-correlation matrix between views
        c = torch.mm(z1.T, z2) / N
        c1 = torch.mm(z1.T, z1) / N
        c2 = torch.mm(z2.T, z2) / N

        # Invariance loss: maximize diagonal of cross-correlation
        # (maximize agreement between views on same dimensions)
        loss_inv = -torch.diagonal(c).sum()

        # Decorrelation loss: minimize off-diagonal elements
        # (reduce redundancy by decorrelating feature dimensions)
        iden = torch.eye(c.shape[0], device=self.device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()

        # Total loss
        loss = loss_inv + self.lambd * (loss_dec1 + loss_dec2)

        return loss

    @torch.no_grad()
    def get_embeddings(self, batch):
        """
        Get all node embeddings (for saving/analysis).

        Args:
            batch: dict containing node features

        Returns:
            [N, out_dim] CPU tensor of embeddings
        """
        self.eval()
        embeddings = self.forward(batch)
        return embeddings.cpu()

    def save_embeddings(self, batch, save_path=None):
        """
        Save embeddings to file.

        Args:
            batch: dict containing node features
            save_path: Optional path to save embeddings
        """
        embeddings = self.get_embeddings(batch)

        if save_path is None:
            save_path = f'./libcity/cache/{self.exp_id}/evaluate_cache/embedding_{self.model_name}_{self.dataset}_{self.out_dim}.npy'

        np.save(save_path, embeddings.numpy())
        self._logger.info(f"Embeddings saved to {save_path}")
