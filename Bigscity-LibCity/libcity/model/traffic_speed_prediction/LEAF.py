"""
LEAF (Large Language Model Enhanced Adaptive Framework) Model for Traffic Prediction
Adapted from: https://github.com/xxx/LEAF

Original files:
- repos/LEAF/model/basic_model.py - Base class with embeddings
- repos/LEAF/model/graph.py - GraphBranch (ST-GCN based)
- repos/LEAF/model/hypergraph.py - HypergraphBranch

Key changes from original:
1. Removed global `args` dependency, replaced with config/data_feature
2. Implemented missing utility functions: norm_adj(), get_predefined_adjs()
3. Integrated GraphBranch as primary predictor (simplified without LLM selection)
4. Adapted to LibCity's batch dictionary format (X, y keys)
5. Added proper device handling via config
6. Implemented required LibCity methods: forward, predict, calculate_loss

Required config parameters:
- hidden_dim: Hidden dimension for embeddings (default: 64)
- stgcn_num_layers: Number of ST-GCN layers in GraphBranch (default: 7)
- dropout: Dropout rate (default: 0.0)
- use_hypergraph: Whether to use HypergraphBranch (default: False)
- hgnn_num_backbone_layers: Number of backbone layers for hypergraph (default: 6)
- hgnn_num_head_layers: Number of head layers for hypergraph (default: 1)
- hgnn_num_hyper_edge: Number of hyperedges (default: 32)
"""

import math
import numpy as np
from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def norm_adj(adj):
    """
    Normalize adjacency matrix using symmetric normalization.
    D^(-1/2) * A * D^(-1/2)

    Args:
        adj: Adjacency matrix tensor of shape (..., N, N)

    Returns:
        Normalized adjacency matrix
    """
    # Compute degree matrix
    deg = adj.sum(dim=-1, keepdim=True)  # (..., N, 1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    deg_inv_sqrt[torch.isnan(deg_inv_sqrt)] = 0.0

    # D^(-1/2) * A * D^(-1/2)
    normalized = deg_inv_sqrt * adj * deg_inv_sqrt.transpose(-1, -2)
    return normalized


def generate_predefined_adjs(adj_mx, num_nodes):
    """
    Generate predefined adjacency matrices for the hypergraph backbone.
    Creates multi-hop adjacency matrices based on graph structure.

    Args:
        adj_mx: Original adjacency matrix (numpy array or tensor)
        num_nodes: Number of nodes

    Returns:
        List of adjacency matrices (2 matrices for 2-branch backbone)
    """
    if isinstance(adj_mx, np.ndarray):
        adj = torch.tensor(adj_mx, dtype=torch.float32)
    else:
        adj = adj_mx.float()

    # First adjacency: original adjacency with self-loops
    adj1 = adj + torch.eye(num_nodes, device=adj.device)

    # Second adjacency: 2-hop connectivity
    adj2 = torch.mm(adj, adj)
    adj2 = (adj2 > 0).float()
    adj2 = adj2 + torch.eye(num_nodes, device=adj.device)

    return [adj1.numpy(), adj2.numpy()]


class BasicLayer(nn.Module):
    """
    ST-GCN layer that combines spatial and temporal graph convolutions.
    Uses a 3-step temporal window for spatiotemporal feature extraction.
    """

    def __init__(self, adjacency_matrix, num_nodes, hidden_dim, dropout, device,
                 use_learned_adj=True, padding=0):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.device = device
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.use_learned_adj = use_learned_adj
        if use_learned_adj:
            self.weights = nn.Parameter(torch.rand((hidden_dim,)))

        # Build predefined adjacency with temporal expansion (3N x 3N)
        predefined_adj = torch.tensor(adjacency_matrix, dtype=torch.float32)
        adj = torch.zeros((3 * num_nodes, 3 * num_nodes), requires_grad=False)
        adj[:num_nodes, :num_nodes] = predefined_adj
        adj[num_nodes:2*num_nodes, num_nodes:2*num_nodes] = predefined_adj
        adj[-num_nodes:, -num_nodes:] = predefined_adj
        identity = torch.eye(num_nodes)
        adj = adj + identity.repeat(3, 3)
        self.register_buffer('predefined_adj', norm_adj(adj.unsqueeze(0)))  # 1 x 3N x 3N
        self.edge_mask = None
        self.padding = padding

    def forward(self, feat):
        """
        Args:
            feat: Input features of shape (B, T, N, D)

        Returns:
            Output features of shape (B, T-2, N, D)
        """
        batchsize = feat.size(0)
        feat_dim = feat.size(-1)

        if self.padding > 0:
            pad = torch.zeros(
                size=(batchsize, self.padding, self.num_nodes, self.hidden_dim),
                device=feat.device
            )
            feat = torch.cat([feat, pad], dim=1)

        if self.use_learned_adj:
            weighted_feat = F.normalize(feat * torch.sigmoid(self.weights), p=2, dim=-1)
        else:
            weighted_feat = None

        feat_list = []
        for i in range(2, feat.size(1)):
            feature = feat[:, i-2:i+1, :, :]  # B x 3 x N x D
            feature = feature.reshape((batchsize, -1, feat_dim))  # B x (3 x N) x D
            feature_sum = feat[:, i, :, :]

            if self.use_learned_adj:
                weighted_feature = weighted_feat[:, i-2:i+1, :, :]
                weighted_feature = weighted_feature.reshape((batchsize, -1, feat_dim))
                learned_adj_matrix = weighted_feature @ weighted_feature.transpose(1, 2)
                learned_adj_matrix = norm_adj(learned_adj_matrix)
                feature_with_learned_adj = learned_adj_matrix @ feature
                feature_with_learned_adj = self.ffn(feature_with_learned_adj[:, -self.num_nodes:, :])
                feature_sum = feature_sum + feature_with_learned_adj
            else:
                if self.edge_mask is None:
                    feature_with_predefined_adj = self.predefined_adj @ feature
                else:
                    feature_with_predefined_adj = norm_adj(self.predefined_adj * self.edge_mask) @ feature
                feature_sum = feature_sum + self.ffn(feature_with_predefined_adj[:, -self.num_nodes:, :])

            feature_sum = self.layer_norm(feature_sum)
            feat_list.append(feature_sum)

        new_feat = torch.stack(feat_list, dim=1)  # B x T x N x D
        return new_feat


class GraphBranchBackbone(nn.Module):
    """
    Backbone network for GraphBranch using stacked BasicLayers.
    """

    def __init__(self, num_layers, adj_mx, num_nodes, hidden_dim, dropout, device):
        super().__init__()
        self.layers = nn.Sequential(*[
            BasicLayer(adj_mx, num_nodes, hidden_dim, dropout, device,
                       use_learned_adj=False, padding=2)
            for _ in range(num_layers)
        ])

    def forward(self, feature):
        """
        Args:
            feature: Input of shape (B, T, N, D)

        Returns:
            Output of shape (B, T, N, D)
        """
        return self.layers(feature)


class HypergraphLearning(nn.Module):
    """
    Hypergraph learning module that learns hyperedge assignments.
    """

    def __init__(self, hidden_dim, num_edges):
        super().__init__()
        self.num_edges = num_edges
        self.hidden_dim = hidden_dim

        # Edge classifier: maps features to hyperedge assignments
        self.edge_clf = nn.Parameter(
            torch.randn(hidden_dim, num_edges) / math.sqrt(num_edges)
        )
        # Edge mapping: transforms hyperedge features
        self.edge_map = nn.Parameter(
            torch.randn(num_edges, num_edges) / math.sqrt(num_edges)
        )
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Args:
            x: Input features of shape (B, T, N, D)

        Returns:
            Output features of shape (B, T, N, D)
        """
        batch_size, time_len, num_nodes, feat_dim = x.shape

        # Reshape to (B, T*N, D)
        feat = x.reshape(batch_size, -1, feat_dim)

        # Compute soft hyperedge assignment: (B, T*N, E)
        hyper_assignment = torch.softmax(feat @ self.edge_clf, dim=-1)

        # Aggregate node features to hyperedges: (B, E, D)
        hyper_feat = hyper_assignment.transpose(1, 2) @ feat

        # Transform hyperedge features: (B, E, D)
        hyper_feat_mapped = self.activation(self.edge_map @ hyper_feat)
        hyper_out = hyper_feat_mapped + hyper_feat

        # Distribute hyperedge features back to nodes: (B, T*N, D)
        y = self.activation(hyper_assignment @ hyper_out)

        # Reshape back and apply residual connection
        y = y.reshape(batch_size, time_len, num_nodes, feat_dim)
        y_final = self.norm(y + x)

        return y_final


class ModuleWithHypergraphLearning(nn.Module):
    """
    Module that stacks multiple hypergraph learning layers.
    """

    def __init__(self, hidden_dim, dropout, depth=3, num_edges=32, hyper=None):
        super().__init__()
        self.depth = depth
        self.hypers = HypergraphLearning(hidden_dim, num_edges) if hyper is None else hyper
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i in range(self.depth):
            x = self.hypers(x)
            if i != self.depth - 1:
                x = self.dropout(x)
        return x


class HypergraphBackbone(nn.Module):
    """
    Backbone for HypergraphBranch using multi-scale adjacency matrices.
    """

    def __init__(self, num_layers, predefined_adjs, num_nodes, hidden_dim, dropout, device):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(*[
                BasicLayer(predefined_adjs[i], num_nodes, hidden_dim, dropout, device,
                           use_learned_adj=False, padding=2)
                for _ in range(num_layers)
            ])
            for i in range(min(len(predefined_adjs), 2))
        ])

    def forward(self, feature):
        """
        Args:
            feature: Input of shape (B, T, N, D)

        Returns:
            Output of shape (B, T, N, D) - max pooled across scales
        """
        feature_list = []
        for layer in self.layers:
            x = layer(feature)
            feature_list.append(x)
        # Max pooling across different scales
        feature = torch.stack(feature_list, dim=3).max(dim=3)[0]
        return feature


class HypergraphBranchEncoder(nn.Module):
    """
    Encoder for HypergraphBranch combining backbone and hypergraph learning.
    """

    def __init__(self, predefined_adjs, adj_mx, num_nodes, hidden_dim, dropout,
                 hgnn_num_backbone_layers, hgnn_num_head_layers, hgnn_num_hyper_edge, device):
        super().__init__()
        self.backbone = HypergraphBackbone(
            hgnn_num_backbone_layers, predefined_adjs, num_nodes, hidden_dim, dropout, device
        )
        self.hyper = HypergraphLearning(hidden_dim, hgnn_num_hyper_edge)
        self.hgnn_core = ModuleWithHypergraphLearning(
            hidden_dim, dropout, depth=hgnn_num_head_layers, hyper=self.hyper
        )

    def forward(self, x):
        x = self.backbone(x)
        output = self.hgnn_core(x)
        return output


class LEAF(AbstractTrafficStateModel):
    """
    LEAF: Large Language Model Enhanced Adaptive Framework for Traffic Prediction.

    This is a simplified version focusing on the core prediction components:
    - GraphBranch: ST-GCN based spatial-temporal learning
    - HypergraphBranch (optional): Hypergraph-based learning

    The LLM-based selection and test-time adaptation features from the original
    paper are not included in this basic implementation.
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._logger = getLogger()

        # Data features
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self._scaler = self.data_feature.get('scaler')

        # Model configuration
        self.device = config.get('device', torch.device('cpu'))
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.dropout = config.get('dropout', 0.0)

        # GraphBranch configuration
        self.stgcn_num_layers = config.get('stgcn_num_layers', 7)

        # HypergraphBranch configuration
        self.use_hypergraph = config.get('use_hypergraph', False)
        self.hgnn_num_backbone_layers = config.get('hgnn_num_backbone_layers', 6)
        self.hgnn_num_head_layers = config.get('hgnn_num_head_layers', 1)
        self.hgnn_num_hyper_edge = config.get('hgnn_num_hyper_edge', 32)

        # Get adjacency matrix
        adj_mx = data_feature.get('adj_mx')
        if adj_mx is None:
            self._logger.warning('No adjacency matrix provided, using identity matrix')
            adj_mx = np.eye(self.num_nodes)

        # Add self-loops
        self.adj_mx = adj_mx + np.eye(self.num_nodes)

        # Generate predefined adjacencies for hypergraph
        if self.use_hypergraph:
            self.predefined_adjs = generate_predefined_adjs(adj_mx, self.num_nodes)
        else:
            self.predefined_adjs = None

        # Embedding layers
        # Note: In LibCity, time features (time_of_day, day_of_week) are concatenated into X
        # when add_time_in_day=True or add_day_in_week=True, so feature_dim already accounts
        # for them. We only need input and node embeddings.
        self.embeddings = nn.ModuleDict({
            'input': nn.Linear(self.feature_dim, self.hidden_dim),
            'node': nn.Embedding(self.num_nodes, self.hidden_dim)
        })

        # Build backbone
        if self.use_hypergraph:
            self._logger.info('Building LEAF with HypergraphBranch')
            self.encoder = HypergraphBranchEncoder(
                predefined_adjs=self.predefined_adjs,
                adj_mx=self.adj_mx,
                num_nodes=self.num_nodes,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
                hgnn_num_backbone_layers=self.hgnn_num_backbone_layers,
                hgnn_num_head_layers=self.hgnn_num_head_layers,
                hgnn_num_hyper_edge=self.hgnn_num_hyper_edge,
                device=self.device
            )
        else:
            self._logger.info('Building LEAF with GraphBranch')
            self.backbone = GraphBranchBackbone(
                num_layers=self.stgcn_num_layers,
                adj_mx=self.adj_mx,
                num_nodes=self.num_nodes,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
                device=self.device
            )

        # Output layer: (B, N, T*D) -> (B, N, output_window * output_dim)
        self.output_layer = nn.Linear(
            self.hidden_dim * self.input_window,
            self.output_dim * self.output_window
        )

        self._logger.info(f'LEAF model initialized with {self.num_nodes} nodes, '
                          f'hidden_dim={self.hidden_dim}, layers={self.stgcn_num_layers}')

    def _embed(self, batch):
        """
        Apply embeddings to input features.

        In LibCity, time features (time_of_day, day_of_week) are concatenated into X
        when add_time_in_day=True or add_day_in_week=True is set in config.
        The input embedding layer handles all features including time.

        Args:
            batch: Dictionary with key 'X' containing input tensor

        Returns:
            Embedded features of shape (B, T, N, D)
        """
        x = batch['X']  # (B, T, N, F) - time features already included if add_time_in_day=True

        # Input embedding (handles all features including time)
        input_emb = self.embeddings['input'](x)  # (B, T, N, D)

        # Node embedding: (N, D) -> (1, 1, N, D)
        node_idx = torch.arange(0, self.num_nodes, device=x.device)
        node_emb = self.embeddings['node'](node_idx).unsqueeze(0).unsqueeze(0)

        # Combine embeddings
        feat = input_emb + node_emb

        return feat

    def _head_forward(self, out_feat):
        """
        Apply output head to produce predictions.

        Args:
            out_feat: Features of shape (B, T, N, D)

        Returns:
            Predictions of shape (B, output_window, N, output_dim)
        """
        batch_size = out_feat.size(0)

        # Reshape: (B, T, N, D) -> (B, N, T, D) -> (B, N, T*D)
        out_feat = out_feat.transpose(1, 2).reshape(batch_size, self.num_nodes, -1)

        # Linear projection: (B, N, T*D) -> (B, N, output_window * output_dim)
        prediction = self.output_layer(out_feat)

        # Reshape: (B, N, output_window * output_dim) -> (B, N, output_window, output_dim)
        prediction = prediction.reshape(batch_size, self.num_nodes, self.output_window, self.output_dim)

        # Transpose: (B, N, T, D) -> (B, T, N, D)
        prediction = prediction.transpose(1, 2)

        return prediction

    def forward(self, batch):
        """
        Forward pass of LEAF model.

        Args:
            batch: Dictionary containing:
                - 'X': Input tensor of shape (B, T_in, N, F)
                - 'y': Target tensor of shape (B, T_out, N, F) (optional, for training)
                - 'time_of_day': Time of day indices (optional)
                - 'day_of_week': Day of week indices (optional)

        Returns:
            Predictions of shape (B, T_out, N, output_dim)
        """
        # Apply embeddings
        feat = self._embed(batch)  # (B, T, N, D)

        # Apply backbone
        if self.use_hypergraph:
            out_feat = self.encoder(feat)
        else:
            out_feat = self.backbone(feat)  # (B, T, N, D)

        # Apply output head
        prediction = self._head_forward(out_feat)  # (B, T_out, N, output_dim)

        return prediction

    def predict(self, batch):
        """
        Generate predictions for a batch.

        Args:
            batch: Input batch dictionary

        Returns:
            Predictions of shape (B, output_window, N, output_dim)
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate training loss.

        Args:
            batch: Batch containing 'X' and 'y'

        Returns:
            Loss tensor (scalar)
        """
        y_true = batch['y']  # (B, T, N, F)
        y_predicted = self.predict(batch)  # (B, T, N, output_dim)

        # Inverse transform to original scale
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        # Compute masked Huber loss (similar to original LEAF)
        return loss.huber_loss(y_predicted, y_true)
