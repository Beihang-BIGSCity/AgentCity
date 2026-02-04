"""
SARN: Structure-Aware Road Network Embedding via Contrastive Learning

This model adapts the original SARN implementation for the LibCity framework.
SARN learns road network embeddings using:
- Graph Attention Networks (GAT) for spatial encoding
- Momentum Contrastive Learning (MoCo) with local and global losses
- Feature embedding for road segment attributes

Original Paper: "Structure-Aware Road Network Embedding via Graph Contrastive Learning"
Original Repository: https://github.com/yc-li/SARN

Key Adaptations for LibCity:
- Inherits from AbstractTrafficStateModel
- Uses LibCity's data_feature and config patterns
- Implements predict() and calculate_loss() methods
- Adapts road segment features from LibCity's batch format
"""

import copy
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger

try:
    from torch_geometric.nn import GATConv
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


# =====================================================
# Feature Embedding Module
# =====================================================
class FeatEmbedding(nn.Module):
    """
    Embeds discrete road segment features into dense vectors.

    Features embedded:
    - highway_cls: road type classification
    - length_code: discretized road length
    - radian_code: discretized road direction
    - lon_code/lat_code: discretized coordinates (start and end)
    """
    def __init__(self, nhighway_code, nlength_code, nradian_code,
                 nlon_code, nlat_code,
                 highway_dim=16, length_dim=16, radian_dim=16, lonlat_dim=32):
        super(FeatEmbedding, self).__init__()

        self.emb_highway = nn.Embedding(nhighway_code, highway_dim)
        self.emb_length = nn.Embedding(nlength_code, length_dim)
        self.emb_radian = nn.Embedding(nradian_code, radian_dim)
        self.emb_lon = nn.Embedding(nlon_code, lonlat_dim)
        self.emb_lat = nn.Embedding(nlat_code, lonlat_dim)

        # Total output dimension
        self.output_dim = highway_dim + length_dim + radian_dim + 4 * lonlat_dim

    def forward(self, inputs):
        """
        Args:
            inputs: [N, n_features] tensor containing feature codes
                   Expected columns: [wayid_code, segid_code, highway_cls, length_code,
                                     radian_code, s_lon_code, s_lat_code, e_lon_code, e_lat_code, ...]
        Returns:
            [N, output_dim] embedded features
        """
        # Map input columns to embeddings
        # Column indices based on original SARN feature ordering
        highway_emb = self.emb_highway(inputs[:, 2])  # highway_cls at index 2
        length_emb = self.emb_length(inputs[:, 3])    # length_code at index 3
        radian_emb = self.emb_radian(inputs[:, 4])    # radian_code at index 4
        s_lon_emb = self.emb_lon(inputs[:, 5])        # s_lon_code at index 5
        s_lat_emb = self.emb_lat(inputs[:, 6])        # s_lat_code at index 6
        e_lon_emb = self.emb_lon(inputs[:, 7])        # e_lon_code at index 7
        e_lat_emb = self.emb_lat(inputs[:, 8])        # e_lat_code at index 8

        return torch.cat((
            highway_emb, length_emb, radian_emb,
            s_lon_emb, s_lat_emb, e_lon_emb, e_lat_emb
        ), dim=1)


# =====================================================
# GAT Encoder Module (using PyTorch Geometric)
# =====================================================
class GATEncoder(nn.Module):
    """
    Graph Attention Network encoder for road segment features.
    Uses PyTorch Geometric's GATConv layers.
    """
    def __init__(self, nfeat, nhid, nout, nhead=4, nlayer=2, dropout=0.2):
        super(GATEncoder, self).__init__()
        assert nlayer >= 1

        self.nlayer = nlayer
        self.dropout = dropout
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(GATConv(nfeat, nhid, heads=nhead,
                                   dropout=dropout, negative_slope=0.2))
        # Hidden layers
        for _ in range(nlayer - 1):
            self.layers.append(GATConv(nhid * nhead, nhid, heads=nhead,
                                       dropout=dropout, negative_slope=0.2))
        # Output layer
        self.layer_out = GATConv(nhead * nhid, nout, heads=1, concat=False,
                                 dropout=dropout, negative_slope=0.2)

    def forward(self, x, edge_index):
        """
        Args:
            x: [N, nfeat] node features
            edge_index: [2, E] edge indices in COO format
        Returns:
            [N, nout] node embeddings
        """
        for l in range(self.nlayer):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers[l](x, edge_index)
            x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_out(x, edge_index)

        return x


class GATEncoderSimple(nn.Module):
    """
    Fallback GAT encoder when PyTorch Geometric is not available.
    Uses standard attention mechanism.
    """
    def __init__(self, nfeat, nhid, nout, nhead=4, nlayer=2, dropout=0.2):
        super(GATEncoderSimple, self).__init__()

        self.dropout = dropout
        self.nlayer = nlayer
        self.nhead = nhead

        # Simple MLP-based encoder as fallback
        layers = []
        in_dim = nfeat
        for i in range(nlayer):
            layers.append(nn.Linear(in_dim, nhid))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout))
            in_dim = nhid
        layers.append(nn.Linear(nhid, nout))

        self.encoder = nn.Sequential(*layers)

        self._logger = getLogger()
        self._logger.warning("PyTorch Geometric not available. Using simple MLP encoder instead of GAT.")

    def forward(self, x, edge_index):
        """Fallback forward without graph convolution."""
        return self.encoder(x)


# =====================================================
# Projector Module for Contrastive Learning
# =====================================================
class Projector(nn.Module):
    """MLP projector for contrastive learning."""
    def __init__(self, nin, nhid, nout):
        super(Projector, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(nin, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nout)
        )
        self._reset_parameters()

    def forward(self, x):
        return self.mlp(x)

    def _reset_parameters(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.414)
                nn.init.zeros_(m.bias)


# =====================================================
# Momentum Queue for MoCo
# =====================================================
class MomentumQueue(nn.Module):
    """Multi-queue momentum buffer for contrastive learning."""
    def __init__(self, nhid, queue_size, nqueue):
        super(MomentumQueue, self).__init__()
        self.queue_size = queue_size
        self.nqueue = nqueue

        self.register_buffer("queue", torch.randn(nqueue, nhid, queue_size))
        self.queue = F.normalize(self.queue, dim=1)

        self.register_buffer("ids", torch.full([nqueue, queue_size], -1, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros((nqueue,), dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, k, elem_id, q_id):
        """
        Update queue with new key.

        Args:
            k: feature vector
            elem_id: element identifier
            q_id: queue identifier
        """
        ptr = int(self.queue_ptr[q_id].item())
        self.queue[q_id, :, ptr] = k.T
        self.ids[q_id, ptr] = elem_id

        ptr = (ptr + 1) % self.queue_size
        self.queue_ptr[q_id] = ptr


# =====================================================
# MoCo Module for Contrastive Learning
# =====================================================
class MoCo(nn.Module):
    """
    Momentum Contrast (MoCo) module with local and global losses.
    Uses dual GAT encoders (query and key) with momentum updates.
    """
    def __init__(self, nfeat, nemb, nout, queue_size, nqueue,
                 mmt=0.999, temperature=0.07, use_torch_geometric=True):
        super(MoCo, self).__init__()

        self.queue_size = queue_size
        self.nqueue = nqueue
        self.mmt = mmt
        self.temperature = temperature

        # Select encoder type
        EncoderClass = GATEncoder if (use_torch_geometric and HAS_TORCH_GEOMETRIC) else GATEncoderSimple

        # Query and Key encoders
        self.encoder_q = EncoderClass(nfeat=nfeat, nhid=nfeat // 2, nout=nemb, nhead=4, nlayer=2)
        self.encoder_k = EncoderClass(nfeat=nfeat, nhid=nfeat // 2, nout=nemb, nhead=4, nlayer=2)

        # Projectors
        self.mlp_q = Projector(nemb, nemb, nout)
        self.mlp_k = Projector(nemb, nemb, nout)

        # Initialize key encoder with query encoder parameters
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Momentum queues
        self.queues = MomentumQueue(nout, queue_size, nqueue)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.mmt + param_q.data * (1. - self.mmt)

        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data = param_k.data * self.mmt + param_q.data * (1. - self.mmt)

    def forward(self, inputs_q, edge_index_q, idx_in_adjsub_q,
                inputs_k, edge_index_k, idx_in_adjsub_k,
                q_ids, elem_ids):
        """
        Forward pass for contrastive learning.

        Args:
            inputs_q: Query node features
            edge_index_q: Query edge indices
            idx_in_adjsub_q: Mapping indices for query
            inputs_k: Key node features
            edge_index_k: Key edge indices
            idx_in_adjsub_k: Mapping indices for key
            q_ids: Queue IDs for each sample
            elem_ids: Element IDs for each sample

        Returns:
            logits_local, labels_local, logits_global, labels_global
        """
        # Compute query features
        q = self.mlp_q(self.encoder_q(inputs_q, edge_index_q))
        q = F.normalize(q, dim=1)

        # Compute key features with momentum encoder
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.mlp_k(self.encoder_k(inputs_k, edge_index_k))
            k = F.normalize(k, dim=1)

        # Select batch samples
        q = q[idx_in_adjsub_q]
        k = k[idx_in_adjsub_k]

        # Positive logits (local)
        l_pos_local = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # Negative logits (local) - from same queue
        neg_local = self.queues.queue[q_ids].clone().detach()
        l_neg_local = torch.einsum('nc,nck->nk', q, neg_local)

        # Mask out same elements
        neg_local_ids = self.queues.ids[q_ids].clone().detach()
        l_neg_local[neg_local_ids == elem_ids.unsqueeze(1).repeat(1, neg_local_ids.shape[1])] = -9e15

        # Negative logits (global) - mean readout across all queues
        neg_global = torch.mean(self.queues.queue.clone().detach(), dim=2)
        l_neg_global = torch.einsum('nc,ck->nk', q, neg_global.T)

        # Combine logits
        logits_local = torch.cat([l_pos_local, l_neg_local], dim=1)
        logits_global = l_neg_global

        # Apply temperature
        logits_local = logits_local / self.temperature
        logits_global = logits_global / self.temperature

        # Labels
        labels_local = torch.zeros_like(l_pos_local, dtype=torch.long).squeeze(1)
        labels_global = q_ids.clone().detach()

        # Update queues
        for i, q_id in enumerate(q_ids):
            self.queues.dequeue_and_enqueue(k[i, :], elem_ids[i].item(), q_id)

        return logits_local, labels_local, logits_global, labels_global

    def loss_mtl(self, logits_local, labels_local, logits_global, labels_global,
                 w_local, w_global):
        """
        Multi-task learning loss combining local and global contrastive losses.

        Args:
            logits_local: Local contrastive logits
            labels_local: Local labels
            logits_global: Global contrastive logits
            labels_global: Global labels (queue indices)
            w_local: Weight for local loss
            w_global: Weight for global loss

        Returns:
            Combined weighted loss
        """
        sfmax_local = F.softmax(logits_local, dim=1)
        sfmax_global = F.softmax(logits_global, dim=1)

        p_local = torch.log(sfmax_local.gather(1, labels_local.view(-1, 1)))
        p_global = torch.log(sfmax_global.gather(1, labels_global.view(-1, 1)))

        loss_local = F.nll_loss(p_local, torch.zeros_like(labels_local))
        loss_global = F.nll_loss(p_global, torch.zeros_like(labels_local))

        return loss_local * w_local + loss_global * w_global

    def encode(self, inputs, edge_index):
        """
        Get embeddings without contrastive learning (for inference).

        Args:
            inputs: Node features
            edge_index: Edge indices

        Returns:
            Normalized embeddings
        """
        embs = self.encoder_q(inputs, edge_index)
        return F.normalize(embs, dim=1)


# =====================================================
# Edge Index Utilities
# =====================================================
class EdgeIndexUtil:
    """
    Utility class for managing edge indices in the graph.
    Handles subgraph extraction and edge augmentation.
    """
    def __init__(self, edges, tweight=None, sweight=None, nnode=None):
        """
        Args:
            edges: [E, 2] numpy array of edges
            tweight: topology weights
            sweight: spatial weights
            nnode: number of nodes
        """
        self.edges = np.array(edges, dtype=np.int64)
        self.nnode = nnode if nnode else int(edges.max()) + 1
        self.tweight = tweight if tweight is not None else np.ones(len(edges))
        self.sweight = sweight if sweight is not None else np.zeros(len(edges))
        self.node_neighbours = None

    def length(self):
        return self.edges.shape[0]

    def remove_edges(self, idxs):
        """Remove edges by indices."""
        self.edges = np.delete(self.edges, idxs, axis=0)
        self.tweight = np.delete(self.tweight, idxs)
        self.sweight = np.delete(self.sweight, idxs)
        self.node_neighbours = None

    def create_adj_index(self):
        """Create adjacency list representation."""
        self.node_neighbours = [([], []) for _ in range(self.nnode)]
        for x, y in self.edges:
            self.node_neighbours[x][1].append(y)
            self.node_neighbours[y][0].append(x)

    def sub_edge_index(self, sub_idx):
        """
        Extract subgraph edge index for given node indices.

        Args:
            sub_idx: List of node indices in the subgraph

        Returns:
            sub_edge_index: [2, E'] edge indices for subgraph
            new_x_idx: indices of nodes in original graph
            mapping_to_origin_idx: mapping from sub_idx to new indices
        """
        if self.node_neighbours is None:
            self.create_adj_index()

        idx = sorted(list(set(sub_idx)))
        idx1 = []
        sub_edge_list = []

        for _i in idx:
            if _i < len(self.node_neighbours):
                idx1.extend(self.node_neighbours[_i][0])
                sub_edge_list.extend([(x, _i) for x in self.node_neighbours[_i][0]])

        idx1 = sorted(list(set(idx1 + idx)))
        idx1_to_newidx = [-1] * self.nnode

        for i, v in enumerate(idx1):
            idx1_to_newidx[v] = i

        sub_edge_index = np.array(
            [(idx1_to_newidx[i], idx1_to_newidx[j]) for (i, j) in sub_edge_list if idx1_to_newidx[i] >= 0 and idx1_to_newidx[j] >= 0],
            dtype=np.int64
        ).T if sub_edge_list else np.zeros((2, 0), dtype=np.int64)

        mapping_to_origin_idx = [idx1_to_newidx[_i] for _i in sub_idx if _i < len(idx1_to_newidx) and idx1_to_newidx[_i] >= 0]

        return sub_edge_index, idx1, mapping_to_origin_idx


def graph_augment_edge_index(edge_index_util, break_prob=0.2):
    """
    Graph augmentation by randomly removing edges based on weights.

    Args:
        edge_index_util: EdgeIndexUtil object
        break_prob: Probability of edge removal

    Returns:
        Augmented EdgeIndexUtil
    """
    aug = copy.deepcopy(edge_index_util)
    n_edges = aug.length()

    if n_edges == 0:
        return aug

    # Compute removal probabilities based on topology weights
    weights = aug.tweight.copy()
    weights_zero = (weights == 0)
    n_topo = n_edges - sum(weights_zero)

    if n_topo > 0:
        max_weight = max(weights) + 1.5
        weights = np.log(max_weight - weights) / np.log(1.5)
        weights[weights_zero] = 0

        sum_weight = sum(weights)
        if sum_weight > 0:
            weights = weights / sum_weight
        else:
            weights = np.ones(n_edges) / n_edges

        n_remove = int(break_prob * n_topo)
        if n_remove > 0:
            edges_to_remove = np.random.choice(n_edges, p=weights, size=n_remove, replace=False)
            aug.remove_edges(list(edges_to_remove))

    return aug


# =====================================================
# Main SARN Model for LibCity
# =====================================================
class SARN(AbstractTrafficStateModel):
    """
    SARN: Structure-Aware Road Network Embedding via Contrastive Learning

    This model learns road segment embeddings using graph attention networks
    and momentum contrastive learning with local and global objectives.

    Config Parameters:
        - output_dim: Dimension of output embeddings (default: 128)
        - feat_dim: Dimension of feature embeddings (default: 176)
        - out_dim: Dimension of contrastive projection (default: 32)
        - moco_queue_size: Size of each momentum queue (default: 100)
        - moco_temperature: Contrastive learning temperature (default: 0.05)
        - moco_loss_local_weight: Weight for local loss (default: 0.4)
        - break_edge_prob: Probability of edge removal for augmentation (default: 0.2)
        - highway_dim: Embedding dim for highway type (default: 16)
        - length_dim: Embedding dim for length code (default: 16)
        - radian_dim: Embedding dim for radian code (default: 16)
        - lonlat_dim: Embedding dim for coordinates (default: 32)
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._logger = getLogger()

        # Data features
        self.adj_mx = data_feature.get('adj_mx')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self._scaler = data_feature.get('scaler')

        # Device
        self.device = config.get('device', torch.device('cpu'))
        self.model = config.get('model', 'SARN')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)

        # Model hyperparameters
        self.output_dim = config.get('output_dim', 128)
        self.feat_dim = config.get('feat_dim', 176)
        self.out_dim = config.get('out_dim', 32)

        # Feature embedding dimensions
        self.highway_dim = config.get('highway_dim', 16)
        self.length_dim = config.get('length_dim', 16)
        self.radian_dim = config.get('radian_dim', 16)
        self.lonlat_dim = config.get('lonlat_dim', 32)

        # MoCo hyperparameters
        self.moco_queue_size = config.get('moco_queue_size', 100)
        self.moco_temperature = config.get('moco_temperature', 0.05)
        self.moco_loss_local_weight = config.get('moco_loss_local_weight', 0.4)
        self.moco_loss_global_weight = 1.0 - self.moco_loss_local_weight
        self.moco_nqueue = config.get('moco_nqueue', 10)  # Number of spatial queues

        # Graph augmentation
        self.break_edge_prob = config.get('break_edge_prob', 0.2)

        # Feature counts (should be provided by data_feature or config)
        self.nhighway_code = config.get('nhighway_code', data_feature.get('nhighway_code', 20))
        self.nlength_code = config.get('nlength_code', data_feature.get('nlength_code', 200))
        self.nradian_code = config.get('nradian_code', data_feature.get('nradian_code', 50))
        self.nlon_code = config.get('nlon_code', data_feature.get('nlon_code', 500))
        self.nlat_code = config.get('nlat_code', data_feature.get('nlat_code', 500))

        # Build edge index from adjacency matrix
        self._build_edge_index()

        # Initialize feature embedding module
        self.feat_emb = FeatEmbedding(
            nhighway_code=self.nhighway_code,
            nlength_code=self.nlength_code,
            nradian_code=self.nradian_code,
            nlon_code=self.nlon_code,
            nlat_code=self.nlat_code,
            highway_dim=self.highway_dim,
            length_dim=self.length_dim,
            radian_dim=self.radian_dim,
            lonlat_dim=self.lonlat_dim
        )

        # Update feat_dim based on actual embedding output
        self.feat_dim = self.feat_emb.output_dim

        # Initialize MoCo model
        self.moco = MoCo(
            nfeat=self.feat_dim,
            nemb=self.output_dim,
            nout=self.out_dim,
            queue_size=self.moco_queue_size,
            nqueue=self.moco_nqueue,
            mmt=0.999,
            temperature=self.moco_temperature,
            use_torch_geometric=HAS_TORCH_GEOMETRIC
        )

        self._logger.info(f"SARN initialized with output_dim={self.output_dim}, "
                         f"feat_dim={self.feat_dim}, num_nodes={self.num_nodes}, "
                         f"moco_nqueue={self.moco_nqueue}")

    def _build_edge_index(self):
        """Build edge index from adjacency matrix."""
        if self.adj_mx is not None:
            # Handle sparse matrix
            if hasattr(self.adj_mx, 'row') and hasattr(self.adj_mx, 'col'):
                edges = np.stack([self.adj_mx.row, self.adj_mx.col], axis=1)
                data = self.adj_mx.data if hasattr(self.adj_mx, 'data') else np.ones(len(edges))
            elif hasattr(self.adj_mx, 'tocoo'):
                coo = self.adj_mx.tocoo()
                edges = np.stack([coo.row, coo.col], axis=1)
                data = coo.data
            else:
                # Dense matrix
                adj = np.array(self.adj_mx)
                rows, cols = np.where(adj > 0)
                edges = np.stack([rows, cols], axis=1)
                data = adj[rows, cols]

            self.edge_index_util = EdgeIndexUtil(
                edges=edges,
                tweight=data,
                nnode=self.num_nodes
            )
            self.edge_index = torch.tensor(edges.T, dtype=torch.long, device=self.device)
        else:
            # Empty edge index
            self.edge_index_util = EdgeIndexUtil(
                edges=np.zeros((0, 2), dtype=np.int64),
                nnode=self.num_nodes
            )
            self.edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)

    def forward(self, batch):
        """
        Forward pass for the SARN model.

        Args:
            batch: dict containing:
                - 'node_features': [N, n_features] tensor of node features
                - Optional: 'edge_index' override

        Returns:
            [N, output_dim] node embeddings
        """
        # Get node features
        if 'node_features' in batch:
            node_features = batch['node_features']
        elif 'X' in batch:
            # Traffic state format: (batch, time, nodes, features) -> take last timestep
            x = batch['X']
            if x.dim() == 4:
                node_features = x[:, -1, :, :].reshape(-1, x.shape[-1])
            else:
                node_features = x
        else:
            raise ValueError("batch must contain 'node_features' or 'X'")

        # Move to device and ensure long type for embeddings
        node_features = node_features.to(self.device)
        if node_features.dtype != torch.long:
            node_features = node_features.long()

        # Embed features
        embedded_features = self.feat_emb(node_features)

        # Get edge index
        if 'edge_index' in batch:
            edge_index = batch['edge_index'].to(self.device)
        else:
            edge_index = self.edge_index.to(self.device)

        # Get embeddings from encoder
        embeddings = self.moco.encode(embedded_features, edge_index)

        return embeddings

    def predict(self, batch):
        """
        Get embeddings for prediction/inference and save to disk.

        Args:
            batch: dict containing node features

        Returns:
            [N, output_dim] normalized embeddings
        """
        embeddings = self.forward(batch)

        # Save embeddings to disk (following ChebConv pattern)
        save_path = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.npy'.format(
            self.exp_id, self.model, self.dataset, self.output_dim)
        np.save(save_path, embeddings.detach().cpu().numpy())
        self._logger.info('Saved embeddings to {}'.format(save_path))

        return embeddings

    def calculate_loss(self, batch):
        """
        Calculate contrastive learning loss.

        For road representation, we use MoCo contrastive loss with:
        - Local loss: positive pairs from same node, negatives from same spatial queue
        - Global loss: negatives from all queues

        Args:
            batch: dict containing:
                - 'node_features': [N, n_features]
                - 'cell_ids': [N] spatial cell IDs for each node (optional)
                - 'node_ids': [N] unique node IDs (optional)

        Returns:
            loss tensor
        """
        # Get node features
        if 'node_features' in batch:
            node_features = batch['node_features']
        elif 'X' in batch:
            x = batch['X']
            if x.dim() == 4:
                node_features = x[:, -1, :, :].reshape(-1, x.shape[-1])
            else:
                node_features = x
        else:
            raise ValueError("batch must contain 'node_features' or 'X'")

        node_features = node_features.to(self.device)
        if node_features.dtype != torch.long:
            node_features = node_features.long()

        n_nodes = node_features.shape[0]

        # Create augmented graphs
        edge_index_1 = graph_augment_edge_index(self.edge_index_util, self.break_edge_prob)
        edge_index_2 = graph_augment_edge_index(self.edge_index_util, self.break_edge_prob)

        # Sample batch of nodes
        batch_size = min(128, n_nodes)
        batch_indices = random.sample(range(n_nodes), batch_size)

        # Get subgraph for view 1
        sub_edge_1, new_x_idx_1, mapping_1 = edge_index_1.sub_edge_index(batch_indices)
        sub_edge_1 = torch.tensor(sub_edge_1, dtype=torch.long, device=self.device)
        sub_features_1 = self.feat_emb(node_features[new_x_idx_1])

        # Get subgraph for view 2
        sub_edge_2, new_x_idx_2, mapping_2 = edge_index_2.sub_edge_index(batch_indices)
        sub_edge_2 = torch.tensor(sub_edge_2, dtype=torch.long, device=self.device)
        sub_features_2 = self.feat_emb(node_features[new_x_idx_2])

        # Get cell IDs and node IDs
        if 'cell_ids' in batch:
            cell_ids = batch['cell_ids'][batch_indices].to(self.device)
        else:
            # Assign random queue IDs if not provided
            cell_ids = torch.randint(0, self.moco_nqueue, (len(batch_indices),), device=self.device)

        if 'node_ids' in batch:
            node_ids = batch['node_ids'][batch_indices].to(self.device)
        else:
            node_ids = torch.tensor(batch_indices, dtype=torch.long, device=self.device)

        # Ensure mappings are valid
        if len(mapping_1) == 0 or len(mapping_2) == 0:
            # Fallback: use simple autoencoding loss
            embeddings = self.forward(batch)
            if 'node_labels' in batch:
                return loss.masked_mse_torch(embeddings, batch['node_labels'].to(self.device))
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        mapping_1 = torch.tensor(mapping_1[:len(batch_indices)], dtype=torch.long, device=self.device)
        mapping_2 = torch.tensor(mapping_2[:len(batch_indices)], dtype=torch.long, device=self.device)

        # Ensure consistent batch size
        min_len = min(len(mapping_1), len(mapping_2), len(cell_ids), len(node_ids))
        mapping_1 = mapping_1[:min_len]
        mapping_2 = mapping_2[:min_len]
        cell_ids = cell_ids[:min_len]
        node_ids = node_ids[:min_len]

        # Forward through MoCo
        logits_local, labels_local, logits_global, labels_global = self.moco(
            sub_features_1, sub_edge_1, mapping_1,
            sub_features_2, sub_edge_2, mapping_2,
            cell_ids, node_ids
        )

        # Calculate loss
        contrastive_loss = self.moco.loss_mtl(
            logits_local, labels_local,
            logits_global, labels_global,
            self.moco_loss_local_weight,
            self.moco_loss_global_weight
        )

        return contrastive_loss

    @torch.no_grad()
    def get_embeddings(self, batch):
        """
        Get all node embeddings (for saving/analysis).

        Args:
            batch: dict containing node features

        Returns:
            [N, output_dim] CPU tensor of embeddings
        """
        self.eval()
        embeddings = self.predict(batch)
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
            save_path = f'./libcity/cache/{self.exp_id}/evaluate_cache/embedding_{self.model}_{self.dataset}_{self.output_dim}.npy'

        np.save(save_path, embeddings.numpy())
        self._logger.info(f"Embeddings saved to {save_path}")
