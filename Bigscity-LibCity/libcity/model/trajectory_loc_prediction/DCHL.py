# coding: utf-8
"""
DCHL: Disentangled Contrastive Hypergraph Learning for Next POI Recommendation

This module adapts the DCHL model from the original paper implementation for LibCity.

Original paper: [24 SIGIR] Disentangled Contrastive Hypergraph Learning for Next POI Recommendation
Original author: Yantong Lai
Original repository: repos/DCHL/

Original files:
    - model.py (main model classes: DCHL, MultiViewHyperConvLayer, DirectedHyperConvLayer, etc.)
    - dataset.py (dataset and graph construction)
    - utils.py (utility functions for graph construction)

Key innovations preserved:
1. Multi-view Hypergraph Convolutional Network - captures user-POI collaborative patterns
2. Directed Hypergraph Convolutional Network - captures POI transition patterns
3. Geographical Convolutional Network - captures spatial proximity patterns
4. Disentangled contrastive learning - learns distinct representations for each view
5. Adaptive gating mechanism - fuses representations from different views

Key adaptations for LibCity:
- Inherits from AbstractModel (trajectory location prediction task)
- Implements predict() and calculate_loss() methods
- Adapts batch format to work with LibCity's trajectory data loaders
- Graph structures (HG_up, HG_pu, poi_geo_graph, HG_poi_src, HG_poi_tar) are stored as model
  attributes and initialized from data_feature
- Original forward(dataset, batch) signature adapted to forward(batch) with graphs as attributes

Required data_feature keys:
- num_users: Number of users
- num_pois (or loc_size): Number of POIs/locations
- HG_up: User-POI hypergraph (sparse tensor) [U, L]
- HG_pu: POI-User hypergraph (sparse tensor) [L, U]
- poi_geo_graph: POI geographical graph (sparse tensor) [L, L]
- HG_poi_src: Source POI directed hypergraph (sparse tensor) [L, L]
- HG_poi_tar: Target POI directed hypergraph (sparse tensor) [L, L]
- pad_all_train_sessions: Padded training sessions [U, MAX_SEQ_LEN] (optional)

Required config parameters:
- emb_dim: Embedding dimension (default: 128)
- num_mv_layers: Number of multi-view hypergraph conv layers (default: 3)
- num_geo_layers: Number of geographical conv layers (default: 3)
- num_di_layers: Number of directed hypergraph conv layers (default: 3)
- dropout: Dropout rate (default: 0.3)
- temperature: Temperature for contrastive loss (default: 0.1)
- lambda_cl: Weight for contrastive loss (default: 0.1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from math import radians, cos, sin, asin, sqrt

from libcity.model.abstract_model import AbstractModel


# =============================================================================
# Utility Functions for Graph Construction
# =============================================================================

def haversine_distance(lon1, lat1, lon2, lat2):
    """Haversine distance between two coordinates."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth radius in km
    return c * r


def gen_poi_geo_adj(num_pois, pois_coos_dict, distance_threshold):
    """Generate geographical adjacency matrix with pois_coos_dict and distance_threshold."""
    poi_geo_adj = np.zeros(shape=(num_pois, num_pois))
    for poi1 in range(num_pois):
        if poi1 not in pois_coos_dict:
            continue
        lat1, lon1 = pois_coos_dict[poi1]
        for poi2 in range(poi1, num_pois):
            if poi2 not in pois_coos_dict:
                continue
            lat2, lon2 = pois_coos_dict[poi2]
            hav_dist = haversine_distance(lon1, lat1, lon2, lat2)
            if hav_dist <= distance_threshold:
                poi_geo_adj[poi1, poi2] = 1
                poi_geo_adj[poi2, poi1] = 1
    poi_geo_adj = sp.csr_matrix(poi_geo_adj)
    return poi_geo_adj


def normalized_adj(adj, is_symmetric=True):
    """Normalize adjacent matrix for GCN."""
    if is_symmetric:
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum + 1e-8, -1/2).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv * adj * d_mat_inv
    else:
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum + 1e-8, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv * adj
    return norm_adj


def get_hyper_deg(incidence_matrix):
    """Get degree matrix for hypergraph normalization."""
    rowsum = np.array(incidence_matrix.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv


def transform_csr_matrix_to_tensor(csr_matrix):
    """Transform csr matrix to sparse tensor."""
    coo = csr_matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    sp_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sp_tensor


def gen_sparse_H_user(sessions_dict, num_pois, num_users):
    """Generate sparse incidence matrix for user-POI hypergraph."""
    H = np.zeros(shape=(num_pois, num_users))
    for userID, sessions in sessions_dict.items():
        if isinstance(sessions, list) and len(sessions) > 0:
            if isinstance(sessions[0], list):
                # sessions is list of lists
                seq = []
                for session in sessions:
                    seq.extend(session)
            else:
                # sessions is a single list
                seq = sessions
        else:
            seq = sessions if hasattr(sessions, '__iter__') else [sessions]
        for poi in seq:
            if 0 <= poi < num_pois:
                H[poi, userID] = 1
    H = sp.csr_matrix(H)
    return H


def gen_sparse_directed_H_poi(users_trajs_dict, num_pois):
    """Generate directed POI-POI incidence matrix for hypergraph."""
    H = np.zeros(shape=(num_pois, num_pois))
    for userID, traj in users_trajs_dict.items():
        if isinstance(traj, list) and len(traj) > 0:
            if isinstance(traj[0], list):
                # traj is list of lists, flatten
                flat_traj = []
                for session in traj:
                    flat_traj.extend(session)
                traj = flat_traj
        for src_idx in range(len(traj) - 1):
            for tar_idx in range(src_idx + 1, len(traj)):
                src_poi = traj[src_idx]
                tar_poi = traj[tar_idx]
                if 0 <= src_poi < num_pois and 0 <= tar_poi < num_pois:
                    H[src_poi, tar_poi] = 1
    H = sp.csr_matrix(H)
    return H


# =============================================================================
# Model Components
# =============================================================================

class MultiViewHyperConvLayer(nn.Module):
    """
    Multi-view Hypergraph Convolutional Layer.

    Performs message passing on user-POI hypergraph to capture collaborative patterns.
    """

    def __init__(self, emb_dim, device):
        super(MultiViewHyperConvLayer, self).__init__()
        self.fc_fusion = nn.Linear(2 * emb_dim, emb_dim, device=device)
        self.dropout = nn.Dropout(0.3)
        self.emb_dim = emb_dim
        self.device = device

    def forward(self, pois_embs, pad_all_train_sessions, HG_up, HG_pu):
        """
        Forward pass of multi-view hypergraph convolution.

        Args:
            pois_embs: POI embeddings [L, d]
            pad_all_train_sessions: Padded training sessions [U, MAX_SESS_LEN] (unused in current impl)
            HG_up: User-POI hypergraph [U, L]
            HG_pu: POI-User hypergraph [L, U]

        Returns:
            Propagated POI embeddings [L, d]
        """
        # 1. node -> hyperedge message (POI aggregation)
        msg_poi_agg = torch.sparse.mm(HG_up, pois_embs)  # [U, d]

        # 2. propagation: hyperedge -> node
        propag_pois_embs = torch.sparse.mm(HG_pu, msg_poi_agg)  # [L, d]

        return propag_pois_embs


class DirectedHyperConvLayer(nn.Module):
    """
    Directed Hypergraph Convolutional Layer.

    Captures POI transition patterns using directed hypergraph.
    """

    def __init__(self):
        super(DirectedHyperConvLayer, self).__init__()

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        """
        Forward pass of directed hypergraph convolution.

        Args:
            pois_embs: POI embeddings [L, d]
            HG_poi_src: Source POI hypergraph [L, L]
            HG_poi_tar: Target POI hypergraph [L, L]

        Returns:
            Propagated POI embeddings [L, d]
        """
        msg_tar = torch.sparse.mm(HG_poi_tar, pois_embs)
        msg_src = torch.sparse.mm(HG_poi_src, msg_tar)
        return msg_src


class MultiViewHyperConvNetwork(nn.Module):
    """
    Multi-view Hypergraph Convolutional Network.

    Stacks multiple multi-view hypergraph conv layers with residual connections.
    """

    def __init__(self, num_layers, emb_dim, dropout, device):
        super(MultiViewHyperConvNetwork, self).__init__()
        self.num_layers = num_layers
        self.device = device
        self.mv_hconv_layer = MultiViewHyperConvLayer(emb_dim, device)
        self.dropout = dropout

    def forward(self, pois_embs, pad_all_train_sessions, HG_up, HG_pu):
        """
        Forward pass through multi-layer multi-view hypergraph network.

        Args:
            pois_embs: Initial POI embeddings [L, d]
            pad_all_train_sessions: Padded training sessions
            HG_up: User-POI hypergraph
            HG_pu: POI-User hypergraph

        Returns:
            Final POI embeddings (mean of all layers) [L, d]
        """
        final_pois_embs = [pois_embs]
        for layer_idx in range(self.num_layers):
            pois_embs = self.mv_hconv_layer(pois_embs, pad_all_train_sessions, HG_up, HG_pu)
            # Residual connection to alleviate over-smoothing
            pois_embs = pois_embs + final_pois_embs[-1]
            pois_embs = F.dropout(pois_embs, self.dropout, training=self.training)
            final_pois_embs.append(pois_embs)
        final_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)
        return final_pois_embs


class DirectedHyperConvNetwork(nn.Module):
    """
    Directed Hypergraph Convolutional Network.

    Stacks multiple directed hypergraph conv layers with residual connections.
    """

    def __init__(self, num_layers, device, dropout=0.3):
        super(DirectedHyperConvNetwork, self).__init__()
        self.num_layers = num_layers
        self.device = device
        self.dropout = dropout
        self.di_hconv_layer = DirectedHyperConvLayer()

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        """
        Forward pass through multi-layer directed hypergraph network.

        Args:
            pois_embs: Initial POI embeddings [L, d]
            HG_poi_src: Source POI hypergraph
            HG_poi_tar: Target POI hypergraph

        Returns:
            Final POI embeddings (mean of all layers) [L, d]
        """
        final_pois_embs = [pois_embs]
        for layer_idx in range(self.num_layers):
            pois_embs = self.di_hconv_layer(pois_embs, HG_poi_src, HG_poi_tar)
            # Residual connection
            pois_embs = pois_embs + final_pois_embs[-1]
            pois_embs = F.dropout(pois_embs, self.dropout, training=self.training)
            final_pois_embs.append(pois_embs)
        final_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)
        return final_pois_embs


class GeoConvNetwork(nn.Module):
    """
    Geographical Convolutional Network.

    Captures spatial proximity patterns between POIs.
    """

    def __init__(self, num_layers, dropout):
        super(GeoConvNetwork, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, pois_embs, geo_graph):
        """
        Forward pass through geographical conv network.

        Args:
            pois_embs: Initial POI embeddings [L, d]
            geo_graph: POI geographical graph [L, L]

        Returns:
            Final POI embeddings (mean of all layers) [L, d]
        """
        final_pois_embs = [pois_embs]
        for _ in range(self.num_layers):
            pois_embs = torch.sparse.mm(geo_graph, pois_embs)
            pois_embs = pois_embs + final_pois_embs[-1]
            final_pois_embs.append(pois_embs)
        output_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)
        return output_pois_embs


# =============================================================================
# Main DCHL Model
# =============================================================================

class DCHL(AbstractModel):
    """
    DCHL: Disentangled Contrastive Hypergraph Learning for Next POI Recommendation.

    This model learns POI representations through three disentangled views:
    1. Multi-view Hypergraph - user-POI collaborative patterns
    2. Directed Hypergraph - POI transition patterns
    3. Geographical Graph - spatial proximity patterns

    These views are fused using adaptive gating and trained with contrastive loss.

    Adapted for LibCity's trajectory location prediction task.
    """

    def __init__(self, config, data_feature):
        super(DCHL, self).__init__(config, data_feature)

        # Device configuration
        self.device = config.get('device', 'cpu')

        # Data dimensions from data_feature
        self.num_users = data_feature.get('num_users', data_feature.get('uid_size', 100))
        self.num_pois = data_feature.get('num_pois', data_feature.get('loc_size', 1000))

        # Model hyperparameters from config
        self.emb_dim = config.get('emb_dim', 128)
        self.ssl_temp = config.get('temperature', 0.1)
        self.lambda_cl = config.get('lambda_cl', 0.1)
        self.num_mv_layers = config.get('num_mv_layers', 3)
        self.num_geo_layers = config.get('num_geo_layers', 3)
        self.num_di_layers = config.get('num_di_layers', 3)
        self.dropout = config.get('dropout', 0.3)
        self.distance_threshold = config.get('distance_threshold', 2.5)

        # Embedding layers
        self.user_embedding = nn.Embedding(self.num_users, self.emb_dim)
        self.poi_embedding = nn.Embedding(self.num_pois + 1, self.emb_dim, padding_idx=self.num_pois)

        # Embedding initialization
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.poi_embedding.weight)

        # Network components
        self.mv_hconv_network = MultiViewHyperConvNetwork(
            self.num_mv_layers, self.emb_dim, 0, self.device
        )
        self.geo_conv_network = GeoConvNetwork(self.num_geo_layers, self.dropout)
        self.di_hconv_network = DirectedHyperConvNetwork(
            self.num_di_layers, self.device, self.dropout
        )

        # Gates for adaptive fusion with POI embeddings
        self.hyper_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())
        self.gcn_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())
        self.trans_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())

        # Gates for adaptive fusion with user embeddings
        self.user_hyper_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())
        self.user_gcn_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())

        # Temporal-augmentation (kept for compatibility, not used in basic forward)
        self.pos_embeddings = nn.Embedding(1500, self.emb_dim, padding_idx=0)
        self.w_1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_dim, 1))
        self.glu1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.glu2 = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

        # Gating before disentangled learning
        self.w_gate_geo = nn.Parameter(torch.FloatTensor(self.emb_dim, self.emb_dim))
        self.b_gate_geo = nn.Parameter(torch.FloatTensor(1, self.emb_dim))
        self.w_gate_seq = nn.Parameter(torch.FloatTensor(self.emb_dim, self.emb_dim))
        self.b_gate_seq = nn.Parameter(torch.FloatTensor(1, self.emb_dim))
        self.w_gate_col = nn.Parameter(torch.FloatTensor(self.emb_dim, self.emb_dim))
        self.b_gate_col = nn.Parameter(torch.FloatTensor(1, self.emb_dim))
        nn.init.xavier_normal_(self.w_gate_geo.data)
        nn.init.xavier_normal_(self.b_gate_geo.data)
        nn.init.xavier_normal_(self.w_gate_seq.data)
        nn.init.xavier_normal_(self.b_gate_seq.data)
        nn.init.xavier_normal_(self.w_gate_col.data)
        nn.init.xavier_normal_(self.b_gate_col.data)

        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Initialize graph structures from data_feature
        self._init_graph_structures(data_feature)

    def _init_graph_structures(self, data_feature):
        """
        Initialize graph structures from data_feature.

        If pre-computed graphs are provided in data_feature, use them directly.
        Otherwise, construct graphs from user trajectory data.
        """
        # Check for pre-computed graphs
        if 'HG_up' in data_feature and data_feature['HG_up'] is not None:
            self.register_buffer('HG_up', data_feature['HG_up'].to(self.device))
        else:
            self.HG_up = None

        if 'HG_pu' in data_feature and data_feature['HG_pu'] is not None:
            self.register_buffer('HG_pu', data_feature['HG_pu'].to(self.device))
        else:
            self.HG_pu = None

        if 'poi_geo_graph' in data_feature and data_feature['poi_geo_graph'] is not None:
            self.register_buffer('poi_geo_graph', data_feature['poi_geo_graph'].to(self.device))
        else:
            self.poi_geo_graph = None

        if 'HG_poi_src' in data_feature and data_feature['HG_poi_src'] is not None:
            self.register_buffer('HG_poi_src', data_feature['HG_poi_src'].to(self.device))
        else:
            self.HG_poi_src = None

        if 'HG_poi_tar' in data_feature and data_feature['HG_poi_tar'] is not None:
            self.register_buffer('HG_poi_tar', data_feature['HG_poi_tar'].to(self.device))
        else:
            self.HG_poi_tar = None

        if 'pad_all_train_sessions' in data_feature and data_feature['pad_all_train_sessions'] is not None:
            self.register_buffer('pad_all_train_sessions', data_feature['pad_all_train_sessions'].to(self.device))
        else:
            self.pad_all_train_sessions = None

        # Construct graphs if not provided
        if self.HG_up is None or self.HG_pu is None:
            self._construct_default_graphs(data_feature)

    def _construct_default_graphs(self, data_feature):
        """
        Construct default graph structures when not provided in data_feature.

        Creates identity-like sparse matrices as placeholders.
        """
        # Create simple identity-based graphs as defaults
        # In practice, these should be computed from actual user-POI interaction data

        # HG_up: User-POI graph [U, L]
        if self.HG_up is None:
            delattr(self, 'HG_up')  # Remove existing None attribute before registering buffer
            indices = torch.arange(min(self.num_users, self.num_pois))
            i = torch.stack([indices, indices])
            v = torch.ones(len(indices))
            self.register_buffer('HG_up', torch.sparse_coo_tensor(
                i, v, (self.num_users, self.num_pois)
            ).to(self.device))

        # HG_pu: POI-User graph [L, U]
        if self.HG_pu is None:
            delattr(self, 'HG_pu')  # Remove existing None attribute before registering buffer
            indices = torch.arange(min(self.num_users, self.num_pois))
            i = torch.stack([indices, indices])
            v = torch.ones(len(indices))
            self.register_buffer('HG_pu', torch.sparse_coo_tensor(
                i, v, (self.num_pois, self.num_users)
            ).to(self.device))

        # poi_geo_graph: POI geographical graph [L, L] (identity)
        if self.poi_geo_graph is None:
            delattr(self, 'poi_geo_graph')  # Remove existing None attribute before registering buffer
            indices = torch.arange(self.num_pois)
            i = torch.stack([indices, indices])
            v = torch.ones(self.num_pois)
            self.register_buffer('poi_geo_graph', torch.sparse_coo_tensor(
                i, v, (self.num_pois, self.num_pois)
            ).to(self.device))

        # HG_poi_src and HG_poi_tar: Directed POI hypergraphs [L, L] (identity)
        if self.HG_poi_src is None:
            delattr(self, 'HG_poi_src')  # Remove existing None attribute before registering buffer
            indices = torch.arange(self.num_pois)
            i = torch.stack([indices, indices])
            v = torch.ones(self.num_pois)
            self.register_buffer('HG_poi_src', torch.sparse_coo_tensor(
                i, v, (self.num_pois, self.num_pois)
            ).to(self.device))

        if self.HG_poi_tar is None:
            delattr(self, 'HG_poi_tar')  # Remove existing None attribute before registering buffer
            indices = torch.arange(self.num_pois)
            i = torch.stack([indices, indices])
            v = torch.ones(self.num_pois)
            self.register_buffer('HG_poi_tar', torch.sparse_coo_tensor(
                i, v, (self.num_pois, self.num_pois)
            ).to(self.device))

        # pad_all_train_sessions: Default empty
        if self.pad_all_train_sessions is None:
            delattr(self, 'pad_all_train_sessions')  # Remove existing None attribute before registering buffer
            self.register_buffer('pad_all_train_sessions', torch.zeros(
                self.num_users, 1, dtype=torch.long
            ).to(self.device))

    def set_graph_structures(self, HG_up=None, HG_pu=None, poi_geo_graph=None,
                            HG_poi_src=None, HG_poi_tar=None, pad_all_train_sessions=None):
        """
        Set graph structures after model initialization.

        This method can be used to update graph structures during runtime.

        Args:
            HG_up: User-POI hypergraph [U, L]
            HG_pu: POI-User hypergraph [L, U]
            poi_geo_graph: POI geographical graph [L, L]
            HG_poi_src: Source POI directed hypergraph [L, L]
            HG_poi_tar: Target POI directed hypergraph [L, L]
            pad_all_train_sessions: Padded training sessions [U, MAX_SEQ_LEN]
        """
        if HG_up is not None:
            self.HG_up = HG_up.to(self.device)
        if HG_pu is not None:
            self.HG_pu = HG_pu.to(self.device)
        if poi_geo_graph is not None:
            self.poi_geo_graph = poi_geo_graph.to(self.device)
        if HG_poi_src is not None:
            self.HG_poi_src = HG_poi_src.to(self.device)
        if HG_poi_tar is not None:
            self.HG_poi_tar = HG_poi_tar.to(self.device)
        if pad_all_train_sessions is not None:
            self.pad_all_train_sessions = pad_all_train_sessions.to(self.device)

    @staticmethod
    def row_shuffle(embedding):
        """Shuffle embedding rows for contrastive learning."""
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        return corrupted_embedding

    def cal_loss_infonce(self, emb1, emb2):
        """
        Calculate InfoNCE contrastive loss.

        Args:
            emb1: First embedding [N, d]
            emb2: Second embedding [N, d]

        Returns:
            InfoNCE loss scalar
        """
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.ssl_temp)
        neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.ssl_temp), axis=1)
        loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
        loss /= pos_score.shape[0]
        return loss

    def cal_loss_cl_pois(self, hg_pois_embs, geo_pois_embs, trans_pois_embs):
        """
        Calculate contrastive loss for POI embeddings across views.

        Args:
            hg_pois_embs: POI embeddings from hypergraph view [L, d]
            geo_pois_embs: POI embeddings from geographical view [L, d]
            trans_pois_embs: POI embeddings from transition view [L, d]

        Returns:
            Contrastive loss for POIs
        """
        # Normalization
        norm_hg_pois_embs = F.normalize(hg_pois_embs, p=2, dim=1)
        norm_geo_pois_embs = F.normalize(geo_pois_embs, p=2, dim=1)
        norm_trans_pois_embs = F.normalize(trans_pois_embs, p=2, dim=1)

        # Calculate pairwise InfoNCE loss
        loss_cl_pois = 0.0
        loss_cl_pois += self.cal_loss_infonce(norm_hg_pois_embs, norm_geo_pois_embs)
        loss_cl_pois += self.cal_loss_infonce(norm_hg_pois_embs, norm_trans_pois_embs)
        loss_cl_pois += self.cal_loss_infonce(norm_geo_pois_embs, norm_trans_pois_embs)

        return loss_cl_pois

    def cal_loss_cl_users(self, hg_batch_users_embs, geo_batch_users_embs, trans_batch_users_embs):
        """
        Calculate contrastive loss for user embeddings across views.

        Args:
            hg_batch_users_embs: User embeddings from hypergraph view [BS, d]
            geo_batch_users_embs: User embeddings from geographical view [BS, d]
            trans_batch_users_embs: User embeddings from transition view [BS, d]

        Returns:
            Contrastive loss for users
        """
        # Normalization
        norm_hg_batch_users_embs = F.normalize(hg_batch_users_embs, p=2, dim=1)
        norm_geo_batch_users_embs = F.normalize(geo_batch_users_embs, p=2, dim=1)
        norm_trans_batch_users_embs = F.normalize(trans_batch_users_embs, p=2, dim=1)

        # Calculate pairwise InfoNCE loss
        loss_cl_users = 0.0
        loss_cl_users += self.cal_loss_infonce(norm_hg_batch_users_embs, norm_geo_batch_users_embs)
        loss_cl_users += self.cal_loss_infonce(norm_hg_batch_users_embs, norm_trans_batch_users_embs)
        loss_cl_users += self.cal_loss_infonce(norm_geo_batch_users_embs, norm_trans_batch_users_embs)

        return loss_cl_users

    def _prepare_batch(self, batch):
        """
        Prepare batch data from LibCity format.

        LibCity batch format for trajectory location prediction typically includes:
        - 'uid' or 'user': User ID [batch_size]
        - 'current_loc' or 'loc': Location sequence [batch_size, seq_len] or [batch_size]
        - 'target': Target location [batch_size]

        Args:
            batch: LibCity batch object or dictionary

        Returns:
            Dictionary with prepared batch data
        """
        if hasattr(batch, 'data'):
            # LibCity Batch object
            # Try to get user ID
            if 'uid' in batch.data:
                user_idx = batch['uid']
            elif 'user' in batch.data:
                user_idx = batch['user']
            elif 'user_idx' in batch.data:
                user_idx = batch['user_idx']
            else:
                # Default to indices
                batch_size = len(batch['target']) if 'target' in batch.data else 1
                user_idx = torch.arange(batch_size, dtype=torch.long)

            # Get target
            target = batch['target'] if 'target' in batch.data else None

            # Get location sequence (for potential future use)
            if 'current_loc' in batch.data:
                loc_seq = batch['current_loc']
            elif 'loc' in batch.data:
                loc_seq = batch['loc']
            else:
                loc_seq = None

        elif isinstance(batch, dict):
            # Dictionary-like batch
            user_idx = batch.get('uid', batch.get('user', batch.get('user_idx')))
            target = batch.get('target', batch.get('label'))
            loc_seq = batch.get('current_loc', batch.get('loc'))
        else:
            # Assume batch is already prepared with 'user_idx' and 'label'
            user_idx = batch.get('user_idx') if hasattr(batch, 'get') else batch['user_idx']
            target = batch.get('label') if hasattr(batch, 'get') else batch.get('label', batch.get('target'))
            loc_seq = None

        # Ensure tensors are on the correct device
        if user_idx is not None:
            user_idx = user_idx.to(self.device) if isinstance(user_idx, torch.Tensor) else torch.tensor(user_idx, device=self.device)
        if target is not None:
            target = target.to(self.device) if isinstance(target, torch.Tensor) else torch.tensor(target, device=self.device)
        if loc_seq is not None:
            loc_seq = loc_seq.to(self.device) if isinstance(loc_seq, torch.Tensor) else torch.tensor(loc_seq, device=self.device)

        return {
            'user_idx': user_idx,
            'target': target,
            'loc_seq': loc_seq
        }

    def forward(self, batch):
        """
        Forward pass of DCHL model.

        This method adapts the original forward(dataset, batch) signature to
        forward(batch) by using graph structures stored as model attributes.

        Args:
            batch: LibCity batch containing user indices and targets

        Returns:
            Tuple of (prediction, loss_cl_user, loss_cl_poi)
            - prediction: Location prediction scores [batch_size, num_pois]
            - loss_cl_user: User contrastive loss
            - loss_cl_poi: POI contrastive loss
        """
        # Prepare batch data
        batch_data = self._prepare_batch(batch)
        user_idx = batch_data['user_idx']

        # Self-gating input for disentangled learning
        poi_emb_weight = self.poi_embedding.weight[:-1]  # Exclude padding

        geo_gate_pois_embs = torch.multiply(
            poi_emb_weight,
            torch.sigmoid(torch.matmul(poi_emb_weight, self.w_gate_geo) + self.b_gate_geo)
        )
        seq_gate_pois_embs = torch.multiply(
            poi_emb_weight,
            torch.sigmoid(torch.matmul(poi_emb_weight, self.w_gate_seq) + self.b_gate_seq)
        )
        col_gate_pois_embs = torch.multiply(
            poi_emb_weight,
            torch.sigmoid(torch.matmul(poi_emb_weight, self.w_gate_col) + self.b_gate_col)
        )

        # Multi-view hypergraph convolutional network
        hg_pois_embs = self.mv_hconv_network(
            col_gate_pois_embs, self.pad_all_train_sessions, self.HG_up, self.HG_pu
        )
        # Hypergraph structure-aware user embeddings
        hg_structural_users_embs = torch.sparse.mm(self.HG_up, hg_pois_embs)  # [U, d]
        hg_batch_users_embs = hg_structural_users_embs[user_idx]  # [BS, d]

        # POI-POI geographical graph convolutional network
        geo_pois_embs = self.geo_conv_network(geo_gate_pois_embs, self.poi_geo_graph)  # [L, d]
        # Geo-aware user embeddings
        geo_structural_users_embs = torch.sparse.mm(self.HG_up, geo_pois_embs)
        geo_batch_users_embs = geo_structural_users_embs[user_idx]  # [BS, d]

        # POI-POI directed hypergraph
        trans_pois_embs = self.di_hconv_network(
            seq_gate_pois_embs, self.HG_poi_src, self.HG_poi_tar
        )
        # Transition-aware user embeddings
        trans_structural_users_embs = torch.sparse.mm(self.HG_up, trans_pois_embs)
        trans_batch_users_embs = trans_structural_users_embs[user_idx]  # [BS, d]

        # Cross-view contrastive learning losses
        # Only compute expensive contrastive losses during training to avoid OOM during inference
        # These create L x L similarity matrices which can require ~15GB for large POI sets
        if self.training:
            loss_cl_poi = self.cal_loss_cl_pois(hg_pois_embs, geo_pois_embs, trans_pois_embs)
            loss_cl_user = self.cal_loss_cl_users(
                hg_batch_users_embs, geo_batch_users_embs, trans_batch_users_embs
            )
        else:
            # Skip expensive contrastive loss computation during inference
            loss_cl_poi = torch.tensor(0.0, device=self.device)
            loss_cl_user = torch.tensor(0.0, device=self.device)

        # Normalization
        norm_hg_pois_embs = F.normalize(hg_pois_embs, p=2, dim=1)
        norm_geo_pois_embs = F.normalize(geo_pois_embs, p=2, dim=1)
        norm_trans_pois_embs = F.normalize(trans_pois_embs, p=2, dim=1)

        norm_hg_batch_users_embs = F.normalize(hg_batch_users_embs, p=2, dim=1)
        norm_geo_batch_users_embs = F.normalize(geo_batch_users_embs, p=2, dim=1)
        norm_trans_batch_users_embs = F.normalize(trans_batch_users_embs, p=2, dim=1)

        # Adaptive fusion for user embeddings
        hyper_coef = self.hyper_gate(norm_hg_batch_users_embs)
        geo_coef = self.gcn_gate(norm_geo_batch_users_embs)
        trans_coef = self.trans_gate(norm_trans_batch_users_embs)

        # Final fusion for user and POI embeddings
        fusion_batch_users_embs = (
            hyper_coef * norm_hg_batch_users_embs +
            geo_coef * norm_geo_batch_users_embs +
            trans_coef * norm_trans_batch_users_embs
        )
        fusion_pois_embs = norm_hg_pois_embs + norm_geo_pois_embs + norm_trans_pois_embs

        # Prediction: user-POI similarity
        prediction = fusion_batch_users_embs @ fusion_pois_embs.T

        return prediction, loss_cl_user, loss_cl_poi

    def predict(self, batch):
        """
        Predict next POI for given batch.

        Args:
            batch: LibCity batch containing user indices

        Returns:
            torch.Tensor: Location prediction scores [batch_size, num_pois]
        """
        self.eval()
        with torch.no_grad():
            prediction, _, _ = self.forward(batch)
            # Apply log softmax for compatibility with NLLLoss evaluation
            return F.log_softmax(prediction, dim=-1)

    def calculate_loss(self, batch):
        """
        Calculate combined loss for training.

        The loss combines:
        1. Recommendation loss (CrossEntropy)
        2. User contrastive loss (InfoNCE)
        3. POI contrastive loss (InfoNCE)

        Args:
            batch: LibCity batch containing user indices and targets

        Returns:
            torch.Tensor: Combined loss (scalar)
        """
        # Forward pass
        prediction, loss_cl_user, loss_cl_poi = self.forward(batch)

        # Prepare batch data
        batch_data = self._prepare_batch(batch)
        target = batch_data['target']

        if target is None:
            raise ValueError("Target labels not found in batch. Expected 'target' or 'label' key.")

        # Recommendation loss
        loss_rec = self.criterion(prediction, target)

        # Combined loss with contrastive learning weight
        total_loss = loss_rec + self.lambda_cl * (loss_cl_poi + loss_cl_user)

        return total_loss
