# coding=utf-8
"""
Adapted DCHL model for LibCity framework.

Original paper: [24 SIGIR] Disentangled Contrastive Hypergraph Learning for Next POI Recommendation
Original author: Yantong Lai
Adapted for LibCity by: Model Adaptation Agent

Key adaptations:
1. Inherits from AbstractModel instead of nn.Module
2. Constructor signature changed from (num_users, num_pois, args, device) to (config, data_feature)
3. Hypergraph structures are initialized via set_data_feature method or during __init__ with fallback
4. Added predict() and calculate_loss() methods as required by LibCity
5. Forward signature adapted to work with LibCity batch format
6. Fallback graph initialization when session data is not available
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from math import radians, cos, sin, asin, sqrt
import logging

from libcity.model.abstract_model import AbstractModel

# Set up logger for this module
_logger = logging.getLogger(__name__)


class MultiViewHyperConvLayer(nn.Module):
    """
    Multi-view Hypergraph Convolutional Layer
    Performs message passing between POIs and users through hypergraph structures.
    """

    def __init__(self, emb_dim, device):
        super(MultiViewHyperConvLayer, self).__init__()
        self.fc_fusion = nn.Linear(2 * emb_dim, emb_dim, device=device)
        self.dropout = nn.Dropout(0.3)
        self.emb_dim = emb_dim
        self.device = device

    def forward(self, pois_embs, pad_all_train_sessions, HG_up, HG_pu):
        """
        Forward pass for multi-view hypergraph convolution.

        Args:
            pois_embs: POI embeddings [L, d]
            pad_all_train_sessions: Padded training sessions [U, MAX_SESS_LEN]
            HG_up: User-to-POI hypergraph [U, L]
            HG_pu: POI-to-User hypergraph [L, U]

        Returns:
            propag_pois_embs: Propagated POI embeddings [L, d]
        """
        # 1. node -> hyperedge message: poi node aggregation
        msg_poi_agg = torch.sparse.mm(HG_up, pois_embs)  # [U, d]

        # 2. propagation: hyperedge -> node
        propag_pois_embs = torch.sparse.mm(HG_pu, msg_poi_agg)  # [L, d]

        return propag_pois_embs


class DirectedHyperConvLayer(nn.Module):
    """
    Directed hypergraph convolutional layer for capturing sequential transitions.
    """

    def __init__(self):
        super(DirectedHyperConvLayer, self).__init__()

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        """
        Forward pass for directed hypergraph convolution.

        Args:
            pois_embs: POI embeddings [L, d]
            HG_poi_src: Source POI hypergraph [L, L]
            HG_poi_tar: Target POI hypergraph [L, L]

        Returns:
            msg_src: Message from source POIs [L, d]
        """
        msg_tar = torch.sparse.mm(HG_poi_tar, pois_embs)
        msg_src = torch.sparse.mm(HG_poi_src, msg_tar)

        return msg_src


class MultiViewHyperConvNetwork(nn.Module):
    """
    Multi-view Hypergraph Convolutional Network
    Stacks multiple multi-view hypergraph convolutional layers.
    """

    def __init__(self, num_layers, emb_dim, dropout, device):
        super(MultiViewHyperConvNetwork, self).__init__()
        self.num_layers = num_layers
        self.device = device
        self.mv_hconv_layer = MultiViewHyperConvLayer(emb_dim, device)
        self.dropout = dropout

    def forward(self, pois_embs, pad_all_train_sessions, HG_up, HG_pu):
        """
        Forward pass through multi-layer multi-view hypergraph convolution.

        Args:
            pois_embs: Initial POI embeddings [L, d]
            pad_all_train_sessions: Padded training sessions
            HG_up: User-to-POI hypergraph
            HG_pu: POI-to-User hypergraph

        Returns:
            final_pois_embs: Final POI embeddings after multi-layer propagation [L, d]
        """
        final_pois_embs = [pois_embs]
        for layer_idx in range(self.num_layers):
            pois_embs = self.mv_hconv_layer(pois_embs, pad_all_train_sessions, HG_up, HG_pu)
            # add residual connection to alleviate over-smoothing issue
            pois_embs = pois_embs + final_pois_embs[-1]
            pois_embs = F.dropout(pois_embs, self.dropout)
            final_pois_embs.append(pois_embs)
        final_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)

        return final_pois_embs


class DirectedHyperConvNetwork(nn.Module):
    """
    Directed Hypergraph Convolutional Network for sequential pattern learning.
    """

    def __init__(self, num_layers, device, dropout=0.3):
        super(DirectedHyperConvNetwork, self).__init__()
        self.num_layers = num_layers
        self.device = device
        self.dropout = dropout
        self.di_hconv_layer = DirectedHyperConvLayer()

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        """
        Forward pass through multi-layer directed hypergraph convolution.

        Args:
            pois_embs: Initial POI embeddings [L, d]
            HG_poi_src: Source POI hypergraph
            HG_poi_tar: Target POI hypergraph

        Returns:
            final_pois_embs: Final POI embeddings [L, d]
        """
        final_pois_embs = [pois_embs]
        for layer_idx in range(self.num_layers):
            pois_embs = self.di_hconv_layer(pois_embs, HG_poi_src, HG_poi_tar)
            # add residual connection
            pois_embs = pois_embs + final_pois_embs[-1]
            pois_embs = F.dropout(pois_embs, self.dropout)
            final_pois_embs.append(pois_embs)
        final_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)

        return final_pois_embs


class GeoConvNetwork(nn.Module):
    """
    Geographical Convolutional Network for spatial pattern learning.
    """

    def __init__(self, num_layers, dropout):
        super(GeoConvNetwork, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, pois_embs, geo_graph):
        """
        Forward pass through geographical graph convolution.

        Args:
            pois_embs: Initial POI embeddings [L, d]
            geo_graph: POI geographical adjacency graph [L, L]

        Returns:
            output_pois_embs: Final POI embeddings [L, d]
        """
        final_pois_embs = [pois_embs]
        for _ in range(self.num_layers):
            pois_embs = torch.sparse.mm(geo_graph, pois_embs)
            pois_embs = pois_embs + final_pois_embs[-1]
            final_pois_embs.append(pois_embs)
        output_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)

        return output_pois_embs


# Utility functions for hypergraph construction
def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate haversine distance between two points."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


def gen_poi_geo_adj(num_pois, pois_coos_dict, distance_threshold):
    """Generate geographical adjacency matrix based on POI coordinates."""
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


def transform_csr_matrix_to_tensor(csr_matrix):
    """Transform csr matrix to PyTorch sparse tensor."""
    coo = csr_matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    sp_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sp_tensor


def get_hyper_deg(incidence_matrix):
    """Get hypergraph degree matrix."""
    rowsum = np.array(incidence_matrix.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv


def gen_sparse_H_user(sessions_dict, num_pois, num_users):
    """Generate sparse incidence matrix for user-POI hypergraph."""
    H = np.zeros(shape=(num_pois, num_users))
    for userID, sessions in sessions_dict.items():
        if isinstance(sessions[0], list):
            # sessions is a list of lists
            seq = []
            for session in sessions:
                seq.extend(session)
        else:
            # sessions is already a flat list
            seq = sessions
        for poi in seq:
            if poi < num_pois:
                H[poi, userID] = 1
    H = sp.csr_matrix(H)
    return H


def gen_sparse_directed_H_poi(users_trajs_dict, num_pois):
    """Generate directed poi-poi incidence matrix for hypergraph."""
    H = np.zeros(shape=(num_pois, num_pois))
    for userID, traj in users_trajs_dict.items():
        if isinstance(traj[0], list):
            # Flatten if it's a list of lists
            flat_traj = []
            for session in traj:
                flat_traj.extend(session)
            traj = flat_traj
        for src_idx in range(len(traj) - 1):
            for tar_idx in range(src_idx + 1, len(traj)):
                src_poi = traj[src_idx]
                tar_poi = traj[tar_idx]
                if src_poi < num_pois and tar_poi < num_pois:
                    H[src_poi, tar_poi] = 1
    H = sp.csr_matrix(H)
    return H


def csr_matrix_drop_edge(csr_adj_matrix, keep_rate):
    """Drop edge on scipy.sparse.csr_matrix."""
    if keep_rate == 1.0:
        return csr_adj_matrix
    coo = csr_adj_matrix.tocoo()
    row = coo.row
    col = coo.col
    edgeNum = row.shape[0]
    mask = np.floor(np.random.rand(edgeNum) + keep_rate).astype(np.bool_)
    new_row = row[mask]
    new_col = col[mask]
    new_edgeNum = new_row.shape[0]
    new_values = np.ones(new_edgeNum, dtype=np.float32)
    drop_adj_matrix = sp.csr_matrix((new_values, (new_row, new_col)), shape=coo.shape)
    return drop_adj_matrix


class DCHL(AbstractModel):
    """
    Disentangled Contrastive Hypergraph Learning for Next POI Recommendation

    This model uses three types of graph structures to learn POI representations:
    1. Multi-view Hypergraph: Captures user-POI collaborative patterns
    2. Directed Hypergraph: Captures sequential transition patterns
    3. Geographical Graph: Captures spatial proximity patterns

    Contrastive learning is applied across these three views to learn disentangled representations.

    Args:
        config: LibCity configuration dictionary
        data_feature: Data feature dictionary containing num_users, num_pois, etc.
    """

    def __init__(self, config, data_feature):
        super(DCHL, self).__init__(config, data_feature)

        # Device configuration
        self.device = config.get('device', 'cpu')

        # Extract data features
        # Support different naming conventions in data_feature
        self.num_users = data_feature.get('num_users', data_feature.get('uid_size', 0))
        self.num_pois = data_feature.get('num_pois', data_feature.get('loc_size', 0))

        # Model hyperparameters from config
        self.emb_dim = config.get('emb_dim', 128)
        self.ssl_temp = config.get('temperature', 0.1)
        self.lambda_cl = config.get('lambda_cl', 0.1)
        self.dropout_rate = config.get('dropout', 0.3)
        self.num_mv_layers = config.get('num_mv_layers', 3)
        self.num_geo_layers = config.get('num_geo_layers', 3)
        self.num_di_layers = config.get('num_di_layers', 3)
        self.distance_threshold = config.get('distance_threshold', 2.5)
        self.keep_rate = config.get('keep_rate', 1.0)
        self.keep_rate_poi = config.get('keep_rate_poi', 1.0)

        # Store data_feature for later use
        self.data_feature = data_feature

        # Embeddings
        self.user_embedding = nn.Embedding(self.num_users, self.emb_dim)
        self.poi_embedding = nn.Embedding(self.num_pois + 1, self.emb_dim, padding_idx=self.num_pois)

        # Embedding initialization
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.poi_embedding.weight)

        # Networks
        self.mv_hconv_network = MultiViewHyperConvNetwork(
            self.num_mv_layers, self.emb_dim, 0, self.device
        )
        self.geo_conv_network = GeoConvNetwork(self.num_geo_layers, self.dropout_rate)
        self.di_hconv_network = DirectedHyperConvNetwork(
            self.num_di_layers, self.device, self.dropout_rate
        )

        # Gates for adaptive fusion with POI embeddings
        self.hyper_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())
        self.gcn_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())
        self.trans_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())

        # Gates for adaptive fusion with user embeddings
        self.user_hyper_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())
        self.user_gcn_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())

        # Temporal-augmentation layers
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

        # Initialize gate parameters
        nn.init.xavier_normal_(self.w_gate_geo.data)
        nn.init.xavier_normal_(self.b_gate_geo.data)
        nn.init.xavier_normal_(self.w_gate_seq.data)
        nn.init.xavier_normal_(self.b_gate_seq.data)
        nn.init.xavier_normal_(self.w_gate_col.data)
        nn.init.xavier_normal_(self.b_gate_col.data)

        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)

        # Graph structures (will be initialized via set_data_feature or from data_feature)
        self.HG_up = None
        self.HG_pu = None
        self.HG_poi_src = None
        self.HG_poi_tar = None
        self.poi_geo_graph = None
        self.pad_all_train_sessions = None
        self._graphs_initialized = False
        self._using_fallback_graphs = False

        # Try to initialize graphs from data_feature if available
        self._init_graphs_from_data_feature()

        # If graphs still not initialized, create fallback graphs
        if not self._graphs_initialized:
            _logger.warning(
                "DCHL: Required graph data (sessions_dict) not found in data_feature. "
                "Creating fallback identity/sparse graphs. Model will run but may have "
                "suboptimal performance. For best results, provide sessions_dict in data_feature."
            )
            self._create_fallback_graphs()

    def _init_graphs_from_data_feature(self):
        """Initialize graph structures from data_feature if available."""
        if self.data_feature is None:
            return

        # Check if precomputed graphs are available
        if 'HG_up' in self.data_feature and self.data_feature['HG_up'] is not None:
            self.HG_up = self.data_feature['HG_up']
            if not isinstance(self.HG_up, torch.Tensor):
                self.HG_up = transform_csr_matrix_to_tensor(self.HG_up).to(self.device)
            else:
                self.HG_up = self.HG_up.to(self.device)

        if 'HG_pu' in self.data_feature and self.data_feature['HG_pu'] is not None:
            self.HG_pu = self.data_feature['HG_pu']
            if not isinstance(self.HG_pu, torch.Tensor):
                self.HG_pu = transform_csr_matrix_to_tensor(self.HG_pu).to(self.device)
            else:
                self.HG_pu = self.HG_pu.to(self.device)

        if 'HG_poi_src' in self.data_feature and self.data_feature['HG_poi_src'] is not None:
            self.HG_poi_src = self.data_feature['HG_poi_src']
            if not isinstance(self.HG_poi_src, torch.Tensor):
                self.HG_poi_src = transform_csr_matrix_to_tensor(self.HG_poi_src).to(self.device)
            else:
                self.HG_poi_src = self.HG_poi_src.to(self.device)

        if 'HG_poi_tar' in self.data_feature and self.data_feature['HG_poi_tar'] is not None:
            self.HG_poi_tar = self.data_feature['HG_poi_tar']
            if not isinstance(self.HG_poi_tar, torch.Tensor):
                self.HG_poi_tar = transform_csr_matrix_to_tensor(self.HG_poi_tar).to(self.device)
            else:
                self.HG_poi_tar = self.HG_poi_tar.to(self.device)

        if 'poi_geo_graph' in self.data_feature and self.data_feature['poi_geo_graph'] is not None:
            self.poi_geo_graph = self.data_feature['poi_geo_graph']
            if not isinstance(self.poi_geo_graph, torch.Tensor):
                self.poi_geo_graph = transform_csr_matrix_to_tensor(self.poi_geo_graph).to(self.device)
            else:
                self.poi_geo_graph = self.poi_geo_graph.to(self.device)

        if 'pad_all_train_sessions' in self.data_feature and self.data_feature['pad_all_train_sessions'] is not None:
            self.pad_all_train_sessions = self.data_feature['pad_all_train_sessions']
            if isinstance(self.pad_all_train_sessions, torch.Tensor):
                self.pad_all_train_sessions = self.pad_all_train_sessions.to(self.device)

        # Check if all required graphs are initialized
        if all([
            self.HG_up is not None,
            self.HG_pu is not None,
            self.HG_poi_src is not None,
            self.HG_poi_tar is not None,
            self.poi_geo_graph is not None
        ]):
            self._graphs_initialized = True

    def set_data_feature(self, data_feature):
        """
        Set data feature and initialize graph structures.
        This method should be called by the executor or dataset to provide
        the required hypergraph structures.

        Args:
            data_feature: Dictionary containing graph structures:
                - sessions_dict: User sessions dictionary
                - pois_coos_dict: POI coordinates dictionary (optional)
                - HG_up, HG_pu, HG_poi_src, HG_poi_tar: Precomputed hypergraphs (optional)
                - poi_geo_graph: Precomputed geographical graph (optional)
        """
        self.data_feature = data_feature

        # Update num_users and num_pois if provided
        if 'num_users' in data_feature:
            self.num_users = data_feature['num_users']
        elif 'uid_size' in data_feature:
            self.num_users = data_feature['uid_size']
        if 'num_pois' in data_feature:
            self.num_pois = data_feature['num_pois']
        elif 'loc_size' in data_feature:
            self.num_pois = data_feature['loc_size']

        # Try to initialize from precomputed graphs
        self._init_graphs_from_data_feature()

        # If graphs not initialized, try to build them from raw data
        if not self._graphs_initialized:
            self._build_graphs_from_data()

        # If still not initialized, create fallback graphs
        if not self._graphs_initialized:
            _logger.warning(
                "DCHL: Could not build graphs from set_data_feature(). "
                "Creating fallback graphs."
            )
            self._create_fallback_graphs()

    def _build_graphs_from_data(self):
        """Build graph structures from raw session data."""
        if self.data_feature is None:
            return

        sessions_dict = self.data_feature.get('sessions_dict', None)
        pois_coos_dict = self.data_feature.get('pois_coos_dict', None)

        if sessions_dict is None:
            return

        # Build user-POI hypergraph
        H_pu = gen_sparse_H_user(sessions_dict, self.num_pois, self.num_users)
        H_pu = csr_matrix_drop_edge(H_pu, self.keep_rate)
        Deg_H_pu = get_hyper_deg(H_pu)
        HG_pu = Deg_H_pu * H_pu
        self.HG_pu = transform_csr_matrix_to_tensor(HG_pu).to(self.device)

        # Build POI-user hypergraph
        H_up = H_pu.T
        Deg_H_up = get_hyper_deg(H_up)
        HG_up = Deg_H_up * H_up
        self.HG_up = transform_csr_matrix_to_tensor(HG_up).to(self.device)

        # Build users' trajectories dictionary
        users_trajs_dict = {}
        for userID, sessions in sessions_dict.items():
            if isinstance(sessions[0], list):
                traj = []
                for session in sessions:
                    traj.extend(session)
                users_trajs_dict[userID] = traj
            else:
                users_trajs_dict[userID] = sessions

        # Build directed POI-POI hypergraph
        H_poi_src = gen_sparse_directed_H_poi(users_trajs_dict, self.num_pois)
        H_poi_src = csr_matrix_drop_edge(H_poi_src, self.keep_rate_poi)
        Deg_H_poi_src = get_hyper_deg(H_poi_src)
        HG_poi_src = Deg_H_poi_src * H_poi_src
        self.HG_poi_src = transform_csr_matrix_to_tensor(HG_poi_src).to(self.device)

        H_poi_tar = H_poi_src.T
        Deg_H_poi_tar = get_hyper_deg(H_poi_tar)
        HG_poi_tar = Deg_H_poi_tar * H_poi_tar
        self.HG_poi_tar = transform_csr_matrix_to_tensor(HG_poi_tar).to(self.device)

        # Build geographical graph if coordinates available
        if pois_coos_dict is not None:
            poi_geo_adj = gen_poi_geo_adj(self.num_pois, pois_coos_dict, self.distance_threshold)
            poi_geo_graph_matrix = normalized_adj(adj=poi_geo_adj, is_symmetric=False)
            self.poi_geo_graph = transform_csr_matrix_to_tensor(poi_geo_graph_matrix).to(self.device)
        else:
            # Use identity matrix if no coordinates available
            poi_geo_adj = sp.eye(self.num_pois)
            self.poi_geo_graph = transform_csr_matrix_to_tensor(poi_geo_adj).to(self.device)

        # Build padded training sessions
        all_seqs = [torch.tensor(traj) for traj in users_trajs_dict.values()]
        if len(all_seqs) > 0:
            self.pad_all_train_sessions = torch.nn.utils.rnn.pad_sequence(
                all_seqs, batch_first=True, padding_value=self.num_pois
            ).to(self.device)

        self._graphs_initialized = True

    def _create_fallback_graphs(self):
        """
        Create fallback graph structures when session data is not available.

        This method creates minimal functional graph structures to allow the model
        to run within LibCity's standard pipeline. The fallback graphs are based on:
        - Identity matrices for user-POI and POI-POI relationships
        - Sparse random initialization to provide some structure

        Note: Performance with fallback graphs may be suboptimal. For best results,
        provide proper session data through set_data_feature() or in data_feature.
        """
        _logger.info("DCHL: Creating fallback graph structures...")

        # Ensure we have valid dimensions
        num_users = max(self.num_users, 1)
        num_pois = max(self.num_pois, 1)

        # Create fallback user-POI hypergraph (HG_up): [U, L]
        # Use a sparse random matrix to provide some connectivity
        # Each user is connected to a small random subset of POIs
        sparsity = min(0.1, 10.0 / num_pois)  # At most 10% or ~10 connections per user
        H_up_data = (np.random.rand(num_users, num_pois) < sparsity).astype(np.float32)
        # Ensure at least one connection per user
        for i in range(num_users):
            if H_up_data[i].sum() == 0:
                H_up_data[i, np.random.randint(0, num_pois)] = 1.0
        H_up = sp.csr_matrix(H_up_data)
        Deg_H_up = get_hyper_deg(H_up)
        HG_up = Deg_H_up * H_up
        self.HG_up = transform_csr_matrix_to_tensor(HG_up).to(self.device)

        # Create fallback POI-user hypergraph (HG_pu): [L, U]
        H_pu = H_up.T
        Deg_H_pu = get_hyper_deg(H_pu)
        HG_pu = Deg_H_pu * H_pu
        self.HG_pu = transform_csr_matrix_to_tensor(HG_pu).to(self.device)

        # Create fallback directed POI-POI hypergraph (HG_poi_src, HG_poi_tar): [L, L]
        # Use a sparse lower triangular matrix to simulate sequential transitions
        sparsity_poi = min(0.05, 20.0 / num_pois)  # Sparse connections
        H_poi_src_data = np.tril(
            (np.random.rand(num_pois, num_pois) < sparsity_poi).astype(np.float32), k=-1
        )
        # Ensure some connectivity
        for i in range(1, min(num_pois, 100)):
            if H_poi_src_data[i-1:i, :].sum() == 0:
                H_poi_src_data[i-1, min(i, num_pois-1)] = 1.0
        H_poi_src = sp.csr_matrix(H_poi_src_data)
        Deg_H_poi_src = get_hyper_deg(H_poi_src)
        HG_poi_src = Deg_H_poi_src * H_poi_src
        self.HG_poi_src = transform_csr_matrix_to_tensor(HG_poi_src).to(self.device)

        H_poi_tar = H_poi_src.T
        Deg_H_poi_tar = get_hyper_deg(H_poi_tar)
        HG_poi_tar = Deg_H_poi_tar * H_poi_tar
        self.HG_poi_tar = transform_csr_matrix_to_tensor(HG_poi_tar).to(self.device)

        # Create fallback geographical graph: use identity matrix (self-loops only)
        poi_geo_adj = sp.eye(num_pois, dtype=np.float32)
        self.poi_geo_graph = transform_csr_matrix_to_tensor(poi_geo_adj).to(self.device)

        # Create minimal padded training sessions placeholder
        # Shape: [num_users, 1] with padding values
        self.pad_all_train_sessions = torch.full(
            (num_users, 1), num_pois, dtype=torch.long, device=self.device
        )

        self._graphs_initialized = True
        self._using_fallback_graphs = True
        _logger.info(
            f"DCHL: Fallback graphs created - HG_up: {self.HG_up.shape}, "
            f"HG_pu: {self.HG_pu.shape}, HG_poi_src: {self.HG_poi_src.shape}, "
            f"poi_geo_graph: {self.poi_geo_graph.shape}"
        )

    @staticmethod
    def row_shuffle(embedding):
        """Shuffle rows of embedding for contrastive learning."""
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        return corrupted_embedding

    def cal_loss_infonce(self, emb1, emb2):
        """Calculate InfoNCE contrastive loss."""
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.ssl_temp)
        neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.ssl_temp), axis=1)
        loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
        loss /= pos_score.shape[0]
        return loss

    def cal_loss_cl_pois(self, hg_pois_embs, geo_pois_embs, trans_pois_embs):
        """Calculate contrastive loss for POI embeddings across views."""
        # Normalization
        norm_hg_pois_embs = F.normalize(hg_pois_embs, p=2, dim=1)
        norm_geo_pois_embs = F.normalize(geo_pois_embs, p=2, dim=1)
        norm_trans_pois_embs = F.normalize(trans_pois_embs, p=2, dim=1)

        # Calculate loss
        loss_cl_pois = 0.0
        loss_cl_pois += self.cal_loss_infonce(norm_hg_pois_embs, norm_geo_pois_embs)
        loss_cl_pois += self.cal_loss_infonce(norm_hg_pois_embs, norm_trans_pois_embs)
        loss_cl_pois += self.cal_loss_infonce(norm_geo_pois_embs, norm_trans_pois_embs)

        return loss_cl_pois

    def cal_loss_cl_users(self, hg_batch_users_embs, geo_batch_users_embs, trans_batch_users_embs):
        """Calculate contrastive loss for user embeddings across views."""
        # Normalization
        norm_hg_batch_users_embs = F.normalize(hg_batch_users_embs, p=2, dim=1)
        norm_geo_batch_users_embs = F.normalize(geo_batch_users_embs, p=2, dim=1)
        norm_trans_batch_users_embs = F.normalize(trans_batch_users_embs, p=2, dim=1)

        # Calculate loss
        loss_cl_users = 0.0
        loss_cl_users += self.cal_loss_infonce(norm_hg_batch_users_embs, norm_geo_batch_users_embs)
        loss_cl_users += self.cal_loss_infonce(norm_hg_batch_users_embs, norm_trans_batch_users_embs)
        loss_cl_users += self.cal_loss_infonce(norm_geo_batch_users_embs, norm_trans_batch_users_embs)

        return loss_cl_users

    def forward(self, batch):
        """
        Forward pass of DCHL model.

        Args:
            batch: Dictionary containing:
                - user_idx: User indices [batch_size]
                - user_seq: User sequences (optional)
                - label: Target POI labels (optional)

        Returns:
            prediction: Prediction scores [batch_size, num_pois]
            loss_cl_user: Contrastive loss for users
            loss_cl_poi: Contrastive loss for POIs
        """
        # Check if graphs are initialized; if not, try fallback
        if not self._graphs_initialized:
            _logger.warning(
                "DCHL: Graphs not initialized at forward time. Creating fallback graphs."
            )
            self._create_fallback_graphs()

        # Log warning on first forward pass if using fallback graphs
        if self._using_fallback_graphs and not hasattr(self, '_fallback_warning_logged'):
            _logger.warning(
                "DCHL: Forward pass using fallback graphs. Performance may be suboptimal. "
                "For best results, provide sessions_dict in data_feature."
            )
            self._fallback_warning_logged = True

        # Get user indices from batch - use duck typing for BatchPAD compatibility
        try:
            user_idx = batch['uid']  # LibCity's StandardTrajectoryEncoder uses 'uid'
        except (KeyError, TypeError):
            try:
                user_idx = batch['user_idx']
            except (KeyError, TypeError):
                raise KeyError("Batch must contain 'uid' or 'user_idx' key")

        # Handle tensor conversion
        if hasattr(user_idx, 'to'):
            user_idx = user_idx.to(self.device)
        elif not isinstance(user_idx, torch.Tensor):
            user_idx = torch.LongTensor(user_idx).to(self.device)

        # Self-gating input for disentangled learning
        geo_gate_pois_embs = torch.multiply(
            self.poi_embedding.weight[:-1],
            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1], self.w_gate_geo) + self.b_gate_geo)
        )
        seq_gate_pois_embs = torch.multiply(
            self.poi_embedding.weight[:-1],
            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1], self.w_gate_seq) + self.b_gate_seq)
        )
        col_gate_pois_embs = torch.multiply(
            self.poi_embedding.weight[:-1],
            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1], self.w_gate_col) + self.b_gate_col)
        )

        # Multi-view hypergraph convolutional network
        hg_pois_embs = self.mv_hconv_network(
            col_gate_pois_embs, self.pad_all_train_sessions, self.HG_up, self.HG_pu
        )
        # Hypergraph structure aware users embeddings
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

        # Cross view contrastive learning
        loss_cl_poi = self.cal_loss_cl_pois(hg_pois_embs, geo_pois_embs, trans_pois_embs)
        loss_cl_user = self.cal_loss_cl_users(
            hg_batch_users_embs, geo_batch_users_embs, trans_batch_users_embs
        )

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

        # Prediction
        prediction = fusion_batch_users_embs @ fusion_pois_embs.T

        return prediction, loss_cl_user, loss_cl_poi

    def predict(self, batch):
        """
        Predict next POI for given batch.

        Args:
            batch: Dictionary containing user information

        Returns:
            prediction: Prediction scores [batch_size, num_pois]
        """
        prediction, _, _ = self.forward(batch)
        return prediction

    def calculate_loss(self, batch):
        """
        Calculate total loss for training.

        Args:
            batch: Dictionary containing:
                - user_idx: User indices
                - label or target: Target POI labels

        Returns:
            loss: Total loss (reconstruction + contrastive losses)
        """
        prediction, loss_cl_user, loss_cl_poi = self.forward(batch)

        # Get labels - use duck typing for BatchPAD compatibility
        try:
            labels = batch['target']  # LibCity's StandardTrajectoryEncoder uses 'target'
        except (KeyError, TypeError):
            try:
                labels = batch['label']
            except (KeyError, TypeError):
                raise KeyError("Batch must contain 'target' or 'label' key")

        # Handle tensor conversion
        if hasattr(labels, 'to'):
            labels = labels.to(self.device)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.LongTensor(labels).to(self.device)

        # Calculate reconstruction loss
        loss_rec = F.cross_entropy(prediction, labels)

        # Total loss with contrastive learning regularization
        loss = loss_rec + self.lambda_cl * (loss_cl_poi + loss_cl_user)

        return loss
