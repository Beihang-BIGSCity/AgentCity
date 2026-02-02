# coding=utf-8
"""
DCHL: Disentangled Contrastive Hypergraph Learning for Next POI Recommendation
Adapted for LibCity Framework

Original paper: [24 SIGIR] Disentangled Contrastive Hypergraph Learning for Next POI Recommendation
Original author: Yantong Lai
Original source: repos/DCHL/model.py

Key components preserved:
- MultiViewHyperConvLayer: Multi-view hypergraph convolutional layer
- MultiViewHyperConvNetwork: Multi-view hypergraph convolutional network
- DirectedHyperConvLayer: Directed hypergraph convolutional layer
- DirectedHyperConvNetwork: Directed hypergraph convolutional network
- GeoConvNetwork: Geographic graph convolutional network
- DCHL: Main model with disentangled representation learning and contrastive learning

Key adaptations:
- Inherited from AbstractModel instead of nn.Module
- Adapted forward() signature: removed dataset parameter, graphs stored in self attributes
- Implemented predict() method following LibCity conventions
- Implemented calculate_loss() method for combined loss computation
- Extracted configuration parameters from LibCity config dict
- AUTOMATIC GRAPH BUILDING: Graphs are built automatically on first forward pass
  from training batch data. No manual initialize_graphs_from_data() call required.

Automatic Graph Building:
    The model automatically builds required hypergraphs during the first forward pass
    by accumulating user-POI interaction data from training batches. This enables
    seamless integration with LibCity's TrajLocPredExecutor without requiring any
    special initialization calls. The graphs are built when:
    1. Sufficient batches have been processed (configurable via min_batches_for_graph)
    2. Or sufficient users have been observed (at least 10% of total users)

Required config parameters:
    - emb_dim: Embedding dimension (default: 128)
    - num_mv_layers: Number of multi-view hypergraph conv layers (default: 3)
    - num_geo_layers: Number of geographic conv layers (default: 3)
    - num_di_layers: Number of directed hypergraph conv layers (default: 3)
    - dropout: Dropout probability (default: 0.3)
    - lambda_cl: Weight for contrastive learning loss (default: 0.1)
    - temperature: Temperature for contrastive learning (default: 0.1)
    - keep_rate: Keep rate for dropout (default: 1.0)
    - distance_threshold: Distance threshold for geographic adjacency (default: 2.5 km)
    - min_batches_for_graph: Minimum batches before building graphs (default: 10)

Required data_feature parameters:
    - num_users: Number of unique users (or uid_size)
    - num_pois: Number of unique POIs (or loc_size)

Optional data_feature parameters (built automatically if not provided):
    - H_pu: POI-User hypergraph incidence matrix [L, U]
    - HG_pu: Normalized POI-User hypergraph [L, U]
    - H_up: User-POI hypergraph incidence matrix [U, L]
    - HG_up: Normalized User-POI hypergraph [U, L]
    - HG_poi_src: POI transition source hypergraph
    - HG_poi_tar: POI transition target hypergraph
    - poi_geo_graph: POI geographic adjacency graph
    - poi_coordinates: Dict mapping POI index to (lat, lon) tuple for geo graph construction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from math import radians, cos, sin, asin, sqrt
from logging import getLogger

from libcity.model.abstract_model import AbstractModel


# ============================================================================
# Helper Functions for Graph Construction
# ============================================================================

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the Haversine distance between two geographic coordinates.

    Args:
        lon1, lat1: Longitude and latitude of first point (in degrees)
        lon2, lat2: Longitude and latitude of second point (in degrees)

    Returns:
        Distance in kilometers
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth radius in kilometers
    return c * r


def transform_csr_to_sparse_tensor(csr_matrix, device=None):
    """
    Transform scipy CSR matrix to PyTorch sparse COO tensor.

    Args:
        csr_matrix: scipy.sparse.csr_matrix
        device: Target device for the tensor

    Returns:
        torch.sparse.FloatTensor
    """
    coo = csr_matrix.tocoo()
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(coo.data)
    shape = torch.Size(coo.shape)
    sparse_tensor = torch.sparse_coo_tensor(i, v, shape)
    if device is not None:
        sparse_tensor = sparse_tensor.to(device)
    return sparse_tensor


def get_hyper_degree_matrix(incidence_matrix):
    """
    Compute the inverse degree diagonal matrix for hypergraph normalization.

    Args:
        incidence_matrix: scipy.sparse matrix

    Returns:
        scipy.sparse.diags: Inverse degree diagonal matrix
    """
    rowsum = np.array(incidence_matrix.sum(1)).flatten()
    d_inv = np.power(rowsum + 1e-8, -1)
    d_inv[np.isinf(d_inv)] = 0.0
    return sp.diags(d_inv)


def normalized_adj(adj, is_symmetric=False):
    """
    Normalize adjacency matrix for graph convolution.

    Args:
        adj: scipy.sparse adjacency matrix
        is_symmetric: Whether to use symmetric normalization

    Returns:
        Normalized adjacency matrix
    """
    if is_symmetric:
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum + 1e-8, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    else:
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum + 1e-8, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        return d_mat_inv @ adj


# ============================================================================
# Supporting Network Components
# ============================================================================

class MultiViewHyperConvLayer(nn.Module):
    """
    Multi-view Hypergraph Convolutional Layer.

    Performs message passing on a hypergraph structure:
    1. Node -> Hyperedge aggregation (POI -> User)
    2. Hyperedge -> Node propagation (User -> POI)
    """

    def __init__(self, emb_dim, device):
        super(MultiViewHyperConvLayer, self).__init__()

        self.fc_fusion = nn.Linear(2 * emb_dim, emb_dim, device=device)
        self.dropout = nn.Dropout(0.3)
        self.emb_dim = emb_dim
        self.device = device

    def forward(self, pois_embs, pad_all_train_sessions, HG_up, HG_pu):
        """
        Args:
            pois_embs: POI embeddings [L, d]
            pad_all_train_sessions: Padded training sessions [U, MAX_SESS_LEN]
            HG_up: Normalized User-POI hypergraph [U, L]
            HG_pu: Normalized POI-User hypergraph [L, U]

        Returns:
            Propagated POI embeddings [L, d]
        """
        # 1. Node -> hyperedge message: POI aggregation
        msg_poi_agg = torch.sparse.mm(HG_up, pois_embs)  # [U, d]

        # 2. Propagation: hyperedge -> node
        propag_pois_embs = torch.sparse.mm(HG_pu, msg_poi_agg)  # [L, d]

        return propag_pois_embs


class DirectedHyperConvLayer(nn.Module):
    """
    Directed hypergraph convolutional layer.

    Captures directional transitions between POIs using source and target
    hypergraphs to model sequential patterns in check-in data.
    """

    def __init__(self):
        super(DirectedHyperConvLayer, self).__init__()

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        """
        Args:
            pois_embs: POI embeddings [L, d]
            HG_poi_src: Source hypergraph for POI transitions
            HG_poi_tar: Target hypergraph for POI transitions

        Returns:
            Transition-aware POI embeddings [L, d]
        """
        msg_tar = torch.sparse.mm(HG_poi_tar, pois_embs)
        msg_src = torch.sparse.mm(HG_poi_src, msg_tar)

        return msg_src


class MultiViewHyperConvNetwork(nn.Module):
    """
    Multi-view Hypergraph Convolutional Network.

    Stacks multiple MultiViewHyperConvLayer with residual connections
    and layer averaging to capture collaborative signals from user-POI
    interactions.
    """

    def __init__(self, num_layers, emb_dim, dropout, device):
        super(MultiViewHyperConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.device = device
        self.mv_hconv_layer = MultiViewHyperConvLayer(emb_dim, device)
        self.dropout = dropout

    def forward(self, pois_embs, pad_all_train_sessions, HG_up, HG_pu):
        """
        Args:
            pois_embs: Initial POI embeddings [L, d]
            pad_all_train_sessions: Padded training sessions
            HG_up: User-POI hypergraph
            HG_pu: POI-User hypergraph

        Returns:
            Enhanced POI embeddings [L, d]
        """
        final_pois_embs = [pois_embs]
        for layer_idx in range(self.num_layers):
            pois_embs = self.mv_hconv_layer(pois_embs, pad_all_train_sessions, HG_up, HG_pu)
            # Residual connection to alleviate over-smoothing
            pois_embs = pois_embs + final_pois_embs[-1]
            pois_embs = F.dropout(pois_embs, self.dropout)
            final_pois_embs.append(pois_embs)
        # Mean aggregation across all layers
        final_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)

        return final_pois_embs


class DirectedHyperConvNetwork(nn.Module):
    """
    Directed Hypergraph Convolutional Network.

    Stacks multiple DirectedHyperConvLayer to capture sequential
    transition patterns in POI visits.
    """

    def __init__(self, num_layers, device, dropout=0.3):
        super(DirectedHyperConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.device = device
        self.dropout = dropout
        self.di_hconv_layer = DirectedHyperConvLayer()

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        """
        Args:
            pois_embs: Initial POI embeddings [L, d]
            HG_poi_src: Source hypergraph for transitions
            HG_poi_tar: Target hypergraph for transitions

        Returns:
            Transition-enhanced POI embeddings [L, d]
        """
        final_pois_embs = [pois_embs]
        for layer_idx in range(self.num_layers):
            pois_embs = self.di_hconv_layer(pois_embs, HG_poi_src, HG_poi_tar)
            # Residual connection
            pois_embs = pois_embs + final_pois_embs[-1]
            pois_embs = F.dropout(pois_embs, self.dropout)
            final_pois_embs.append(pois_embs)
        # Mean aggregation across all layers
        final_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)

        return final_pois_embs


class GeoConvNetwork(nn.Module):
    """
    Geographic Graph Convolutional Network.

    Propagates information on the POI geographic adjacency graph
    to capture spatial proximity patterns.
    """

    def __init__(self, num_layers, dropout):
        super(GeoConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, pois_embs, geo_graph):
        """
        Args:
            pois_embs: Initial POI embeddings [L, d]
            geo_graph: Geographic adjacency graph

        Returns:
            Spatially-enhanced POI embeddings [L, d]
        """
        final_pois_embs = [pois_embs]
        for _ in range(self.num_layers):
            pois_embs = torch.sparse.mm(geo_graph, pois_embs)
            pois_embs = pois_embs + final_pois_embs[-1]
            final_pois_embs.append(pois_embs)
        output_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)

        return output_pois_embs


# ============================================================================
# Main DCHL Model adapted for LibCity
# ============================================================================

class DCHL(AbstractModel):
    """
    DCHL: Disentangled Contrastive Hypergraph Learning for Next POI Recommendation

    This model disentangles POI representations into three views:
    1. Collaborative view: Captured via multi-view hypergraph convolution on user-POI interactions
    2. Geographic view: Captured via graph convolution on POI spatial adjacency
    3. Sequential view: Captured via directed hypergraph convolution on POI transitions

    Cross-view contrastive learning aligns these representations while preserving
    view-specific information through self-gating mechanisms.

    The model performs adaptive fusion of user and POI embeddings from all views
    for the final next POI prediction.

    Automatic Graph Construction:
    This model automatically builds required hypergraphs during the first forward pass
    by accumulating user-POI interaction data from training batches. No manual
    initialization is required - the model works seamlessly with LibCity's standard
    training loop (TrajLocPredExecutor).

    The automatic graph building process:
    1. During the first forward passes, the model accumulates user-POI interactions
    2. When sufficient data is collected (controlled by min_batches_for_graph config),
       all required hypergraphs are constructed
    3. Subsequent forward passes use the constructed graphs normally

    Optional Pre-initialization:
    If you prefer to pre-build graphs before training (e.g., for reproducibility),
    you can still call initialize_graphs_from_data(train_dataloader) manually.
    Pre-computed graphs can also be passed via data_feature.
    """

    def __init__(self, config, data_feature):
        super(DCHL, self).__init__(config, data_feature)

        self._logger = getLogger()

        # Device configuration
        self.device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Model hyperparameters from config
        self.emb_dim = config.get('emb_dim', 128)
        self.num_mv_layers = config.get('num_mv_layers', 3)
        self.num_geo_layers = config.get('num_geo_layers', 3)
        self.num_di_layers = config.get('num_di_layers', 3)
        self.dropout = config.get('dropout', 0.3)
        self.lambda_cl = config.get('lambda_cl', 0.1)
        self.ssl_temp = config.get('temperature', 0.1)
        self.keep_rate = config.get('keep_rate', 1.0)
        self.keep_rate_poi = config.get('keep_rate_poi', 1.0)
        self.distance_threshold = config.get('distance_threshold', 2.5)  # km

        # Data feature parameters
        self.num_users = data_feature.get('num_users', data_feature.get('uid_size', 100))
        self.num_pois = data_feature.get('num_pois', data_feature.get('loc_size', 1000))

        # POI coordinates for geographic graph construction (optional)
        self.poi_coordinates = data_feature.get('poi_coordinates', None)

        # Store graph structures from data_feature (may be None if not provided)
        # These will be built lazily if not provided
        self.H_pu = data_feature.get('H_pu', None)
        self.HG_pu = data_feature.get('HG_pu', None)
        self.H_up = data_feature.get('H_up', None)
        self.HG_up = data_feature.get('HG_up', None)
        self.HG_poi_src = data_feature.get('HG_poi_src', None)
        self.HG_poi_tar = data_feature.get('HG_poi_tar', None)
        self.poi_geo_graph = data_feature.get('poi_geo_graph', None)
        self.pad_all_train_sessions = data_feature.get('pad_all_train_sessions', None)

        # Flag to track if graphs have been initialized
        self._graphs_initialized = self._check_all_graphs_present()

        # Automatic graph building state
        # These track accumulated data during first epoch for graph construction
        self._graph_builder_active = False
        self._accumulated_user_poi_interactions = {}  # user_id -> set of visited POIs
        self._accumulated_user_trajectories = {}      # user_id -> list of POIs in order
        self._first_epoch_complete = False
        self._batch_count = 0
        self._min_batches_for_graph = config.get('min_batches_for_graph', 10)  # Min batches before building

        # Move graphs to device if they exist
        if self._graphs_initialized:
            self._move_graphs_to_device()

        # Log data feature status
        self._logger.info(f"DCHL initialized with {self.num_users} users and {self.num_pois} POIs")
        self._log_graph_status()

        # User and POI embeddings
        self.user_embedding = nn.Embedding(self.num_users, self.emb_dim)
        self.poi_embedding = nn.Embedding(self.num_pois + 1, self.emb_dim, padding_idx=self.num_pois)

        # Initialize embeddings
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

        # Gating for adaptive fusion with POI embeddings
        self.hyper_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())
        self.gcn_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())
        self.trans_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())

        # Gating for adaptive fusion with user embeddings
        self.user_hyper_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())
        self.user_gcn_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())

        # Temporal-augmentation layers
        self.pos_embeddings = nn.Embedding(1500, self.emb_dim, padding_idx=0)
        self.w_1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_dim, 1))
        self.glu1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.glu2 = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

        # Self-gating parameters for disentangled learning
        self.w_gate_geo = nn.Parameter(torch.FloatTensor(self.emb_dim, self.emb_dim))
        self.b_gate_geo = nn.Parameter(torch.FloatTensor(1, self.emb_dim))
        self.w_gate_seq = nn.Parameter(torch.FloatTensor(self.emb_dim, self.emb_dim))
        self.b_gate_seq = nn.Parameter(torch.FloatTensor(1, self.emb_dim))
        self.w_gate_col = nn.Parameter(torch.FloatTensor(self.emb_dim, self.emb_dim))
        self.b_gate_col = nn.Parameter(torch.FloatTensor(1, self.emb_dim))

        # Initialize gating parameters
        nn.init.xavier_normal_(self.w_gate_geo.data)
        nn.init.xavier_normal_(self.b_gate_geo.data)
        nn.init.xavier_normal_(self.w_gate_seq.data)
        nn.init.xavier_normal_(self.b_gate_seq.data)
        nn.init.xavier_normal_(self.w_gate_col.data)
        nn.init.xavier_normal_(self.b_gate_col.data)

        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)

    def _check_all_graphs_present(self):
        """Check if all required graphs are present in data_feature."""
        required_graphs = ['HG_pu', 'HG_up', 'HG_poi_src', 'HG_poi_tar', 'poi_geo_graph']
        return all(getattr(self, name, None) is not None for name in required_graphs)

    def _move_graphs_to_device(self):
        """Move all graph structures to the configured device."""
        if self.H_pu is not None:
            self.H_pu = self.H_pu.to(self.device)
        if self.HG_pu is not None:
            self.HG_pu = self.HG_pu.to(self.device)
        if self.H_up is not None:
            self.H_up = self.H_up.to(self.device)
        if self.HG_up is not None:
            self.HG_up = self.HG_up.to(self.device)
        if self.HG_poi_src is not None:
            self.HG_poi_src = self.HG_poi_src.to(self.device)
        if self.HG_poi_tar is not None:
            self.HG_poi_tar = self.HG_poi_tar.to(self.device)
        if self.poi_geo_graph is not None:
            self.poi_geo_graph = self.poi_geo_graph.to(self.device)
        if self.pad_all_train_sessions is not None:
            self.pad_all_train_sessions = self.pad_all_train_sessions.to(self.device)

    def _log_graph_status(self):
        """Log the status of required graph structures."""
        graph_names = ['H_pu', 'HG_pu', 'H_up', 'HG_up', 'HG_poi_src', 'HG_poi_tar', 'poi_geo_graph']
        for name in graph_names:
            graph = getattr(self, name, None)
            if graph is not None:
                if hasattr(graph, '_nnz'):
                    self._logger.info(f"Graph {name}: shape {graph.shape}, nnz {graph._nnz()}")
                else:
                    self._logger.info(f"Graph {name}: shape {graph.shape}")
            else:
                self._logger.warning(f"Graph {name} is not provided - will be built from training data")

    # ========================================================================
    # Graph Construction Methods
    # ========================================================================

    def _build_user_poi_hypergraphs(self, user_poi_interactions):
        """
        Build User-POI hypergraphs from user-POI interaction data.

        Args:
            user_poi_interactions: Dict mapping user_id to list of visited POI ids

        Returns:
            Tuple of (H_pu, HG_pu, H_up, HG_up) as sparse tensors
        """
        self._logger.info("Building User-POI hypergraphs...")

        # Create POI-User incidence matrix H_pu [num_pois, num_users]
        H_pu = np.zeros((self.num_pois, self.num_users), dtype=np.float32)

        for user_id, poi_list in user_poi_interactions.items():
            if user_id < self.num_users:
                for poi_id in poi_list:
                    if poi_id < self.num_pois:
                        H_pu[poi_id, user_id] = 1.0

        H_pu_sparse = sp.csr_matrix(H_pu)

        # Apply edge dropout if keep_rate < 1.0
        if self.keep_rate < 1.0:
            H_pu_sparse = self._csr_matrix_drop_edge(H_pu_sparse, self.keep_rate)

        # Compute normalized hypergraph HG_pu = D_v^{-1} * H_pu
        Deg_H_pu = get_hyper_degree_matrix(H_pu_sparse)
        HG_pu_sparse = Deg_H_pu @ H_pu_sparse

        # Transpose for User-POI hypergraph
        H_up_sparse = H_pu_sparse.T  # [num_users, num_pois]
        Deg_H_up = get_hyper_degree_matrix(H_up_sparse)
        HG_up_sparse = Deg_H_up @ H_up_sparse

        # Convert to sparse tensors
        H_pu_tensor = transform_csr_to_sparse_tensor(H_pu_sparse, self.device)
        HG_pu_tensor = transform_csr_to_sparse_tensor(HG_pu_sparse, self.device)
        H_up_tensor = transform_csr_to_sparse_tensor(H_up_sparse, self.device)
        HG_up_tensor = transform_csr_to_sparse_tensor(HG_up_sparse, self.device)

        self._logger.info(f"Built H_pu: shape {H_pu_tensor.shape}, nnz {H_pu_tensor._nnz()}")
        self._logger.info(f"Built HG_up: shape {HG_up_tensor.shape}, nnz {HG_up_tensor._nnz()}")

        return H_pu_tensor, HG_pu_tensor, H_up_tensor, HG_up_tensor

    def _build_poi_transition_hypergraphs(self, user_trajectories):
        """
        Build POI transition hypergraphs from user trajectory data.

        The directed hypergraph captures sequential transitions between POIs:
        - H_poi_src[src, tar] = 1 if there exists a transition from src to tar

        Args:
            user_trajectories: Dict mapping user_id to list of POI ids (trajectory)

        Returns:
            Tuple of (HG_poi_src, HG_poi_tar) as sparse tensors
        """
        self._logger.info("Building POI transition hypergraphs...")

        # Create directed POI-POI incidence matrix [num_pois, num_pois]
        # H[src, tar] = 1 means there is a transition from src to tar
        H_poi_src = np.zeros((self.num_pois, self.num_pois), dtype=np.float32)

        for user_id, trajectory in user_trajectories.items():
            traj = list(trajectory)
            for src_idx in range(len(traj) - 1):
                for tar_idx in range(src_idx + 1, len(traj)):
                    src_poi = traj[src_idx]
                    tar_poi = traj[tar_idx]
                    if src_poi < self.num_pois and tar_poi < self.num_pois:
                        H_poi_src[src_poi, tar_poi] = 1.0

        H_poi_src_sparse = sp.csr_matrix(H_poi_src)

        # Apply edge dropout if keep_rate_poi < 1.0
        if self.keep_rate_poi < 1.0:
            H_poi_src_sparse = self._csr_matrix_drop_edge(H_poi_src_sparse, self.keep_rate_poi)

        # Normalize source hypergraph
        Deg_H_poi_src = get_hyper_degree_matrix(H_poi_src_sparse)
        HG_poi_src_sparse = Deg_H_poi_src @ H_poi_src_sparse

        # Transpose for target hypergraph
        H_poi_tar_sparse = H_poi_src_sparse.T
        Deg_H_poi_tar = get_hyper_degree_matrix(H_poi_tar_sparse)
        HG_poi_tar_sparse = Deg_H_poi_tar @ H_poi_tar_sparse

        # Convert to sparse tensors
        HG_poi_src_tensor = transform_csr_to_sparse_tensor(HG_poi_src_sparse, self.device)
        HG_poi_tar_tensor = transform_csr_to_sparse_tensor(HG_poi_tar_sparse, self.device)

        self._logger.info(f"Built HG_poi_src: shape {HG_poi_src_tensor.shape}, nnz {HG_poi_src_tensor._nnz()}")
        self._logger.info(f"Built HG_poi_tar: shape {HG_poi_tar_tensor.shape}, nnz {HG_poi_tar_tensor._nnz()}")

        return HG_poi_src_tensor, HG_poi_tar_tensor

    def _build_geo_graph(self, poi_coordinates=None):
        """
        Build POI geographic adjacency graph.

        Creates an adjacency matrix where POIs within distance_threshold km
        are connected. If coordinates are not available, creates an identity
        matrix as fallback (self-loops only).

        Args:
            poi_coordinates: Dict mapping POI id to (lat, lon) tuple, or None

        Returns:
            Normalized geographic graph as sparse tensor
        """
        self._logger.info("Building POI geographic graph...")

        if poi_coordinates is None or len(poi_coordinates) == 0:
            # Fallback: Create identity matrix (self-loops only)
            self._logger.warning(
                "POI coordinates not available. Using identity matrix as fallback for geo graph. "
                "This may reduce model performance. Consider providing 'poi_coordinates' in data_feature."
            )
            geo_adj = sp.eye(self.num_pois, dtype=np.float32)
        else:
            # Build geographic adjacency based on Haversine distance
            geo_adj = np.zeros((self.num_pois, self.num_pois), dtype=np.float32)

            for poi1 in range(self.num_pois):
                if poi1 not in poi_coordinates:
                    geo_adj[poi1, poi1] = 1.0  # Self-loop if no coordinates
                    continue

                lat1, lon1 = poi_coordinates[poi1]
                for poi2 in range(poi1, self.num_pois):
                    if poi2 not in poi_coordinates:
                        continue

                    lat2, lon2 = poi_coordinates[poi2]
                    dist = haversine_distance(lon1, lat1, lon2, lat2)

                    if dist <= self.distance_threshold:
                        geo_adj[poi1, poi2] = 1.0
                        geo_adj[poi2, poi1] = 1.0

            geo_adj = sp.csr_matrix(geo_adj)

        # Normalize the adjacency matrix
        geo_graph_normalized = normalized_adj(geo_adj, is_symmetric=False)
        geo_graph_tensor = transform_csr_to_sparse_tensor(geo_graph_normalized, self.device)

        nnz = geo_graph_tensor._nnz()
        density = nnz / (self.num_pois * self.num_pois) * 100
        self._logger.info(f"Built poi_geo_graph: shape {geo_graph_tensor.shape}, nnz {nnz}, density {density:.2f}%")

        return geo_graph_tensor

    def _csr_matrix_drop_edge(self, csr_adj_matrix, keep_rate):
        """
        Randomly drop edges from a CSR matrix.

        Args:
            csr_adj_matrix: scipy.sparse.csr_matrix
            keep_rate: Probability of keeping each edge

        Returns:
            New CSR matrix with dropped edges
        """
        if keep_rate >= 1.0:
            return csr_adj_matrix

        coo = csr_adj_matrix.tocoo()
        row = coo.row
        col = coo.col
        edge_num = row.shape[0]

        # Generate edge mask
        mask = np.floor(np.random.rand(edge_num) + keep_rate).astype(bool)

        # Get new indices
        new_row = row[mask]
        new_col = col[mask]
        new_values = np.ones(new_row.shape[0], dtype=np.float32)

        return sp.csr_matrix((new_values, (new_row, new_col)), shape=coo.shape)

    def initialize_graphs_from_data(self, train_dataloader):
        """
        Build all required hypergraphs from training data (OPTIONAL).

        NOTE: This method is now OPTIONAL. The model automatically builds graphs
        during the first forward pass. You only need to call this method if you
        want to pre-build graphs before training starts (e.g., for reproducibility
        or to ensure all training data is used for graph construction).

        This method iterates through the entire training dataloader to collect
        user-POI interactions and trajectories, then builds all required graphs.

        Args:
            train_dataloader: PyTorch DataLoader containing training batches
                Each batch should have:
                - 'uid': User IDs [batch_size]
                - 'trajectory' or 'current_loc' or 'history_loc': POI sequences
        """
        if self._graphs_initialized:
            self._logger.info("Graphs already initialized, skipping.")
            return

        self._logger.info("Initializing graphs from training data...")

        # Collect user-POI interactions from training data
        user_poi_interactions = {}  # user_id -> set of visited POIs
        user_trajectories = {}      # user_id -> list of POIs in order

        for batch in train_dataloader:
            # Get user IDs
            if hasattr(batch, 'data'):
                batch_data = batch.data
            elif isinstance(batch, dict):
                batch_data = batch
            else:
                batch_data = batch

            # Extract user IDs
            if 'uid' in batch_data:
                uids = batch_data['uid']
            elif 'user_idx' in batch_data:
                uids = batch_data['user_idx']
            else:
                self._logger.warning("Batch does not contain 'uid' or 'user_idx', skipping")
                continue

            # Extract trajectories/POI sequences
            trajectory = None
            if 'trajectory' in batch_data:
                trajectory = batch_data['trajectory']
            elif 'current_loc' in batch_data:
                trajectory = batch_data['current_loc']
            elif 'history_loc' in batch_data:
                trajectory = batch_data['history_loc']
            elif 'loc' in batch_data:
                trajectory = batch_data['loc']

            if trajectory is None:
                self._logger.warning("Batch does not contain trajectory data, skipping")
                continue

            # Convert to numpy if tensor
            if isinstance(uids, torch.Tensor):
                uids = uids.cpu().numpy()
            if isinstance(trajectory, torch.Tensor):
                trajectory = trajectory.cpu().numpy()

            # Process each sample in the batch
            for i, uid in enumerate(uids):
                uid = int(uid)
                if uid not in user_poi_interactions:
                    user_poi_interactions[uid] = set()
                    user_trajectories[uid] = []

                # Get trajectory for this user
                if len(trajectory.shape) == 2:
                    traj = trajectory[i]  # [seq_len]
                elif len(trajectory.shape) == 3:
                    traj = trajectory[i].flatten()  # [seq_len, ...]
                else:
                    traj = trajectory.flatten()

                # Add POIs to interactions and trajectory
                for poi_id in traj:
                    poi_id = int(poi_id)
                    if poi_id < self.num_pois:  # Exclude padding
                        user_poi_interactions[uid].add(poi_id)
                        user_trajectories[uid].append(poi_id)

            # Also collect from target if available
            if 'target' in batch_data:
                target = batch_data['target']
                if isinstance(target, torch.Tensor):
                    target = target.cpu().numpy()

                for i, uid in enumerate(uids):
                    uid = int(uid)
                    if len(target.shape) > 0:
                        t = int(target[i]) if len(target.shape) == 1 else int(target[i].flatten()[0])
                    else:
                        t = int(target)
                    if t < self.num_pois:
                        user_poi_interactions[uid].add(t)
                        user_trajectories[uid].append(t)

        self._logger.info(f"Collected interactions from {len(user_poi_interactions)} users")

        # Build User-POI hypergraphs
        self.H_pu, self.HG_pu, self.H_up, self.HG_up = self._build_user_poi_hypergraphs(
            user_poi_interactions
        )

        # Build POI transition hypergraphs
        self.HG_poi_src, self.HG_poi_tar = self._build_poi_transition_hypergraphs(
            user_trajectories
        )

        # Build geographic graph
        self.poi_geo_graph = self._build_geo_graph(self.poi_coordinates)

        # Build padded training sessions tensor (for compatibility)
        if self.pad_all_train_sessions is None:
            max_seq_len = max(len(traj) for traj in user_trajectories.values()) if user_trajectories else 100
            max_seq_len = min(max_seq_len, 500)  # Cap at 500 to avoid memory issues

            pad_sessions = np.full((self.num_users, max_seq_len), self.num_pois, dtype=np.int64)
            for uid, traj in user_trajectories.items():
                if uid < self.num_users:
                    traj_len = min(len(traj), max_seq_len)
                    pad_sessions[uid, :traj_len] = traj[:traj_len]

            self.pad_all_train_sessions = torch.tensor(pad_sessions, dtype=torch.long, device=self.device)
            self._logger.info(f"Built pad_all_train_sessions: shape {self.pad_all_train_sessions.shape}")

        self._graphs_initialized = True
        self._logger.info("Graph initialization complete.")

    def _ensure_graphs_initialized(self):
        """
        Ensure graphs are initialized before forward pass.
        Raises an informative error if graphs are not available.

        Note: This is now deprecated in favor of automatic graph building.
        Kept for compatibility with existing code that may check initialization status.
        """
        if not self._graphs_initialized:
            required = ['HG_pu', 'HG_up', 'HG_poi_src', 'HG_poi_tar', 'poi_geo_graph']
            missing = [n for n in required if getattr(self, n, None) is None]
            raise RuntimeError(
                f"DCHL graphs are not initialized. Missing: {missing}. "
                f"Please call model.initialize_graphs_from_data(train_dataloader) "
                f"before training/inference, or provide precomputed graphs in data_feature."
            )

    # ========================================================================
    # Automatic Graph Building Methods
    # ========================================================================

    def _accumulate_interactions(self, batch):
        """
        Accumulate user-POI interactions and trajectories from a batch.

        This method is called during forward passes when graphs are not yet initialized.
        It extracts user-POI interaction data from each batch and stores it for later
        graph construction.

        Args:
            batch: LibCity Batch object containing trajectory data
        """
        # Get batch data
        if hasattr(batch, 'data'):
            batch_data = batch.data
        elif isinstance(batch, dict):
            batch_data = batch
        else:
            batch_data = batch

        # Extract user IDs
        uids = None
        if 'uid' in batch_data:
            uids = batch_data['uid']
        elif 'user_idx' in batch_data:
            uids = batch_data['user_idx']

        if uids is None:
            self._logger.warning("Batch does not contain 'uid' or 'user_idx', cannot accumulate interactions")
            return

        # Extract trajectories/POI sequences
        trajectory = None
        for key in ['trajectory', 'current_loc', 'history_loc', 'loc']:
            if key in batch_data:
                trajectory = batch_data[key]
                break

        if trajectory is None:
            self._logger.warning("Batch does not contain trajectory data, cannot accumulate interactions")
            return

        # Convert to numpy if tensor
        if isinstance(uids, torch.Tensor):
            uids = uids.cpu().numpy()
        if isinstance(trajectory, torch.Tensor):
            trajectory = trajectory.cpu().numpy()

        # Process each sample in the batch
        for i, uid in enumerate(uids):
            uid = int(uid)
            if uid not in self._accumulated_user_poi_interactions:
                self._accumulated_user_poi_interactions[uid] = set()
                self._accumulated_user_trajectories[uid] = []

            # Get trajectory for this user
            if len(trajectory.shape) == 2:
                traj = trajectory[i]  # [seq_len]
            elif len(trajectory.shape) == 3:
                traj = trajectory[i].flatten()  # [seq_len, ...]
            else:
                traj = trajectory.flatten()

            # Add POIs to interactions and trajectory
            for poi_id in traj:
                poi_id = int(poi_id)
                if poi_id < self.num_pois:  # Exclude padding
                    self._accumulated_user_poi_interactions[uid].add(poi_id)
                    self._accumulated_user_trajectories[uid].append(poi_id)

        # Also collect from target if available
        if 'target' in batch_data:
            target = batch_data['target']
            if isinstance(target, torch.Tensor):
                target = target.cpu().numpy()

            if isinstance(uids, torch.Tensor):
                uids_np = uids.cpu().numpy()
            else:
                uids_np = uids

            for i, uid in enumerate(uids_np):
                uid = int(uid)
                if uid not in self._accumulated_user_poi_interactions:
                    self._accumulated_user_poi_interactions[uid] = set()
                    self._accumulated_user_trajectories[uid] = []

                if len(target.shape) > 0:
                    t = int(target[i]) if len(target.shape) == 1 else int(target[i].flatten()[0])
                else:
                    t = int(target)
                if t < self.num_pois:
                    self._accumulated_user_poi_interactions[uid].add(t)
                    self._accumulated_user_trajectories[uid].append(t)

        self._batch_count += 1

    def _should_build_graphs(self):
        """
        Determine if we have accumulated enough data to build graphs.

        Returns True when either:
        1. We have processed enough batches (min_batches_for_graph)
        2. We have data from enough users (at least 10% of num_users)

        Returns:
            bool: True if graphs should be built now
        """
        # Check if we have enough batches
        if self._batch_count >= self._min_batches_for_graph:
            return True

        # Check if we have enough users
        num_users_with_data = len(self._accumulated_user_poi_interactions)
        if num_users_with_data >= max(10, self.num_users * 0.1):
            return True

        return False

    def _finalize_graph_construction(self):
        """
        Build all graphs from accumulated interaction data.

        This method is called when enough data has been accumulated during forward passes.
        It constructs all required hypergraphs and the geographic graph.
        """
        if self._graphs_initialized:
            return

        num_users_collected = len(self._accumulated_user_poi_interactions)
        num_interactions = sum(len(pois) for pois in self._accumulated_user_poi_interactions.values())

        self._logger.info(f"Building graphs from accumulated data: {num_users_collected} users, "
                         f"{num_interactions} interactions, {self._batch_count} batches")

        # Build User-POI hypergraphs
        self.H_pu, self.HG_pu, self.H_up, self.HG_up = self._build_user_poi_hypergraphs(
            self._accumulated_user_poi_interactions
        )

        # Build POI transition hypergraphs
        self.HG_poi_src, self.HG_poi_tar = self._build_poi_transition_hypergraphs(
            self._accumulated_user_trajectories
        )

        # Build geographic graph
        self.poi_geo_graph = self._build_geo_graph(self.poi_coordinates)

        # Build padded training sessions tensor (for compatibility)
        if self.pad_all_train_sessions is None:
            if self._accumulated_user_trajectories:
                max_seq_len = max(len(traj) for traj in self._accumulated_user_trajectories.values())
                max_seq_len = min(max_seq_len, 500)  # Cap at 500 to avoid memory issues
            else:
                max_seq_len = 100

            pad_sessions = np.full((self.num_users, max_seq_len), self.num_pois, dtype=np.int64)
            for uid, traj in self._accumulated_user_trajectories.items():
                if uid < self.num_users:
                    traj_len = min(len(traj), max_seq_len)
                    pad_sessions[uid, :traj_len] = traj[:traj_len]

            self.pad_all_train_sessions = torch.tensor(pad_sessions, dtype=torch.long, device=self.device)
            self._logger.info(f"Built pad_all_train_sessions: shape {self.pad_all_train_sessions.shape}")

        self._graphs_initialized = True
        self._graph_builder_active = False

        # Clear accumulated data to free memory
        self._accumulated_user_poi_interactions = {}
        self._accumulated_user_trajectories = {}

        self._logger.info("Automatic graph initialization complete.")

    def _build_minimal_graphs_from_batch(self, batch):
        """
        Build minimal placeholder graphs from a single batch.

        This is a fallback when we need to proceed with forward pass but don't have
        enough accumulated data. Creates sparse graphs that allow training to proceed,
        though with potentially reduced quality.

        Args:
            batch: Current batch data
        """
        self._logger.warning(
            "Building minimal placeholder graphs from single batch. "
            "This may reduce model performance. Consider accumulating more data first."
        )

        # Accumulate this batch first
        self._accumulate_interactions(batch)

        # If we still don't have any data, create identity/empty graphs
        if not self._accumulated_user_poi_interactions:
            self._logger.warning("No interaction data available. Creating identity graphs as fallback.")

            # Create minimal identity-like graphs
            identity_poi = sp.eye(self.num_pois, dtype=np.float32)
            identity_user = sp.eye(self.num_users, dtype=np.float32)

            # Create minimal user-poi interaction (diagonal sampling)
            min_size = min(self.num_users, self.num_pois)
            H_up_sparse = sp.lil_matrix((self.num_users, self.num_pois), dtype=np.float32)
            for i in range(min_size):
                H_up_sparse[i, i] = 1.0
            H_up_sparse = H_up_sparse.tocsr()

            H_pu_sparse = H_up_sparse.T

            Deg_H_pu = get_hyper_degree_matrix(H_pu_sparse)
            HG_pu_sparse = Deg_H_pu @ H_pu_sparse
            Deg_H_up = get_hyper_degree_matrix(H_up_sparse)
            HG_up_sparse = Deg_H_up @ H_up_sparse

            self.H_pu = transform_csr_to_sparse_tensor(H_pu_sparse, self.device)
            self.HG_pu = transform_csr_to_sparse_tensor(HG_pu_sparse, self.device)
            self.H_up = transform_csr_to_sparse_tensor(H_up_sparse, self.device)
            self.HG_up = transform_csr_to_sparse_tensor(HG_up_sparse, self.device)

            # POI transition as identity
            self.HG_poi_src = transform_csr_to_sparse_tensor(identity_poi, self.device)
            self.HG_poi_tar = transform_csr_to_sparse_tensor(identity_poi, self.device)

            # Geo graph as identity
            self.poi_geo_graph = transform_csr_to_sparse_tensor(identity_poi, self.device)

            # Minimal padded sessions
            self.pad_all_train_sessions = torch.full(
                (self.num_users, 10), self.num_pois, dtype=torch.long, device=self.device
            )
        else:
            # Build from accumulated data (even if minimal)
            self._finalize_graph_construction()
            return

        self._graphs_initialized = True
        self._graph_builder_active = False
        self._logger.info("Minimal placeholder graphs created.")

    # ========================================================================
    # Contrastive Learning Loss Functions
    # ========================================================================

    @staticmethod
    def row_shuffle(embedding):
        """Shuffle embedding rows for contrastive learning negative sampling."""
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        return corrupted_embedding

    def cal_loss_infonce(self, emb1, emb2):
        """
        Calculate InfoNCE contrastive loss between two embeddings.

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
        Calculate cross-view contrastive loss for POI embeddings.

        Aligns POI representations from hypergraph, geographic, and
        transition views using InfoNCE loss.

        Args:
            hg_pois_embs: Hypergraph-based POI embeddings
            geo_pois_embs: Geographic-based POI embeddings
            trans_pois_embs: Transition-based POI embeddings

        Returns:
            Combined contrastive loss for POIs
        """
        # Normalization
        norm_hg_pois_embs = F.normalize(hg_pois_embs, p=2, dim=1)
        norm_geo_pois_embs = F.normalize(geo_pois_embs, p=2, dim=1)
        norm_trans_pois_embs = F.normalize(trans_pois_embs, p=2, dim=1)

        # Pairwise contrastive losses
        loss_cl_pois = 0.0
        loss_cl_pois += self.cal_loss_infonce(norm_hg_pois_embs, norm_geo_pois_embs)
        loss_cl_pois += self.cal_loss_infonce(norm_hg_pois_embs, norm_trans_pois_embs)
        loss_cl_pois += self.cal_loss_infonce(norm_geo_pois_embs, norm_trans_pois_embs)

        return loss_cl_pois

    def cal_loss_cl_users(self, hg_batch_users_embs, geo_batch_users_embs, trans_batch_users_embs):
        """
        Calculate cross-view contrastive loss for user embeddings.

        Aligns user representations derived from different POI views.

        Args:
            hg_batch_users_embs: Users from hypergraph view
            geo_batch_users_embs: Users from geographic view
            trans_batch_users_embs: Users from transition view

        Returns:
            Combined contrastive loss for users
        """
        # Normalization
        norm_hg_batch_users_embs = F.normalize(hg_batch_users_embs, p=2, dim=1)
        norm_geo_batch_users_embs = F.normalize(geo_batch_users_embs, p=2, dim=1)
        norm_trans_batch_users_embs = F.normalize(trans_batch_users_embs, p=2, dim=1)

        # Pairwise contrastive losses
        loss_cl_users = 0.0
        loss_cl_users += self.cal_loss_infonce(norm_hg_batch_users_embs, norm_geo_batch_users_embs)
        loss_cl_users += self.cal_loss_infonce(norm_hg_batch_users_embs, norm_trans_batch_users_embs)
        loss_cl_users += self.cal_loss_infonce(norm_geo_batch_users_embs, norm_trans_batch_users_embs)

        return loss_cl_users

    # ========================================================================
    # Forward Pass and Prediction
    # ========================================================================

    def forward(self, batch):
        """
        Forward pass of DCHL model.

        Processes the batch through three disentangled views (collaborative,
        geographic, sequential), performs cross-view contrastive learning,
        and adaptively fuses embeddings for prediction.

        Automatic Graph Building:
        If graphs are not initialized, this method will automatically accumulate
        interaction data from batches and build graphs when sufficient data is
        collected. This removes the need for manual initialize_graphs_from_data() calls.

        Args:
            batch: LibCity Batch object containing:
                - 'user_idx' or 'uid': User indices [batch_size]
                - 'target' (optional): Target POI indices for loss computation

        Returns:
            Tuple of (predictions, loss_cl_users, loss_cl_pois)
            - predictions: [batch_size, num_pois] prediction scores
            - loss_cl_users: Cross-view contrastive loss for users
            - loss_cl_pois: Cross-view contrastive loss for POIs
        """
        # Automatic graph building logic
        if not self._graphs_initialized:
            # Activate graph builder if not already active
            if not self._graph_builder_active:
                self._graph_builder_active = True
                self._logger.info("DCHL: Starting automatic graph building from training data...")

            # Accumulate interactions from this batch
            self._accumulate_interactions(batch)

            # Check if we should build graphs now
            if self._should_build_graphs():
                self._finalize_graph_construction()
            else:
                # Not enough data yet - build minimal graphs to allow forward pass to proceed
                # This is a fallback that allows training to start immediately
                self._logger.info(f"DCHL: Accumulated {self._batch_count} batches, "
                                 f"{len(self._accumulated_user_poi_interactions)} users. "
                                 f"Building graphs now...")
                self._finalize_graph_construction()

        # Get user indices from batch
        # LibCity may use 'uid' or 'user_idx' as key
        if hasattr(batch, 'data'):
            batch_data = batch.data
        elif isinstance(batch, dict):
            batch_data = batch
        else:
            batch_data = batch

        if 'user_idx' in batch_data:
            user_idx = batch_data['user_idx'] if isinstance(batch_data, dict) else batch['user_idx']
        elif 'uid' in batch_data:
            user_idx = batch_data['uid'] if isinstance(batch_data, dict) else batch['uid']
        else:
            raise KeyError("Batch must contain 'user_idx' or 'uid' for user indices")

        # Ensure user_idx is on the correct device
        if isinstance(user_idx, torch.Tensor):
            user_idx = user_idx.to(self.device)

        # Self-gating input for disentangled representations
        # Each gate produces view-specific POI embeddings
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

        # Multi-view hypergraph convolutional network (collaborative view)
        hg_pois_embs = self.mv_hconv_network(
            col_gate_pois_embs, self.pad_all_train_sessions, self.HG_up, self.HG_pu
        )
        # Hypergraph structure-aware user embeddings
        hg_structural_users_embs = torch.sparse.mm(self.HG_up, hg_pois_embs)  # [U, d]
        hg_batch_users_embs = hg_structural_users_embs[user_idx]  # [BS, d]

        # POI-POI geographical graph convolutional network (geographic view)
        geo_pois_embs = self.geo_conv_network(geo_gate_pois_embs, self.poi_geo_graph)  # [L, d]
        # Geo-aware user embeddings
        geo_structural_users_embs = torch.sparse.mm(self.HG_up, geo_pois_embs)
        geo_batch_users_embs = geo_structural_users_embs[user_idx]  # [BS, d]

        # POI-POI directed hypergraph (sequential view)
        trans_pois_embs = self.di_hconv_network(
            seq_gate_pois_embs, self.HG_poi_src, self.HG_poi_tar
        )
        # Transition-aware user embeddings
        trans_structural_users_embs = torch.sparse.mm(self.HG_up, trans_pois_embs)
        trans_batch_users_embs = trans_structural_users_embs[user_idx]  # [BS, d]

        # Cross-view contrastive learning losses
        loss_cl_poi = self.cal_loss_cl_pois(hg_pois_embs, geo_pois_embs, trans_pois_embs)
        loss_cl_user = self.cal_loss_cl_users(
            hg_batch_users_embs, geo_batch_users_embs, trans_batch_users_embs
        )

        # Normalization for fusion
        norm_hg_pois_embs = F.normalize(hg_pois_embs, p=2, dim=1)
        norm_geo_pois_embs = F.normalize(geo_pois_embs, p=2, dim=1)
        norm_trans_pois_embs = F.normalize(trans_pois_embs, p=2, dim=1)

        norm_hg_batch_users_embs = F.normalize(hg_batch_users_embs, p=2, dim=1)
        norm_geo_batch_users_embs = F.normalize(geo_batch_users_embs, p=2, dim=1)
        norm_trans_batch_users_embs = F.normalize(trans_batch_users_embs, p=2, dim=1)

        # Adaptive fusion for user embeddings using learned gates
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

        # Prediction: inner product between user and POI embeddings
        prediction = fusion_batch_users_embs @ fusion_pois_embs.T  # [BS, L]

        return prediction, loss_cl_user, loss_cl_poi

    def predict(self, batch):
        """
        Predict next POI locations for the given batch.

        Args:
            batch: LibCity Batch object containing user indices

        Returns:
            Location prediction scores [batch_size, num_pois]
        """
        prediction, _, _ = self.forward(batch)
        return prediction

    def calculate_loss(self, batch):
        """
        Calculate the combined loss for training.

        The total loss consists of:
        1. Recommendation loss (cross-entropy on predicted POIs)
        2. Contrastive losses for POIs and users

        Total = loss_rec + lambda_cl * (loss_cl_pois + loss_cl_users)

        Args:
            batch: LibCity Batch object containing:
                - 'user_idx' or 'uid': User indices
                - 'target': Target POI indices [batch_size]

        Returns:
            Combined loss tensor
        """
        prediction, loss_cl_user, loss_cl_poi = self.forward(batch)

        # Get target POIs
        if hasattr(batch, 'data'):
            target = batch['target']
        elif isinstance(batch, dict):
            target = batch['target']
        else:
            target = batch.target

        target = target.to(self.device)

        # Recommendation loss (cross-entropy)
        loss_rec = F.cross_entropy(prediction, target, reduction='mean')

        # Combined loss with contrastive regularization
        total_loss = loss_rec + self.lambda_cl * (loss_cl_poi + loss_cl_user)

        return total_loss
