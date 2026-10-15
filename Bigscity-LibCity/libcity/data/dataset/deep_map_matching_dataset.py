"""
DeepMapMatchingDataset for deep learning-based map matching models (GraphMM, etc.)

This dataset class extends MapMatchingDataset to provide:
1. Grid-based trajectory representations
2. Graph structures (road adjacency, trace graphs)
3. Proper train/valid/test DataLoader objects
4. Batch formatting compatible with GraphMM and similar models

Road-Centric Data Alignment (v2):
The key challenge is that LibCity stores GPS trajectories (e.g., 2503 points)
and road sequences (e.g., 179 segments) separately with different lengths.
GraphMM expects aligned trace-road pairs where len(gps) approximately equals len(roads).

This implementation uses road-centric trajectory processing:
1. Uses road sequence length as the primary reference for cutting
2. Proportionally samples GPS points to match road segments
3. Applies downsampling to GPS points (not roads)
4. Filters out invalid samples with empty road or trace sequences

Original preprocessing logic adapted from:
- repos/GraphMM/data_loader.py
- repos/GraphMM/graph_data.py
- repos/GraphMM/data_preprocess/data_process.py

Author: LibCity adaptation
Version: 2.0 (road-centric alignment fix)
"""

import os
import pickle
import math
import random
from logging import getLogger
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from torch_sparse import SparseTensor
    HAS_TORCH_SPARSE = True
except (ImportError, OSError):
    # OSError covers dlopen/ABI failures of prebuilt torch_sparse wheels
    # (missing libcudart, undefined torch symbols) on CPU-only installs.
    SparseTensor = None
    HAS_TORCH_SPARSE = False

import networkx as nx

from libcity.data.dataset.map_matching_dataset import MapMatchingDataset
from libcity.utils.utils import ensure_dir


# Constants for GPS to grid conversion
GRID_SIZE = 50  # meters
LAT_PER_METER = 8.993203677616966e-06
LNG_PER_METER = 1.1700193970443768e-05


def gps2grid(lat, lng, min_lat, min_lng, grid_size=GRID_SIZE):
    """
    Convert GPS coordinates to grid cell indices.

    Args:
        lat: Latitude
        lng: Longitude
        min_lat: Minimum latitude of the region
        min_lng: Minimum longitude of the region
        grid_size: Grid cell size in meters (default 50m)

    Returns:
        tuple: (grid_x, grid_y) grid cell indices
    """
    lat_unit = LAT_PER_METER * grid_size
    lng_unit = LNG_PER_METER * grid_size

    locgrid_x = int((lat - min_lat) / lat_unit) + 1
    locgrid_y = int((lng - min_lng) / lng_unit) + 1

    return locgrid_x, locgrid_y


def grid2gps(gridx1, gridy1, gridx2, gridy2, min_lat, min_lng, grid_size=GRID_SIZE):
    """
    Convert grid cell indices back to GPS coordinates (cell boundaries).

    Returns:
        tuple: (lat1, lng1, lat2, lng2) coordinates of grid cell boundaries
    """
    lat_unit = LAT_PER_METER * grid_size
    lng_unit = LNG_PER_METER * grid_size

    lat1 = (gridx1 - 1) * lat_unit + min_lat
    lng1 = (gridy1 - 1) * lng_unit + min_lng
    lat2 = gridx2 * lat_unit + min_lat
    lng2 = gridy2 * lng_unit + min_lng

    return lat1, lng1, lat2, lng2


def padding_collate_fn(batch):
    """
    Collate function for padding variable-length trajectories in a batch.

    Args:
        batch: List of (grid_traces, roads, traces_gps, sample_idx) tuples

    Returns:
        Padded tensors for batch processing

    Note: Filters out invalid samples (empty traces or roads) to prevent
    IndexError during model training.
    """
    # Filter out invalid samples with empty traces or roads
    valid_batch = [sample for sample in batch if len(sample[0]) > 0 and len(sample[1]) > 0]

    if len(valid_batch) == 0:
        # Return minimal valid batch to prevent crashes
        return {
            'grid_traces': torch.LongTensor([[1]]),
            'tgt_roads': torch.LongTensor([[0]]),
            'traces_gps': torch.FloatTensor([[[0.0, 0.0]]]),
            'sample_Idx': torch.LongTensor([[0]]),
            'traces_lens': [1],
            'road_lens': [1]
        }

    trace_lens = [len(sample[0]) for sample in valid_batch]
    road_lens = [len(sample[1]) for sample in valid_batch]
    max_tlen, max_rlen = max(trace_lens), max(road_lens)

    # Ensure minimum length of 1
    max_tlen = max(max_tlen, 1)
    max_rlen = max(max_rlen, 1)

    x, y, z, w = [], [], [], []
    # 0: [PAD] token for grid traces
    for sample in valid_batch:
        x.append(list(sample[0]) + [0] * (max_tlen - len(sample[0])))
        y.append(list(sample[1]) + [-1] * (max_rlen - len(sample[1])))
        z.append(list(sample[2]) + [[0.0, 0.0]] * (max_tlen - len(sample[2])))
        w.append(list(sample[3]) + [-1] * (max_tlen - len(sample[3])))

    return {
        'grid_traces': torch.LongTensor(x),
        'tgt_roads': torch.LongTensor(y),
        'traces_gps': torch.FloatTensor(z),
        'sample_Idx': torch.LongTensor(w),
        'traces_lens': trace_lens,
        'road_lens': road_lens
    }


class MapMatchingTorchDataset(Dataset):
    """
    PyTorch Dataset wrapper for map matching trajectories.

    Provides __getitem__ and __len__ for DataLoader compatibility.
    Includes validation to ensure no empty sequences.
    """

    def __init__(self, traces_ls, roads_ls, traces_gps_ls, sample_idx_ls):
        """
        Args:
            traces_ls: List of grid trace sequences
            roads_ls: List of target road sequences
            traces_gps_ls: List of GPS coordinate sequences
            sample_idx_ls: List of sample indices
        """
        # Filter out invalid samples during initialization
        valid_indices = [
            i for i in range(len(traces_ls))
            if len(traces_ls[i]) > 0 and len(roads_ls[i]) > 0
        ]

        self.traces_ls = [traces_ls[i] for i in valid_indices]
        self.roads_ls = [roads_ls[i] for i in valid_indices]
        self.traces_gps_ls = [traces_gps_ls[i] for i in valid_indices]
        self.sample_idx_ls = [sample_idx_ls[i] for i in valid_indices]
        self.length = len(self.traces_ls)

        if len(valid_indices) < len(traces_ls):
            from logging import getLogger
            logger = getLogger()
            logger.warning('Filtered out {} invalid samples with empty sequences'.format(
                len(traces_ls) - len(valid_indices)))

        assert len(self.traces_ls) == len(self.roads_ls) == len(self.traces_gps_ls) == len(self.sample_idx_ls)

    def __getitem__(self, index):
        return (
            self.traces_ls[index],
            self.roads_ls[index],
            self.traces_gps_ls[index],
            self.sample_idx_ls[index]
        )

    def __len__(self):
        return self.length


class DeepMapMatchingDataset(MapMatchingDataset):
    """
    Dataset class for deep learning-based map matching models.

    Extends MapMatchingDataset to provide:
    - Grid-based trajectory representations
    - Road network graph structures
    - Trace graph (grid connectivity based on trajectory flow)
    - Proper train/valid/test DataLoader objects
    - Batch formatting compatible with GraphMM

    Config Parameters:
        grid_size: Grid cell size in meters (default: 50)
        train_rate: Training data ratio (default: 0.7)
        eval_rate: Validation data ratio (default: 0.1)
        batch_size: Batch size for DataLoader (default: 32)
        downsample_rate: GPS trajectory downsampling rate (default: 0.5)
        max_road_len: Maximum road sequence length (default: 25)
        min_road_len: Minimum road sequence length (default: 15)
        layer: Number of layers for adjacency polynomial (default: 2)
        gamma: Gamma parameter for adjacency polynomial (default: 1.0)
        num_workers: Number of workers for DataLoader (default: 0)
    """

    def __init__(self, config):
        # Initialize parent class to load basic map matching data
        super().__init__(config)

        self._logger = getLogger()

        # Ensure parent data is loaded from cache if needed
        # When parent MapMatchingDataset loads from cache, it skips file loading
        # and rd_nwk, trajectory, route remain None. We need to load them from
        # the parent's cache file to build our data structures.
        if self.rd_nwk is None and self.cache_dataset:
            parent_cache_file = self.cache_file_name  # This is the parent's cache file
            if os.path.exists(parent_cache_file):
                with open(parent_cache_file, 'rb') as f:
                    parent_cache = pickle.load(f)
                self.rd_nwk = parent_cache.get('rd_nwk')
                self.trajectory = parent_cache.get('trajectory')
                self.route = parent_cache.get('route')
                self.multi_traj = parent_cache.get('multi_traj', False)
                self._logger.info('Loaded parent data from cache: {}'.format(parent_cache_file))

        # Configuration parameters
        self.grid_size = config.get('grid_size', GRID_SIZE)
        self.train_rate = config.get('train_rate', 0.7)
        self.eval_rate = config.get('eval_rate', 0.1)
        self.batch_size = config.get('batch_size', 32)
        self.eval_batch_size = config.get('eval_batch_size', self.batch_size)
        self.downsample_rate = config.get('downsample_rate', 0.5)
        self.max_road_len = config.get('max_road_len', 25)
        self.min_road_len = config.get('min_road_len', 15)
        self.layer = config.get('layer', 2)
        self.gamma = config.get('gamma', 1.0)
        self.num_workers = config.get('num_workers', 0)
        self.shuffle = config.get('shuffle', True)

        # Cache paths - version 2 uses road-centric alignment
        self.cache_file_folder = './libcity/cache/dataset_cache/'
        ensure_dir(self.cache_file_folder)
        self._cache_version = 'v3'  # v3 = road ID index mapping fix
        self.deep_mm_cache_file = os.path.join(
            self.cache_file_folder,
            'deep_map_matching_{}_{}_{}.pkl'.format(
                self.dataset,
                str(self.downsample_rate),
                self._cache_version
            )
        )

        # Initialize graph data structures
        self.num_roads = 0
        self.num_grids = 0
        self.road_graph = None
        self.trace_graph = None
        self.grid2traceid_dict = {}
        self.traceid2grid_dict = {}
        self.geo_id_to_idx = {}  # Mapping from geo_id to consecutive index
        self.road_adj = None
        self.road_x = None
        self.trace_weight = None
        self.trace_in_edge_index = None
        self.trace_out_edge_index = None
        self.map_matrix = None
        self.A_list = None
        self.singleton_grid_mask = None
        self.singleton_grid_location = None

        # Dataset splits
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

        # MBR (Minimum Bounding Rectangle) of the road network
        self.min_lat = None
        self.min_lng = None
        self.max_lat = None
        self.max_lng = None

        # Process data if not cached
        if self.cache_dataset and os.path.exists(self.deep_mm_cache_file):
            self._logger.info('Loading cached deep map matching data from {}'.format(
                self.deep_mm_cache_file))
            self._load_cache()
        else:
            self._logger.info('Building deep map matching data structures...')
            self._build_data_structures()
            if self.cache_dataset:
                self._save_cache()

    def _compute_mbr(self):
        """Compute the Minimum Bounding Rectangle from road network."""
        min_lat, min_lng = float('inf'), float('inf')
        max_lat, max_lng = float('-inf'), float('-inf')

        for u, v, data in self.rd_nwk.edges(data=True):
            # Get coordinates from nodes
            u_lat = self.rd_nwk.nodes[u].get('lat', 0)
            u_lng = self.rd_nwk.nodes[u].get('lon', 0)
            v_lat = self.rd_nwk.nodes[v].get('lat', 0)
            v_lng = self.rd_nwk.nodes[v].get('lon', 0)

            min_lat = min(min_lat, u_lat, v_lat)
            min_lng = min(min_lng, u_lng, v_lng)
            max_lat = max(max_lat, u_lat, v_lat)
            max_lng = max(max_lng, u_lng, v_lng)

        self.min_lat = min_lat
        self.min_lng = min_lng
        self.max_lat = max_lat
        self.max_lng = max_lng

        self._logger.info('MBR: lat [{:.6f}, {:.6f}], lng [{:.6f}, {:.6f}]'.format(
            min_lat, max_lat, min_lng, max_lng))

    def _build_road_graph(self):
        """
        Build road graph from the road network.

        Creates:
        - self.road_graph: NetworkX graph of roads
        - self.road_x: Road features tensor
        - self.road_adj: Sparse adjacency tensor
        - self.A_list: Adjacency polynomial for CRF
        """
        self._logger.info('Building road graph...')

        # Create a mapping from geo_id to consecutive indices
        edge_list = list(self.rd_nwk.edges(data=True))
        geo_ids = set()
        for u, v, data in edge_list:
            if 'geo_id' in data:
                geo_ids.add(data['geo_id'])

        # If no geo_ids, use edge indices
        if not geo_ids:
            geo_ids = set(range(len(edge_list)))

        geo_id_list = sorted(list(geo_ids))
        geo_id_to_idx = {gid: idx for idx, gid in enumerate(geo_id_list)}
        self.geo_id_to_idx = geo_id_to_idx  # Save mapping for trajectory processing
        self.num_roads = len(geo_id_list)

        self._logger.info('Number of roads: {}'.format(self.num_roads))

        # Build road graph
        self.road_graph = nx.Graph()

        # Add nodes for each road segment
        for u, v, data in edge_list:
            geo_id = data.get('geo_id', edge_list.index((u, v, data)))
            if geo_id not in geo_id_to_idx:
                continue

            idx = geo_id_to_idx[geo_id]

            # Get coordinates
            u_lat = self.rd_nwk.nodes[u].get('lat', 0)
            u_lng = self.rd_nwk.nodes[u].get('lon', 0)
            v_lat = self.rd_nwk.nodes[v].get('lat', 0)
            v_lng = self.rd_nwk.nodes[v].get('lon', 0)

            # Convert to grid
            grid_x1, grid_y1 = gps2grid(u_lat, u_lng, self.min_lat, self.min_lng, self.grid_size)
            grid_x2, grid_y2 = gps2grid(v_lat, v_lng, self.min_lat, self.min_lng, self.grid_size)

            self.road_graph.add_node(idx,
                x1=min(grid_x1, grid_x2),
                y1=min(grid_y1, grid_y2),
                x2=max(grid_x1, grid_x2),
                y2=max(grid_y1, grid_y2),
                lat1=u_lat, lng1=u_lng,
                lat2=v_lat, lng2=v_lng,
                geo_id=geo_id)

        # Add edges based on shared nodes in original road network
        node_to_roads = defaultdict(list)
        for u, v, data in edge_list:
            geo_id = data.get('geo_id', edge_list.index((u, v, data)))
            if geo_id in geo_id_to_idx:
                idx = geo_id_to_idx[geo_id]
                node_to_roads[u].append(idx)
                node_to_roads[v].append(idx)

        for node, roads in node_to_roads.items():
            for i in range(len(roads)):
                for j in range(i + 1, len(roads)):
                    if roads[i] != roads[j]:
                        self.road_graph.add_edge(roads[i], roads[j])

        # Build road features (x)
        x_feat = []
        for i in range(self.num_roads):
            if i not in self.road_graph.nodes:
                # Add placeholder for missing nodes
                x_feat.append([0] * 28)
                continue

            node_data = dict(self.road_graph.nodes[i])
            min_x = node_data.get('x1', 0)
            max_x = node_data.get('x2', 0)
            min_y = node_data.get('y1', 0)
            max_y = node_data.get('y2', 0)
            min_lat = node_data.get('lat1', 0)
            min_lng = node_data.get('lng1', 0)
            max_lat = node_data.get('lat2', 0)
            max_lng = node_data.get('lng2', 0)

            # Build feature vector similar to original
            tmp_feat = []
            self_grid = [(min_x, min_y), (max_x, max_y)]

            xy_ls = [
                (min_x, min_y), (max_x, max_y),
                ((min_x + max_x) // 2, min_y), ((min_x + max_x) // 2, max_y),
                (max_x, min_y), (min_x, max_y),
                (min_x, (min_y + max_y) // 2), (max_x, (min_y + max_y) // 2)
            ]

            for xy in xy_ls:
                x, y = xy
                min_dis = 1000
                for sx, sy in self_grid:
                    min_dis = min(min_dis, abs(sx - x) + abs(sy - y))
                tmp_feat.extend([x, y, min_dis])

            tmp_feat.extend([min_lat, min_lng, max_lat, max_lng])
            x_feat.append(tmp_feat)

        self.road_x = torch.tensor(x_feat, dtype=torch.float)

        # Build edge index
        edge_index = [[], []]
        for u, v in self.road_graph.edges():
            edge_index[0].extend([u, v])
            edge_index[1].extend([v, u])

        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # Build sparse adjacency
        if HAS_TORCH_SPARSE:
            self.road_adj = SparseTensor(
                row=edge_index[0],
                col=edge_index[1],
                sparse_sizes=(self.num_roads, self.num_roads)
            )
        else:
            # Fallback to dense adjacency
            self.road_adj = torch.zeros(self.num_roads, self.num_roads)
            for i in range(edge_index.shape[1]):
                self.road_adj[edge_index[0, i], edge_index[1, i]] = 1

        # Build adjacency polynomial for CRF
        A = torch.eye(self.num_roads)
        for u, v in self.road_graph.edges():
            A[u, v] = 1
            A[v, u] = 1

        self.A_list = self._get_adj_poly(A, self.layer, self.gamma)

        self._logger.info('Road graph built with {} nodes, {} edges'.format(
            self.road_graph.number_of_nodes(),
            self.road_graph.number_of_edges()))

    def _get_adj_poly(self, A, layer, gamma):
        """
        Compute adjacency polynomial A^k for CRF.

        Args:
            A: Adjacency matrix (with self-loops)
            layer: Number of hops (k)
            gamma: Penalty for non-adjacent pairs

        Returns:
            torch.Tensor: Adjacency polynomial matrix
        """
        ans = A.clone()
        for _ in range(layer - 1):
            ans = ans @ A
        ans[ans != 0] = 1.0
        ans[ans == 0] = -gamma
        return ans

    def _build_trace_graph(self):
        """
        Build trace graph from trajectories.

        The trace graph captures grid connectivity based on trajectory flow.

        Creates:
        - self.trace_graph: NetworkX DiGraph of grid cells
        - self.grid2traceid_dict: Mapping from (grid_x, grid_y) to trace node id
        - self.trace_weight: Edge weights tensor
        - self.trace_in_edge_index, self.trace_out_edge_index: Edge indices
        """
        self._logger.info('Building trace graph...')

        grid2id_dict = {}
        trace_dict = {}  # (from_id, to_id) -> count

        # Process all trajectories
        for usr_id, usr_trajs in self.trajectory.items():
            for traj_id, traj_points in usr_trajs.items():
                lst_id = -1
                for point in traj_points:
                    # point format: [dyna_id, lon, lat, ...] or [dyna_id, lon, lat]
                    if len(point) >= 3:
                        lon, lat = point[1], point[2]
                    else:
                        continue

                    gridx, gridy = gps2grid(lat, lon, self.min_lat, self.min_lng, self.grid_size)

                    if gridx < 0 or gridy < 0:
                        continue

                    if (gridx, gridy) not in grid2id_dict:
                        grid2id_dict[(gridx, gridy)] = len(grid2id_dict)

                    tmp_id = grid2id_dict[(gridx, gridy)]

                    if lst_id != -1 and lst_id != tmp_id:
                        key = (lst_id, tmp_id)
                        trace_dict[key] = trace_dict.get(key, 0) + 1

                    lst_id = tmp_id

        self.grid2traceid_dict = grid2id_dict
        self.traceid2grid_dict = {v: k for k, v in grid2id_dict.items()}
        self.num_grids = len(grid2id_dict)

        self._logger.info('Number of grid cells: {}'.format(self.num_grids))

        # Build trace graph
        self.trace_graph = nx.DiGraph()
        weighted_edges = [(k[0], k[1], v) for k, v in trace_dict.items()]
        self.trace_graph.add_weighted_edges_from(weighted_edges)

        # Add isolated nodes
        for grid, node_id in grid2id_dict.items():
            if node_id not in self.trace_graph.nodes():
                self.trace_graph.add_node(node_id)
            self.trace_graph.nodes[node_id]['gridx'] = grid[0]
            self.trace_graph.nodes[node_id]['gridy'] = grid[1]

        # Build PyTorch tensors for trace graph
        edge_index = [[], []]
        weight = []

        for u, v in self.trace_graph.edges():
            edge_index[0].append(u)
            edge_index[1].append(v)
            weight.append(self.trace_graph[u][v]['weight'])

        if len(weight) > 0:
            out_edge_index = torch.tensor(edge_index, dtype=torch.long)
            in_edge_index = out_edge_index[[1, 0]]
            weight = torch.tensor(weight, dtype=torch.float)
        else:
            # Handle case with no edges
            out_edge_index = torch.zeros((2, 0), dtype=torch.long)
            in_edge_index = torch.zeros((2, 0), dtype=torch.long)
            weight = torch.zeros(0, dtype=torch.float)

        self.trace_in_edge_index = in_edge_index
        self.trace_out_edge_index = out_edge_index
        self.trace_weight = weight

        # Build grid features (x)
        grid_x = []
        for i in range(self.num_grids):
            if i in self.traceid2grid_dict:
                gridx, gridy = self.traceid2grid_dict[i]
                coords = grid2gps(gridx, gridy, gridx, gridy,
                                  self.min_lat, self.min_lng, self.grid_size)
                grid_x.append(coords)
            else:
                grid_x.append((0, 0, 0, 0))

        self.trace_x = torch.tensor(grid_x, dtype=torch.float)

        self._logger.info('Trace graph built with {} nodes, {} edges'.format(
            self.trace_graph.number_of_nodes(),
            self.trace_graph.number_of_edges()))

    def _build_mapping_matrix(self):
        """
        Build mapping matrix from roads to grids.

        Creates:
        - self.map_matrix: (num_grids, num_roads) mapping matrix
        - self.singleton_grid_mask: Indices of grids with no road mapping
        - self.singleton_grid_location: Locations of singleton grids
        """
        self._logger.info('Building grid-road mapping matrix...')

        # Build grid2road and road2grid mappings
        grid2road_dict = {}
        road2grid_dict = {}

        for i in range(self.num_roads):
            if i not in self.road_graph.nodes:
                road2grid_dict[i] = []
                continue

            node_data = dict(self.road_graph.nodes[i])
            x1 = node_data.get('x1', 0)
            x2 = node_data.get('x2', 0)
            y1 = node_data.get('y1', 0)
            y2 = node_data.get('y2', 0)

            road2grid_dict[i] = []
            for gx in range(x1, x2 + 1):
                for gy in range(y1, y2 + 1):
                    if (gx, gy) in self.grid2traceid_dict:
                        road2grid_dict[i].append((gx, gy))
                        if (gx, gy) not in grid2road_dict:
                            grid2road_dict[(gx, gy)] = []
                        grid2road_dict[(gx, gy)].append(i)

        # Build traceid2road mapping
        traceid2road_dict = {}
        for grid, roads in grid2road_dict.items():
            if grid in self.grid2traceid_dict:
                traceid = self.grid2traceid_dict[grid]
                traceid2road_dict[traceid] = roads

        # Build mapping matrix
        self.map_matrix = torch.zeros(self.num_grids, self.num_roads)
        for traceid, roads in traceid2road_dict.items():
            for road in roads:
                self.map_matrix[traceid, road] = 1
            if len(roads) > 0:
                self.map_matrix[traceid] /= len(roads)

        # Find singleton grids (grids with no road mapping)
        singleton_mask = []
        singleton_location = []
        for i in range(self.num_grids):
            if i not in traceid2road_dict or len(traceid2road_dict[i]) == 0:
                singleton_mask.append(i)
                if i < len(self.trace_x):
                    singleton_location.append(self.trace_x[i])
                else:
                    singleton_location.append(torch.zeros(4))

        if len(singleton_location) > 0:
            self.singleton_grid_location = torch.stack(singleton_location, dim=0)
        else:
            self.singleton_grid_location = torch.zeros((0, 4))

        self.singleton_grid_mask = torch.tensor(singleton_mask, dtype=torch.long)

        self._logger.info('Mapping matrix built. Singleton grids: {}'.format(
            len(singleton_mask)))

    def _process_trajectories(self):
        """
        Process trajectories into format suitable for model training.

        This method implements road-centric trajectory processing:
        1. Uses road sequence as the primary reference (not GPS points)
        2. Cuts trajectories based on road segment count
        3. Samples GPS points proportionally to match road segments
        4. Creates proper alignment between GPS traces and roads

        The key insight is that LibCity stores GPS trajectories (e.g., 2503 points)
        and road sequences (e.g., 179 segments) separately. GraphMM expects aligned
        pairs where len(gps_points) approximately equals len(roads) after processing.
        """
        self._logger.info('Processing trajectories for training (road-centric approach)...')

        all_traces = []  # List of (traces_grid, roads, traces_gps, sample_idx)
        skipped_empty = 0
        skipped_short = 0

        for usr_id, usr_trajs in self.trajectory.items():
            for traj_id, traj_points in usr_trajs.items():
                # Get corresponding ground truth route
                if self.route is None:
                    continue
                if usr_id not in self.route or traj_id not in self.route[usr_id]:
                    continue

                route_data = self.route[usr_id][traj_id]

                # Get road sequence from route (this is the primary reference)
                # Map geo_id to consecutive index using self.geo_id_to_idx
                roads = []
                for route_point in route_data:
                    if len(route_point) >= 2:
                        road_id = route_point[1]
                        if isinstance(road_id, (int, np.integer)):
                            road_id = int(road_id)
                        elif isinstance(road_id, str) and road_id.isdigit():
                            road_id = int(road_id)
                        else:
                            continue
                        # Map geo_id to consecutive index
                        if road_id in self.geo_id_to_idx:
                            roads.append(self.geo_id_to_idx[road_id])

                if len(roads) < self.min_road_len:
                    skipped_short += 1
                    continue

                # Convert GPS points to grid
                all_traces_grid = []
                all_traces_gps = []
                all_original_idx = []

                for orig_idx, point in enumerate(traj_points):
                    if len(point) >= 3:
                        lon, lat = point[1], point[2]
                    else:
                        continue

                    gridx, gridy = gps2grid(lat, lon, self.min_lat, self.min_lng, self.grid_size)

                    if (gridx, gridy) in self.grid2traceid_dict:
                        # +1 to reserve 0 for padding
                        traceid = self.grid2traceid_dict[(gridx, gridy)] + 1
                        all_traces_grid.append(traceid)
                        all_traces_gps.append([lat, lon])
                        all_original_idx.append(orig_idx)

                if len(all_traces_grid) == 0:
                    skipped_empty += 1
                    continue

                # Cut trajectories based on ROAD count (road-centric cutting)
                traces_list = self._cut_trajectory_road_centric(
                    all_traces_grid, all_traces_gps, all_original_idx, roads)

                all_traces.extend(traces_list)

        self._logger.info('Processing complete: {} samples created'.format(len(all_traces)))
        self._logger.info('Skipped: {} empty GPS, {} short road sequences'.format(
            skipped_empty, skipped_short))

        if len(all_traces) == 0:
            self._logger.warning('No valid trajectory samples found! Creating dummy data.')
            # Create minimal dummy data to avoid errors
            dummy_trace = [1, 2, 3]
            dummy_roads = [0, 1, 2]
            dummy_gps = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            dummy_idx = [0, 1, 2]
            all_traces = [(dummy_trace, dummy_roads, dummy_gps, dummy_idx)] * 10

        # Shuffle and split
        random.shuffle(all_traces)
        n_samples = len(all_traces)
        n_train = int(n_samples * self.train_rate)
        n_eval = int(n_samples * self.eval_rate)

        train_data = all_traces[:n_train]
        eval_data = all_traces[n_train:n_train + n_eval]
        test_data = all_traces[n_train + n_eval:]

        # Create datasets
        self.train_dataset = MapMatchingTorchDataset(
            [d[0] for d in train_data],
            [d[1] for d in train_data],
            [d[2] for d in train_data],
            [d[3] for d in train_data]
        )

        self.eval_dataset = MapMatchingTorchDataset(
            [d[0] for d in eval_data],
            [d[1] for d in eval_data],
            [d[2] for d in eval_data],
            [d[3] for d in eval_data]
        )

        self.test_dataset = MapMatchingTorchDataset(
            [d[0] for d in test_data],
            [d[1] for d in test_data],
            [d[2] for d in test_data],
            [d[3] for d in test_data]
        )

        self._logger.info('Split: train={}, eval={}, test={}'.format(
            len(train_data), len(eval_data), len(test_data)))

    def _cut_trajectory_road_centric(self, all_traces_grid, all_traces_gps, all_original_idx, roads):
        """
        Cut and align trajectories using road-centric approach.

        This method ensures proper alignment between GPS traces and road sequences
        by using road count as the primary reference:
        1. Cuts based on road sequence length (max_road_len)
        2. Proportionally samples GPS points to match road segments
        3. Applies downsampling to GPS points

        Args:
            all_traces_grid: Full GPS trajectory converted to grid IDs
            all_traces_gps: Full GPS coordinates [lat, lon]
            all_original_idx: Original indices of GPS points
            roads: Target road sequence (ground truth)

        Returns:
            List of (traces_grid, roads, traces_gps, sample_idx) tuples
            where len(traces_grid) approximately equals len(roads)
        """
        results = []
        n_roads = len(roads)
        n_gps = len(all_traces_grid)

        if n_roads < self.min_road_len:
            return results

        if n_gps == 0:
            return results

        # Calculate GPS-to-road ratio for proportional sampling
        gps_road_ratio = n_gps / n_roads

        # Cut based on ROAD count
        if n_roads <= self.max_road_len:
            # Single segment - sample GPS points proportionally
            sampled_traces, sampled_gps, sample_idx = self._sample_gps_for_roads(
                all_traces_grid, all_traces_gps, all_original_idx,
                0, n_roads, gps_road_ratio, n_gps
            )

            if len(sampled_traces) > 0 and len(roads) > 0:
                results.append((sampled_traces, roads, sampled_gps, sample_idx))
            return results

        # Multiple segments - cut based on road count
        road_start = 0
        while road_start < n_roads:
            road_end = min(road_start + self.max_road_len, n_roads)
            seg_len = road_end - road_start

            # Handle last segment - merge if too short
            if seg_len < self.min_road_len:
                if len(results) > 0:
                    # Merge with previous segment
                    prev = results[-1]
                    prev_road_start = road_start - len(prev[1])

                    # Re-sample GPS for merged segment
                    sampled_traces, sampled_gps, sample_idx = self._sample_gps_for_roads(
                        all_traces_grid, all_traces_gps, all_original_idx,
                        prev_road_start, road_end, gps_road_ratio, n_gps
                    )

                    merged_roads = list(prev[1]) + roads[road_start:road_end]

                    if len(sampled_traces) > 0 and len(merged_roads) > 0:
                        results[-1] = (sampled_traces, merged_roads, sampled_gps, sample_idx)
                break

            # Sample GPS points for this road segment
            sampled_traces, sampled_gps, sample_idx = self._sample_gps_for_roads(
                all_traces_grid, all_traces_gps, all_original_idx,
                road_start, road_end, gps_road_ratio, n_gps
            )

            seg_roads = roads[road_start:road_end]

            if len(sampled_traces) > 0 and len(seg_roads) > 0:
                results.append((sampled_traces, seg_roads, sampled_gps, sample_idx))

            road_start = road_end

        return results

    def _sample_gps_for_roads(self, all_traces_grid, all_traces_gps, all_original_idx,
                               road_start, road_end, gps_road_ratio, n_gps):
        """
        Sample GPS points to match a road segment range.

        Uses proportional mapping to select GPS points that correspond to
        the given road segment range, then applies downsampling.

        Args:
            all_traces_grid: Full GPS grid IDs
            all_traces_gps: Full GPS coordinates
            all_original_idx: Original indices
            road_start: Start index in road sequence
            road_end: End index in road sequence
            gps_road_ratio: Ratio of GPS points to road segments
            n_gps: Total GPS points

        Returns:
            sampled_traces, sampled_gps, sample_idx
        """
        # Calculate corresponding GPS range
        gps_start = int(road_start * gps_road_ratio)
        gps_end = min(int(road_end * gps_road_ratio), n_gps)

        # Ensure we have at least some GPS points
        if gps_end <= gps_start:
            gps_end = min(gps_start + 1, n_gps)

        # Extract segment
        seg_traces = all_traces_grid[gps_start:gps_end]
        seg_gps = all_traces_gps[gps_start:gps_end]
        seg_original_idx = all_original_idx[gps_start:gps_end]

        if len(seg_traces) == 0:
            return [], [], []

        # Apply downsampling
        if self.downsample_rate < 1.0:
            return self._downsample_segment(seg_traces, seg_gps, seg_original_idx)
        else:
            # Create sample indices relative to segment start
            sample_idx = list(range(len(seg_traces)))
            return seg_traces, seg_gps, sample_idx

    def _downsample_segment(self, traces, gps, original_idx):
        """
        Downsample a trajectory segment while keeping first and last points.

        Args:
            traces: Grid trace sequence
            gps: GPS coordinate sequence
            original_idx: Original indices

        Returns:
            Downsampled traces, gps, and sample indices
        """
        if len(traces) <= 2:
            return list(traces), list(gps), list(range(len(traces)))

        sampled_traces = [traces[0]]
        sampled_gps = [gps[0]]
        sample_idx = [0]

        for i in range(1, len(traces) - 1):
            if random.random() <= self.downsample_rate:
                sampled_traces.append(traces[i])
                sampled_gps.append(gps[i])
                sample_idx.append(i)

        sampled_traces.append(traces[-1])
        sampled_gps.append(gps[-1])
        sample_idx.append(len(traces) - 1)

        return sampled_traces, sampled_gps, sample_idx

    def _downsample(self, traces, gps, rate):
        """
        Randomly downsample trajectory points.

        Args:
            traces: Grid trace sequence
            gps: GPS coordinate sequence
            rate: Sampling rate (0-1)

        Returns:
            Downsampled traces, gps, and sample indices
        """
        if len(traces) <= 2:
            return traces, gps, list(range(len(traces)))

        sampled_traces = [traces[0]]
        sampled_gps = [gps[0]]
        sample_idx = [0]

        for i in range(1, len(traces) - 1):
            if random.random() <= rate:
                sampled_traces.append(traces[i])
                sampled_gps.append(gps[i])
                sample_idx.append(i)

        sampled_traces.append(traces[-1])
        sampled_gps.append(gps[-1])
        sample_idx.append(len(traces) - 1)

        return sampled_traces, sampled_gps, sample_idx

    def _cut_trajectory(self, traces, gps, sample_idx, roads):
        """
        Cut long trajectories into smaller segments.

        Args:
            traces: Grid trace sequence
            gps: GPS coordinate sequence
            sample_idx: Sample indices
            roads: Target road sequence

        Returns:
            List of (traces, roads, gps, sample_idx) tuples
        """
        results = []
        n = len(traces)

        if n < self.min_road_len:
            return results

        if n <= self.max_road_len:
            results.append((traces, roads, gps, sample_idx))
            return results

        # Cut into segments
        num_segments = (n - 1) // self.max_road_len + 1

        for i in range(num_segments):
            start = i * self.max_road_len
            end = min((i + 1) * self.max_road_len, n)

            if end - start < self.min_road_len:
                # Merge with previous segment
                if len(results) > 0:
                    prev = results[-1]
                    merged_traces = list(prev[0]) + traces[start:end]
                    merged_roads = list(prev[1]) + roads[start:min(end, len(roads))]
                    merged_gps = list(prev[2]) + gps[start:end]
                    merged_idx = list(prev[3]) + sample_idx[start:end]
                    results[-1] = (merged_traces, merged_roads, merged_gps, merged_idx)
                continue

            seg_traces = traces[start:end]
            seg_roads = roads[start:min(end, len(roads))]
            seg_gps = gps[start:end]
            seg_idx = sample_idx[start:end]

            results.append((seg_traces, seg_roads, seg_gps, seg_idx))

        return results

    def _build_data_structures(self):
        """Build all data structures needed for deep map matching."""
        # Compute MBR from road network
        self._compute_mbr()

        # Build road graph
        self._build_road_graph()

        # Build trace graph
        self._build_trace_graph()

        # Build mapping matrix
        self._build_mapping_matrix()

        # Process trajectories
        self._process_trajectories()

    def _save_cache(self):
        """Save processed data to cache file."""
        cache_data = {
            'num_roads': self.num_roads,
            'num_grids': self.num_grids,
            'grid2traceid_dict': self.grid2traceid_dict,
            'traceid2grid_dict': self.traceid2grid_dict,
            'geo_id_to_idx': self.geo_id_to_idx,
            'min_lat': self.min_lat,
            'min_lng': self.min_lng,
            'max_lat': self.max_lat,
            'max_lng': self.max_lng,
            'road_x': self.road_x,
            'road_adj': self.road_adj,
            'trace_weight': self.trace_weight,
            'trace_in_edge_index': self.trace_in_edge_index,
            'trace_out_edge_index': self.trace_out_edge_index,
            'trace_x': self.trace_x,
            'map_matrix': self.map_matrix,
            'A_list': self.A_list,
            'singleton_grid_mask': self.singleton_grid_mask,
            'singleton_grid_location': self.singleton_grid_location,
            'train_dataset': self.train_dataset,
            'eval_dataset': self.eval_dataset,
            'test_dataset': self.test_dataset,
        }

        with open(self.deep_mm_cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        self._logger.info('Saved cache to {}'.format(self.deep_mm_cache_file))

    def _load_cache(self):
        """Load processed data from cache file."""
        with open(self.deep_mm_cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        self.num_roads = cache_data['num_roads']
        self.num_grids = cache_data['num_grids']
        self.grid2traceid_dict = cache_data['grid2traceid_dict']
        self.traceid2grid_dict = cache_data['traceid2grid_dict']
        self.geo_id_to_idx = cache_data.get('geo_id_to_idx', {})
        self.min_lat = cache_data['min_lat']
        self.min_lng = cache_data['min_lng']
        self.max_lat = cache_data['max_lat']
        self.max_lng = cache_data['max_lng']
        self.road_x = cache_data['road_x']
        self.road_adj = cache_data['road_adj']
        self.trace_weight = cache_data['trace_weight']
        self.trace_in_edge_index = cache_data['trace_in_edge_index']
        self.trace_out_edge_index = cache_data['trace_out_edge_index']
        self.trace_x = cache_data['trace_x']
        self.map_matrix = cache_data['map_matrix']
        self.A_list = cache_data['A_list']
        self.singleton_grid_mask = cache_data['singleton_grid_mask']
        self.singleton_grid_location = cache_data['singleton_grid_location']
        self.train_dataset = cache_data['train_dataset']
        self.eval_dataset = cache_data['eval_dataset']
        self.test_dataset = cache_data['test_dataset']

    def get_data(self):
        """
        Get train, validation, and test DataLoaders.

        Returns:
            tuple: (train_dataloader, eval_dataloader, test_dataloader)
        """
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=padding_collate_fn,
            num_workers=self.num_workers
        )

        eval_dataloader = DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=padding_collate_fn,
            num_workers=self.num_workers
        )

        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=padding_collate_fn,
            num_workers=self.num_workers
        )

        self._logger.info('Created DataLoaders: train={}, eval={}, test={}'.format(
            len(train_dataloader), len(eval_dataloader), len(test_dataloader)))

        return train_dataloader, eval_dataloader, test_dataloader

    def get_data_feature(self):
        """
        Get data features for model initialization.

        Returns:
            dict: Dictionary containing data features:
                - num_roads: Number of road segments
                - num_grids: Number of grid cells
                - road_adj: Road adjacency (SparseTensor or dense tensor)
                - road_x: Road features
                - trace_in_edge_index: Trace graph incoming edges
                - trace_out_edge_index: Trace graph outgoing edges
                - trace_weight: Trace edge weights
                - map_matrix: Grid-to-road mapping matrix
                - A_list: Adjacency polynomial for CRF
                - singleton_grid_mask: Indices of singleton grids
                - singleton_grid_location: Locations of singleton grids
                - grid2traceid_dict: Grid to trace ID mapping
                - traceid2grid_dict: Trace ID to grid mapping
                - with_time: Whether data includes time information
                - delta_time: Whether time is in delta format
        """
        res = super().get_data_feature()

        res.update({
            'num_roads': self.num_roads,
            'num_grids': self.num_grids,
            'road_adj': self.road_adj,
            'road_x': self.road_x,
            'trace_in_edge_index': self.trace_in_edge_index,
            'trace_out_edge_index': self.trace_out_edge_index,
            'trace_weight': self.trace_weight,
            'map_matrix': self.map_matrix,
            'A_list': self.A_list,
            'singleton_grid_mask': self.singleton_grid_mask,
            'singleton_grid_location': self.singleton_grid_location,
            'grid2traceid_dict': self.grid2traceid_dict,
            'traceid2grid_dict': self.traceid2grid_dict,
        })

        return res
