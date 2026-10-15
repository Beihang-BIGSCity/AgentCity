# coding=utf-8
"""
DiffMMDataset: Dataset adapter for DiffMM map matching model.

This dataset provides the specialized batch format required by the DiffMM model.

Batch Format (after collate_fn):
The dataset returns plain dictionaries with keys matching DiffMM model expectations:
- seq_len: Actual sequence lengths [batch] (tensor)
- current_loc: Normalized GPS coordinates [batch, seq_len, 3] (norm_lat, norm_lng, norm_time)
- target: Ground truth road segment IDs [batch, seq_len]
- trg_onehot: One-hot encoded targets [batch, seq_len, num_roads]
- candidate_segs: Candidate segment IDs [batch, seq_len, num_cands]
- candidate_feats: Segment features [batch, seq_len, num_cands, 9]
- candidate_mask: Mask for valid candidates [batch, seq_len, num_cands]

Key Mapping (internal -> model expected):
- norm_gps_seq -> current_loc
- trg_rid -> target
- segs_id -> candidate_segs
- segs_feat -> candidate_feats
- segs_mask -> candidate_mask
- lengths -> seq_len (converted to tensor)

The dataset extends MapMatchingDataset to leverage its road network
preprocessing while providing DiffMM-specific batch formatting.

Key Features:
1. Candidate segment generation based on spatial proximity
2. Road segment feature computation (distance, angle, etc.)
3. One-hot encoding of target segments for diffusion training
4. Proper normalization of GPS coordinates

Executor Compatibility:
- Uses DeepMapMatchingExecutor (not TrajLocPredExecutor)
- Returns plain dicts (not Batch objects with to_tensor() method)

Author: LibCity adaptation for DiffMM
"""

import os
import pickle
import math
from logging import getLogger
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from libcity.data.dataset.map_matching_dataset import MapMatchingDataset
from libcity.utils.utils import ensure_dir


# Constants for GPS to grid conversion
GRID_SIZE = 50  # meters
LAT_PER_METER = 8.993203677616966e-06
LNG_PER_METER = 1.1700193970443768e-05


def gps2grid(lat, lng, min_lat, min_lng, grid_size=GRID_SIZE):
    """Convert GPS coordinates to grid cell indices."""
    lat_unit = LAT_PER_METER * grid_size
    lng_unit = LNG_PER_METER * grid_size

    locgrid_x = int((lat - min_lat) / lat_unit) + 1
    locgrid_y = int((lng - min_lng) / lng_unit) + 1

    return locgrid_x, locgrid_y


def haversine_distance(lat1, lng1, lat2, lng2):
    """Calculate the haversine distance between two GPS points in meters."""
    R = 6371000  # Earth radius in meters
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lng = math.radians(lng2 - lng1)

    a = math.sin(delta_lat / 2) ** 2 + \
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def compute_bearing(lat1, lng1, lat2, lng2):
    """Compute bearing angle between two GPS points in radians."""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lng = math.radians(lng2 - lng1)

    x = math.sin(delta_lng) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lng)

    bearing = math.atan2(x, y)
    return bearing


class DiffMMTorchDataset(Dataset):
    """PyTorch Dataset for DiffMM model training.

    Each sample contains:
    - norm_gps_seq: Normalized GPS sequence (seq_len, 3)
    - trg_rid: Target road segment IDs (seq_len,)
    - trg_onehot: One-hot encoded targets (seq_len, num_roads)
    - segs_id: Candidate segment IDs (seq_len, num_cands)
    - segs_feat: Candidate segment features (seq_len, num_cands, 9)
    - segs_mask: Candidate validity mask (seq_len, num_cands)
    """

    def __init__(self, samples, num_roads, num_cands=10):
        """
        Args:
            samples: List of processed trajectory samples
            num_roads: Total number of road segments
            num_cands: Maximum number of candidate segments per GPS point
        """
        self.samples = samples
        self.num_roads = num_roads
        self.num_cands = num_cands

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'norm_gps_seq': sample['norm_gps_seq'],
            'trg_rid': sample['trg_rid'],
            'trg_onehot': sample['trg_onehot'],
            'segs_id': sample['segs_id'],
            'segs_feat': sample['segs_feat'],
            'segs_mask': sample['segs_mask'],
            'length': sample['length']
        }


def diffmm_collate_fn(batch):
    """Collate function for DiffMM DataLoader.

    Pads variable-length sequences to the maximum length in the batch.

    Returns:
        Dictionary with padded tensors (using DiffMM model's expected keys):
        - seq_len: (batch_size,) - Actual sequence lengths (as tensor)
        - current_loc: (batch_size, max_seq_len, 3) - Normalized GPS coordinates
        - target: (batch_size, max_seq_len) - Ground truth road segment IDs
        - trg_onehot: (batch_size, max_seq_len, num_roads) - One-hot encoded targets
        - candidate_segs: (batch_size, max_seq_len, num_cands) - Candidate segment IDs
        - candidate_feats: (batch_size, max_seq_len, num_cands, 9) - Candidate features
        - candidate_mask: (batch_size, max_seq_len, num_cands) - Candidate validity mask
    """
    # Get maximum sequence length in this batch
    lengths = [item['length'] for item in batch]
    max_len = max(lengths)

    # Get dimensions from first sample
    num_roads = batch[0]['trg_onehot'].shape[1]
    num_cands = batch[0]['segs_id'].shape[1]

    # Prepare padded tensors
    batch_size = len(batch)
    norm_gps_seq = torch.zeros(batch_size, max_len, 3)
    trg_rid = torch.zeros(batch_size, max_len, dtype=torch.long)
    trg_onehot = torch.zeros(batch_size, max_len, num_roads)
    segs_id = torch.zeros(batch_size, max_len, num_cands, dtype=torch.long)
    segs_feat = torch.zeros(batch_size, max_len, num_cands, 9)
    segs_mask = torch.zeros(batch_size, max_len, num_cands)

    for i, item in enumerate(batch):
        length = item['length']
        norm_gps_seq[i, :length] = item['norm_gps_seq']
        trg_rid[i, :length] = item['trg_rid']
        trg_onehot[i, :length] = item['trg_onehot']
        segs_id[i, :length] = item['segs_id']
        segs_feat[i, :length] = item['segs_feat']
        segs_mask[i, :length] = item['segs_mask']

    # Return with keys matching DiffMM model's expected format
    return {
        'seq_len': torch.tensor(lengths, dtype=torch.long),  # Changed from 'lengths'
        'current_loc': norm_gps_seq,  # Changed from 'norm_gps_seq'
        'target': trg_rid,  # Changed from 'trg_rid'
        'trg_onehot': trg_onehot,  # Kept for training
        'candidate_segs': segs_id,  # Changed from 'segs_id'
        'candidate_feats': segs_feat,  # Changed from 'segs_feat'
        'candidate_mask': segs_mask  # Changed from 'segs_mask'
    }


class DiffMMDataset(MapMatchingDataset):
    """
    Dataset class for DiffMM map matching model.

    Extends MapMatchingDataset to provide the specialized batch format
    required by DiffMM, including:
    - Candidate segment generation
    - Road segment feature computation
    - One-hot target encoding
    - Normalized GPS coordinates

    Config Parameters:
        num_cands: Maximum number of candidate segments per GPS point (default: 10)
        cand_search_radius: Search radius for candidates in meters (default: 100)
        train_rate: Training data ratio (default: 0.7)
        eval_rate: Validation data ratio (default: 0.15)
        batch_size: Batch size for DataLoader (default: 32)
        max_seq_len: Maximum sequence length (default: 100)
        min_seq_len: Minimum sequence length (default: 5)
        num_workers: Number of workers for DataLoader (default: 0)
    """

    def __init__(self, config):
        # Initialize parent class to load basic map matching data
        super().__init__(config)

        self._logger = getLogger()

        # Load parent data from cache if needed
        if self.rd_nwk is None and self.cache_dataset:
            parent_cache_file = self.cache_file_name
            if os.path.exists(parent_cache_file):
                with open(parent_cache_file, 'rb') as f:
                    parent_cache = pickle.load(f)
                self.rd_nwk = parent_cache.get('rd_nwk')
                self.trajectory = parent_cache.get('trajectory')
                self.route = parent_cache.get('route')
                self.multi_traj = parent_cache.get('multi_traj', False)
                self._logger.info('Loaded parent data from cache: {}'.format(parent_cache_file))

        # DiffMM-specific configuration parameters
        self.num_cands = config.get('num_cands', 10)
        self.cand_search_radius = config.get('cand_search_radius', 100)  # meters
        self.train_rate = config.get('train_rate', 0.7)
        self.eval_rate = config.get('eval_rate', 0.15)
        self.batch_size = config.get('batch_size', 32)
        self.eval_batch_size = config.get('eval_batch_size', self.batch_size)
        self.max_seq_len = config.get('max_seq_len', 100)
        self.min_seq_len = config.get('min_seq_len', 5)
        self.num_workers = config.get('num_workers', 0)
        self.shuffle = config.get('shuffle', True)

        # Cache paths
        self.cache_file_folder = './libcity/cache/dataset_cache/'
        ensure_dir(self.cache_file_folder)
        self.diffmm_cache_file = os.path.join(
            self.cache_file_folder,
            'diffmm_dataset_{}_{}_{}.pkl'.format(
                self.dataset,
                self.num_cands,
                self.cand_search_radius
            )
        )

        # Data structures
        self.num_roads = 0
        self.geo_id_to_idx = {}
        self.idx_to_geo_id = {}
        self.road_coords = {}  # geo_id -> (start_lat, start_lng, end_lat, end_lng)
        self.road_features = {}  # geo_id -> feature dict

        # MBR (Minimum Bounding Rectangle)
        self.min_lat = None
        self.min_lng = None
        self.max_lat = None
        self.max_lng = None

        # Datasets
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

        # Process data
        if self.cache_dataset and os.path.exists(self.diffmm_cache_file):
            self._logger.info('Loading cached DiffMM data from {}'.format(
                self.diffmm_cache_file))
            self._load_cache()
        else:
            self._logger.info('Building DiffMM data structures...')
            self._build_data_structures()
            if self.cache_dataset:
                self._save_cache()

    def _compute_mbr(self):
        """Compute the Minimum Bounding Rectangle from road network."""
        min_lat, min_lng = float('inf'), float('inf')
        max_lat, max_lng = float('-inf'), float('-inf')

        for u, v, data in self.rd_nwk.edges(data=True):
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

    def _build_road_index(self):
        """Build road segment index and feature structures."""
        self._logger.info('Building road index...')

        # Create mapping from geo_id to consecutive indices
        edge_list = list(self.rd_nwk.edges(data=True))
        geo_ids = set()
        for u, v, data in edge_list:
            if 'geo_id' in data:
                geo_ids.add(data['geo_id'])

        # If no geo_ids, use edge indices
        if not geo_ids:
            for idx, (u, v, data) in enumerate(edge_list):
                data['geo_id'] = idx
                geo_ids.add(idx)

        geo_id_list = sorted(list(geo_ids))
        self.geo_id_to_idx = {gid: idx for idx, gid in enumerate(geo_id_list)}
        self.idx_to_geo_id = {idx: gid for gid, idx in self.geo_id_to_idx.items()}
        self.num_roads = len(geo_id_list)

        self._logger.info('Number of roads: {}'.format(self.num_roads))

        # Build road coordinates and features
        for u, v, data in edge_list:
            geo_id = data.get('geo_id')
            if geo_id is None or geo_id not in self.geo_id_to_idx:
                continue

            # Get coordinates
            u_lat = self.rd_nwk.nodes[u].get('lat', 0)
            u_lng = self.rd_nwk.nodes[u].get('lon', 0)
            v_lat = self.rd_nwk.nodes[v].get('lat', 0)
            v_lng = self.rd_nwk.nodes[v].get('lon', 0)

            self.road_coords[geo_id] = (u_lat, u_lng, v_lat, v_lng)

            # Compute road features
            mid_lat = (u_lat + v_lat) / 2
            mid_lng = (u_lng + v_lng) / 2
            length = haversine_distance(u_lat, u_lng, v_lat, v_lng)
            bearing = compute_bearing(u_lat, u_lng, v_lat, v_lng)

            self.road_features[geo_id] = {
                'start_lat': u_lat,
                'start_lng': u_lng,
                'end_lat': v_lat,
                'end_lng': v_lng,
                'mid_lat': mid_lat,
                'mid_lng': mid_lng,
                'length': length,
                'bearing': bearing,
                'speed': data.get('speed', 30.0)  # default speed in km/h
            }

    def _find_candidates(self, lat, lng, radius=None):
        """Find candidate road segments within radius of a GPS point.

        Args:
            lat: GPS latitude
            lng: GPS longitude
            radius: Search radius in meters (default: self.cand_search_radius)

        Returns:
            List of (geo_id, distance) tuples sorted by distance
        """
        if radius is None:
            radius = self.cand_search_radius

        candidates = []

        for geo_id, coords in self.road_coords.items():
            start_lat, start_lng, end_lat, end_lng = coords

            # Calculate distance to road midpoint as approximation
            mid_lat = (start_lat + end_lat) / 2
            mid_lng = (start_lng + end_lng) / 2
            dist = haversine_distance(lat, lng, mid_lat, mid_lng)

            if dist <= radius:
                candidates.append((geo_id, dist))

        # Sort by distance
        candidates.sort(key=lambda x: x[1])

        return candidates[:self.num_cands]

    def _compute_segment_features(self, lat, lng, geo_id, prev_lat=None, prev_lng=None):
        """Compute features for a candidate segment relative to a GPS point.

        Returns a 9-dimensional feature vector:
        [dist_to_start, dist_to_end, dist_to_mid,
         bearing_diff, length_norm, speed_norm,
         lat_diff, lng_diff, time_norm]
        """
        features = self.road_features.get(geo_id, {})

        start_lat = features.get('start_lat', 0)
        start_lng = features.get('start_lng', 0)
        end_lat = features.get('end_lat', 0)
        end_lng = features.get('end_lng', 0)
        mid_lat = features.get('mid_lat', 0)
        mid_lng = features.get('mid_lng', 0)
        length = features.get('length', 0)
        bearing = features.get('bearing', 0)
        speed = features.get('speed', 30.0)

        # Distances
        dist_to_start = haversine_distance(lat, lng, start_lat, start_lng)
        dist_to_end = haversine_distance(lat, lng, end_lat, end_lng)
        dist_to_mid = haversine_distance(lat, lng, mid_lat, mid_lng)

        # Bearing difference (if previous point available)
        bearing_diff = 0.0
        if prev_lat is not None and prev_lng is not None:
            traj_bearing = compute_bearing(prev_lat, prev_lng, lat, lng)
            bearing_diff = abs(bearing - traj_bearing)
            if bearing_diff > math.pi:
                bearing_diff = 2 * math.pi - bearing_diff

        # Normalize features
        dist_to_start_norm = dist_to_start / max(self.cand_search_radius, 1)
        dist_to_end_norm = dist_to_end / max(self.cand_search_radius, 1)
        dist_to_mid_norm = dist_to_mid / max(self.cand_search_radius, 1)
        bearing_diff_norm = bearing_diff / math.pi
        length_norm = length / 1000  # normalize to km
        speed_norm = speed / 120  # normalize to max 120 km/h
        lat_diff = (lat - mid_lat) * 1000  # scale lat diff
        lng_diff = (lng - mid_lng) * 1000  # scale lng diff

        return [
            dist_to_start_norm,
            dist_to_end_norm,
            dist_to_mid_norm,
            bearing_diff_norm,
            length_norm,
            speed_norm,
            lat_diff,
            lng_diff,
            0.0  # time placeholder
        ]

    def _normalize_gps(self, lat, lng, time_delta=0.0):
        """Normalize GPS coordinates to [0, 1] range based on MBR.

        Returns:
            (norm_lat, norm_lng, norm_time)
        """
        lat_range = self.max_lat - self.min_lat
        lng_range = self.max_lng - self.min_lng

        if lat_range == 0:
            lat_range = 1
        if lng_range == 0:
            lng_range = 1

        norm_lat = (lat - self.min_lat) / lat_range
        norm_lng = (lng - self.min_lng) / lng_range
        norm_time = min(time_delta / 3600, 1.0)  # normalize time to hours, cap at 1

        return norm_lat, norm_lng, norm_time

    def _process_trajectory(self, traj_points, route_data):
        """Process a single trajectory into DiffMM format.

        Args:
            traj_points: List of GPS points [dyna_id, lon, lat, time?, ...]
            route_data: Ground truth route [[dyna_id, road_id], ...]

        Returns:
            Processed sample dict or None if invalid
        """
        # Build mapping from dyna_id to road_id
        dyna_to_road = {}
        for route_point in route_data:
            if len(route_point) >= 2:
                dyna_id = route_point[0]
                road_id = route_point[1]
                if isinstance(road_id, (int, np.integer)):
                    dyna_to_road[dyna_id] = int(road_id)
                elif isinstance(road_id, str) and road_id.isdigit():
                    dyna_to_road[dyna_id] = int(road_id)

        # Process GPS points
        norm_gps_list = []
        trg_rid_list = []
        segs_id_list = []
        segs_feat_list = []
        segs_mask_list = []

        prev_lat, prev_lng = None, None
        prev_time = 0

        for point in traj_points:
            if len(point) < 3:
                continue

            dyna_id = point[0]
            lon, lat = point[1], point[2]
            time_val = point[3] if len(point) > 3 else 0

            # Skip points outside MBR
            if lat < self.min_lat or lat > self.max_lat or \
               lon < self.min_lng or lon > self.max_lng:
                continue

            # Get target road
            target_road = dyna_to_road.get(dyna_id)
            if target_road is None or target_road not in self.geo_id_to_idx:
                continue

            target_idx = self.geo_id_to_idx[target_road]

            # Normalize GPS
            if isinstance(time_val, (int, float)):
                time_delta = time_val - prev_time if prev_time else 0
                prev_time = time_val
            else:
                time_delta = 0

            norm_lat, norm_lng, norm_time = self._normalize_gps(lat, lon, time_delta)
            norm_gps_list.append([norm_lat, norm_lng, norm_time])

            # Target road ID
            trg_rid_list.append(target_idx)

            # Find candidate segments
            candidates = self._find_candidates(lat, lon)

            # Ensure target is in candidates (add if missing)
            cand_ids = [c[0] for c in candidates]
            if target_road not in cand_ids:
                candidates = [(target_road, 0.0)] + candidates[:self.num_cands - 1]
            else:
                candidates = candidates[:self.num_cands]

            # Pad candidates if needed
            while len(candidates) < self.num_cands:
                candidates.append((0, 0.0))  # padding candidate

            # Build candidate features
            cand_ids = []
            cand_feats = []
            cand_masks = []

            for cand_geo_id, _ in candidates:
                if cand_geo_id in self.geo_id_to_idx:
                    cand_idx = self.geo_id_to_idx[cand_geo_id]
                    feats = self._compute_segment_features(lat, lon, cand_geo_id, prev_lat, prev_lng)
                    cand_ids.append(cand_idx)
                    cand_feats.append(feats)
                    cand_masks.append(1.0)
                else:
                    cand_ids.append(0)
                    cand_feats.append([0.0] * 9)
                    cand_masks.append(0.0)

            segs_id_list.append(cand_ids)
            segs_feat_list.append(cand_feats)
            segs_mask_list.append(cand_masks)

            prev_lat, prev_lng = lat, lon

        # Check minimum sequence length
        seq_len = len(norm_gps_list)
        if seq_len < self.min_seq_len:
            return None

        # Truncate if needed
        if seq_len > self.max_seq_len:
            norm_gps_list = norm_gps_list[:self.max_seq_len]
            trg_rid_list = trg_rid_list[:self.max_seq_len]
            segs_id_list = segs_id_list[:self.max_seq_len]
            segs_feat_list = segs_feat_list[:self.max_seq_len]
            segs_mask_list = segs_mask_list[:self.max_seq_len]
            seq_len = self.max_seq_len

        # Convert to tensors
        norm_gps_seq = torch.tensor(norm_gps_list, dtype=torch.float)
        trg_rid = torch.tensor(trg_rid_list, dtype=torch.long)
        segs_id = torch.tensor(segs_id_list, dtype=torch.long)
        segs_feat = torch.tensor(segs_feat_list, dtype=torch.float)
        segs_mask = torch.tensor(segs_mask_list, dtype=torch.float)

        # Build one-hot targets
        trg_onehot = torch.zeros(seq_len, self.num_roads)
        for i, rid in enumerate(trg_rid_list):
            if rid < self.num_roads:
                trg_onehot[i, rid] = 1.0

        return {
            'norm_gps_seq': norm_gps_seq,
            'trg_rid': trg_rid,
            'trg_onehot': trg_onehot,
            'segs_id': segs_id,
            'segs_feat': segs_feat,
            'segs_mask': segs_mask,
            'length': seq_len
        }

    def _process_all_trajectories(self):
        """Process all trajectories into DiffMM format samples."""
        self._logger.info('Processing trajectories for DiffMM...')

        all_samples = []

        for usr_id, usr_trajs in self.trajectory.items():
            for traj_id, traj_points in usr_trajs.items():
                # Get corresponding ground truth route
                if self.route is None:
                    continue
                if usr_id not in self.route or traj_id not in self.route[usr_id]:
                    continue

                route_data = self.route[usr_id][traj_id]
                sample = self._process_trajectory(traj_points, route_data)

                if sample is not None:
                    all_samples.append(sample)

        self._logger.info('Total processed samples: {}'.format(len(all_samples)))

        if len(all_samples) == 0:
            self._logger.warning('No valid trajectory samples found! Creating dummy data.')
            # Create minimal dummy data
            for _ in range(10):
                dummy_sample = {
                    'norm_gps_seq': torch.rand(self.min_seq_len, 3),
                    'trg_rid': torch.randint(0, max(1, self.num_roads), (self.min_seq_len,)),
                    'trg_onehot': torch.zeros(self.min_seq_len, max(1, self.num_roads)),
                    'segs_id': torch.randint(0, max(1, self.num_roads), (self.min_seq_len, self.num_cands)),
                    'segs_feat': torch.rand(self.min_seq_len, self.num_cands, 9),
                    'segs_mask': torch.ones(self.min_seq_len, self.num_cands),
                    'length': self.min_seq_len
                }
                all_samples.append(dummy_sample)

        # Shuffle and split
        np.random.shuffle(all_samples)
        n_samples = len(all_samples)

        # Ensure at least 1 sample in each split when possible
        if n_samples >= 3:
            n_train = max(1, int(n_samples * self.train_rate))
            n_eval = max(1, int(n_samples * self.eval_rate))
            # Ensure we don't exceed total
            if n_train + n_eval >= n_samples:
                n_train = n_samples - 2
                n_eval = 1
            n_test = n_samples - n_train - n_eval
        elif n_samples == 2:
            n_train = 1
            n_eval = 1
            n_test = 0
        elif n_samples == 1:
            # Use same sample for all splits (for testing)
            self._logger.warning("Only 1 sample available - using it for train only")
            n_train = 1
            n_eval = 0
            n_test = 0
        else:
            raise ValueError("Dataset has no valid samples")

        train_samples = all_samples[:n_train]
        eval_samples = all_samples[n_train:n_train + n_eval]
        test_samples = all_samples[n_train + n_eval:] if n_samples > 1 else all_samples

        # Create datasets
        self.train_dataset = DiffMMTorchDataset(train_samples, self.num_roads, self.num_cands)
        self.eval_dataset = DiffMMTorchDataset(eval_samples, self.num_roads, self.num_cands)
        self.test_dataset = DiffMMTorchDataset(test_samples, self.num_roads, self.num_cands)

        self._logger.info('Split: train={}, eval={}, test={}'.format(
            len(train_samples), len(eval_samples), len(test_samples)))

    def _build_data_structures(self):
        """Build all data structures needed for DiffMM."""
        # Compute MBR
        self._compute_mbr()

        # Build road index
        self._build_road_index()

        # Process trajectories
        self._process_all_trajectories()

    def _save_cache(self):
        """Save processed data to cache file."""
        cache_data = {
            'num_roads': self.num_roads,
            'geo_id_to_idx': self.geo_id_to_idx,
            'idx_to_geo_id': self.idx_to_geo_id,
            'road_coords': self.road_coords,
            'road_features': self.road_features,
            'min_lat': self.min_lat,
            'min_lng': self.min_lng,
            'max_lat': self.max_lat,
            'max_lng': self.max_lng,
            'train_samples': self.train_dataset.samples if self.train_dataset else [],
            'eval_samples': self.eval_dataset.samples if self.eval_dataset else [],
            'test_samples': self.test_dataset.samples if self.test_dataset else [],
        }

        with open(self.diffmm_cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        self._logger.info('Saved cache to {}'.format(self.diffmm_cache_file))

    def _load_cache(self):
        """Load processed data from cache file."""
        with open(self.diffmm_cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        self.num_roads = cache_data['num_roads']
        self.geo_id_to_idx = cache_data['geo_id_to_idx']
        self.idx_to_geo_id = cache_data['idx_to_geo_id']
        self.road_coords = cache_data['road_coords']
        self.road_features = cache_data['road_features']
        self.min_lat = cache_data['min_lat']
        self.min_lng = cache_data['min_lng']
        self.max_lat = cache_data['max_lat']
        self.max_lng = cache_data['max_lng']

        # Recreate datasets
        train_samples = cache_data.get('train_samples', [])
        eval_samples = cache_data.get('eval_samples', [])
        test_samples = cache_data.get('test_samples', [])

        self.train_dataset = DiffMMTorchDataset(train_samples, self.num_roads, self.num_cands)
        self.eval_dataset = DiffMMTorchDataset(eval_samples, self.num_roads, self.num_cands)
        self.test_dataset = DiffMMTorchDataset(test_samples, self.num_roads, self.num_cands)

    def get_data(self):
        """Get train, validation, and test DataLoaders.

        Returns:
            tuple: (train_dataloader, eval_dataloader, test_dataloader)
        """
        # Safety check: ensure datasets have samples before creating DataLoaders
        train_len = len(self.train_dataset) if self.train_dataset else 0
        eval_len = len(self.eval_dataset) if self.eval_dataset else 0
        test_len = len(self.test_dataset) if self.test_dataset else 0

        if train_len == 0:
            self._logger.warning("Train dataset is empty - DataLoader may fail")

        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=min(self.batch_size, max(1, train_len)),
            shuffle=self.shuffle and train_len > 0,
            collate_fn=diffmm_collate_fn,
            num_workers=self.num_workers
        ) if train_len > 0 else None

        eval_dataloader = DataLoader(
            dataset=self.eval_dataset,
            batch_size=min(self.eval_batch_size, max(1, eval_len)),
            shuffle=False,
            collate_fn=diffmm_collate_fn,
            num_workers=self.num_workers
        ) if eval_len > 0 else None

        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=min(self.eval_batch_size, max(1, test_len)),
            shuffle=False,
            collate_fn=diffmm_collate_fn,
            num_workers=self.num_workers
        ) if test_len > 0 else None

        self._logger.info('Created DataLoaders: train={}, eval={}, test={}'.format(
            len(train_dataloader) if train_dataloader else 0,
            len(eval_dataloader) if eval_dataloader else 0,
            len(test_dataloader) if test_dataloader else 0))

        return train_dataloader, eval_dataloader, test_dataloader

    def get_data_feature(self):
        """Get data features for DiffMM model initialization.

        Returns:
            dict: Dictionary containing data features:
                - id_size: Number of road segments (required by DiffMM)
                - num_roads: Same as id_size (alias)
                - num_cands: Number of candidate segments per GPS point
                - geo_id_to_idx: Mapping from geo_id to index
                - idx_to_geo_id: Mapping from index to geo_id
                - with_time: Whether data includes time information
                - delta_time: Whether time is in delta format
        """
        res = super().get_data_feature()

        res.update({
            'id_size': self.num_roads,  # Required by DiffMM
            'num_roads': self.num_roads,
            'num_cands': self.num_cands,
            'geo_id_to_idx': self.geo_id_to_idx,
            'idx_to_geo_id': self.idx_to_geo_id,
        })

        return res
