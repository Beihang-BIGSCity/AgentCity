"""
PLMTrajRec Encoder

This encoder transforms LibCity trajectory data into the format expected by PLMTrajRec model.

PLMTrajRec expects:
- GPS coordinates (src_lat, src_lng)
- Masking information (mask_index, padd_index)
- Road candidate information (src_candi_id)
- Temporal information (forward_delta_t, backward_delta_t, etc.)
- Road condition indices (road_condition_xyt_index)
- Target labels (target_road_id, target_rate)

LibCity provides:
- Discrete location IDs with timestamps
- GPS coordinates accessible via geo file

Key Transformations:
1. Extract GPS from geo file using location ID mappings
2. Generate random masking for trajectory recovery task
3. Create dummy road candidates (can be enhanced with real road network data)
4. Compute temporal features from timestamps
5. Generate road condition grid indices from GPS coordinates
"""

import os
import math
import random
import numpy as np
import pandas as pd
from libcity.data.dataset.trajectory_encoder.abstract_trajectory_encoder import AbstractTrajectoryEncoder
from libcity.utils import parse_time


parameter_list = ['dataset', 'min_session_len', 'min_sessions', 'traj_encoder', 'cut_method',
                  'window_size', 'min_checkins', 'max_session_len']


def parse_coordinate(coordinate):
    """Parse coordinate string from geo file."""
    items = coordinate[1:-1].split(',')
    return float(items[0]), float(items[1])


class PLMTrajRecEncoder(AbstractTrajectoryEncoder):
    """
    Encoder for PLMTrajRec model.

    Transforms LibCity trajectory format to PLMTrajRec's expected batch format.
    """

    def __init__(self, config):
        super().__init__(config)
        self.uid = 0
        self.location2id = {}  # Original loc ID -> encoded ID
        self.id2location = {}  # Encoded ID -> original loc ID
        self.loc_id = 0

        # PLMTrajRec specific parameters
        self.id_size = config.get('id_size', 2505)  # Road network size
        self.soft_traj_num = config.get('soft_traj_num', 128)
        self.hidden_dim = config.get('hidden_dim', 512)
        self.keep_ratio = config.get('default_keep_ratio', 0.125)
        self.max_len = config.get('max_session_len', 100)

        # Road condition grid parameters
        self.grid_size = config.get('grid_size', 64)
        self.time_slots = config.get('time_slots', 24)  # 24 hours

        # Minimum Bounding Rectangle (MBR) for GPS normalization
        # Will be computed from data if not provided
        self.mbr = config.get('mbr', None)
        self._lat_list = []
        self._lng_list = []

        # Load geo file for GPS coordinates
        self.dataset = config.get('dataset', '')
        self.geo_file = config.get('geo_file', self.dataset)
        self.geo_data = None
        self._load_geo_data()

        # Location to GPS mapping
        self.loc2gps = {}

        # Feature dictionary - defines output format
        # NOTE: prompt_token removed - now a learnable model parameter
        # NOTE: src_candi_id changed to sparse int (location ID) instead of dense one-hot
        self.feature_dict = {
            'src_lat': 'float',
            'src_lng': 'float',
            'mask_index': 'int',
            'padd_index': 'int',
            'src_candi_id': 'int',  # (T,) - sparse location ID instead of (T, id_size) one-hot
            'traj_length': 'no_tensor',
            'road_condition_xyt_index': 'int',  # (T, 3)
            'forward_delta_t': 'float',
            'backward_delta_t': 'float',
            'forward_index': 'int',
            'backward_index': 'int',
            'target_road_id': 'int',
            'target_rate': 'float',
            'uid': 'int',  # User ID (scalar per sample, batched into 1D tensor)
            'target': 'int'  # Next location target (scalar per sample, batched into 1D tensor)
        }

        # Cache file name
        parameters_str = ''
        for key in parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'trajectory_plmtrajrec_{}.json'.format(parameters_str))

    def _load_geo_data(self):
        """Load geo data containing GPS coordinates."""
        geo_path = './raw_data/{}/{}.geo'.format(self.dataset, self.geo_file)
        if os.path.exists(geo_path):
            self.geo_data = pd.read_csv(geo_path)
        else:
            self.geo_data = None

    def _get_gps_for_location(self, loc_id):
        """Get GPS coordinates for a location ID."""
        if loc_id in self.loc2gps:
            return self.loc2gps[loc_id]

        if self.geo_data is not None:
            try:
                row = self.geo_data[self.geo_data['geo_id'] == loc_id]
                if len(row) > 0:
                    coord_str = row.iloc[0]['coordinates']
                    lng, lat = parse_coordinate(coord_str)
                    self.loc2gps[loc_id] = (lat, lng)
                    return (lat, lng)
            except Exception:
                pass

        # Fallback: generate synthetic GPS from location ID
        # This creates a grid-based mapping for testing purposes
        grid_lat = 30.65 + (loc_id % 100) * 0.001
        grid_lng = 104.04 + (loc_id // 100) * 0.001
        self.loc2gps[loc_id] = (grid_lat, grid_lng)
        return (grid_lat, grid_lng)

    def _normalize_gps(self, lat, lng):
        """Normalize GPS coordinates to [0, 1] using MBR."""
        if self.mbr is None:
            return lat, lng

        norm_lat = (lat - self.mbr['min_lat']) / (self.mbr['max_lat'] - self.mbr['min_lat'] + 1e-6)
        norm_lng = (lng - self.mbr['min_lng']) / (self.mbr['max_lng'] - self.mbr['min_lng'] + 1e-6)
        return max(0, min(1, norm_lat)), max(0, min(1, norm_lng))

    def _get_grid_index(self, lat, lng):
        """Convert GPS to grid indices for road condition lookup."""
        norm_lat, norm_lng = self._normalize_gps(lat, lng)
        x = int(norm_lat * (self.grid_size - 1))
        y = int(norm_lng * (self.grid_size - 1))
        return min(x, self.grid_size - 1), min(y, self.grid_size - 1)

    def _time_to_slot(self, time_obj):
        """Convert time to slot index (0-23 for hourly slots)."""
        return time_obj.hour

    def _generate_mask(self, seq_len):
        """Generate random mask for trajectory recovery task.

        Args:
            seq_len: Length of trajectory sequence

        Returns:
            mask_index: Binary mask where 1 indicates masked positions
            keep_indices: List of unmasked position indices
        """
        keep_count = max(2, int(seq_len * self.keep_ratio))
        all_indices = list(range(seq_len))
        keep_indices = sorted(random.sample(all_indices, min(keep_count, seq_len)))

        # Ensure first and last are kept for reference
        if 0 not in keep_indices:
            keep_indices = [0] + keep_indices[:-1] if len(keep_indices) > 1 else [0]
        if seq_len - 1 not in keep_indices and seq_len > 1:
            keep_indices = keep_indices[:-1] + [seq_len - 1] if len(keep_indices) > 1 else [seq_len - 1]

        mask_index = [0 if i in keep_indices else 1 for i in range(seq_len)]
        return mask_index, keep_indices

    def _compute_temporal_features(self, timestamps, mask_index):
        """Compute forward and backward temporal features.

        For masked positions, find the nearest unmasked positions and compute time deltas.

        Args:
            timestamps: List of datetime objects
            mask_index: Binary mask (1 = masked)

        Returns:
            forward_delta_t: Time delta to nearest forward unmasked point
            backward_delta_t: Time delta to nearest backward unmasked point
            forward_index: Index of nearest forward unmasked point
            backward_index: Index of nearest backward unmasked point
        """
        seq_len = len(timestamps)
        forward_delta_t = [0.0] * seq_len
        backward_delta_t = [0.0] * seq_len
        forward_index = list(range(seq_len))
        backward_index = list(range(seq_len))

        # Find unmasked positions
        unmasked = [i for i, m in enumerate(mask_index) if m == 0]

        for i in range(seq_len):
            if mask_index[i] == 0:
                # Unmasked position - delta is 0, index is self
                continue

            # Find nearest backward unmasked
            backward_candidates = [u for u in unmasked if u < i]
            if backward_candidates:
                back_idx = backward_candidates[-1]  # Nearest backward
                backward_index[i] = back_idx
                delta = (timestamps[i] - timestamps[back_idx]).total_seconds() / 3600.0  # hours
                backward_delta_t[i] = delta
            else:
                backward_index[i] = 0
                backward_delta_t[i] = 0.0

            # Find nearest forward unmasked
            forward_candidates = [u for u in unmasked if u > i]
            if forward_candidates:
                fwd_idx = forward_candidates[0]  # Nearest forward
                forward_index[i] = fwd_idx
                delta = (timestamps[fwd_idx] - timestamps[i]).total_seconds() / 3600.0  # hours
                forward_delta_t[i] = delta
            else:
                forward_index[i] = seq_len - 1
                forward_delta_t[i] = 0.0

        return forward_delta_t, backward_delta_t, forward_index, backward_index

    def encode(self, uid, trajectories, negative_sample=None):
        """Encode trajectories for PLMTrajRec model.

        Args:
            uid: User ID
            trajectories: List of trajectories, each trajectory is a list of points
                         Point format: [dyna_id, entity_id, time, timezone, location, ...]
            negative_sample: Not used for PLMTrajRec

        Returns:
            encoded_trajectories: List of encoded trajectory samples
        """
        uid = self.uid
        self.uid += 1
        encoded_trajectories = []

        for traj in trajectories:
            if len(traj) < 2:
                continue

            # Extract locations, timestamps, and GPS
            locations = []
            timestamps = []
            lats = []
            lngs = []

            for point in traj:
                loc = point[4]  # location field
                time_str = point[2]  # time field

                # Recode location
                if loc not in self.location2id:
                    self.location2id[loc] = self.loc_id
                    self.id2location[self.loc_id] = loc
                    self.loc_id += 1

                locations.append(self.location2id[loc])
                timestamps.append(parse_time(time_str))

                # Get GPS coordinates
                lat, lng = self._get_gps_for_location(loc)
                lats.append(lat)
                lngs.append(lng)

                # Track for MBR computation
                self._lat_list.append(lat)
                self._lng_list.append(lng)

            seq_len = len(locations)

            # Generate mask for trajectory recovery
            mask_index, keep_indices = self._generate_mask(seq_len)

            # Padding mask (all zeros for now - no padding within sequence)
            padd_index = [0] * seq_len

            # Compute temporal features
            forward_delta_t, backward_delta_t, forward_index, backward_index = \
                self._compute_temporal_features(timestamps, mask_index)

            # Generate road condition grid indices
            road_condition_xyt_index = []
            for i, (lat, lng, ts) in enumerate(zip(lats, lngs, timestamps)):
                x, y = self._get_grid_index(lat, lng)
                t = self._time_to_slot(ts)
                road_condition_xyt_index.append([x, y, t])

            # Generate sparse road candidate IDs (just the location ID modulo id_size)
            # Changed from dense one-hot to sparse integer for cache size efficiency
            src_candi_id = [loc % self.id_size for loc in locations]

            # Target road IDs (use location IDs modulo id_size)
            target_road_id = [loc % self.id_size for loc in locations]

            # Target rates (dummy - uniform distribution)
            target_rate = [0.5] * seq_len

            # NOTE: prompt_token removed - now a learnable model parameter

            # Compile encoded sample
            trace = []
            trace.append(lats)  # src_lat
            trace.append(lngs)  # src_lng
            trace.append(mask_index)  # mask_index
            trace.append(padd_index)  # padd_index
            trace.append(src_candi_id)  # src_candi_id (sparse int)
            trace.append(seq_len)  # traj_length (no_pad_int)
            trace.append(road_condition_xyt_index)  # road_condition_xyt_index
            trace.append(forward_delta_t)  # forward_delta_t
            trace.append(backward_delta_t)  # backward_delta_t
            trace.append(forward_index)  # forward_index
            trace.append(backward_index)  # backward_index
            trace.append(target_road_id)  # target_road_id
            trace.append(target_rate)  # target_rate
            trace.append(uid)  # uid
            # For next-location prediction, use the last position as the target
            trace.append(target_road_id[-1] if len(target_road_id) > 0 else 0)  # target: single next location

            encoded_trajectories.append(trace)

        return encoded_trajectories

    def gen_data_feature(self):
        """Generate data features after encoding all trajectories."""
        # Compute MBR from collected GPS coordinates
        if self._lat_list and self._lng_list:
            self.mbr = {
                'min_lat': min(self._lat_list),
                'max_lat': max(self._lat_list),
                'min_lng': min(self._lng_list),
                'max_lng': max(self._lng_list)
            }
        else:
            # Default MBR (Chengdu area)
            self.mbr = {
                'min_lat': 30.655,
                'max_lat': 30.727,
                'min_lng': 104.043,
                'max_lng': 104.129
            }

        # Padding configuration
        # NOTE: src_candi_id changed to sparse int (0) instead of dense one-hot vector
        self.pad_item = {
            'src_lat': 0.0,
            'src_lng': 0.0,
            'mask_index': 0,
            'padd_index': 1,  # Padding positions marked as 1
            'src_candi_id': 0,  # Sparse int padding
            'road_condition_xyt_index': [0, 0, 0],
            'forward_delta_t': 0.0,
            'backward_delta_t': 0.0,
            'forward_index': 0,
            'backward_index': 0,
            'target_road_id': 0,
            'target_rate': 0.0
            # NOTE: 'uid' and 'target' are no_pad_int fields, excluded from padding
        }

        # Data feature dictionary for model initialization
        self.data_feature = {
            'loc_size': self.loc_id + 1,
            'uid_size': self.uid,
            'id_size': self.id_size,
            'mbr': self.mbr,
            'grid_size': self.grid_size,
            'time_slots': self.time_slots,
            'hidden_dim': self.hidden_dim,
            'soft_traj_num': self.soft_traj_num,
            # Road condition placeholder - should be loaded from actual data
            'road_condition': None
        }
