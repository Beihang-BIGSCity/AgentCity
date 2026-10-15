"""
JGRM Encoder - Trajectory Encoder for JGRM Model

This encoder transforms standard LibCity trajectory data into the format expected by JGRM
(Joint GPS and Route Modeling for Trajectory Representation Learning).

JGRM requires specialized data fields that standard LibCity trajectory encoders don't provide:
- route_data: Temporal features per road segment (weekday, minute, delta_time)
- route_assign_mat: Road segment sequence (indices into vocab)
- gps_data: GPS point features (8 dimensions per point)
- gps_assign_mat: GPS-to-road assignments (which road segment each GPS point belongs to)
- gps_length: Number of GPS points per road segment

Since standard foursquare/gowalla datasets don't have actual GPS traces or road networks,
this encoder creates a compatibility layer with pseudo-mappings:
- route_assign_mat: Uses encoded location IDs (POI IDs) as "road segments"
- route_data: Extracts temporal features from timestamps
- gps_data: Creates synthetic GPS-like features from location/time data
- gps_assign_mat: Maps each trajectory point to itself (1-to-1 mapping)
- gps_length: Set to 1 per location (simplified)

Key Assumptions:
1. Each POI is treated as a "road segment" for the route encoder
2. Each check-in is treated as a single GPS point on that segment
3. The graph is built from location co-occurrence in trajectories
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict

from libcity.data.dataset.trajectory_encoder.abstract_trajectory_encoder import AbstractTrajectoryEncoder
from libcity.utils import parse_time


parameter_list = ['dataset', 'min_session_len', 'min_sessions', 'traj_encoder', 'cut_method',
                  'window_size', 'history_type', 'min_checkins', 'max_session_len']


class JGRMEncoder(AbstractTrajectoryEncoder):
    """
    Encoder for JGRM model.

    Transforms LibCity trajectory format to JGRM's expected batch format with
    route_data, route_assign_mat, gps_data, gps_assign_mat, and gps_length.
    """

    def __init__(self, config):
        super().__init__(config)
        self.uid = 0
        self.location2id = {}  # Original loc ID -> encoded ID
        self.id2location = {}  # Encoded ID -> original loc ID
        self.loc_id = 0
        self.tim_max = 47  # Time encoding: weekday hours 0-23, weekend 24-47

        # JGRM specific parameters
        self.gps_feat_num = config.get('gps_feat_num', 8)
        self.route_max_len = config.get('route_max_len', 100)
        self.history_type = config.get('history_type', 'splice')

        # Dataset info for geo file loading
        self.dataset = config.get('dataset', '')
        self.geo_file = config.get('geo_file', self.dataset)
        self.geo_data = None
        self._load_geo_data()

        # Location to GPS mapping
        self.loc2gps = {}

        # Track all GPS coordinates for normalization
        self._lat_list = []
        self._lng_list = []

        # Edge tracking for graph construction
        self.edge_pairs = defaultdict(int)  # (src, dst) -> count

        # Feature dictionary - defines output format for JGRM
        self.feature_dict = {
            'route_data': 'float',           # (T, 3) - weekday, minute, delta_time
            'route_assign_mat': 'int',       # (T,) - road segment indices
            'gps_data': 'float',             # (T, gps_feat_num) - GPS point features
            'gps_assign_mat': 'int',         # (T,) - GPS-to-road assignments
            'gps_length': 'int',             # (T,) - GPS points per road segment
            'history_route_data': 'float',   # History route data for context
            'history_route_assign_mat': 'int',
            'history_gps_data': 'float',
            'history_gps_assign_mat': 'int',
            'history_gps_length': 'int',
            'current_route_data': 'float',   # Current route data
            'current_route_assign_mat': 'int',
            'current_gps_data': 'float',
            'current_gps_assign_mat': 'int',
            'current_gps_length': 'int',
            'target': 'int',                 # Target location
            'target_tim': 'int',             # Target time
            'uid': 'int'                     # User ID
        }

        # For cut_off mode, use array types
        if self.history_type == 'cut_off':
            self.feature_dict['history_route_data'] = 'array of float'
            self.feature_dict['history_route_assign_mat'] = 'array of int'
            self.feature_dict['history_gps_data'] = 'array of float'
            self.feature_dict['history_gps_assign_mat'] = 'array of int'
            self.feature_dict['history_gps_length'] = 'array of int'

        # Cache file name
        parameters_str = ''
        for key in parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'trajectory_jgrm_{}.json'.format(parameters_str))

    def _load_geo_data(self):
        """Load geo data containing GPS coordinates if available."""
        geo_path = './raw_data/{}/{}.geo'.format(self.dataset, self.geo_file)
        if os.path.exists(geo_path):
            try:
                self.geo_data = pd.read_csv(geo_path)
            except Exception:
                self.geo_data = None
        else:
            self.geo_data = None

    def _parse_coordinate(self, coordinate):
        """Parse coordinate string from geo file."""
        try:
            items = coordinate[1:-1].split(',')
            return float(items[0]), float(items[1])
        except Exception:
            return None, None

    def _get_gps_for_location(self, loc_id):
        """Get GPS coordinates for a location ID."""
        if loc_id in self.loc2gps:
            return self.loc2gps[loc_id]

        if self.geo_data is not None:
            try:
                row = self.geo_data[self.geo_data['geo_id'] == loc_id]
                if len(row) > 0:
                    coord_str = row.iloc[0]['coordinates']
                    lng, lat = self._parse_coordinate(coord_str)
                    if lat is not None and lng is not None:
                        self.loc2gps[loc_id] = (lat, lng)
                        return (lat, lng)
            except Exception:
                pass

        # Fallback: generate synthetic GPS from location ID hash
        # Creates a pseudo-random but consistent mapping
        np.random.seed(hash(str(loc_id)) % (2**31))
        grid_lat = float(35.0 + np.random.random() * 5.0)  # Range: 35-40
        grid_lng = float(135.0 + np.random.random() * 5.0)  # Range: 135-140
        self.loc2gps[loc_id] = (grid_lat, grid_lng)
        return (grid_lat, grid_lng)

    def _time_encode(self, time):
        """Encode time to 0-47 (workday hours 0-23, weekend 24-47)."""
        if time.weekday() in [0, 1, 2, 3, 4]:
            return time.hour
        else:
            return time.hour + 24

    def _create_route_data(self, locations, timestamps):
        """
        Create route_data tensor: (seq_len, 3) with [weekday, minute, delta_time].

        Args:
            locations: List of encoded location IDs
            timestamps: List of datetime objects

        Returns:
            route_data: List of [weekday, minute_of_day, delta_time] per location
        """
        route_data = []
        for i, (loc, ts) in enumerate(zip(locations, timestamps)):
            weekday = ts.weekday() + 1  # 1-7, 0 reserved for mask
            minute_of_day = ts.hour * 60 + ts.minute + 1  # 1-1440, 0 reserved for mask

            # Delta time: time difference from previous point (in minutes)
            if i == 0:
                delta_time = 0.0
            else:
                delta = (ts - timestamps[i-1]).total_seconds() / 60.0
                delta_time = min(delta, 1440.0)  # Cap at 24 hours

            route_data.append([weekday, minute_of_day, delta_time])

        return route_data

    def _create_gps_data(self, locations, timestamps):
        """
        Create gps_data tensor: (seq_len, gps_feat_num) with GPS-like features.

        For standard POI datasets without actual GPS traces, we synthesize 8 features:
        1. Normalized latitude
        2. Normalized longitude
        3. Sin(hour)
        4. Cos(hour)
        5. Sin(day_of_week)
        6. Cos(day_of_week)
        7. Speed proxy (distance/time from prev point, normalized)
        8. Heading proxy (direction from prev point, as sin+cos)

        Args:
            locations: List of original location IDs (before encoding)
            timestamps: List of datetime objects

        Returns:
            gps_data: List of feature vectors per GPS point
        """
        gps_data = []

        # Get GPS coordinates for all locations
        gps_coords = [self._get_gps_for_location(loc) for loc in locations]

        for i, (loc, ts) in enumerate(zip(locations, timestamps)):
            lat, lng = gps_coords[i]

            # Track for normalization
            self._lat_list.append(lat)
            self._lng_list.append(lng)

            # Normalize lat/lng to [0, 1] approximately
            norm_lat = (lat - 30.0) / 20.0  # Rough normalization
            norm_lng = (lng - 100.0) / 50.0

            # Temporal features - convert to Python float for JSON serialization
            hour = ts.hour
            sin_hour = float(np.sin(2 * np.pi * hour / 24.0))
            cos_hour = float(np.cos(2 * np.pi * hour / 24.0))

            day = ts.weekday()
            sin_day = float(np.sin(2 * np.pi * day / 7.0))
            cos_day = float(np.cos(2 * np.pi * day / 7.0))

            # Speed and heading proxies
            if i == 0:
                speed_proxy = 0.0
                heading_proxy = 0.0
            else:
                prev_lat, prev_lng = gps_coords[i-1]
                delta_lat = lat - prev_lat
                delta_lng = lng - prev_lng
                distance = float(np.sqrt(delta_lat**2 + delta_lng**2))
                delta_time = max((ts - timestamps[i-1]).total_seconds(), 1.0)
                speed_proxy = float(min(distance / (delta_time / 3600.0), 1.0))  # Normalized
                heading_proxy = float(np.arctan2(delta_lat, delta_lng) / np.pi)  # -1 to 1

            # Ensure all features are Python native floats for JSON serialization
            features = [
                float(norm_lat), float(norm_lng),
                sin_hour, cos_hour,
                sin_day, cos_day,
                speed_proxy, heading_proxy
            ]

            # Pad or truncate to gps_feat_num
            while len(features) < self.gps_feat_num:
                features.append(0.0)
            features = features[:self.gps_feat_num]

            gps_data.append(features)

        return gps_data

    def _track_edges(self, locations):
        """Track location transitions for graph construction."""
        for i in range(len(locations) - 1):
            src = locations[i]
            dst = locations[i + 1]
            self.edge_pairs[(src, dst)] += 1

    def encode(self, uid, trajectories, negative_sample=None):
        """
        Encode trajectories for JGRM model.

        Args:
            uid: User ID
            trajectories: List of trajectories, each trajectory is a list of points
                         Point format: [dyna_id, entity_id, time, timezone, location, ...]
            negative_sample: Not used for JGRM

        Returns:
            encoded_trajectories: List of encoded trajectory samples
        """
        uid = self.uid
        self.uid += 1
        encoded_trajectories = []

        history_route_data = []
        history_route_assign_mat = []
        history_gps_data = []
        history_gps_assign_mat = []
        history_gps_length = []

        for index, traj in enumerate(trajectories):
            # Extract and encode locations
            current_loc = []
            current_tim = []
            original_locs = []  # Track original location IDs for GPS lookup
            timestamps = []

            for point in traj:
                loc = point[4]  # location field
                time_str = point[2]  # time field

                # Recode location
                if loc not in self.location2id:
                    self.location2id[loc] = self.loc_id
                    self.id2location[self.loc_id] = loc
                    self.loc_id += 1

                encoded_loc = self.location2id[loc]
                current_loc.append(encoded_loc)
                original_locs.append(loc)

                now_time = parse_time(time_str)
                timestamps.append(now_time)
                time_code = self._time_encode(now_time)
                current_tim.append(time_code)

            # Track edges for graph construction
            self._track_edges(current_loc)

            # Create JGRM-specific data for current trajectory
            current_route_data = self._create_route_data(current_loc, timestamps)
            current_route_assign_mat = current_loc[:]  # Location IDs as "road segments"
            current_gps_data = self._create_gps_data(original_locs, timestamps)
            current_gps_assign_mat = current_loc[:]  # Each GPS point maps to its location
            current_gps_length = [1] * len(current_loc)  # 1 GPS point per "road segment"

            # Skip first trajectory as it becomes history
            if index == 0:
                if self.history_type == 'splice':
                    history_route_data += current_route_data
                    history_route_assign_mat += current_route_assign_mat
                    history_gps_data += current_gps_data
                    history_gps_assign_mat += current_gps_assign_mat
                    history_gps_length += current_gps_length
                else:
                    history_route_data.append(current_route_data)
                    history_route_assign_mat.append(current_route_assign_mat)
                    history_gps_data.append(current_gps_data)
                    history_gps_assign_mat.append(current_gps_assign_mat)
                    history_gps_length.append(current_gps_length)
                continue

            # Generate training samples: predict each point from previous points
            for i in range(len(current_loc) - 1):
                trace = []
                target = current_loc[i + 1]
                target_tim = current_tim[i + 1]

                # History data (from previous trajectories)
                if self.history_type == 'splice':
                    trace.append(history_route_data.copy())
                    trace.append(history_route_assign_mat.copy())
                    trace.append(history_gps_data.copy())
                    trace.append(history_gps_assign_mat.copy())
                    trace.append(history_gps_length.copy())
                else:
                    trace.append([rd[:] for rd in history_route_data])
                    trace.append([rm[:] for rm in history_route_assign_mat])
                    trace.append([gd[:] for gd in history_gps_data])
                    trace.append([ga[:] for ga in history_gps_assign_mat])
                    trace.append([gl[:] for gl in history_gps_length])

                # Current trajectory up to position i
                trace.append(current_route_data[:i + 1])
                trace.append(current_route_assign_mat[:i + 1])
                trace.append(current_gps_data[:i + 1])
                trace.append(current_gps_assign_mat[:i + 1])
                trace.append(current_gps_length[:i + 1])

                # Target
                trace.append(target)
                trace.append(target_tim)
                trace.append(uid)

                encoded_trajectories.append(trace)

            # Update history
            if self.history_type == 'splice':
                history_route_data += current_route_data
                history_route_assign_mat += current_route_assign_mat
                history_gps_data += current_gps_data
                history_gps_assign_mat += current_gps_assign_mat
                history_gps_length += current_gps_length
            else:
                history_route_data.append(current_route_data)
                history_route_assign_mat.append(current_route_assign_mat)
                history_gps_data.append(current_gps_data)
                history_gps_assign_mat.append(current_gps_assign_mat)
                history_gps_length.append(current_gps_length)

        return encoded_trajectories

    def _build_edge_index(self):
        """
        Build edge_index for GAT from tracked location transitions.

        Returns:
            edge_index: numpy array of shape (2, num_edges) with [sources, targets]
        """
        if not self.edge_pairs:
            # Return minimal edge (self-loop on node 0)
            return np.array([[0], [0]], dtype=np.int64)

        sources = []
        targets = []
        for (src, dst), count in self.edge_pairs.items():
            # Add edge (can add multiple times for weight, or just once)
            sources.append(src)
            targets.append(dst)
            # Add reverse edge for undirected graph
            sources.append(dst)
            targets.append(src)

        edge_index = np.array([sources, targets], dtype=np.int64)
        return edge_index

    def gen_data_feature(self):
        """Generate data features after encoding all trajectories."""
        vocab_size = self.loc_id  # Number of unique locations (road segments)
        edge_index = self._build_edge_index()

        # Padding configuration
        # route_data is (T, 3), needs padding with zeros
        # route_assign_mat uses vocab_size as padding value (standard convention)
        self.pad_item = {
            'route_data': [0.0, 0.0, 0.0],
            'route_assign_mat': vocab_size,  # Padding index
            'gps_data': [0.0] * self.gps_feat_num,
            'gps_assign_mat': vocab_size,
            'gps_length': 0,
            'history_route_data': [0.0, 0.0, 0.0],
            'history_route_assign_mat': vocab_size,
            'history_gps_data': [0.0] * self.gps_feat_num,
            'history_gps_assign_mat': vocab_size,
            'history_gps_length': 0,
            'current_route_data': [0.0, 0.0, 0.0],
            'current_route_assign_mat': vocab_size,
            'current_gps_data': [0.0] * self.gps_feat_num,
            'current_gps_assign_mat': vocab_size,
            'current_gps_length': 0
        }

        # Maximum lengths for padding
        self.feature_max_len = {
            'route_data': self.route_max_len,
            'route_assign_mat': self.route_max_len,
            'gps_data': self.route_max_len,
            'gps_assign_mat': self.route_max_len,
            'gps_length': self.route_max_len,
            'history_route_data': self.route_max_len,
            'history_route_assign_mat': self.route_max_len,
            'history_gps_data': self.route_max_len,
            'history_gps_assign_mat': self.route_max_len,
            'history_gps_length': self.route_max_len,
            'current_route_data': self.route_max_len,
            'current_route_assign_mat': self.route_max_len,
            'current_gps_data': self.route_max_len,
            'current_gps_assign_mat': self.route_max_len,
            'current_gps_length': self.route_max_len
        }

        # Data feature dictionary for model initialization
        self.data_feature = {
            'loc_size': vocab_size + 1,
            'vocab_size': vocab_size,  # Required by JGRM for embedding layer
            'edge_index': edge_index.tolist(),  # Convert numpy array to list for JSON serialization
            'tim_size': self.tim_max + 2,
            'uid_size': self.uid,
            'loc_pad': vocab_size,
            'tim_pad': self.tim_max + 1,
            'gps_feat_num': self.gps_feat_num,
            'route_max_len': self.route_max_len
        }
