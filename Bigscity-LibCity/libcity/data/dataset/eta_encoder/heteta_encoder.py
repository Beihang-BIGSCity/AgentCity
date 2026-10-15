"""
HetETA Encoder for LibCity ETA task.

This encoder bridges trajectory data from ETADataset with the HetETA model's
expected traffic state format. HetETA uses heterogeneous information networks
(road network + vehicle trajectories) to predict link-level speeds, then
computes ETA as: ETA = sum(distance / speed).

Reference:
    Original HetETA data loading: repos/HetETA/codes/util/data_loader.py
    LibCity HetETA model: Bigscity-LibCity/libcity/model/eta/HetETA.py

Key adaptations:
    - Converts trajectory points to link-level speed observations
    - Creates multi-period temporal features (recent/daily/weekly)
    - Generates spatiotemporal grid format: X[batch, time_steps, num_nodes, features]
    - Provides adjacency matrices for heterogeneous graph convolution
    - Computes ETA by aggregating predicted speeds over route links

Architecture Overview:
    1. Trajectory -> Link speed extraction
    2. Speed observations -> Temporal feature matrix [T, N, F]
    3. Each sample: temporal window around departure time
    4. Output: 'X' tensor for speed prediction, route info for ETA calculation
"""

import os
import numpy as np
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
from collections import defaultdict

from libcity.data.dataset.eta_encoder.abstract_eta_encoder import AbstractETAEncoder


parameter_list = [
    'dataset', 'eta_encoder', 'seq_len', 'days', 'weeks', 'num_nodes',
]


def geo_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    Returns distance in kilometers.
    """
    lon1, lat1, lon2, lat2 = tuple(map(lambda x: radians(x), (lon1, lat1, lon2, lat2)))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


class HetETAEncoder(AbstractETAEncoder):
    """
    Encoder for HetETA (Heterogeneous ETA) model.

    This encoder transforms trajectory data into the traffic state format
    expected by HetETA:
        - X: [batch, input_window, num_nodes, feature_dim]
        - y: [batch, output_window, num_nodes, output_dim]

    The model predicts link-level speeds, then computes ETA by:
        ETA = sum(link_distance / predicted_speed)

    Required config parameters:
        - seq_len: Number of recent time steps (default: 4)
        - days: Number of daily pattern time steps (default: 4)
        - weeks: Number of weekly pattern time steps (default: 4)
        - num_nodes: Number of road links/nodes (default: 100)
        - per_period: Minutes per time period (default: 5)
        - feature_dim: Number of features per node (default: 1, speed only)

    Output format (matching HetETA model expectations):
        - X: Historical speed features [batch, T, N, F] where T = weeks + days + seq_len
        - link_distances: Distance traveled on each link [batch, N]
        - gt_eta_time: Ground truth travel time in seconds
        - Additional metadata for evaluation

    Note on simplified implementation:
        Since LibCity ETA datasets don't have pre-computed link-level speed history,
        this encoder creates a simplified representation:
        - Uses trajectory points to estimate current link speeds
        - Creates placeholder historical features with temporal patterns
        - Road network adjacency is approximated from trajectory data
    """

    def __init__(self, config):
        super().__init__(config)

        # Temporal configuration (matching HetETA model)
        self.seq_len = config.get('seq_len', 4)
        self.days = config.get('days', 4)
        self.weeks = config.get('weeks', 4)
        self.input_window = self.weeks + self.days + self.seq_len  # Total temporal steps
        self.output_window = config.get('output_window', 1)

        # Spatial configuration
        self.num_nodes = config.get('num_nodes', 100)  # Number of road links
        self.feature_dim = config.get('feature_dim', 1)  # Speed features
        self.output_dim = config.get('output_dim', 1)

        # Time period settings
        self.per_period = config.get('per_period', 5)  # Minutes per period

        # Grid discretization for spatial mapping
        self.grid_size = config.get('grid_size', 10)  # sqrt(num_nodes) for grid mapping

        # Feature dictionary defining output format
        # Order MUST match the order in encode() output tuple
        self.feature_dict = {
            # Spatiotemporal features (main input to model)
            'X': 'float',  # [input_window, num_nodes, feature_dim]
            'y': 'float',  # [output_window, num_nodes, output_dim]

            # Route information for ETA calculation
            'link_distances': 'float',  # [num_nodes] - distance on each link
            'link_ids': 'int',  # Variable length list of link IDs in route

            # Ground truth and metadata (matching ETAExecutor expectations)
            'time': 'float',  # Ground truth travel time (seconds) - expected by ETAExecutor
            'traj_len': 'int',
            'traj_id': 'int',
            'uid': 'int',
            'weekid': 'int',
            'timeid': 'int',
            'start_timestamp': 'int',
        }

        # Index of 'traj_len' for sorting (used by ETADataset)
        self.traj_len_idx = 5  # Position of traj_len in feature_dict

        # Build cache file name
        parameters_str = ''
        for key in parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'heteta_eta{}.json'.format(parameters_str))

        # Statistics tracking
        self.uid_size = 0
        self.time_list = []
        self.speed_observations = []  # Collect speed data for building historical features
        self.link_adjacency = defaultdict(set)  # Track link connectivity

        # Coordinate bounds for grid mapping
        self.lon_min = float('inf')
        self.lon_max = float('-inf')
        self.lat_min = float('inf')
        self.lat_max = float('-inf')

    def _update_bounds(self, lon, lat):
        """Update coordinate bounds for grid mapping."""
        self.lon_min = min(self.lon_min, lon)
        self.lon_max = max(self.lon_max, lon)
        self.lat_min = min(self.lat_min, lat)
        self.lat_max = max(self.lat_max, lat)

    def _coord_to_node(self, lon, lat):
        """
        Map coordinates to a node ID using grid discretization.
        This provides a simplified link identification from coordinates.

        Args:
            lon: Longitude
            lat: Latitude

        Returns:
            node_id: Integer node ID in range [0, num_nodes)
        """
        # Normalize to [0, 1] range
        if self.lon_max > self.lon_min and self.lat_max > self.lat_min:
            norm_lon = (lon - self.lon_min) / (self.lon_max - self.lon_min)
            norm_lat = (lat - self.lat_min) / (self.lat_max - self.lat_min)
        else:
            norm_lon = 0.5
            norm_lat = 0.5

        # Clamp to valid range
        norm_lon = max(0, min(0.999, norm_lon))
        norm_lat = max(0, min(0.999, norm_lat))

        # Map to grid
        grid_x = int(norm_lon * self.grid_size)
        grid_y = int(norm_lat * self.grid_size)

        # Convert to linear index
        node_id = grid_y * self.grid_size + grid_x

        return min(node_id, self.num_nodes - 1)

    def _extract_link_speeds(self, traj, dyna_feature_column):
        """
        Extract link-level speed observations from a trajectory.

        Args:
            traj: List of trajectory points
            dyna_feature_column: Column mapping for .dyna file

        Returns:
            link_speeds: Dict mapping node_id -> list of speed observations
            link_distances: Dict mapping node_id -> total distance on link
            timestamps: List of timestamps
            total_distance: Total trajectory distance (km)
        """
        link_speeds = defaultdict(list)
        link_distances = defaultdict(float)
        timestamps = []
        coordinates = []

        # First pass: collect coordinates and update bounds
        for point in traj:
            coord = eval(point[dyna_feature_column["coordinates"]])
            lon, lat = float(coord[0]), float(coord[1])
            coordinates.append((lon, lat))
            self._update_bounds(lon, lat)

            time_str = point[dyna_feature_column["time"]]
            timestamps.append(datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%SZ'))

        # Second pass: compute speeds for each segment
        total_distance = 0.0
        prev_node = None

        for i in range(len(coordinates) - 1):
            lon1, lat1 = coordinates[i]
            lon2, lat2 = coordinates[i + 1]

            # Calculate segment distance (km)
            segment_dist = geo_distance(lon1, lat1, lon2, lat2)
            total_distance += segment_dist

            # Calculate segment time (hours)
            time_delta = (timestamps[i + 1] - timestamps[i]).total_seconds() / 3600.0

            # Calculate speed (km/h)
            if time_delta > 0:
                speed = segment_dist / time_delta
                # Clip to reasonable range
                speed = max(1.0, min(speed, 120.0))
            else:
                speed = 30.0  # Default speed

            # Map to node
            node_id = self._coord_to_node(lon1, lat1)
            link_speeds[node_id].append(speed)
            link_distances[node_id] += segment_dist

            # Track adjacency
            if prev_node is not None and prev_node != node_id:
                self.link_adjacency[prev_node].add(node_id)
                self.link_adjacency[node_id].add(prev_node)
            prev_node = node_id

        return link_speeds, link_distances, timestamps, total_distance

    def _create_temporal_features(self, link_speeds, departure_time):
        """
        Create multi-period temporal feature tensor.

        Since we don't have historical speed data, this creates a simplified
        representation with temporal patterns:
        - Recent: Current speed observations with small variations
        - Daily: Speed with daily pattern variation
        - Weekly: Speed with weekly pattern variation

        Args:
            link_speeds: Dict mapping node_id -> list of speed observations
            departure_time: Departure datetime

        Returns:
            X: Tensor of shape [input_window, num_nodes, feature_dim]
        """
        # Initialize feature tensor
        X = np.zeros((self.input_window, self.num_nodes, self.feature_dim), dtype=np.float32)

        # Base speed per node (average of observations, or default)
        base_speeds = np.ones(self.num_nodes, dtype=np.float32) * 30.0  # Default 30 km/h

        for node_id, speeds in link_speeds.items():
            if speeds:
                base_speeds[node_id] = np.mean(speeds)

        # Time-based variation factors
        hour = departure_time.hour
        weekday = departure_time.weekday()

        # Morning rush (7-9), evening rush (17-19) have lower speeds
        def get_time_factor(h):
            if 7 <= h <= 9 or 17 <= h <= 19:
                return 0.7  # Rush hour
            elif 22 <= h or h <= 5:
                return 1.2  # Night (less traffic)
            else:
                return 1.0

        # Weekend has higher speeds
        def get_day_factor(d):
            return 1.1 if d >= 5 else 1.0

        # Fill temporal features
        # [0:weeks] - weekly patterns
        for w in range(self.weeks):
            week_hour = (hour + w * 24) % 24
            factor = get_time_factor(week_hour) * get_day_factor((weekday - w) % 7)
            noise = np.random.normal(1.0, 0.1, self.num_nodes)
            X[w, :, 0] = base_speeds * factor * noise

        # [weeks:weeks+days] - daily patterns
        for d in range(self.days):
            day_hour = (hour + d * 2) % 24  # Shift by 2 hours per step
            factor = get_time_factor(day_hour)
            noise = np.random.normal(1.0, 0.05, self.num_nodes)
            X[self.weeks + d, :, 0] = base_speeds * factor * noise

        # [weeks+days:weeks+days+seq_len] - recent patterns
        for s in range(self.seq_len):
            recent_hour = (hour - (self.seq_len - s - 1)) % 24
            factor = get_time_factor(recent_hour)
            noise = np.random.normal(1.0, 0.02, self.num_nodes)
            X[self.weeks + self.days + s, :, 0] = base_speeds * factor * noise

        # Normalize speeds to m/s for model
        X = X / 3.6  # km/h -> m/s

        return X

    def _create_target(self, link_speeds):
        """
        Create target tensor (current speeds).

        Args:
            link_speeds: Dict mapping node_id -> list of speed observations

        Returns:
            y: Tensor of shape [output_window, num_nodes, output_dim]
        """
        y = np.ones((self.output_window, self.num_nodes, self.output_dim),
                    dtype=np.float32) * 30.0  # Default 30 km/h

        for node_id, speeds in link_speeds.items():
            if speeds:
                y[0, node_id, 0] = np.mean(speeds)

        # Convert to m/s
        y = y / 3.6

        return y

    def _create_link_distance_vector(self, link_distances):
        """
        Create dense link distance vector for ETA calculation.

        Args:
            link_distances: Dict mapping node_id -> distance (km)

        Returns:
            distances: Array of shape [num_nodes] in meters
        """
        distances = np.zeros(self.num_nodes, dtype=np.float32)

        for node_id, dist in link_distances.items():
            distances[node_id] = dist * 1000  # km -> meters

        return distances

    def encode(self, uid, trajectories, dyna_feature_column):
        """
        Encode trajectories for HetETA model.

        Args:
            uid: User/entity ID
            trajectories: List of trajectories, each is a list of points
            dyna_feature_column: Column indices for features in .dyna file

        Returns:
            List of encoded trajectories (tuples matching feature_dict order)
        """
        self.uid_size = max(uid + 1, self.uid_size)
        encoded_trajectories = []

        for traj in trajectories:
            if len(traj) < 2:
                continue

            # Extract link-level speed data
            link_speeds, link_distances, timestamps, total_distance = \
                self._extract_link_speeds(traj, dyna_feature_column)

            if not link_speeds:
                continue

            # Get temporal information
            begin_time = timestamps[0]
            end_time = timestamps[-1]
            weekid = begin_time.weekday() + 1  # 1-7
            timeid = begin_time.hour * 60 + begin_time.minute + 1  # 1-1440

            # Ground truth travel time (seconds)
            gt_eta_time = (end_time - begin_time).total_seconds()
            self.time_list.append(gt_eta_time)

            # Create spatiotemporal features
            X = self._create_temporal_features(link_speeds, begin_time)
            y = self._create_target(link_speeds)

            # Create link distance vector for ETA calculation
            link_dist_vector = self._create_link_distance_vector(link_distances)

            # List of link IDs in route order
            link_ids = list(link_speeds.keys())

            # Trajectory metadata
            traj_len = len(traj)
            traj_id = int(traj[-1][dyna_feature_column["traj_id"]]) if "traj_id" in dyna_feature_column else 0
            start_timestamp = int(begin_time.timestamp())

            # Store speed observations for later normalization
            for speeds in link_speeds.values():
                self.speed_observations.extend(speeds)

            # Format output according to feature_dict order
            encoded_trajectories.append([
                X.tolist(),  # [input_window, num_nodes, feature_dim]
                y.tolist(),  # [output_window, num_nodes, output_dim]
                link_dist_vector.tolist(),  # [num_nodes]
                link_ids,  # Variable length
                [gt_eta_time],
                [traj_len],
                [traj_id],
                [uid],
                [weekid],
                [timeid],
                [start_timestamp],
            ])

        return encoded_trajectories

    def gen_data_feature(self):
        """
        Generate data features and padding items after encoding all trajectories.
        """
        # Padding items for variable-length sequences
        self.pad_item = {
            'link_ids': 0,
        }

        # Build adjacency matrix from observed trajectories
        adj_mx = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for node, neighbors in self.link_adjacency.items():
            for neighbor in neighbors:
                adj_mx[node, neighbor] = 1.0
                adj_mx[neighbor, node] = 1.0

        # Add self-loops
        np.fill_diagonal(adj_mx, 1.0)

        # Data features for model initialization
        self.data_feature = {
            'num_nodes': self.num_nodes,
            'feature_dim': self.feature_dim,
            'output_dim': self.output_dim,
            'adj_mx': adj_mx.tolist(),  # For JSON serialization
            'traj_len_idx': self.traj_len_idx,
            'uid_size': self.uid_size,
            'input_window': self.input_window,
            'output_window': self.output_window,
        }

        # Add normalization statistics
        if len(self.time_list) > 0:
            self.data_feature['time_mean'] = float(np.mean(self.time_list))
            self.data_feature['time_std'] = float(np.std(self.time_list))
        else:
            self.data_feature['time_mean'] = 0.0
            self.data_feature['time_std'] = 1.0

        if len(self.speed_observations) > 0:
            self.data_feature['speed_mean'] = float(np.mean(self.speed_observations))
            self.data_feature['speed_std'] = float(np.std(self.speed_observations))
        else:
            self.data_feature['speed_mean'] = 30.0
            self.data_feature['speed_std'] = 15.0

    def gen_scalar_data_feature(self, train_data):
        """
        Generate scalar data features from training data.

        Args:
            train_data: List of encoded training trajectories

        Returns:
            Dictionary of scalar features for normalization
        """
        travel_times = []
        speeds = []

        # Build column mapping
        scalar_feature_column = {}
        for i, key in enumerate(self.feature_dict):
            scalar_feature_column[key] = i

        # Extract statistics from training data
        for data in train_data:
            travel_times.extend(data[scalar_feature_column["time"]])

            # Extract speeds from X tensor (last recent time step)
            X = np.array(data[scalar_feature_column["X"]])
            if X.ndim == 3:
                # Get speeds from last time step, all nodes
                recent_speeds = X[-1, :, 0]  # [num_nodes]
                speeds.extend(recent_speeds[recent_speeds > 0].tolist())

        # Compute statistics
        scalar_data_feature = {
            'time_mean': float(np.mean(travel_times)) if travel_times else 0.0,
            'time_std': float(np.std(travel_times)) if travel_times else 1.0,
            'speed_mean': float(np.mean(speeds)) if speeds else 8.3,  # ~30 km/h in m/s
            'speed_std': float(np.std(speeds)) if speeds else 4.0,
            'uid_size': self.uid_size,
            'num_nodes': self.num_nodes,
            'feature_dim': self.feature_dim,
            'output_dim': self.output_dim,
            'input_window': self.input_window,
            'output_window': self.output_window,
            'traj_len_idx': self.traj_len_idx,
        }

        # Reconstruct adjacency matrix for model use
        adj_mx = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for node, neighbors in self.link_adjacency.items():
            for neighbor in neighbors:
                adj_mx[node, neighbor] = 1.0
                adj_mx[neighbor, node] = 1.0
        np.fill_diagonal(adj_mx, 1.0)
        scalar_data_feature['adj_mx'] = adj_mx

        # Add train_ prefix for normalization stats used by model
        scalar_data_feature['train_gt_eta_time_mean'] = scalar_data_feature['time_mean']
        scalar_data_feature['train_gt_eta_time_std'] = scalar_data_feature['time_std']

        for k, v in scalar_data_feature.items():
            if isinstance(v, (int, float, str)):
                self._logger.info("{}: {}".format(k, v))

        return scalar_data_feature
