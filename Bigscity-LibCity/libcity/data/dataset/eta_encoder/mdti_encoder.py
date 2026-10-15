"""
MDTI Encoder for LibCity ETA task.

This encoder prepares trajectory data for the MDTI (Multi-modal Dual Transformer
for Travel Time Estimation) model, which requires:
1. Grid-based features (grid_image, grid_traj, grid_time_emb, grid_feature)
2. Road network features (g_input_feature, g_edge_index, road_traj, road_weeks, road_minutes, road_type)
3. Travel time as ground truth

Reference:
    Original preprocessing: repos/MDTI/dataset/preprocess.py
    Grid utilities: repos/MDTI/dataset/grid_space.py
    MDTI DataLoader: repos/MDTI/dataset/mdti_loader.py
    TTE DataLoader: repos/MDTI/dataset/tte_loader.py

Key adaptations:
    - Implements LibCity's AbstractETAEncoder interface
    - Generates multimodal features for grid and road networks
    - Handles grid-space rasterization similar to original MDTI
    - Generates temporal embeddings using Date2Vec-like approach
"""

import os
import math
import numpy as np
from datetime import datetime
from math import radians, cos, sin, asin, sqrt, atan2, degrees

from libcity.data.dataset.eta_encoder.abstract_eta_encoder import AbstractETAEncoder


parameter_list = [
    'dataset', 'eta_encoder', 'grid_size', 'hidden_emb_dim',
]
parameter_list_cut = [
    'dataset', 'eta_encoder', 'grid_size', 'hidden_emb_dim',
    'cut_method', 'min_session_len', 'max_session_len', 'min_sessions', 'window_size',
]


def geo_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).

    Returns:
        Distance in kilometers
    """
    lon1, lat1, lon2, lat2 = tuple(map(lambda x: radians(x), (lon1, lat1, lon2, lat2)))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


class GridSpace:
    """
    Grid space for trajectory rasterization.

    Adapted from repos/MDTI/dataset/grid_space.py
    """

    def __init__(self, x_unit: int, y_unit: int, x_min, y_min, x_max, y_max):
        assert x_unit > 0 and y_unit > 0

        # Convert grid size from meters to degrees
        self.x_unit = x_unit * 360 / (2 * math.pi * 6378137 * math.cos((y_min + y_max) * math.pi / 360))
        self.y_unit = y_unit * 360 / (2 * math.pi * 6378137)

        # Whole space MBR range
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

        self.x_size = int(math.ceil((x_max - x_min) / self.x_unit))
        self.y_size = int(math.ceil((y_max - y_min) / self.y_unit))
        self.grid_num = self.x_size * self.y_size

    def get_xyidx_by_point(self, x, y):
        """Get grid cell indices for a coordinate."""
        # Clamp to valid range
        x = max(self.x_min, min(x, self.x_max))
        y = max(self.y_min, min(y, self.y_max))

        i_x = int(math.floor((x - self.x_min) / self.x_unit))
        i_y = int(math.floor((y - self.y_min) / self.y_unit))

        # Ensure indices are valid
        i_x = max(0, min(i_x, self.x_size - 1))
        i_y = max(0, min(i_y, self.y_size - 1))

        return (i_x, i_y)

    def get_gridid_by_xyidx(self, i_x: int, i_y: int):
        """Convert 2D indices to 1D grid ID."""
        return i_x * self.y_size + i_y

    def get_gridid_by_point(self, x, y):
        """Get 1D grid ID for a coordinate."""
        i_x, i_y = self.get_xyidx_by_point(x, y)
        return self.get_gridid_by_xyidx(i_x, i_y)

    def get_center_point(self, i_x: int, i_y: int):
        """Get center coordinate of a grid cell."""
        return (self.x_min + self.x_unit / 2 + self.x_unit * i_x,
                self.y_min + self.y_unit / 2 + self.y_unit * i_y)


def generate_spatial_features(src, gs):
    """
    Generate spatial features for trajectory points.

    For each point, generates [x_norm, y_norm, distance, bearing].

    Args:
        src: List of [lon, lat] coordinates
        gs: GridSpace instance

    Returns:
        List of [x_norm, y_norm, distance, bearing] for each point
    """
    tgt = []

    for i in range(1, len(src)):
        coord1 = [src[i - 1][0], src[i - 1][1]]
        coord2 = [src[i][0], src[i][1]]

        # Calculate distance (need to swap for geodesic)
        distance = geo_distance(coord1[0], coord1[1], coord2[0], coord2[1])

        # Calculate bearing
        bearing = atan2(coord2[1] - coord1[1], coord2[0] - coord1[0])
        bearing = (degrees(bearing) + 360) % 360 / 360

        # Normalize coordinates
        x = (coord2[0] - gs.x_min) / (gs.x_max - gs.x_min) if gs.x_max != gs.x_min else 0
        y = (coord2[1] - gs.y_min) / (gs.y_max - gs.y_min) if gs.y_max != gs.y_min else 0

        tgt.append([x, y, distance, bearing])

    # Add first point features
    x = (src[0][0] - gs.x_min) / (gs.x_max - gs.x_min) if gs.x_max != gs.x_min else 0
    y = (src[0][1] - gs.y_min) / (gs.y_max - gs.y_min) if gs.y_max != gs.y_min else 0
    tgt.insert(0, [x, y, 0.0, 0.0])

    return tgt


def generate_time_embedding(timestamp, hidden_dim):
    """
    Generate Date2Vec-style time embedding.

    Uses sinusoidal encoding similar to the original Date2Vec model.

    Args:
        timestamp: datetime object
        hidden_dim: Embedding dimension

    Returns:
        numpy array of shape [hidden_dim]
    """
    # Extract time features
    t = [timestamp.hour, timestamp.minute, timestamp.second,
         timestamp.year % 100, timestamp.month, timestamp.day]

    # Create sinusoidal encoding
    k = hidden_dim
    k1 = k // 2
    k2 = k - k1

    # Linear features
    linear_part = np.zeros(k1)
    for i in range(min(len(t), k1)):
        linear_part[i] = t[i % len(t)] / max(t[i % len(t)] + 1, 1)

    # Sinusoidal features
    sin_part = np.zeros(k2)
    for i in range(k2):
        freq = (i + 1) / k2
        val = sum(t[j] * freq for j in range(len(t)))
        sin_part[i] = np.sin(val * np.pi / 180)

    return np.concatenate([linear_part, sin_part])


class MDTIEncoder(AbstractETAEncoder):
    """
    Encoder for MDTI (Multi-modal Dual Transformer) model.

    This encoder prepares trajectory data for the MDTI model which uses
    both grid-based and road network representations.

    Required config parameters:
        - grid_size: Grid cell size in meters (default: 100)
        - hidden_emb_dim: Hidden embedding dimension (default: 256)
        - grid_H: Grid height (optional, computed automatically)
        - grid_W: Grid width (optional, computed automatically)
        - min_lon, max_lon, min_lat, max_lat: Bounding box (computed if not provided)

    Output features:
        Grid features:
            - grid_image: Trajectory rasterized to RGB grid image (H x W x 3)
            - grid_traj: Sequence of grid cell indices
            - grid_time_emb: Temporal embeddings for each point
            - grid_feature: Spatial features (x_norm, y_norm, distance, bearing)

        Road network features:
            - g_input_feature: Road node features (num_roads x g_fea_size)
            - g_edge_index: Road network graph connectivity
            - road_traj: Sequence of road segment IDs
            - road_weeks: Day-of-week for each step
            - road_minutes: Minute-of-day for each step
            - road_type: Road type for each segment

        Target:
            - travel_time: Ground truth travel time in minutes
    """

    def __init__(self, config):
        super().__init__(config)

        # Grid configuration
        self.grid_size = config.get('grid_size', 100)  # Grid cell size in meters
        self.hidden_emb_dim = config.get('hidden_emb_dim', 256)

        # Bounding box (will be computed during encoding if not provided)
        self.min_lon = config.get('min_lon', None)
        self.max_lon = config.get('max_lon', None)
        self.min_lat = config.get('min_lat', None)
        self.max_lat = config.get('max_lat', None)

        # Grid dimensions (computed from bounding box)
        self.grid_H = config.get('grid_H', None)
        self.grid_W = config.get('grid_W', None)

        # Road special tokens
        self.road_special_tokens = config.get('road_special_tokens', {
            'padding_token': 0,
            'cls_token': 1,
            'mask_token': 2,
        })
        self.grid_special_tokens = config.get('grid_special_tokens', {
            'padding_token': 0,
            'cls_token': 1,
        })

        # Feature dictionary defining output format
        # The order here must match the order of features in encode() output
        # NOTE: grid_image is NOT stored per-trajectory to save memory.
        # Instead, it is stored once in data_feature and accessed by the model.
        self.feature_dict = {
            'grid_traj': 'int',                 # Shape: [seq_len] - grid cell indices
            'grid_time_emb': 'float',           # Shape: [seq_len, hidden_dim] - temporal embeddings
            'grid_feature': 'float',            # Shape: [seq_len, 4] - spatial features
            'g_input_feature': 'no_pad_float',  # Shape: (num_roads, g_fea_size) - road features
            'g_edge_index': 'no_pad_int',       # Shape: (2, num_edges) - graph connectivity
            'road_traj': 'int',                 # Shape: [seq_len] - road segment IDs
            'road_weeks': 'int',                # Shape: [seq_len] - day of week
            'road_minutes': 'int',              # Shape: [seq_len] - minute of day
            'road_type': 'int',                 # Shape: [seq_len] - road type
            'travel_time': 'float',             # Shape: [1] - travel time in minutes
            'traj_len': 'int',                  # Shape: [1] - trajectory length
            'traj_id': 'int',                   # Shape: [1] - trajectory ID
            'start_timestamp': 'int',           # Shape: [1] - start timestamp
        }

        self.traj_len_idx = len(self.feature_dict) - 4  # Index of traj_len (adjusted after removing grid_image)

        # Build cache file name
        parameters_str = ''
        need_cut = self.config.get("need_cut", False)
        self.parameter_list = parameter_list_cut if need_cut else parameter_list
        for key in self.parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'mdti_eta{}.json'.format(parameters_str))

        # Statistics tracking
        self.all_lons = []
        self.all_lats = []
        self.normalization_computed = False
        self.grid_space = None

        # Road network statistics
        self.road_num = 0
        self.road_type_num = 0
        self.g_fea_size = 0

        # Global grid image (will be built after first pass)
        self.global_grid_image = None

        # Road network features (placeholder, should be loaded from data)
        self.road_features = None
        self.road_edge_index = None

        # Time statistics
        self.time_list = []

    def _compute_bounding_box(self, trajectories, dyna_feature_column):
        """
        Compute bounding box from trajectory data.

        Args:
            trajectories: List of trajectory data
            dyna_feature_column: Column mapping for trajectory features
        """
        for traj in trajectories:
            for point in traj:
                coordinate = eval(point[dyna_feature_column["coordinates"]])
                lon, lat = float(coordinate[0]), float(coordinate[1])
                self.all_lons.append(lon)
                self.all_lats.append(lat)

        if len(self.all_lons) > 0:
            self.min_lon = min(self.all_lons)
            self.max_lon = max(self.all_lons)
            self.min_lat = min(self.all_lats)
            self.max_lat = max(self.all_lats)

    def _initialize_grid_space(self):
        """Initialize grid space after bounding box is computed."""
        if self.min_lon is None or self.grid_space is not None:
            return

        self.grid_space = GridSpace(
            self.grid_size, self.grid_size,
            self.min_lon, self.min_lat,
            self.max_lon, self.max_lat
        )

        # Set grid dimensions
        if self.grid_H is None:
            self.grid_H = self.grid_space.x_size
        if self.grid_W is None:
            self.grid_W = self.grid_space.y_size

        self._logger.info(f"Grid space initialized: {self.grid_H} x {self.grid_W} "
                         f"(lon: {self.min_lon:.4f}-{self.max_lon:.4f}, "
                         f"lat: {self.min_lat:.4f}-{self.max_lat:.4f})")

    def _construct_grid_image(self, trajectories, dyna_feature_column):
        """
        Construct global grid image from trajectories.

        The grid image has 3 channels:
            - Channel 0: Normalized x coordinate
            - Channel 1: Normalized y coordinate
            - Channel 2: Traffic density (normalized count)

        Args:
            trajectories: List of trajectories
            dyna_feature_column: Column mapping

        Returns:
            numpy array of shape (H, W, 3)
        """
        if self.grid_space is None:
            return None

        # Build traffic density image
        traffic_image = np.zeros((self.grid_space.x_size, self.grid_space.y_size))

        for traj in trajectories:
            for point in traj:
                coordinate = eval(point[dyna_feature_column["coordinates"]])
                lon, lat = float(coordinate[0]), float(coordinate[1])
                try:
                    p_x, p_y = self.grid_space.get_xyidx_by_point(lon, lat)
                    traffic_image[p_x, p_y] += 1
                except (AssertionError, ValueError):
                    continue

        # Normalize traffic image
        if np.max(traffic_image) > np.min(traffic_image):
            traffic_image = (traffic_image - np.min(traffic_image)) / (np.max(traffic_image) - np.min(traffic_image))

        # Build coordinate images
        x_image = np.zeros((self.grid_space.x_size, self.grid_space.y_size))
        y_image = np.zeros((self.grid_space.x_size, self.grid_space.y_size))

        for i_x in range(self.grid_space.x_size):
            for i_y in range(self.grid_space.y_size):
                c_x, c_y = self.grid_space.get_center_point(i_x, i_y)
                x_image[i_x, i_y] = c_x
                y_image[i_x, i_y] = c_y

        # Normalize coordinate images
        x_max = self.grid_space.x_size * self.grid_space.x_unit + self.grid_space.x_min
        y_max = self.grid_space.y_size * self.grid_space.y_unit + self.grid_space.y_min

        if x_max != self.grid_space.x_min:
            x_image = (x_image - self.grid_space.x_min) / (x_max - self.grid_space.x_min)
        if y_max != self.grid_space.y_min:
            y_image = (y_image - self.grid_space.y_min) / (y_max - self.grid_space.y_min)

        # Stack channels
        grid_image = np.stack([x_image, y_image, traffic_image], axis=-1)

        return grid_image

    def _encode_trajectory(self, traj, dyna_feature_column):
        """
        Encode a single trajectory.

        Args:
            traj: Single trajectory (list of points)
            dyna_feature_column: Column mapping

        Returns:
            Dictionary containing encoded features
        """
        if len(traj) < 2:
            return None

        # Extract coordinates and times
        coordinates = []
        times = []
        road_ids = []
        road_types = []

        for point in traj:
            # Get coordinates
            coord = eval(point[dyna_feature_column["coordinates"]])
            lon, lat = float(coord[0]), float(coord[1])
            coordinates.append([lon, lat])

            # Get time
            time_str = point[dyna_feature_column["time"]]
            times.append(datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%SZ'))

            # Get road segment ID if available
            if "road_id" in dyna_feature_column:
                road_id = int(point[dyna_feature_column["road_id"]])
            elif "segment_id" in dyna_feature_column:
                road_id = int(point[dyna_feature_column["segment_id"]])
            else:
                # Use grid ID as pseudo-road ID
                if self.grid_space is not None:
                    road_id = self.grid_space.get_gridid_by_point(lon, lat)
                else:
                    road_id = 0
            road_ids.append(road_id)
            self.road_num = max(road_id + 1, self.road_num)

            # Get road type if available
            if "road_type" in dyna_feature_column:
                road_type = int(point[dyna_feature_column["road_type"]])
            elif "highway_type" in dyna_feature_column:
                road_type = int(point[dyna_feature_column["highway_type"]])
            else:
                road_type = 0
            road_types.append(road_type)
            self.road_type_num = max(road_type + 1, self.road_type_num)

        # Generate grid trajectory
        grid_traj = []
        if self.grid_space is not None:
            for lon, lat in coordinates:
                try:
                    grid_id = self.grid_space.get_gridid_by_point(lon, lat)
                    # Add offset for special tokens
                    grid_traj.append(grid_id + len(self.grid_special_tokens))
                except (AssertionError, ValueError):
                    grid_traj.append(self.grid_special_tokens['padding_token'])
        else:
            grid_traj = [0] * len(coordinates)

        # Generate grid features (x_norm, y_norm, distance, bearing)
        if self.grid_space is not None:
            grid_feature = generate_spatial_features(coordinates, self.grid_space)
        else:
            grid_feature = [[0.0, 0.0, 0.0, 0.0]] * len(coordinates)

        # Generate temporal embeddings
        grid_time_emb = []
        for t in times:
            emb = generate_time_embedding(t, self.hidden_emb_dim)
            grid_time_emb.append(emb.tolist())

        # Generate road trajectory with special token offset
        road_traj = [rid + len(self.road_special_tokens) for rid in road_ids]

        # Generate temporal features
        road_weeks = [(t.weekday() + 1) for t in times]  # 1-7, 0 for padding
        road_minutes = [(t.hour * 60 + t.minute + 1) for t in times]  # 1-1440, 0 for padding

        # Calculate travel time
        begin_time = times[0]
        end_time = times[-1]
        travel_time = (end_time - begin_time).total_seconds()
        self.time_list.append(travel_time)

        # Metadata
        traj_len = len(traj)
        traj_id = int(traj[-1][dyna_feature_column["traj_id"]]) if "traj_id" in dyna_feature_column else 0
        start_timestamp = int(begin_time.timestamp())

        return {
            'grid_traj': grid_traj,
            'grid_time_emb': grid_time_emb,
            'grid_feature': grid_feature,
            'road_traj': road_traj,
            'road_weeks': road_weeks,
            'road_minutes': road_minutes,
            'road_type': road_types,
            'travel_time': travel_time,
            'traj_len': traj_len,
            'traj_id': traj_id,
            'start_timestamp': start_timestamp,
        }

    def encode(self, uid, trajectories, dyna_feature_column):
        """
        Encode trajectories for MDTI model.

        Args:
            uid: User ID (not used but kept for interface compatibility)
            trajectories: List of trajectories, each is a list of points
            dyna_feature_column: Column indices for features in .dyna file

        Returns:
            List of encoded trajectories
        """
        # First pass: compute bounding box if not already done
        if not self.normalization_computed:
            self._compute_bounding_box(trajectories, dyna_feature_column)
            self._initialize_grid_space()

            # Build global grid image
            if self.grid_space is not None:
                self.global_grid_image = self._construct_grid_image(trajectories, dyna_feature_column)

            self.normalization_computed = True

        encoded_trajectories = []

        for traj in trajectories:
            encoded = self._encode_trajectory(traj, dyna_feature_column)
            if encoded is None:
                continue

            # NOTE: grid_image is NOT stored per-trajectory to save memory.
            # The global grid_image is stored once in data_feature and accessed by the model.

            # Get road network features (placeholder)
            # In practice, these should be loaded from preprocessed data
            if self.road_features is not None:
                g_input_feature = self.road_features.tolist()
            else:
                # Create placeholder road features
                g_fea_size = max(self.g_fea_size, 1)
                g_input_feature = np.random.randn(max(self.road_num, 1), g_fea_size).tolist()

            if self.road_edge_index is not None:
                g_edge_index = self.road_edge_index.tolist()
            else:
                # Create placeholder edge index (self-loops)
                num_roads = max(self.road_num, 1)
                g_edge_index = [list(range(num_roads)), list(range(num_roads))]

            # Format output according to feature_dict order (grid_image removed)
            encoded_trajectories.append([
                encoded['grid_traj'],                # grid_traj
                encoded['grid_time_emb'],            # grid_time_emb
                encoded['grid_feature'],             # grid_feature
                g_input_feature,                     # g_input_feature
                g_edge_index,                        # g_edge_index
                encoded['road_traj'],                # road_traj
                encoded['road_weeks'],               # road_weeks
                encoded['road_minutes'],             # road_minutes
                encoded['road_type'],                # road_type
                [encoded['travel_time']],            # travel_time
                [encoded['traj_len']],               # traj_len
                [encoded['traj_id']],                # traj_id
                [encoded['start_timestamp']],        # start_timestamp
            ])

        return encoded_trajectories

    def gen_data_feature(self):
        """
        Generate data features and padding items after encoding all trajectories.

        NOTE: The global grid_image is stored here in data_feature to avoid
        per-trajectory duplication which causes OOM issues.
        """
        # Padding items for variable-length sequences
        self.pad_item = {
            'grid_traj': self.grid_special_tokens['padding_token'],
            'grid_time_emb': [0.0] * self.hidden_emb_dim,
            'grid_feature': [0.0, 0.0, 0.0, 0.0],
            'road_traj': self.road_special_tokens['padding_token'],
            'road_weeks': 0,
            'road_minutes': 0,
            'road_type': 0,
        }

        # Data features for model initialization
        self.data_feature = {
            'traj_len_idx': self.traj_len_idx,
            'grid_H': self.grid_H or 1,
            'grid_W': self.grid_W or 1,
            'grid_num': (self.grid_H or 1) * (self.grid_W or 1),
            'road_num': max(self.road_num, 1),
            'road_type': max(self.road_type_num, 1),
            'g_fea_size': max(self.g_fea_size, 1),
            'hidden_emb_dim': self.hidden_emb_dim,
            'road_special_tokens': self.road_special_tokens,
            'grid_special_tokens': self.grid_special_tokens,
            'min_lon': self.min_lon,
            'max_lon': self.max_lon,
            'min_lat': self.min_lat,
            'max_lat': self.max_lat,
        }

        # Store global grid_image in data_feature (instead of per-trajectory)
        # This significantly reduces memory usage
        if self.global_grid_image is not None:
            self.data_feature['grid_image'] = self.global_grid_image.tolist()
            self._logger.info(f"Stored global grid_image in data_feature: "
                            f"shape {self.global_grid_image.shape}")
        else:
            # Fallback: create a minimal grid image
            grid_H = max(1, self.grid_H or 1)
            grid_W = max(1, self.grid_W or 1)
            self.data_feature['grid_image'] = np.zeros((grid_H, grid_W, 3)).tolist()
            self._logger.info(f"Created fallback grid_image: shape ({grid_H}, {grid_W}, 3)")

        # Add time statistics
        if len(self.time_list) > 0:
            self.data_feature['time_mean'] = np.mean(self.time_list)
            self.data_feature['time_std'] = np.std(self.time_list)
        else:
            self.data_feature['time_mean'] = 0.0
            self.data_feature['time_std'] = 1.0

    def gen_scalar_data_feature(self, train_data):
        """
        Generate scalar data features from training data.

        Args:
            train_data: List of encoded training trajectories

        Returns:
            Dictionary of scalar features for normalization
        """
        travel_times = []
        traj_lens = []

        scalar_feature_column = {}
        for i, key in enumerate(self.feature_dict):
            scalar_feature_column[key] = i

        for data in train_data:
            travel_times.extend(data[scalar_feature_column["travel_time"]])
            traj_lens.extend(data[scalar_feature_column["traj_len"]])

        scalar_data_feature = {
            'time_mean': np.mean(travel_times) if len(travel_times) > 0 else 0.0,
            'time_std': np.std(travel_times) if len(travel_times) > 0 else 1.0,
            'avg_traj_len': np.mean(traj_lens) if len(traj_lens) > 0 else 1,
            'min_lon': self.min_lon,
            'max_lon': self.max_lon,
            'min_lat': self.min_lat,
            'max_lat': self.max_lat,
            'grid_H': self.grid_H,
            'grid_W': self.grid_W,
            'road_num': self.road_num,
        }

        for k, v in scalar_data_feature.items():
            self._logger.info("{}: {}".format(k, v))

        return scalar_data_feature
