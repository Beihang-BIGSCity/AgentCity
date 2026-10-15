import os
import numpy as np
from datetime import datetime

from libcity.data.dataset.eta_encoder.abstract_eta_encoder import AbstractETAEncoder


parameter_list = [
    'dataset', 'eta_encoder', 'split',
]
parameter_list_cut = [
    'dataset', 'eta_encoder', 'split', 'cut_method', 'min_session_len', 'max_session_len', 'min_sessions', 'window_size',
]


class DOTEncoder(AbstractETAEncoder):
    """
    DOT (Deep OD-Time) Encoder for ETA prediction.

    This encoder converts GPS trajectory sequences into Pixelated Trajectory (PiT) images
    as described in the DOT paper. The PiT representation consists of 3 channels:
    - Channel 0: Binary mask (1 if cell visited, -1 otherwise)
    - Channel 1: Normalized daytime (time of day when cell was first visited)
    - Channel 2: Normalized time offset (elapsed time fraction when cell was first visited)

    Additionally, it extracts OD-Time features:
    - Origin coordinates (lng_norm, lat_norm)
    - Destination coordinates (lng_norm, lat_norm)
    - Departure time (daytime_norm)

    Reference:
        Original implementation: repos/DOT/dataset.py (TrajectoryDataset class)
    """

    def __init__(self, config):
        super().__init__(config)

        # Grid split parameter (default 20x20 grid)
        self.split = config.get('split', 20)

        # Feature dictionary defining output format
        # The order here must match the order of features in encode() output
        self.feature_dict = {
            'images': 'no_pad_float',   # Shape: (3, split, split) - PiT image (fixed size, no padding)
            'odt': 'no_pad_float',      # Shape: (5,) - ODT features (fixed size, no padding)
            'time': 'float',            # Travel time in minutes
            'traj_len': 'int',          # Number of points in trajectory
            'traj_id': 'int',           # Trajectory ID
            'start_timestamp': 'int',   # Start timestamp
        }
        self.traj_len_idx = len(self.feature_dict) - 1

        # Build cache file name
        parameters_str = ''
        need_cut = self.config.get("need_cut", False)
        self.parameter_list = parameter_list_cut if need_cut else parameter_list
        for key in self.parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'eta{}.json'.format(parameters_str))

        # Bounding box statistics (will be computed during encoding)
        self.lng_min = float('inf')
        self.lng_max = float('-inf')
        self.lat_min = float('inf')
        self.lat_max = float('-inf')

        # Collect all trajectories for normalization (first pass)
        self.all_lngs = []
        self.all_lats = []
        self.normalization_computed = False

    def _compute_normalization(self, trajectories, dyna_feature_column):
        """
        Compute bounding box normalization parameters from trajectory data.

        Args:
            trajectories: List of trajectory data
            dyna_feature_column: Column mapping for trajectory features
        """
        for traj in trajectories:
            for point in traj:
                coordinate = eval(point[dyna_feature_column["coordinates"]])
                lng, lat = float(coordinate[0]), float(coordinate[1])
                self.lng_min = min(self.lng_min, lng)
                self.lng_max = max(self.lng_max, lng)
                self.lat_min = min(self.lat_min, lat)
                self.lat_max = max(self.lat_max, lat)
                self.all_lngs.append(lng)
                self.all_lats.append(lat)

    def _normalize_coord(self, value, min_val, max_val):
        """
        Normalize coordinate to [-1, 1] range.

        Args:
            value: Raw coordinate value
            min_val: Minimum value in dataset
            max_val: Maximum value in dataset

        Returns:
            Normalized value in [-1, 1] range
        """
        if max_val == min_val:
            return 0.0
        return ((value - min_val) / (max_val - min_val)) * 2 - 1

    def _compute_cell_index(self, lng, lat):
        """
        Compute the cell index in the grid for a given coordinate.

        Args:
            lng: Longitude value
            lat: Latitude value

        Returns:
            Cell index (integer) in the flattened grid
        """
        if self.lng_max == self.lng_min:
            x_index = 0
        else:
            x_index = int(round((lng - self.lng_min) /
                                ((self.lng_max - self.lng_min) / (self.split - 1))))

        if self.lat_max == self.lat_min:
            y_index = 0
        else:
            y_index = int(round((lat - self.lat_min) /
                                ((self.lat_max - self.lat_min) / (self.split - 1))))

        # Clamp to valid range
        x_index = max(0, min(x_index, self.split - 1))
        y_index = max(0, min(y_index, self.split - 1))

        return y_index * self.split + x_index

    def _to_image(self, traj, dyna_feature_column):
        """
        Convert a trajectory to a Pixelated Trajectory (PiT) image.

        This method implements the core PiT generation algorithm from the DOT paper.

        Args:
            traj: Single trajectory (list of spatiotemporal points)
            dyna_feature_column: Column mapping for trajectory features

        Returns:
            image: numpy array of shape (3, split, split)
            odt: numpy array of shape (5,) - OD-Time features
            travel_time: float - travel time in minutes
        """
        if len(traj) < 2:
            # Handle edge case: trajectory too short
            # Return empty image and zero features
            image = np.ones((3, self.split, self.split)) * -1
            odt = np.zeros(5)
            return image, odt, 0.0

        # Extract coordinates and timestamps
        points = []
        for point in traj:
            coordinate = eval(point[dyna_feature_column["coordinates"]])
            lng, lat = float(coordinate[0]), float(coordinate[1])
            time_str = point[dyna_feature_column["time"]]
            timestamp = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%SZ')
            time_seconds = timestamp.timestamp()
            points.append({
                'lng': lng,
                'lat': lat,
                'time': time_seconds,
                'timestamp': timestamp
            })

        # Compute normalized coordinates
        for p in points:
            p['lng_norm'] = self._normalize_coord(p['lng'], self.lng_min, self.lng_max)
            p['lat_norm'] = self._normalize_coord(p['lat'], self.lat_min, self.lat_max)

        # Compute daytime (time of day) normalized to [-1, 1]
        for p in points:
            daytime_seconds = p['time'] % (24 * 60 * 60)  # Seconds since midnight
            p['daytime_norm'] = (daytime_seconds / (24 * 60 * 60)) * 2 - 1

        # Compute time offset normalized to [-1, 1]
        start_time = points[0]['time']
        end_time = points[-1]['time']
        duration = end_time - start_time

        if duration <= 0:
            # All points have same timestamp, assign offset = 0
            for p in points:
                p['offset'] = 0.0
        else:
            for p in points:
                p['offset'] = ((p['time'] - start_time) / duration) * 2 - 1

        # Compute cell indices
        for p in points:
            p['cell_index'] = self._compute_cell_index(p['lng'], p['lat'])

        # Remove consecutive duplicates (same cell)
        cell_points = []
        prev_cell = None
        for p in points:
            if p['cell_index'] != prev_cell:
                cell_points.append(p)
                prev_cell = p['cell_index']

        # Reverse to ensure first visit values are used (following original implementation)
        cell_points = cell_points[::-1]

        # Create 3-channel image
        image = np.ones((3, self.split * self.split)) * -1  # Initialize with -1

        for p in cell_points:
            idx = p['cell_index']
            image[0, idx] = 1                    # Binary mask
            image[1, idx] = p['daytime_norm']    # Normalized daytime
            image[2, idx] = p['offset']          # Normalized time offset

        # Reshape to (C, H, W)
        image = image.reshape(3, self.split, self.split)

        # OD-Time features: [o_lng, o_lat, d_lng, d_lat, daytime]
        odt = np.array([
            points[0]['lng_norm'],
            points[0]['lat_norm'],
            points[-1]['lng_norm'],
            points[-1]['lat_norm'],
            points[0]['daytime_norm']
        ])

        # Travel time in minutes
        travel_time = duration

        return image, odt, travel_time

    def encode(self, uid, trajectories, dyna_feature_column):
        """
        Encode trajectories of user uid.

        Args:
            uid: User ID (not used in DOT but kept for interface compatibility)
            trajectories: List of trajectories, each trajectory is a list of points
            dyna_feature_column: Column mapping for trajectory features

        Returns:
            List of encoded trajectories, each containing:
            [images, odt, time, traj_len, traj_id, start_timestamp]
        """
        # First pass: collect all coordinates for normalization
        if not self.normalization_computed:
            self._compute_normalization(trajectories, dyna_feature_column)

        encoded_trajectories = []

        for traj in trajectories:
            # Convert trajectory to PiT image
            image, odt, travel_time = self._to_image(traj, dyna_feature_column)

            # Extract metadata
            traj_len = len(traj)
            traj_id = int(traj[-1][dyna_feature_column["traj_id"]])

            begin_time_str = traj[0][dyna_feature_column["time"]]
            begin_time = datetime.strptime(begin_time_str, '%Y-%m-%dT%H:%M:%SZ')
            start_timestamp = int(datetime.timestamp(begin_time))

            # Format output according to feature_dict order
            # Convert numpy arrays to lists for JSON serialization compatibility
            encoded_trajectories.append([
                image.tolist(),     # images: (3, split, split) -> nested list
                odt.tolist(),       # odt: (5,) -> list
                [travel_time],      # time: travel time in minutes
                [traj_len],         # traj_len
                [traj_id],          # traj_id
                [start_timestamp],  # start_timestamp
            ])

        return encoded_trajectories

    def gen_data_feature(self):
        """
        Generate data features after encoding all trajectories.

        Sets pad_item for features that need padding and data_feature
        for model initialization.
        """
        # PiT images and ODT don't need padding as they have fixed size
        # Other scalar features wrapped in lists don't need padding
        self.pad_item = {}

        # Data features for model initialization
        self.data_feature = {
            'traj_len_idx': self.traj_len_idx,
            'split': self.split,
            'num_channels': 3,
            'odt_size': 5,
            'lng_min': self.lng_min,
            'lng_max': self.lng_max,
            'lat_min': self.lat_min,
            'lat_max': self.lat_max,
        }

        # Mark normalization as computed
        self.normalization_computed = True

    def gen_scalar_data_feature(self, train_data):
        """
        Generate scalar normalization statistics from training data.

        Args:
            train_data: List of encoded training trajectories

        Returns:
            Dictionary containing normalization statistics
        """
        travel_times = []
        scalar_feature_column = {}
        for i, key in enumerate(self.feature_dict):
            scalar_feature_column[key] = i

        for data in train_data:
            travel_times.extend(data[scalar_feature_column["time"]])

        scalar_data_feature = {
            'time_mean': np.mean(travel_times) if travel_times else 0.0,
            'time_std': np.std(travel_times) if travel_times else 1.0,
            'lng_min': self.lng_min,
            'lng_max': self.lng_max,
            'lat_min': self.lat_min,
            'lat_max': self.lat_max,
        }

        for k, v in scalar_data_feature.items():
            self._logger.info("{}: {}".format(k, v))

        return scalar_data_feature
