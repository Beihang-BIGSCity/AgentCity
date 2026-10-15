"""
DutyTTE Encoder for LibCity ETA task.

This encoder prepares trajectory data for the DutyTTE model, which requires:
1. Segment IDs for road segment embeddings
2. Segment travel time features (distribution information)
3. Origin-destination node IDs
4. Start time information
5. Number of segments in trajectory

The encoder extracts and encodes these features from raw trajectory data.
"""

import os
import numpy as np
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

from libcity.data.dataset.eta_encoder.abstract_eta_encoder import AbstractETAEncoder


parameter_list = [
    'dataset', 'eta_encoder',
]
parameter_list_cut = [
    'dataset', 'eta_encoder', 'cut_method', 'min_session_len', 'max_session_len', 'min_sessions', 'window_size',
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


class DutyTTEEncoder(AbstractETAEncoder):
    """
    Encoder for DutyTTE model.

    This encoder prepares trajectory data for the DutyTTE uncertainty quantification model.
    It extracts segment-level features and trajectory-level metadata.

    Required input features in .dyna file:
        - coordinates: GPS coordinates of trajectory points
        - time: Timestamp of each point
        - segment_id (optional): Road segment ID for each point
        - node_id (optional): Node/intersection ID

    Output features:
        - segments: Road segment IDs along the trajectory
        - segment_travel_time: Distribution features for each segment (m*2+1 dim)
        - num_segments: Number of segments in trajectory
        - start_time: Start time bucket (10-minute granularity)
        - origin: Origin node ID
        - destination: Destination node ID
        - time: Total travel time (ground truth)
    """

    def __init__(self, config):
        super().__init__(config)

        # Number of distribution parameters (histogram bins)
        self.m = config.get('m', 5)

        # Feature dictionary defines the output format
        self.feature_dict = {
            'segments': 'int',                    # Road segment IDs [seq_len]
            'segment_travel_time': 'float',       # Segment features [seq_len, m*2+1]
            'num_segments': 'int',                # Number of segments [1]
            'start_time': 'int',                  # Start time bucket [1]
            'origin': 'int',                      # Origin node ID [1]
            'destination': 'int',                 # Destination node ID [1]
            'time': 'float',                      # Total travel time [1]
            'traj_len': 'int',                    # Trajectory length [1]
            'traj_id': 'int',                     # Trajectory ID [1]
            'uid': 'int',                         # User ID [1]
        }

        self.traj_len_idx = len(self.feature_dict) - 3  # Index of traj_len

        # Build cache file name
        parameters_str = ''
        need_cut = self.config.get("need_cut", False)
        self.parameter_list = parameter_list_cut if need_cut else parameter_list
        for key in self.parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'duty_tte{}.json'.format(parameters_str))

        # Track statistics for data_feature
        self.uid_size = 0
        self.segment_size = 0
        self.node_size = 0
        self.time_list = []

    def _encode_segment_travel_time(self, travel_times, m):
        """
        Encode segment travel times into distribution features.

        For each segment, creates a feature vector of size m*2+1:
        - First m values: histogram of travel time distribution
        - Next m values: cumulative distribution
        - Last value: count/normalization factor

        Args:
            travel_times: List of travel times for each segment
            m: Number of histogram bins

        Returns:
            List of feature vectors [num_segments, m*2+1]
        """
        features = []
        for tt in travel_times:
            # Create a simple feature representation
            # In practice, this would be computed from historical data
            feature = np.zeros(m * 2 + 1)

            # Use travel time to create pseudo-histogram features
            # Normalize by a typical range (e.g., 0-600 seconds per segment)
            normalized_tt = min(tt / 600.0, 1.0)

            # Fill histogram bins based on travel time
            bin_idx = min(int(normalized_tt * m), m - 1)
            feature[bin_idx] = 1.0

            # Cumulative distribution
            for i in range(m):
                feature[m + i] = 1.0 if i >= bin_idx else 0.0

            # Count/normalization
            feature[-1] = tt

            features.append(feature.tolist())

        return features

    def encode(self, uid, trajectories, dyna_feature_column):
        """
        Encode trajectories for DutyTTE model.

        Args:
            uid: User ID
            trajectories: List of trajectories, each is a list of points
            dyna_feature_column: Column indices for features in .dyna file

        Returns:
            List of encoded trajectories
        """
        self.uid_size = max(uid + 1, self.uid_size)
        encoded_trajectories = []

        for traj in trajectories:
            if len(traj) < 2:
                continue

            # Extract coordinates and times
            coordinates = []
            times = []
            segment_ids = []
            node_ids = []

            for point in traj:
                # Get coordinates
                coord = eval(point[dyna_feature_column["coordinates"]])
                longi, lati = float(coord[0]), float(coord[1])
                coordinates.append((longi, lati))

                # Get time
                time_str = point[dyna_feature_column["time"]]
                times.append(datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%SZ'))

                # Get segment ID if available
                if "segment_id" in dyna_feature_column:
                    seg_id = int(point[dyna_feature_column["segment_id"]])
                else:
                    # Use a hash of coordinates as pseudo-segment ID
                    seg_id = hash((round(longi, 4), round(lati, 4))) % 10000
                segment_ids.append(seg_id)
                self.segment_size = max(seg_id + 1, self.segment_size)

                # Get node ID if available
                if "node_id" in dyna_feature_column:
                    node_id = int(point[dyna_feature_column["node_id"]])
                else:
                    # Use first/last point hash as node ID
                    node_id = hash((round(longi, 3), round(lati, 3))) % 5000
                node_ids.append(node_id)
                self.node_size = max(node_id + 1, self.node_size)

            # Calculate travel times between consecutive points
            segment_travel_times = []
            for i in range(1, len(times)):
                tt = (times[i] - times[i-1]).total_seconds()
                segment_travel_times.append(max(tt, 0.1))  # Avoid zero

            # Add a placeholder for the first segment
            if len(segment_travel_times) > 0:
                segment_travel_times = [segment_travel_times[0]] + segment_travel_times
            else:
                segment_travel_times = [1.0]

            # Encode segment travel time features
            segment_features = self._encode_segment_travel_time(segment_travel_times, self.m)

            # Calculate total travel time
            begin_time = times[0]
            end_time = times[-1]
            total_time = (end_time - begin_time).total_seconds()
            self.time_list.append(total_time)

            # Calculate start time bucket (10-minute granularity)
            # 144 buckets per day (24 * 60 / 10 = 144)
            hour = begin_time.hour
            minute = begin_time.minute
            start_time_bucket = (hour * 60 + minute) // 10

            # Get origin and destination
            origin = node_ids[0]
            destination = node_ids[-1]

            # Number of segments
            num_segments = len(segment_ids)

            # Trajectory metadata
            traj_len = len(traj)
            traj_id = int(traj[-1][dyna_feature_column["traj_id"]]) if "traj_id" in dyna_feature_column else 0

            # Create encoded trajectory
            encoded_trajectories.append([
                segment_ids,           # segments
                segment_features,      # segment_travel_time
                [num_segments],        # num_segments
                [start_time_bucket],   # start_time
                [origin],              # origin
                [destination],         # destination
                [total_time],          # time
                [traj_len],            # traj_len
                [traj_id],             # traj_id
                [uid],                 # uid
            ])

        return encoded_trajectories

    def gen_data_feature(self):
        """
        Generate data features and padding items after encoding all trajectories.
        """
        self.pad_item = {
            'segments': 0,
            'segment_travel_time': [0.0] * (self.m * 2 + 1),
        }

        self.data_feature = {
            'traj_len_idx': self.traj_len_idx,
            'uid_size': self.uid_size,
            'segment_dims': max(self.segment_size, 1),
            'node_dims': max(self.node_size, 1),
            'm': self.m,
        }

        # Add time statistics if available
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
        time_list = []
        num_segments_list = []

        scalar_feature_column = {}
        for i, key in enumerate(self.feature_dict):
            scalar_feature_column[key] = i

        for data in train_data:
            time_list.extend(data[scalar_feature_column["time"]])
            num_segments_list.extend(data[scalar_feature_column["num_segments"]])

        scalar_data_feature = {
            'time_mean': np.mean(time_list) if len(time_list) > 0 else 0.0,
            'time_std': np.std(time_list) if len(time_list) > 0 else 1.0,
            'avg_num_segments': np.mean(num_segments_list) if len(num_segments_list) > 0 else 1,
        }

        for k, v in scalar_data_feature.items():
            self._logger.info("{}: {}".format(k, v))

        return scalar_data_feature
