"""
LightPath Encoder for ETA Task

This encoder prepares trajectory data for the LightPath model.
It extracts road segment IDs, timestamps, and travel time information
from trajectory data.
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
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = tuple(map(lambda x: radians(x), (lon1, lat1, lon2, lat2)))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


class LightPathEncoder(AbstractETAEncoder):
    """
    Encoder for LightPath model.

    This encoder extracts:
    - road_segments: Sequence of road segment IDs
    - timestamps: Sequence of time indices (minute of day)
    - time: Total travel time in seconds
    - lens: Actual sequence length
    - Additional features for model training
    """

    def __init__(self, config):
        super().__init__(config)
        self.feature_dict = {
            'road_segments': 'int',    # Road segment IDs
            'timestamps': 'int',        # Time indices (minute of day)
            'uid': 'int',               # User ID
            'weekid': 'int',            # Day of week
            'timeid': 'int',            # Minute of day at start
            'dist': 'float',            # Total distance
            'time': 'float',            # Total travel time (target)
            'lens': 'int',              # Actual trajectory length
            'traj_id': 'int',           # Trajectory ID
            'start_timestamp': 'int',   # Start timestamp
        }
        self.traj_len_idx = len(self.feature_dict) - 3  # Index of 'lens'

        parameters_str = ''
        need_cut = self.config.get("need_cut", False)
        self.parameter_list = parameter_list_cut if need_cut else parameter_list
        for key in self.parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'eta_lightpath{}.json'.format(parameters_str))

        self.uid_size = 0
        self.vocab_size = 0  # Max road segment ID + 1
        self.time_size = 1440  # Minutes in a day
        self.max_seq_len = config.get('max_seq_len', 100)

    def encode(self, uid, trajectories, dyna_feature_column):
        """
        Encode trajectories for LightPath model.

        Args:
            uid: User ID
            trajectories: List of trajectories
            dyna_feature_column: Column indices for features

        Returns:
            List of encoded trajectories
        """
        self.uid_size = max(uid + 1, self.uid_size)
        encoded_trajectories = []

        for traj in trajectories:
            road_segments = []
            timestamps = []

            begin_time = datetime.strptime(traj[0][dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')
            end_time = datetime.strptime(traj[-1][dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')
            weekid = int(begin_time.weekday())
            timeid = int(begin_time.strftime('%H')) * 60 + int(begin_time.strftime('%M'))
            time = (end_time - begin_time).seconds
            print(time)
            traj_len = len(traj)
            traj_id = int(traj[-1][dyna_feature_column["traj_id"]])
            start_timestamp = datetime.timestamp(begin_time)

            # Track distance
            total_dist = 0
            prev_longi, prev_lati = None, None

            for point in traj:
                # Extract road segment ID if available
                if "road_segment" in dyna_feature_column:
                    road_seg = int(point[dyna_feature_column["road_segment"]])
                elif "location" in dyna_feature_column:
                    road_seg = int(point[dyna_feature_column["location"]])
                else:
                    # Use a hash of coordinates as fallback
                    coordinate = eval(point[dyna_feature_column["coordinates"]])
                    road_seg = hash((round(coordinate[0], 4), round(coordinate[1], 4))) % 90000

                road_segments.append(road_seg)
                self.vocab_size = max(road_seg + 1, self.vocab_size)

                # Extract time index (minute of day)
                point_time = datetime.strptime(point[dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')
                minute_of_day = int(point_time.strftime('%H')) * 60 + int(point_time.strftime('%M'))
                timestamps.append(minute_of_day)

                # Calculate cumulative distance
                if "coordinates" in dyna_feature_column:
                    coordinate = eval(point[dyna_feature_column["coordinates"]])
                    longi, lati = float(coordinate[0]), float(coordinate[1])
                    if prev_longi is not None:
                        total_dist += geo_distance(prev_longi, prev_lati, longi, lati)
                    prev_longi, prev_lati = longi, lati

            # Use distance from data if available
            if "current_dis" in dyna_feature_column:
                total_dist = float(traj[-1][dyna_feature_column["current_dis"]] -
                                   traj[0][dyna_feature_column["current_dis"]])

            # Truncate if needed
            if len(road_segments) > self.max_seq_len:
                road_segments = road_segments[:self.max_seq_len]
                timestamps = timestamps[:self.max_seq_len]
                traj_len = self.max_seq_len

            encoded_trajectories.append([
                road_segments[:],
                timestamps[:],
                [uid],
                [weekid],
                [timeid],
                [total_dist],
                [time],
                [traj_len],
                [traj_id],
                [int(start_timestamp)],
            ])

        return encoded_trajectories

    def gen_data_feature(self):
        """Generate data features and padding items."""
        self.pad_item = {
            'road_segments': 0,
            'timestamps': 0,
        }
        self.data_feature = {
            'traj_len_idx': self.traj_len_idx,
            'uid_size': self.uid_size,
            'vocab_size': max(self.vocab_size, 90000),  # Minimum vocabulary size
            'time_size': self.time_size,
        }

    def gen_scalar_data_feature(self, train_data):
        """
        Generate scalar data features from training data.

        Args:
            train_data: List of encoded training trajectories

        Returns:
            Dictionary of scalar statistics
        """
        dist_list = []
        time_list = []
        lens_list = []

        scalar_feature_column = {}
        for i, key in enumerate(self.feature_dict):
            scalar_feature_column[key] = i

        for data in train_data:
            dist_list.extend(data[scalar_feature_column["dist"]])
            time_list.extend(data[scalar_feature_column["time"]])
            lens_list.extend(data[scalar_feature_column["lens"]])

        scalar_data_feature = {
            'dist_mean': np.mean(dist_list),
            'dist_std': np.std(dist_list) if np.std(dist_list) > 0 else 1.0,
            'time_mean': np.mean(time_list),
            'time_std': np.std(time_list) if np.std(time_list) > 0 else 1.0,
            'avg_seq_len': np.mean(lens_list),
            'max_seq_len': max(lens_list) if lens_list else self.max_seq_len,
        }

        for k, v in scalar_data_feature.items():
            self._logger.info("{}: {}".format(k, v))

        return scalar_data_feature
