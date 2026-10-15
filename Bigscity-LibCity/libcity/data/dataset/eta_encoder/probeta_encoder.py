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
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


class ProbETAEncoder(AbstractETAEncoder):
    """
    ProbETA Encoder for trajectory data.

    This encoder extracts road segment sequences from trajectories for ProbETA model.
    Unlike other ETA encoders that focus on GPS coordinates, this encoder extracts
    road segment IDs that the trajectory passes through.

    Expected features:
    - road_segments: Sequence of road segment IDs (required)
    - device_ids: Device identifier for similarity computation (optional)
    - time: Travel time label (required)
    - time_slots: Time slot IDs for temporal features (optional)
    """

    def __init__(self, config):
        super().__init__(config)
        self.feature_dict = {
            'road_segments': 'int',  # Road segment ID sequence
            'device_ids': 'int',     # Device ID for similarity
            'time_slots': 'int',     # Time slot ID
            'uid': 'int',            # User ID
            'weekid': 'int',         # Day of week
            'timeid': 'int',         # Time of day in minutes
            'time': 'float',         # Travel time (label)
            'traj_len': 'int',       # Trajectory length
            'traj_id': 'int',        # Trajectory ID
            'start_timestamp': 'int', # Start timestamp
            'current_longi': 'float',  # For visualization/debugging
            'current_lati': 'float',   # For visualization/debugging
        }
        self.traj_len_idx = len(self.feature_dict) - 1
        parameters_str = ''
        need_cut = self.config.get("need_cut", False)
        self.parameter_list = parameter_list_cut if need_cut else parameter_list
        for key in self.parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'probeta{}.json'.format(parameters_str))

        self.uid_size = 0
        self.road_num = 0

    def encode(self, uid, trajectories, dyna_feature_column):
        """
        Encode trajectories for ProbETA model.

        Args:
            uid: User ID
            trajectories: List of trajectories
            dyna_feature_column: Column mapping for dyna features

        Returns:
            List of encoded trajectories
        """
        self.uid_size = max(uid + 1, self.uid_size)
        encoded_trajectories = []

        for traj in trajectories:
            # Extract road segments
            road_segments = []
            current_longi = []
            current_lati = []

            begin_time = datetime.strptime(traj[0][dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')
            end_time = datetime.strptime(traj[-1][dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')
            weekid = int(begin_time.weekday())
            timeid = int(begin_time.strftime('%H')) * 60 + int(begin_time.strftime('%M'))
            time = (end_time - begin_time).seconds
            traj_len = len(traj)
            traj_id = int(traj[-1][dyna_feature_column["traj_id"]])
            start_timestamp = int(datetime.timestamp(begin_time))

            # Extract road segments from trajectory
            # ProbETA expects road segment IDs, which could be in 'location' or 'road_id' field
            for point in traj:
                # Try to get road segment ID from location field
                if "location" in dyna_feature_column:
                    road_id = int(point[dyna_feature_column["location"]])
                elif "road_id" in dyna_feature_column:
                    road_id = int(point[dyna_feature_column["road_id"]])
                else:
                    # Fallback: use location ID if available
                    # This assumes location IDs are road segment IDs
                    road_id = int(point[dyna_feature_column.get("location", 0)])

                road_segments.append(road_id)
                self.road_num = max(self.road_num, road_id + 1)

                # Also extract coordinates for compatibility
                if "coordinates" in dyna_feature_column:
                    coordinate = eval(point[dyna_feature_column["coordinates"]])
                    longi, lati = float(coordinate[0]), float(coordinate[1])
                    current_longi.append(longi)
                    current_lati.append(lati)
                else:
                    current_longi.append(0.0)
                    current_lati.append(0.0)

            # Device ID (if available, otherwise use uid)
            if "device_id" in dyna_feature_column:
                device_id = int(traj[0][dyna_feature_column["device_id"]])
            else:
                device_id = uid  # Use user ID as device ID

            # Time slot (if available)
            if "time_slot" in dyna_feature_column:
                time_slot = int(traj[0][dyna_feature_column["time_slot"]])
            else:
                # Compute time slot from hour (e.g., divide day into 48 30-min slots)
                time_slot = int(begin_time.hour * 2 + begin_time.minute // 30)

            encoded_trajectories.append([
                road_segments[:],      # road_segments
                [device_id],           # device_ids
                [time_slot],           # time_slots
                [uid],                 # uid
                [weekid],              # weekid
                [timeid],              # timeid
                [time],                # time (label)
                [traj_len],            # traj_len
                [traj_id],             # traj_id
                [start_timestamp],     # start_timestamp
                current_longi[:],      # current_longi
                current_lati[:],       # current_lati
            ])

        return encoded_trajectories

    def gen_data_feature(self):
        """Generate data feature dictionary for model initialization."""
        self.pad_item = {
            'road_segments': 0,     # Pad road segments with 0
            'current_longi': 0.0,
            'current_lati': 0.0,
        }
        self.data_feature = {
            'traj_len_idx': self.traj_len_idx,
            'uid_size': self.uid_size,
            'road_num': self.road_num,  # Important for ProbETA embedding layer
        }

    def gen_scalar_data_feature(self, train_data):
        """
        Generate scalar statistics from training data.

        Args:
            train_data: List of encoded trajectories

        Returns:
            Dictionary of scalar statistics
        """
        time_list = []
        road_freq = {}

        scalar_feature_column = {}
        for i, key in enumerate(self.feature_dict):
            scalar_feature_column[key] = i

        for data in train_data:
            # Collect travel times
            time_list.extend(data[scalar_feature_column["time"]])

            # Count road segment frequencies
            road_segments = data[scalar_feature_column["road_segments"]]
            for road_id in road_segments:
                road_freq[road_id] = road_freq.get(road_id, 0) + 1

        scalar_data_feature = {
            'time_mean': np.mean(time_list),
            'time_std': np.std(time_list),
            'time_min': np.min(time_list),
            'time_max': np.max(time_list),
            'num_unique_roads': len(road_freq),
            'avg_road_freq': np.mean(list(road_freq.values())) if road_freq else 0,
        }

        for k, v in scalar_data_feature.items():
            self._logger.info("{}: {}".format(k, v))

        return scalar_data_feature
