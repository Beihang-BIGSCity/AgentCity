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


class MVSTMEncoder(AbstractETAEncoder):
    """
    Encoder for MVSTM (Multi-View Spatial-Temporal Model) for Travel Time Estimation.

    This encoder processes trajectories and extracts features required by the MVSTM model:
    - Link sequences: link_ids, link_time, link_ratio, link_current_status
    - Order-level features: dist, simple_eta, driver_id, slice_id, weekday, weather, temps
    - Target: travel time (eta)

    The encoder handles variable-length link sequences and provides all necessary
    normalization statistics for the model.
    """

    def __init__(self, config):
        super().__init__(config)
        self.feature_dict = {
            # Link sequence features
            'link_ids': 'int',
            'link_time': 'float',
            'link_ratio': 'float',
            'link_current_status': 'float',
            'link_len': 'float',
            # Order-level numerical features
            'dist': 'float',
            'simple_eta': 'float',
            'high_temp': 'float',
            'low_temp': 'float',
            # Order-level categorical features
            'driver_id': 'int',
            'slice_id': 'int',
            'weekday': 'int',
            'weather': 'int',
            # Target and metadata
            'eta': 'float',
            'time': 'float',
            'traj_len': 'int',
            'traj_id': 'int',
            'start_timestamp': 'int',
        }
        self.traj_len_idx = len(self.feature_dict) - 1
        parameters_str = ''
        need_cut = self.config.get("need_cut", False)
        self.parameter_list = parameter_list_cut if need_cut else parameter_list
        for key in self.parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'mvstm_eta{}.json'.format(parameters_str))

        self.uid_size = 0
        self.link_size = 0
        self.driver_size = 0

        # Link to ID mapping
        self.link2id = {}
        self.driver2id = {}

    def _get_link_id(self, link):
        """Get or create ID for a link."""
        link_str = str(link)
        if link_str not in self.link2id:
            self.link2id[link_str] = len(self.link2id) + 1  # 0 is reserved for padding/unknown
        return self.link2id[link_str]

    def _get_driver_id(self, driver):
        """Get or create ID for a driver."""
        driver_str = str(driver)
        if driver_str not in self.driver2id:
            self.driver2id[driver_str] = len(self.driver2id)
        return self.driver2id[driver_str]

    def encode(self, uid, trajectories, dyna_feature_column):
        self.uid_size = max(uid + 1, self.uid_size)
        encoded_trajectories = []

        for traj in trajectories:
            # Extract trajectory timestamps
            begin_time = datetime.strptime(traj[0][dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')
            end_time = datetime.strptime(traj[-1][dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')

            # Calculate temporal features
            weekday = int(begin_time.weekday())
            # Time slice: 5-minute intervals, 288 per day
            slice_id = (int(begin_time.strftime('%H')) * 60 + int(begin_time.strftime('%M'))) // 5
            travel_time = (end_time - begin_time).total_seconds()

            traj_len = len(traj)
            traj_id = int(traj[-1][dyna_feature_column["traj_id"]])
            start_timestamp = datetime.timestamp(begin_time)

            # Extract link sequence features
            link_ids = []
            link_time = []
            link_ratio = []
            link_current_status = []

            # Calculate total distance
            total_dist = 0
            last_longi, last_lati = None, None

            for i, point in enumerate(traj):
                # Get coordinates
                coordinate = eval(point[dyna_feature_column["coordinates"]])
                longi, lati = float(coordinate[0]), float(coordinate[1])

                # Calculate distance
                if last_longi is not None:
                    total_dist += geo_distance(last_longi, last_lati, longi, lati)
                last_longi, last_lati = longi, lati

                # Extract link_id if available
                if "link_id" in dyna_feature_column:
                    link_id = self._get_link_id(point[dyna_feature_column["link_id"]])
                else:
                    # Use grid-based ID as fallback
                    grid_id = int((longi * 1000) % 10000) * 10000 + int((lati * 1000) % 10000)
                    link_id = self._get_link_id(grid_id)
                link_ids.append(link_id)

                # Extract or compute link_time
                if "link_time" in dyna_feature_column:
                    lt = float(point[dyna_feature_column["link_time"]])
                elif i > 0:
                    prev_time = datetime.strptime(traj[i-1][dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')
                    curr_time = datetime.strptime(point[dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')
                    lt = (curr_time - prev_time).total_seconds()
                else:
                    lt = 0.0
                link_time.append(lt)

                # Extract or default link_ratio
                if "link_ratio" in dyna_feature_column:
                    lr = float(point[dyna_feature_column["link_ratio"]])
                else:
                    lr = 1.0
                link_ratio.append(lr)

                # Extract or default link_current_status
                if "link_current_status" in dyna_feature_column:
                    lcs = float(point[dyna_feature_column["link_current_status"]])
                elif "current_state" in dyna_feature_column:
                    lcs = float(point[dyna_feature_column["current_state"]])
                else:
                    lcs = 1.0
                link_current_status.append(lcs)

            link_len = len(link_ids)

            # Extract driver_id
            if "driver_id" in dyna_feature_column:
                driver_id = self._get_driver_id(traj[0][dyna_feature_column["driver_id"]])
            else:
                driver_id = self._get_driver_id(uid)

            # Total distance in km
            dist = total_dist if total_dist > 0 else 0.1

            # Simple ETA estimation (average speed assumption: 30 km/h)
            simple_eta = dist / 30 * 3600  # in seconds

            # Weather features (default to reasonable values if not available)
            if "weather" in dyna_feature_column:
                weather = int(traj[0][dyna_feature_column["weather"]])
            else:
                weather = 0  # default: clear

            if "hightemp" in dyna_feature_column:
                high_temp = float(traj[0][dyna_feature_column["hightemp"]])
            elif "high_temp" in dyna_feature_column:
                high_temp = float(traj[0][dyna_feature_column["high_temp"]])
            else:
                high_temp = 25.0  # default temperature

            if "lowtemp" in dyna_feature_column:
                low_temp = float(traj[0][dyna_feature_column["lowtemp"]])
            elif "low_temp" in dyna_feature_column:
                low_temp = float(traj[0][dyna_feature_column["low_temp"]])
            else:
                low_temp = 20.0  # default temperature

            encoded_trajectories.append([
                link_ids[:],
                link_time[:],
                link_ratio[:],
                link_current_status[:],
                [link_len],
                [dist],
                [simple_eta],
                [high_temp],
                [low_temp],
                [driver_id],
                [slice_id],
                [weekday],
                [weather],
                [travel_time],  # eta
                [travel_time],  # time (same as eta for this model)
                [traj_len],
                [traj_id],
                [start_timestamp],
            ])

        return encoded_trajectories

    def gen_data_feature(self):
        self.pad_item = {
            'link_ids': 0,
            'link_time': 0,
            'link_ratio': 0,
            'link_current_status': 0,
        }
        self.link_size = len(self.link2id) + 1  # +1 for padding
        self.driver_size = len(self.driver2id)
        self.data_feature = {
            'traj_len_idx': self.traj_len_idx,
            'uid_size': self.uid_size,
            'num_links': self.link_size,
            'num_drivers': self.driver_size if self.driver_size > 0 else self.uid_size,
        }

    def gen_scalar_data_feature(self, train_data):
        """
        Generate normalization statistics from training data.
        These are used by the MVSTM model for feature normalization.
        """
        dist_list = []
        time_list = []
        simple_eta_list = []
        high_temp_list = []
        low_temp_list = []
        link_time_list = []

        scalar_feature_column = {}
        for i, key in enumerate(self.feature_dict):
            scalar_feature_column[key] = i

        for data in train_data:
            # Collect order-level features
            dist_list.extend(data[scalar_feature_column["dist"]])
            time_list.extend(data[scalar_feature_column["time"]])
            simple_eta_list.extend(data[scalar_feature_column["simple_eta"]])
            high_temp_list.extend(data[scalar_feature_column["high_temp"]])
            low_temp_list.extend(data[scalar_feature_column["low_temp"]])

            # Collect link-level features
            link_time_list.extend(data[scalar_feature_column["link_time"]])

        # Log-transform distance and time for better normalization
        log_dist_list = [np.log(max(d, 0.001)) for d in dist_list]
        log_time_list = [np.log(max(t, 1.0)) for t in time_list]
        log_simple_eta_list = [np.log(max(e, 1.0)) for e in simple_eta_list]

        scalar_data_feature = {
            # Log-transformed statistics (matching original MVSTM)
            'eta_mean': np.mean(log_time_list),
            'eta_std': np.std(log_time_list),
            'dist_mean': np.mean(log_dist_list),
            'dist_std': np.std(log_dist_list),
            'simple_eta_mean': np.mean(log_simple_eta_list),
            'simple_eta_std': np.std(log_simple_eta_list),
            # Temperature statistics
            'high_temp_mean': np.mean(high_temp_list),
            'high_temp_std': np.std(high_temp_list) if np.std(high_temp_list) > 0 else 1.0,
            'low_temp_mean': np.mean(low_temp_list),
            'low_temp_std': np.std(low_temp_list) if np.std(low_temp_list) > 0 else 1.0,
            # Link time statistics
            'link_time_min': np.min(link_time_list) if link_time_list else 0.0,
            'link_time_max': np.max(link_time_list) if link_time_list else 3000.0,
            'link_time_mean': np.mean(link_time_list) if link_time_list else 10.0,
            'link_time_std': np.std(link_time_list) if link_time_list and np.std(link_time_list) > 0 else 10.0,
            # Raw statistics (for reference)
            'time_mean': np.mean(time_list),
            'time_std': np.std(time_list),
        }

        for k, v in scalar_data_feature.items():
            self._logger.info("{}: {}".format(k, v))

        return scalar_data_feature
