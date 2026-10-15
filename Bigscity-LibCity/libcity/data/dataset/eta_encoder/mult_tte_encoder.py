import os
import numpy as np
import pandas as pd
import random
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


class MultTTEEncoder(AbstractETAEncoder):
    """
    Encoder for MulT-TTE model.

    This encoder prepares data for the MulT-TTE model which requires:
    - Road link features: highway type, length, cumulative length
    - Temporal features: week, date (day of year), time (minute of day)
    - GPS coordinates
    - Link indices for BERT-based segment embedding (with masking)
    """

    def __init__(self, config):
        super().__init__(config)

        # Feature dictionary defines the output format
        self.feature_dict = {
            'links': 'float',  # [highway_type, length, cum_length, week, date, time, lon1, lat1, lon2, lat2]
            'lens': 'int',  # sequence length
            'linkindex': 'int',  # masked link indices for BERT
            'rawlinks': 'int',  # original link indices
            'encoder_attention_mask': 'int',  # attention mask for BERT
            'mask_label': 'int',  # labels for masked positions
            'time': 'float',  # ground truth travel time
            'traj_id': 'int',
            'start_timestamp': 'int',
        }

        self.traj_len_idx = 1  # Index of 'lens' in feature list

        parameters_str = ''
        need_cut = self.config.get("need_cut", False)
        self.parameter_list = parameter_list_cut if need_cut else parameter_list
        for key in self.parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'eta{}.json'.format(parameters_str))

        # Load road network data for link information
        self.dataset = self.config.get('dataset', '')
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)

        # Road link information (geo_id -> properties)
        self.link_info = {}

        # Try to load .geo file for link information
        try:
            path = "./raw_data/{}/{}.geo".format(self.dataset, self.geo_file)
            f_geo = pd.read_csv(path)
            self._logger.info("Loaded file " + self.geo_file + '.geo')

            for row in f_geo.itertuples():
                geo_id = getattr(row, 'geo_id')
                # Extract highway type (default to 0 if not available)
                highway_type = getattr(row, 'highway_type', 0) if hasattr(row, 'highway_type') else 0
                # Extract length (default to 0 if not available)
                length = getattr(row, 'length', 0) if hasattr(row, 'length') else 0

                self.link_info[geo_id] = {
                    'highway_type': highway_type,
                    'length': length,
                }
        except Exception as e:
            self._logger.warning(f"Failed to load .geo file: {e}. Will use default values.")

        # Mask rate for BERT-based segment prediction
        self.mask_rate = config.get('mask_rate', 0.4)
        self.vocab_size = config.get('vocab_size', 27300)
        self.mask_token_id = self.vocab_size  # Special token ID for masking

        self.uid_size = 0

    def encode(self, uid, trajectories, dyna_feature_column):
        """
        Encode trajectories for MulT-TTE model.

        For each trajectory point, we extract:
        - Highway type (from geo data or location ID)
        - Length and cumulative length
        - Week (0-6), date (day of year 1-366), time (minute of day 0-1440)
        - GPS coordinates (lon1, lat1, lon2, lat2)
        - Link index for BERT embedding
        """
        self.uid_size = max(uid + 1, self.uid_size)
        encoded_trajectories = []

        for traj in trajectories:
            # Parse temporal information
            begin_time = datetime.strptime(traj[0][dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')
            end_time = datetime.strptime(traj[-1][dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')

            weekid = int(begin_time.weekday())  # 0-6
            traj_len = len(traj)
            traj_id = int(traj[-1][dyna_feature_column["traj_id"]])
            start_timestamp = int(datetime.timestamp(begin_time))
            time = (end_time - begin_time).seconds

            # Feature arrays for this trajectory
            links_features = []  # [highway_type, length, cum_length, week, date, time, lon1, lat1, lon2, lat2]
            link_indices = []  # Original link indices

            cum_length = 0.0
            prev_longi, prev_lati = None, None

            for idx, point in enumerate(traj):
                # Get location/link ID
                location = point[dyna_feature_column["location"]] if "location" in dyna_feature_column else idx

                # Get coordinates
                coordinate = eval(point[dyna_feature_column["coordinates"]])
                longi, lati = float(coordinate[0]), float(coordinate[1])

                # Get highway type and length from link info
                if location in self.link_info:
                    highway_type = self.link_info[location]['highway_type']
                    length = self.link_info[location]['length']
                else:
                    # Default values if link info not available
                    highway_type = 0
                    # Calculate length from GPS if previous point exists
                    if prev_longi is not None and prev_lati is not None:
                        length = geo_distance(prev_longi, prev_lati, longi, lati)
                    else:
                        length = 0.0

                # Update cumulative length
                cum_length += length

                # Temporal features for this point
                point_time = datetime.strptime(point[dyna_feature_column["time"]], '%Y-%m-%dT%H:%M:%SZ')
                week = int(point_time.weekday())  # 0-6
                date = int(point_time.timetuple().tm_yday)  # Day of year (1-366)
                minute_of_day = int(point_time.hour) * 60 + int(point_time.minute)  # 0-1439

                # GPS representation (current and previous points)
                lon1 = prev_longi if prev_longi is not None else longi
                lat1 = prev_lati if prev_lati is not None else lati
                lon2 = longi
                lat2 = lati

                # Assemble features: [highway_type, length, cum_length, week, date, time, lon1, lat1, lon2, lat2]
                link_feature = [
                    highway_type,
                    length,
                    cum_length,
                    week,
                    date,
                    minute_of_day,
                    lon1,
                    lat1,
                    lon2,
                    lat2
                ]
                links_features.append(link_feature)

                # Link index (use location ID, capped at vocab_size)
                link_idx = int(location) if isinstance(location, (int, float)) else hash(location) % self.vocab_size
                link_idx = min(link_idx, self.vocab_size - 1)
                link_indices.append(link_idx)

                # Update previous coordinates
                prev_longi, prev_lati = longi, lati

            # Create masked version for BERT
            masked_link_indices, attention_mask, mask_labels = self._create_masked_sequence(link_indices)

            encoded_trajectories.append([
                links_features,  # List of feature vectors
                traj_len,  # Sequence length
                masked_link_indices,  # Masked link indices
                link_indices,  # Original link indices (raw)
                attention_mask,  # Attention mask
                mask_labels,  # Mask labels
                time,  # Ground truth travel time
                traj_id,
                start_timestamp,
            ])

        return encoded_trajectories

    def _create_masked_sequence(self, link_indices):
        """
        Create masked sequence for BERT-based segment prediction.

        Args:
            link_indices: Original link indices

        Returns:
            masked_indices: Indices with some positions masked
            attention_mask: 1 for valid positions, 0 for padding
            mask_labels: Original indices at masked positions, -100 elsewhere
        """
        seq_len = len(link_indices)

        # Create copies
        masked_indices = link_indices.copy()
        mask_labels = [-100] * seq_len  # -100 is ignored in BERT loss
        attention_mask = [1] * seq_len

        # Randomly mask positions
        num_to_mask = max(1, int(seq_len * self.mask_rate))
        mask_positions = random.sample(range(seq_len), num_to_mask)

        for pos in mask_positions:
            mask_labels[pos] = link_indices[pos]  # Store original value
            masked_indices[pos] = self.mask_token_id  # Replace with mask token

        return masked_indices, attention_mask, mask_labels

    def gen_data_feature(self):
        """Generate data features and padding configuration."""
        self.pad_item = {
            'links': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10-dimensional feature vector
            'linkindex': self.vocab_size + 1,  # Padding token ID
            'rawlinks': self.vocab_size + 1,
            'encoder_attention_mask': 0,
            'mask_label': -100,
        }

        self.data_feature = {
            'traj_len_idx': self.traj_len_idx,
            'uid_size': self.uid_size,
            'vocab_size': self.vocab_size,
            'pad_token_id': self.vocab_size + 1,
        }

    def gen_scalar_data_feature(self, train_data):
        """
        Generate scalar statistics from training data.
        Used for normalization in the model.
        """
        time_list = []
        length_list = []
        cum_length_list = []

        for data in train_data:
            # Extract time (index 6 in encoded data)
            time_list.extend([data[6]])

            # Extract lengths from links features (index 0)
            links = data[0]
            for link in links:
                length_list.append(link[1])  # length
                cum_length_list.append(link[2])  # cumulative length

        scalar_data_feature = {
            'time_mean': np.mean(time_list),
            'time_std': np.std(time_list),
            'length_mean': np.mean(length_list),
            'length_std': np.std(length_list),
            'cum_length_mean': np.mean(cum_length_list),
            'cum_length_std': np.std(cum_length_list),
        }

        for k, v in scalar_data_feature.items():
            self._logger.info("{}: {}".format(k, v))

        return scalar_data_feature
