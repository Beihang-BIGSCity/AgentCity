"""
HierETA Encoder for LibCity ETA task.

This encoder prepares trajectory data for the HierETA (Hierarchical ETA) model,
which requires a hierarchical representation of routes:
    Route -> Links (road segments between intersections) -> Segments (sub-parts of links)

Reference:
    Original HierETA data loading: repos/HierETA/dataloading.py
    LibCity model: Bigscity-LibCity/libcity/model/eta/HierETA.py

Key adaptations:
    - Implements LibCity's AbstractETAEncoder interface
    - Generates hierarchical route structure from flat trajectory data
    - Creates segment and link features with proper masking
    - Handles variable-length sequences with padding

IMPORTANT SIMPLIFICATIONS:
    Since LibCity ETA datasets don't have pre-computed hierarchical link/segment structure,
    this encoder creates a simplified hierarchy:
    - Each trajectory point becomes a "segment"
    - Groups of consecutive segments form "links"
    - Link boundaries are created every N segments (configurable)
    - Road network attributes are placeholder values (to be enhanced with real data)
"""

import os
import numpy as np
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

from libcity.data.dataset.eta_encoder.abstract_eta_encoder import AbstractETAEncoder


parameter_list = [
    'dataset', 'eta_encoder', 'link_num', 'segment_num', 'segments_per_link',
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


class HierETAEncoder(AbstractETAEncoder):
    """
    Encoder for HierETA (Hierarchical ETA) model.

    This encoder transforms flat trajectory data into HierETA's hierarchical format
    with external features, segment features, and link (crossing) features.

    Required config parameters:
        - link_num: Maximum number of links per route (default: 31)
        - segment_num: Maximum number of segments per link (default: 50)
        - segments_per_link: How many trajectory points form one link (default: 10)

    Output features (matching HierETA model requirements):
        External features (route-level):
            - weekID: Day of week (0-6)
            - timeID: Minute of day (0-1439)
            - driverID: Driver ID (placeholder, using entity_id)

        Segment features (point-level, reshaped to [batch, link_num*segment_num]):
            - segID: Segment ID (using trajectory point index as proxy)
            - segment_functional_level: Road functional level (placeholder 0-8)
            - roadState: Road traffic state (placeholder 0-5)
            - laneNum: Number of lanes (placeholder 0-6)
            - roadLevel: Road level (placeholder 0-7)
            - wid: Road width (placeholder, normalized distance)
            - speedLimit: Speed limit (placeholder, derived from distance/time)
            - time: Time spent on segment (seconds)
            - len: Segment length (meters)

        Link features (intersection-level, [batch, link_num]):
            - crossID: Crossing/intersection ID (placeholder)
            - delayTime: Delay at crossing (placeholder, using time gaps)

        Masks and metadata:
            - road_segment_mask: Mask for valid segments
            - road_link_mask: Mask for valid links
            - link_seg_lens: Number of segments in each link
            - link_lens: Number of links in route
            - gt_eta_time: Ground truth travel time (normalized)
    """

    def __init__(self, config):
        super().__init__(config)

        # Hierarchical structure configuration
        self.link_num = config.get('link_num', 31)
        self.segment_num = config.get('segment_num', 50)
        self.segments_per_link = config.get('segments_per_link', 10)
        self.max_segments_total = self.link_num * self.segment_num

        # Feature dictionary defining output format
        # Order MUST match the order in encode() output tuple
        self.feature_dict = {
            # External features (route-level, scalar per batch)
            'weekID': 'int',
            'timeID': 'int',
            'driverID': 'int',

            # Segment features (sequence, padded to link_num*segment_num)
            'segID': 'int',
            'segment_functional_level': 'int',
            'roadState': 'int',
            'laneNum': 'int',
            'roadLevel': 'int',
            'wid': 'float',
            'speedLimit': 'float',
            'time': 'float',
            'len': 'float',

            # Link features (sequence, padded to link_num)
            'crossID': 'int',
            'delayTime': 'float',

            # Masks and metadata
            'road_segment_mask': 'float',
            'road_link_mask': 'float',
            'link_seg_lens': 'int',
            'link_lens': 'int',
            'gt_eta_time': 'float',

            # Auxiliary
            'traj_id': 'int',
            'start_timestamp': 'int',
        }

        # Index of 'link_lens' in feature_dict (used by ETADataset for trajectory length sorting)
        self.traj_len_idx = 17

        # Build cache file name
        parameters_str = ''
        for key in parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'hiereta_eta{}.json'.format(parameters_str))

        # Statistics tracking
        self.uid_size = 0
        self.seg_id_size = 0
        self.cross_id_size = 0
        self.time_list = []

    def _create_hierarchical_structure(self, traj, dyna_feature_column):
        """
        Create hierarchical link->segment structure from flat trajectory.

        Args:
            traj: Flat trajectory (list of points)
            dyna_feature_column: Column mapping

        Returns:
            dict with hierarchical features
        """
        traj_len = len(traj)

        # Extract basic features from trajectory points
        coordinates = []
        times = []
        segments = []

        for idx, point in enumerate(traj):
            # Get coordinates
            coord = eval(point[dyna_feature_column["coordinates"]])
            lon, lat = float(coord[0]), float(coord[1])
            coordinates.append([lon, lat])

            # Get time
            time_str = point[dyna_feature_column["time"]]
            times.append(datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%SZ'))

            # Segment ID: use a hash of coordinates or just point index
            # In real scenario, this should come from map-matching to road network
            seg_id = idx  # Simplified: just use index
            segments.append(seg_id)
            self.seg_id_size = max(self.seg_id_size, seg_id + 1)

        # Create hierarchical structure by grouping segments into links
        # Each link contains up to segments_per_link consecutive segments
        num_links = min((traj_len + self.segments_per_link - 1) // self.segments_per_link, self.link_num)

        # Initialize hierarchical arrays
        link_seg_lens = []
        segment_features = {
            'segID': [],
            'segment_functional_level': [],
            'roadState': [],
            'laneNum': [],
            'roadLevel': [],
            'wid': [],
            'speedLimit': [],
            'time': [],
            'len': [],
        }
        cross_features = {
            'crossID': [],
            'delayTime': [],
        }

        # Process each link
        seg_idx = 0
        for link_idx in range(num_links):
            # Determine how many segments in this link
            start_seg = link_idx * self.segments_per_link
            end_seg = min(start_seg + self.segments_per_link, traj_len)
            num_segs_in_link = end_seg - start_seg
            link_seg_lens.append(num_segs_in_link)

            # Process segments in this link
            for i in range(start_seg, end_seg):
                # Segment ID
                segment_features['segID'].append(segments[i] + 1)  # +1 to reserve 0 for padding

                # Placeholder categorical features (would come from road network in real scenario)
                segment_features['segment_functional_level'].append(np.random.randint(1, 9))
                segment_features['roadState'].append(np.random.randint(1, 6))
                segment_features['laneNum'].append(np.random.randint(1, 7))
                segment_features['roadLevel'].append(np.random.randint(1, 8))

                # Calculate segment length
                if i < traj_len - 1:
                    seg_len = geo_distance(
                        coordinates[i][0], coordinates[i][1],
                        coordinates[i + 1][0], coordinates[i + 1][1]
                    ) * 1000  # Convert km to meters
                else:
                    seg_len = 0.0
                segment_features['len'].append(seg_len)

                # Calculate time on segment
                if i < traj_len - 1:
                    time_delta = (times[i + 1] - times[i]).total_seconds()
                else:
                    time_delta = 0.0
                segment_features['time'].append(time_delta)

                # Placeholder continuous features
                segment_features['wid'].append(seg_len / 100.0)  # Normalized width

                # Speed limit (placeholder, derived from distance/time)
                if time_delta > 0:
                    speed = (seg_len / 1000.0) / (time_delta / 3600.0)  # km/h
                    speed_limit = min(max(speed * 1.2, 30.0), 120.0)
                else:
                    speed_limit = 50.0
                segment_features['speedLimit'].append(speed_limit)

            # Pad segments within this link to segment_num
            for _ in range(self.segment_num - num_segs_in_link):
                segment_features['segID'].append(0)
                segment_features['segment_functional_level'].append(0)
                segment_features['roadState'].append(0)
                segment_features['laneNum'].append(0)
                segment_features['roadLevel'].append(0)
                segment_features['wid'].append(0.0)
                segment_features['speedLimit'].append(0.0)
                segment_features['time'].append(0.0)
                segment_features['len'].append(0.0)

            # Create crossing/link feature (at the end of this link)
            cross_id = link_idx  # Simplified: use link index as crossing ID
            self.cross_id_size = max(self.cross_id_size, cross_id + 1)
            cross_features['crossID'].append(cross_id + 1)  # +1 to reserve 0 for padding

            # Delay time at crossing (placeholder, using time between links)
            if link_idx < num_links - 1:
                next_link_start = (link_idx + 1) * self.segments_per_link
                if next_link_start < traj_len:
                    delay = (times[next_link_start] - times[end_seg - 1]).total_seconds()
                    cross_features['delayTime'].append(max(delay, 0.0))
                else:
                    cross_features['delayTime'].append(0.0)
            else:
                cross_features['delayTime'].append(0.0)

        # Pad links to link_num
        for _ in range(self.link_num - num_links):
            link_seg_lens.append(0)
            cross_features['crossID'].append(0)
            cross_features['delayTime'].append(0.0)

            # Pad segments for this empty link
            for _ in range(self.segment_num):
                segment_features['segID'].append(0)
                segment_features['segment_functional_level'].append(0)
                segment_features['roadState'].append(0)
                segment_features['laneNum'].append(0)
                segment_features['roadLevel'].append(0)
                segment_features['wid'].append(0.0)
                segment_features['speedLimit'].append(0.0)
                segment_features['time'].append(0.0)
                segment_features['len'].append(0.0)

        # Create masks
        segment_mask = [1.0 if seg_id > 0 else 0.0 for seg_id in segment_features['segID']]
        link_mask = [1.0 if num_segs > 0 else 0.0 for num_segs in link_seg_lens]

        return {
            'num_links': num_links,
            'link_seg_lens': link_seg_lens,
            'segment_features': segment_features,
            'cross_features': cross_features,
            'segment_mask': segment_mask,
            'link_mask': link_mask,
            'times': times,
            'coordinates': coordinates,
        }

    def encode(self, uid, trajectories, dyna_feature_column):
        """
        Encode trajectories for HierETA model.

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

            # Create hierarchical structure
            hier_data = self._create_hierarchical_structure(traj, dyna_feature_column)

            # Extract external features (route-level)
            begin_time = hier_data['times'][0]
            weekID = begin_time.weekday() + 1  # 1-7, 0 reserved for padding
            timeID = begin_time.hour * 60 + begin_time.minute + 1  # 1-1440, 0 reserved for padding
            driverID = uid + 1  # Use entity_id as driver_id, +1 to reserve 0

            # Calculate ground truth ETA
            end_time = hier_data['times'][-1]
            gt_eta_time = (end_time - begin_time).total_seconds()
            self.time_list.append(gt_eta_time)

            # Metadata
            traj_id = int(traj[-1][dyna_feature_column["traj_id"]]) if "traj_id" in dyna_feature_column else 0
            start_timestamp = int(begin_time.timestamp())

            # Format output according to feature_dict order
            encoded_trajectories.append([
                # External features (scalars)
                [weekID],
                [timeID],
                [driverID],

                # Segment features (sequences of length link_num*segment_num)
                hier_data['segment_features']['segID'],
                hier_data['segment_features']['segment_functional_level'],
                hier_data['segment_features']['roadState'],
                hier_data['segment_features']['laneNum'],
                hier_data['segment_features']['roadLevel'],
                hier_data['segment_features']['wid'],
                hier_data['segment_features']['speedLimit'],
                hier_data['segment_features']['time'],
                hier_data['segment_features']['len'],

                # Link features (sequences of length link_num)
                hier_data['cross_features']['crossID'],
                hier_data['cross_features']['delayTime'],

                # Masks and metadata
                hier_data['segment_mask'],
                hier_data['link_mask'],
                hier_data['link_seg_lens'],
                [hier_data['num_links']],
                [gt_eta_time],

                # Auxiliary
                [traj_id],
                [start_timestamp],
            ])

        return encoded_trajectories

    def gen_data_feature(self):
        """
        Generate data features and padding items after encoding all trajectories.
        """
        # Padding items for variable-length sequences
        # Note: Most sequences are fixed-length (link_num*segment_num or link_num)
        # but we still define padding for consistency
        self.pad_item = {}  # No padding needed as sequences are pre-padded

        # Data features for model initialization
        self.data_feature = {
            'uid_size': self.uid_size,
            'seg_vocab_size': max(self.seg_id_size + 1, 100),  # +1 for the +1 offset applied to IDs
            'cross_vocab_size': max(self.cross_id_size + 1, 100),  # +1 for the +1 offset applied to IDs
            'driver_vocab_size': max(self.uid_size + 1, 100),  # +1 for the +1 offset applied to IDs
            'link_num': self.link_num,
            'segment_num': self.segment_num,
            'traj_len_idx': self.traj_len_idx,
        }

        # Add normalization statistics
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

        # Build column mapping
        scalar_feature_column = {}
        for i, key in enumerate(self.feature_dict):
            scalar_feature_column[key] = i

        # Extract travel times
        for data in train_data:
            travel_times.extend(data[scalar_feature_column["gt_eta_time"]])

        # Compute statistics
        scalar_data_feature = {
            'time_mean': np.mean(travel_times) if len(travel_times) > 0 else 0.0,
            'time_std': np.std(travel_times) if len(travel_times) > 0 else 1.0,
            'link_num': self.link_num,
            'segment_num': self.segment_num,
            'traj_len_idx': self.traj_len_idx,
        }

        # Only include vocab sizes if they were actually computed (non-zero)
        # This prevents overwriting cached values when loading from cache
        if self.uid_size > 0:
            scalar_data_feature['uid_size'] = self.uid_size
            scalar_data_feature['driver_vocab_size'] = self.uid_size + 1  # +1 for offset
        if self.seg_id_size > 0:
            scalar_data_feature['seg_vocab_size'] = self.seg_id_size + 1  # +1 for offset
        if self.cross_id_size > 0:
            scalar_data_feature['cross_vocab_size'] = self.cross_id_size + 1  # +1 for offset

        # Add train_ prefix for normalization stats used by model
        scalar_data_feature['train_gt_eta_time_mean'] = scalar_data_feature['time_mean']
        scalar_data_feature['train_gt_eta_time_std'] = scalar_data_feature['time_std']

        for k, v in scalar_data_feature.items():
            self._logger.info("{}: {}".format(k, v))

        return scalar_data_feature
