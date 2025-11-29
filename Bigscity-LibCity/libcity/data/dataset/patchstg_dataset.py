import os
import pandas as pd
from libcity.data.dataset import TrafficStatePointDataset


class PatchSTGDataset(TrafficStatePointDataset):
    """
    Custom dataset for PatchSTG model.
    Extends TrafficStatePointDataset to provide geo information for spatial patching.
    """

    def __init__(self, config):
        super().__init__(config)
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'patchstg_{}.npz'.format(self.parameters_str))

    def get_data_feature(self):
        """
        Returns dataset features including geo information for spatial patching.

        Returns:
            dict: Dictionary containing dataset features including geo data
        """
        # Get base features
        features = super().get_data_feature()

        # Add geo data for spatial patching
        geo_file_path = self.data_path + self.geo_file + '.geo'
        if os.path.exists(geo_file_path):
            geo_data = pd.read_csv(geo_file_path)
            features['geo'] = geo_data
            self._logger.info("Added geo information for PatchSTG spatial patching")
        else:
            features['geo'] = None
            self._logger.warning(f"Geo file not found at {geo_file_path}, "
                               "PatchSTG will use sequential patching")

        return features
