# coding=utf-8
"""
DeepMMSeq2SeqDataset: A dataset class for DeepMM seq2seq map matching model.

This dataset provides tokenized GPS trajectory data and road segment data for
training seq2seq-based map matching models like DeepMM. Unlike MapMatchingDataset
which only provides test data (traditional algorithms don't need training),
this dataset provides train/val/test splits with proper tokenization.

Adapted from the original DeepMM data utilities:
    ./repos/DeepMM/DeepMM/data_utils.py

Key differences from MapMatchingDataset:
    - Provides train/val/test DataLoaders instead of (None, None, test_data)
    - Tokenizes GPS locations to grid cell IDs
    - Tokenizes road segments using geo_ids
    - Supports time encoding (optional)
    - Returns batches in the format expected by DeepMM model

Note: This is different from DeepMapMatchingDataset which is designed for
      GraphMM and similar graph-based map matching models.

Batch Format for DeepMM model:
    {
        'input_src': torch.LongTensor,      # Source location sequences [batch_size, seq_len]
        'input_trg': torch.LongTensor,      # Target segments for teacher forcing [batch_size, seq_len]
        'output_trg' or 'target': torch.LongTensor,  # Ground truth segments [batch_size, seq_len]
        'uid': torch.LongTensor,            # Dummy user IDs (batch indices) for TrajLocPredExecutor compatibility
    }
"""

import os
import pickle
import json
import operator
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from logging import getLogger

from libcity.data.dataset import AbstractDataset
from libcity.utils.utils import ensure_dir


class DeepMMBatch:
    """Batch wrapper for DeepMM that provides to_tensor() method for LibCity executor compatibility.

    This wrapper class holds pre-tensorized data from the collate_fn and provides
    the to_tensor() method expected by TrajLocPredExecutor. Since data is already
    in tensor format, to_tensor() simply moves tensors to the specified device.

    The class supports both dictionary-style access (batch['key']) and attribute-style
    access (batch.data['key']) for maximum compatibility with different model implementations.
    """

    def __init__(self, data_dict):
        """Initialize the batch wrapper.

        Args:
            data_dict: Dictionary containing tensor data with keys like
                       'input_src', 'input_trg', 'output_trg', 'target'
        """
        self.data = data_dict

    def __getitem__(self, key):
        """Enable dictionary-style access: batch['key']."""
        return self.data[key]

    def __setitem__(self, key, value):
        """Enable dictionary-style assignment: batch['key'] = value."""
        self.data[key] = value

    def __contains__(self, key):
        """Enable 'key in batch' checks."""
        return key in self.data

    def get(self, key, default=None):
        """Enable batch.get('key', default) access."""
        return self.data.get(key, default)

    def keys(self):
        """Return data keys."""
        return self.data.keys()

    def values(self):
        """Return data values."""
        return self.data.values()

    def items(self):
        """Return data items."""
        return self.data.items()

    def to_tensor(self, device):
        """Move all tensors to the specified device.

        This method is called by TrajLocPredExecutor before passing the batch
        to the model. Since the data is already in tensor format (from collate_fn),
        we just need to move each tensor to the target device.

        Args:
            device: torch.device or string specifying the target device (e.g., 'cuda:0', 'cpu')
        """
        for key in self.data:
            if isinstance(self.data[key], torch.Tensor):
                self.data[key] = self.data[key].to(device)


class DeepMMSeq2SeqTorchDataset(Dataset):
    """PyTorch Dataset wrapper for DeepMM data."""

    def __init__(self, src_data, trg_data, src_vocab, trg_vocab,
                 max_src_len=100, max_trg_len=100, time_encoding='NoEncoding',
                 time_data=None, time_vocab=None):
        """
        Args:
            src_data: List of source location sequences (GPS grid cell tokens)
            trg_data: List of target segment sequences (road segment tokens)
            src_vocab: Dict mapping location tokens to IDs
            trg_vocab: Dict mapping segment tokens to IDs
            max_src_len: Maximum source sequence length
            max_trg_len: Maximum target sequence length
            time_encoding: Time encoding method ('NoEncoding', 'OneEncoding', 'TwoEncoding')
            time_data: Time data for time encoding (optional)
            time_vocab: Time vocabulary for time encoding (optional)
        """
        self.src_data = src_data
        self.trg_data = trg_data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len
        self.time_encoding = time_encoding
        self.time_data = time_data
        self.time_vocab = time_vocab

        # Special tokens
        self.pad_id_src = src_vocab.get('<pad>', 1)
        self.pad_id_trg = trg_vocab.get('<pad>', 1)
        self.sos_id_trg = trg_vocab.get('<s>', 0)
        self.eos_id_trg = trg_vocab.get('</s>', 2)
        self.unk_id_src = src_vocab.get('<unk>', 3)
        self.unk_id_trg = trg_vocab.get('<unk>', 3)

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src_seq = self.src_data[idx]
        trg_seq = self.trg_data[idx]

        # Convert tokens to IDs with <s> and </s>
        src_ids = [self.src_vocab.get('<s>', 0)]
        for tok in src_seq:
            src_ids.append(self.src_vocab.get(tok, self.unk_id_src))
        src_ids.append(self.src_vocab.get('</s>', 2))

        trg_ids = [self.sos_id_trg]
        for tok in trg_seq:
            trg_ids.append(self.trg_vocab.get(tok, self.unk_id_trg))
        trg_ids.append(self.eos_id_trg)

        # Truncate to max length
        src_ids = src_ids[:self.max_src_len]
        trg_ids = trg_ids[:self.max_trg_len]

        # For teacher forcing: input is [<s>, token1, ..., tokenN]
        # output target is [token1, ..., tokenN, </s>]
        input_trg = trg_ids[:-1]
        output_trg = trg_ids[1:]

        return {
            'input_src': src_ids,
            'input_trg': input_trg,
            'output_trg': output_trg,
            'src_len': len(src_ids),
            'trg_len': len(output_trg)
        }


def collate_fn(batch):
    """Collate function for DeepMM DataLoader.

    Pads sequences to the same length within a batch.
    """
    # Get max lengths in this batch
    max_src_len = max(item['src_len'] for item in batch)
    max_trg_len = max(item['trg_len'] for item in batch)

    # Pad token IDs
    pad_id_src = 1  # <pad> token ID for source
    pad_id_trg = 1  # <pad> token ID for target

    batch_input_src = []
    batch_input_trg = []
    batch_output_trg = []

    for item in batch:
        # Pad source
        src_padded = item['input_src'] + [pad_id_src] * (max_src_len - item['src_len'])
        batch_input_src.append(src_padded)

        # Pad target input
        trg_in_padded = item['input_trg'] + [pad_id_trg] * (max_trg_len - item['trg_len'])
        batch_input_trg.append(trg_in_padded)

        # Pad target output
        trg_out_padded = item['output_trg'] + [pad_id_trg] * (max_trg_len - item['trg_len'])
        batch_output_trg.append(trg_out_padded)

    # Wrap the data dictionary in DeepMMBatch for LibCity executor compatibility
    # This provides the to_tensor() method expected by TrajLocPredExecutor
    batch_dict = {
        'input_src': torch.LongTensor(batch_input_src),
        'input_trg': torch.LongTensor(batch_input_trg),
        'output_trg': torch.LongTensor(batch_output_trg),
        'target': torch.LongTensor(batch_output_trg),  # Alias for compatibility
        'uid': torch.arange(len(batch)),  # Add dummy user IDs (batch indices) for TrajLocPredExecutor compatibility
    }
    return DeepMMBatch(batch_dict)


class DeepMMSeq2SeqDataset(AbstractDataset):
    """
    Dataset class for DeepMM seq2seq map matching model.

    This is specifically designed for DeepMM's Seq2Seq architecture and
    provides tokenized data in the format expected by the DeepMM model.

    Note: For GraphMM and similar graph-based models, use DeepMapMatchingDataset
          from libcity.data.dataset.deep_map_matching_dataset instead.

    This dataset provides:
    - Tokenized GPS locations (discretized to grid cells)
    - Tokenized road segments (using geo_ids from road network)
    - Train/validation/test DataLoaders
    - Vocabularies for source locations and target segments

    Data Format:
        Input batch contains:
        - input_src: Source location sequences [batch_size, src_seq_len]
        - input_trg: Target segments for teacher forcing [batch_size, trg_seq_len]
        - output_trg/target: Ground truth segments [batch_size, trg_seq_len]
    """

    def __init__(self, config):
        """
        Initialize the DeepMapMatchingDataset.

        Args:
            config: Configuration dict containing dataset parameters
        """
        self.config = config
        self.dataset = config.get('dataset', '')
        self._logger = getLogger()

        # Data paths
        self.data_path = './raw_data/' + self.dataset + '/'
        if not os.path.exists(self.data_path):
            raise ValueError("Dataset {} not exist! Please ensure the path "
                             "'./raw_data/{}/' exist!".format(self.dataset, self.dataset))

        # File names
        self.geo_file = config.get('geo_file', self.dataset)
        self.rel_file = config.get('rel_file', self.dataset)
        self.dyna_file = config.get('dyna_file', self.dataset)
        self.usr_file = config.get('usr_file', self.dataset)
        self.truth_file = config.get('truth_file', self.dataset + '_truth')

        # Model parameters
        self.batch_size = config.get('batch_size', 64)
        self.max_src_length = config.get('max_src_length', 100)
        self.max_trg_length = config.get('max_trg_length', 100)
        self.time_encoding = config.get('time_encoding', 'NoEncoding')
        self.num_workers = config.get('num_workers', 0)

        # Train/val/test split ratios
        self.train_rate = config.get('train_rate', 0.7)
        self.eval_rate = config.get('eval_rate', 0.15)
        # test_rate = 1 - train_rate - eval_rate

        # Trajectory segmentation parameters for handling long trajectories
        # This allows splitting one long trajectory into multiple training samples
        self.enable_segmentation = config.get('enable_segmentation', True)
        self.segment_window_src = config.get('segment_window_src', None)  # Default from max_src_length
        self.segment_window_trg = config.get('segment_window_trg', None)  # Default from max_trg_length
        self.segment_stride_ratio = config.get('segment_stride_ratio', 0.5)  # Stride as ratio of window
        self.min_segment_length = config.get('min_segment_length', 10)  # Minimum segment length

        # Grid tokenization parameters
        self.grid_size = config.get('grid_size', 0.001)  # Grid cell size in degrees
        self.lat_min = config.get('lat_min', None)
        self.lat_max = config.get('lat_max', None)
        self.lon_min = config.get('lon_min', None)
        self.lon_max = config.get('lon_max', None)

        # Cache settings
        self.cache_dataset = config.get('cache_dataset', True)
        self.parameters_str = '{}_{}_{}_{}'.format(
            self.dataset, self.train_rate, self.eval_rate, self.grid_size
        )
        self.cache_file_folder = './libcity/cache/dataset_cache/'
        self.cache_file_name = os.path.join(
            self.cache_file_folder,
            'deep_map_matching_{}.pkl'.format(self.parameters_str)
        )
        ensure_dir(self.cache_file_folder)

        # Vocabularies
        self.src_loc2id = None
        self.src_id2loc = None
        self.trg_seg2id = None
        self.trg_id2seg = None

        # Time vocabularies (if time_encoding is used)
        self.src_tim2id = None
        self.src_id2tim = None

        # Data
        self.src_data = None
        self.trg_data = None
        self.time_data = None

        # Data features
        self.data_feature = {}

    def _construct_vocab(self, lines):
        """
        Construct vocabulary from tokenized lines.

        Adapted from DeepMM's construct_vocab function.

        Args:
            lines: List of tokenized sequences

        Returns:
            word2id: Dict mapping tokens to IDs
            id2word: Dict mapping IDs to tokens
        """
        vocab = {}
        for line in lines:
            for word in line:
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1

        # Remove special tokens if present
        for special in ['<s>', '<pad>', '</s>', '<unk>']:
            if special in vocab:
                del vocab[special]

        # Initialize with special tokens
        word2id = {
            '<s>': 0,
            '<pad>': 1,
            '</s>': 2,
            '<unk>': 3,
        }

        id2word = {
            0: '<s>',
            1: '<pad>',
            2: '</s>',
            3: '<unk>',
        }

        # Sort by frequency (descending)
        sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
        sorted_words = [x[0] for x in sorted_vocab]

        # Assign IDs starting from 4
        for ind, word in enumerate(sorted_words):
            word2id[word] = ind + 4
            id2word[ind + 4] = word

        return word2id, id2word

    def _latlon_to_grid_id(self, lat, lon):
        """
        Convert latitude/longitude to a grid cell ID string.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Grid cell ID as string
        """
        if self.lat_min is not None and self.lon_min is not None:
            # Use predefined bounds
            lat_idx = int((lat - self.lat_min) / self.grid_size)
            lon_idx = int((lon - self.lon_min) / self.grid_size)
        else:
            # Use raw grid indices
            lat_idx = int(lat / self.grid_size)
            lon_idx = int(lon / self.grid_size)

        return '{}_{}'.format(lat_idx, lon_idx)

    def _load_geo_and_rel(self):
        """
        Load road network from .geo and .rel files.

        Returns:
            geo_ids: List of road segment geo_ids
            geo_data: Dict mapping geo_id to road segment properties
        """
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')

        geo_ids = []
        geo_data = {}

        for _, row in geofile.iterrows():
            geo_id = int(row['geo_id'])
            geo_ids.append(geo_id)
            coordinates = eval(row['coordinates'])
            geo_data[geo_id] = {
                'coordinates': coordinates,
            }

        self._logger.info("Loaded file {}.geo, num_segments={}".format(
            self.geo_file, len(geo_ids)))

        return geo_ids, geo_data

    def _load_trajectories(self):
        """
        Load GPS trajectories from .dyna file.

        Returns:
            trajectories: Dict mapping (usr_id, traj_id) to list of GPS points
        """
        dynafile = pd.read_csv(self.data_path + self.dyna_file + '.dyna')

        trajectories = {}
        multi_traj = 'traj_id' in dynafile.keys()

        for _, row in dynafile.iterrows():
            usr_id = row['entity_id']
            traj_id = row['traj_id'] if multi_traj else 0
            coordinates = eval(row['coordinates'])
            time = row.get('time', None)

            key = (usr_id, traj_id)
            if key not in trajectories:
                trajectories[key] = []

            point = {
                'lon': coordinates[0],
                'lat': coordinates[1],
                'time': time
            }
            trajectories[key].append(point)

        self._logger.info("Loaded file {}.dyna, num_trajectories={}".format(
            self.dyna_file, len(trajectories)))

        return trajectories

    def _load_ground_truth(self):
        """
        Load ground truth matched routes from truth file.

        Returns:
            routes: Dict mapping (usr_id, traj_id) to list of matched segment IDs
        """
        truth_path = self.data_path + self.truth_file + '.dyna'
        if not os.path.exists(truth_path):
            self._logger.warning("Ground truth file not found: {}".format(truth_path))
            return None

        truth_dyna = pd.read_csv(truth_path)

        routes = {}
        multi_traj = 'traj_id' in truth_dyna.keys()

        for _, row in truth_dyna.iterrows():
            usr_id = row['entity_id']
            traj_id = row['traj_id'] if multi_traj else 0
            location = row['location']

            key = (usr_id, traj_id)
            if key not in routes:
                routes[key] = []

            routes[key].append(str(location))

        self._logger.info("Loaded file {}.dyna, num_routes={}".format(
            self.truth_file, len(routes)))

        return routes

    def _segment_trajectory(self, src_tokens, trg_tokens):
        """
        Segment a long trajectory into multiple overlapping windows.

        This method breaks long trajectories into multiple training samples using
        sliding windows with overlap. This is essential when:
        - Dataset has only one or few very long trajectories
        - Train/val/test split would result in 0 samples for some splits

        Args:
            src_tokens: List of source GPS grid cell tokens
            trg_tokens: List of target road segment tokens

        Returns:
            segments: List of (src_segment, trg_segment) tuples
        """
        if not self.enable_segmentation:
            return [(src_tokens, trg_tokens)]

        # Use configured window sizes or fall back to max_*_length
        window_src = self.segment_window_src if self.segment_window_src else self.max_src_length
        window_trg = self.segment_window_trg if self.segment_window_trg else self.max_trg_length

        src_len = len(src_tokens)
        trg_len = len(trg_tokens)

        # If trajectory is short enough, return as single sample
        if src_len <= window_src and trg_len <= window_trg:
            return [(src_tokens, trg_tokens)]

        # Calculate stride based on ratio
        stride_src = max(1, int(window_src * self.segment_stride_ratio))

        # Calculate the ratio between source and target lengths
        # This helps maintain alignment between GPS points and road segments
        if src_len > 0:
            trg_src_ratio = trg_len / src_len
        else:
            trg_src_ratio = 1.0

        segments = []
        src_start = 0

        while src_start < src_len:
            # Calculate source window end
            src_end = min(src_start + window_src, src_len)

            # Calculate corresponding target window based on position ratio
            # This maintains approximate alignment between GPS and road segments
            trg_start = int(src_start * trg_src_ratio)
            trg_end = min(int(src_end * trg_src_ratio), trg_len)

            # Ensure target window does not exceed max length
            if trg_end - trg_start > window_trg:
                trg_end = trg_start + window_trg

            # Extract segments
            src_segment = src_tokens[src_start:src_end]
            trg_segment = trg_tokens[trg_start:trg_end]

            # Only add if both segments meet minimum length requirement
            if len(src_segment) >= self.min_segment_length and len(trg_segment) >= self.min_segment_length:
                segments.append((src_segment, trg_segment))

            # Move to next window
            src_start += stride_src

            # Break if we have processed all source tokens
            if src_end >= src_len:
                break

        # If no segments were created (e.g., trajectory too short), return original
        if len(segments) == 0:
            segments = [(src_tokens, trg_tokens)]

        return segments

    def _tokenize_data(self, trajectories, routes):
        """
        Tokenize GPS trajectories and matched routes, then segment into samples.

        This method first converts GPS points to grid cell tokens and road segments
        to segment ID tokens. Then, if segmentation is enabled, it breaks long
        trajectories into multiple overlapping windows to create more training samples.

        Args:
            trajectories: Dict mapping (usr_id, traj_id) to GPS points
            routes: Dict mapping (usr_id, traj_id) to matched segment IDs

        Returns:
            src_loc_lines: List of tokenized GPS sequences (one per sample)
            trg_seg_lines: List of tokenized segment sequences (one per sample)
        """
        src_loc_lines = []
        trg_seg_lines = []

        # Compute bounds if not provided
        if self.lat_min is None or self.lon_min is None:
            all_lats = []
            all_lons = []
            for key in trajectories:
                for point in trajectories[key]:
                    all_lats.append(point['lat'])
                    all_lons.append(point['lon'])

            if all_lats and all_lons:
                self.lat_min = min(all_lats)
                self.lat_max = max(all_lats)
                self.lon_min = min(all_lons)
                self.lon_max = max(all_lons)

        total_original_trajectories = 0
        total_segments_created = 0

        # Process each trajectory
        for key in trajectories:
            if routes is not None and key not in routes:
                continue

            total_original_trajectories += 1

            # Tokenize GPS points to grid cells
            src_tokens = []
            for point in trajectories[key]:
                grid_id = self._latlon_to_grid_id(point['lat'], point['lon'])
                src_tokens.append(grid_id)

            # Get matched segment tokens
            if routes is not None:
                trg_tokens = routes[key]
            else:
                # If no ground truth, use empty sequence (for test only)
                trg_tokens = []

            if len(src_tokens) > 0 and len(trg_tokens) > 0:
                # Segment the trajectory into multiple samples
                segments = self._segment_trajectory(src_tokens, trg_tokens)

                for src_segment, trg_segment in segments:
                    src_loc_lines.append(src_segment)
                    trg_seg_lines.append(trg_segment)
                    total_segments_created += 1

        self._logger.info("Tokenized {} original trajectories into {} samples".format(
            total_original_trajectories, total_segments_created))

        if self.enable_segmentation:
            self._logger.info("Segmentation enabled: window_src={}, window_trg={}, stride_ratio={}".format(
                self.segment_window_src if self.segment_window_src else self.max_src_length,
                self.segment_window_trg if self.segment_window_trg else self.max_trg_length,
                self.segment_stride_ratio))

        return src_loc_lines, trg_seg_lines

    def _load_and_process_data(self):
        """
        Load data from files, tokenize, and build vocabularies.
        """
        # Load road network
        geo_ids, geo_data = self._load_geo_and_rel()

        # Load trajectories
        trajectories = self._load_trajectories()

        # Load ground truth
        routes = self._load_ground_truth()

        if routes is None:
            raise ValueError("Ground truth file is required for training DeepMM model")

        # Tokenize data
        src_loc_lines, trg_seg_lines = self._tokenize_data(trajectories, routes)

        # Build vocabularies
        self.src_loc2id, self.src_id2loc = self._construct_vocab(src_loc_lines)
        self.trg_seg2id, self.trg_id2seg = self._construct_vocab(trg_seg_lines)

        self._logger.info("Source vocabulary size: {}".format(len(self.src_loc2id)))
        self._logger.info("Target vocabulary size: {}".format(len(self.trg_seg2id)))

        self.src_data = src_loc_lines
        self.trg_data = trg_seg_lines

    def _split_data(self):
        """
        Split data into train/validation/test sets.

        Returns:
            train_src, train_trg: Training data
            val_src, val_trg: Validation data
            test_src, test_trg: Test data
        """
        n_samples = len(self.src_data)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        n_train = int(n_samples * self.train_rate)
        n_val = int(n_samples * self.eval_rate)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        train_src = [self.src_data[i] for i in train_indices]
        train_trg = [self.trg_data[i] for i in train_indices]

        val_src = [self.src_data[i] for i in val_indices]
        val_trg = [self.trg_data[i] for i in val_indices]

        test_src = [self.src_data[i] for i in test_indices]
        test_trg = [self.trg_data[i] for i in test_indices]

        self._logger.info("Data split: train={}, val={}, test={}".format(
            len(train_src), len(val_src), len(test_src)))

        return train_src, train_trg, val_src, val_trg, test_src, test_trg

    def _create_dataloaders(self, train_src, train_trg, val_src, val_trg, test_src, test_trg):
        """
        Create PyTorch DataLoaders for train/val/test sets.

        Returns:
            train_loader, val_loader, test_loader
        """
        train_dataset = DeepMMSeq2SeqTorchDataset(
            train_src, train_trg,
            self.src_loc2id, self.trg_seg2id,
            self.max_src_length, self.max_trg_length,
            self.time_encoding
        )

        val_dataset = DeepMMSeq2SeqTorchDataset(
            val_src, val_trg,
            self.src_loc2id, self.trg_seg2id,
            self.max_src_length, self.max_trg_length,
            self.time_encoding
        )

        test_dataset = DeepMMSeq2SeqTorchDataset(
            test_src, test_trg,
            self.src_loc2id, self.trg_seg2id,
            self.max_src_length, self.max_trg_length,
            self.time_encoding
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

        return train_loader, val_loader, test_loader

    def get_data(self):
        """
        Load and process data, return DataLoaders.

        Returns:
            tuple: (train_dataloader, eval_dataloader, test_dataloader)
        """
        # Check cache
        if self.cache_dataset and os.path.exists(self.cache_file_name):
            self._logger.info("Loading cached dataset from {}".format(self.cache_file_name))
            with open(self.cache_file_name, 'rb') as f:
                cache_data = pickle.load(f)

            self.src_loc2id = cache_data['src_loc2id']
            self.src_id2loc = cache_data['src_id2loc']
            self.trg_seg2id = cache_data['trg_seg2id']
            self.trg_id2seg = cache_data['trg_id2seg']
            self.src_data = cache_data['src_data']
            self.trg_data = cache_data['trg_data']
            self.data_feature = cache_data.get('data_feature', {})
        else:
            # Load and process data
            self._load_and_process_data()

            # Cache data
            if self.cache_dataset:
                cache_data = {
                    'src_loc2id': self.src_loc2id,
                    'src_id2loc': self.src_id2loc,
                    'trg_seg2id': self.trg_seg2id,
                    'trg_id2seg': self.trg_id2seg,
                    'src_data': self.src_data,
                    'trg_data': self.trg_data,
                    'data_feature': self.data_feature
                }
                with open(self.cache_file_name, 'wb') as f:
                    pickle.dump(cache_data, f)
                self._logger.info("Saved cached dataset to {}".format(self.cache_file_name))

        # Update data features
        self.data_feature.update({
            'src_loc_vocab_size': len(self.src_loc2id),
            'trg_seg_vocab_size': len(self.trg_seg2id),
            'pad_token_src_loc': self.src_loc2id['<pad>'],
            'pad_token_trg': self.trg_seg2id['<pad>'],
            'sos_token_trg': self.trg_seg2id['<s>'],
            'eos_token_trg': self.trg_seg2id['</s>'],
            'src_loc2id': self.src_loc2id,
            'src_id2loc': self.src_id2loc,
            'trg_seg2id': self.trg_seg2id,
            'trg_id2seg': self.trg_id2seg,
        })

        # Add time vocabulary info if time encoding is used
        if self.time_encoding != 'NoEncoding':
            self.data_feature['src_tim_vocab_size'] = len(self.src_tim2id) if self.src_tim2id else 0

        # Split data
        train_src, train_trg, val_src, val_trg, test_src, test_trg = self._split_data()

        # Create dataloaders
        train_loader, val_loader, test_loader = self._create_dataloaders(
            train_src, train_trg, val_src, val_trg, test_src, test_trg
        )

        return train_loader, val_loader, test_loader

    def get_data_feature(self):
        """
        Return data features required by the DeepMM model.

        Returns:
            dict: Data features including vocabulary sizes and special token IDs
        """
        if not self.data_feature:
            # If get_data hasn't been called, return basic features
            self.data_feature = {
                'src_loc_vocab_size': len(self.src_loc2id) if self.src_loc2id else 0,
                'trg_seg_vocab_size': len(self.trg_seg2id) if self.trg_seg2id else 0,
                'pad_token_src_loc': 1,
                'pad_token_trg': 1,
                'sos_token_trg': 0,
                'eos_token_trg': 2,
            }

        return self.data_feature
