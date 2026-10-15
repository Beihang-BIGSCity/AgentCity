"""
Contrastive Split Language Model Dataset for START.
Contains ContrastiveSplitLMDataset - the primary dataset class for START model.

Migrated from: /home/wangwenrui/shk/AgentCity/repos/START/libcity/data/dataset/bertlm_contrastive_split_dataset.py
"""

import os
import torch
import pickle
import pandas as pd
from libcity.data.dataset.bertlm_dataset import BERTSubDataset, noise_mask
from libcity.data.dataset.contrastive_split_dataset import collate_unsuperv_contrastive_split
from libcity.data.dataset.bertlm_contrastive_dataset import collate_unsuperv_mask, ContrastiveLMDataset


class ContrastiveSplitLMDataset(ContrastiveLMDataset):
    """
    Contrastive Split Language Model Dataset.

    This is the primary dataset class for the START model, combining:
    - Contrastive learning with different data augmentations
    - Masked language modeling objective

    It creates two augmented views of trajectories and also prepares
    data for masked language model training.
    """

    def __init__(self, config):
        """
        Initialize ContrastiveSplitLMDataset.

        Args:
            config: Configuration dictionary containing:
                - out_data_argument1: First augmentation type (e.g., 'trim')
                - out_data_argument2: Second augmentation type (e.g., 'time')
                - data_argument1: List of additional arguments for view 1
                - data_argument2: List of additional arguments for view 2
                - Other base config parameters
        """
        super().__init__(config)
        self.argu1 = config.get('out_data_argument1', 'trim')
        self.argu2 = config.get('out_data_argument2', 'time')
        self.data_argument1 = self.config.get("data_argument1", [])
        self.data_argument2 = self.config.get("data_argument2", [])
        self.collate_fn = collate_unsuperv_contrastive_split_lm

    def _gen_dataset(self):
        """Generate split datasets with different augmentations and LM support."""
        train_dataset = TrajectoryProcessingDatasetSplitLM(
            data_name=self.dataset, data_type='train', vocab=self.vocab,
            seq_len=self.seq_len, add_cls=self.add_cls, merge=self.merge, min_freq=self.min_freq,
            max_train_size=self.max_train_size,
            data_argument1=self.data_argument1,
            data_argument2=self.data_argument2,
            masking_ratio=self.masking_ratio, masking_mode=self.masking_mode,
            distribution=self.distribution, avg_mask_len=self.avg_mask_len,
            argu1=self.argu1, argu2=self.argu2)
        eval_dataset = TrajectoryProcessingDatasetSplitLM(
            data_name=self.dataset, data_type='eval', vocab=self.vocab,
            seq_len=self.seq_len, add_cls=self.add_cls, merge=self.merge, min_freq=self.min_freq,
            max_train_size=None,
            data_argument1=self.data_argument1,
            data_argument2=self.data_argument2,
            masking_ratio=self.masking_ratio, masking_mode=self.masking_mode,
            distribution=self.distribution, avg_mask_len=self.avg_mask_len,
            argu1=self.argu1, argu2=self.argu2)
        test_dataset = TrajectoryProcessingDatasetSplitLM(
            data_name=self.dataset, data_type='test', vocab=self.vocab,
            seq_len=self.seq_len, add_cls=self.add_cls, merge=self.merge, min_freq=self.min_freq,
            max_train_size=None,
            data_argument1=self.data_argument1,
            data_argument2=self.data_argument2,
            masking_ratio=self.masking_ratio, masking_mode=self.masking_mode,
            distribution=self.distribution, avg_mask_len=self.avg_mask_len,
            argu1=self.argu1, argu2=self.argu2)
        return train_dataset, eval_dataset, test_dataset


class TrajectoryProcessingDatasetSplitLM(BERTSubDataset):
    """
    Trajectory Dataset with Split Augmentation and Language Modeling.

    Extends BERTSubDataset to support loading two different augmented
    views of trajectories along with the original for language modeling.
    """

    def __init__(self, data_name, data_type, vocab, seq_len=512, add_cls=True,
                 merge=True, min_freq=1, max_train_size=None,
                 data_argument1=None, data_argument2=None,
                 masking_ratio=0.2, masking_mode='together',
                 distribution='random', avg_mask_len=3, argu1=None, argu2=None):
        """
        Initialize TrajectoryProcessingDatasetSplitLM.

        Args:
            data_name: Name of the dataset
            data_type: Type of data ('train', 'eval', or 'test')
            vocab: WordVocab instance
            seq_len: Maximum sequence length
            add_cls: Whether to add CLS token
            merge: Whether to use merged vocabulary
            min_freq: Minimum frequency for vocabulary
            max_train_size: Maximum training samples
            data_argument1: List of arguments for view 1 (e.g., ['mask'])
            data_argument2: List of arguments for view 2 (e.g., ['mask'])
            masking_ratio: Ratio for masking
            masking_mode: 'together' or 'separate'
            distribution: 'random' or 'geometric'
            avg_mask_len: Average mask span length
            argu1: First augmentation type (e.g., 'trim')
            argu2: Second augmentation type (e.g., 'time')
        """
        if argu1 is not None:
            self.data_path1 = 'raw_data/{}/{}_{}_enhancedby{}.csv'.format(
                data_name, data_name, data_type, argu1)
            self.cache_path1 = 'raw_data/{}/cache_{}_{}_{}_{}_{}_enhancedby{}.pkl'.format(
                data_name, data_name, data_type, add_cls, merge, min_freq, argu1)
        else:
            self.data_path1 = 'raw_data/{}/{}_{}.csv'.format(
                data_name, data_name, data_type)
            self.cache_path1 = 'raw_data/{}/cache_{}_{}_{}_{}_{}.pkl'.format(
                data_name, data_name, data_type, add_cls, merge, min_freq)
        if argu2 is not None:
            self.data_path2 = 'raw_data/{}/{}_{}_enhancedby{}.csv'.format(
                data_name, data_name, data_type, argu2)
            self.cache_path2 = 'raw_data/{}/cache_{}_{}_{}_{}_{}_enhancedby{}.pkl'.format(
                data_name, data_name, data_type, add_cls, merge, min_freq, argu2)
        else:
            self.data_path2 = 'raw_data/{}/{}_{}.csv'.format(
                data_name, data_name, data_type)
            self.cache_path2 = 'raw_data/{}/cache_{}_{}_{}_{}_{}.pkl'.format(
                data_name, data_name, data_type, add_cls, merge, min_freq)

        self.temporal_mat_path1 = self.cache_path1[:-4] + '_temporal_mat.pkl'
        self.temporal_mat_path2 = self.cache_path2[:-4] + '_temporal_mat.pkl'

        super().__init__(data_name, data_type, vocab, seq_len, add_cls, merge,
                         min_freq, max_train_size, masking_ratio, masking_mode, distribution, avg_mask_len)
        self._logger.info('Init TrajectoryProcessingDatasetSplitLM!')
        self.data_argument1 = data_argument1
        self.data_argument2 = data_argument2

        self._load_data_split()

    def _load_data_split(self):
        """Load augmented data from cache or process from CSV files."""
        if os.path.exists(self.cache_path1) and os.path.exists(self.temporal_mat_path1) \
                and os.path.exists(self.cache_path2) and os.path.exists(self.temporal_mat_path2):
            self.traj_list1 = pickle.load(open(self.cache_path1, 'rb'))
            self.temporal_mat_list1 = pickle.load(open(self.temporal_mat_path1, 'rb'))
            self.traj_list2 = pickle.load(open(self.cache_path2, 'rb'))
            self.temporal_mat_list2 = pickle.load(open(self.temporal_mat_path2, 'rb'))
            self._logger.info('Load dataset from {}, {}'.format(self.cache_path1, self.cache_path2))
        else:
            origin_data_df1 = pd.read_csv(self.data_path1, sep=';')
            origin_data_df2 = pd.read_csv(self.data_path2, sep=';')
            assert origin_data_df1.shape == origin_data_df2.shape
            self.traj_list1, self.temporal_mat_list1 = self.data_processing(
                origin_data_df1, self.data_path1, cache_path=self.cache_path1, tmat_path=self.temporal_mat_path1)
            self.traj_list2, self.temporal_mat_list2 = self.data_processing(
                origin_data_df2, self.data_path2, cache_path=self.cache_path2, tmat_path=self.temporal_mat_path2)

    def __len__(self):
        assert len(self.traj_list1) == len(self.traj_list2) == len(self.traj_list)
        return len(self.traj_list)

    def __getitem__(self, ind):
        traj_ind, mask, temporal_mat = super().__getitem__(ind)
        traj_ind1 = self.traj_list1[ind]  # (seq_length, feat_dim)
        traj_ind2 = self.traj_list2[ind]  # (seq_length, feat_dim)
        temporal_mat1 = self.temporal_mat_list1[ind]  # (seq_length, seq_length)
        temporal_mat2 = self.temporal_mat_list2[ind]  # (seq_length, seq_length)
        mask1 = None
        mask2 = None
        if 'mask' in self.data_argument1:
            mask1 = noise_mask(traj_ind1, self.masking_ratio, self.avg_mask_len, self.masking_mode, self.distribution,
                               self.exclude_feats, self.add_cls)  # (seq_length, feat_dim) boolean array
        if 'mask' in self.data_argument2:
            mask2 = noise_mask(traj_ind2, self.masking_ratio, self.avg_mask_len, self.masking_mode, self.distribution,
                               self.exclude_feats, self.add_cls)  # (seq_length, feat_dim) boolean array
        return traj_ind, mask, temporal_mat, \
               torch.LongTensor(traj_ind1), torch.LongTensor(traj_ind2), \
               torch.LongTensor(temporal_mat1), torch.LongTensor(temporal_mat2), \
               torch.LongTensor(mask1) if mask1 is not None else None, \
               torch.LongTensor(mask2) if mask2 is not None else None


def collate_unsuperv_contrastive_split_lm(data, max_len=None, vocab=None, add_cls=True):
    """
    Collate function for contrastive split learning with language modeling.

    Combines contrastive split data processing with masked language model data.

    Args:
        data: List of tuples containing:
            (features, masks, temporal_mat, features1, features2,
             temporal_mat1, temporal_mat2, mask1, mask2)
        max_len: Maximum sequence length
        vocab: WordVocab instance
        add_cls: Whether CLS token is present

    Returns:
        Tuple of:
            - X1, X2: Contrastive views
            - padding_masks1, padding_masks2: Padding masks for views
            - batch_temporal_mat1, batch_temporal_mat2: Temporal matrices
            - masked_x: Masked input for LM
            - targets: Target tokens for LM
            - target_masks: Mask for targets
            - padding_masks: Padding mask for LM
            - batch_temporal_mat: Temporal matrix for LM
    """
    features, masks, temporal_mat, features1, features2, temporal_mat1, temporal_mat2, mask1, mask2 = zip(*data)
    data_for_mask = list(zip(features, masks, temporal_mat))
    dara_for_contra = list(zip(features1, features2, temporal_mat1, temporal_mat2, mask1, mask2))

    X1, X2, padding_masks1, padding_masks2, batch_temporal_mat1, batch_temporal_mat2 \
        = collate_unsuperv_contrastive_split(data=dara_for_contra, max_len=max_len, vocab=vocab, add_cls=add_cls)

    masked_x, targets, target_masks, padding_masks, batch_temporal_mat = collate_unsuperv_mask(
        data=data_for_mask, max_len=max_len, vocab=vocab, add_cls=add_cls)
    return X1, X2, padding_masks1, padding_masks2, batch_temporal_mat1, batch_temporal_mat2, \
           masked_x, targets, target_masks, padding_masks, batch_temporal_mat
