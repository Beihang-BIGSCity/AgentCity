"""
Contrastive Language Model Dataset for START.
Contains ContrastiveLMDataset class and collate function.

Migrated from: /home/wangwenrui/shk/AgentCity/repos/START/libcity/data/dataset/bertlm_contrastive_dataset.py
"""

import torch
from libcity.data.dataset.bertlm_dataset import BERTLMDataset, collate_unsuperv_mask


class ContrastiveLMDataset(BERTLMDataset):
    """
    Contrastive Learning + Language Model Dataset.

    This dataset combines contrastive learning with masked language modeling
    for trajectory representation learning.
    """

    def __init__(self, config):
        """
        Initialize ContrastiveLMDataset.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.collate_fn = collate_unsuperv_contrastive_lm


def collate_unsuperv_contrastive_lm(data, max_len=None, vocab=None, add_cls=True):
    """
    Collate function for contrastive learning with language modeling.

    Creates two views of the data (both same for basic contrastive learning)
    along with masked language model inputs.

    Args:
        data: List of (features, masks, temporal_mat) tuples
        max_len: Maximum sequence length for padding
        vocab: WordVocab instance
        add_cls: Whether CLS token is present

    Returns:
        Tuple of (contra_view1, contra_view2, masked_x, targets, target_masks, padding_masks, batch_temporal_mat)
    """
    batch_size = len(data)
    features, masks, temporal_mat = zip(*data)

    lengths = [X.shape[0] for X in features]
    if max_len is None:
        max_len = max(lengths)
    contra_view1 = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        contra_view1[i, :end, :] = features[i][:end, :]
    masked_x, targets, target_masks, padding_masks, batch_temporal_mat = collate_unsuperv_mask(
        data=data, max_len=max_len, vocab=vocab, add_cls=add_cls)
    return contra_view1.long(), contra_view1.long().clone(), masked_x, targets, target_masks, padding_masks, batch_temporal_mat
