# DeepMMSeq2SeqDataset Migration Summary

## Overview

Created a new dataset class `DeepMMSeq2SeqDataset` to provide proper training data for the DeepMM seq2seq map matching model. This solves the problem where `MapMatchingDataset` returns `(None, None, test_data)` because traditional map matching algorithms do not require training data.

## Problem Statement

The DeepMM model is a neural network-based seq2seq map matching algorithm that requires:
- Training data with tokenized GPS sequences and corresponding matched road segments
- Proper train/validation/test data splits
- Batch format compatible with seq2seq architecture (source sequence, target input for teacher forcing, target output)

The existing `MapMatchingDataset` only provides test data, which is insufficient for training neural models.

## Solution

Created `DeepMMSeq2SeqDataset` class that:

1. **Inherits from `AbstractDataset`** - Following LibCity conventions
2. **Loads GPS trajectories and road network** from LibCity's standard data format (.geo, .dyna files)
3. **Tokenizes GPS locations** - Discretizes GPS coordinates to grid cells using configurable grid size
4. **Tokenizes road segments** - Uses geo_ids from road network as token IDs
5. **Builds vocabularies** - Creates loc2id, seg2id, id2loc, id2seg mappings with special tokens (<s>, </s>, <pad>, <unk>)
6. **Splits data** - Configurable train/validation/test splits (default 70/15/15)
7. **Creates PyTorch DataLoaders** - With proper batch collation and padding

## Files Created/Modified

### Created Files

1. **`/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/dataset_subclass/deep_map_matching_dataset.py`**

   Contains:
   - `DeepMMSeq2SeqTorchDataset`: PyTorch Dataset wrapper class
   - `collate_fn`: Batch collation function with padding
   - `DeepMMSeq2SeqDataset`: Main LibCity dataset class

### Modified Files

1. **`/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/__init__.py`**
   - Added import for `DeepMMSeq2SeqDataset`
   - Added class to `__all__` list

2. **`/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`**
   - Updated DeepMM model entry to use `DeepMMSeq2SeqDataset` instead of `MapMatchingDataset`

3. **`/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/map_matching/DeepMM.json`**
   - Added dataset-related configuration parameters

## Batch Format

The dataset returns batches in the following format:

```python
{
    'input_src': torch.LongTensor,      # Source location sequences [batch_size, src_seq_len]
    'input_trg': torch.LongTensor,      # Target segments for teacher forcing [batch_size, trg_seq_len]
    'output_trg': torch.LongTensor,     # Ground truth segments [batch_size, trg_seq_len]
    'target': torch.LongTensor,         # Alias for output_trg (compatibility)
}
```

## Data Features

The dataset provides the following features to the model via `get_data_feature()`:

```python
{
    'src_loc_vocab_size': int,      # Number of unique GPS grid cells
    'trg_seg_vocab_size': int,      # Number of unique road segments
    'pad_token_src_loc': int,       # Padding token ID for source (default: 1)
    'pad_token_trg': int,           # Padding token ID for target (default: 1)
    'sos_token_trg': int,           # Start-of-sequence token ID (default: 0)
    'eos_token_trg': int,           # End-of-sequence token ID (default: 2)
    'src_loc2id': dict,             # GPS grid cell to ID mapping
    'src_id2loc': dict,             # ID to GPS grid cell mapping
    'trg_seg2id': dict,             # Road segment to ID mapping
    'trg_id2seg': dict,             # ID to road segment mapping
}
```

## Configuration Parameters

The following parameters can be configured in the model config or passed to the dataset:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 64 | Batch size for DataLoader |
| `max_src_length` | 100 | Maximum source sequence length |
| `max_trg_length` | 100 | Maximum target sequence length |
| `time_encoding` | 'NoEncoding' | Time encoding method |
| `grid_size` | 0.001 | Grid cell size in degrees |
| `train_rate` | 0.7 | Training data ratio |
| `eval_rate` | 0.15 | Validation data ratio |
| `cache_dataset` | True | Whether to cache processed data |
| `num_workers` | 0 | DataLoader workers |

## Data Requirements

The dataset expects the following files in `./raw_data/{dataset}/`:

1. **`{dataset}.geo`** - Road network with geo_id and coordinates
2. **`{dataset}.dyna`** - GPS trajectories with entity_id, traj_id, coordinates
3. **`{dataset}_truth.dyna`** - Ground truth matched routes with entity_id, traj_id, location (geo_id)

## Usage Example

```python
from libcity.data.dataset import DeepMMSeq2SeqDataset

config = {
    'dataset': 'your_dataset_name',
    'batch_size': 64,
    'max_src_length': 100,
    'max_trg_length': 100,
    'grid_size': 0.001,
    'train_rate': 0.7,
    'eval_rate': 0.15,
}

dataset = DeepMMSeq2SeqDataset(config)
train_loader, val_loader, test_loader = dataset.get_data()
data_feature = dataset.get_data_feature()

# Use with DeepMM model
model = DeepMM(config, data_feature)
```

## Differences from DeepMapMatchingDataset

Note that `DeepMapMatchingDataset` (in the main dataset directory) is designed for GraphMM and similar graph-based map matching models. It provides:
- Grid-based trajectory representations
- Road network graph structures
- Trace graphs with grid connectivity

`DeepMMSeq2SeqDataset` is specifically designed for DeepMM's seq2seq architecture with:
- Simple tokenized sequences
- Teacher forcing format
- No graph structures

## Original Reference

The tokenization logic was adapted from the original DeepMM implementation:
- Source: `./repos/DeepMM/DeepMM/data_utils.py`
