# DiffMM Dataset Adapter - Migration Summary

## Overview

This document describes the dataset adapter created to bridge LibCity's standard datasets with DiffMM's expected batch format.

## Problem Statement

DiffMM requires a specialized batch format that standard LibCity datasets do not provide:

**Required by DiffMM:**
- `lengths` - Sequence lengths for each trajectory
- `norm_gps_seq` - Normalized GPS coordinates (batch, seq_len, 3)
- `trg_rid` - Target road segment IDs
- `trg_onehot` - One-hot encoded target segments
- `segs_id` - Candidate segment IDs (batch, seq_len, num_cands)
- `segs_feat` - Segment features (batch, seq_len, num_cands, 9)
- `segs_mask` - Mask for valid candidates

**Available from standard datasets:**
- MapMatchingDataset: trajectory, rd_nwk, route dictionaries
- TrajectoryDataset: history_loc, history_tim, current_loc, target, uid
- DeepMapMatchingDataset: grid_traces, tgt_roads, traces_gps

## Solution Implemented

### 1. Created DiffMMDataset Class

**File:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/dataset_subclass/diffmm_dataset.py`

The DiffMMDataset extends MapMatchingDataset and provides:

- **Candidate segment generation** - Finds road segments within a configurable search radius of each GPS point
- **Road segment feature computation** - Calculates 9 features per candidate:
  - dist_to_start_norm
  - dist_to_end_norm
  - dist_to_mid_norm
  - bearing_diff_norm
  - length_norm
  - speed_norm
  - lat_diff
  - lng_diff
  - time_norm (placeholder)
- **One-hot encoding of targets** - Creates one-hot vectors for diffusion training
- **GPS normalization** - Normalizes coordinates to [0,1] based on road network MBR

### 2. Updated DiffMM Model's _batch2model() Method

**Files:**
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/DiffMM.py`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/DiffMM.py`

The model now handles multiple batch formats:

1. **DiffMMDataset format (preferred)** - Uses data directly
2. **DeepMapMatchingDataset format** - Adapts grid_traces, tgt_roads, traces_gps
3. **Unknown formats** - Provides helpful error message with required keys

Key methods added:
- `_batch2model()` - Routes to appropriate handler based on detected format
- `_process_diffmm_batch()` - Processes native DiffMMDataset format
- `_adapt_deepmm_batch()` - Creates synthetic structures from DeepMapMatchingDataset

### 3. Configuration Updates

**Model Config:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/map_matching/DiffMM.json`

Added parameters:
```json
{
    "dataset_class": "DiffMMDataset",
    "num_cands": 10,
    "cand_search_radius": 100,
    "max_seq_len": 100,
    "min_seq_len": 5,
    "train_rate": 0.7,
    "eval_rate": 0.15
}
```

**Task Config:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`

Updated both map_matching and traj_loc_pred sections to use DiffMMDataset.

### 4. Module Registration

**Files updated:**
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/__init__.py`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/dataset_subclass/__init__.py`

Added imports and exports for DiffMMDataset.

## Batch Format Details

### DiffMMDataset Collate Function Output

```python
{
    'lengths': torch.LongTensor,           # (batch_size,)
    'norm_gps_seq': torch.FloatTensor,     # (batch_size, max_seq_len, 3)
    'trg_rid': torch.LongTensor,           # (batch_size, max_seq_len)
    'trg_onehot': torch.FloatTensor,       # (batch_size, max_seq_len, num_roads)
    'segs_id': torch.LongTensor,           # (batch_size, max_seq_len, num_cands)
    'segs_feat': torch.FloatTensor,        # (batch_size, max_seq_len, num_cands, 9)
    'segs_mask': torch.FloatTensor         # (batch_size, max_seq_len, num_cands)
}
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| num_cands | 10 | Maximum candidate segments per GPS point |
| cand_search_radius | 100 | Search radius in meters |
| max_seq_len | 100 | Maximum trajectory sequence length |
| min_seq_len | 5 | Minimum trajectory sequence length |
| train_rate | 0.7 | Training data ratio |
| eval_rate | 0.15 | Validation data ratio |

## Usage

To use DiffMM with LibCity:

```python
from libcity.pipeline import run_model

run_model(
    task='map_matching',
    dataset='your_dataset',
    model='DiffMM'
)
```

The framework will automatically use DiffMMDataset based on the task_config.json configuration.

## Limitations and Assumptions

1. **Candidate generation uses distance to road midpoint** - This is an approximation; a more accurate implementation would compute point-to-segment distance.

2. **Feature computation is simplified** - Some features (like time normalization) use placeholders when timestamp data is not available.

3. **Fallback adapter for DeepMapMatchingDataset** - Creates synthetic data structures that are not optimal for training. Always prefer using DiffMMDataset for best results.

4. **Ground truth required** - The dataset requires ground truth route files for training and evaluation.

## Files Modified/Created

### Created
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/dataset_subclass/diffmm_dataset.py`

### Modified
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/__init__.py`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/dataset_subclass/__init__.py`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/DiffMM.py`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/DiffMM.py`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/map_matching/DiffMM.json`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
