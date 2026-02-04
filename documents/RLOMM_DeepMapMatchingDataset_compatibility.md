# RLOMM Model: DeepMapMatchingDataset Compatibility Fix

## Overview

This document describes the modifications made to the RLOMM (Reinforcement Learning Online Map Matching) model to make it compatible with the DeepMapMatchingDataset batch format.

## Problem Statement

The RLOMM model was originally designed to work with a specific batch format that includes:
- `traces`: (batch_size, seq_len, 2) with [grid_id, timestamp]
- `tgt_roads`: (batch_size, seq_len) - indices into candidates
- `candidates_id`: (batch_size, seq_len, num_candidates) - candidate road IDs
- `trace_lens`: list of actual trace lengths

However, the DeepMapMatchingDataset (used by other models like GraphMM) provides a different format:
- `grid_traces`: (batch_size, seq_len) - grid cell IDs (1-indexed, 0=padding)
- `tgt_roads`: (batch_size, seq_len) - ground truth road IDs (not indices)
- `traces_lens`: list of actual trace lengths
- `road_lens`: list of actual road lengths
- `traces_gps`: (batch_size, seq_len, 2) - GPS coordinates

The key missing component was `candidates_id`, which is critical for the RL-based approach.

## Solution Implemented

The RLOMM model was modified to support both batch formats by:

### 1. Storing map_matrix for Candidate Generation

In `__init__()`:
```python
# Store map_matrix for candidate generation
map_matrix = data_feature.get('map_matrix', None)
if map_matrix is not None:
    self.map_matrix = map_matrix
else:
    self.map_matrix = torch.eye(self.num_grids, self.num_roads)

# Pre-compute grid-to-candidate mapping
self._precompute_grid_candidates()
```

### 2. Pre-computing Grid-to-Candidate Mapping

New method `_precompute_grid_candidates()`:
- Creates a lookup table mapping each grid cell to its candidate roads
- Uses map_matrix to find roads connected to each grid
- Sorts candidates by mapping score (proximity)

### 3. Building Road Adjacency for Fallback Candidates

New method `_build_road_adjacency()`:
- Builds adjacency list from road graph
- Used when a grid cell has no direct road mapping
- Allows using adjacent roads from previous match as candidates

### 4. On-the-fly Candidate Generation

New method `_generate_candidates_from_grid()`:
- Takes grid_traces and tgt_roads as input
- Generates candidates for each grid position:
  1. Always includes target road first (ensures training signal)
  2. Adds roads from grid mapping (via map_matrix)
  3. Adds adjacent roads from previous candidates
  4. Fills remaining slots with random roads
- Returns candidates_id and tgt_indices (indices into candidates)

### 5. Unified Batch Preprocessing

New method `_preprocess_batch()`:
- Detects batch format (original vs DeepMapMatchingDataset)
- For original format: passes through as-is
- For DeepMapMatchingDataset format:
  - Extracts grid_traces and tgt_roads
  - Generates candidates using `_generate_candidates_from_grid()`
  - Constructs traces tensor with [grid_id, timestamp]
  - Returns unified format for model consumption

### 6. Updated Model Methods

Modified methods to use `_preprocess_batch()`:
- `forward()`: Now handles both formats transparently
- `predict()`: Now handles both formats transparently
- `calculate_loss()`: Now handles both formats transparently

## File Modified

**Path**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/RLOMM.py`

## Key Changes Summary

| Component | Change |
|-----------|--------|
| `__init__()` | Added map_matrix storage, num_grids, and candidate pre-computation |
| `_precompute_grid_candidates()` | New method - pre-computes grid-to-candidate mapping |
| `_build_road_adjacency()` | New method - builds road adjacency for fallback candidates |
| `_generate_candidates_from_grid()` | New method - generates candidates on-the-fly |
| `_preprocess_batch()` | New method - handles both batch formats |
| `forward()` | Updated to use `_preprocess_batch()` |
| `predict()` | Updated to use `_preprocess_batch()` |
| `calculate_loss()` | Updated to use `_preprocess_batch()` |
| `update_graph_data()` | Updated to recompute candidates when graph changes |
| Module docstring | Updated to document both batch formats |

## Assumptions

1. **Grid IDs are 1-indexed**: In DeepMapMatchingDataset, grid_traces uses 1-based indexing where 0 represents padding. The candidate generation adjusts for this.

2. **Target road should be in candidates**: For proper training, the target road is always included as the first candidate. This ensures a valid training signal even if the grid-to-road mapping doesn't include the target.

3. **Timestamps can be simulated**: When timestamps are not provided (DeepMapMatchingDataset doesn't include them), position indices are used as timestamps.

## Usage Example

```python
# Using with DeepMapMatchingDataset
from libcity.data.dataset import DeepMapMatchingDataset
from libcity.model.map_matching import RLOMM

# Create dataset
dataset = DeepMapMatchingDataset(config)
train_loader, eval_loader, test_loader = dataset.get_data()

# Create model with data features (includes map_matrix)
data_feature = dataset.get_data_feature()
model = RLOMM(config, data_feature)

# Training - batch format is handled automatically
for batch in train_loader:
    loss = model.calculate_loss(batch)  # Works with DeepMapMatchingDataset format
```

## Limitations

1. **Candidate quality**: Auto-generated candidates may not be as accurate as pre-computed candidates that consider GPS proximity.

2. **No timestamp information**: When using DeepMapMatchingDataset, position is used as timestamp, which may affect time-aware features.

3. **Random fallback**: When grid mapping is sparse, random roads are used to fill candidate slots, which may slow training.

## Future Improvements

1. Use GPS coordinates (traces_gps) to compute proximity-based candidates
2. Cache generated candidates for frequently used grid patterns
3. Consider road connectivity when generating fallback candidates
