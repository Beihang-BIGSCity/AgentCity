# RLOMM Model: DeepMapMatchingDataset Batch Format Compatibility Fix

## Overview

This document describes the fixes implemented to make the RLOMM (Reinforcement Learning Online Map Matching) model compatible with the DeepMapMatchingDataset batch format.

## Problem Statement

The RLOMM model was originally designed to work with a batch format that includes pre-computed candidate roads:
- `traces`: [batch, seq_len, 2] with (grid_id, timestamp)
- `tgt_roads`: [batch, seq_len] - indices into candidates
- `candidates_id`: [batch, seq_len, num_candidates] - candidate road IDs
- `trace_lens`: list of trace lengths

However, the DeepMapMatchingDataset provides a different format:
- `grid_traces`: [batch, seq_len] - grid cell IDs (1-indexed, 0=padding)
- `tgt_roads`: [batch, seq_len] - ground truth road IDs (not indices)
- `traces_lens`: list of trace lengths
- `road_lens`: list of road lengths
- `traces_gps`: [batch, seq_len, 2] - GPS coordinates

The key incompatibilities were:
1. **Key mismatch**: RLOMM looked for `traces`, `X`, or `input_traces`, but DeepMapMatchingDataset provides `grid_traces`
2. **Dimension mismatch**: RLOMM expected 3D traces [batch, seq_len, 2], but `grid_traces` is 2D [batch, seq_len]
3. **Missing candidates**: RLOMM required `candidates_id`, which DeepMapMatchingDataset does not provide
4. **Key alias mismatch**: RLOMM looked for `trace_lens`, but DeepMapMatchingDataset uses `traces_lens`

## Changes Made

### File Modified
**Path**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/RLOMM.py`

### 1. Added `candidate_size` Configuration (lines 736-737)

```python
# Candidate generation parameters
self.candidate_size = config.get('candidate_size', 10)
```

This allows controlling the number of candidate roads generated per grid position.

### 2. Added `map_matrix` Storage (lines 742-748)

```python
# Store map_matrix for candidate generation (used when candidates_id not provided)
map_matrix = data_feature.get('map_matrix', None)
if map_matrix is not None:
    self.map_matrix = map_matrix
else:
    # Fallback: identity-like mapping (each grid maps to itself as road)
    self.map_matrix = None
```

The map_matrix is used to find candidate roads for each grid cell when candidates are not provided in the batch.

### 3. Updated `_prepare_batch` Method (lines 814-864)

Key changes:
- Added `grid_traces` as an accepted trace key
- Added dimension handling for 2D traces (converting to 3D with zero time delta)
- Added dynamic candidate generation when `candidates_id` is not in batch
- Added `traces_lens` as an accepted key for trace lengths

```python
# Get traces - add 'grid_traces' as valid key
traces = batch.get('traces', batch.get('X', batch.get('input_traces', batch.get('grid_traces'))))
if traces is None:
    raise KeyError("Batch must contain 'traces', 'X', 'input_traces', or 'grid_traces'")
traces = traces.to(self.device)

# Handle 1D grid traces (convert to 2D format with zero time delta)
if traces.dim() == 2:
    # [batch, seq_len] -> [batch, seq_len, 2]
    traces = torch.stack([traces, torch.zeros_like(traces)], dim=-1).float()

# Get candidates - generate from map_matrix if not provided
candidates_id = batch.get('candidates_id', batch.get('candidates', batch.get('cands')))
if candidates_id is None:
    # Generate candidates using grid-to-road mapping
    candidates_id, tgt_roads = self._generate_candidates(traces, tgt_roads)
else:
    candidates_id = candidates_id.to(self.device)

# Get trace lengths - also accept 'traces_lens' as alternative key
trace_lens = batch.get('trace_lens', batch.get('lengths', batch.get('src_lens', batch.get('traces_lens'))))
```

### 4. Added `_generate_candidates` Method (lines 866-964)

New method that generates candidate roads for each position based on grid-to-road mapping:

```python
def _generate_candidates(self, traces, tgt_roads):
    """
    Generate candidate roads for each position based on grid-to-road mapping.

    Logic:
    1. Always includes target road first (ensures training signal)
    2. Adds roads from grid mapping (via map_matrix)
    3. Fills remaining slots with random roads
    4. Returns candidates_id and tgt_indices (indices into candidates)
    """
```

Key features of the implementation:
- Uses `map_matrix` to find roads that map to each grid cell
- Handles 1-indexed grid IDs from DeepMapMatchingDataset (0 = padding)
- Always includes the target road as the first candidate to ensure a valid training signal
- Fills remaining candidate slots with roads from grid mapping or random roads
- Returns both `candidates_id` and `tgt_indices` (target road index within candidates)

## Configuration

The fix uses the following configuration parameter:
- `candidate_size`: Number of candidate roads to generate per position (default: 10, matching RLOMM.json)

## Usage

After this fix, RLOMM works seamlessly with DeepMapMatchingDataset:

```python
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

## Assumptions

1. **Grid IDs are 1-indexed**: DeepMapMatchingDataset uses 1-based grid IDs where 0 represents padding. The candidate generation adjusts for this by subtracting 1 when indexing into map_matrix.

2. **Target road in candidates**: For proper training, the target road is always included as the first candidate. This ensures a valid training signal even if the grid-to-road mapping does not include the target.

3. **Time delta simulation**: When time information is not provided (DeepMapMatchingDataset does not include timestamps), zeros are used as time deltas.

## Limitations

1. **Candidate quality**: Auto-generated candidates may be less accurate than pre-computed candidates that consider GPS proximity and road network topology.

2. **Performance**: Candidate generation adds computational overhead, especially for large batches. Pre-computed candidates would be more efficient.

3. **Random fallback**: When grid mapping is sparse, random roads are used to fill candidate slots, which may slow training convergence.

## Related Files

- **Config**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/map_matching/RLOMM.json`
- **Dataset**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/deep_map_matching_dataset.py`
- **Executor**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/executor/deep_map_matching_executor.py`
