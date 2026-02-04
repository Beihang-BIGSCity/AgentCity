# DeepMapMatchingDataset Migration Summary

## Overview

This document describes the creation of `DeepMapMatchingDataset`, a dataset class for deep learning-based map matching models like GraphMM in the LibCity framework.

## Problem Addressed

The existing `MapMatchingDataset` in LibCity was designed for traditional map matching algorithms and returns `None` for training and validation data. Deep learning models like GraphMM require:
- Training/validation/test data splits with proper DataLoaders
- Grid-based trajectory representations
- Graph structures (road adjacency, trace graphs)
- Proper batch formatting with padding

## Files Created/Modified

### New Files

1. **Dataset Class**
   - Path: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/deep_map_matching_dataset.py`
   - Contains: `DeepMapMatchingDataset` class, helper functions, and `MapMatchingTorchDataset` wrapper

2. **Dataset Configuration**
   - Path: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/data/DeepMapMatchingDataset.json`
   - Contains: Default configuration parameters for the dataset

### Modified Files

1. **Dataset Init**
   - Path: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/__init__.py`
   - Changes: Added import and export for `DeepMapMatchingDataset`

2. **Task Config**
   - Path: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
   - Changes: Updated GraphMM to use `DeepMapMatchingDataset`

3. **Executor**
   - Path: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/executor/deep_map_matching_executor.py`
   - Changes: Added support for dictionary batch format from DeepMapMatchingDataset

## Class Structure

```python
class DeepMapMatchingDataset(MapMatchingDataset):
    """
    Dataset class for deep learning-based map matching models.

    Provides:
    - Grid-based trajectory representations
    - Road network graph structures
    - Trace graph (grid connectivity based on trajectory flow)
    - Proper train/valid/test DataLoader objects
    """
```

## Key Features

### Data Processing

1. **GPS to Grid Conversion**
   - Uses 50m x 50m grid cells (configurable)
   - Converts GPS trajectories to grid cell sequences
   - Preserves GPS coordinates for attention mechanism

2. **Graph Construction**
   - **Road Graph**: Built from road network topology
     - Nodes: Road segments
     - Edges: Connections between adjacent roads
     - Features: Grid coordinates, GPS coordinates, distances

   - **Trace Graph**: Built from trajectory patterns
     - Nodes: Grid cells visited by trajectories
     - Edges: Transitions between grid cells
     - Weights: Transition frequency

3. **Mapping Matrix**
   - Maps grid cells to overlapping road segments
   - Handles singleton grids (grids with no road mapping)

### Batch Format

The dataset returns batches with the following structure:

```python
{
    'grid_traces': torch.LongTensor,      # (batch, max_trace_len)
    'tgt_roads': torch.LongTensor,        # (batch, max_road_len)
    'traces_gps': torch.FloatTensor,      # (batch, max_trace_len, 2)
    'sample_Idx': torch.LongTensor,       # (batch, max_trace_len)
    'traces_lens': list,                  # Original trace lengths
    'road_lens': list                     # Original road lengths
}
```

### Data Features

The `get_data_feature()` method returns:

```python
{
    'num_roads': int,                      # Number of road segments
    'num_grids': int,                      # Number of grid cells
    'road_adj': SparseTensor/Tensor,       # Road adjacency matrix
    'road_x': torch.Tensor,                # Road features (num_roads, 28)
    'trace_in_edge_index': torch.Tensor,   # Incoming edges for trace graph
    'trace_out_edge_index': torch.Tensor,  # Outgoing edges for trace graph
    'trace_weight': torch.Tensor,          # Edge weights for trace graph
    'map_matrix': torch.Tensor,            # (num_grids, num_roads) mapping
    'A_list': torch.Tensor,                # Adjacency polynomial for CRF
    'singleton_grid_mask': torch.Tensor,   # Indices of singleton grids
    'singleton_grid_location': torch.Tensor, # Locations of singleton grids
    'grid2traceid_dict': dict,             # Grid to trace ID mapping
    'traceid2grid_dict': dict,             # Trace ID to grid mapping
    'with_time': bool,                     # Time information flag
    'delta_time': bool                     # Delta time flag
}
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| grid_size | 50 | Grid cell size in meters |
| train_rate | 0.7 | Training data ratio |
| eval_rate | 0.1 | Validation data ratio |
| batch_size | 32 | Training batch size |
| eval_batch_size | 64 | Evaluation batch size |
| downsample_rate | 0.5 | GPS trajectory downsampling rate |
| max_road_len | 25 | Maximum road sequence length |
| min_road_len | 15 | Minimum road sequence length |
| layer | 2 | Adjacency polynomial layers |
| gamma | 1.0 | CRF penalty parameter |
| num_workers | 0 | DataLoader workers |
| shuffle | true | Shuffle training data |
| cache_dataset | true | Cache processed data |

## Usage Example

```python
from libcity.data.dataset import DeepMapMatchingDataset

# Configuration
config = {
    'dataset': 'Seattle',
    'grid_size': 50,
    'train_rate': 0.7,
    'eval_rate': 0.1,
    'batch_size': 32,
    'cache_dataset': True
}

# Create dataset
dataset = DeepMapMatchingDataset(config)

# Get DataLoaders
train_loader, eval_loader, test_loader = dataset.get_data()

# Get data features for model initialization
data_feature = dataset.get_data_feature()
```

## Original Source References

The implementation is adapted from:
- `repos/GraphMM/data_loader.py` - Dataset loading and padding
- `repos/GraphMM/graph_data.py` - Graph data structures
- `repos/GraphMM/data_preprocess/utils.py` - GPS to grid conversion
- `repos/GraphMM/data_preprocess/build_trace_graph.py` - Trace graph construction
- `repos/GraphMM/data_preprocess/build_road_graph.py` - Road graph construction
- `repos/GraphMM/data_preprocess/build_grid_road_matrix.py` - Mapping matrix
- `repos/GraphMM/data_preprocess/build_A.py` - Adjacency matrix
- `repos/GraphMM/data_preprocess/data_process.py` - Data splitting

## Dependencies

- torch
- torch_sparse (optional, for SparseTensor support)
- networkx
- numpy
- pandas

## Limitations

1. The dataset requires proper LibCity format data files (.geo, .rel, .dyna, .usr)
2. Ground truth routes are required for training (in _truth.dyna file)
3. Large datasets may require significant preprocessing time on first run
4. The grid size is fixed during preprocessing; changing it requires cache invalidation

## Future Improvements

1. Support for variable grid sizes per region
2. Pre-computed graph embeddings for faster training
3. Support for incremental data updates
4. Integration with other deep learning map matching models (DeepMM, DiffMM, TRMMA)
