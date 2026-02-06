# RLOMM Model Migration Summary

## Overview

**Model Name**: RLOMM (Reinforcement Learning for Online Map Matching)

**Task Type**: Map Matching

**Base Class**: AbstractModel

**Source Repository**: `/home/wangwenrui/shk/AgentCity/repos/RLOMM`

**Target Location**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/RLOMM.py`

## Original Model Architecture

RLOMM uses a Double DQN architecture with contrastive learning for map matching GPS trajectories to road segment sequences. The original implementation consists of:

### Key Components

1. **RoadGIN** (`model/road_gin.py`)
   - Graph Isomorphism Network for road network encoding
   - Uses PyTorch Geometric's GINConv layers
   - Multi-layer with max pooling across layers

2. **TraceGCN** (`model/trace_gcn.py`)
   - Directed GCN for GPS trace graph encoding
   - Bidirectional encoding (separate GCNs for in/out edges)
   - Concatenates incoming and outgoing embeddings

3. **QNetwork** (`model/Q_network.py`)
   - Combines road graph and trace graph encodings
   - RNN-based sequence modeling for traces and matched segments
   - Attention mechanism for candidate selection
   - Outputs Q-values for each candidate action

4. **MMAgent** (`model/mm_agent.py`)
   - Main reinforcement learning agent
   - Implements Double DQN with experience replay
   - Manages main and target networks
   - Computes shaped rewards based on:
     - Correct match reward
     - Continuous success bonus
     - Connectivity penalty
     - Detour penalty

5. **Memory** (`memory.py`)
   - Experience replay buffer using named tuples
   - Stores (state, action, next_state, reward) transitions

## Adaptations for LibCity

### Structural Changes

1. **Unified File**: All components consolidated into single `RLOMM.py`
2. **Base Class**: Inherits from `AbstractModel` instead of `nn.Module`
3. **Constructor**: Changed signature to `__init__(self, config, data_feature)`
4. **Required Methods**: Implemented `predict()` and `calculate_loss()`

### Data Format Adaptations

The model now accepts LibCity batch dictionaries with the following keys:

| Original Key | LibCity Key Options | Shape |
|-------------|---------------------|-------|
| traces | traces, X, input_traces | [batch, seq_len, 2] |
| tgt_roads | tgt_roads, y, target, output_trg | [batch, seq_len] |
| candidates_id | candidates_id, candidates, cands | [batch, seq_len, num_cands] |
| trace_lens | trace_lens, lengths, src_lens | [batch] |

### Graph Data Handling

The model requires road and trace graph data, which can be provided in two ways:

1. **Via data_feature dictionary**:
   - `road_x`: Road node features
   - `road_adj`: Road adjacency (SparseTensor)
   - `trace_in_edge_index`, `trace_out_edge_index`: Trace graph edges
   - `trace_weight`, `map_matrix`: Additional graph data
   - `connectivity_distances`: Precomputed road distances

2. **Via set_graphs() method**:
   - Executor can provide pre-loaded `RoadGraphData` and `TraceGraphData` objects

### Training Adaptations

The `calculate_loss()` method combines:
1. **RL Loss (Double DQN)**: Smooth L1 loss on TD error
2. **Contrastive Loss**: Aligns trace and road embeddings

Training updates target network periodically based on `target_update_interval`.

## Configuration Parameters

```json
{
  "road_emb_dim": 128,
  "traces_emb_dim": 128,
  "num_layers": 3,
  "gin_depth": 3,
  "gcn_depth": 3,
  "gamma": 0.8,
  "match_interval": 4,
  "memory_capacity": 100,
  "target_update_interval": 10,
  "optimize_batch_size": 32,
  "correct_reward": 5.0,
  "mask_reward": 0.0,
  "continuous_success_reward": 1.0,
  "connectivity_reward": 1.0,
  "detour_penalty": 1.0,
  "lambda_ctr": 0.1,
  "batch_size": 512,
  "learning_rate": 0.001,
  "max_epoch": 100
}
```

## Dependencies

- PyTorch (required)
- PyTorch Geometric (optional, for full GNN functionality)
- torch_sparse (optional, for SparseTensor support)

The model includes fallback implementations when PyTorch Geometric is not available, using simple MLP layers instead of GNN layers.

## Files Modified/Created

1. **Created**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/RLOMM.py`
   - Complete model implementation with all components

2. **Modified**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/__init__.py`
   - Added RLOMM import and registration

3. **Updated**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/map_matching/RLOMM.json`
   - Configuration file with model hyperparameters

## Usage Example

```python
from libcity.model.map_matching import RLOMM

# Configuration
config = {
    'device': 'cuda',
    'road_emb_dim': 128,
    'traces_emb_dim': 128,
    'gamma': 0.8,
    'match_interval': 4,
    # ... other parameters
}

# Data features (including graph data)
data_feature = {
    'num_roads': 8533,
    'num_grids': 10551,
    'road_x': road_features_tensor,
    'road_adj': road_adjacency_sparse,
    'trace_in_edge_index': trace_in_edges,
    'trace_out_edge_index': trace_out_edges,
    'trace_weight': edge_weights,
    'map_matrix': grid_to_road_mapping,
}

# Create model
model = RLOMM(config, data_feature)

# Training
loss = model.calculate_loss(batch)

# Inference
predictions = model.predict(batch)
```

## Notes and Limitations

1. **RL Training**: The model uses online RL training, which may require special executor handling for proper episode-based training.

2. **Graph Data**: Road and trace graph data must be provided either through data_feature or via set_graphs() before training/inference.

3. **Memory Management**: Experience replay buffer is stored in model. For distributed training, consider clearing memory between epochs.

4. **Fallback Mode**: When PyTorch Geometric is not installed, the model uses MLP fallbacks instead of GNN layers, which may reduce performance.

## Migration Date

2024-02-04
