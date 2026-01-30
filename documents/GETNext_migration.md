# GETNext Model Migration to LibCity

## Overview

This document describes the adaptation of the GETNext model (Trajectory Flow Map Enhanced Transformer for Next POI Recommendation) from its original PyTorch implementation to the LibCity framework.

**Original Paper**: Yang et al. "GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation" (SIGIR 2022)

**Original Repository**: https://github.com/songyangme/GETNext

## Source Files

Original model location:
- `/home/wangwenrui/shk/AgentCity/repos/GETNext/model.py`

Supporting files analyzed:
- `/home/wangwenrui/shk/AgentCity/repos/GETNext/train.py`
- `/home/wangwenrui/shk/AgentCity/repos/GETNext/dataloader.py`
- `/home/wangwenrui/shk/AgentCity/repos/GETNext/utils.py`
- `/home/wangwenrui/shk/AgentCity/repos/GETNext/param_parser.py`

## Output Files

LibCity model file:
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/GETNext.py`

Configuration file:
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/GETNext.json`

Modified files:
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`

## Model Architecture

The GETNext model consists of 8 main components that were all consolidated into a single LibCity model class:

1. **GCN (Graph Convolutional Network)**: Generates POI embeddings from the trajectory flow graph
2. **NodeAttnMap**: Creates attention maps for graph-based prediction adjustment
3. **UserEmbeddings**: Embedding layer for user IDs
4. **CategoryEmbeddings**: Embedding layer for POI categories
5. **Time2Vec**: Time encoding using sine/cosine activation
6. **FuseEmbeddings**: Fusion layer for combining embeddings
7. **PositionalEncoding**: Standard transformer positional encoding
8. **TransformerSeqModel**: Main transformer encoder for sequence modeling

## Key Adaptations

### 1. Class Structure
- Original: 8 separate model classes initialized and trained independently
- LibCity: Single `GETNext` class inheriting from `AbstractModel` that contains all sub-models

### 2. Data Handling
- Original: Custom dataset classes with trajectory data parsed from CSV files
- LibCity: Uses `data_feature` dictionary to get dimensions and graph data
  - `loc_size`: Number of POI locations
  - `uid_size`: Number of users
  - `cat_size`: Number of categories
  - `graph_A`: Adjacency matrix (optional)
  - `graph_X`: Node features (optional)
  - `poi_idx2cat_idx`: POI to category mapping

### 3. Batch Format
- Original: Custom collate function returning lists of tuples
- LibCity: Dictionary-based batch with keys:
  - `current_loc`: (batch_size, seq_len) - POI indices
  - `current_tim`: (batch_size, seq_len) - time values
  - `uid`: (batch_size,) - user indices
  - `target`: (batch_size,) or (batch_size, seq_len) - target POI indices

### 4. Loss Computation
- Original: Separate loss computation in training loop
- LibCity: Implemented `calculate_loss()` method with multi-task loss:
  - Cross-entropy for POI prediction
  - MSE for time prediction
  - Cross-entropy for category prediction
  - Combined loss = loss_poi + time_loss_weight * loss_time + loss_cat

### 5. Prediction
- Original: Returns predictions for all positions
- LibCity: Implemented `predict()` method returning predictions for the last position

### 6. Graph Data
- Original: Loaded from CSV files and processed externally
- LibCity: Can be passed through `data_feature` or uses identity matrix as default

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| poi_embed_dim | 128 | POI embedding dimension |
| user_embed_dim | 128 | User embedding dimension |
| time_embed_dim | 32 | Time embedding dimension |
| cat_embed_dim | 32 | Category embedding dimension |
| gcn_nhid | [32, 64] | GCN hidden layer dimensions |
| gcn_dropout | 0.3 | GCN dropout rate |
| transformer_nhid | 1024 | Transformer FFN hidden dimension |
| transformer_nlayers | 2 | Number of transformer layers |
| transformer_nhead | 2 | Number of attention heads |
| transformer_dropout | 0.3 | Transformer dropout rate |
| node_attn_nhid | 128 | Node attention hidden dimension |
| time_loss_weight | 10.0 | Weight for time prediction loss |

## Usage

```python
from libcity.model.trajectory_loc_prediction import GETNext

# Configuration
config = {
    'device': 'cuda',
    'poi_embed_dim': 128,
    'user_embed_dim': 128,
    'time_embed_dim': 32,
    'cat_embed_dim': 32,
    'transformer_nlayers': 2,
    'transformer_nhead': 2,
}

# Data features
data_feature = {
    'loc_size': 1000,
    'uid_size': 100,
    'cat_size': 50,
    'tim_size': 48,
    'poi_idx2cat_idx': {...},  # POI to category mapping
    'graph_A': adj_matrix,      # Optional
    'graph_X': node_features,   # Optional
}

# Initialize model
model = GETNext(config, data_feature)

# Training
loss = model.calculate_loss(batch)

# Prediction
predictions = model.predict(batch)
```

## Limitations and Notes

1. **Graph Data**: If graph_A and graph_X are not provided in data_feature, identity matrices are used as defaults. For best performance, provide the actual trajectory flow graph.

2. **Category Mapping**: If poi_idx2cat_idx is not provided, a dummy mapping (POI_idx % num_cats) is used.

3. **Time2Vec**: Uses sine activation by default (as in original implementation).

4. **Batch Processing**: The current implementation processes POI-to-category mapping in a loop, which may be slower for large batches. Future optimization could vectorize this operation.

5. **Multi-task Learning**: The model supports multi-task prediction (POI, time, category), but only POI predictions are returned by the `predict()` method.

## Testing

To verify the model imports correctly:

```python
import torch
from libcity.model.trajectory_loc_prediction import GETNext

config = {'device': 'cpu'}
data_feature = {'loc_size': 100, 'uid_size': 10, 'cat_size': 5, 'tim_size': 48}

model = GETNext(config, data_feature)
print(f"Model created successfully with {sum(p.numel() for p in model.parameters())} parameters")
```
