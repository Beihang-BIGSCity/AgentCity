# MVGRL Migration Summary

## Model Information

- **Model Name**: MVGRL (Multi-View Graph Representation Learning)
- **Original Paper**: "Contrastive Multi-View Representation Learning on Graphs" (ICML 2020)
- **Original Repository**: https://github.com/kavehhassani/mvgrl
- **Task Type**: Traffic State Prediction (adapted from unsupervised graph representation learning)

## Original Model Overview

MVGRL is a contrastive learning framework for graph representation learning that:
- Uses two graph views: original adjacency matrix and PPR (Personalized PageRank) diffusion matrix
- Learns representations by maximizing mutual information between local (node-level) and global (graph-level) representations across views
- Originally designed for unsupervised node/graph classification

### Original Architecture Components:
1. **GCN**: Graph Convolutional Network layer with PReLU activation
2. **Readout**: Graph-level pooling via mean aggregation
3. **Discriminator**: Bilinear layer for contrastive scoring
4. **Model**: Two-branch GCN encoder (adjacency + diffusion views)

## Adaptation Approach

Since MVGRL is fundamentally an unsupervised contrastive learning model, significant adaptation was required to make it suitable for supervised traffic state prediction:

### Key Transformations

1. **Multi-View Spatial Encoding Preserved**:
   - Retained the core two-view (adjacency + PPR diffusion) GCN architecture
   - This captures both local connectivity and global diffusion patterns in traffic networks

2. **Temporal Modeling Added**:
   - Added temporal convolution layers or GRU for processing time-series traffic data
   - Processes spatial embeddings from each time step through temporal layers

3. **Prediction Head Added**:
   - Added MLP-based prediction head to output traffic forecasts
   - Maps temporal features to output window predictions

4. **Contrastive Loss Made Optional**:
   - Contrastive loss can be used as an auxiliary loss during training
   - Controlled via `use_contrastive_loss` config parameter

5. **LibCity Interface Implemented**:
   - Inherits from `AbstractTrafficStateModel`
   - Implements `forward()`, `predict()`, `calculate_loss()` methods
   - Handles LibCity's batch format `{'X': tensor, 'y': tensor}`

## Files Created/Modified

### New Files

1. **Model File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_speed_prediction/MVGRL.py`
   - Contains: `compute_ppr_matrix()`, `normalize_adj_matrix()`, `GCNLayer`, `MVGRLEncoder`, `Readout`, `Discriminator`, `TemporalConv`, `MVGRL`

2. **Config File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traffic_state_pred/MVGRL.json`

### Modified Files

1. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_speed_prediction/__init__.py`
   - Added MVGRL import and export

2. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
   - Added MVGRL to allowed models and task configuration

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 64 | Hidden dimension for GCN layers |
| `num_gcn_layers` | 2 | Number of GCN layers per view |
| `ppr_alpha` | 0.2 | Teleport probability for PPR computation |
| `use_contrastive_loss` | false | Whether to add contrastive auxiliary loss |
| `contrastive_weight` | 0.1 | Weight for contrastive loss when enabled |
| `temporal_type` | "conv" | Temporal modeling type ("conv" or "gru") |
| `dropout` | 0.1 | Dropout probability |

## Usage Example

```python
# Run with default config
python run_model.py --task traffic_state_pred --model MVGRL --dataset METR_LA

# With contrastive auxiliary loss
python run_model.py --task traffic_state_pred --model MVGRL --dataset METR_LA \
    --use_contrastive_loss true --contrastive_weight 0.1
```

## Model Architecture Diagram

```
Input: X (batch, time, nodes, features)
         |
         v
[Input Projection] -> (batch, time, nodes, hidden_dim)
         |
         v
For each time step:
    |
    +---> [GCN Branch 1 (Adjacency)] -> h_adj
    |
    +---> [GCN Branch 2 (PPR Diffusion)] -> h_diff
    |
    +---> [Concatenate] -> h_t (batch, nodes, hidden_dim*2)
         |
         v
[Stack temporal] -> (batch, time, nodes, hidden_dim*2)
         |
         v
[Temporal Conv/GRU] -> (batch*nodes, time, hidden_dim)
         |
         v
[Flatten + Prediction Head] -> (batch, output_window, nodes, output_dim)
```

## Limitations and Notes

1. **PPR Computation**: The PPR diffusion matrix is computed once at initialization using matrix inversion, which may be slow for very large graphs. For graphs with >5000 nodes, consider using approximate PPR methods.

2. **Memory Usage**: The model stores both adjacency and PPR diffusion matrices, doubling the graph memory footprint compared to single-view methods.

3. **Contrastive Loss**: When enabled, adds computational overhead for negative sample generation and discriminator scoring.

4. **Original vs Adapted**: The original MVGRL was designed for transductive node classification; this adaptation is for inductive traffic prediction. The spatial encoding mechanism is preserved, but the training objective is changed.

## Original Source Files

- Primary implementation: `/home/wangwenrui/shk/AgentCity/repos/mvgrl/node/train.py` (lines 89-122)
- Supporting classes: `GCN` (lines 10-40), `Readout` (lines 44-53), `Discriminator` (lines 57-86)
- Utility functions: `/home/wangwenrui/shk/AgentCity/repos/mvgrl/utils.py`
