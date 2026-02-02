# JGRM Model Migration Summary

## Model Information

**Model Name**: JGRM (Joint GPS and Route Modeling)

**Task Type**: Trajectory Representation Learning (placed in trajectory_loc_prediction for framework compatibility)

**Original Repository**: `/home/wangwenrui/shk/AgentCity/repos/JGRM/`

**Original Files**:
- `JGRM.py` - Main model class (JGRMModel)
- `basemodel.py` - Base class definition
- `cl_loss.py` - GPS-Route matching loss function
- `dcl.py` - Decoupled Contrastive Loss
- `dataloader.py` - Data loading and preprocessing
- `JGRM_train.py` - Training script
- `config/chengdu.json` - Configuration file

## LibCity Adapted Files

**Model File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/JGRM.py`

**Config File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/JGRM.json`

**Registration**: Updated `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py`

## Architecture Overview

JGRM is a dual-branch trajectory representation learning model:

### 1. Route Encoding Branch
- **Graph Encoder (GAT)**: Encodes road network topology using Graph Attention Networks
- **Transformer Encoder**: Processes route segment sequences with temporal features (weekday, minute, delta)
- **Output**: Per-segment and trajectory-level route representations

### 2. GPS Encoding Branch
- **Intra-segment GRU**: Processes GPS points within each road segment
- **Inter-segment GRU**: Models relationships across road segments
- **Output**: Per-segment and trajectory-level GPS representations

### 3. Joint Encoding (Shared Transformer)
- Combines GPS and route representations with modal embeddings
- Enables cross-modal learning and alignment
- Produces final joint representations for both modalities

## Training Objectives

Three loss functions are combined during training:

1. **Route MLM Loss**: Masked Language Model loss for predicting masked route segments from route representations
2. **GPS MLM Loss**: Masked Language Model loss for predicting masked segments from GPS representations
3. **GPS-Route Matching Loss**: Contrastive loss for aligning GPS and route trajectory representations

Combined loss formula: `(route_mlm_loss + gps_mlm_loss + 2 * match_loss) / 3`

## Key Adaptations Made

### 1. Class Inheritance
- Changed from `BaseModel` to `AbstractModel` (LibCity convention)
- Constructor signature: `__init__(self, config, data_feature)`

### 2. Integrated Sub-modules
All sub-modules are now defined within the same file:
- `GraphEncoder`: GAT-based road network encoder
- `TransformerModel`: Vanilla transformer encoder
- `IntervalEmbedding`: Continuous time interval embedding
- `DCL`: Decoupled Contrastive Loss

### 3. Loss Functions
- `get_traj_match_loss()`: GPS-Route matching loss method
- `random_mask()`: MLM masking logic integrated as class method

### 4. LibCity Interface Methods
- `forward(batch)`: Full forward pass returning all representations
- `predict(batch)`: Returns trajectory embeddings for inference
- `calculate_loss(batch)`: Computes combined training loss
- `get_embeddings(batch)`: Convenience method for representation extraction

### 5. Data Handling
Adapted to use LibCity's batch dictionary format with keys:
- `route_data`: Temporal features (batch, seq_len, 3)
- `route_assign_mat`: Route segment indices
- `gps_data`: GPS point features
- `gps_assign_mat`: GPS-to-segment assignments
- `gps_length`: Number of GPS points per segment

### 6. Device Handling
- Replaced `.cuda()` calls with configurable device from `config['device']`
- Edge index moved to device dynamically

### 7. Optional Dependencies
- torch_geometric (for GAT): Falls back to MLP if not available
- sklearn (for kmeans): Not required for this model

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| route_max_len | 100 | Maximum route sequence length |
| road_feat_num | 1 | Number of road features |
| road_embed_size | 128 | Road segment embedding dimension |
| gps_feat_num | 8 | Number of GPS point features |
| gps_embed_size | 128 | GPS embedding dimension |
| route_embed_size | 128 | Route representation dimension |
| hidden_size | 256 | Hidden layer dimension |
| drop_edge_rate | 0.1 | GAT edge dropout rate |
| drop_route_rate | 0.1 | Route encoder dropout rate |
| drop_road_rate | 0.1 | Shared transformer dropout rate |
| mask_length | 2 | Consecutive masked segment length |
| mask_prob | 0.2 | Masking probability for MLM |
| tau | 0.07 | Temperature for contrastive loss |
| mode | "x" | "p" for pretrain embeddings, "x" for GAT |
| mlm_loss_weight | 1.0 | Weight for MLM losses |
| match_loss_weight | 2.0 | Weight for matching loss |
| queue_size | 2048 | Size of contrastive learning queue |
| route_transformer_layers | 4 | Number of route Transformer layers |
| route_transformer_heads | 8 | Number of route Transformer heads |
| shared_transformer_layers | 2 | Number of shared Transformer layers |
| shared_transformer_heads | 4 | Number of shared Transformer heads |

## Required Data Features

| Feature | Type | Description |
|---------|------|-------------|
| vocab_size | int | Number of road segments in network |
| edge_index | tensor/ndarray | Road network adjacency (2, num_edges) |
| route_max_len | int | Maximum route sequence length |

## Expected Batch Format

```python
batch = {
    'route_data': torch.Tensor,      # (batch, seq_len, 3) - weekday, minute, delta
    'route_assign_mat': torch.Tensor, # (batch, seq_len) - segment indices
    'gps_data': torch.Tensor,         # (batch, gps_len, gps_feat_num)
    'gps_assign_mat': torch.Tensor,   # (batch, gps_len) - segment assignments
    'gps_length': torch.Tensor        # (batch, seq_len) - GPS points per segment
}
```

## Usage Notes

### For Training
```python
model = JGRM(config, data_feature)
loss = model.calculate_loss(batch)
loss.backward()
```

### For Inference (Embedding Extraction)
```python
model.eval()
embeddings = model.predict(batch)  # (batch_size, hidden_size)
```

## Limitations and Assumptions

1. **Data Preprocessing**: The model expects pre-processed trajectory data with GPS-to-segment assignments. LibCity's default trajectory datasets may need custom preprocessing.

2. **Road Network Graph**: The model requires road network adjacency information (`edge_index`). This needs to be provided in `data_feature` or loaded separately.

3. **Task Category**: JGRM is fundamentally a representation learning model, not a location prediction model. It's placed in `trajectory_loc_prediction` for framework compatibility, but the primary output is trajectory embeddings.

4. **torch_geometric Dependency**: For full GAT functionality, torch_geometric should be installed. The model includes a fallback to simple MLP layers.

5. **Batch Size**: The GPS-Route matching loss uses hard negative mining which assumes a reasonable batch size (recommended >= 32).

## Differences from Original Implementation

1. **Unified File**: All components combined into single file vs. multiple source files
2. **Device Agnostic**: Uses configurable device instead of hardcoded CUDA
3. **Masking in Training**: Masking is applied within `calculate_loss()` rather than externally
4. **Inference Mode**: Added `predict()` method that returns embeddings without masking
5. **Loss Computation**: All loss computations integrated into `calculate_loss()`

## Testing Recommendations

1. Verify torch_geometric installation for GAT functionality
2. Test with small batch sizes first (GPU memory considerations)
3. Validate data preprocessing matches expected format
4. Check edge_index format (COO format expected: shape (2, num_edges))
