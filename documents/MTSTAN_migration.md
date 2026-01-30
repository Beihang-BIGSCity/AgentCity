# MTSTAN Migration Documentation

## Model Overview

**MTSTAN (Multi-Task Spatio-Temporal Attention Network)** is a travel time estimation model that combines spatio-temporal attention mechanisms with multi-task learning.

### Original Implementation
- **Repository**: `./repos/MTSTAN`
- **Framework**: TensorFlow 1.12.0
- **Task Type**: Travel Time Estimation (ETA)

### Migrated Implementation
- **File Location**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/MTSTAN.py`
- **Framework**: PyTorch
- **Base Class**: `AbstractTrafficStateModel`

## Architecture Components

### 1. MultiHeadAttention
- Converted from TensorFlow's custom `multihead_attention` function
- Implements scaled dot-product attention with multi-head mechanism
- Supports residual connection and layer normalization

### 2. SpatialTransformer
- Applies multi-head attention across spatial dimension (nodes/sites)
- Stacks multiple attention blocks with feed-forward networks
- Adapted from `model/spatial_attention.py`

### 3. TemporalTransformer
- Applies multi-head attention across temporal dimension
- Returns attention weights for interpretability
- Adapted from `model/temporal_attention.py`

### 4. STBlock (Spatio-Temporal Block)
- Combines spatial and temporal attention
- Uses gated fusion mechanism to merge spatial and temporal features
- Adapted from `model/st_block.py`

### 5. BridgeTransformer
- Cross-attention between historical encodings and future queries
- Connects encoder outputs with decoder for prediction
- Adapted from `model/bridge.py`

### 6. InferenceModule
- CNN and dense layers for final prediction
- Produces speed predictions from encoded features
- Adapted from `model/inference.py`

### 7. EmbeddingLayer
- Position embedding for spatial locations
- Temporal embeddings for week, day, hour, minute
- Adapted from `model/embedding.py`

## Key Transformations

### TensorFlow to PyTorch Conversions

| TensorFlow | PyTorch |
|------------|---------|
| `tf.layers.dense` | `nn.Linear` |
| `tf.layers.conv1d` | `nn.Conv1d` |
| `tf.nn.conv2d` | `nn.Conv2d` |
| `tf.variable_scope` | `nn.Module` |
| `tf.nn.embedding_lookup` | `nn.Embedding` |
| `tf.nn.softmax` | `F.softmax` |
| `tf.matmul` | `torch.bmm` |

### Data Format Adaptation

**Original Format (TensorFlow)**:
- Speed: `[batch, input_length, site_num, feature_s]`
- Temporal indices: Separate placeholders for week, day, hour, minute
- Labels: `[batch, site_num, output_length]`

**LibCity Format (PyTorch)**:
- Input `X`: `[batch, input_length, num_nodes, feature_dim]`
- Target `y`: `[batch, output_length, num_nodes, feature_dim]`
- Uses batch dictionary with keys 'X' and 'y'

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `emb_size` | 64 | Embedding dimension |
| `site_num` | 108 | Number of spatial nodes |
| `input_length` | 12 | Input sequence length |
| `output_length` | 6 | Output prediction length |
| `num_heads` | 4 | Number of attention heads |
| `num_blocks` | 1 | Number of transformer blocks |
| `dropout` | 0.0 | Dropout probability |
| `feature_dim` | 1 | Input feature dimension |
| `alpha1` | 0.3 | Weight for speed prediction loss |
| `alpha2` | 0.4 | Weight for total travel time loss |
| `alpha3` | 0.3 | Weight for segment travel time loss |

## Missing Files from Original Repository

The following files were referenced but not found in the original repository:
- `model/utils.py` - Contains utility functions (`gatedFusion`, `FC`, `STEmbedding`)
- `model/trajectory_inf.py` - Contains `STANClass` for trajectory inference

These components were reconstructed based on:
1. Import statements and usage patterns in existing code
2. Related implementations (GMAN, ASTGCN models)
3. Standard transformer architecture patterns

## Assumptions and Limitations

1. **Simplified Multi-Task Learning**: The original model had three tasks (speed prediction, total travel time, segment travel time). The current implementation focuses on speed prediction as the primary task.

2. **Temporal Feature Extraction**: The model expects temporal features (week, day, hour, minute) in the batch. If not provided, default values are used.

3. **Trajectory Features**: The original model had trajectory-specific features (vehicle ID, type, distances). These are not used in the current LibCity adaptation as they require a specific encoder.

4. **Gated Fusion**: Implemented based on common gated fusion patterns as the original utils.py was not available.

## Files Created/Modified

### Created Files
1. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/MTSTAN.py` - Main model implementation
2. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/eta/MTSTAN.json` - Model configuration

### Modified Files
1. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/__init__.py` - Added MTSTAN import
2. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json` - Added MTSTAN to eta task

## Usage Example

```python
from libcity.pipeline import run_model

# Run MTSTAN model
run_model(
    task='eta',
    model_name='MTSTAN',
    dataset='Chengdu_Taxi_Sample1',
    config_file=None
)
```

## Testing Recommendations

1. Verify model instantiation with default configuration
2. Test forward pass with synthetic data
3. Validate loss calculation
4. Compare output dimensions with expected format

## Future Improvements

1. Add full multi-task learning support with trajectory features
2. Implement custom ETA encoder for trajectory data
3. Add attention weight visualization
4. Support for variable-length sequences
