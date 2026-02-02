# TrajSDE Model Migration Summary

## Overview
Migration of TrajSDE (Stochastic Differential Equation-based Trajectory Prediction) model to LibCity framework.

## Source Information
- **Original Repository**: `/home/wangwenrui/shk/AgentCity/repos/TrajSDE`
- **Main Model Class**: `PredictionModelSDENet` in `models/model_base_mix_sde.py`
- **Target Location**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/TrajSDE.py`

## Critical Bug Fix (2026-02-02)

### Problem
The original adapted model raised:
```
KeyError: 'rotate_mat is not in the batch'
```
at line 1297 when trying to set `data['rotate_mat'] = None`.

### Root Cause
LibCity's `Batch` class has an immutable key set. The `__setitem__` method raises a `KeyError` if you try to set a key that doesn't exist in `feature_name`:
```python
def __setitem__(self, key, value):
    if key in self.data:
        self.data[key] = value
    else:
        raise KeyError('{} is not in the batch'.format(key))
```

### Solution
Completely rewrote the model to:
1. Never modify the input batch object
2. Extract data into local variables in `_prepare_data()`
3. Use simplified components that don't require graph structures or lane information

## Architecture (Updated for LibCity)

### New Simplified Components

| Component | Class Name | Description |
|-----------|------------|-------------|
| Encoder | `LocalEncoderSimple` | SDE-based temporal encoding without graph message passing |
| AA Encoder | `AAEncoderSimple` | Self-attention based Actor-Actor interaction |
| Aggregator | `GlobalInteractorSimple` | Standard multi-head attention for global interactions |
| Decoder | `SDEDecoderSimple` | POI probability prediction instead of continuous coordinates |

### Removed Components
- **AL Encoder**: Lane information not available in POI data
- **Graph Message Passing**: Replaced with standard attention
- **Rotation Normalization**: Not applicable to discrete POI prediction
- **Continuous Trajectory Output**: Changed to discrete location probabilities

## Key Dependencies
- **Required**: PyTorch
- **Optional**: torchsde >= 0.2.5 (model provides fallback without it)
- **Optional**: PyTorch Geometric (no longer required for core functionality)

## Migration Changes

### Data Format Adaptation

**Original TrajSDE expected:**
- Autonomous vehicle trajectory data
- Graph structures with edge indices
- Lane information (positions, paddings, actor indices, vectors)
- Rotation matrices and angles
- Continuous 2D coordinates

**LibCity provides:**
- POI check-in data with discrete locations
- `current_loc`: Location sequence (batch_size, seq_len)
- `current_tim`: Time sequence (batch_size, seq_len)
- `target`: Target location (batch_size,)

### New Embedding Layers
```python
self.loc_embedding = nn.Embedding(self.loc_size, self.embed_dim, padding_idx=self.loc_pad)
self.tim_embedding = nn.Embedding(self.tim_size, self.embed_dim // 2)
self.input_proj = nn.Linear(self.embed_dim + self.embed_dim // 2, self.embed_dim)
```

### Loss Function Change
- **Before**: Laplace NLL loss for continuous coordinates
- **After**: Cross-entropy loss for POI classification with mode probability weighting

## Configuration Parameters (Updated)

| Parameter | Default | Description |
|-----------|---------|-------------|
| embed_dim | 64 | Embedding dimension |
| num_modes | 6 | Number of prediction modes |
| num_heads | 8 | Number of attention heads |
| dropout | 0.1 | Dropout rate |
| historical_steps | 10 | Number of historical time steps |
| hidden_size | 128 | Hidden layer size |
| num_global_layers | 3 | Number of global interaction layers |
| sde_layers | 2 | Number of SDE layers |
| rtol | 0.001 | Relative tolerance for SDE solver |
| atol | 0.001 | Absolute tolerance for SDE solver |
| step_size | 0.1 | Step size for SDE solver |

### Removed Parameters
- `future_steps`: Now fixed at 1 for single-step POI prediction
- `local_radius`: Not applicable without graph structure
- `node_dim`, `edge_dim`: Replaced with embedding dimensions
- `min_scale`: Not used for classification
- `rotate`, `is_gtabs`: Not applicable to POI data

## Usage Example

```python
from libcity.model.trajectory_loc_prediction import TrajSDE

# Configuration
config = {
    'device': 'cuda',
    'embed_dim': 64,
    'num_modes': 6,
    'num_heads': 8,
    'historical_steps': 10,
}

# Data features (from dataset)
data_feature = {
    'loc_size': 5000,
    'tim_size': 48,
    'uid_size': 1000,
    'loc_pad': 0
}

# Initialize model
model = TrajSDE(config, data_feature)

# Forward pass
output = model(batch)
# output['loc_logits']: (num_modes, batch_size, loc_size)
# output['pi']: (batch_size, num_modes)

# Prediction
log_probs = model.predict(batch)  # (batch_size, loc_size)

# Loss calculation
loss = model.calculate_loss(batch)  # Cross-entropy based loss
```

## Data Format Requirements

The model now expects LibCity's standard trajectory batch format:

```python
batch = {
    'current_loc': Tensor,   # [batch_size, seq_len] - location IDs
    'current_tim': Tensor,   # [batch_size, seq_len] - time slot IDs
    'target': Tensor,        # [batch_size] - target location ID
}
```

## Limitations and Notes

1. **Simplified Model**: The model no longer uses graph-based message passing or lane information.

2. **SDE Optional**: The model works without torchsde, using a simple fallback for temporal encoding.

3. **Multi-Modal Output**: Maintains multi-modal prediction capability, but may not be beneficial for all POI datasets.

4. **GPU Memory**: SDE integration can be memory-intensive. Consider reducing `num_modes` or `historical_steps` if needed.

## Files Modified

1. **Rewritten**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/TrajSDE.py`
2. **Already registered**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py`
3. **Updated**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/TrajSDE.json`

## Original Files Referenced

- `repos/TrajSDE/models/model_base_mix_sde.py` - Main model class
- `repos/TrajSDE/models/encoders/enc_hivt_nusargo_sde_sep2.py` - Encoder
- `repos/TrajSDE/models/aggregators/agg_hivt.py` - Aggregator
- `repos/TrajSDE/models/decoders/dec_hivt_nusargo_sde.py` - Decoder
- `repos/TrajSDE/models/utils/util.py` - Utilities
- `repos/TrajSDE/models/utils/embedding.py` - Embedding layers
- `repos/TrajSDE/models/utils/ode_utils.py` - ODE utilities
- `repos/TrajSDE/models/utils/sde_utils.py` - SDE utilities
- `repos/TrajSDE/models/utils/sdeint.py` - Custom SDE integration
- `repos/TrajSDE/losses/laplace_nll_loss.py` - Loss function
