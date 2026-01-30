# UrbanDiT Model Migration Documentation

## Overview

This document describes the migration of the UrbanDiT model to the LibCity framework.

**Original Paper**: "UrbanDiT: A Foundation Model for Open-World Urban Spatio-Temporal Learning"

**Original Repository**: https://github.com/YuanYuan98/UrbanDiT

## Source Files

### Original Implementation
- **Main Model**: `repos/UrbanDiT/src/models.py` (lines 288-638)
  - `UrbanDiT` class: Core diffusion transformer
  - `UrbanDiTBlock`: Transformer block with temporal and spatial attention
  - `FinalLayer`: Output projection layer
  - `TimestepEmbedder`: Diffusion timestep embedding
  - `Memory`: Prompt memory module

- **Embeddings**: `repos/UrbanDiT/src/Embed.py`
  - `TokenEmbedding_S`: Spatial token embedding (2D conv)
  - `TokenEmbedding_ST`: Spatio-temporal token embedding (3D conv)
  - `TimeEmbedding`: Temporal embedding with weekday/hour
  - `GraphEmbedding`: Graph-based embedding for non-grid data

- **Diffusion**: `repos/UrbanDiT/src/diffusion/`
  - `gaussian_diffusion.py`: Gaussian diffusion process
  - `respace.py`: Timestep respacing utilities
  - `diffusion_utils.py`: Helper functions

- **Masking**: `repos/UrbanDiT/src/mask_generator.py`
  - `VideoMaskGenerator`: Multi-task mask generation

## Adapted Files

### LibCity Implementation
- **Model**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_speed_prediction/UrbanDiT.py`
- **Config**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traffic_state_pred/UrbanDiT.json`

## Key Adaptations

### 1. Class Structure
- Inherits from `AbstractTrafficStateModel` instead of `nn.Module`
- Implements required LibCity methods: `__init__`, `forward`, `predict`, `calculate_loss`

### 2. Data Format Conversion
- **Input**: LibCity uses `(B, T, N, F)` format (batch, time, nodes, features)
- **Model expects**: `(B, T, C, H, W)` format (batch, time, channels, height, width)
- Added `_reshape_to_grid()` and `_reshape_from_grid()` methods for conversion
- Auto-detects grid dimensions from `num_nodes`

### 3. Diffusion Process
- **Original**: Uses `diffusers` library with `FlowMatchEulerDiscreteScheduler`
- **Adapted**: Implemented standalone `FlowMatchingScheduler` class
- Simplified flow matching with configurable steps

### 4. Components Included
All necessary components are self-contained in the single file:
- Positional embeddings (1D and 2D sinusoidal)
- Token embeddings (spatial and spatio-temporal)
- Time embeddings
- Timestep embeddings for diffusion
- Memory/prompt modules
- Attention modules (from timm)
- Transformer blocks with adaptive layer norm
- Flow matching scheduler
- Video mask generator

### 5. Simplifications
- Focused on forecasting task (mask_idx=0)
- Removed graph-specific data handling for initial version
- Removed classifier-free guidance
- Simplified prompt system (can be enabled via config)

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 384 | Transformer hidden dimension |
| `depth` | 6 | Number of transformer blocks |
| `num_heads` | 6 | Number of attention heads |
| `mlp_ratio` | 2 | MLP hidden ratio |
| `patch_size` | 2 | Spatial patch size |
| `t_patch_len` | 2 | Temporal patch length |
| `stride` | 2 | Temporal stride |
| `diffusion_steps` | 200 | Number of diffusion training steps |
| `num_inference_steps` | 20 | Number of inference denoising steps |
| `is_prompt` | 0 | Enable prompt learning (0=off, 1=on) |
| `grid_height` | null | Grid height (auto-detected if null) |
| `grid_width` | null | Grid width (auto-detected if null) |

## Model Variants

Three pre-configured variants are available:
- `UrbanDiT_S_1`: depth=4, hidden_size=256, num_heads=4 (smallest)
- `UrbanDiT_S_2`: depth=6, hidden_size=384, num_heads=6 (default)
- `UrbanDiT_S_3`: depth=12, hidden_size=384, num_heads=12 (largest)

## Dependencies

Required packages:
- `torch`: PyTorch deep learning framework
- `einops`: Tensor operations (used for rearrange)
- `numpy`: Numerical operations

## Usage Example

```python
from libcity.model.traffic_speed_prediction import UrbanDiT

# Via LibCity config
config = {
    'device': torch.device('cuda:0'),
    'input_window': 12,
    'output_window': 12,
    'hidden_size': 384,
    'depth': 6,
    'num_heads': 6,
    'diffusion_steps': 200,
    'num_inference_steps': 20,
}

data_feature = {
    'num_nodes': 1024,  # 32x32 grid
    'output_dim': 1,
    'feature_dim': 3,
    'scaler': scaler_instance,
}

model = UrbanDiT(config, data_feature)
```

## Limitations and Future Work

1. **Grid Data Only**: Current version is optimized for grid-based data. Graph support requires additional work.

2. **Single Task**: Focused on forecasting. Multi-task capabilities (interpolation, imputation) can be added by extending mask generation.

3. **No EMA**: Original uses Exponential Moving Average for inference. Can be added via LibCity executor.

4. **No Classifier-Free Guidance**: Can be added for conditional generation tasks.

## Testing

To test the model import:
```python
from libcity.model.traffic_speed_prediction import UrbanDiT
print(UrbanDiT is not None)  # Should print True
```
