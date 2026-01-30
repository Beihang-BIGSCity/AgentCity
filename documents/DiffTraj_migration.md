# DiffTraj Migration to LibCity

## Overview

This document describes the adaptation of the DiffTraj model (a diffusion-based trajectory generation model) to the LibCity framework.

## Source Repository

**Original Location**: `/home/wangwenrui/shk/AgentCity/repos/DiffTraj`

**Key Source Files**:
- `utils/Traj_UNet.py` - Main model architecture (Guide_UNet, Model, WideAndDeep)
- `utils/EMA.py` - Exponential Moving Average helper
- `utils/config.py` - Default configuration parameters
- `main.py` - Training loop with diffusion process

## Adapted Files

### Model File
**Location**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/DiffTraj.py`

### Configuration File
**Location**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/DiffTraj.json`

### Registry Update
**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py`

Added import and registration for `DiffTraj`.

## Key Adaptations

### 1. Class Structure

**Original**: Standalone `Guide_UNet` class with custom config object (SimpleNamespace)

**Adapted**: `DiffTraj` class inheriting from `AbstractModel` with LibCity-compatible interface

```python
class DiffTraj(AbstractModel):
    def __init__(self, config, data_feature):
        # Uses config.get() pattern for all hyperparameters
        ...

    def forward(self, batch):
        # Returns (predicted_noise, target_noise) tuple
        ...

    def predict(self, batch):
        # Generates trajectories via reverse diffusion
        ...

    def calculate_loss(self, batch):
        # MSE loss between predicted and actual noise
        ...
```

### 2. Configuration Pattern

**Original**: Nested SimpleNamespace objects
```python
config.model.ch
config.diffusion.num_diffusion_timesteps
```

**Adapted**: LibCity config.get() pattern with defaults
```python
self.ch = config.get('ch', 128)
self.num_timesteps = config.get('num_diffusion_timesteps', 500)
```

### 3. Batch Data Handling

**Original**: Separate tensors for trajectory and attributes
```python
traj = ...  # [batch, 2, length]
head = ...  # [batch, 8]
```

**Adapted**: Dictionary-based batch structure
```python
batch['traj'] or batch['X']  # Trajectories
batch['attr'] or batch['attributes']  # Attributes
```

### 4. Diffusion Schedule

The diffusion noise schedule is built using `register_buffer()` for proper device handling:
- `betas`, `alphas`, `alphas_cumprod`
- `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`
- `posterior_variance`, `posterior_mean_coef1`, `posterior_mean_coef2`

### 5. EMA Training

EMA helper is preserved but adapted to work within the LibCity training loop:
- EMA is updated in `calculate_loss()` during training
- EMA weights are applied in `predict()` for inference

## Model Components

### Included Components

1. **get_timestep_embedding**: Sinusoidal timestep embeddings
2. **Normalize**: Group normalization (32 groups)
3. **nonlinearity**: Swish activation
4. **Attention**: Simple attention for embeddings
5. **WideAndDeep**: Trajectory attribute embedding (combines wide linear + deep embedding)
6. **Upsample**: 1D upsampling with optional convolution
7. **Downsample**: 1D downsampling with optional convolution
8. **ResnetBlock**: Residual block with time embedding injection
9. **AttnBlock**: Self-attention block for 1D sequences
10. **UNet**: Main UNet architecture for trajectory denoising
11. **GuideUNet**: Guided UNet with classifier-free guidance
12. **EMAHelper**: Exponential moving average for parameters
13. **DiffTraj**: Main LibCity-compatible model class

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| traj_length | 200 | Length of trajectory sequence |
| in_channels | 2 | Input channels (lat/lon) |
| out_ch | 2 | Output channels |
| attr_dim | 8 | Trajectory attribute dimension |
| ch | 128 | Base channel size |
| ch_mult | [1,2,2,2] | Channel multipliers |
| num_res_blocks | 2 | Residual blocks per level |
| attn_resolutions | [16] | Resolutions for attention |
| dropout | 0.1 | Dropout rate |
| resamp_with_conv | true | Use convolution in resampling |
| guidance_scale | 3.0 | Classifier-free guidance scale |
| num_departure_slots | 288 | Departure time embedding size |
| num_locations | 257 | Location embedding size |
| num_diffusion_timesteps | 500 | Diffusion steps |
| beta_start | 0.0001 | Starting beta |
| beta_end | 0.05 | Ending beta |
| beta_schedule | "linear" | Beta schedule type |
| ema | true | Enable EMA |
| ema_rate | 0.9999 | EMA decay rate |

## Input/Output Format

### Training Input (batch dictionary)
- `traj` or `X`: Trajectories `[batch, 2, length]` or `[batch, length, 2]`
- `attr` or `attributes`: Trajectory attributes `[batch, 8]`
  - Index 0: departure_time (categorical, 0-287)
  - Index 1-5: continuous (trip_distance, trip_time, trip_length, avg_dis, avg_speed)
  - Index 6: start_id (categorical, 0-256)
  - Index 7: end_id (categorical, 0-256)

### Prediction Input
- `attr` or `attributes`: Trajectory attributes `[batch, 8]`

### Output
- Generated trajectories: `[batch, 2, length]`

## Assumptions and Limitations

1. **Fixed trajectory length**: Model expects trajectories of fixed length (default 200 points)
2. **Attribute format**: Expects exactly 8 attributes in specific order
3. **Normalization**: Trajectories should be normalized to approximately [-1, 1] range
4. **Categorical limits**: Departure slots (288), locations (257) - configurable
5. **Generation time**: Reverse diffusion requires 500 sequential steps (slow)

## Usage Example

```python
from libcity.model.trajectory_loc_prediction import DiffTraj

# Initialize model
config = {
    'device': torch.device('cuda'),
    'traj_length': 200,
    'num_diffusion_timesteps': 500,
    # ... other config
}
data_feature = {}

model = DiffTraj(config, data_feature)

# Training
batch = {
    'traj': trajectories,  # [batch, 2, 200]
    'attr': attributes     # [batch, 8]
}
loss = model.calculate_loss(batch)

# Generation
gen_batch = {'attr': attributes}
generated = model.predict(gen_batch)
```

## Notes

- This is a generative model, not a prediction model in the traditional sense
- The model generates complete trajectories from scratch given attributes
- Training uses MSE loss between predicted and actual noise
- Classifier-free guidance improves generation quality at the cost of 2x forward passes
