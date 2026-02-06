# DiffMM Migration Summary - LibCity Integration

## Overview

**Model Name**: DiffMM
**Original Repository**: https://github.com/decisionintelligence/DiffMM
**Paper**: "DiffMM: Efficient Method for Accurate Noisy and Sparse Trajectory Map Matching via One Step Diffusion" (AAAI)
**Task Type**: Map Matching (GPS-to-road-segment matching)
**Migration Date**: 2026-02-02
**Latest Update**: 2026-02-04 (final version in map_matching directory)

---

## Task Category History

### Update 2026-02-04 (Final)
DiffMM has been properly implemented in `map_matching` directory with a complete rewrite following the original repository architecture.

**Current Location**: `libcity/model/map_matching/DiffMM.py`

The implementation includes:
- Complete TrajEncoder with PointEncoder and road segment attention
- Full DiT (Diffusion Transformer) with AdaLN modulation
- ShortCut one-step diffusion for efficient inference
- Bootstrap target generation for training
- Proper LibCity AbstractModel integration
- Full batch format adaptation for LibCity compatibility

### Previous Updates
- 2026-02-03: Temporarily in trajectory_loc_prediction
- 2026-02-02: Initial migration

---

## Files Created/Updated

### Model File
**Path**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/DiffMM.py`

The model file contains all necessary components:
- `Norm`: Layer normalization
- `PositionalEncoder`: Sinusoidal positional encoding
- `MultiHeadAttention`: Multi-head self-attention
- `FeedForward`: Feed-forward network with residual
- `EncoderLayer`: Transformer encoder layer
- `TransformerEncoder`: Full transformer encoder
- `PointEncoder`: GPS point sequence encoder
- `Attention`: Cross-attention for trajectory-road matching
- `TrajEncoder`: Main trajectory encoder
- `SinusoidalPosEmb`: Timestep embedding
- `DiTBlock`: Diffusion Transformer block with adaptive LayerNorm
- `OutputLayer`: Final output with modulation
- `DiT`: Complete Diffusion Transformer
- `get_targets`: Bootstrap target generation function
- `ShortCut`: One-step diffusion wrapper
- `DiffMM`: Main LibCity model class

### Configuration File
**Path**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/map_matching/DiffMM.json`

```json
{
    "model_name": "DiffMM",
    "dataset_class": "DiffMMDataset",
    "hid_dim": 256,
    "num_units": 512,
    "transformer_layers": 2,
    "depth": 2,
    "timesteps": 2,
    "samplingsteps": 1,
    "dropout": 0.1,
    "bootstrap_every": 8,
    "num_heads": 4,
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 1e-6,
    "lr_scheduler": "none",
    "batch_size": 16,
    "max_epoch": 30,
    "clip_grad_norm": 1.0,
    "evaluate_method": "segment",
    "num_cands": 10,
    "cand_search_radius": 100,
    "max_seq_len": 100,
    "min_seq_len": 5,
    "train_rate": 0.7,
    "eval_rate": 0.15
}
```

### Registration
**File Updated**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/__init__.py`

Added:
```python
from libcity.model.map_matching.DiffMM import DiffMM
# And added "DiffMM" to __all__ list
```

---

## Architecture Overview

### Model Pipeline

```
GPS Trajectory + Candidate Segments
        |
        v
+-------------------+
|   TrajEncoder     |
|   - PointEncoder  | -> Transformer encoding of GPS points
|   - RoadEmbed     | -> Embedding of candidate road segments
|   - CrossAttn     | -> Attention between points and segments
+-------------------+
        |
        v (2 * hid_dim condition)
+-------------------+
|   DiT Model       |
|   - NoiseLinear   | -> Project noisy input
|   - TimeEmbed     | -> Sinusoidal time embedding
|   - DiTBlocks     | -> Adaptive LayerNorm Transformer
|   - OutputLayer   | -> Final velocity prediction
+-------------------+
        |
        v
+-------------------+
|   ShortCut        |
|   - Training      | -> Flow matching + Bootstrap targets
|   - Inference     | -> One-step or multi-step denoising
+-------------------+
        |
        v
Road Segment Predictions
```

### Key Features

1. **One-Step Diffusion**: Unlike traditional diffusion models that require many denoising steps, DiffMM learns to denoise in a single step for efficient inference.

2. **Bootstrap Training**: Uses self-consistency training where the model generates its own training targets at coarser time scales.

3. **Flow Matching**: Combines rectified flow objectives with bootstrap targets for stable training.

4. **Candidate Masking**: Spatially filters candidate road segments using masks to improve precision.

5. **Adaptive Layer Normalization**: DiT blocks use modulation for conditioning on trajectory and timestep.

---

## Data Format Requirements

### Input Batch Keys

| Key | Shape | Description |
|-----|-------|-------------|
| `norm_gps_seq` or `X` | (B, L, 3) | Normalized GPS: [lat, lng, time] |
| `lengths` or `current_len` | (B,) | Sequence lengths |
| `trg_rid` or `target` | (B, L) | Ground truth road segment IDs |
| `segs_id` or `candidate_segs` | (B, L, C) | Candidate segment IDs |
| `segs_feat` or `candidate_features` | (B, L, C, 9) | Candidate features |
| `segs_mask` or `candidate_mask` | (B, L, C) | Candidate validity mask |

### Candidate Features (9 dimensions)
1. `err_weight`: Error weight
2. `cosv`: Cosine velocity
3. `cosv_pre`: Previous cosine velocity
4. `cosf`: Cosine feature f
5. `cosl`: Cosine feature l
6. `cos1`: Cosine feature 1
7. `cos2`: Cosine feature 2
8. `cos3`: Cosine feature 3
9. `cosp`: Cosine feature p

---

## Configuration Parameters

### Model Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hid_dim` | 256 | Hidden dimension for encoder |
| `num_units` | 512 | Hidden dimension for DiT |
| `transformer_layers` | 2 | Number of Transformer layers in encoder |
| `depth` | 2 | Number of DiT blocks |
| `timesteps` | 2 | Training timesteps |
| `samplingsteps` | 1 | Inference steps (1 for one-step) |
| `dropout` | 0.1 | Dropout rate |
| `bootstrap_every` | 8 | Bootstrap sample ratio (1/8 of batch) |
| `num_heads` | 4 | Attention heads |

---

## API Methods

### `__init__(config, data_feature)`
Initialize the model with configuration and data features.

### `forward(batch)`
Forward pass returning encoded trajectory conditions.

### `predict(batch)`
Returns list of predicted road segment indices per trajectory.

### `calculate_loss(batch)`
Returns combined MSE + BCE loss for training.

### `get_predictions_with_probs(batch)`
Returns predictions with probability scores and full distributions.

---

## Key Differences from Original

1. **Standalone to AbstractModel**: Wrapped all components into LibCity's AbstractModel interface.

2. **Config-based Parameters**: All hyperparameters extracted to config file.

3. **Batch Transformation**: Added `_batch2model` method to convert LibCity batch format.

4. **Device Management**: Uses LibCity's device configuration.

5. **Self-contained Layers**: All layer definitions are included in the single file (no external dependencies on repos/DiffMM).

6. **Batch Access Pattern**: Uses try/except for BatchPAD compatibility in map_matching task.

---

## Dependencies

### Required
- PyTorch >= 1.9.0
- NumPy

### Optional
- einops (not required, removed dependency)

---

## Usage Example

### Running DiffMM in LibCity

```bash
# Basic usage for map matching task
python run_model.py --task map_matching --model DiffMM --dataset your_dataset

# With custom config
python run_model.py --task map_matching --model DiffMM --dataset your_dataset \
    --hid_dim 256 --num_units 512 --batch_size 16 --max_epoch 30
```

### Python API Usage

```python
from libcity.model.map_matching import DiffMM

# Configuration
config = {
    'device': 'cuda:0',
    'hid_dim': 256,
    'num_units': 512,
    'transformer_layers': 2,
    'depth': 2,
    'timesteps': 2,
    'samplingsteps': 1,
    'dropout': 0.1,
    'bootstrap_every': 8
}

# Data features
data_feature = {
    'id_size': 5000,  # Number of road segments + 1
    'num_cands': 10,
    'feat_dim': 9
}

# Create model
model = DiffMM(config, data_feature)
model = model.to(config['device'])

# Training
loss = model.calculate_loss(batch)
loss.backward()

# Inference
predictions = model.predict(batch)
```

---

## Executor

Uses `DeepMapMatchingExecutor` for training and evaluation.

---

## Evaluation Metrics

The original DiffMM uses the following metrics:
- **Accuracy**: Exact match rate
- **Recall**: Percentage of ground truth segments recovered
- **Precision**: Percentage of predictions that are correct
- **F1 Score**: Harmonic mean of precision and recall

These can be computed using LibCity's evaluation framework or custom evaluators.

---

## Limitations and Notes

1. **Road Network Preprocessing**: The original model requires candidate segment generation using spatial search. This preprocessing should be done in the dataset/encoder stage.

2. **Segment Indexing**: Road segment IDs are expected to start from 1 (0 reserved for padding). The model uses `id_size - 1` for the output dimension.

3. **Bootstrap Training**: During training, the model switches to eval mode temporarily to generate bootstrap targets. This is handled in `get_targets`.

4. **Memory Usage**: The DiT model with high `num_units` can be memory-intensive. Reduce batch size if needed.

---

## References

- Original Paper: "DiffMM: Efficient Method for Accurate Noisy and Sparse Trajectory Map Matching via One Step Diffusion"
- Repository: https://github.com/decisionintelligence/DiffMM
- DiT Paper: "Scalable Diffusion Models with Transformers" (ICCV 2023)

---

**Initial Migration**: 2026-02-02
**Final Migration**: 2026-02-04 (moved to map_matching directory)
**LibCity Version**: Compatible with current version
**Status**: Migration complete, located in map_matching directory
**Executor**: DeepMapMatchingExecutor
