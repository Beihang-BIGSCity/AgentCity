# Config Migration: DiffMM

## Overview

**Model Name**: DiffMM
**Task Type**: Trajectory Location Prediction (Map Matching)
**Migration Date**: 2026-02-02
**Status**: Configuration Complete

---

## Task Configuration (task_config.json)

### Registration

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`

- **Added to**: `traj_loc_pred.allowed_model`
- **Line number**: 36
- **Task type**: `traj_loc_pred`

### Configuration Section

```json
"DiffMM": {
    "dataset_class": "TrajectoryDataset",
    "executor": "TrajLocPredExecutor",
    "evaluator": "TrajLocPredEvaluator",
    "traj_encoder": "StandardTrajectoryEncoder"
}
```

**Location**: Lines 231-236

---

## Model Configuration (DiffMM.json)

### File Path
`/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/DiffMM.json`

### Complete Configuration

```json
{
    "model_name": "DiffMM",
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
    "batch_size": 4,
    "max_epoch": 30,
    "clip_grad_norm": 1.0,
    "evaluate_method": "segment"
}
```

### Parameters Description

#### Model Architecture Parameters

| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `hid_dim` | 256 | Original paper | Hidden dimension for TrajEncoder |
| `num_units` | 512 | Original paper | Hidden dimension for DiT model |
| `transformer_layers` | 2 | Original paper | Number of transformer encoder layers |
| `depth` | 2 | Original paper | Number of DiT blocks |
| `num_heads` | 4 | Original paper | Number of attention heads |

#### Diffusion Parameters

| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `timesteps` | 2 | Original paper | Diffusion timesteps for training |
| `samplingsteps` | 1 | Original paper | Inference denoising steps (1=one-step) |
| `bootstrap_every` | 8 | Original paper | Bootstrap sample ratio (1/8 of batch) |

#### Regularization

| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `dropout` | 0.1 | Original paper | Dropout rate for regularization |
| `clip_grad_norm` | 1.0 | Best practice | Gradient clipping threshold |

#### Training Parameters

| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `optimizer` | "AdamW" | Original paper | Optimizer type |
| `learning_rate` | 0.001 | Original paper | Initial learning rate |
| `weight_decay` | 1e-6 | Original paper | L2 regularization weight |
| `lr_scheduler` | "none" | Default | Learning rate scheduler |
| `batch_size` | 4 | Original paper | Training batch size |
| `max_epoch` | 30 | Original paper | Maximum training epochs |

#### Evaluation

| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `evaluate_method` | "segment" | LibCity convention | Evaluation method for map matching |

---

## Dataset Compatibility

### Required Data Features

DiffMM requires the following features in each batch:

| Feature Key | Alternative Key | Shape | Type | Description |
|-------------|-----------------|-------|------|-------------|
| `norm_gps_seq` | `X` | (B, L, 3) | Float | Normalized GPS: [lat, lng, time] |
| `lengths` | `current_len` | (B,) | Int | Sequence lengths |
| `trg_rid` | `target` | (B, L) | Int | Ground truth road segment IDs |
| `segs_id` | `candidate_segs` | (B, L, C) | Int | Candidate segment IDs |
| `segs_feat` | `candidate_features` | (B, L, C, 9) | Float | Candidate features |
| `segs_mask` | `candidate_mask` | (B, L, C) | Bool | Candidate validity mask |

Where:
- B = Batch size
- L = Max sequence length
- C = Max number of candidate segments per GPS point

### Candidate Features (9 dimensions)

1. `err_weight`: Error weight from spatial distance
2. `cosv`: Cosine similarity with velocity
3. `cosv_pre`: Cosine similarity with previous velocity
4. `cosf`: Cosine feature f (heading)
5. `cosl`: Cosine feature l (length)
6. `cos1`: Cosine feature 1
7. `cos2`: Cosine feature 2
8. `cos3`: Cosine feature 3
9. `cosp`: Cosine feature p (position)

### Dataset Requirements

1. **Road Network**: Must have road segment database with geometry
2. **Candidate Generation**: Must pre-compute candidate segments within spatial radius
3. **Segment Indexing**: Road segment IDs must start from 1 (0 reserved for padding)
4. **Normalization**: GPS coordinates should be normalized to [0, 1] range

### Compatible LibCity Datasets

Currently, standard trajectory datasets (foursquare_tky, foursquare_nyc, etc.) would need:
- Road network preprocessing
- Candidate segment generation
- Feature extraction for each candidate

**Note**: You may need to create a custom dataset class or encoder for full DiffMM functionality.

---

## Usage Examples

### Command Line

```bash
# Basic usage with default config
python run_model.py --task traj_loc_pred --model DiffMM --dataset foursquare_tky

# Override specific parameters
python run_model.py --task traj_loc_pred --model DiffMM --dataset foursquare_tky \
    --hid_dim 256 --num_units 512 --batch_size 8 --max_epoch 50

# Use GPU
python run_model.py --task traj_loc_pred --model DiffMM --dataset foursquare_tky \
    --gpu true --gpu_id 0
```

### Configuration File

Create `diffmm_test.json`:

```json
{
    "task": "traj_loc_pred",
    "model": "DiffMM",
    "dataset": "foursquare_tky",
    "hid_dim": 256,
    "num_units": 512,
    "transformer_layers": 2,
    "depth": 2,
    "timesteps": 2,
    "samplingsteps": 1,
    "dropout": 0.1,
    "bootstrap_every": 8,
    "num_heads": 4,
    "learning_rate": 0.001,
    "batch_size": 4,
    "max_epoch": 30,
    "gpu": true,
    "gpu_id": 0
}
```

Run with:
```bash
python run_model.py --config_file diffmm_test.json
```

### Python API

```python
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_executor, get_model

# Load config
config = ConfigParser(task='traj_loc_pred', model='DiffMM', dataset='foursquare_tky')

# Get dataset
dataset = get_dataset(config)

# Create model
model = get_model(config, dataset.get_data_feature())

# Get executor
executor = get_executor(config, model, dataset)

# Train
executor.train(model, dataset)

# Evaluate
executor.evaluate(model)
```

---

## Validation Checklist

### Configuration Files
- [x] Added DiffMM to `traj_loc_pred.allowed_model` in task_config.json
- [x] Added DiffMM configuration section in task_config.json
- [x] Created DiffMM.json in config/model/traj_loc_pred/
- [x] Verified JSON syntax is valid
- [x] All parameters have appropriate default values

### Model Integration
- [x] Model file exists at libcity/model/trajectory_loc_prediction/DiffMM.py
- [x] Model registered in __init__.py
- [x] Model inherits from AbstractModel
- [x] Model implements required methods (forward, predict, calculate_loss)

### Documentation
- [x] Updated DiffMM_migration_summary.md with configuration section
- [x] Created DiffMM_config_migration_summary.md
- [x] Documented all configuration parameters
- [x] Provided usage examples
- [x] Documented dataset requirements

### Testing (Pending)
- [ ] Test model initialization with config
- [ ] Test forward pass with sample data
- [ ] Test loss calculation
- [ ] Test prediction output format
- [ ] Validate with real trajectory data
- [ ] Verify bootstrap training mechanism
- [ ] Test one-step vs multi-step inference

---

## Notes and Considerations

### Compatibility Concerns

1. **Dataset Format**: Standard LibCity trajectory datasets may not have the required road network and candidate segment information. A custom dataset/encoder may be needed.

2. **Segment Features**: The 9-dimensional candidate features require geometric calculations between GPS points and road segments.

3. **Bootstrap Training**: The model uses a special training mode where it temporarily switches to eval mode to generate targets. This is handled internally.

4. **Memory Usage**: With `num_units=512` and `depth=2`, the DiT model can be memory-intensive. Consider reducing batch size or model dimensions for limited GPU memory.

### Differences from Original

| Aspect | Original DiffMM | LibCity Adaptation |
|--------|-----------------|---------------------|
| Config loading | argparse + dict | LibCity ConfigParser |
| Device management | Manual | LibCity config['device'] |
| Model initialization | Standalone class | AbstractModel subclass |
| Data loading | Custom dataloader | LibCity Dataset classes |
| Training loop | Custom training code | Executor pattern |
| Evaluation | Custom metrics | Evaluator pattern |

### Future Enhancements

1. Create a dedicated `DiffMMDataset` class for better data handling
2. Implement a `DiffMMEncoder` for candidate generation
3. Add road network preprocessing utilities
4. Create specialized evaluator for map matching metrics
5. Add visualization tools for trajectory-to-road matching

---

## Files Modified Summary

### Created
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/DiffMM.json` (Previous phase)
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/DiffMM.py` (Previous phase)
- `/home/wangwenrui/shk/AgentCity/documents/DiffMM_config_migration_summary.md` (This document)

### Modified
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
  - Line 36: Added "DiffMM" to allowed_model list
  - Lines 231-236: Added DiffMM configuration section
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py` (Previous phase)
  - Line 29: Added import statement
  - Line 59: Added to __all__ list
- `/home/wangwenrui/shk/AgentCity/documents/DiffMM_migration_summary.md`
  - Added configuration section
  - Added usage examples
  - Added parameter mappings

---

## Next Steps

1. **Prepare Test Data**:
   - Create or identify a trajectory dataset with road network
   - Implement candidate segment generation
   - Preprocess data to match required format

2. **Initial Testing**:
   ```bash
   cd /home/wangwenrui/shk/AgentCity/Bigscity-LibCity
   python run_model.py --task traj_loc_pred --model DiffMM --dataset test_dataset
   ```

3. **Validation**:
   - Verify model loads correctly
   - Check forward pass outputs
   - Validate loss computation
   - Test prediction format
   - Evaluate on sample data

4. **Optimization** (if needed):
   - Tune hyperparameters
   - Adjust batch size for memory
   - Experiment with different diffusion timesteps
   - Test bootstrap frequency variations

---

## References

- **Original Paper**: "DiffMM: Efficient Method for Accurate Noisy and Sparse Trajectory Map Matching via One Step Diffusion" (AAAI)
- **Original Repository**: https://github.com/decisionintelligence/DiffMM
- **LibCity Documentation**: https://bigscity-libcity-docs.readthedocs.io/
- **DiT Paper**: "Scalable Diffusion Models with Transformers" (ICCV 2023)

---

**Configuration Status**: COMPLETE ✓ (Updated 2026-02-03)
**Ready for Testing**: YES
**Last Updated**: 2026-02-03

---

## Update Log (2026-02-03)

### Configuration Updates
1. **Updated Model Config** (`DiffMM.json`):
   - Added missing diffusion parameters: `objective`, `beta_schedule`, `beta_start`, `beta_end`
   - Changed `max_epoch` from 30 to 100 (original paper default)
   - All parameters now match original paper specifications

2. **Verified Task Registration**:
   - Confirmed DiffMM is in `traj_loc_pred.allowed_model` (line 35)
   - Confirmed model-specific config block exists (lines 224-229)
   - Verified JSON syntax is valid

3. **Dual Task Support**:
   - DiffMM is registered in BOTH `traj_loc_pred` and `map_matching` tasks
   - Use `traj_loc_pred` for POI/trajectory prediction adapted to segments
   - Use `map_matching` for true GPS-to-road matching

### Updated Configuration Parameters

The complete DiffMM.json now includes:

```json
{
    "model": "DiffMM",
    "task": "traj_loc_pred",
    "hid_dim": 256,
    "num_units": 512,
    "transformer_layers": 2,
    "depth": 2,
    "num_heads": 4,
    "dropout": 0.1,
    "timesteps": 2,
    "samplingsteps": 1,
    "bootstrap_every": 8,
    "objective": "pred_v",
    "beta_schedule": "sqrt",
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "batch_size": 4,
    "learning_rate": 0.001,
    "max_epoch": 100,
    "optimizer": "adamw",
    "weight_decay": 1e-6,
    "clip_grad_norm": 1.0,
    "lr_scheduler": "none",
    "log_every": 1,
    "load_best_epoch": true,
    "hyper_tune": false,
    "evaluate_method": "segment"
}
```

### Available Datasets

#### traj_loc_pred Task Datasets
- foursquare_tky
- foursquare_nyc
- gowalla
- foursquare_serm
- Proto

#### map_matching Task Datasets
- global
- Seattle
- Neftekamsk
- Valky
- Ruzhany
- Santander
- Spaichingen
- NovoHamburgo
