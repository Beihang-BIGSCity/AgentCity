# DiffMM Migration to Map Matching Task

## Migration Summary

**Date:** 2026-02-06
**Model:** DiffMM (Diffusion-based Map Matching)
**Action:** Moved from `trajectory_loc_prediction` to `map_matching` task category

## Reason for Migration

Testing revealed that DiffMM is a **map matching model**, NOT a trajectory location prediction model. The model expects:
- GPS coordinates (lat, lng, time)
- Candidate road segments with features
- Road network data (id_size parameter)
- Map matching specific outputs (matching GPS points to road segments)

This is fundamentally different from trajectory location prediction which predicts next POI/location visits.

## Changes Made

### 1. Model File Migration

**From:**
```
/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/DiffMM.py
```

**To:**
```
/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/DiffMM.py
```

**Key Updates in Model File:**
- Updated docstring to reflect Map Matching task
- Confirmed `AbstractModel` base class (correct for neural map matching models like RLOMM)
- No import changes needed - already using correct base class

### 2. Model Registration Updates

#### Map Matching __init__.py
**File:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/__init__.py`

**Added:**
```python
from libcity.model.map_matching.DiffMM import DiffMM
```

**Updated __all__ list:**
```python
__all__ = [
    "STMatching",
    "IVMM",
    "HMMM",
    "FMM",
    "RLOMM",
    "DiffMM"  # Added
]
```

#### Trajectory Location Prediction __init__.py
**File:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py`

**Removed:**
- Import statement: `from libcity.model.trajectory_loc_prediction.DiffMM import DiffMM`
- From __all__ list: `"DiffMM"`

### 3. Configuration Updates

#### Task Config File
**File:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`

**Removed from trajectory_loc_prediction section:**
```json
{
    "allowed_model": [
        // ... other models
        "DiffMM"  // REMOVED
    ],
    // ... removed DiffMM configuration block
}
```

**Already exists in map_matching section:**
```json
{
    "map_matching": {
        "allowed_model": [
            "STMatching",
            "IVMM",
            "HMMM",
            "FMM",
            "STMatch",
            "DeepMM",
            "DiffMM",  // Already present
            "TRMMA",
            "GraphMM",
            "RLOMM"
        ],
        "DiffMM": {
            "dataset_class": "DiffMMDataset",
            "executor": "DeepMapMatchingExecutor",
            "evaluator": "MapMatchingEvaluator"
        }
    }
}
```

#### Model Config File
**File:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/map_matching/DiffMM.json`

**Already exists with proper configuration:**
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
    "beta_schedule": "cosine",
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

## Model Architecture

DiffMM consists of:

1. **TrajEncoder**: Encodes GPS trajectories with candidate road segments
   - PointEncoder: Transformer-based GPS point encoding
   - Road embedding with candidate attention mechanism

2. **DiT (Diffusion Transformer)**: Flow matching model
   - Adaptive Layer Normalization (AdaLN)
   - Multi-head attention blocks
   - Sinusoidal time embeddings

3. **ShortCut**: Fast inference with flow matching
   - Bootstrap training for 1-2 step inference
   - Replaces traditional multi-step diffusion

## Base Class

**AbstractModel** - Used for neural network-based models including:
- Deep learning map matching models (DiffMM, RLOMM, GraphMM, DeepMM)
- Trajectory location prediction models
- Trajectory embedding models

Traditional rule-based map matching models (STMatching, IVMM, HMMM, FMM) use **AbstractTraditionModel**.

## Required Data Features

```python
data_feature = {
    'id_size': int,  # Number of road segments in the network
    # Other features provided by DiffMMDataset
}
```

## Batch Format

Expected batch dictionary keys:
- `current_loc`: [batch, seq_len, 3] GPS coordinates (lat, lng, time)
- `target`: [batch, seq_len] ground truth segment IDs
- `candidate_segs`: [batch, seq_len, max_candidates] candidate segment IDs
- `candidate_feats`: [batch, seq_len, max_candidates, 9] candidate features
- `candidate_mask`: [batch, seq_len, max_candidates] validity mask
- `current_loc_len`: [batch] or list of actual trajectory lengths

## Executor and Evaluator

- **Dataset Class:** DiffMMDataset
- **Executor:** DeepMapMatchingExecutor
- **Evaluator:** MapMatchingEvaluator

## Usage

```python
from libcity.model.map_matching import DiffMM

# Initialize model
config = {...}  # Load from DiffMM.json
data_feature = {'id_size': num_road_segments}
model = DiffMM(config, data_feature)

# Training
loss = model.calculate_loss(batch)

# Inference
predictions = model.predict(batch)
```

## Files Modified

1. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/DiffMM.py` (created)
2. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/__init__.py` (updated)
3. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py` (updated)
4. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json` (updated)

## Files to Delete

The old file is marked for deletion in git:
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/DiffMM.py`

This will be removed when changes are committed.

## Testing Recommendations

1. Test imports:
   ```python
   from libcity.model.map_matching import DiffMM
   ```

2. Test model initialization with DiffMMDataset

3. Verify training with DeepMapMatchingExecutor

4. Validate predictions with MapMatchingEvaluator

## Related Models

Other map matching models in LibCity:
- **Traditional:** STMatching, IVMM, HMMM, FMM
- **Neural:** DeepMM, GraphMM, RLOMM, TRMMA

## Notes

- DiffMM uses flow matching for fast 1-2 step inference instead of traditional multi-step diffusion
- The model already had proper configuration in map_matching section
- Base class `AbstractModel` is correct for neural map matching models
- No changes needed to model logic, only file location and registration

## Completion Status

- [x] Move model file to map_matching directory
- [x] Update map_matching __init__.py
- [x] Remove from trajectory_loc_prediction __init__.py
- [x] Update task_config.json (remove from traj_loc_pred)
- [x] Verify map_matching config exists
- [x] Create migration documentation
- [ ] Commit changes to delete old file
- [ ] Test imports and model loading

## Next Steps

1. Commit the changes to finalize file deletion
2. Test the model with DiffMMDataset
3. Run end-to-end map matching pipeline
4. Update any external documentation or README files
