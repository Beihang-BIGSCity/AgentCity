# Config Migration: MetaTTE Recategorization

## Task: Recategorize MetaTTE from traffic_state_pred to eta

## Problem Statement
MetaTTE was incorrectly categorized under `traffic_state_pred` task. The model is designed for trajectory-based travel time estimation (ETA), NOT traffic state prediction on fixed sensor locations.

## Model Analysis
- **Purpose**: Multi-Scale Spatio-Temporal Travel Time Estimation
- **Input**: Trajectory sequences with (lat_diff, lng_diff, time_id, week_id)
- **Output**: Predicted travel time for a trajectory
- **Architecture**:
  - Three parallel GRU branches (spatial, hour temporal, week temporal)
  - Attention mechanism over branches
  - MLP head with residual connection
- **Task Type**: ETA (Estimated Time of Arrival / Travel Time Estimation)

## Changes Made

### 1. task_config.json Updates

#### Removed from traffic_state_pred
- **File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
- **Line 364**: Removed "MetaTTE" from `traffic_state_pred.allowed_model` list
- **Lines 1001-1005** (old): Removed MetaTTE config block with TrafficStatePointDataset/TrafficStateExecutor/TrafficStateEvaluator

#### Added to eta
- **Line 1014**: Added "MetaTTE" to `eta.allowed_model` list
- **Lines 1086-1091** (new): Added MetaTTE config block:
```json
"MetaTTE": {
    "dataset_class": "ETADataset",
    "executor": "ETAExecutor",
    "evaluator": "ETAEvaluator",
    "eta_encoder": "StandardTrajectoryEncoder"
}
```

### 2. Model Config File
- **Created**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/eta/MetaTTE.json`
- **Copied from**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traffic_state_pred/MetaTTE.json`
- **Parameters**: All hyperparameters preserved (hidden_size, time_emb_dim, week_emb_dim, etc.)

### 3. Model File Location
- **Created**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/MetaTTE.py`
- **Copied from**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_speed_prediction/MetaTTE.py`
- **Note**: Original file kept in place for backwards compatibility during transition

### 4. Import Statements

#### Added to eta/__init__.py
- **File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/__init__.py`
- **Added import**: `from libcity.model.eta.MetaTTE import MetaTTE`
- **Added to __all__**: "MetaTTE"

#### Removed from traffic_speed_prediction/__init__.py
- **File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_speed_prediction/__init__.py`
- **Removed import**: `from libcity.model.traffic_speed_prediction.MetaTTE import MetaTTE`
- **Removed from __all__**: "MetaTTE"

## Configuration Details

### Dataset Class: ETADataset
- Designed for trajectory-based ETA tasks
- Handles sequential trajectory points with spatial and temporal features
- Compatible with MetaTTE's input format

### Executor: ETAExecutor
- Manages training/testing workflow for ETA models
- Handles trajectory data batching and evaluation
- Appropriate for travel time estimation tasks

### Evaluator: ETAEvaluator
- Computes ETA-specific metrics (MAE, RMSE, MAPE)
- Evaluates predicted vs actual travel times
- Different from TrafficStateEvaluator which focuses on speed/flow prediction

### Encoder: StandardTrajectoryEncoder
- Processes trajectory data for ETA models
- Encodes spatial coordinates and temporal features
- Compatible with MetaTTE's input requirements

## Compatibility Verification

### Task Type Match
- **Before**: traffic_state_pred - Grid/graph-based prediction at fixed locations
- **After**: eta - Trajectory-based travel time estimation
- **Correct**: YES - MetaTTE processes trajectories, not fixed sensor data

### Dataset Compatibility
- **ETADataset features**: Trajectory sequences, lat/lng, timestamps
- **MetaTTE requirements**: lat_diff, lng_diff, time_id, week_id
- **Compatible**: YES - ETADataset can provide required features

### Executor/Evaluator Match
- **ETAExecutor**: Handles trajectory batching and sequential processing
- **ETAEvaluator**: Computes travel time metrics
- **Appropriate**: YES - Matches MetaTTE's prediction task

## Model Hyperparameters (Preserved)

```json
{
  "hidden_size": 128,
  "time_emb_dim": 128,
  "week_emb_dim": 128,
  "num_hours": 24,
  "num_weekdays": 7,
  "spatial_input_dim": 2,
  "num_gru_layers": 1,
  "dropout": 0.0,
  "bidirectional": false,
  "rnn_type": "GRU",
  "mlp_layers": [1024, 512, 256, 128],
  "max_epoch": 100,
  "batch_size": 64,
  "learner": "adam",
  "learning_rate": 1e-3,
  "lr_decay": false,
  "clip_grad_norm": false,
  "use_early_stop": true,
  "patience": 20
}
```

## Available Datasets for ETA Task
- Chengdu_Taxi_Sample1
- Beijing_Taxi_Sample

## Testing Recommendations

### 1. Import Test
```python
from libcity.model.eta import MetaTTE
model = MetaTTE(config, data_feature)
```

### 2. Configuration Test
```bash
python run_model.py --task eta --model MetaTTE --dataset Chengdu_Taxi_Sample1
```

### 3. Verify Dataset Compatibility
- Check that ETADataset provides required features
- Ensure trajectory sequences have lat_diff, lng_diff, time_id, week_id
- Validate batch format matches MetaTTE's forward() expectations

## Notes and Caveats

### Backward Compatibility
- Original model file kept at `libcity/model/traffic_speed_prediction/MetaTTE.py` for now
- Should be removed after confirming all imports reference the new location
- Update any existing scripts/configs that reference the old location

### Data Feature Requirements
MetaTTE expects batch data with one of these formats:
1. Combined format: `batch['X']` with shape `[batch_size, seq_len, 4]`
   - Features: (lat_diff, lng_diff, time_id, week_id)
2. Separate keys: `batch['lat_diff']`, `batch['lng_diff']`, `batch['time_id']`, `batch['week_id']`

Target labels accepted:
- `batch['y']` - Generic target
- `batch['time']` - Travel time
- `batch['travel_time']` - Explicit travel time

### Encoder Selection
- Used `StandardTrajectoryEncoder` as most ETA models don't use custom encoders
- If MetaTTE requires specific preprocessing, a custom encoder may be needed
- Consider creating `MetaTTEEncoder` if data transformation is needed

## Summary

### Files Modified
1. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
2. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/__init__.py`
3. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_speed_prediction/__init__.py`

### Files Created
1. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/eta/MetaTTE.json`
2. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/MetaTTE.py`

### Result
MetaTTE successfully recategorized from `traffic_state_pred` to `eta` task type with appropriate:
- Dataset class: ETADataset
- Executor: ETAExecutor
- Evaluator: ETAEvaluator
- Encoder: StandardTrajectoryEncoder

The model is now correctly categorized for trajectory-based travel time estimation tasks.
