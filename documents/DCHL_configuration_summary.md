# Config Migration: DCHL

## Overview
DCHL (Disentangled Contrastive Hypergraph Learning) model has been successfully configured in LibCity's configuration system for trajectory location prediction (next POI recommendation).

**Paper**: SIGIR 2024 - Disentangled Contrastive Hypergraph Learning for Next POI Recommendation
**Original Repository**: https://github.com/icmpnorequest/SIGIR2024_DCHL
**Task Type**: traj_loc_pred (trajectory location prediction / next POI recommendation)

---

## Configuration Status

### 1. task_config.json ✓
**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`

- **Status**: Already registered
- **Location**: Line 31 in `traj_loc_pred.allowed_model` list
- **Task Configuration**: Lines 203-208
  ```json
  "DCHL": {
      "dataset_class": "TrajectoryDataset",
      "executor": "TrajLocPredExecutor",
      "evaluator": "TrajLocPredEvaluator",
      "traj_encoder": "StandardTrajectoryEncoder"
  }
  ```

### 2. Model Configuration ✓
**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/DCHL.json`

- **Status**: Configured with all required hyperparameters
- **Parameters** (from SIGIR 2024 paper):

| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `emb_dim` | 128 | Paper default | Embedding dimension |
| `num_mv_layers` | 3 | Paper default | Multi-view hypergraph layers |
| `num_geo_layers` | 3 | Paper default | Geographic graph layers |
| `num_di_layers` | 3 | Paper default | Directed hypergraph layers |
| `dropout` | 0.3 | Paper default | Dropout rate |
| `temperature` | 0.1 | Paper default | InfoNCE temperature |
| `lambda_cl` | 0.1 | Paper default | Contrastive loss weight |
| `keep_rate` | 1.0 | Paper default | Edge dropout rate |
| `keep_rate_poi` | 1.0 | Paper default | POI edge dropout |
| `distance_threshold` | 2.5 | Paper default | Geographic distance (km) |
| `learning_rate` | 0.001 | Paper default | Learning rate |
| `lr_decay` | 0.1 | Paper default | LR decay factor |
| `weight_decay` | 0.0005 | Paper default | Weight decay |
| `batch_size` | 200 | Paper default | Batch size |
| `max_epoch` | 30 | Paper default | Maximum epochs |

### 3. Model Import ✓
**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py`

- **Status**: Imported and exported
- **Import Line**: Line 27
  ```python
  from libcity.model.trajectory_loc_prediction.DCHL import DCHL
  ```
- **Export**: Line 55 in `__all__` list

### 4. Model Implementation ✓
**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/DCHL.py`

- **Status**: Fully adapted for LibCity
- **Base Class**: `AbstractModel`
- **Key Features**:
  - Multi-view hypergraph convolutional network
  - Directed hypergraph for sequential patterns
  - Geographic graph for spatial patterns
  - InfoNCE contrastive learning across three views
  - Adaptive fusion with learnable gates

---

## Dataset Compatibility

### Compatible Datasets
DCHL uses `TrajectoryDataset` class and is compatible with the following LibCity datasets:

**Allowed Datasets** (from task_config.json):
- `foursquare_tky` - Foursquare Tokyo
- `foursquare_nyc` - Foursquare New York City
- `gowalla` - Gowalla check-in data
- `foursquare_serm` - Foursquare SERM
- `Proto` - Prototype dataset

### Required Data Features

DCHL requires the following data features for proper initialization:

1. **Basic Features**:
   - `num_users` (or `uid_size`): Number of unique users
   - `num_pois` (or `loc_size`): Number of unique POIs

2. **Graph Structures** (can be precomputed or built from raw data):
   - `sessions_dict`: User sessions dictionary (required for graph construction)
   - `pois_coos_dict`: POI coordinates (latitude, longitude) - optional but recommended
   - `HG_up`: User-to-POI hypergraph (precomputed) - optional
   - `HG_pu`: POI-to-User hypergraph (precomputed) - optional
   - `HG_poi_src`: Source POI hypergraph (precomputed) - optional
   - `HG_poi_tar`: Target POI hypergraph (precomputed) - optional
   - `poi_geo_graph`: Geographic adjacency graph (precomputed) - optional

### Dataset Configuration
**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/data/TrajectoryDataset.json`

Default TrajectoryDataset parameters:
```json
{
  "min_session_len": 5,
  "max_session_len": 50,
  "min_sessions": 2,
  "min_checkins": 3,
  "window_size": 12,
  "batch_size": 20,
  "num_workers": 0,
  "cache_dataset": true,
  "train_rate": 0.7,
  "eval_rate": 0.1,
  "history_type": "splice",
  "cut_method": "time_interval",
  "traj_encoder": "StandardTrajectoryEncoder"
}
```

---

## Model Architecture

### Three-View Graph Learning

1. **Multi-View Hypergraph** (Collaborative Filtering):
   - Captures user-POI interaction patterns
   - Uses hypergraph structure where each user is a hyperedge
   - `num_mv_layers` layers with residual connections

2. **Geographical Graph** (Spatial Patterns):
   - POI-POI adjacency based on geographic distance
   - Distance threshold: 2.5 km (configurable)
   - Uses Haversine distance for lat/lon calculation
   - `num_geo_layers` layers

3. **Directed Hypergraph** (Sequential Patterns):
   - Captures temporal transition patterns
   - Source-to-target POI transitions from trajectories
   - `num_di_layers` layers

### Disentangled Learning

- Self-gating mechanism to separate view-specific features
- Three separate gates for geo, seq, and collaborative views
- Applied before view-specific graph convolutions

### Contrastive Learning

- InfoNCE loss across three views
- Temperature parameter: 0.1
- Contrastive loss weight: 0.1
- Applied to both POI and user embeddings

### Adaptive Fusion

- Learnable gates for each view
- Separate gates for user and POI embeddings
- Final prediction from fused representations

---

## Usage Example

```python
# Run DCHL model on Foursquare NYC dataset
python run_model.py --task traj_loc_pred --model DCHL --dataset foursquare_nyc

# With custom hyperparameters
python run_model.py --task traj_loc_pred --model DCHL --dataset foursquare_nyc \
    --emb_dim 256 --num_mv_layers 4 --lambda_cl 0.2
```

---

## Implementation Notes

### LibCity Adaptations

1. **Constructor Signature**:
   - Original: `__init__(num_users, num_pois, args, device)`
   - LibCity: `__init__(config, data_feature)`

2. **Graph Initialization**:
   - Supports both precomputed graphs and on-the-fly construction
   - Graphs can be passed via `data_feature` or built from `sessions_dict`
   - Method `set_data_feature()` allows dynamic graph updates

3. **Forward Pass**:
   - Original: Takes user indices and sequences
   - LibCity: Takes batch dictionary with `user_idx` or `uid`
   - Returns predictions and contrastive losses

4. **Loss Calculation**:
   - Required `calculate_loss()` method for LibCity training
   - Combines reconstruction loss with contrastive losses
   - Formula: `loss = loss_rec + lambda_cl * (loss_cl_poi + loss_cl_user)`

5. **Prediction**:
   - Required `predict()` method for LibCity evaluation
   - Returns prediction scores for all POIs

### Graph Construction

If graphs are not precomputed, DCHL will build them from raw data:

1. **User-POI Hypergraph**: From `sessions_dict`
2. **Directed POI Hypergraph**: From user trajectories
3. **Geographic Graph**: From POI coordinates (if available)
4. **Edge Dropout**: Applied during training with `keep_rate` parameters

---

## Special Requirements

### POI Coordinates
For optimal performance, provide POI coordinates in `pois_coos_dict`:
```python
pois_coos_dict = {
    poi_id: (latitude, longitude),
    ...
}
```

If coordinates are not available, DCHL will use an identity matrix for the geographic graph (no spatial learning).

### Session Format
Sessions can be:
- List of lists: `[[poi1, poi2], [poi3, poi4]]` (multiple sessions per user)
- Flat list: `[poi1, poi2, poi3, poi4]` (single session per user)

Both formats are automatically handled by the model.

---

## Hyperparameter Tuning Recommendations

### Key Parameters to Tune

1. **Embedding Dimension** (`emb_dim`):
   - Default: 128
   - Range: 64-256
   - Higher values for larger datasets

2. **Contrastive Loss Weight** (`lambda_cl`):
   - Default: 0.1
   - Range: 0.01-0.5
   - Balance between reconstruction and contrastive learning

3. **Temperature** (`temperature`):
   - Default: 0.1
   - Range: 0.05-0.2
   - Lower values for harder contrastive learning

4. **Distance Threshold** (`distance_threshold`):
   - Default: 2.5 km
   - Adjust based on city size and POI density
   - Urban: 1-2 km, Suburban: 3-5 km

5. **Number of Layers**:
   - All default to 3
   - Range: 2-5
   - More layers for larger graphs (risk of over-smoothing)

### Edge Dropout for Regularization

- `keep_rate`: User-POI hypergraph edge retention
- `keep_rate_poi`: POI-POI hypergraph edge retention
- Values < 1.0 provide regularization (e.g., 0.8-0.95)

---

## Verification Checklist

- [x] Model registered in `task_config.json`
- [x] Model configuration file created with all hyperparameters
- [x] Model imported in `__init__.py`
- [x] Model implementation adapted for LibCity
- [x] Dataset compatibility verified
- [x] Required methods implemented (`forward`, `predict`, `calculate_loss`)
- [x] Graph initialization methods implemented
- [x] Documentation created

---

## Known Limitations

1. **Memory Requirements**:
   - Hypergraph structures require significant memory for large datasets
   - Consider batch processing for datasets with >100K POIs

2. **Graph Construction Time**:
   - Building hypergraphs from scratch can be slow
   - Recommend precomputing and caching graphs for large datasets

3. **Geographic Data**:
   - Performance degrades without POI coordinates
   - Spatial view uses identity matrix if coordinates unavailable

4. **Sparse Data**:
   - Requires sufficient user-POI interactions
   - Minimum sessions per user: 2 (from TrajectoryDataset config)
   - Minimum check-ins: 3

---

## References

- **Paper**: Yantong Lai et al. "Disentangled Contrastive Hypergraph Learning for Next POI Recommendation", SIGIR 2024
- **Original Code**: https://github.com/icmpnorequest/SIGIR2024_DCHL
- **LibCity Framework**: https://github.com/LibCity/Bigscity-LibCity

---

## Configuration Files

### File Paths
1. Task config: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
2. Model config: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/DCHL.json`
3. Dataset config: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/data/TrajectoryDataset.json`
4. Model implementation: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/DCHL.py`

---

**Configuration Date**: 2026-02-04
**LibCity Branch**: shk
**Status**: ✓ Complete and Verified
