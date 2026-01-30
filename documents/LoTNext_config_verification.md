# LoTNext Configuration Verification Report

## Config Migration: LoTNext

**Date:** 2026-01-30
**Model:** LoTNext (Long-tail Next POI Prediction)
**Task:** trajectory_loc_prediction (traj_loc_pred)
**LibCity Path:** Bigscity-LibCity

---

## 1. Task Configuration Verification

### task_config.json Entry

**Location:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`

**Status:** VERIFIED

- **Added to:** `traj_loc_pred.allowed_model` (Line 20)
- **Configuration Block:** Lines 119-124

```json
"LoTNext": {
    "dataset_class": "TrajectoryDataset",
    "executor": "TrajLocPredExecutor",
    "evaluator": "TrajLocPredEvaluator",
    "traj_encoder": "StandardTrajectoryEncoder"
}
```

### Verification Results

| Component | Expected | Actual | Status |
|-----------|----------|--------|--------|
| dataset_class | TrajectoryDataset | TrajectoryDataset | PASS |
| executor | TrajLocPredExecutor | TrajLocPredExecutor | PASS |
| evaluator | TrajLocPredEvaluator | TrajLocPredEvaluator | PASS |
| traj_encoder | StandardTrajectoryEncoder | StandardTrajectoryEncoder | PASS |

**Notes:**
- TrajectoryDataset is appropriate for trajectory-based POI prediction
- Uses standard trajectory encoder (no custom encoder needed)
- Executor and evaluator match other trajectory prediction models (DeepMove, GETNext)

---

## 2. Model Registration Verification

### __init__.py Entry

**Location:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py`

**Status:** VERIFIED

- **Import Statement:** Line 15 - `from libcity.model.trajectory_loc_prediction.LoTNext import LoTNext`
- **__all__ Entry:** Line 32 - `"LoTNext"`

---

## 3. Hyperparameter Configuration

### Original Paper Parameters

Source: `/home/wangwenrui/shk/AgentCity/repos/LoTNext/setting.py`

| Parameter | Gowalla Default | Foursquare Default | Source Line |
|-----------|----------------|-------------------|-------------|
| hidden_dim | 10 | 10 | 86 |
| learning_rate | 0.01 | 0.01 | 88 |
| epochs | 100 | 100 | 89 |
| batch_size | 200 | 256 | 160, 169 |
| lambda_t | 0.1 | 0.1 | 162, 171 |
| lambda_s | 1000 | 100 | 163, 172 |
| transformer_nhid | 32 | 32 | 95 |
| transformer_nlayers | 2 | 2 | 99 |
| transformer_nhead | 2 | 2 | 103 |
| transformer_dropout | 0.3 | 0.3 | 107 |
| attention_dropout_rate | 0.1 | 0.1 | 111 |
| time_embed_dim | 32 | 32 | 115 |
| user_embed_dim | 128 | 128 | 119 |
| sequence_length | 20 | 20 | 56 |
| weight_decay | 0.0 | 0.0 | 87 |

### Updated LoTNext.json Configuration

**Location:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/LoTNext.json`

**Status:** UPDATED (Gowalla defaults)

```json
{
    "hidden_size": 10,
    "loc_emb_size": 10,
    "time_emb_size": 32,
    "user_emb_size": 128,
    "rnn_type": "LSTM",
    "batch_size": 200,
    "sequence_length": 20,
    "transformer_nhid": 32,
    "transformer_nlayers": 2,
    "transformer_nhead": 2,
    "transformer_dropout": 0.3,
    "attention_dropout_rate": 0.1,
    "lambda_loc": 1.0,
    "lambda_user": 1.0,
    "lambda_t": 0.1,
    "lambda_s": 1000,
    "use_graph_user": false,
    "use_spatial_graph": false,
    "loc_loss_weight": 1.0,
    "time_loss_weight": 0.1,
    "num_time_slots": 168,
    "spatial_decay": 1000,
    "learning_rate": 0.01,
    "weight_decay": 0.0,
    "max_epoch": 100,
    "validate_epoch": 5,
    "optimizer": "adam",
    "dropout_p": 0.3,
    "logit_adj_post": 1,
    "tro_post_range": [0.25, 0.5, 0.75, 1, 1.5, 2],
    "logit_adj_train": 1,
    "tro_train": 1.0,
    "min_checkins": 101
}
```

### Parameter Mapping: Original -> LibCity

| Original Parameter | LibCity Parameter | Notes |
|--------------------|------------------|-------|
| hidden_dim | hidden_size | Core embedding dimension: **10 (not 128)** |
| hidden_dim | loc_emb_size | Location embedding matches hidden_dim |
| time_embed_dim | time_emb_size | Time2Vec output dimension: **32** |
| user_embed_dim | user_emb_size | User embedding dimension: **128** |
| lr | learning_rate | 0.01 (original paper) |
| lambda_t | lambda_t | Temporal decay: **0.1** |
| lambda_s | lambda_s | Spatial decay: **1000** (Gowalla) / **100** (Foursquare) |
| lambda_s | spatial_decay | Used in _compute_spatial_weight() |
| epochs | max_epoch | 100 epochs |

### Critical Corrections Made

1. **hidden_size**: Changed from **128** to **10** (original paper value)
2. **loc_emb_size**: Changed from **128** to **10** (matches hidden_size)
3. **time_emb_size**: Changed from **6** to **32** (original paper value)
4. **batch_size**: Changed from **32** to **200** (Gowalla default)
5. **transformer_nhid**: Changed from **256** to **32** (original paper value)
6. **transformer_nhead**: Changed from **4** to **2** (original paper value)
7. **transformer_dropout**: Changed from **0.1** to **0.3** (original paper value)
8. **spatial_decay**: Changed from **100.0** to **1000** (Gowalla default)
9. **learning_rate**: Changed from **0.001** to **0.01** (original paper value)

### Additional Parameters Added

1. **transformer_nlayers**: 2 (from original paper)
2. **user_emb_size**: 128 (from original paper)
3. **lambda_t**: 0.1 (temporal decay parameter)
4. **lambda_s**: 1000 (spatial decay for Gowalla)
5. **weight_decay**: 0.0 (L2 regularization)
6. **validate_epoch**: 5 (validation frequency)
7. **logit_adj_post**: 1 (long-tail logit adjustment)
8. **tro_post_range**: [0.25, 0.5, 0.75, 1, 1.5, 2] (temperature range)
9. **logit_adj_train**: 1 (training logit adjustment)
10. **tro_train**: 1.0 (training temperature)
11. **min_checkins**: 101 (minimum check-ins filter)

---

## 4. Dataset Compatibility

### Required Data Features

LoTNext expects the following data fields from LibCity datasets:

| Field | Type | Shape | Required | Used For |
|-------|------|-------|----------|----------|
| current_loc | Tensor | [batch_size, seq_len] | Yes | POI sequence |
| current_tim | Tensor | [batch_size, seq_len] | Recommended | Time slots (0-167) |
| uid | Tensor | [batch_size] | Recommended | User IDs |
| current_coord | Tensor | [batch_size, seq_len, 2] | Optional | GPS coordinates (lat, lon) |
| target | Tensor | [batch_size] | Yes | Target POI location |
| target_tim | Tensor | [batch_size] | Optional | Target time slot |

### Compatible LibCity Datasets

From `task_config.json` line 22-27:

- foursquare_tky
- foursquare_nyc
- gowalla
- foursquare_serm
- Proto

### Data Requirements Analysis

1. **POI Sequences**: Uses `current_loc` for trajectory locations
2. **Temporal Data**: Converts `current_tim` to normalized time slots (0-1 range)
3. **User Embeddings**: Uses `uid` for user-specific patterns
4. **Spatial Features**: Optionally uses `current_coord` for Haversine distance-based spatial weighting
5. **Multi-task Learning**: Predicts both location (required) and time slot (optional)

### TrajectoryDataset Compatibility

**Status:** COMPATIBLE

The standard `TrajectoryDataset` class provides:
- POI sequences via standard trajectory encoding
- User IDs via data_feature
- Time slot information
- Coordinate data (if available in raw data)

**Notes:**
- Spatial attention requires coordinate data in the dataset
- Time prediction is optional (gracefully disabled if target_tim is missing)
- Graph features (transition_graph, interact_graph) are optional

---

## 5. Model-Specific Features

### Graph-Enhanced Embeddings (Optional)

The model supports three types of graphs via `set_graphs()` method:

1. **Transition Graph** (transition_graph):
   - POI-to-POI temporal transitions
   - Enhanced with lambda_loc weighting
   - Used for location embedding refinement

2. **Spatial Graph** (spatial_graph):
   - Spatial proximity between POIs
   - Currently not actively used in forward pass

3. **Interaction Graph** (interact_graph):
   - User-POI bipartite graph
   - Processed through DenoisingGCNNet
   - Combines user and POI embeddings

### Multi-Task Learning

1. **Location Prediction** (Primary):
   - CrossEntropyLoss
   - Weight: `loc_loss_weight` (1.0)

2. **Time Slot Prediction** (Secondary):
   - MSELoss
   - Weight: `time_loss_weight` (0.1)
   - Optional (only if target_tim provided)

### Spatial Attention Weighting

- Uses Haversine distance for spatial proximity
- Exponential decay: exp(-distance / spatial_decay)
- spatial_decay = 1000 km (Gowalla) / 100 km (Foursquare)
- Only active if coordinates are available

### Temporal Encoding

- Time2Vec with sine activation
- Converts time slots to continuous embeddings
- Dimension: `time_emb_size` (32)
- Fused with location embeddings

---

## 6. Configuration Notes and Warnings

### Critical Parameter: hidden_size = 10

The original LoTNext paper uses an **extremely small hidden dimension of 10**. This is unusual but intentional:
- Designed for long-tail POI prediction with limited data
- Prevents overfitting on rare POIs
- Uses graph embeddings and transformer to compensate

**Warning:** Do NOT increase hidden_size to 128 or similar values, as this will:
- Deviate from the original paper
- May cause overfitting on long-tail POIs
- Requires much more training data

### Dataset-Specific Parameters

For **Foursquare** datasets, modify:
```json
{
    "batch_size": 256,
    "lambda_s": 100,
    "spatial_decay": 100
}
```

For **Gowalla** datasets, use current config (already set):
```json
{
    "batch_size": 200,
    "lambda_s": 1000,
    "spatial_decay": 1000
}
```

### torch_geometric Dependency

The model includes a fallback for missing torch_geometric:
- Full GCN functionality requires: `pip install torch-geometric`
- Without it, uses simple linear layers (reduced performance)

### Graph Data Preparation

To use graph-enhanced embeddings:
1. Construct transition graph (temporal POI-POI)
2. Construct interaction graph (user-POI bipartite)
3. Call `model.set_graphs(transition_graph, spatial_graph, interact_graph)` before training

Without graphs, model falls back to standard embeddings (still functional).

---

## 7. Verification Checklist

- [x] Model registered in `allowed_model` list
- [x] task_config.json entry complete with all required fields
- [x] dataset_class is appropriate (TrajectoryDataset)
- [x] executor is correct (TrajLocPredExecutor)
- [x] evaluator is appropriate (TrajLocPredEvaluator)
- [x] traj_encoder is set (StandardTrajectoryEncoder)
- [x] Model imported in __init__.py
- [x] Model added to __all__ list
- [x] LoTNext.json created with all hyperparameters
- [x] Hyperparameters match original paper (Gowalla defaults)
- [x] Critical correction: hidden_size = 10 (not 128)
- [x] Critical correction: time_emb_size = 32 (not 6)
- [x] Dataset compatibility verified
- [x] Optional features documented (graphs, coordinates, time)

---

## 8. Testing Recommendations

### Basic Import Test
```python
from libcity.model.trajectory_loc_prediction import LoTNext
print("LoTNext imported successfully")
```

### Configuration Test
```python
from libcity.config import ConfigParser
config = ConfigParser(task='traj_loc_pred', model='LoTNext',
                      dataset='gowalla')
print(config)
```

### Training Test
```python
from libcity.pipeline import run_model

# Basic training (no graphs)
run_model(
    task='traj_loc_pred',
    model='LoTNext',
    dataset='gowalla'
)

# With custom config for Foursquare
run_model(
    task='traj_loc_pred',
    model='LoTNext',
    dataset='foursquare_tky',
    config_file='custom_foursquare_config.json'
)
```

### Foursquare Custom Config
```json
{
    "batch_size": 256,
    "lambda_s": 100,
    "spatial_decay": 100
}
```

---

## 9. Summary

### Status: VERIFIED and UPDATED

The LoTNext configuration has been thoroughly reviewed and updated to match the original paper's hyperparameters. Key corrections include:

1. **Critical**: hidden_size corrected from 128 to 10 (original paper value)
2. **Critical**: time_emb_size corrected from 6 to 32 (original paper value)
3. All transformer parameters verified against original implementation
4. Dataset-specific defaults (Gowalla) applied
5. Long-tail reweighting parameters added

### Files Modified

1. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/LoTNext.json` - UPDATED
2. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json` - Already correct
3. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py` - Already correct

### Compatibility

- **LibCity Framework**: COMPATIBLE
- **TrajectoryDataset**: COMPATIBLE
- **Standard Executor/Evaluator**: COMPATIBLE
- **Reproduciblity**: HIGH (matches original paper defaults)

### Next Steps

1. Test model import and instantiation
2. Run training on Gowalla dataset with default config
3. Optionally prepare graph data for enhanced performance
4. Create Foursquare-specific config if needed
5. Evaluate on standard trajectory prediction metrics (Acc@1, Acc@5, Acc@10)
