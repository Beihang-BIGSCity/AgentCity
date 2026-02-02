# Config Migration: DCHL

## Model Overview
- **Model Name**: DCHL (Disentangled Contrastive Hypergraph Learning)
- **Paper**: [SIGIR 2024] Disentangled Contrastive Hypergraph Learning for Next POI Recommendation
- **Task Type**: Trajectory Location Prediction (Next POI Recommendation)
- **Model File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/DCHL.py`

## Configuration Files Status

### 1. task_config.json
- **Location**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
- **Status**: ALREADY REGISTERED
- **Registration**: Added to `traj_loc_pred.allowed_model` (line 31)
- **Configuration Block**: Lines 198-203
```json
"DCHL": {
    "dataset_class": "TrajectoryDataset",
    "executor": "TrajLocPredExecutor",
    "evaluator": "TrajLocPredEvaluator",
    "traj_encoder": "StandardTrajectoryEncoder"
}
```

### 2. Model Configuration File
- **Location**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/DCHL.json`
- **Status**: UPDATED (added `keep_rate_poi` parameter)
- **Configuration**:

```json
{
    "emb_dim": 128,
    "num_mv_layers": 3,
    "num_geo_layers": 3,
    "num_di_layers": 3,
    "dropout": 0.3,
    "temperature": 0.1,
    "lambda_cl": 0.1,
    "keep_rate": 1.0,
    "keep_rate_poi": 1.0,
    "distance_threshold": 2.5,
    "learning_rate": 0.001,
    "lr_decay": 0.1,
    "weight_decay": 0.0005,
    "batch_size": 200,
    "max_epoch": 30
}
```

### 3. Model Import Registration
- **Location**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py`
- **Status**: ALREADY REGISTERED
- **Import**: Line 23 - `from libcity.model.trajectory_loc_prediction.DCHL import DCHL`
- **Export**: Line 47 - Added to `__all__` list

## Hyperparameter Details

### Core Model Parameters (from original paper)
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `emb_dim` | 128 | Paper default | Embedding dimension for users and POIs |
| `num_mv_layers` | 3 | Paper default | Number of multi-view hypergraph conv layers |
| `num_geo_layers` | 3 | Paper default | Number of geographic graph conv layers |
| `num_di_layers` | 3 | Paper default | Number of directed hypergraph conv layers |
| `dropout` | 0.3 | Paper default | Dropout probability for regularization |
| `lambda_cl` | 0.1 | Paper default | Weight for contrastive learning loss |
| `temperature` | 0.1 | Paper default | Temperature parameter for contrastive learning |
| `keep_rate` | 1.0 | Default (no dropout) | Keep rate for general dropout augmentation |
| `keep_rate_poi` | 1.0 | Default (no dropout) | Keep rate for POI-specific dropout augmentation |

### Training Parameters
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `learning_rate` | 0.001 | Paper/LibCity | Initial learning rate for optimizer |
| `lr_decay` | 0.1 | LibCity convention | Learning rate decay factor |
| `weight_decay` | 0.0005 | Paper default | L2 regularization weight |
| `batch_size` | 200 | Paper default | Training batch size |
| `max_epoch` | 30 | Paper default | Maximum training epochs |

### Data Processing Parameters
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `distance_threshold` | 2.5 | Paper (2.5km) | Distance threshold for POI geographic adjacency graph construction |

## Required Data Features

DCHL requires extensive precomputed graph structures in the `data_feature` dictionary:

### Essential Graph Structures
1. **H_pu**: POI-User hypergraph incidence matrix [num_pois, num_users]
2. **HG_pu**: Normalized POI-User hypergraph [num_pois, num_users]
3. **H_up**: User-POI hypergraph incidence matrix [num_users, num_pois]
4. **HG_up**: Normalized User-POI hypergraph [num_users, num_pois]
5. **HG_poi_src**: POI transition source hypergraph (directed)
6. **HG_poi_tar**: POI transition target hypergraph (directed)
7. **poi_geo_graph**: POI geographic adjacency graph (based on spatial distance)

### Additional Features
- **num_users**: Total number of unique users in dataset
- **num_pois**: Total number of unique POIs/locations in dataset
- **pad_all_train_sessions**: Padded training sessions [num_users, max_session_length] (optional)

### Graph Construction Requirements
- **Geographic Graph**: Requires POI coordinates (latitude, longitude) to compute geographic proximity within `distance_threshold` (default 2.5km)
- **Hypergraph Structures**: Require complete user-POI interaction trajectories
- **Directed Hypergraphs**: Require sequential POI transitions from trajectories

## Dataset Compatibility

### Original Paper Datasets
- **NYC**: 834 users, 3,835 POIs (Foursquare check-ins)
- **TKY/Tokyo**: 2,173 users, 7,038 POIs (Foursquare check-ins)

### LibCity Compatible Datasets
According to `task_config.json`, the following datasets are allowed for trajectory location prediction:
- `foursquare_tky` - Tokyo Foursquare dataset
- `foursquare_nyc` - New York City Foursquare dataset
- `gowalla` - Gowalla check-in dataset
- `foursquare_serm` - Foursquare SERM dataset
- `Proto` - Prototype dataset

### Dataset Requirements
For DCHL to work properly, datasets must include:
1. **User-POI interaction sequences** (trajectories)
2. **POI coordinates** (latitude, longitude) for geographic graph construction
3. **Temporal information** for sequential pattern learning
4. **Sufficient interaction density** for hypergraph construction

### Compatibility Notes
- DCHL uses `TrajectoryDataset` class (standard LibCity trajectory dataset)
- Standard trajectory encoder: `StandardTrajectoryEncoder`
- The model requires preprocessing to construct all hypergraph structures
- Datasets with sparse interactions may not work well due to hypergraph requirements

## Model Architecture Summary

### Three-View Disentangled Learning
1. **Collaborative View**: Multi-view hypergraph convolution on user-POI interactions
   - Captures collaborative filtering signals
   - Uses bidirectional user-POI hypergraphs (H_pu, HG_pu, H_up, HG_up)

2. **Geographic View**: Graph convolution on POI spatial adjacency
   - Captures spatial proximity patterns
   - Requires POI coordinates and geographic graph

3. **Sequential View**: Directed hypergraph convolution on POI transitions
   - Captures sequential transition patterns
   - Uses directed hypergraphs (HG_poi_src, HG_poi_tar)

### Cross-View Contrastive Learning
- Aligns representations from different views using InfoNCE loss
- Applied to both POI and user embeddings
- Controlled by `lambda_cl` and `temperature` parameters

### Adaptive Fusion
- Self-gating mechanisms to preserve view-specific information
- Learned gates for combining multi-view embeddings
- Final prediction via user-POI inner product

## Implementation Notes

### LibCity Adaptations
- Inherited from `AbstractModel` (LibCity framework)
- Implemented `predict()` method for inference
- Implemented `calculate_loss()` for combined loss computation
- Graph structures stored as model attributes (loaded from data_feature)
- Standard LibCity batch format expected

### Loss Computation
Total loss = `loss_rec + lambda_cl * (loss_cl_pois + loss_cl_users)`
- `loss_rec`: Cross-entropy recommendation loss
- `loss_cl_pois`: Cross-view contrastive loss for POIs
- `loss_cl_users`: Cross-view contrastive loss for users

### Special Considerations
1. **Memory Requirements**: Model maintains multiple graph structures in GPU memory
2. **Preprocessing Overhead**: Hypergraph construction is computationally expensive
3. **Data Sparsity**: Performance degrades with sparse user-POI interactions
4. **Geographic Data**: Essential for geographic view; missing coordinates will cause errors

## Usage Example

```python
# Example configuration for running DCHL
{
    "task": "traj_loc_pred",
    "model": "DCHL",
    "dataset": "foursquare_nyc",
    "emb_dim": 128,
    "num_mv_layers": 3,
    "num_geo_layers": 3,
    "num_di_layers": 3,
    "dropout": 0.3,
    "lambda_cl": 0.1,
    "temperature": 0.1,
    "keep_rate": 1.0,
    "keep_rate_poi": 1.0,
    "distance_threshold": 2.5,
    "learning_rate": 0.001,
    "batch_size": 200,
    "max_epoch": 30
}
```

## Validation Checklist

- [x] Model registered in `task_config.json` allowed_model list
- [x] Model configuration block exists in `task_config.json`
- [x] Model config file created at `config/model/traj_loc_pred/DCHL.json`
- [x] All hyperparameters from paper included
- [x] `keep_rate_poi` parameter added
- [x] Model imported in `__init__.py`
- [x] Model exported in `__all__` list
- [x] Uses standard `TrajectoryDataset` class
- [x] Compatible with `TrajLocPredExecutor` and `TrajLocPredEvaluator`
- [ ] Data preprocessing pipeline for graph construction (requires implementation)

## Known Limitations

1. **Graph Preprocessing**: DCHL requires extensive graph preprocessing that is not currently automated in LibCity's standard data pipeline. Users need to:
   - Compute hypergraph incidence matrices (H_pu, H_up)
   - Normalize hypergraphs (HG_pu, HG_up)
   - Build directed transition hypergraphs (HG_poi_src, HG_poi_tar)
   - Construct geographic adjacency graph (poi_geo_graph)

2. **Dataset Requirements**: Not all LibCity trajectory datasets may have the required POI coordinate information for geographic graph construction.

3. **Computational Complexity**: The model maintains multiple large graph structures, requiring significant GPU memory for large-scale datasets.

4. **Interaction Density**: The hypergraph-based approach requires sufficient user-POI interaction density. Very sparse datasets may produce degenerate hypergraphs.

## Future Work

- Implement automated graph preprocessing pipeline in dataset class
- Add support for datasets without geographic coordinates (disable geographic view)
- Optimize memory usage for large-scale datasets
- Add parameter validation and informative error messages for missing graphs

## References

- Original Paper: "Disentangled Contrastive Hypergraph Learning for Next POI Recommendation" (SIGIR 2024)
- Original Author: Yantong Lai
- LibCity Framework: https://github.com/LibCity/Bigscity-LibCity
