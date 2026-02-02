# Config Migration: JGRM

## Overview
JGRM (Joint GPS and Route Multimodal Model) for Trajectory Representation Learning has been successfully configured in LibCity.

**Model Type:** Trajectory Location Prediction (traj_loc_pred)
**Paper:** "More Than Routing: Joint GPS and Route Modeling for Refine Trajectory Representation Learning" (WWW 2024)
**Repository:** https://github.com/mamazi0131/JGRM

## 1. task_config.json Registration

### Status: ALREADY REGISTERED
- **Task Type:** traj_loc_pred
- **Location:** Line 32 in /home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json
- **Encoder:** JGRMEncoder
- **Dataset Class:** TrajectoryDataset
- **Executor:** TrajLocPredExecutor
- **Evaluator:** TrajLocPredEvaluator

```json
"JGRM": {
    "dataset_class": "TrajectoryDataset",
    "executor": "TrajLocPredExecutor",
    "evaluator": "TrajLocPredEvaluator",
    "traj_encoder": "JGRMEncoder"
}
```

## 2. Model Configuration

### File Location
`/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/JGRM.json`

### Updated Parameters

#### Core Model Architecture
| Parameter | Value | Source |
|-----------|-------|--------|
| model_name | "JGRM" | LibCity convention |
| route_max_len | 100 | Paper default |
| road_embed_size | 128 | Paper default |
| gps_feat_num | 8 | GPS features: timestamp, lng, lat, speed, acceleration, angle_delta, interval, distance |
| gps_embed_size | 128 | Paper default |
| route_embed_size | 128 | Paper default |
| hidden_size | 256 | Paper default |

#### Transformer Configuration
| Parameter | Value | Source |
|-----------|-------|--------|
| route_transformer_layers | 4 | Paper default |
| route_transformer_heads | 8 | Paper default |
| shared_transformer_layers | 2 | Paper default |
| shared_transformer_heads | 4 | Paper default |

#### Training Configuration
| Parameter | Value | Source |
|-----------|-------|--------|
| mode | "x" | Use graph encoding (GAT) |
| dropout (drop_route_rate) | 0.1 | Paper default |
| dropout (drop_road_rate) | 0.1 | Paper default |
| drop_edge_rate | 0.1 | Graph edge dropout |

#### Pre-training Objectives
| Parameter | Value | Source |
|-----------|-------|--------|
| mask_prob | 0.2 | Masking probability for MLM |
| mask_len | 2 | Masked span length |
| queue_size | 2048 | Contrastive learning queue |
| tau | 0.07 | Temperature for contrastive loss |
| mlm_loss_weight | 1.0 | MLM loss weight |
| match_loss_weight | 2.0 | Matching loss weight |

#### Optimization
| Parameter | Value | Source |
|-----------|-------|--------|
| optimizer | "AdamW" | Paper/Specification |
| learning_rate | 0.0005 | Specification (5e-4) |
| weight_decay | 1e-6 | Paper default |
| lr_scheduler | "linear_warmup" | Specification |
| warmup_step | 1000 | Paper default |
| batch_size | 64 | Paper default |
| epochs | 100 | Standard pre-training |

#### Additional Parameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| output_dim | 1 | LibCity convention |
| embed_dim | 32 | LibCity convention |

## 3. Dataset Compatibility

### Required Data Format

JGRM requires a specialized dual-stream data format with the following components:

#### Route Stream (Road Segment Sequence)
- **route_assign_mat**: Sequence of road segment IDs
- **route_data**: Temporal features [week, minute, delta_time]
  - week: Day of week (0-6)
  - minute: Minute of day (0-1439)
  - delta_time: Time interval between segments
- **masked_route_assign_mat**: Masked version for MLM pre-training

#### GPS Stream (GPS Point Sequence)
- **gps_data**: GPS features (8 dimensions)
  1. timestamp
  2. longitude
  3. latitude
  4. speed
  5. acceleration
  6. angle_delta (change in heading)
  7. interval (time since last point)
  8. distance (distance since last point)
- **masked_gps_assign_mat**: GPS-to-road segment assignments
- **gps_length**: List of GPS point counts per road segment

#### Road Network Graph
- **edge_index**: Adjacency matrix for GAT encoding
- **pretrained_road_embed**: Optional pre-trained road embeddings

### Compatible LibCity Datasets

According to task_config.json, the allowed datasets for traj_loc_pred are:
- foursquare_tky
- foursquare_nyc
- gowalla
- foursquare_serm
- Proto

**Important Compatibility Notes:**

1. **Standard Datasets Require Adaptation:**
   - Standard LibCity trajectory datasets use POI-based check-ins
   - JGRM requires road network + GPS trajectory data
   - May need custom dataset class or encoder to bridge the gap

2. **Custom Dataset Requirements:**
   - Road network topology (edge_index)
   - GPS traces with rich features (8-dimensional)
   - Road segment sequences aligned with GPS points
   - Temporal metadata (week, minute, intervals)

3. **Recommended Approach:**
   - Create a custom `JGRMDataset` class extending `TrajectoryDataset`
   - Or use a custom `JGRMEncoder` to transform standard trajectory data
   - Current configuration uses `JGRMEncoder` as specified in task_config.json

## 4. Model Architecture Details

### Dual-Stream Processing

#### GPS Stream Pipeline:
1. **Linear Projection**: Project 8D GPS features to embedding space
2. **Intra-road GRU**: Bidirectional GRU within each road segment
3. **Inter-road GRU**: Bidirectional GRU across road segments
4. **Projection Head**: Map to shared representation space

#### Route Stream Pipeline:
1. **Node Embedding**: Look up road segment embeddings
2. **Graph Encoder (Optional)**: 2-layer GAT for topology encoding
3. **Time Embedding**: Week + Minute + Interval embeddings
4. **Transformer Encoder**: 4-layer transformer with 8 heads
5. **Projection Head**: Map to shared representation space

### Joint Encoding
- **Shared Transformer**: 2-layer transformer with 4 heads
- **Modal Embeddings**: Distinguish GPS vs Route modalities
- **Position Embeddings**: Sequence position encoding

### Pre-training Objectives

1. **Masked Language Modeling (MLM)**:
   - Predict masked road segments from GPS stream
   - Predict masked road segments from Route stream
   - Loss weight: 1.0 each

2. **GPS-Route Matching**:
   - Contrastive learning with queue (2048 negatives)
   - Temperature-scaled similarity (tau=0.07)
   - Loss weight: 2.0

3. **Combined Loss**:
   ```
   total_loss = (mlm_gps + mlm_route + 2 * matching) / 4
   ```

## 5. Special Requirements

### Dependencies
- **torch-geometric**: Required for GAT layers
  ```bash
  pip install torch-geometric
  ```
- If torch-geometric is not available, model falls back to plain embeddings (mode='p')

### Device Management
- Model automatically uses device from config
- Supports GPU training with automatic device placement

### Queue-based Contrastive Learning
- Maintains momentum queues for GPS and Route representations
- Queue size: 2048
- Automatically updates during training

## 6. Usage Example

### Basic Training
```python
from libcity.pipeline import run_model

run_model(
    task='traj_loc_pred',
    model='JGRM',
    dataset='your_dataset',
    config_file='your_config.json'
)
```

### Custom Configuration
```json
{
    "task": "traj_loc_pred",
    "model": "JGRM",
    "dataset": "your_dataset",
    "hidden_size": 512,
    "learning_rate": 0.001,
    "batch_size": 128,
    "epochs": 200
}
```

### Extract Trajectory Representations
```python
model = JGRM(config, data_feature)
representations = model.get_trajectory_representation(batch)

# Access different representations:
# - representations['gps_traj']: GPS trajectory embedding
# - representations['route_traj']: Route trajectory embedding
# - representations['joint_gps_traj']: Joint GPS embedding
# - representations['joint_route_traj']: Joint Route embedding
# - representations['combined']: Concatenated joint embeddings
```

## 7. Comparison with Original Implementation

### Parameter Mapping
| Original | LibCity | Value |
|----------|---------|-------|
| hidden_dim | hidden_size | 256 |
| lr | learning_rate | 0.0005 |
| num_layers | route_transformer_layers | 4 |
| n_heads | route_transformer_heads | 8 |
| max_len | route_max_len | 100 |
| dropout | drop_route_rate | 0.1 |

### Adaptations for LibCity
1. **Base Class**: Changed from custom BaseModel to AbstractModel
2. **Config Management**: Use LibCity's config and data_feature pattern
3. **Loss Calculation**: Implemented calculate_loss() method
4. **Prediction Interface**: Implemented predict() method
5. **Device Handling**: Integrated with LibCity's device management

## 8. Known Limitations and Notes

### Data Requirements
- **High Data Complexity**: Requires synchronized GPS and road segment data
- **Graph Data**: Needs road network topology (adjacency matrix)
- **Feature Engineering**: 8D GPS features may need careful preprocessing

### Computational Costs
- **Dual-stream Processing**: Approximately 2x computation vs single-stream
- **Contrastive Queue**: Additional 2048 * hidden_size memory
- **Graph Encoding**: GAT layers add computational overhead

### Dataset Adaptation
- Standard POI check-in datasets may not directly work
- May require custom data preprocessing or encoder
- GPS trajectory datasets with map-matching are ideal

## 9. Future Enhancements

### Potential Improvements
1. **Custom Dataset Class**: Create `JGRMDataset` for better data handling
2. **Pre-trained Embeddings**: Support loading pre-trained road embeddings
3. **Fine-tuning Interface**: Add methods for downstream task fine-tuning
4. **Visualization Tools**: Tools to visualize learned representations
5. **Data Augmentation**: Trajectory augmentation strategies

### Research Extensions
1. **Multi-modal Fusion**: Additional modalities (traffic, weather)
2. **Hierarchical Modeling**: Multi-scale spatial-temporal modeling
3. **Transfer Learning**: Pre-train on large datasets, fine-tune on small ones
4. **Zero-shot Learning**: Generalize to unseen road networks

## 10. Testing Checklist

- [x] Model config file created (JGRM.json)
- [x] Model registered in task_config.json
- [x] All required hyperparameters included
- [x] Optimizer and scheduler configured
- [x] Loss weights specified
- [ ] Test with sample dataset (requires JGRM-compatible data)
- [ ] Verify encoder implementation (JGRMEncoder)
- [ ] Test training loop
- [ ] Test prediction interface
- [ ] Validate representation quality

## 11. References

- **Paper**: "More Than Routing: Joint GPS and Route Modeling for Refine Trajectory Representation Learning", WWW 2024
- **Original Repository**: https://github.com/mamazi0131/JGRM
- **Model Implementation**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/JGRM.py`
- **Config File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/JGRM.json`
- **Task Config**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`

## 12. Contact and Support

For issues related to:
- **LibCity Integration**: Check LibCity documentation and GitHub issues
- **JGRM Model**: Refer to original paper and repository
- **Custom Dataset**: May need to implement custom encoder or dataset class

---

**Migration Date**: 2026-02-02
**Status**: Configuration Complete, Testing Pending
**Next Steps**: Implement/verify JGRMEncoder and test with compatible dataset
