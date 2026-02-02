# JGRM Configuration Migration Summary

## Config Migration: JGRM

### 1. task_config.json Updates

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`

#### Changes Made:
- **Added to**: `traj_loc_pred.allowed_model` list (line 32)
- **Task configuration** (lines 203-208):
  ```json
  "JGRM": {
      "dataset_class": "TrajectoryDataset",
      "executor": "TrajLocPredExecutor",
      "evaluator": "TrajLocPredEvaluator",
      "traj_encoder": "StandardTrajectoryEncoder"
  }
  ```

### 2. Model Config

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/JGRM.json`

#### Complete Parameters List:

| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| **Sequence Parameters** |
| `route_max_len` | 100 | Original config | Maximum route sequence length |
| **Embedding Dimensions** |
| `road_feat_num` | 1 | Original config | Number of road features |
| `road_embed_size` | 128 | Original config | Road segment embedding dimension |
| `gps_feat_num` | 8 | Original config | Number of GPS point features (speed, acceleration, angle, etc.) |
| `gps_embed_size` | 128 | Original config | GPS feature embedding dimension |
| `route_embed_size` | 128 | Original config | Route representation dimension |
| `hidden_size` | 256 | Original config | Hidden layer dimension for all encoders |
| **Dropout Rates** |
| `drop_edge_rate` | 0.1 | Original config | Dropout rate for GAT edges |
| `drop_route_rate` | 0.1 | Original config | Dropout rate for route transformer |
| `drop_road_rate` | 0.1 | Original config | Dropout rate for shared transformer |
| **Masking Parameters (MLM)** |
| `mask_length` | 2 | Original config | Length of consecutive masked segments |
| `mask_prob` | 0.2 | Original config | Probability for masking in MLM |
| **Loss Parameters** |
| `tau` | 0.07 | Original config | Temperature for contrastive loss |
| `mlm_loss_weight` | 1.0 | Default | Weight for MLM losses |
| `match_loss_weight` | 2.0 | Default | Weight for GPS-Route matching loss |
| **Model Architecture** |
| `mode` | "x" | Original config | "p" for pretrain embeddings, "x" for GAT |
| `route_transformer_layers` | 4 | Original implementation | Number of layers in route transformer |
| `route_transformer_heads` | 8 | Original implementation | Number of attention heads in route transformer |
| `shared_transformer_layers` | 2 | Original implementation | Number of layers in shared transformer |
| `shared_transformer_heads` | 4 | Original implementation | Number of attention heads in shared transformer |
| **Training Parameters** |
| `learning_rate` | 0.003 | Original config | Initial learning rate |
| `weight_decay` | 1e-6 | Original config | Weight decay for regularization |
| `batch_size` | 64 | Original config | Training batch size |
| `epochs` | 20 | Original config | Number of training epochs |
| `warmup_step` | 1000 | Original config | Number of warmup steps for learning rate |

### 3. Model Code Updates

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/JGRM.py`

#### Changes Made:

1. **Added transformer architecture parameters to config loading** (lines 238-242):
   ```python
   # Transformer architecture parameters
   self.route_transformer_layers = config.get('route_transformer_layers', 4)
   self.route_transformer_heads = config.get('route_transformer_heads', 8)
   self.shared_transformer_layers = config.get('shared_transformer_layers', 2)
   self.shared_transformer_heads = config.get('shared_transformer_heads', 4)
   ```

2. **Updated route transformer instantiation** (lines 264-267):
   ```python
   self.route_encoder = TransformerModel(
       self.hidden_size, self.route_transformer_heads, self.hidden_size,
       self.route_transformer_layers, self.drop_route_rate
   )
   ```
   - Previously hardcoded: `num_heads=8, num_layers=4`
   - Now configurable via config parameters

3. **Updated shared transformer instantiation** (lines 288-291):
   ```python
   self.sharedtransformer = TransformerModel(
       self.hidden_size, self.shared_transformer_heads, self.hidden_size,
       self.shared_transformer_layers, self.drop_road_rate
   )
   ```
   - Previously hardcoded: `num_heads=4, num_layers=2`
   - Now configurable via config parameters

### 4. Dataset Compatibility

#### Allowed Datasets (from task_config.json):
- foursquare_tky
- foursquare_nyc
- gowalla
- foursquare_serm
- Proto

#### Dataset Requirements:

**IMPORTANT**: JGRM has specialized data requirements beyond standard trajectory datasets:

1. **GPS Point Features** (8 features):
   - Speed
   - Acceleration
   - Angle/Bearing
   - Temporal features
   - Spatial features

2. **Road Network Data**:
   - `vocab_size`: Number of road segments in the network
   - `edge_index`: Road network adjacency graph (COO format: shape [2, num_edges])
   - Road segment to GPS point mapping

3. **Trajectory Sequences**:
   - Route segment sequences (road IDs)
   - GPS point sequences aligned with road segments
   - Temporal features: weekday, minute of day, time delta

#### Expected Batch Format:

```python
batch = {
    'route_data': torch.Tensor,         # (batch, seq_len, 3) - weekday, minute, delta
    'route_assign_mat': torch.Tensor,   # (batch, seq_len) - segment indices
    'gps_data': torch.Tensor,           # (batch, gps_len, 8) - GPS features
    'gps_assign_mat': torch.Tensor,     # (batch, gps_len) - GPS-to-segment assignments
    'gps_length': torch.Tensor          # (batch, seq_len) - GPS points per segment
}
```

**Note**: Standard LibCity trajectory datasets (Foursquare, Gowalla) likely need custom preprocessing to generate these data structures. JGRM was originally designed for road-based GPS trajectories with map-matching.

### 5. Configuration Consistency Verification

#### Config-to-Code Mapping:

| Config Parameter | Model Code Reference | Status |
|-----------------|---------------------|--------|
| `route_max_len` | `self.route_max_len` | ✓ Used |
| `road_feat_num` | `self.road_feat_num` | ✓ Used |
| `road_embed_size` | `self.road_embed_size` | ✓ Used |
| `gps_feat_num` | `self.gps_feat_num` | ✓ Used |
| `gps_embed_size` | `self.gps_embed_size` | ✓ Used |
| `route_embed_size` | `self.route_embed_size` | ✓ Used |
| `hidden_size` | `self.hidden_size` | ✓ Used |
| `drop_edge_rate` | `self.drop_edge_rate` | ✓ Used in `encode_graph()` |
| `drop_route_rate` | `self.drop_route_rate` | ✓ Used in route transformer |
| `drop_road_rate` | `self.drop_road_rate` | ✓ Used in shared transformer |
| `mask_length` | `self.mask_length` | ✓ Used in `random_mask()` |
| `mask_prob` | `self.mask_prob` | ✓ Used in `random_mask()` |
| `tau` | `self.tau` | ✓ Used in `get_traj_match_loss()` |
| `mode` | `self.mode` | ✓ Used in `encode_route()` |
| `mlm_loss_weight` | `self.mlm_loss_weight` | ✓ Used in `calculate_loss()` |
| `match_loss_weight` | `self.match_loss_weight` | ✓ Used in `calculate_loss()` |
| `route_transformer_layers` | `self.route_transformer_layers` | ✓ Used in route encoder |
| `route_transformer_heads` | `self.route_transformer_heads` | ✓ Used in route encoder |
| `shared_transformer_layers` | `self.shared_transformer_layers` | ✓ Used in shared transformer |
| `shared_transformer_heads` | `self.shared_transformer_heads` | ✓ Used in shared transformer |

All parameters are correctly read from config and used in model initialization.

### 6. Notes and Limitations

#### Compatibility Concerns:

1. **Data Preprocessing Required**:
   - JGRM expects GPS trajectories with road segment assignments (map-matching)
   - Standard POI check-in datasets (Foursquare, Gowalla) lack road network structure
   - Custom dataset preprocessing pipeline needed

2. **Road Network Dependency**:
   - Model requires road network graph (`edge_index`) in `data_feature`
   - GAT encoder expects torch_geometric library
   - Fallback to MLP if torch_geometric not available

3. **GPU Memory Considerations**:
   - Dual-branch architecture + joint transformer is memory-intensive
   - Recommend batch_size <= 64 for typical GPU memory
   - GPS sequence length can vary significantly (impacts memory)

4. **Task Mismatch**:
   - JGRM is fundamentally a trajectory representation learning model
   - Placed in `traj_loc_pred` for framework compatibility
   - Primary output is trajectory embeddings, not next location predictions
   - May need custom evaluator for representation quality metrics

#### Recommended Usage:

```python
# For trajectory embedding extraction
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.model import JGRM

# Load config
config = ConfigParser(task='traj_loc_pred', model='JGRM', dataset='custom_gps_dataset')

# Initialize model
model = JGRM(config, data_feature)

# Training
loss = model.calculate_loss(batch)

# Inference - get embeddings
embeddings = model.predict(batch)  # Returns (batch_size, hidden_size=256)
```

### 7. Validation Checklist

- [x] Model added to `allowed_model` list in task_config.json
- [x] Task-specific configuration added for JGRM
- [x] All hyperparameters from original paper/code included in config
- [x] Transformer architecture parameters made configurable
- [x] Model code updated to use configurable parameters
- [x] Config parameter consistency verified
- [x] JSON syntax validated
- [x] Model registered in `__init__.py`
- [x] Documentation updated

### 8. Testing Recommendations

1. **Import Test**:
   ```python
   from libcity.model.trajectory_loc_prediction import JGRM
   ```

2. **Config Loading Test**:
   ```python
   config = ConfigParser(task='traj_loc_pred', model='JGRM')
   print(config)  # Verify all parameters loaded
   ```

3. **Model Initialization Test**:
   ```python
   model = JGRM(config, data_feature)
   print(model)  # Check architecture
   ```

4. **Forward Pass Test** (requires proper batch format):
   ```python
   output = model(batch)
   loss = model.calculate_loss(batch)
   embeddings = model.predict(batch)
   ```

### 9. Dataset Preparation Guide

For using JGRM with custom datasets:

1. **Prepare Road Network**:
   - Extract road network topology
   - Create edge_index in COO format
   - Assign road segment IDs (0 to vocab_size-1)

2. **Map-Matching**:
   - Match GPS trajectories to road segments
   - Generate route_assign_mat (road segment sequences)
   - Generate gps_assign_mat (GPS-to-segment mapping)
   - Calculate gps_length (GPS points per segment)

3. **Feature Engineering**:
   - Extract GPS features: speed, acceleration, bearing, etc.
   - Extract temporal features: weekday, minute, time delta
   - Normalize features appropriately

4. **Data Loading**:
   - Implement custom dataset class if needed
   - Or adapt TrajectoryDataset to output required batch format
   - Ensure data_feature contains vocab_size and edge_index

---

## Summary

The JGRM model has been successfully configured for LibCity framework:

- **Task Type**: `traj_loc_pred`
- **Model Config**: `/Bigscity-LibCity/libcity/config/model/traj_loc_pred/JGRM.json`
- **Task Config**: Updated with JGRM entry
- **Model Code**: Updated to use configurable transformer architecture
- **All Parameters**: Sourced from original implementation and paper defaults
- **Status**: Ready for testing with appropriate dataset preprocessing
