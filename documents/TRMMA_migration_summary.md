## Repository: TRMMA
- **URL**: https://github.com/derekwtian/TRMMA
- **Cloned to**: /home/wangwenrui/shk/AgentCity/repos/TRMMA
- **Paper**: Efficient Methods for Accurate Sparse Trajectory Recovery and Map Matching (ICDE 2025)

### Key Files

#### Model Files
- **Main Model**: `/repos/TRMMA/models/trmma.py`
  - **Primary Class**: `TrajRecovery` (line 913)
  - **Architecture**: Encoder-Decoder with DualFormer (GPS + Route Transformer)
  - **Components**:
    - `GPSEncoder`: GPS trajectory encoder using GPSFormer
    - `GREncoder`: GPS+Route dual encoder using GRFormer
    - `DecoderMulti`: Multi-task decoder (segment ID + position ratio prediction)
    - `DAPlanner`: Route planning using Direction-Aware algorithm

- **Supporting Model**: `/repos/TRMMA/models/mma.py`
  - **Class**: `GPS2Seg` (Map Matching Attention model)
  - Used for pre-processing/map matching step

- **Layers**: `/repos/TRMMA/models/layers.py`
  - `GPSFormer`: GPS-only transformer encoder
  - `GRFormer`: GPS-Route dual transformer (DualFormer)
  - `Attention`: Custom attention mechanism
  - `MultiHeadAttention`, `FeedForward`, `Norm`: Standard transformer components

#### Training Scripts
- **Training**: `/repos/TRMMA/train_trmma.py`
  - Main training loop with multi-task loss (segment ID + position ratio)
  - Uses teacher forcing during training
  
- **Inference**: `/repos/TRMMA/infer_trmma.py`
  - Trajectory recovery inference
  
- **MMA Training**: `/repos/TRMMA/train_mma.py`
  - Map matching model training (preprocessing step)

#### Configuration
- **Args**: `/repos/TRMMA/args.py`
  - Workspace and data preparation arguments
  - Command-line argument parsing in training scripts

#### Data & Preprocessing
- **Dataset Classes**:
  - `TrajRecData`: Training/validation data loader (in trmma.py)
  - `TrajRecTestData`: Test data loader with route planning
  - `GPS2SegData`: Map matching data loader (in mma.py)

- **Preprocessing**: `/repos/TRMMA/preprocess/`
  - `dam.py`: Direction-Aware Matrix for route planning
  - `seg_info.py`: Road segment information extraction
  - `reformat.py`: Data reformatting utilities
  - `node2vec_.py`: Graph embedding (optional)

#### Utilities
- `/repos/TRMMA/utils/`
  - `map.py`: Road network map with R-tree spatial indexing
  - `spatial_func.py`: Spatial functions (distance, projection, GPS conversions)
  - `trajectory_func.py`: Trajectory data structures (STPoint)
  - `evaluation_utils.py`: Evaluation metrics (LCS, accuracy, MAE, RMSE)
  - `candidate_point.py`: Candidate segment generation
  - `mbr.py`: Minimum Bounding Rectangle utilities
  - `model_utils.py`: Model utility functions (gps2grid, normalization)

### Dependencies

#### Python Version
- **Python**: 3.8

#### Core Dependencies
- **PyTorch**: 1.13 with CUDA 11
- **numpy**: Standard numerical operations
- **pandas**: Data processing
- **networkx**: Graph operations (road network, routing)
- **tqdm**: Progress bars
- **pickle**: Data serialization

#### Spatial Libraries (Likely Required)
- Shapefile processing (for OSM map data - nodes.shp, edges.shp)
- R-tree spatial indexing (implied by map.py)

### Model Architecture Overview

#### Task Type
**Trajectory Recovery (Map Matching + Route Interpolation)**

The model performs two related tasks:
1. **Map Matching (MMA)**: Match sparse GPS points to road segments
2. **Trajectory Recovery (TRMMA)**: Recover high-sampling trajectories from sparse trajectories

#### Architecture Components

**TRMMA Model (`TrajRecovery` class):**

1. **Input Encoding**:
   - GPS coordinates (lat, lng, normalized time)
   - Optional: Source segment information
   - Optional: Positional embeddings
   - Route embeddings (from DA planner or learned)

2. **Encoder** (Dual options):
   - **GPSFormer**: Self-attention on GPS points only
   - **GRFormer (DualFormer)**: Dual transformer with GPS-Route interaction
     - GPS self-attention
     - Route self-attention
     - Cross-attention (Route attends to GPS)

3. **Decoder**:
   - GRU-based sequential decoder
   - Multi-task outputs:
     - Segment ID prediction (attention over candidate routes)
     - Position ratio prediction (position within segment)
   - Destination-aware decoding
   - Teacher forcing support

4. **Route Planning**:
   - Direction-Aware (DA) algorithm
   - Alternative planners: shortest, fastest
   - Uses historical traffic patterns

#### Key Features
- **Multi-task Learning**: Joint prediction of segment IDs and position ratios
- **DualFormer**: Novel dual-stream transformer for GPS + route
- **Direction-Aware Planning**: Traffic-aware route generation
- **Teacher Forcing**: Configurable teaching ratio during training
- **Positional Encoding**: Learnable position embeddings
- **Attention Mechanisms**: Custom attention for route selection

### Data Format

#### Input Data
- **GPS Trajectories**: CSV format with:
  - Moving object ID
  - Trajectory ID
  - GPS points: [lat, lng, timestamp]
  - Map-matched segments (ground truth)
  - Segment sequences with speeds and durations

- **Map Data**: OpenStreetMap format
  - `nodes.shp`: Road network nodes
  - `edges.shp`: Road network edges

#### Expected Data Structure
```
data/
  └── {city}/
      ├── roadnet/
      │   ├── nodes.shp
      │   └── edges.shp
      ├── train.pkl
      ├── valid.pkl
      └── test_output.pkl
```

### Recommended LibCity Category

**Category**: `trajectory_loc_prediction` (Map Matching / Trajectory Recovery)

**Model Type**: Should inherit from **`AbstractModel`** (NOT AbstractTrafficStateModel)

**Reasoning**:
1. This is a trajectory-level task, not a traffic state prediction task
2. Works with individual trajectory sequences, not spatiotemporal grids
3. Similar to other trajectory models in LibCity:
   - Input: Sparse GPS trajectories
   - Output: Complete map-matched trajectories (segment IDs + positions)
4. Fits the map matching and trajectory recovery paradigm

### Migration Challenges

#### High Priority Challenges

1. **Complex Data Pipeline**:
   - Requires road network preprocessing (OSM to internal format)
   - R-tree spatial indexing for candidate generation
   - Direction-Aware Matrix (DAM) pre-computation
   - Multiple preprocessing steps (map matching, route planning)

2. **Custom Road Network Representation**:
   - `RoadNetworkMapFull` class with spatial indexing
   - Segment feature extraction (18 dimensions)
   - Graph-based routing (NetworkX)
   - May need to adapt to LibCity's road network format

3. **Two-Stage Training**:
   - First train MMA (map matching model)
   - Then train TRMMA (trajectory recovery)
   - Need to handle model dependencies

4. **Route Planning Integration**:
   - DAPlanner requires pre-computed matrices
   - Traffic pattern data (vehicle counts by hour)
   - May need to provide simplified alternatives

5. **Multi-Task Loss**:
   - Two loss components (segment ID BCE + position ratio L1)
   - Weighted combination (lambda1, lambda2)
   - Need careful hyperparameter tuning

#### Medium Priority Challenges

6. **Custom Data Loaders**:
   - `TrajRecData` and `TrajRecTestData` classes
   - Complex collate functions
   - Trajectory subsampling logic
   - Need to adapt to LibCity's data loading framework

7. **City-Specific Configurations**:
   - Different time spans, UTC offsets, zone ranges per city
   - Hardcoded city parameters in training script
   - Should externalize to config files

8. **Evaluation Metrics**:
   - Custom metrics in `evaluation_utils.py`
   - LCS (Longest Common Subsequence) for route matching
   - GPS distance metrics (MAE, RMSE)
   - Need to integrate with LibCity's evaluation framework

#### Lower Priority Challenges

9. **Model Complexity**:
   - Many configurable flags (da_route_flag, srcseg_flag, gps_flag, etc.)
   - Multiple encoder/decoder variants
   - Need to determine optimal default configuration

10. **No Standard Requirements File**:
    - Dependencies only listed in README
    - May have hidden dependencies (e.g., shapefile libraries, rtree)
    - Need to identify and document all dependencies

### Model Parameters

**Key Hyperparameters**:
- `hid_dim`: Hidden dimension (default: 64-256)
- `transformer_layers`: Number of transformer layers (default: 4)
- `heads`: Number of attention heads (default: 4)
- `keep_ratio`: Sparsity level (0.1 - 0.5)
- `lambda1`: Weight for segment ID loss (default: 10)
- `lambda2`: Weight for position ratio loss (default: 5)
- `tf_ratio`: Teacher forcing ratio (default: 1.0)
- `grid_size`: Spatial grid size (default: 50m)
- `time_span`: Time interval (city-specific: 12-60s)
- `candi_size`: Number of candidate segments (default: 10)
- `search_dist`: Candidate search distance (default: 50m)

**Model Size**: ~256-512 hidden dim typical, multiple transformer layers

### Next Steps for Migration

1. **Create LibCity Model Adapter**:
   - Inherit from `AbstractModel`
   - Implement required methods: `calculate_loss`, `predict`, `forward`
   - Adapt input/output formats to LibCity standards

2. **Create Config File** (`TRMMA.json`):
   - Define all hyperparameters
   - Set sensible defaults
   - Document parameter meanings

3. **Implement Data Executor**:
   - Create custom executor for trajectory recovery task
   - Handle road network loading
   - Implement candidate generation
   - Support route planning

4. **Adapt Road Network Integration**:
   - Use LibCity's road network format if possible
   - Or provide converters for OSM data
   - Implement spatial indexing efficiently

5. **Simplify Preprocessing**:
   - Provide scripts for DAM generation
   - Document data preparation steps
   - Consider optional features vs. required features

6. **Testing Strategy**:
   - Start with MMA model (simpler map matching)
   - Then integrate TRMMA (full trajectory recovery)
   - Test on Porto dataset first (well-documented)

### Additional Notes

- The model is designed for **sparse trajectory recovery**, a specific use case
- Requires **road network data** (not just trajectory data)
- Performance depends heavily on **preprocessing quality**
- May benefit from **pre-trained MMA model** for better results
- The paper reports strong results on Porto, Xi'an, Beijing, Chengdu datasets

### Model Class Hierarchy

```
TrajRecovery (nn.Module)
├── emb_id (Embedding layer for road segments)
├── pos_embedding_gps (Optional positional encoding)
├── pos_embedding_route (Optional positional encoding)
├── fc_in_gps (GPS feature projection)
├── fc_in_route (Route feature projection)
├── encoder (GPSEncoder or GREncoder)
│   ├── transformer (GPSFormer or GRFormer)
│   └── temporal (Optional temporal embedding)
└── decoder (DecoderMulti)
    ├── rnn (GRU)
    ├── attn_route (Attention)
    └── fc_rate_out (Position ratio prediction)
```

### Related Models

- **MMA**: Map Matching with Attention (preprocessing model)
- Both models can be used independently or sequentially

---

## Migration Completion Summary

### Date: 2026-02-02

### Status: COMPLETED

### Files Created

1. **Model File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/TRMMA.py`
   - Full adaptation of TrajRecovery model
   - Includes all supporting classes (GPSFormer, GRFormer, DecoderMulti, etc.)
   - Inherits from AbstractModel
   - Implements forward(), predict(), calculate_loss() methods

2. **Config File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/TRMMA.json`
   - Default hyperparameters
   - All configuration options documented

3. **Registration**: Updated `__init__.py` to include TRMMA

### Key Adaptations Made

| Original | Adapted |
|----------|---------|
| `parameters` object | `config` dict + `data_feature` dict |
| `nn.Module` base | `AbstractModel` base |
| Custom data loaders | LibCity batch dictionary |
| External DAPlanner | Simplified (routes in batch) |
| Multiple files | Single consolidated file |

### Supported Configuration Parameters

```json
{
    "hid_dim": 128,
    "id_emb_dim": 128,
    "transformer_layers": 2,
    "heads": 4,
    "dropout": 0.1,
    "learn_pos": false,
    "da_route_flag": true,
    "srcseg_flag": true,
    "rid_feats_flag": false,
    "rate_flag": true,
    "dest_type": 1,
    "prog_flag": false,
    "pro_features_flag": true,
    "pro_input_dim": 48,
    "pro_output_dim": 8,
    "tf_ratio": 0.5,
    "lambda1": 1.0,
    "lambda2": 0.5
}
```

### Expected Batch Data Format

```python
batch = {
    'src_grid': tensor,      # (batch, src_len, 3)
    'src_len': tensor,       # (batch,)
    'trg_id': tensor,        # (batch, trg_len)
    'trg_rate': tensor,      # (batch, trg_len, 1)
    'trg_len': tensor,       # (batch,)
    'routes': tensor,        # (batch, route_len)
    'route_len': tensor,     # (batch,)
    'd_rid': tensor,         # (batch,)
    'd_rate': tensor,        # (batch, 1)
    'labels': tensor,        # (batch, trg_len-2, route_len)
    'pro_features': tensor,  # (batch,) - optional
    'src_seg': tensor,       # (batch, src_len) - optional
    'src_seg_feat': tensor,  # (batch, src_len, 1) - optional
}
```

### Simplifications

1. **DAPlanner**: Removed external route planning dependency. Routes must be pre-computed and provided in batch data.

2. **GPS2Seg Module**: Not included. Preprocessing for candidate point generation should be done separately.

3. **Road Network Features**: Simplified handling. External segment info files not required.

4. **Spatial Indexing**: Not included. R-tree based candidate search should be done during data preprocessing.

### Usage Example

```python
from libcity.model.trajectory_loc_prediction import TRMMA

config = {
    'device': 'cuda',
    'hid_dim': 128,
    'transformer_layers': 2,
    'heads': 4,
}

data_feature = {
    'id_size': 10000,
}

model = TRMMA(config, data_feature)

# Training
loss = model.calculate_loss(batch)

# Inference
predictions = model.predict(batch)
```

### Migration Iterations

#### Iteration 1: Task Categorization Fix
**Issue**: Model was placed in `trajectory_loc_prediction` but is actually a `map_matching` model.
- TRMMA expects GPS trajectory recovery data (src_grid, routes, trg_id, trg_rate)
- traj_loc_pred provides POI prediction data (history_loc, current_loc, target)
**Fix**: Moved model from trajectory_loc_prediction to map_matching category

#### Iteration 2: Executor Interface Fix
**Issue**: MapMatchingExecutor expects traditional models with run() method, but TRMMA is a neural model
- Error: AttributeError: 'TRMMA' object has no attribute 'run'
**Fix**: Created DeepMapMatchingExecutor for neural map matching models

#### Iteration 3: Configuration and Dataset Incompatibility
**Issues Identified**:
1. Missing executor config: DeepMapMatchingExecutor.json (FIXED)
2. Dataset format mismatch: MapMatchingDataset returns raw dictionaries, not tensor batches

**Current Status**: PARTIAL MIGRATION
- ✅ Model code adapted and moved to correct location
- ✅ DeepMapMatchingExecutor created
- ✅ DeepMapMatchingExecutor.json config created
- ✅ task_config.json updated
- ❌ Custom dataset needed for tensor batch generation

### Final Migration Status: INCOMPLETE

**Reason**: TRMMA requires a custom dataset class that LibCity does not currently provide.

**What Works**:
- Model architecture fully adapted to LibCity conventions
- Model registered in map_matching task category
- DeepMapMatchingExecutor created for neural map matching
- All configuration files in place

**What's Missing**:
- **DeepMapMatchingDataset**: Custom dataset class to convert trajectory data into tensor batches with TRMMA's expected format (src_grid, routes, trg_id, trg_rate, labels, etc.)
- The existing MapMatchingDataset is designed for traditional algorithms and returns raw data dictionaries

**Required for Completion**:
1. Create `Bigscity-LibCity/libcity/data/dataset/deep_map_matching_dataset.py`
2. Implement preprocessing for:
   - GPS trajectory gridding and normalization
   - Road network loading and candidate generation
   - Route planning (DA algorithm or alternatives)
   - Tensor batch creation with proper collation
3. Register in dataset/__init__.py
4. Update task_config.json to use DeepMapMatchingDataset for TRMMA

**Estimated Additional Work**:
- Dataset implementation: 300-500 lines
- Preprocessing utilities: 200-300 lines
- Testing and debugging: 2-4 hours

### Recommendations for Future Work

1. **High Priority**: Create DeepMapMatchingDataset for TRMMA, DeepMM, and DiffMM
2. Integrate DAPlanner for dynamic route generation
3. Add GPS2Seg preprocessing module
4. Add evaluation metrics specific to trajectory recovery (LCS, route accuracy)
5. Support beam search decoding
6. Create documentation for data preparation pipeline

### Files Modified/Created

**Created**:
- `/Bigscity-LibCity/libcity/model/map_matching/TRMMA.py` (1198 lines)
- `/Bigscity-LibCity/libcity/config/model/map_matching/TRMMA.json`
- `/Bigscity-LibCity/libcity/executor/deep_map_matching_executor.py` (300+ lines)
- `/Bigscity-LibCity/libcity/config/executor/DeepMapMatchingExecutor.json`

**Modified**:
- `/Bigscity-LibCity/libcity/model/map_matching/__init__.py` (added TRMMA import)
- `/Bigscity-LibCity/libcity/executor/__init__.py` (added DeepMapMatchingExecutor)
- `/Bigscity-LibCity/libcity/config/task_config.json` (TRMMA mapping updated)

**Deleted**:
- `/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/TRMMA.py` (moved to map_matching)
- `/Bigscity-LibCity/libcity/config/model/traj_loc_pred/TRMMA.json` (moved to map_matching)


