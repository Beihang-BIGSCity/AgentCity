# Config Migration: GraphMM

## Migration Summary
Successfully configured GraphMM (Graph-based Map Matching) model in LibCity's task configuration system.

## 1. task_config.json Registration

### Location
- **File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
- **Task Type**: `traj_loc_pred`
- **Line Number**: 35 (in allowed_model list)

### Changes Made
Added "GraphMM" to the `allowed_model` list in the `traj_loc_pred` task section:
```json
"allowed_model": [
    ...
    "JGRM",
    "TrajSDE",
    "RNTrajRec",
    "GraphMM"
]
```

### Model Configuration Section
Added GraphMM configuration block (lines 224-229):
```json
"GraphMM": {
    "dataset_class": "TrajectoryDataset",
    "executor": "TrajLocPredExecutor",
    "evaluator": "TrajLocPredEvaluator",
    "traj_encoder": "StandardTrajectoryEncoder"
}
```

## 2. Model Config File

### Location
- **File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/GraphMM.json`
- **Status**: Already exists with complete configuration

### Hyperparameters

#### Core Model Parameters
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `model_name` | "GraphMM" | - | Model identifier |
| `emb_dim` | 256 | Original paper | Embedding dimension for all components |
| `layer` | 4 | Original paper | K-hop neighbors for adjacency polynomial (A^k) |
| `tf_ratio` | 0.5 | Original paper | Teacher forcing ratio during training |
| `drop_prob` | 0.5 | Original paper | Dropout probability |
| `gamma` | 10000 | Original paper | Penalty for unreachable roads in CRF |
| `topn` | 5 | Original paper | Top-N candidates for CRF Viterbi decoding |
| `neg_nums` | 800 | Original paper | Number of negative samples for CRF training |

#### Model Architecture Flags
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `use_crf` | true | Original paper | Whether to use CRF layer for structured prediction |
| `bi` | true | Original paper | Whether to use bidirectional GRU encoder |
| `atten_flag` | true | Original paper | Whether to use attention mechanism in Seq2Seq |

#### Feature Dimensions
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `road_feat_dim` | 28 | Original paper | Road segment feature dimension |
| `trace_feat_dim` | 4 | Original paper | Trajectory trace feature dimension |
| `gps_feat_dim` | 2 | Original paper | GPS coordinate dimension |

#### Graph Architecture Parameters
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `gin_depth` | 3 | Original paper | Number of GIN layers in RoadGIN encoder |
| `gin_mlp_layers` | 2 | Original paper | MLP layers per GIN layer |
| `digcn_depth` | 2 | Original paper | Number of DiGCN layers in TraceGCN encoder |

#### Training Parameters
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `optimizer` | "AdamW" | Original paper | Optimizer type |
| `learning_rate` | 0.0001 | Original paper | Learning rate |
| `weight_decay` | 1e-8 | Original paper | Weight decay for regularization |
| `batch_size` | 32 | Original paper | Batch size (for CRF mode) |
| `epochs` | 200 | Original paper | Maximum training epochs |
| `grad_clip_norm` | 5.0 | Original paper | Gradient clipping norm |
| `output_dim` | 1 | LibCity convention | Output dimension |

## 3. Model File Location

### Implementation
- **File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/GraphMM.py`
- **Status**: Already implemented
- **Size**: 1311 lines

### Module Registration
- **File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py`
- **Status**: Already registered (line 28, line 58)

## 4. Dataset Compatibility

### Required Data Features

GraphMM requires specialized graph-based data structures that must be provided in the `data_feature` dictionary:

#### Road Network Graph
- `num_roads` or `loc_size`: Number of road segments
- `road_x` or `road_features`: Road segment features (shape: [num_roads, road_feat_dim])
- `road_adj` or `road_edge_index`: Road graph adjacency (SparseTensor or edge_index format)
- `A` or `adjacency_matrix`: Base adjacency matrix for computing A^k polynomial

#### Trajectory Trace Graph
- `num_grids`: Number of grid cells in the trace graph
- `trace_in_edge_index`: Incoming edges in trace graph (shape: [2, num_edges])
- `trace_out_edge_index`: Outgoing edges in trace graph (shape: [2, num_edges])
- `trace_weight`: Edge weights for trace graph (shape: [num_edges])

#### Grid-Road Mapping
- `map_matrix` or `grid_road_map`: Mapping matrix from grids to roads (shape: [num_grids, num_roads])
- `singleton_grid_mask`: Boolean mask for singleton grids (optional)
- `singleton_grid_location`: Location features for singleton grids (optional)

#### Batch Data Requirements
Each training/inference batch must contain:
- `grid_traces` or `X`: Grid cell IDs of trajectory points (shape: [batch_size, seq_len])
- `tgt_roads` or `target` or `y`: Ground truth road segments (shape: [batch_size, road_len])
- `traces_gps` or `gps`: GPS coordinates (shape: [batch_size, seq_len, 2])
- `sample_Idx` or `sample_idx`: Sample indices (shape: [batch_size, seq_len])
- `traces_lens` or `trace_lens`: List of actual trajectory lengths
- `road_lens`: List of actual road sequence lengths

### Preprocessing Requirements

1. **Road Network Graph Construction**
   - Build PyG-compatible edge indices from road network
   - Extract or compute road segment features (e.g., length, direction, type, etc.)
   - Compute adjacency matrix powers A^k for k=layer (default: 4)

2. **Trajectory Trace Graph Construction**
   - Create grid discretization of geographic space
   - Build directed trace graph with incoming/outgoing edges
   - Compute edge weights based on trajectory transition frequencies

3. **Grid-Road Mapping**
   - Map each grid cell to candidate road segments
   - Create sparse mapping matrix
   - Identify and handle singleton grids (grids not mapped to roads)

4. **Data Encoding**
   - Convert GPS trajectories to grid cell sequences
   - Align road segment ground truth with grid sequences
   - Compute sample indices for sequence ordering

### Dataset Compatibility Notes

- **Standard LibCity Datasets**: May require significant preprocessing
  - Most trajectory datasets (foursquare_tky, gowalla, etc.) are POI-based, not road-based
  - GraphMM is designed for GPS trajectory map matching on road networks

- **Recommended Datasets**:
  - Custom map matching datasets with road network graphs
  - Datasets with explicit road segment labels (e.g., from original GraphMM paper)
  - GPS trajectory datasets with road network metadata

- **Data Preparation**:
  - A custom dataset class may be needed to handle graph preprocessing
  - Consider creating a `GraphMMDataset` class extending `TrajectoryDataset`
  - Preprocessing scripts required for:
    - Road network graph extraction from OSM or similar sources
    - Grid discretization and trace graph construction
    - Grid-to-road mapping generation

## 5. Model Architecture

### Component Overview

1. **RoadGIN** (Road Graph Encoder)
   - Graph Isomorphism Network with `gin_depth` layers
   - Each layer uses MLP with `gin_mlp_layers` layers
   - Max-pooling over layer outputs for multi-hop information
   - Input: Road features + Road adjacency
   - Output: Road embeddings (num_roads, emb_dim)

2. **TraceGCN** (Trace Graph Encoder)
   - Bidirectional DiGCN with `digcn_depth` layers
   - Separate encoders for incoming and outgoing edges
   - Concatenates bidirectional representations
   - Input: Grid features + Trace graph edges
   - Output: Grid embeddings (num_grids, 2*emb_dim)

3. **Seq2Seq** (Sequence Decoder)
   - Bidirectional GRU encoder
   - Unidirectional GRU decoder with optional attention
   - Teacher forcing during training (controlled by `tf_ratio`)
   - Input: Trajectory sequence with GPS and sample indices
   - Output: Emission probabilities for road segments

4. **CRF** (Conditional Random Field)
   - Structured prediction layer (optional, controlled by `use_crf`)
   - Uses road embeddings to compute transition scores
   - Negative sampling for efficient training (`neg_nums` samples)
   - Viterbi decoding with top-k candidates (`topn`)
   - Penalty for unreachable roads (`gamma`)

### Dependencies

- **PyTorch Geometric**: Required for GINConv, GCNConv, MLP layers
- **torch-sparse**: Required for SparseTensor (optional but recommended)
- **PyTorch**: Standard neural network modules

## 6. Comparison with Similar Models

### vs. DeepMM
- **DeepMM**: Deep learning-based map matching in LibCity's `map_matching` task
- **GraphMM**: Graph-based map matching in `traj_loc_pred` task
- **Difference**: GraphMM uses dual-graph encoders (road + trace) with CRF, while DeepMM may use different architecture

### Task Assignment Rationale
- GraphMM is placed in `traj_loc_pred` task (not `map_matching` task)
- Reason: Model outputs road segment sequences (similar to trajectory location prediction)
- The model file is located in `trajectory_loc_prediction/` directory
- Uses standard trajectory prediction executor and evaluator

## 7. Testing Recommendations

### Before Testing
1. Verify dependencies are installed:
   ```bash
   pip install torch-geometric torch-sparse
   ```

2. Prepare specialized dataset with:
   - Road network graph data
   - Trace graph data
   - Grid-road mapping
   - Preprocessed trajectories

### Test Configuration
Create a test config file (e.g., `graphmm_test_config.json`):
```json
{
    "task": "traj_loc_pred",
    "model": "GraphMM",
    "dataset": "your_custom_dataset",
    "emb_dim": 256,
    "layer": 4,
    "use_crf": true,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "epochs": 200
}
```

### Run Test
```bash
python run_model.py --task traj_loc_pred --model GraphMM --dataset your_dataset --config_file graphmm_test_config.json
```

## 8. Migration Status

- [x] Model file created: GraphMM.py
- [x] Model config created: GraphMM.json
- [x] Added to task_config.json allowed_model list
- [x] Added model configuration block in task_config.json
- [x] Model registered in __init__.py
- [x] Configuration parameters documented
- [x] Dataset requirements documented

## 9. Next Steps

1. **Dataset Preparation**
   - Create or adapt a dataset with road network graph data
   - Implement preprocessing pipeline for graph construction
   - Consider creating a custom `GraphMMDataset` class

2. **Testing**
   - Test with synthetic/minimal data first
   - Validate graph data loading
   - Check model forward pass and loss computation
   - Verify CRF decoding works correctly

3. **Optimization**
   - Tune hyperparameters for specific datasets
   - Adjust `layer`, `gamma`, `neg_nums`, `topn` based on road network size
   - Experiment with `tf_ratio` and `drop_prob`

4. **Documentation**
   - Add dataset preparation guide
   - Create example preprocessing scripts
   - Document expected input/output formats

## 10. Important Notes

### Unique Characteristics
- **Dual-Graph Architecture**: GraphMM is unique in using both road network graph and trajectory trace graph
- **CRF Integration**: Optional CRF layer for structured prediction with negative sampling
- **Graph Requirements**: More complex data requirements than standard trajectory models

### Potential Issues
1. **Data Availability**: Most LibCity datasets may not have required graph structures
2. **Preprocessing Complexity**: Significant preprocessing needed for road networks and trace graphs
3. **Memory Requirements**: Graph data structures can be memory-intensive for large road networks
4. **Dependency Management**: Requires PyTorch Geometric and torch-sparse

### Compatibility
- **Framework**: Fully integrated with LibCity conventions
- **Executor**: Uses standard TrajLocPredExecutor
- **Evaluator**: Uses standard TrajLocPredEvaluator
- **Encoder**: Uses StandardTrajectoryEncoder (though model has custom graph encoders)

## References

### Original Paper
"GraphMM: Graph-based Vehicular Map Matching by Leveraging Trajectory and Road Correlations"

### Code Location
- Model: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/GraphMM.py`
- Config: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/GraphMM.json`
- Task Config: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`

## Summary

GraphMM has been successfully configured in LibCity's task configuration system. The model is registered in the `traj_loc_pred` task with complete hyperparameter configuration. However, due to its specialized graph-based architecture and unique data requirements, careful dataset preparation and preprocessing will be essential for successful deployment. The model's dual-graph encoder design (RoadGIN + TraceGCN) combined with CRF-based decoding makes it particularly suitable for map matching tasks with rich road network information.
