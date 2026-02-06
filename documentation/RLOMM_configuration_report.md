# RLOMM Configuration Setup Report

## Model Information
- **Model Name**: RLOMM (Reinforcement Learning for Online Map Matching)
- **Task Type**: Map Matching
- **Architecture**: Double DQN with Contrastive Learning
- **Paper**: "RLOMM: An Efficient Reinforcement Learning-Based Approach for Online Map Matching"

## Configuration Status

### 1. Task Config Registration ✓

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`

- **Task**: `map_matching`
- **Added to `allowed_model`**: Line 1111 ✓
- **Model Configuration Block**: Lines 1168-1172 ✓

```json
"map_matching": {
    "allowed_model": [
        "STMatching",
        "IVMM",
        "HMMM",
        "FMM",
        "STMatch",
        "DeepMM",
        "DiffMM",
        "TRMMA",
        "GraphMM",
        "RLOMM"  // Line 1111
    ],
    "allowed_dataset": [
        "global",
        "Seattle",
        "Neftekamsk",
        "Valky",
        "Ruzhany",
        "Santander",
        "Spaichingen",
        "NovoHamburgo"
    ],
    ...
    "RLOMM": {
        "dataset_class": "DeepMapMatchingDataset",
        "executor": "DeepMapMatchingExecutor",
        "evaluator": "MapMatchingEvaluator"
    }
}
```

### 2. Model Config File ✓

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/map_matching/RLOMM.json`

**Status**: Created and verified ✓

#### Model Architecture Parameters
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `road_emb_dim` | 128 | Paper default | Road embedding dimension |
| `traces_emb_dim` | 128 | Paper default | Trace embedding dimension |
| `num_layers` | 3 | Paper default | Number of RNN layers |
| `gin_depth` | 3 | Paper default | Graph Isomorphism Network depth |
| `gcn_depth` | 3 | Paper default | Graph Convolutional Network depth |
| `attention_dim` | 128 | Paper default | Attention mechanism dimension |

#### Reinforcement Learning Parameters
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `gamma` | 0.8 | Beijing config | RL discount factor |
| `match_interval` | 4 | Porto config | Points to match at once |
| `target_update_interval` | 10 | Model code | Target network update interval |
| `memory_capacity` | 100 | Porto config | Experience replay buffer size |
| `optimize_batch_size` | 32 | Porto config | RL optimization batch size |

#### Reward Function Parameters
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `correct_reward` | 5.0 | Beijing config | Reward for correct match |
| `mask_reward` | 0.0 | Paper default | Reward for masked positions |
| `continuous_success_reward` | 1.0 | Beijing config | Bonus for consecutive successes |
| `connectivity_reward` | 1.0 | Beijing config | Reward for road connectivity |
| `detour_penalty` | 1.0 | Beijing config | Penalty for detours |
| `lambda_ctr` | 0.1 | Paper default | Contrastive loss weight |

#### Feature Dimensions (Fixed by Dataset)
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `road_feat_dim` | 28 | Dataset | Road feature dimension (3×8 + 4) |
| `trace_feat_dim` | 4 | Dataset | Trace feature dimension |

#### Training Parameters
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `batch_size` | 512 | Beijing config | Training batch size |
| `train_batch_size` | 512 | Beijing config | Same as batch_size |
| `eval_batch_size` | 500 | Beijing config | Evaluation batch size |
| `learning_rate` | 0.001 | Paper default | Initial learning rate |
| `max_epoch` | 100 | Paper default | Maximum training epochs |
| `optimizer` | "adam" | Paper default | Optimizer type |
| `learner` | "adam" | LibCity convention | Same as optimizer |
| `lr_decay` | false | Model code | Learning rate decay flag |
| `lr_scheduler` | "none" | Model code | Learning rate scheduler |
| `clip_grad_norm` | true | Paper default | Gradient clipping flag |
| `max_grad_norm` | 5.0 | Paper default | Maximum gradient norm |

#### Evaluation & Logging
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `use_early_stop` | true | LibCity convention | Enable early stopping |
| `patience` | 20 | Paper default | Early stopping patience |
| `log_every` | 1 | LibCity convention | Log frequency (epochs) |
| `saved` | true | LibCity convention | Save model flag |
| `save_mode` | "best" | LibCity convention | Save best model only |
| `train_loss` | "none" | Model code | Loss computed in model |
| `metrics` | ["Accuracy", "RLCS"] | Paper | Evaluation metrics |

#### Dataset & System
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `cache_dataset` | true | LibCity convention | Cache preprocessed data |
| `num_workers` | 0 | LibCity convention | DataLoader workers |
| `load_best_epoch` | true | LibCity convention | Load best checkpoint |
| `debug` | false | LibCity convention | Debug mode flag |
| `hyper_tune` | false | LibCity convention | Hyperparameter tuning flag |

### 3. Model Implementation ✓

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/RLOMM.py`

**Status**: Implemented and verified ✓

**Key Components**:
1. **RoadGIN**: Graph Isomorphism Network for road graph encoding
2. **TraceGCN**: Bidirectional GCN for GPS trace graph encoding
3. **QNetwork**: Deep Q-Network with attention mechanism
4. **MMAgent**: Double DQN agent with experience replay
5. **Memory**: Experience replay buffer for RL training

**LibCity Integration**:
- Inherits from `AbstractModel` ✓
- Implements `predict()` method ✓
- Implements `calculate_loss()` method ✓
- Batch format compatible with LibCity ✓

### 4. Model Registration ✓

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/__init__.py`

```python
from libcity.model.map_matching.RLOMM import RLOMM

__all__ = [
    "STMatching",
    "IVMM",
    "HMMM",
    "FMM",
    "GraphMM",
    "DeepMM",
    "DiffMM",
    "RLOMM"  # ✓ Registered
]
```

## Dataset Compatibility

### Compatible Datasets

RLOMM uses `DeepMapMatchingDataset` which is compatible with the following map matching datasets:

| Dataset | Location | Type | Status |
|---------|----------|------|--------|
| **Seattle** | USA | Urban road network | ✓ Available |
| **Neftekamsk** | Russia | Small city | ✓ Available |
| **Valky** | Ukraine | Rural/small town | ✓ Available |
| **Ruzhany** | Belarus | Small town | ✓ Available |
| **Santander** | Spain | Coastal city | ✓ Available |
| **Spaichingen** | Germany | Small city | ✓ Available |
| **NovoHamburgo** | Brazil | Urban area | ✓ Available |

### Dataset Requirements

RLOMM requires the following data features (provided by `DeepMapMatchingDataset`):

#### Road Graph Data
- `num_roads`: Number of road segments in network
- `road_x`: Road node features [num_roads, 28]
- `road_adj`: Road adjacency matrix (SparseTensor)
- `connectivity_distances`: Precomputed road connectivity

#### Trace Graph Data
- `num_grids`: Number of grid cells for spatial discretization
- `trace_in_edge_index`: Incoming edges in trace graph
- `trace_out_edge_index`: Outgoing edges in trace graph
- `trace_weight`: Edge weights for trace graph
- `map_matrix`: Grid-to-road mapping matrix
- `singleton_grid_mask`: Mask for singleton grids
- `singleton_grid_location`: Location features for singleton grids

#### Trajectory Data (per batch)
- `traces`: GPS trace grid IDs + timestamps [batch, seq_len, 2]
- `tgt_roads`: Ground truth road indices [batch, seq_len]
- `candidates_id`: Candidate road IDs [batch, seq_len, num_candidates]
- `trace_lens`: Sequence lengths [batch]

### Dataset Configuration

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/data/DeepMapMatchingDataset.json`

```json
{
  "delta_time": true,
  "grid_size": 50,
  "train_rate": 0.7,
  "eval_rate": 0.1,
  "batch_size": 32,
  "eval_batch_size": 64,
  "downsample_rate": 0.5,
  "max_road_len": 25,
  "min_road_len": 15,
  "layer": 2,
  "gamma": 1.0,
  "num_workers": 0,
  "shuffle": true,
  "cache_dataset": true
}
```

## Executor & Evaluator

### Executor

**Class**: `DeepMapMatchingExecutor`

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/executor/deep_map_matching_executor.py`

**Responsibilities**:
- Training loop management
- Batch preparation and device transfer
- Loss computation via `model.calculate_loss()`
- Gradient updates and optimization
- Model checkpointing
- Validation and evaluation

**Special Features for RLOMM**:
- Supports RL-specific training (experience replay, target network updates)
- Compatible with models that compute their own loss
- Handles variable-length trajectories

### Evaluator

**Class**: `MapMatchingEvaluator`

**Metrics**:
1. **Accuracy**: Percentage of correctly matched road segments
2. **RLCS**: Route Length-based Continuous Similarity

**Evaluation Mode**: Segment-based matching evaluation

## Training Process

### RLOMM Training Flow

1. **Initialization**:
   - Load road graph and trace graph data
   - Initialize main network and target network
   - Create experience replay buffer

2. **Training Loop** (per epoch):
   - Process trajectories in intervals of `match_interval` points
   - Forward pass through main network
   - Select actions (road segments) using Q-values
   - Compute rewards based on:
     - Correctness of match
     - Road connectivity
     - Continuous success streaks
     - Detour detection
   - Store experiences in replay buffer
   - Sample mini-batch and compute loss:
     - RL loss: Smooth L1 loss on TD error (Double DQN)
     - Contrastive loss: Align trace and road embeddings
   - Update main network
   - Periodically update target network

3. **Evaluation**:
   - Run greedy policy (no exploration)
   - Compute Accuracy and RLCS metrics

### Key Differences from Standard Models

| Aspect | Standard Models | RLOMM |
|--------|----------------|-------|
| Loss Function | Cross-entropy | RL loss + Contrastive loss |
| Training | Supervised | Reinforcement learning |
| Network | Single network | Main + Target networks |
| Memory | Batch-based | Experience replay buffer |
| Optimization | Per-batch | Double DQN updates |

## Configuration Notes

### Important Considerations

1. **PyTorch Geometric Dependency**:
   - RLOMM requires `torch_geometric` for GIN and GCN layers
   - Fallback to simple MLP if not available (reduced performance)

2. **Memory Requirements**:
   - Large road graphs (>10K segments) may require GPU
   - Experience replay buffer size: `memory_capacity` × episode data

3. **Hyperparameter Sensitivity**:
   - `gamma`: Controls long-term vs. short-term rewards
   - `lambda_ctr`: Balance between RL and contrastive learning
   - Reward values: Affect convergence speed and stability

4. **Dataset Preprocessing**:
   - Grid-based spatial discretization (50m cells)
   - Road graph construction (adjacency, features)
   - Trace graph construction (temporal, spatial edges)

5. **Batch Size Considerations**:
   - `batch_size`: Number of trajectories per batch
   - `optimize_batch_size`: RL optimization batch from replay buffer
   - Large `batch_size` speeds up training but needs more memory

## Testing Commands

### Run RLOMM on Seattle Dataset

```bash
# Basic training
python run_model.py --task map_matching --model RLOMM --dataset Seattle

# Custom configuration
python run_model.py --task map_matching --model RLOMM --dataset Seattle \
    --batch_size 256 --learning_rate 0.0005 --max_epoch 50

# With different reward settings
python run_model.py --task map_matching --model RLOMM --dataset Seattle \
    --correct_reward 10.0 --lambda_ctr 0.2
```

### Run on Multiple Datasets

```bash
# Seattle (urban, large)
python run_model.py --task map_matching --model RLOMM --dataset Seattle

# Santander (medium city)
python run_model.py --task map_matching --model RLOMM --dataset Santander

# Neftekamsk (small city)
python run_model.py --task map_matching --model RLOMM --dataset Neftekamsk
```

## Configuration Files Summary

### Files Created/Modified

1. **Model Config**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/map_matching/RLOMM.json` ✓
2. **Task Config**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json` (verified) ✓
3. **Model Implementation**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/RLOMM.py` (exists) ✓
4. **Model Registration**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/__init__.py` (verified) ✓

### Dataset Compatibility Matrix

| Dataset | Road Network | Trajectories | Graph Data | Status |
|---------|--------------|--------------|------------|--------|
| Seattle | ✓ | ✓ | Auto-generated | ✓ Ready |
| Neftekamsk | ✓ | ✓ | Auto-generated | ✓ Ready |
| Valky | ✓ | ✓ | Auto-generated | ✓ Ready |
| Ruzhany | ✓ | ✓ | Auto-generated | ✓ Ready |
| Santander | ✓ | ✓ | Auto-generated | ✓ Ready |
| Spaichingen | ✓ | ✓ | Auto-generated | ✓ Ready |
| NovoHamburgo | ✓ | ✓ | Auto-generated | ✓ Ready |

Note: Graph data (road adjacency, trace graphs) is automatically constructed by `DeepMapMatchingDataset` from raw trajectory and road network data.

## Troubleshooting

### Common Issues

1. **PyTorch Geometric not found**:
   - Install: `pip install torch-geometric torch-sparse`
   - Model will use fallback MLP layers (reduced performance)

2. **CUDA out of memory**:
   - Reduce `batch_size` (try 128 or 256)
   - Reduce `memory_capacity` (try 50)
   - Use CPU: set `device: "cpu"` in config

3. **Loss not decreasing**:
   - Check reward balance (correct_reward vs. penalties)
   - Adjust `lambda_ctr` (contrastive loss weight)
   - Verify graph data is loaded correctly

4. **Slow training**:
   - Increase `match_interval` (process more points at once)
   - Reduce `target_update_interval`
   - Use GPU acceleration

## Conclusion

RLOMM configuration is **COMPLETE** and ready for testing:

✓ Registered in task_config.json
✓ Model config file created with all hyperparameters
✓ Model implementation exists and follows LibCity conventions
✓ Model registered in __init__.py
✓ Compatible with 7 map matching datasets
✓ Uses appropriate executor (DeepMapMatchingExecutor)
✓ Uses appropriate evaluator (MapMatchingEvaluator)
✓ All dependencies documented

**Next Steps**:
1. Install PyTorch Geometric (recommended)
2. Run initial test on Seattle dataset
3. Fine-tune hyperparameters based on results
4. Test on other datasets for generalization

---

**Report Generated**: 2026-02-04
**LibCity Path**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity`
