# Config Migration: PLMTrajRec

## Model Overview
PLMTrajRec (Pre-trained Language Model based Trajectory Recovery) is a trajectory location prediction model that uses a pre-trained BERT model with LoRA fine-tuning for sparse trajectory recovery.

## Task Registration

### task_config.json
- **Status**: Already registered
- **Task type**: `traj_loc_pred`
- **Location**: Line 22 in allowed_model list
- **Configuration**:
  ```json
  "PLMTrajRec": {
      "dataset_class": "TrajectoryDataset",
      "executor": "TrajLocPredExecutor",
      "evaluator": "TrajLocPredEvaluator",
      "traj_encoder": "PLMTrajRecEncoder"
  }
  ```

## Model Configuration

### File Location
`/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/PLMTrajRec.json`

### Hyperparameters

#### Model Architecture Parameters
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `hidden_dim` | 512 | main.py line 27 | Hidden dimension for embeddings |
| `conv_kernel` | 9 | main.py line 26 | Kernel size for trajectory convolution |
| `soft_traj_num` | 128 | main.py line 35 | Number of soft trajectory prompts |
| `road_candi` | true | main.py line 34 | Whether to use road candidate information |
| `dropout` | 0.3 | PLMTrajRec.py line 534 | Dropout rate |

#### BERT Configuration
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `bert_model_path` | "bert-base-uncased" | PLMTrajRec.py line 540 | Path to BERT model |
| `use_lora` | true | PLMTrajRec.py line 541 | Whether to use LoRA fine-tuning |
| `lora_r` | 8 | layer.py line 158 | LoRA rank |
| `lora_alpha` | 32 | layer.py line 159 | LoRA alpha parameter |
| `lora_dropout` | 0.01 | layer.py line 161 | LoRA dropout rate |

#### Loss Configuration
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `lambda1` | 10 | main.py line 29 | Weight for rate prediction loss |

#### Prompt Configuration
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `default_keep_ratio` | 0.125 | PLMTrajRec.py line 547 | Default GPS keep ratio for prompt |

#### Data Dimensions
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `id_size` | 2505 | main.py line 52 (Chengdu) | Number of road segment IDs |
| `grid_size` | 64 | Config default | Grid size for spatial encoding |
| `time_slots` | 24 | Config default | Number of time slots |

#### Training Parameters
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `batch_size` | 64 | main.py line 25 | Batch size |
| `learning_rate` | 0.0001 | main.py line 28 | Learning rate (1e-4) |
| `max_epoch` | 50 | Config default | Maximum training epochs |
| `optimizer` | "adamw" | Config default | Optimizer type |
| `clip` | 1.0 | Config default | Gradient clipping |
| `lr_step` | 10 | Config default | Learning rate scheduler step |
| `lr_decay` | 0.5 | Config default | Learning rate decay factor |
| `lr_scheduler` | "steplr" | Config default | Learning rate scheduler |

## Model Registration

### File Location
`/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py`

**Status**: Already imported (line 24)

## Required Data Features

The model requires the following data features from the dataset:

### Required Features
- `src_lat`: Source latitudes [batch, seq_len]
- `src_lng`: Source longitudes [batch, seq_len]
- `mask_index`: Mask for unknown positions [batch, seq_len]
- `padd_index`: Padding mask [batch, seq_len]
- `traj_length`: Trajectory lengths [batch]
- `trg_id`: Target road segment IDs [batch, seq_len] (for training)
- `trg_rate`: Target movement rates [batch, seq_len] (for training)

### Optional Features
- `src_candi_id`: Road candidate IDs [batch, seq_len, num_candidates]
- `prompt_token`: Pre-computed prompt tokens [batch, prompt_len, hidden_dim]
- `road_condition`: Road condition matrix [T, N, N]
- `road_condition_xyt_index`: XYT indices [batch, seq_len, 3]
- `forward_delta_t`: Time delta to forward neighbor [batch, seq_len]
- `backward_delta_t`: Time delta to backward neighbor [batch, seq_len]
- `forward_index`: Forward neighbor index [batch, seq_len]
- `backward_index`: Backward neighbor index [batch, seq_len]
- `keep_ratio`: GPS keep ratio for prompt generation

### Data Feature Dict
- `id_size`: Number of road segment IDs (from dataset)
- `grid_size`: Grid size for spatial encoding (optional)
- `time_slots`: Number of time slots (optional)

## Dataset Compatibility

### Compatible Datasets
PLMTrajRec works with LibCity's `TrajectoryDataset` class and requires:
1. GPS coordinates (lat/lng)
2. Road segment IDs
3. Trajectory timestamps

### Recommended Datasets
- Custom trajectory datasets with road network information
- Porto dataset (id_size=2225)
- Chengdu dataset (id_size=2505)

### Data Preprocessing Requirements
- Road network map matching for `src_candi_id`
- Road condition flow matrix (optional but recommended)
- Temporal interpolation indices for forward/backward neighbors

## Model Components

### Key Features
1. **Learnable Fourier Positional Encoding**: For GPS coordinates
2. **Soft Trajectory Prompts**: Learnable task-specific prompts
3. **ReprogrammingLayer**: Cross-attention for trajectory transformation
4. **BERT Backbone**: Pre-trained language model with LoRA fine-tuning
5. **Dual-head Decoder**: Separate heads for road ID and movement rate prediction
6. **Spatial-Temporal Convolution**: For road condition processing

### Loss Function
Combined loss with two components:
- Road ID prediction: NLL loss
- Movement rate prediction: MSE loss (weighted by lambda1)

Total loss = ID_loss + lambda1 * rate_loss

## Dependencies

### Required Libraries
- `torch`: PyTorch framework
- `transformers`: For BERT model (optional, has fallback)
- `peft`: For LoRA fine-tuning (optional, has fallback)

### Fallback Mechanism
If BERT/LoRA libraries are not available, the model uses:
- Fallback transformer encoder (6 layers, 8 heads, 512 hidden_dim)
- Learnable token embeddings for MASK and PAD

## Notes and Considerations

### BERT Model Path
- Default: "bert-base-uncased"
- Can be changed via config parameter
- Model will attempt to download from HuggingFace if not found locally
- Fallback encoder is used if download fails

### Road Candidate Feature
- Setting `road_candi: true` requires `src_candi_id` in batch data
- This feature significantly improves performance on road networks
- If unavailable, set to `false` (uses GPS only)

### Memory Considerations
- BERT model requires significant GPU memory
- LoRA reduces memory footprint for fine-tuning
- Fallback encoder is more memory-efficient

### Performance Tips
1. Use road candidate information when available
2. Enable LoRA for efficient fine-tuning
3. Adjust `soft_traj_num` based on dataset complexity
4. Tune `lambda1` to balance ID and rate prediction

## Migration Status

- [x] Model registered in task_config.json
- [x] Model config file created
- [x] Model imported in __init__.py
- [x] All hyperparameters documented
- [x] Data requirements specified
- [x] Dependencies listed

## Testing Recommendations

### Basic Test
```python
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.model import get_model

# Load config
config = ConfigParser(task='traj_loc_pred', model='PLMTrajRec',
                     dataset='your_dataset')

# Get dataset
dataset = get_dataset(config)

# Get model
model = get_model(config, dataset.get_data_feature())

# Run forward pass
batch = next(iter(dataset.get_data_loader('train')))
output = model.predict(batch)
```

### Hyperparameter Tuning
Recommended parameters to tune:
1. `learning_rate`: [1e-5, 5e-5, 1e-4, 5e-4]
2. `lambda1`: [5, 10, 20, 50]
3. `soft_traj_num`: [64, 128, 256]
4. `dropout`: [0.1, 0.2, 0.3, 0.4]
5. `conv_kernel`: [5, 7, 9, 11]

## References

### Original Repository
Path: `/home/wangwenrui/shk/AgentCity/repos/PLMTrajRec`

### Key Files
- `main.py`: Training script with default hyperparameters
- `finetune_main.py`: Fine-tuning script
- `model/model.py`: Original PTR model implementation
- `model/layer.py`: BERT wrapper and LoRA configuration
- `model/loss_fn.py`: Custom loss functions

### LibCity Implementation
- Model: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/PLMTrajRec.py`
- Config: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/PLMTrajRec.json`

## Date
2026-02-02
