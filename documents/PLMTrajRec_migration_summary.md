# PLMTrajRec Migration Summary

## Overview
This document describes the migration of the PLMTrajRec (Pre-trained Language Model based Trajectory Recovery) model from its original implementation to the LibCity framework.

## Source Files
- **Original Repository**: `/home/wangwenrui/shk/AgentCity/repos/PLMTrajRec/`
- **Key Source Files**:
  - `repos/PLMTrajRec/model/model.py` - Main PTR model, Decoder, ReprogrammingLayer
  - `repos/PLMTrajRec/model/layer.py` - LearnableFourierPositionalEncoding, TemporalPositionalEncoding, BERT wrapper
  - `repos/PLMTrajRec/model/loss_fn.py` - Custom accuracy metrics (cal_id_acc, check_dis_loss)

## Output Files
- **Model File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/PLMTrajRec.py`
- **Config File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/PLMTrajRec.json`

## Key Components Migrated

### 1. LearnableFourierPositionalEncoding
- Purpose: Encodes GPS coordinates (latitude, longitude) using learnable Fourier features
- Reference: https://arxiv.org/pdf/2106.02795.pdf
- Used for encoding observed GPS points in the trajectory

### 2. TemporalPositionalEncoding
- Standard sinusoidal positional encoding for sequence positions
- Used to provide temporal position information to the BERT encoder

### 3. ReprogrammingLayer
- Multi-head cross-attention layer for trajectory reprogramming
- Uses soft trajectory prompts to transform trajectory embeddings
- Enables the model to adapt pre-trained language model to trajectory domain

### 4. Decoder
- Dual-head decoder for:
  - Road segment ID prediction (classification)
  - Movement rate prediction (regression)
- Outputs are masked based on trajectory length

### 5. TrajConv
- 1D convolution layer for trajectory feature extraction
- Applied after combining GPS and road candidate embeddings

### 6. SpatialTemporalConv
- 2D spatial convolution followed by 1D temporal convolution
- Processes road condition matrices for mask prompt generation

### 7. BERTWrapper
- Wrapper for HuggingFace BERT model with optional LoRA fine-tuning
- Includes fallback transformer encoder if BERT is not available
- Supports configurable BERT model path

### 8. SimpleBERTTokenEmbedding
- Learnable token embeddings for MASK and PAD tokens
- Used when BERT is not available or for custom token initialization

## Adaptations for LibCity

### Base Class
- Inherits from `AbstractModel` as required for trajectory location prediction tasks

### Required Methods Implemented
1. `__init__(config, data_feature)`: Initializes all model components with configuration
2. `forward(batch)`: Processes batch and returns road ID and rate predictions
3. `predict(batch)`: Returns predictions for evaluation
4. `calculate_loss(batch)`: Computes combined ID + rate loss

### Batch Input Format
The model expects a LibCity batch dictionary with the following keys:

**Required keys:**
- `src_lat`: Source latitudes [batch, seq_len]
- `src_lng`: Source longitudes [batch, seq_len]
- `mask_index`: Mask for unknown positions [batch, seq_len]
- `padd_index`: Padding mask [batch, seq_len]
- `traj_length`: Trajectory lengths [batch]

**Optional keys:**
- `src_candi_id`: Road candidate IDs [batch, seq_len, num_candidates]
- `prompt_token`: Pre-computed prompt tokens [batch, prompt_len, hidden_dim]
- `road_condition`: Road condition matrix [T, N, N]
- `road_condition_xyt_index`: XYT indices [batch, seq_len, 3]
- `forward_delta_t`: Time delta to forward neighbor [batch, seq_len]
- `backward_delta_t`: Time delta to backward neighbor [batch, seq_len]
- `forward_index`: Forward neighbor index [batch, seq_len]
- `backward_index`: Backward neighbor index [batch, seq_len]

**Target keys (for training):**
- `trg_id`: Target road segment IDs [batch, seq_len]
- `trg_rate`: Target movement rates [batch, seq_len]

### Output Format
- `outputs_id`: Road ID log probabilities [seq_len, batch, id_size + 1]
- `outputs_rate`: Movement rate predictions [seq_len, batch, 1]

## Configuration Parameters

### Model Hyperparameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_dim | 512 | Hidden dimension for embeddings |
| conv_kernel | 9 | Kernel size for trajectory convolution |
| soft_traj_num | 128 | Number of soft trajectory prompts |
| road_candi | true | Whether to use road candidate information |
| dropout | 0.3 | Dropout rate |
| lambda1 | 10 | Weight for rate prediction loss |

### BERT Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| bert_model_path | bert-base-uncased | Path to BERT model |
| use_lora | true | Whether to use LoRA fine-tuning |
| lora_r | 8 | LoRA rank |
| lora_alpha | 32 | LoRA alpha parameter |
| lora_dropout | 0.01 | LoRA dropout rate |

### Prompt Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| default_keep_ratio | 0.125 | Default GPS keep ratio for prompt |

### Data Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| id_size | 2505 | Number of road segment IDs |
| grid_size | 64 | Grid size for spatial encoding |
| time_slots | 24 | Number of time slots |

### Training Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| batch_size | 64 | Batch size |
| learning_rate | 0.0001 | Learning rate |
| max_epoch | 50 | Maximum training epochs |
| optimizer | adamw | Optimizer type |
| clip | 1.0 | Gradient clipping value |

## Dependencies
- PyTorch
- NumPy
- transformers (optional, for BERT)
- peft (optional, for LoRA)

## Special Considerations

### BERT Fallback
If the `transformers` or `peft` libraries are not available, the model uses a fallback transformer encoder. This ensures the model can run without HuggingFace dependencies.

### Road Candidate Information
The model can optionally use road candidate information for each GPS point. This is controlled by the `road_candi` parameter. When enabled, the model combines GPS embeddings with road candidate embeddings.

### Sparse Trajectory Recovery Task
This model is designed for sparse trajectory recovery, where the goal is to predict:
1. Road segment IDs for each point in the trajectory
2. Movement rates (position along the road segment)

This is different from standard next location prediction tasks in LibCity.

### Loss Function
The total loss combines:
- NLL loss for road ID prediction
- MSE loss for rate prediction (weighted by lambda1)

```
total_loss = id_loss + lambda1 * rate_loss
```

## Changes from Original Implementation

1. **Unified Model Class**: Combined PTR, layer components, and BERT wrapper into a single PLMTrajRec class
2. **Configurable BERT Path**: Made BERT model path configurable instead of hardcoded
3. **Fallback Encoder**: Added fallback transformer encoder when BERT is unavailable
4. **LibCity Batch Format**: Adapted input handling for LibCity batch dictionary format
5. **Token Embeddings**: Replaced direct BERT token lookups with learnable embeddings
6. **Device Handling**: Added proper device management through config

## Usage Example

```python
from libcity.model.trajectory_loc_prediction import PLMTrajRec

config = {
    'device': 'cuda',
    'hidden_dim': 512,
    'conv_kernel': 9,
    'soft_traj_num': 128,
    'road_candi': True,
    'dropout': 0.3,
    'lambda1': 10,
    'bert_model_path': 'bert-base-uncased',
    'use_lora': True,
}

data_feature = {
    'id_size': 2505,
}

model = PLMTrajRec(config, data_feature)
```
