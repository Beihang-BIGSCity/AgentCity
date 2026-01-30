# PLMTrajRec Migration Documentation

## Overview

This document describes the adaptation of the PLMTrajRec (Pre-trained Language Model for Trajectory Recovery) model to the LibCity framework.

## Original Model Information

- **Repository**: `/home/wangwenrui/shk/AgentCity/repos/PLMTrajRec`
- **Main Model File**: `repos/PLMTrajRec/model/model.py` (PTR class at line 164)
- **Supporting Files**:
  - `repos/PLMTrajRec/model/layer.py` - Helper classes (LearnableFourierPositionalEncoding, TemporalPositionalEncoding, BERT)
  - `repos/PLMTrajRec/model/loss_fn.py` - Loss functions and evaluation metrics
  - `repos/PLMTrajRec/trainer.py` - Training logic
  - `repos/PLMTrajRec/load_data/datasets.py` - Dataset and data processing

## Adapted Files

### Model File
**Path**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/PLMTrajRec.py`

### Configuration File
**Path**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/PLMTrajRec.json`

## Key Adaptations

### 1. Base Class Inheritance
- Changed from `nn.Module` to `AbstractModel` from LibCity
- Implemented required methods: `__init__()`, `forward()`, `predict()`, `calculate_loss()`

### 2. Data Handling
The original model used a custom data loader with specific batch structure. The adapted model expects LibCity's batch dictionary format with keys:
- `src_lat`: Source latitude tensor (B, T)
- `src_lng`: Source longitude tensor (B, T)
- `mask_index`: Mask indices for unknown positions (B, T)
- `padd_index`: Padding indices (B, T)
- `src_candi_id`: Road candidate IDs (B, T, id_size)
- `traj_length`: Trajectory lengths list
- `prompt_token`: BERT prompt tokens (B, prompt_len, hidden_dim)
- `road_condition_xyt_index`: Road condition indices (B, T, 3)
- `forward_delta_t`: Forward time deltas (B, T)
- `backward_delta_t`: Backward time deltas (B, T)
- `forward_index`: Forward known point indices (B, T)
- `backward_index`: Backward known point indices (B, T)
- `target_road_id`: Target road IDs (B, T) - for loss calculation
- `target_rate`: Target movement rates (B, T) - for loss calculation

### 3. BERT Model Path
- Changed from hardcoded path (`/data/WeiTongLong/code/llm/BERT/BERT-small`) to configurable parameter
- Default: `bert-base-uncased` (HuggingFace model)

### 4. Configuration Parameters
Made all key hyperparameters configurable through LibCity's config system:

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_dim | 512 | Hidden dimension size |
| conv_kernel | 9 | Kernel size for trajectory convolution |
| soft_traj_num | 128 | Number of soft trajectory tokens |
| road_candi | true | Whether to use road candidate features |
| dropout | 0.3 | Dropout rate |
| lambda1 | 10 | Weight for rate prediction loss |
| bert_model_path | bert-base-uncased | Path to BERT model |
| use_lora | true | Whether to use LoRA fine-tuning |
| lora_r | 8 | LoRA attention dimension |
| lora_alpha | 32 | LoRA alpha scaling parameter |
| lora_dropout | 0.01 | LoRA dropout rate |

### 5. Data Feature Requirements
The model expects the following from `data_feature`:
- `id_size`: Number of road segment IDs
- `mbr`: Minimum Bounding Rectangle dictionary
- `road_condition`: Road condition flow data (numpy array or tensor)

## Model Components

### Preserved from Original
1. **LearnableFourierPositionalEncoding**: Encodes GPS coordinates using learnable Fourier features
2. **TemporalPositionalEncoding**: Sinusoidal positional encoding for sequence positions
3. **ReprogrammingLayer**: Multi-head attention for adapting embeddings with soft prompts
4. **SpatialTemporalConv**: Processes road condition grids with spatial and temporal convolutions
5. **TrajConv**: 1D convolution along trajectory dimension
6. **Decoder**: FC layers for road ID and rate prediction
7. **BERTEncoder**: BERT with optional LoRA adapters

### Modified for LibCity
1. **PLMTrajRec class**: Main model class inheriting from AbstractModel
2. **forward()**: Adapted to accept batch dictionary
3. **predict()**: Returns dictionary with road_id and road_rate predictions
4. **calculate_loss()**: Computes combined classification and regression loss

## Dependencies

### Required
- PyTorch
- NumPy

### Optional (for full functionality)
- transformers (HuggingFace) - for BERT model
- peft - for LoRA fine-tuning

## Usage Example

```python
from libcity.model.trajectory_loc_prediction import PLMTrajRec

# Configuration
config = {
    'device': 'cuda',
    'hidden_dim': 512,
    'conv_kernel': 9,
    'bert_model_path': 'bert-base-uncased',
    'use_lora': True
}

# Data features
data_feature = {
    'id_size': 2505,
    'mbr': {
        'min_lat': 30.655,
        'max_lat': 30.727,
        'min_lng': 104.043,
        'max_lng': 104.129
    },
    'road_condition': road_condition_tensor
}

# Initialize model
model = PLMTrajRec(config, data_feature)

# Training
loss = model.calculate_loss(batch)

# Prediction
predictions = model.predict(batch)
```

## Limitations and Notes

1. **Custom Data Encoder**: The model requires specific batch format with trajectory recovery specific fields. A custom trajectory encoder may need to be implemented for the LibCity data pipeline.

2. **BERT Dependency**: The model requires HuggingFace transformers library. If not installed, the model will fail to initialize.

3. **LoRA Dependency**: For LoRA fine-tuning, the peft library is required. If not available, the model falls back to standard BERT fine-tuning.

4. **Road Condition Data**: The model expects pre-computed road condition flow data as a 3D tensor (T, N, N) where T is time slots and N is grid size.

5. **Prompt Tokens**: The model expects pre-computed BERT prompt tokens. In the original implementation, these are generated from task description text.

## Registration

The model has been registered in:
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
