# LightPath Model Migration Documentation

## Overview

This document describes the adaptation of the LightPath model to the LibCity framework for the ETA (Estimated Time of Arrival) prediction task.

## Original Model Information

- **Repository**: ./repos/LightPath
- **Main Model File**: ./repos/LightPath/LightPath/models/MAERecRR.py
- **Model Class**: MaskedAutoencoderViT
- **Original Task**: Trajectory representation learning with Masked Autoencoder

## Model Architecture

LightPath uses a Masked Autoencoder (MAE) architecture with Vision Transformer backbone:

1. **Patch Embedding**:
   - Pre-trained node2vec embeddings for road segments
   - Pre-trained time2vec embeddings for temporal information

2. **Encoder**:
   - 12 transformer blocks (configurable via `depth`)
   - 128-dimensional embeddings (configurable via `embed_dim`)
   - 8 attention heads (configurable via `num_heads`)
   - Sinusoidal position embeddings
   - Random masking for self-supervised learning

3. **Decoder**:
   - 1 transformer block (configurable via `decoder_depth`)
   - 128-dimensional embeddings (configurable via `decoder_embed_dim`)
   - Reconstructs masked patches

4. **Dual Training Objectives**:
   - **Reconstruction Loss**: MSE loss on masked patches
   - **Relational Reasoning Loss**: Binary cross-entropy for contrastive learning

5. **ETA Prediction Head** (added for LibCity):
   - 3-layer MLP for travel time estimation
   - Takes CLS token as input

## File Locations

### Created Files

1. **Model File**:
   `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/LightPath.py`

2. **Configuration File**:
   `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/eta/LightPath.json`

3. **Data Encoder**:
   `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/eta_encoder/lightpath_encoder.py`

### Modified Files

1. **Model Registry**:
   `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/__init__.py`
   - Added: `from libcity.model.eta.LightPath import LightPath`
   - Added: `"LightPath"` to `__all__`

2. **Encoder Registry**:
   `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/eta_encoder/__init__.py`
   - Added: `from .lightpath_encoder import LightPathEncoder`
   - Added: `"LightPathEncoder"` to `__all__`

3. **Task Configuration**:
   `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
   - Added `"LightPath"` to ETA allowed_model list
   - Added LightPath configuration entry

## Key Adaptations

### 1. Base Class Inheritance

```python
# Original
class MaskedAutoencoderViT(nn.Module):

# Adapted
class LightPath(AbstractTrafficStateModel):
```

### 2. Constructor Signature

```python
# Original
def __init__(self, node2vec, time2vec, num_patches=100, ...):

# Adapted
def __init__(self, config, data_feature):
    # Parameters loaded from config dictionary
    self.embed_dim = config.get('embed_dim', 128)
    self.num_patches = config.get('num_patches', 100)
    # ...
```

### 3. Pre-trained Embeddings

The original model requires pre-trained node2vec and time2vec embeddings. The adapted version:

- Supports optional loading of pre-trained embeddings via `node2vec_path` and `time2vec_path` config
- Falls back to learnable embeddings when pre-trained embeddings are not available
- Handles multiple embedding file formats (numpy, torch, pickle dict)

```python
# Configuration options
"use_pretrained_embeddings": false,  # Set to true to use pre-trained
"node2vec_path": "/path/to/node2vec.pkl",
"time2vec_path": "/path/to/time2vec.pkl",
```

### 4. Training Modes

The model supports two training modes:

1. **Pre-training Mode** (`train_mode: "pretrain"`):
   - Uses reconstruction + relational reasoning losses
   - For learning trajectory representations

2. **Fine-tuning Mode** (`train_mode: "finetune"`):
   - Uses ETA prediction loss (MSE)
   - For travel time estimation task

### 5. Batch Data Format

```python
# Expected batch keys from LightPathEncoder
batch = {
    'road_segments': tensor([batch, seq_len]),  # Road segment IDs
    'timestamps': tensor([batch, seq_len]),      # Time indices
    'time': tensor([batch]),                     # Ground truth travel time
    'lens': tensor([batch]),                     # Sequence lengths
    # Additional context features...
}
```

### 6. Required Methods

```python
def forward(self, batch):
    """Process batch and return predictions or training outputs."""

def predict(self, batch):
    """Return travel time predictions."""

def calculate_loss(self, batch):
    """Compute training loss."""
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| embed_dim | 128 | Embedding dimension |
| num_patches | 100 | Maximum sequence length |
| depth | 12 | Number of encoder transformer blocks |
| num_heads | 8 | Number of attention heads in encoder |
| decoder_embed_dim | 128 | Decoder embedding dimension |
| decoder_depth | 1 | Number of decoder transformer blocks |
| decoder_num_heads | 8 | Number of attention heads in decoder |
| mlp_ratio | 4.0 | MLP hidden dimension ratio |
| mask_ratio1 | 0.7 | First masking ratio (pre-training) |
| mask_ratio2 | 0.8 | Second masking ratio (pre-training) |
| mask_ratio_eval | 0.0 | Masking ratio during evaluation |
| rec_weight | 1.0 | Reconstruction loss weight |
| rr_weight | 1.0 | Relational reasoning loss weight |
| train_mode | "finetune" | Training mode ("pretrain" or "finetune") |
| use_pretrained_embeddings | false | Whether to use pre-trained embeddings |
| vocab_size | 90000 | Road segment vocabulary size |
| time_size | 10000 | Time vocabulary size |
| eta_hidden_dim | 128 | ETA head hidden dimension |

## Dependencies

- PyTorch >= 1.8.0
- timm >= 0.3.2 (optional, falls back to built-in implementation)
- numpy

## Usage Example

```python
from libcity.pipeline import run_model

# Run with default configuration
run_model(
    task='eta',
    dataset='Chengdu_Taxi_Sample1',
    model='LightPath',
    config_file=None
)

# Or with custom configuration
run_model(
    task='eta',
    dataset='Chengdu_Taxi_Sample1',
    model='LightPath',
    config_file={
        'train_mode': 'finetune',
        'embed_dim': 128,
        'depth': 6,
        'num_heads': 4,
        'max_epoch': 50
    }
)
```

## Limitations and Notes

1. **Pre-trained Embeddings**: The model works best with pre-trained node2vec and time2vec embeddings specific to the road network. Without these, it uses learnable embeddings which may require more training data.

2. **Sequence Length**: Trajectories longer than `num_patches` are truncated. Consider adjusting this parameter based on your data.

3. **timm Dependency**: While timm is optional, using it provides optimized transformer blocks. The fallback implementation is functional but may be slower.

4. **Memory Usage**: The MAE architecture with dual forward passes during pre-training can be memory-intensive. Consider reducing batch size if needed.

## Original Source References

- Original Paper: LightPath: Lightweight and Scalable Path Representation Learning
- Original Repository: ./repos/LightPath
- Key Files:
  - ./repos/LightPath/LightPath/models/MAERecRR.py (main model)
  - ./repos/LightPath/LightPath/layers/patch_embed.py (embeddings)
  - ./repos/LightPath/LightPath/layers/weight_init.py (initialization)
  - ./repos/LightPath/LightPath/utils/pos_embed.py (position embeddings)
