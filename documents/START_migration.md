# START Model Migration Report

## Overview

The START (Self-supervised Trajectory Representation learning with Contrastive Pre-training) model has been successfully adapted to the LibCity framework.

## Source Information

- **Original Repository**: `/home/wangwenrui/shk/AgentCity/repos/START`
- **Main Model File**: `/home/wangwenrui/shk/AgentCity/repos/START/libcity/model/trajectory_embedding/BERT.py`
- **Task Type**: trajectory_embedding (new task type created for LibCity)

## Files Created/Modified

### Created Files

1. **Model File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_embedding/START.py`
   - Contains all model classes:
     - `START` - Main wrapper class inheriting from `AbstractModel`
     - `BERT` - Base transformer encoder
     - `BERTLM` - BERT with masked language model
     - `BERTContrastive` - BERT with contrastive learning
     - `BERTContrastiveLM` - Combined contrastive + MLM model
     - `BERTDownstream` - BERT for downstream tasks
     - `LinearETA` - ETA prediction head
     - `LinearClassify` - Classification head
     - `LinearSim` - Similarity learning head
     - `LinearNextLoc` - Next location prediction head
     - Supporting classes: `GAT`, `GATLayer`, `GATLayerImp3`, `BERTEmbedding`, `TransformerBlock`, etc.

2. **Module Init**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_embedding/__init__.py`
   - Exports all model classes

3. **Configuration File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/trajectory_embedding/START.json`
   - Default hyperparameters for the START model

### Modified Files

1. **Task Config**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
   - Added new `trajectory_embedding` task type with model configurations

## Model Architecture

The START model uses a BERT-based architecture enhanced with:

1. **Graph Attention Network (GAT)** for location embeddings
   - Learns spatial relationships between locations
   - Supports transition probability features

2. **Positional Embeddings**
   - Sinusoidal positional encoding
   - Optional time-of-day and day-of-week embeddings

3. **Transformer Blocks**
   - Multi-head self-attention with temporal bias
   - Position-wise feed-forward networks
   - Stochastic depth (drop path)

4. **Training Objectives**
   - Contrastive learning (SimCLR, SimCSE support)
   - Masked Language Modeling (MLM)

## LibCity Integration

### AbstractModel Compliance

The `START` class properly inherits from `AbstractModel` and implements:

- `__init__(config, data_feature)`: Initializes model with configuration
- `forward(batch)`: Forward pass returning (view1_emb, view2_emb, mlm_predictions)
- `predict(batch)`: Returns dictionary with embeddings and predictions
- `calculate_loss(batch)`: Computes combined contrastive + MLM loss

### Expected Batch Format

The model expects a batch dictionary with:
```python
{
    'contra_view1': tensor,          # (B, T, F) first augmented view
    'contra_view2': tensor,          # (B, T, F) second augmented view
    'masked_input': tensor,          # (B, T, F) masked input for MLM
    'padding_masks': tensor,         # (B, T) boolean mask
    'batch_temporal_mat': tensor,    # (B, T, T) temporal distances
    'targets': tensor,               # (B, T, F) MLM targets
    'target_masks': tensor,          # (B, T, F) MLM target masks
    'graph_dict': {                  # Optional graph information
        'node_features': tensor,     # (vocab_size, node_fea_dim)
        'edge_index': tensor,        # (2, E)
        'loc_trans_prob': tensor     # (E, 1)
    }
}
```

### Data Features Required

- `vocab_size`: Number of unique locations
- `usr_num`: Number of users (optional)
- `node_fea_dim`: Node feature dimension for GAT (default: 10)

## Configuration Parameters

Key parameters in `START.json`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| d_model | 768 | Hidden dimension |
| n_layers | 6 | Number of transformer layers |
| attn_heads | 8 | Number of attention heads |
| mlp_ratio | 4 | FFN hidden dimension ratio |
| dropout | 0.1 | General dropout rate |
| mlm_ratio | 1.0 | Weight for MLM loss |
| contra_ratio | 1.0 | Weight for contrastive loss |
| temperature | 0.05 | Contrastive loss temperature |
| contra_loss_type | "simclr" | Contrastive loss type |

## Special Notes

1. **New Task Type**: This migration creates a new `trajectory_embedding` task type in LibCity. The existing executor and evaluator infrastructure may need to be extended to fully support this task.

2. **Dataset Requirements**: The model requires specialized datasets with contrastive views and masked inputs. Custom dataset classes may be needed:
   - `ContrastiveSplitLMDataset`
   - `ContrastiveLMDataset`
   - `BERTLMDataset`

3. **GAT Dependencies**: The model includes a full Graph Attention Network implementation. No external graph library (like PyTorch Geometric) is required.

4. **Downstream Tasks**: The model supports multiple downstream tasks through specialized heads:
   - ETA prediction
   - Trajectory classification
   - Trajectory similarity
   - Next location prediction

## Usage Example

```python
from libcity.model.trajectory_embedding import START

# Configuration
config = {
    'd_model': 768,
    'n_layers': 6,
    'attn_heads': 8,
    'device': torch.device('cuda'),
    # ... other parameters
}

# Data features
data_feature = {
    'vocab_size': 10000,
    'usr_num': 500,
    'node_fea_dim': 10
}

# Create model
model = START(config, data_feature)

# Training
loss = model.calculate_loss(batch)
loss.backward()

# Inference
predictions = model.predict(batch)
embeddings = predictions['embedding_view1']
```

## Compatibility Notes

- Tested with PyTorch 1.x and 2.x
- No CUDA-specific code (device-agnostic)
- Compatible with mixed precision training
- Supports gradient checkpointing for memory efficiency
