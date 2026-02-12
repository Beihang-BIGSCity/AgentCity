# DeepMM Quick Reference Guide

## Model Overview

**DeepMM** is a Seq2Seq model with attention for map matching (GPS → Road Segments).

```
GPS Locations → BiLSTM Encoder → Attention → LSTM Decoder → Road Segments
```

## Quick Start

### 1. Basic Usage

```python
from libcity.model.trajectory_loc_prediction import DeepMM

# Configure model
config = {
    'dim_loc_src': 256,
    'dim_seg_trg': 256,
    'src_hidden_dim': 512,
    'trg_hidden_dim': 512,
    'n_layers_src': 2,
    'bidirectional': True,
    'dropout': 0.5,
    'rnn_type': 'LSTM',
    'attn_type': 'dot',
    'device': torch.device('cuda')
}

data_feature = {
    'loc_size': 10000,      # GPS vocabulary size
    'road_num': 5000,       # Road segment vocabulary
    'loc_pad': 0,
    'road_pad': 0
}

model = DeepMM(config, data_feature)
```

### 2. Training

```python
# Batch format
batch = {
    'current_loc': torch.LongTensor([[1, 2, 3, ...]]),    # [batch, seq_len]
    'target_seg': torch.LongTensor([[10, 11, 12, ...]]),  # [batch, seq_len]
    'target': torch.LongTensor([[11, 12, 13, ...]])       # Shifted by 1
}

# Forward pass
loss = model.calculate_loss(batch)
loss.backward()
optimizer.step()
```

### 3. Prediction

```python
# Inference
predictions = model.predict(batch)  # Returns [batch, seq_len] with segment IDs
```

## Configuration Parameters

### Core Architecture

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `dim_loc_src` | 256 | 64-512 | GPS location embedding size |
| `dim_seg_trg` | 256 | 64-512 | Road segment embedding size |
| `src_hidden_dim` | 512 | 128-1024 | Encoder hidden dimension |
| `trg_hidden_dim` | 512 | 128-1024 | Decoder hidden dimension |
| `n_layers_src` | 2 | 1-4 | Encoder LSTM layers |
| `dropout` | 0.5 | 0.0-0.7 | Dropout probability |

### Advanced Options

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `bidirectional` | true | true/false | Use bidirectional encoder |
| `rnn_type` | "LSTM" | LSTM/GRU/RNN | Encoder RNN type |
| `attn_type` | "dot" | dot/general/mlp | Attention mechanism |

### Training Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.001 | Initial learning rate |
| `batch_size` | 128 | Training batch size |
| `max_src_length` | 40 | Max GPS sequence length |
| `max_trg_length` | 54 | Max road segment sequence length |

## Batch Format

### Required Keys

```python
batch = {
    # Input GPS locations (integer indices)
    'current_loc': torch.LongTensor [batch, seq_len],

    # Decoder input (with <start> token prepended)
    'target_seg': torch.LongTensor [batch, seq_len],

    # Ground truth (for loss calculation)
    'target': torch.LongTensor [batch, seq_len]
}
```

### Example

```python
batch = {
    'current_loc': [[1, 5, 8, 12, 0, 0]],    # GPS cells (0 = padding)
    'target_seg': [[2, 10, 15, 20, 25, 1]],  # Segments (2 = <start>, 1 = <end>)
    'target': [[10, 15, 20, 25, 1, 0]]       # Ground truth (shifted)
}
```

## Model Methods

### `__init__(config, data_feature)`
Initialize model with configuration and data features.

### `forward(batch) → decoder_logit`
Forward pass returning logits over vocabulary.
- **Input**: batch dict
- **Output**: [batch, seq_len, vocab_size]

### `predict(batch) → predictions`
Generate predictions (argmax).
- **Input**: batch dict
- **Output**: [batch, seq_len] segment IDs

### `calculate_loss(batch) → loss`
Compute training loss with padding mask.
- **Input**: batch dict
- **Output**: scalar loss tensor

### `decode(logits) → probabilities`
Convert logits to probabilities.
- **Input**: [batch, seq_len, vocab_size]
- **Output**: [batch, seq_len, vocab_size] (softmax)

## Architecture Details

### Encoder
```
Input [batch, seq_len]
  ↓
Embedding [batch, seq_len, 256]
  ↓
BiLSTM (2 layers, 256×2 hidden)
  ↓
Context [batch, seq_len, 512]
```

### Decoder
```
Target [batch, seq_len]
  ↓
Embedding [batch, seq_len, 256]
  ↓
LSTM + Attention (1 layer, 512 hidden)
  ├─ Attend to encoder context
  └─ Output hidden state
  ↓
Linear Projection [batch, seq_len, vocab_size]
```

### Attention
```
Query: decoder_state [batch, 512]
Keys: encoder_outputs [batch, seq_len, 512]
  ↓
Scores = softmax(query · keys^T)
  ↓
Context = Σ(scores × encoder_outputs)
  ↓
Output = tanh(W[context; query])
```

## Common Issues & Solutions

### 1. Out of Memory
**Problem**: GPU OOM during training
**Solutions**:
- Reduce `batch_size` (try 64 or 32)
- Reduce `src_hidden_dim` and `trg_hidden_dim` (try 256)
- Reduce `max_src_length` and `max_trg_length`

### 2. Poor Accuracy
**Problem**: Low validation accuracy
**Solutions**:
- Increase `dim_loc_src` and `dim_seg_trg` (try 512)
- Increase `src_hidden_dim` (try 768 or 1024)
- Increase `n_layers_src` (try 3 or 4)
- Reduce `dropout` (try 0.3)

### 3. Slow Training
**Problem**: Training takes too long
**Solutions**:
- Reduce `max_src_length` and `max_trg_length`
- Use smaller `dim_loc_src` and `dim_seg_trg` (try 128)
- Reduce `n_layers_src` (try 1)

### 4. Overfitting
**Problem**: High train accuracy, low validation accuracy
**Solutions**:
- Increase `dropout` (try 0.6 or 0.7)
- Use data augmentation
- Reduce model capacity (smaller hidden dims)

## Performance Tuning

### For Accuracy
```python
config = {
    'dim_loc_src': 512,
    'dim_seg_trg': 512,
    'src_hidden_dim': 1024,
    'trg_hidden_dim': 1024,
    'n_layers_src': 3,
    'dropout': 0.3
}
```

### For Speed
```python
config = {
    'dim_loc_src': 128,
    'dim_seg_trg': 128,
    'src_hidden_dim': 256,
    'trg_hidden_dim': 256,
    'n_layers_src': 1,
    'dropout': 0.5
}
```

### For Memory Efficiency
```python
config = {
    'dim_loc_src': 128,
    'dim_seg_trg': 128,
    'src_hidden_dim': 256,
    'trg_hidden_dim': 256,
    'n_layers_src': 1,
    'dropout': 0.5,
    'max_src_length': 30,
    'max_trg_length': 40
}
```

## File Locations

```
Bigscity-LibCity/
├── libcity/
│   ├── model/
│   │   └── trajectory_loc_prediction/
│   │       ├── DeepMM.py              # Model implementation
│   │       └── __init__.py            # Model registration
│   └── config/
│       └── model/
│           └── trajectory_loc_prediction/
│               └── DeepMM.json        # Default config

documents/
└── DeepMM_adaptation.md               # Full documentation
```

## Command Line Usage

```bash
# Train DeepMM
python run_model.py --task trajectory_loc_prediction \
    --model DeepMM \
    --dataset your_dataset \
    --config_file DeepMM.json

# Custom config
python run_model.py --task trajectory_loc_prediction \
    --model DeepMM \
    --dataset your_dataset \
    --dim_loc_src 512 \
    --src_hidden_dim 1024 \
    --learning_rate 0.0005
```

## Data Requirements

### Minimum
- GPS location vocabulary (discretized coordinates)
- Road segment vocabulary
- Paired trajectory sequences

### Recommended
- 10,000+ training trajectories
- 100+ unique GPS locations
- 100+ unique road segments
- Average sequence length: 20-50

## Citation

If you use this model, please cite:

```bibtex
@inproceedings{deepmm2018,
  title={Deep Sequence Learning with Auxiliary Information for Traffic Prediction},
  author={[Original Authors]},
  booktitle={KDD},
  year={2018}
}

@article{libcity,
  title={LibCity: An Open Library for Traffic Prediction},
  author={[LibCity Authors]},
  journal={ACM SIGSPATIAL},
  year={2021}
}
```

## Version Info

- **Model Version**: 1.0
- **LibCity Compatibility**: Latest
- **PyTorch Version**: ≥1.7.0
- **Last Updated**: 2026-02-06
