# GraphMM Quick Reference - Trajectory Location Prediction

## Quick Start

### 1. Import Model
```python
from libcity.model.trajectory_loc_prediction import GraphMM
```

### 2. Initialize
```python
config = {
    'emb_dim': 128,
    'use_crf': True,
    'device': torch.device('cuda')
}

data_feature = {
    'num_roads': 1000,
    'num_grids': 5000
}

model = GraphMM(config, data_feature)
```

### 3. Load Graph Data (Required!)
```python
model.load_graph_data(graph_data)
```

### 4. Train
```python
loss = model.calculate_loss(batch)
loss.backward()
optimizer.step()
```

### 5. Predict
```python
predictions = model.predict(batch)
```

## Configuration Presets

### Small (Fast Training)
```json
{
  "emb_dim": 64,
  "use_crf": false,
  "batch_size": 256,
  "learning_rate": 0.001
}
```

### Medium (Balanced)
```json
{
  "emb_dim": 128,
  "use_crf": true,
  "topn": 30,
  "neg_nums": 100,
  "batch_size": 128,
  "learning_rate": 0.001
}
```

### Large (Best Accuracy)
```json
{
  "emb_dim": 256,
  "use_crf": true,
  "topn": 50,
  "neg_nums": 200,
  "batch_size": 64,
  "learning_rate": 0.0005
}
```

## Batch Format

```python
batch = {
    'grid_traces': torch.LongTensor,    # (B, seq_len) - grid IDs
    'tgt_roads': torch.LongTensor,      # (B, road_len) - target roads
    'traces_gps': torch.FloatTensor,    # (B, seq_len, 2) - GPS coordinates
    'sample_Idx': torch.LongTensor,     # (B, seq_len) - sample indices
    'trace_lens': List[int],            # trajectory lengths
    'road_lens': List[int]              # road sequence lengths
}
```

## Common Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| emb_dim | 64/128/256 | Embedding size |
| use_crf | true/false | Enable CRF layer |
| tf_ratio | 0.0-1.0 | Teacher forcing |
| drop_prob | 0.3-0.5 | Dropout rate |
| learning_rate | 0.0001-0.001 | Learning rate |
| batch_size | 32-256 | Batch size |

## Memory Management

### Reduce Memory Usage
```json
{
  "emb_dim": 64,
  "use_crf": false,
  "batch_size": 64,
  "neg_nums": 50
}
```

### Increase Accuracy (if memory allows)
```json
{
  "emb_dim": 256,
  "use_crf": true,
  "topn": 50,
  "neg_nums": 200
}
```

## Common Issues

### "Graph data must be loaded first"
```python
# BEFORE training/prediction:
model.load_graph_data(graph_data)
```

### Out of Memory
```python
# Reduce these parameters:
config['batch_size'] = 32
config['emb_dim'] = 64
config['use_crf'] = False
config['neg_nums'] = 50
```

### Slow Training
```python
# Increase batch size, disable CRF:
config['batch_size'] = 256
config['use_crf'] = False
```

## Files

- **Model**: `Bigscity-LibCity/libcity/model/trajectory_loc_prediction/GraphMM.py`
- **Config**: `Bigscity-LibCity/libcity/config/model/trajectory_loc_prediction/GraphMM.json`
- **Docs**: `documents/GraphMM_trajectory_loc_prediction_adaptation.md`
