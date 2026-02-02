# RNTrajRec Quick Reference

## Model Overview
**RNTrajRec** - Road Network-aware Trajectory Recovery with Spatial-Temporal Transformer
- **Task:** Trajectory Location Prediction (traj_loc_pred)
- **Architecture:** Transformer Encoder + GRU Decoder with Attention
- **Paper:** "RNTrajRec: Road Network Enhanced Trajectory Recovery with Spatial-Temporal Transformer"
- **Original Repo:** https://github.com/WenMellors/RNTrajRec

---

## Quick Start

### Basic Usage
```bash
python run_model.py --task traj_loc_pred --model RNTrajRec --dataset foursquare_tky
```

### With Custom Config
```bash
python run_model.py --task traj_loc_pred --model RNTrajRec --dataset gowalla \
    --hid_dim 512 --batch_size 64 --max_epoch 50
```

---

## Key Configuration Parameters

### Architecture (from Paper)
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `hid_dim` | 512 | 128-768 | Hidden dimension |
| `loc_emb_dim` | 512 | 128-768 | Location embedding size |
| `transformer_layers` | 2 | 1-6 | Encoder layers |
| `num_heads` | 8 | 4-16 | Attention heads |
| `dropout` | 0.1 | 0.0-0.5 | Dropout rate |
| `tim_emb_dim` | 64 | 32-128 | Time embedding size |
| `teacher_forcing_ratio` | 0.5 | 0.0-1.0 | TF probability |

### Training
| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 64 | Training batch size |
| `learning_rate` | 0.0001 | Adam learning rate |
| `max_epoch` | 50 | Maximum epochs |
| `clip` | 5.0 | Gradient clipping |
| `lr_scheduler` | "steplr" | LR scheduler type |
| `lr_step` | 10 | Scheduler step |
| `lr_decay` | 0.5 | LR decay factor |

---

## Model Files

### Configuration
- **Model Config:** `/Bigscity-LibCity/libcity/config/model/traj_loc_pred/RNTrajRec.json`
- **Task Config:** `/Bigscity-LibCity/libcity/config/task_config.json` (line 34, 217-222)

### Implementation
- **Model Code:** `/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/RNTrajRec.py`
- **Init File:** `/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py`

### Documentation
- **Full Guide:** `/documentation/RNTrajRec_config_summary.md`
- **This File:** `/documentation/RNTrajRec_quick_reference.md`

---

## Supported Datasets

Compatible with **TrajectoryDataset** datasets:
- `foursquare_tky` - Tokyo check-ins
- `foursquare_nyc` - NYC check-ins
- `gowalla` - Gowalla check-ins
- `foursquare_serm` - SERM dataset
- `Proto` - Custom trajectory data

---

## Model Components

### 1. TrajEncoder (Transformer-based)
- Location + Time embeddings
- Multi-layer transformer encoder
- Positional encoding
- Attention masking for variable-length sequences

### 2. TrajDecoder (GRU + Attention)
- GRU-based autoregressive decoder
- Bahdanau attention over encoder outputs
- Teacher forcing during training
- Greedy decoding during inference

---

## Input/Output Format

### Input Batch (Required)
```python
batch = {
    'current_loc': torch.LongTensor,  # (batch, seq_len)
    'target': torch.LongTensor,        # (batch,)
}
```

### Input Batch (Optional)
```python
batch = {
    'current_loc': torch.LongTensor,   # (batch, seq_len)
    'current_tim': torch.LongTensor,   # (batch, seq_len) - if use_time=True
    'target': torch.LongTensor,        # (batch,)
}
```

### Output
```python
scores = model.predict(batch)  # (batch, loc_size) - log probabilities
```

---

## Performance Tuning

### For Small Datasets (< 10K trajectories)
```json
{
    "hid_dim": 256,
    "transformer_layers": 2,
    "batch_size": 32,
    "max_epoch": 100
}
```

### For Large Datasets (> 100K trajectories)
```json
{
    "hid_dim": 512,
    "transformer_layers": 3,
    "batch_size": 128,
    "max_epoch": 50
}
```

### For Fast Training
```json
{
    "hid_dim": 256,
    "transformer_layers": 1,
    "batch_size": 128,
    "use_time": false
}
```

### For Best Performance
```json
{
    "hid_dim": 768,
    "transformer_layers": 4,
    "num_heads": 12,
    "batch_size": 64,
    "max_epoch": 100,
    "teacher_forcing_ratio": 0.7
}
```

---

## Common Issues & Solutions

### Out of Memory
```json
{"hid_dim": 256, "batch_size": 32, "transformer_layers": 1}
```

### Slow Training
```json
{"batch_size": 128, "max_output_len": 64, "use_time": false}
```

### Poor Performance
```json
{"hid_dim": 512, "transformer_layers": 3, "teacher_forcing_ratio": 0.7}
```

### Overfitting
```json
{"dropout": 0.3, "lr_decay": 0.5, "max_epoch": 30}
```

---

## Evaluation Metrics

Uses **TrajLocPredEvaluator**:
- Accuracy@1, Accuracy@5, Accuracy@10
- Mean Reciprocal Rank (MRR)
- NDCG@5, NDCG@10
- Macro-F1, Micro-F1

---

## Differences from Original

| Feature | Original | LibCity Adaptation |
|---------|----------|-------------------|
| Road Network | DGL-based GNN | Learnable embeddings |
| Graph Structure | Explicit topology | Implicit (embedded) |
| Dependencies | DGL + PyTorch | PyTorch only |
| Output | Road segments + positions | Location indices |

---

## Command Line Examples

### Standard Training
```bash
python run_model.py \
    --task traj_loc_pred \
    --model RNTrajRec \
    --dataset foursquare_tky
```

### Custom Architecture
```bash
python run_model.py \
    --task traj_loc_pred \
    --model RNTrajRec \
    --dataset gowalla \
    --hid_dim 768 \
    --transformer_layers 4 \
    --num_heads 12
```

### Fast Training
```bash
python run_model.py \
    --task traj_loc_pred \
    --model RNTrajRec \
    --dataset foursquare_nyc \
    --hid_dim 256 \
    --batch_size 128 \
    --max_epoch 30
```

### High Performance
```bash
python run_model.py \
    --task traj_loc_pred \
    --model RNTrajRec \
    --dataset gowalla \
    --hid_dim 512 \
    --transformer_layers 3 \
    --teacher_forcing_ratio 0.7 \
    --max_epoch 100
```

---

## Hyperparameter Search Ranges

For automatic hyperparameter tuning:

```python
search_space = {
    'hid_dim': [128, 256, 512, 768],
    'transformer_layers': [1, 2, 3, 4],
    'num_heads': [4, 8, 12],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [1e-4, 5e-4, 1e-3],
    'teacher_forcing_ratio': [0.3, 0.5, 0.7],
    'batch_size': [32, 64, 128]
}
```

---

## Model Characteristics

- **Complexity:** Medium (Transformer + GRU)
- **Memory:** Moderate (attention matrices)
- **Training Speed:** Medium
- **Inference Speed:** Fast (autoregressive)
- **Accuracy:** High (with proper tuning)

---

## Related LibCity Models

**Similar Architecture:**
- PLMTrajRec - Pre-trained language model
- GETNext - Graph-enhanced transformer
- LoTNext - Long-tail aware

**Similar Purpose:**
- JGRM - Joint graph representation
- TrajSDE - Stochastic modeling
- ROTAN - Rotation-based attention

---

## Integration Checklist

- [x] Model registered in task_config.json
- [x] Configuration file created
- [x] Uses TrajectoryDataset
- [x] Uses TrajLocPredExecutor
- [x] Uses TrajLocPredEvaluator
- [x] Uses StandardTrajectoryEncoder
- [x] Compatible with existing datasets
- [x] Documentation complete

---

## Support & References

**Full Documentation:** `/documentation/RNTrajRec_config_summary.md`
**Original Paper:** "RNTrajRec: Road Network Enhanced Trajectory Recovery with Spatial-Temporal Transformer"
**Original Code:** https://github.com/WenMellors/RNTrajRec
**LibCity Docs:** https://bigscity-libcity.readthedocs.io/

---

**Last Updated:** 2026-02-02
**Status:** Production Ready
