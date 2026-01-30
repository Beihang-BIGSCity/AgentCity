# START Model Quick Reference

## Model Overview

**START**: Self-supervised Trajectory Representation learning with Contrastive Pre-training
- **Task**: trajectory_embedding
- **Architecture**: BERT-based with GAT embeddings
- **Training**: Contrastive learning + Masked Language Model

---

## Configuration Files

All configs in: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/trajectory_embedding/`

| Model | Config File | Purpose |
|-------|-------------|---------|
| START | START.json | Main pre-training (contrastive + MLM) |
| BERTLM | BERTLM.json | MLM only pre-training |
| BERTContrastive | BERTContrastive.json | Contrastive only pre-training |
| BERTContrastiveLM | BERTContrastiveLM.json | Same as START |
| LinearETA | LinearETA.json | ETA prediction (downstream) |
| LinearClassify | LinearClassify.json | Classification (downstream) |
| LinearSim | LinearSim.json | Similarity (downstream) |
| LinearNextLoc | LinearNextLoc.json | Next location (downstream) |

---

## Key Parameters (START.json)

### Critical Settings
```json
{
    "d_model": 768,           // Hidden dimension
    "n_layers": 12,           // Transformer layers
    "attn_heads": 12,         // Attention heads
    "seq_len": 128,           // Max sequence length
    "batch_size": 32,         // Batch size
    "learning_rate": 0.0002,  // 2e-4
    "mlm_ratio": 0.6,         // MLM loss weight
    "contra_ratio": 0.4       // Contrastive loss weight
}
```

### Common Adjustments

**For small datasets**:
```json
{
    "n_layers": 6,
    "d_model": 512,
    "batch_size": 16,
    "dropout": 0.2
}
```

**For long trajectories**:
```json
{
    "seq_len": 256,
    "batch_size": 16
}
```

**Disable GAT** (faster, less memory):
```json
{
    "add_gat": false
}
```

---

## Data Requirements

### data_feature Dictionary
```python
data_feature = {
    'vocab_size': 10000,      # Number of unique locations
    'usr_num': 500,           # Number of users (optional)
    'node_fea_dim': 10        # Node features for GAT
}
```

### Input Batch Format
```python
batch = {
    'contra_view1': (B, T, 5),      # [loc, ts, time_in_day, day_in_week, user]
    'contra_view2': (B, T, 5),      # Second augmented view
    'masked_input': (B, T, 5),      # Masked input
    'padding_masks': (B, T),        # Boolean mask
    'batch_temporal_mat': (B, T, T), # Temporal distances
    'targets': (B, T, 5),           # MLM targets
    'target_masks': (B, T, 5),      # MLM masks
    'graph_dict': {
        'node_features': (V, F),    # V=vocab_size, F=node_fea_dim
        'edge_index': (2, E),       # Edge list
        'loc_trans_prob': (E, 1)    # Transition probs
    }
}
```

---

## Usage Examples

### Pre-training
```bash
python run_model.py \
    --task trajectory_embedding \
    --model START \
    --dataset porto \
    --config_file START.json \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.0002
```

### Fine-tuning (ETA)
```bash
python run_model.py \
    --task trajectory_embedding \
    --model LinearETA \
    --dataset porto \
    --config_file LinearETA.json \
    --load_pretrained \
    --pretrained_model_path ./saved_models/START_best.pth
```

### Python API
```python
from libcity.model.trajectory_embedding import START

# Configuration
config = {
    'd_model': 768,
    'n_layers': 12,
    'attn_heads': 12,
    'seq_len': 128,
    'batch_size': 32,
    'learning_rate': 0.0002,
    'mlm_ratio': 0.6,
    'contra_ratio': 0.4,
    'device': torch.device('cuda')
}

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

---

## Model Variants

### 1. START (Default)
- **Use**: Best overall performance
- **Training**: Contrastive + MLM
- **Config**: START.json

### 2. BERTLM
- **Use**: Generative pre-training only
- **Training**: MLM only
- **Config**: BERTLM.json

### 3. BERTContrastive
- **Use**: Pure representation learning
- **Training**: Contrastive only
- **Config**: BERTContrastive.json

### 4. LinearETA
- **Use**: Travel time prediction
- **Training**: Supervised fine-tuning
- **Config**: LinearETA.json

### 5. LinearClassify
- **Use**: Mode detection, user ID
- **Training**: Supervised fine-tuning
- **Config**: LinearClassify.json

### 6. LinearSim
- **Use**: Trajectory search
- **Training**: Supervised fine-tuning
- **Config**: LinearSim.json

### 7. LinearNextLoc
- **Use**: Location prediction
- **Training**: Supervised fine-tuning
- **Config**: LinearNextLoc.json

---

## Troubleshooting

### Out of Memory
```json
// Reduce these parameters
{
    "batch_size": 16,
    "seq_len": 64,
    "n_layers": 6
}
```

### Slow Training
```json
// Disable GAT or use caching
{
    "add_gat": false
}
```

### Poor Performance
```json
// Increase model capacity
{
    "n_layers": 12,
    "attn_heads": 12,
    "d_model": 768
}
```

### Dataset Compatibility
- Ensure ContrastiveSplitLMDataset is implemented
- Check executor/evaluator availability
- Verify graph_dict is properly constructed

---

## Performance Tips

1. **Use mixed precision**: `--amp` flag
2. **Gradient accumulation**: For larger effective batch sizes
3. **Warmup learning rate**: First 10 epochs
4. **Cache GAT embeddings**: If graph is static
5. **Early stopping**: Monitor validation loss

---

## File Locations

- **Model Code**: `Bigscity-LibCity/libcity/model/trajectory_embedding/START.py`
- **Configs**: `Bigscity-LibCity/libcity/config/model/trajectory_embedding/`
- **Task Config**: `Bigscity-LibCity/libcity/config/task_config.json`
- **Documentation**: `documents/START_config_validation.md`

---

## Contact & References

- **Paper**: "START: Self-supervised Trajectory Representation learning with Contrastive Pre-training"
- **Framework**: LibCity v3.0
- **Config Status**: ✅ Validated and Production-Ready
