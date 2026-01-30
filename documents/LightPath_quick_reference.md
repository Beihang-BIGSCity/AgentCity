# LightPath Configuration Quick Reference

## Status: ✓ VERIFIED - Ready for Testing

---

## File Locations

### Model Files
- **Model**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/LightPath.py`
- **Config**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/eta/LightPath.json`
- **Encoder**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/eta_encoder/lightpath_encoder.py`

### Configuration Files
- **Task Config**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json` (Lines 769-804)

---

## Task Configuration (task_config.json)

```json
"eta": {
    "allowed_model": ["DeepTTE", "TTPNet", "MulT_TTE", "LightPath"],
    "allowed_dataset": ["Chengdu_Taxi_Sample1", "Beijing_Taxi_Sample"],
    "LightPath": {
        "dataset_class": "ETADataset",
        "executor": "ETAExecutor",
        "evaluator": "ETAEvaluator",
        "eta_encoder": "LightPathEncoder"
    }
}
```

**Status**: ✓ Correct - Matches pattern of other ETA models

---

## Model Configuration (LightPath.json)

### Key Parameters

**Architecture** (from paper):
- embed_dim: 128
- num_patches: 100
- depth: 12
- num_heads: 8
- decoder_embed_dim: 128
- decoder_depth: 1
- decoder_num_heads: 8
- mlp_ratio: 4.0

**Training**:
- max_epoch: 100
- batch_size: 64
- learning_rate: 0.001
- learner: "adam"
- weight_decay: 0.00001
- lr_scheduler: "ReduceLROnPlateau"
- clip_grad_norm: true
- use_early_stop: true
- patience: 15

**Task-Specific**:
- train_mode: "finetune" (for ETA prediction)
- use_pretrained_embeddings: false (uses learnable embeddings)
- vocab_size: 90000
- time_size: 10000
- eta_hidden_dim: 128

**Status**: ✓ All parameters validated against original paper

---

## Registry Status

### Model Registry
File: `libcity/model/eta/__init__.py`
```python
from libcity.model.eta.LightPath import LightPath
__all__ = ["DeepTTE", "TTPNet", "MulT_TTE", "LightPath"]
```
**Status**: ✓ Registered

### Encoder Registry
File: `libcity/data/dataset/eta_encoder/__init__.py`
```python
from .lightpath_encoder import LightPathEncoder
__all__ = ["DeeptteEncoder", "TtpnetEncoder", "MultTTEEncoder", "LightPathEncoder"]
```
**Status**: ✓ Registered

---

## Usage Example

### Basic Test Run
```bash
python run_model.py --task eta --model LightPath --dataset Chengdu_Taxi_Sample1
```

### With Custom Config
```python
from libcity.pipeline import run_model

run_model(
    task='eta',
    dataset='Chengdu_Taxi_Sample1',
    model='LightPath',
    config_file={
        'train_mode': 'finetune',
        'max_epoch': 50,
        'batch_size': 64
    }
)
```

---

## Expected Batch Format

The encoder provides:
```python
{
    'road_segments': tensor([batch, seq_len]),  # Road segment IDs
    'timestamps': tensor([batch, seq_len]),      # Time indices (minute of day)
    'time': tensor([batch]),                     # Ground truth travel time (seconds)
    'lens': tensor([batch]),                     # Actual sequence lengths
    'uid': tensor([batch]),                      # User IDs
    'weekid': tensor([batch]),                   # Day of week
    'timeid': tensor([batch]),                   # Start time (minute of day)
    'dist': tensor([batch]),                     # Total distance
}
```

---

## Dependencies

**Required**:
- PyTorch >= 1.8.0
- NumPy

**Optional**:
- timm >= 0.3.2 (for optimized transformers, fallback provided)

---

## Common Configuration Adjustments

### For Low Memory
```json
{
  "batch_size": 32,
  "depth": 6,
  "embed_dim": 64
}
```

### For Pre-training
```json
{
  "train_mode": "pretrain",
  "mask_ratio1": 0.75,
  "mask_ratio2": 0.85,
  "max_epoch": 200
}
```

### With Pre-trained Embeddings
```json
{
  "use_pretrained_embeddings": true,
  "node2vec_path": "/path/to/node2vec.pkl",
  "time2vec_path": "/path/to/time2vec.pkl"
}
```

---

## Verification Checklist

- [x] Model registered in task_config.json (Line 774)
- [x] Model config entry in task_config.json (Lines 798-803)
- [x] Configuration file created and validated (LightPath.json)
- [x] Model implementation complete (LightPath.py)
- [x] Data encoder implementation complete (lightpath_encoder.py)
- [x] Model registered in eta/__init__.py
- [x] Encoder registered in eta_encoder/__init__.py
- [x] Hyperparameters validated against paper
- [x] Compatible with ETADataset, ETAExecutor, ETAEvaluator
- [x] All required methods implemented (forward, predict, calculate_loss)

---

## Known Limitations

1. **Sequence Length**: Trajectories longer than `num_patches` (100) are truncated
2. **Pre-trained Embeddings**: Not required but recommended for best performance
3. **Memory**: MAE architecture can be memory-intensive (adjust batch_size if needed)
4. **timm Dependency**: Optional but recommended for better performance

---

## Next Steps

1. **Run Initial Test**: Use default configuration
2. **Monitor Metrics**: Check MAE, RMSE, MAPE on validation set
3. **Tune if Needed**: Adjust batch_size, learning_rate based on results
4. **Consider Pre-training**: If large dataset available

---

## Documentation

- **Full Verification Report**: `/home/wangwenrui/shk/AgentCity/documents/LightPath_config_verification.md`
- **Migration Documentation**: `/home/wangwenrui/shk/AgentCity/documents/LightPath_migration.md`

---

**Status**: READY FOR TESTING
**Last Updated**: 2026-01-30
