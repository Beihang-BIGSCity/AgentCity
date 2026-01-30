## DutyTTE Configuration Quick Reference

### Critical Updates Applied

1. **top_k**: 2 → 4 (matches paper k=4)
2. **hidden_size**: 128 → 256 (matches paper E_U=256)
3. **n_embed**: 128 → 256 (consistency with hidden_size)

### Configuration File Locations

- **Model Config**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/eta/DutyTTE.json`
- **Task Config**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json` (line 783)
- **Model Code**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/DutyTTE.py`
- **Encoder Code**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/eta_encoder/dutytte_encoder.py`

### Verified Components

| Component | Status | Location |
|-----------|--------|----------|
| task_config.json entry | ✅ Verified | Line 783 in task_config.json |
| Model config file | ✅ Updated | DutyTTE.json |
| Model registration | ✅ Verified | eta/__init__.py (line 6) |
| Encoder registration | ✅ Verified | eta_encoder/__init__.py (line 6, 14) |
| Dataset compatibility | ✅ Verified | ETADataset |

### Paper Parameters vs LibCity Config

| Paper | LibCity | Value | Status |
|-------|---------|-------|--------|
| E_U | hidden_size | 256 | ✅ Updated |
| C | num_experts | 8 | ✅ Verified |
| k | top_k | 4 | ✅ Updated |
| m | m | 5 | ✅ Verified |
| batch_size | batch_size | 128 | ✅ Verified |
| lr | learning_rate | 0.001 | ✅ Verified |
| rho | alpha | 0.1 | ✅ Verified |
| early_stop | patience | 20 | ✅ Verified |

### Usage

```bash
# Basic usage
python run_model.py --task eta --model DutyTTE --dataset Chengdu_Taxi_Sample1

# With custom parameters
python run_model.py --task eta --model DutyTTE --dataset Beijing_Taxi_Sample \
    --batch_size 64 --learning_rate 0.0005
```

### Model Features

- **Uncertainty Quantification**: Provides prediction intervals (lower/upper bounds)
- **Sparse MoE**: Uses 8 experts, selects top-4 per input
- **Custom Loss**: Mean Interval Score (MIS) + load balancing
- **Flexible Encoding**: Works with or without segment_id/node_id in dataset

### All Checks Passed ✅

Configuration is complete and ready for training.
