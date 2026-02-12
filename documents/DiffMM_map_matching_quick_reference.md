# DiffMM Quick Reference - Map Matching

## Model Location
```
/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/DiffMM.py
```

## Import
```python
from libcity.model.map_matching import DiffMM
```

## Task Type
**Map Matching** - Matches GPS trajectories to road network segments

## Configuration File
```
/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/map_matching/DiffMM.json
```

## Quick Start

### 1. Model Configuration
```json
{
    "model": "DiffMM",
    "dataset": "your_dataset_name",
    "dataset_class": "DiffMMDataset",
    "executor": "DeepMapMatchingExecutor",
    "evaluator": "MapMatchingEvaluator"
}
```

### 2. Key Parameters
```json
{
    "hid_dim": 256,           // Hidden dimension
    "num_units": 512,         // Denoising units
    "transformer_layers": 2,   // Trajectory encoder layers
    "depth": 2,               // DiT blocks depth
    "timesteps": 2,           // Training timesteps
    "samplingsteps": 1,       // Inference steps (1-2 for fast)
    "bootstrap_every": 8,     // Bootstrap frequency
    "dropout": 0.1
}
```

### 3. Data Requirements
```python
data_feature = {
    'id_size': <number_of_road_segments>  # Required
}

batch = {
    'current_loc': tensor,      # [B, L, 3] GPS (lat, lng, time)
    'target': tensor,           # [B, L] ground truth segment IDs
    'candidate_segs': tensor,   # [B, L, C] candidate IDs
    'candidate_feats': tensor,  # [B, L, C, 9] candidate features
    'candidate_mask': tensor,   # [B, L, C] validity mask
    'current_loc_len': list     # Actual lengths
}
```

## Components

### TrajEncoder
- GPS point encoding with transformer
- Candidate road segment attention
- Outputs trajectory embeddings

### DiT (Diffusion Transformer)
- Adaptive Layer Normalization
- Time and timestep embeddings
- Multi-head attention blocks

### ShortCut (Flow Matching)
- Bootstrap training
- 1-2 step inference
- Fast map matching

## Methods

### Training
```python
model = DiffMM(config, data_feature)
loss = model.calculate_loss(batch)
loss.backward()
```

### Inference
```python
predictions = model.predict(batch)
# Returns: [N, 1, id_size-1] probabilities
```

## Execution Pipeline

1. **Dataset:** DiffMMDataset loads GPS trajectories and road network
2. **Executor:** DeepMapMatchingExecutor handles training/evaluation
3. **Evaluator:** MapMatchingEvaluator computes metrics
4. **Model:** DiffMM performs map matching

## Performance

- **Fast Inference:** 1-2 denoising steps vs 100+ in traditional diffusion
- **Bootstrap Training:** Improves multi-step guidance
- **Flow Matching:** Direct velocity prediction

## Comparison with Other Map Matching Models

| Model | Type | Base Class | Characteristics |
|-------|------|------------|-----------------|
| STMatching | Traditional | AbstractTraditionModel | Spatial-temporal analysis |
| IVMM | Traditional | AbstractTraditionModel | Interactive voting |
| HMMM | Traditional | AbstractTraditionModel | Hidden Markov Model |
| FMM | Traditional | AbstractTraditionModel | Fast map matching |
| DeepMM | Neural | AbstractModel | Seq2Seq encoder-decoder |
| GraphMM | Neural | AbstractModel | Graph neural network |
| RLOMM | Neural | AbstractModel | Reinforcement learning |
| **DiffMM** | **Neural** | **AbstractModel** | **Diffusion flow matching** |

## Troubleshooting

### Import Error
```python
# Wrong (old location)
from libcity.model.trajectory_loc_prediction import DiffMM

# Correct (new location)
from libcity.model.map_matching import DiffMM
```

### Missing id_size
```python
# Error: data_feature must contain 'id_size'
# Solution: Ensure DiffMMDataset provides id_size
data_feature = {'id_size': road_network.num_segments}
```

### Batch Size Issues
```python
# batch_size must be divisible by bootstrap_every
# Default: batch_size=16, bootstrap_every=8 ✓
# If changed: ensure batch_size % bootstrap_every == 0
```

## Related Documentation

- Full migration details: `documents/DiffMM_migration_to_map_matching.md`
- Original adaptation: `documents/DiffMM_adaptation.md`
- Quick start guide: `documents/DiffMM_quickstart.md`

## Command Line Usage

```bash
# Run DiffMM map matching
python run_model.py --task map_matching --model DiffMM --dataset your_dataset

# With custom config
python run_model.py --task map_matching --model DiffMM --dataset your_dataset \
    --config_file custom_diffmm.json
```
