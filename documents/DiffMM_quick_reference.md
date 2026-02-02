# DiffMM Quick Reference Guide

## Basic Information
- **Repository**: https://github.com/decisionintelligence/DiffMM
- **Local Path**: /home/wangwenrui/shk/AgentCity/repos/DiffMM
- **Task**: Map Matching (GPS trajectory to road segment matching)
- **Paper**: AAAI - One-step diffusion for accurate map matching

## Main Model Classes

### 1. TrajEncoder (models/model.py:6-52)
```python
TrajEncoder(parameters, device)
# Encodes GPS trajectories with road segment candidates
# Input: GPS seq (B,L,3), segment IDs, features, masks
# Output: Trajectory embeddings (B,L,2*hid_dim)
```

### 2. ShortCut (models/short_cut.py:238-277)
```python
ShortCut(model, infer_steps, seq_length, bootstrap_every=8)
# Primary model - one-step diffusion
# Forward: training with MSE + BCE loss
# Inference: single-step generation with flow matching
```

### 3. DiT (models/short_cut.py:173-235)
```python
DiT(out_dim, hid_dim, depth, cond_dim)
# Diffusion Transformer backbone
# Uses adaptive layer normalization
# Conditions on time, timestep, and trajectory embeddings
```

### 4. GaussianDiffusion (models/diffusion.py:270-711)
```python
GaussianDiffusion(model, seq_length, timesteps, ...)
# Alternative multi-step diffusion model
# Supports DDIM/DDPM sampling
# More accurate but slower than ShortCut
```

## Key Configuration Parameters

```python
# Model Architecture
hid_dim: 256              # Hidden dimension
num_units: 512            # Denoising network units
transformer_layers: 2     # Number of transformer layers
depth: 2                  # DiT depth

# Training
learning_rate: 1e-3       # Adam learning rate
batch_size: 512           # Training batch size
epochs: 30                # Training epochs

# Diffusion
timesteps: 2              # Training timesteps
samplingsteps: 1          # Inference steps (1 for ShortCut)
beta_schedule: 'cosine'   # Beta schedule type
objective: 'pred_x0'      # Prediction objective

# Data
keep_ratio: 0.1           # GPS sampling ratio (10%)
search_dist: 50           # Candidate search distance (meters)
grid_size: 50             # Grid size (meters)
```

## Dependencies
```
python==3.11
torch==2.4.0
rtree==1.0.1
geopandas==0.14.4
networkx==3.3
einops==0.8.0
tqdm
```

## Data Format

**Trajectory File** (traj_train.txt):
```
timestamp lat lng segment_id
timestamp lat lng segment_id
...
-{count}  # Trajectory separator
```

**Road Network** (data/{city}/roadnetwork/):
- nodeOSM.txt
- edgeOSM.txt
- wayTypeOSM.txt
- rn_dict.json

## Training Command
```bash
python main.py \
  --city porto \
  --keep_ratio 0.1 \
  --epochs 30 \
  --batch_size 512 \
  --gpu_id 0 \
  --train_flag \
  --test_flag
```

## Evaluation Metrics
- Accuracy: Exact match rate
- Recall: Ground truth coverage
- Precision: Prediction correctness
- F1 Score: Harmonic mean of precision/recall

## File Locations

**Models**:
- /home/wangwenrui/shk/AgentCity/repos/DiffMM/models/model.py
- /home/wangwenrui/shk/AgentCity/repos/DiffMM/models/short_cut.py
- /home/wangwenrui/shk/AgentCity/repos/DiffMM/models/diffusion.py
- /home/wangwenrui/shk/AgentCity/repos/DiffMM/models/layers.py

**Data**:
- /home/wangwenrui/shk/AgentCity/repos/DiffMM/dataset.py
- /home/wangwenrui/shk/AgentCity/repos/DiffMM/main.py

**Utilities**:
- /home/wangwenrui/shk/AgentCity/repos/DiffMM/utils/map.py
- /home/wangwenrui/shk/AgentCity/repos/DiffMM/utils/evaluation_utils.py

## Model Flow

1. **Data Loading** (dataset.py)
   - Parse trajectory files
   - Load road network
   - Generate candidate segments
   - Normalize GPS coordinates

2. **Encoding** (TrajEncoder)
   - Encode GPS points with Transformer
   - Embed road segments
   - Cross-attention between trajectory and candidates

3. **Training** (ShortCut)
   - Bootstrap target generation
   - Flow matching with velocity prediction
   - MSE + BCE loss

4. **Inference** (ShortCut)
   - Single-step generation from noise
   - Masked softmax over candidates
   - Argmax for final segment selection

## Migration Priority Tasks

1. Create LibCity model wrapper for ShortCut
2. Implement map_matching dataset encoder
3. Add road network data loader
4. Configure evaluation metrics
5. Create JSON config files
6. Test with porto/beijing datasets

## Notes
- ShortCut is the primary model (faster)
- GaussianDiffusion is alternative (more accurate)
- Requires spatial libraries (rtree, geopandas)
- Road network preprocessing needed for new datasets
- Data is preprocessed and cached as pickle files
