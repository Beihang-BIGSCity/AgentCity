# JGRM Dataset Requirements and Configuration Reference

## Quick Overview

JGRM (Joint GPS and Route Multimodal Model) is now fully configured in LibCity. All necessary files are in place:

- **Model File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/JGRM.py`
- **Config File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/JGRM.json`
- **Encoder**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/trajectory_encoder/jgrm_encoder.py`
- **Task Config**: Registered in `task_config.json` at line 32 and 204-209

## Dataset Requirements

### Raw Data Format

JGRM requires trajectory data with the following characteristics:

#### Minimum Requirements (Handled by JGRMEncoder)
1. **Trajectory sequences**: Check-in records with location and timestamp
2. **Location information**: POI IDs or road segment IDs
3. **Temporal information**: Timestamps for each check-in

#### Ideal Data Format (For Real GPS Trajectories)
1. **GPS traces**: Sequences of GPS coordinates with timestamps
2. **Road network**: Graph structure with road segment connectivity
3. **Map-matched trajectories**: GPS points aligned with road segments
4. **Rich GPS features**: Speed, acceleration, heading, etc.

### How JGRMEncoder Handles Standard Datasets

The `JGRMEncoder` provides a compatibility layer for standard LibCity trajectory datasets (foursquare, gowalla):

```python
# Standard POI check-in data is transformed:
POI check-in → "Road segment" (route view)
               + Synthetic GPS point (GPS view)
               + Temporal features (week, minute, delta)
```

#### Data Transformations

1. **Route Stream** (from POI sequences):
   - `route_assign_mat`: POI IDs treated as "road segments"
   - `route_data`: [weekday, minute_of_day, delta_time]

2. **GPS Stream** (synthesized from POI data):
   - `gps_data`: 8D features synthesized from POI location + time
     - Features: [norm_lat, norm_lng, sin_hour, cos_hour, sin_day, cos_day, speed_proxy, heading_proxy]
   - `gps_assign_mat`: 1-to-1 mapping (each POI = 1 GPS point)
   - `gps_length`: [1, 1, 1, ...] (1 GPS point per POI)

3. **Road Network Graph**:
   - Constructed from co-occurrence patterns in trajectories
   - Edge (A, B) created when location B follows location A
   - Used for GAT-based graph encoding

### GPS Feature Details

The 8 GPS features synthesized by JGRMEncoder:

| Feature | Description | Source |
|---------|-------------|--------|
| 1. norm_lat | Normalized latitude | POI coordinates or synthetic |
| 2. norm_lng | Normalized longitude | POI coordinates or synthetic |
| 3. sin_hour | Sin encoding of hour | Timestamp |
| 4. cos_hour | Cos encoding of hour | Timestamp |
| 5. sin_day | Sin encoding of day of week | Timestamp |
| 6. cos_day | Cos encoding of day of week | Timestamp |
| 7. speed_proxy | Distance/time between points | Computed from trajectory |
| 8. heading_proxy | Direction of movement | Computed from trajectory |

## Compatible Datasets

### Current LibCity Datasets

According to `task_config.json`, these datasets can be used with JGRM:

1. **foursquare_tky** - Foursquare check-ins in Tokyo
2. **foursquare_nyc** - Foursquare check-ins in New York
3. **gowalla** - Gowalla check-in dataset
4. **foursquare_serm** - Foursquare dataset for SERM
5. **Proto** - Prototype dataset

### Dataset Selection Criteria

**Best fit for JGRM:**
- Datasets with actual GPS coordinates in `.geo` file
- Dense trajectories with many points
- Urban areas with clear movement patterns

**Works but limited:**
- POI check-in datasets without GPS
- Relies on synthetic GPS features
- Graph structure learned from co-occurrence

### Creating Custom JGRM Dataset

For optimal performance, create a custom dataset with:

```python
# Required files:
dataset_name/
├── dataset_name.dyna       # Trajectory data
├── dataset_name.geo        # GPS coordinates
├── dataset_name.usr        # User info (optional)
└── dataset_name.rel        # Road network (optional)

# .dyna format with GPS features:
dyna_id,entity_id,timestamp,location,latitude,longitude,speed,heading,...
0,user_0,2024-01-01T08:00:00Z,road_123,35.6762,139.6503,30.5,45.2,...
1,user_0,2024-01-01T08:01:00Z,road_124,35.6765,139.6510,32.1,46.8,...
...

# .geo format with road network:
geo_id,type,coordinates,properties
road_123,road,"[139.6503, 35.6762]","{\"name\": \"Main St\"}"
road_124,road,"[139.6510, 35.6765]","{\"name\": \"2nd Ave\"}"
...
```

## Example Configuration Files

### 1. Basic Training Configuration

```json
{
    "task": "traj_loc_pred",
    "model": "JGRM",
    "dataset": "foursquare_tky",
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 0.0005,
    "gpu": true,
    "gpu_id": 0
}
```

### 2. Advanced Configuration (Override defaults)

```json
{
    "task": "traj_loc_pred",
    "model": "JGRM",
    "dataset": "your_custom_dataset",

    "# Model Architecture": "",
    "hidden_size": 512,
    "route_embed_size": 256,
    "gps_embed_size": 256,
    "route_transformer_layers": 6,
    "route_transformer_heads": 8,
    "shared_transformer_layers": 3,
    "shared_transformer_heads": 4,

    "# Training": "",
    "optimizer": "AdamW",
    "learning_rate": 0.0005,
    "weight_decay": 1e-6,
    "lr_scheduler": "linear_warmup",
    "warmup_step": 2000,
    "batch_size": 128,
    "epochs": 200,

    "# Pre-training": "",
    "mask_prob": 0.25,
    "mask_len": 3,
    "mlm_loss_weight": 1.0,
    "match_loss_weight": 2.0,
    "queue_size": 4096,
    "tau": 0.05,

    "# Data": "",
    "route_max_len": 150,
    "gps_feat_num": 8,
    "min_session_len": 10,
    "max_session_len": 100,

    "# Device": "",
    "gpu": true,
    "gpu_id": 0
}
```

### 3. Fine-tuning Configuration

```json
{
    "task": "traj_loc_pred",
    "model": "JGRM",
    "dataset": "your_dataset",

    "# Load pre-trained model": "",
    "load_cache": true,
    "cache_path": "/path/to/pretrained_jgrm.pth",

    "# Fine-tuning settings": "",
    "learning_rate": 0.0001,
    "epochs": 50,
    "freeze_encoder": false,

    "# Downstream task specific": "",
    "downstream_task": "next_location",
    "num_classes": 1000
}
```

## Usage Examples

### 1. Train from Scratch

```bash
python run_model.py --task traj_loc_pred --model JGRM --dataset foursquare_tky
```

### 2. Train with Custom Config

```bash
python run_model.py --task traj_loc_pred --model JGRM \
    --dataset foursquare_nyc \
    --config_file my_jgrm_config.json
```

### 3. Python API Usage

```python
from libcity.pipeline import run_model
from libcity.utils import get_executor

# Basic usage
run_model(
    task='traj_loc_pred',
    model='JGRM',
    dataset='gowalla',
    config_file='jgrm_config.json'
)

# Advanced usage with custom parameters
config = {
    'task': 'traj_loc_pred',
    'model': 'JGRM',
    'dataset': 'foursquare_tky',
    'hidden_size': 512,
    'learning_rate': 0.001,
    'batch_size': 128,
    'epochs': 200,
    'gpu': True,
    'gpu_id': 0
}

run_model(config_file=None, **config)
```

### 4. Extract Trajectory Representations

```python
from libcity.data.dataset import TrajectoryDataset
from libcity.model.trajectory_loc_prediction import JGRM
import torch

# Load dataset and model
dataset = TrajectoryDataset(config)
model = JGRM(config, dataset.data_feature)

# Load pre-trained weights
model.load_state_dict(torch.load('pretrained_jgrm.pth'))
model.eval()

# Get representations
batch = next(iter(dataset.train_dataloader))
representations = model.get_trajectory_representation(batch)

# Access different representation types:
gps_traj = representations['gps_traj']              # GPS trajectory embedding
route_traj = representations['route_traj']          # Route trajectory embedding
joint_gps = representations['joint_gps_traj']       # Joint GPS embedding
joint_route = representations['joint_route_traj']   # Joint Route embedding
combined = representations['combined']              # Concatenated (recommended)
```

## Testing the Setup

### 1. Quick Test

```python
# Test if JGRM is properly registered
from libcity.model import trajectory_loc_prediction

assert hasattr(trajectory_loc_prediction, 'JGRM')
print("✓ JGRM model registered")

# Test encoder
from libcity.data.dataset.trajectory_encoder import JGRMEncoder

print("✓ JGRMEncoder available")

# Test config
import json
with open('Bigscity-LibCity/libcity/config/model/traj_loc_pred/JGRM.json') as f:
    config = json.load(f)
    assert config['model_name'] == 'JGRM'
    print("✓ JGRM config valid")
```

### 2. Full Integration Test

```bash
# Run a short training to verify everything works
python run_model.py --task traj_loc_pred --model JGRM \
    --dataset foursquare_tky \
    --epochs 1 \
    --batch_size 16
```

## Performance Expectations

### Training Time (approximate)

| Dataset Size | Batch Size | Epochs | GPU | Training Time |
|--------------|------------|--------|-----|---------------|
| 10K trajectories | 64 | 100 | V100 | ~2 hours |
| 50K trajectories | 64 | 100 | V100 | ~8 hours |
| 100K trajectories | 64 | 100 | V100 | ~15 hours |

### Memory Requirements

| Configuration | GPU Memory | Notes |
|---------------|------------|-------|
| Default (hidden=256) | ~4GB | Standard |
| Large (hidden=512) | ~8GB | Better performance |
| Extra Large (hidden=1024) | ~16GB | Research use |

## Troubleshooting

### Common Issues

#### 1. torch-geometric not found
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-{version}.html
```

If GAT is not needed, set `mode: "p"` in config to use plain embeddings.

#### 2. GPU memory overflow
- Reduce `batch_size`
- Reduce `hidden_size`, `route_embed_size`, `gps_embed_size`
- Reduce `queue_size`
- Reduce `route_max_len`

#### 3. Data format errors
- Ensure dataset has `.dyna` and `.geo` files
- Check timestamp format is compatible with `parse_time()`
- Verify location IDs are consistent across files

#### 4. Poor performance with POI datasets
- Expected: JGRM designed for GPS+Route data
- POI check-ins lack rich GPS features
- Try increasing `hidden_size` and `epochs`
- Consider creating custom dataset with real GPS

## Next Steps

1. **Test with standard dataset**: Run on foursquare_tky or gowalla
2. **Monitor training**: Check loss convergence and validation metrics
3. **Extract representations**: Use `get_trajectory_representation()` for downstream tasks
4. **Fine-tune**: Adapt pre-trained model for specific applications
5. **Create custom dataset**: For best results, use real GPS trajectory data

## References

- **Full Migration Doc**: `/home/wangwenrui/shk/AgentCity/documents/JGRM_config_migration_summary.md`
- **Model Code**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/JGRM.py`
- **Encoder Code**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/trajectory_encoder/jgrm_encoder.py`
- **Config**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/JGRM.json`

---

**Last Updated**: 2026-02-02
**Status**: Ready for testing
