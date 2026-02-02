# DiffMM Phase 2 Migration Plan

## Overview
This document outlines the detailed plan for migrating DiffMM to LibCity framework.

## Phase 2: Model Integration

### Task 1: Create Model File Structure
**Location**: `Bigscity-LibCity/libcity/model/map_matching/`

**Files to Create**:
1. `DiffMM.py` - Main model class
2. Update `__init__.py` to register DiffMM

### Task 2: Port Model Components

#### 2.1 TrajEncoder (from repos/DiffMM/models/model.py)
```python
class TrajEncoder(nn.Module):
    # Lines 6-52 from original
    # Modifications needed:
    # - Adapt to LibCity's data format
    # - Use LibCity's config system
```

#### 2.2 ShortCut Model (from repos/DiffMM/models/short_cut.py)
```python
class ShortCut(nn.Module):
    # Lines 238-277 from original
    # Keep inference logic intact
    # Adapt training interface
```

#### 2.3 DiT Architecture (from repos/DiffMM/models/short_cut.py)
```python
class DiT(nn.Module):
    # Lines 173-235 from original
    # Core model architecture - minimal changes
```

#### 2.4 Layer Components (from repos/DiffMM/models/layers.py)
```python
# Port all layer classes:
# - PointEncoder
# - MultiHeadAttention
# - TransformerEncoder
# - Attention
# - FeedForward
# - Norm
```

### Task 3: Create LibCity Model Wrapper

```python
from libcity.model.abstract_model import AbstractModel

class DiffMM(AbstractModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        
        # Initialize encoder
        self.encoder = TrajEncoder(config, device)
        
        # Initialize diffusion model
        if config.get('use_shortcut', True):
            dit = DiT(...)
            self.diffusion = ShortCut(dit, ...)
        else:
            self.diffusion = GaussianDiffusion(...)
    
    def forward(self, batch):
        """Forward pass for training"""
        # Extract batch data
        # Run encoder
        # Run diffusion
        # Return predictions
    
    def calculate_loss(self, batch):
        """Calculate training loss"""
        # MSE + BCE loss
    
    def predict(self, batch):
        """Inference mode"""
        # Single-step generation
        # Return matched segments
```

### Task 4: Create Configuration File

**Location**: `Bigscity-LibCity/libcity/config/model/map_matching/DiffMM.json`

```json
{
    "model": "DiffMM",
    "task": "map_matching",
    
    "hid_dim": 256,
    "num_units": 512,
    "transformer_layers": 2,
    "depth": 2,
    "id_emb_dim": 128,
    "dropout": 0.1,
    
    "use_shortcut": true,
    "timesteps": 2,
    "samplingsteps": 1,
    "beta_schedule": "cosine",
    "objective": "pred_x0",
    "loss_type": "l2",
    
    "search_dist": 50,
    "grid_size": 50,
    "keep_ratio": 0.1,
    
    "learning_rate": 0.001,
    "max_epoch": 30,
    "batch_size": 512,
    "optimizer": "AdamW",
    "lr_scheduler": "constant",
    
    "gpu": true,
    "gpu_id": "0"
}
```

### Task 5: Create Dataset Encoder

**Location**: `Bigscity-LibCity/libcity/data/dataset/mm_encoder/`

**New File**: `DiffMMEncoder.py`

```python
from libcity.data.dataset import AbstractDataset

class DiffMMDataset(AbstractDataset):
    def __init__(self, config):
        super().__init__(config)
        
        # Load road network
        self.rn = self._load_road_network()
        
        # Load trajectories
        self.trajectories = self._load_trajectories()
    
    def _load_road_network(self):
        """Load road network from data/"""
        # Read nodeOSM.txt, edgeOSM.txt, rn_dict.json
        # Create RoadNetworkMapFull instance
    
    def _load_trajectories(self):
        """Load trajectory data"""
        # Read traj_train.txt, traj_valid.txt, traj_test.txt
        # Parse format: timestamp lat lng segment_id
    
    def get_data_feature(self):
        """Return data features for model initialization"""
        return {
            'id_size': self.rn.valid_edge_cnt_one,
            'grid_size': self.config.get('grid_size', 50),
            'time_span': self.config.get('time_span', 15)
        }
    
    def __getitem__(self, idx):
        """Return single trajectory with candidates"""
        # Generate candidate segments
        # Normalize GPS coordinates
        # Return batch format
```

### Task 6: Update Task Configuration

**Location**: `Bigscity-LibCity/libcity/config/task_config.json`

Add map_matching task:
```json
"map_matching": {
    "allowed_model": ["DiffMM"],
    "allowed_dataset": ["porto", "beijing"],
    "DiffMM": {
        "dataset_class": "DiffMMDataset",
        "executor": "MapMatchingExecutor",
        "evaluator": "MapMatchingEvaluator"
    }
}
```

## Phase 3: Testing

### Test 1: Model Instantiation
```python
# Test model creation
config = load_config('DiffMM.json')
model = DiffMM(config, data_feature)
assert model is not None
```

### Test 2: Forward Pass
```python
# Test forward pass with dummy data
batch = create_dummy_batch()
output = model(batch)
assert output.shape == expected_shape
```

### Test 3: Loss Calculation
```python
# Test loss computation
loss = model.calculate_loss(batch)
assert loss.item() > 0
```

### Test 4: Inference
```python
# Test prediction
predictions = model.predict(batch)
assert len(predictions) == len(batch)
```

### Test 5: End-to-End Training
```python
# Test on small porto dataset
python run_model.py --task map_matching --model DiffMM \
  --dataset porto --batch_size 32 --max_epoch 5
```

## Phase 4: Documentation

### Doc 1: Model Documentation
Create `DiffMM.md` in LibCity docs:
- Model description
- Architecture details
- Usage examples
- Configuration parameters

### Doc 2: Dataset Guide
Document map matching data format:
- Trajectory file format
- Road network format
- Preprocessing steps

### Doc 3: Tutorial
Create Jupyter notebook:
- Load porto dataset
- Train DiffMM model
- Evaluate results
- Visualize matched trajectories

## Dependencies to Add

Update `requirements.txt`:
```
rtree==1.0.1
geopandas==0.14.4
networkx==3.3
einops==0.8.0
```

## Success Criteria

1. Model trains without errors
2. Achieves similar accuracy to original implementation
3. Inference runs in reasonable time
4. All tests pass
5. Documentation complete

## Timeline

- **Day 1-2**: Tasks 1-3 (Model porting)
- **Day 3**: Tasks 4-6 (Configuration and data)
- **Day 4**: Phase 3 (Testing)
- **Day 5**: Phase 4 (Documentation)

## Risk Mitigation

### Risk 1: Road Network Compatibility
- **Mitigation**: Create adapter layer for RoadNetworkMapFull
- **Fallback**: Use simplified road network representation

### Risk 2: Spatial Library Dependencies
- **Mitigation**: Make rtree/geopandas optional imports
- **Fallback**: Provide simplified candidate generation

### Risk 3: Performance Issues
- **Mitigation**: Profile code and optimize bottlenecks
- **Fallback**: Document performance requirements

## Next Actions

1. Create directory structure in LibCity
2. Start porting TrajEncoder
3. Set up test environment
4. Begin configuration files

---

**Date Created**: 2026-02-02
**Status**: Ready to begin Phase 2
**Priority**: High
