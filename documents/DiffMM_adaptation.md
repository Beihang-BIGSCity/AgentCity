# DiffMM Model Adaptation for LibCity Framework

## Overview

**Model Name**: DiffMM (Diffusion-based Map Matching)
**Original Repository**: `./repos/DiffMM`
**Task Type**: Map Matching (adapted to trajectory location prediction framework)
**Framework**: LibCity
**Base Class**: `AbstractModel`

## Original Model Architecture

DiffMM is a generative diffusion model for map matching that uses flow matching with bootstrapping for efficient inference.

### Core Components

1. **TrajEncoder** (`models/model.py`)
   - Point Encoder: Encodes GPS trajectory points using transformer layers
   - Road Embedding: Learns embeddings for road segments
   - Attention Mechanism: Attends over candidate road segments for each GPS point
   - Output: Trajectory representation conditioned on candidate segments

2. **DiT (Diffusion Transformer)** (`models/short_cut.py`)
   - Adaptive Layer Normalization (AdaLN) blocks
   - Time and timestep embeddings for flow matching
   - Multi-head self-attention with modulation
   - Feed-forward network with gating

3. **ShortCut (Flow Matching)** (`models/short_cut.py`)
   - Flow matching from noise to target distribution
   - Bootstrap sampling for improved training
   - Fast inference with minimal sampling steps
   - Deterministic ODE-based generation

### Key Features

- **Flow Matching**: Uses continuous normalizing flows instead of traditional diffusion
- **Bootstrap Training**: Improves sample quality with multi-step refinement
- **Fast Inference**: Requires only 1-2 sampling steps (vs 100+ for traditional diffusion)
- **Candidate-based**: Works with pre-computed candidate road segments
- **Point-wise Prediction**: Predicts road segment distribution for each GPS point independently

## LibCity Adaptation

### File Locations

- **Model Implementation**: `Bigscity-LibCity/libcity/model/trajectory_loc_prediction/DiffMM.py`
- **Configuration**: `Bigscity-LibCity/libcity/config/model/trajectory_loc_prediction/DiffMM.json`
- **Registration**: Updated `Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py`

### Key Adaptations

1. **Inheritance from AbstractModel**
   - Implements required methods: `__init__`, `forward`, `predict`, `calculate_loss`
   - Follows LibCity's model interface conventions

2. **Batch Format Transformation**
   - Original DiffMM expects:
     - GPS sequences with variable lengths
     - Candidate segment IDs and features per point
     - Ground truth road segment IDs

   - LibCity batch format:
     ```python
     batch = {
         'current_loc': [batch, seq_len, 3],      # GPS coordinates (lng, lat, time)
         'target': [batch, seq_len],              # Ground truth segment IDs
         'candidate_segs': [batch, seq_len, max_candidates],  # Candidate IDs
         'candidate_feats': [batch, seq_len, max_candidates, 9],  # Features
         'candidate_mask': [batch, seq_len, max_candidates],  # Validity mask
         'seq_len': [batch]                       # Actual sequence lengths
     }
     ```

3. **Model Structure Preserved**
   - All original layers and components preserved
   - TrajEncoder architecture unchanged
   - DiT blocks with AdaLN preserved
   - Flow matching algorithm identical to original

4. **Training Process**
   - Forward pass generates flow matching targets with bootstrapping
   - Loss combines MSE (flow prediction) and BCE (probability distribution)
   - Gradient clipping applied (default: 1.0)

5. **Inference Process**
   - Starts from Gaussian noise
   - Iteratively applies flow field for `sampling_steps` iterations
   - Returns softmax probabilities over road segments

### Data Flow

```
Input GPS Trajectory
    ↓
[TrajEncoder]
    ├─ PointEncoder (Transformer on GPS points)
    ├─ RoadEmbedding (Candidate segment embeddings)
    └─ Attention (GPS points attend to candidates)
    ↓
Trajectory Condition [batch, seq_len, 2*hid_dim]
    ↓
[Flow Matching Target Generation]
    ├─ Sample time t and timestep dt
    ├─ Interpolate between noise x_0 and target x_1
    ├─ Bootstrap refinement (for subset of batch)
    └─ Generate velocity field v_t
    ↓
[DiT (Diffusion Transformer)]
    ├─ Time embeddings
    ├─ AdaLN-modulated attention
    ├─ Feed-forward with gating
    └─ Output layer
    ↓
Predicted Velocity Field v_pred
    ↓
[Loss Calculation]
    ├─ MSE(v_pred, v_t)
    └─ BCE(softmax(x_t + v_pred), x_1)
    ↓
Final Loss
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hid_dim` | 256 | Hidden dimension for encoders |
| `denoise_units` | 512 | Hidden units in DiT |
| `num_layers` | 2 | Number of transformer layers |
| `timesteps` | 2 | Number of diffusion timesteps in training |
| `sampling_steps` | 1 | Number of inference steps |
| `bootstrap_every` | 8 | Bootstrap frequency (1/bootstrap_every of batch) |
| `keep_ratio` | 0.5 | GPS point sampling ratio |
| `search_dist` | 50 | Candidate segment search radius (meters) |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 0.001 | Initial learning rate |
| `clip_grad_norm` | 1.0 | Gradient clipping threshold |
| `dropout` | 0.1 | Dropout rate |

### Required Data Features

From `data_feature` dictionary:
- `id_size`: Number of road segments in the network (required)
- `seq_length`: Maximum trajectory length (default: 100)

## Differences from Original Implementation

### Preserved

1. ✅ TrajEncoder architecture (PointEncoder + Attention)
2. ✅ DiT blocks with adaptive layer normalization
3. ✅ Flow matching algorithm
4. ✅ Bootstrap training strategy
5. ✅ Fast inference with 1-2 steps
6. ✅ Loss function (MSE + BCE)

### Modified

1. 🔄 **Batch Processing**: Adapted from custom dataloader to LibCity batch format
2. 🔄 **Device Management**: Uses `config['device']` instead of manual device placement
3. 🔄 **Model Interface**: Implements LibCity's `AbstractModel` methods
4. 🔄 **Data Flattening**: Reshapes batch×seq_len to process points independently

### Removed

1. ❌ Custom dataset classes (replaced by LibCity's data loaders)
2. ❌ Training/validation loops (handled by LibCity's executor)
3. ❌ Road network preprocessing utilities (assumed to be in data pipeline)

## Usage Example

```python
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.model import get_model

# Load config
config = ConfigParser(model='DiffMM', dataset='map_matching_dataset')

# Get dataset
dataset = get_dataset(config)
data_feature = dataset.get_data_feature()

# Initialize model
model = get_model(config, data_feature)

# Training (handled by executor)
from libcity.executor import get_executor
executor = get_executor(config, model, dataset)
executor.train()

# Inference
batch = next(iter(dataset.get_data_loader('test')))
predictions = model.predict(batch)
# predictions: [batch, seq_len, id_size] probabilities
```

## Assumptions and Limitations

### Assumptions

1. **Candidate Segments**: Pre-computed candidate road segments are available for each GPS point
2. **Segment Features**: 9-dimensional feature vectors are provided for each candidate segment
3. **Road Network**: Road segment IDs are 1-indexed (0 reserved for padding)
4. **Max Candidates**: A fixed maximum number of candidates per point

### Limitations

1. **Point-wise Independence**: Each GPS point is processed independently after encoding
2. **Memory Requirements**: Full trajectory encoding before flattening can be memory-intensive
3. **Candidate Quality**: Model performance depends on candidate generation quality
4. **Fixed Network**: Requires consistent road network across train/test

## Performance Considerations

### Efficiency Improvements from Original

- **Fast Inference**: 1-2 sampling steps vs 100+ for traditional diffusion
- **Bootstrap Training**: Improves convergence speed
- **Batch Processing**: Efficient parallel processing of points

### Potential Bottlenecks

- **Candidate Attention**: O(seq_len × max_candidates) attention complexity
- **Transformer Encoding**: O(seq_len²) for point encoding
- **Bootstrap Sampling**: Additional forward passes for subset of batch

## Future Enhancements

1. **Dynamic Candidates**: Support variable number of candidates per point
2. **Hierarchical Processing**: Multi-scale trajectory encoding
3. **Uncertainty Estimation**: Output prediction confidence
4. **Online Map Matching**: Streaming trajectory processing

## References

- Original DiffMM Repository: `./repos/DiffMM`
- Flow Matching: "Flow Matching for Generative Modeling" (2022)
- Diffusion Transformers (DiT): "Scalable Diffusion Models with Transformers" (2023)

## Maintenance Notes

- **Version Compatibility**: Tested with PyTorch 1.13+, LibCity 3.0+
- **Device Support**: CPU and CUDA (single/multi-GPU)
- **Code Location**: All components in single file for easier maintenance
- **Dependencies**: Standard LibCity dependencies, no additional requirements

## Contact and Support

For issues or questions:
1. Check LibCity documentation for general model usage
2. Refer to original DiffMM repository for algorithm details
3. Review this documentation for adaptation-specific information
