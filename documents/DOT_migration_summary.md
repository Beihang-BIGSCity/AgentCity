# DOT Model Migration Summary

## 1. Migration Overview

| Field | Details |
|-------|---------|
| **Paper** | Origin-Destination Travel Time Oracle for Map-based Services |
| **Venue** | SIGMOD |
| **Repository** | https://github.com/Logan-Lin/DOT |
| **Status** | SUCCESS |
| **Migration Date** | 2026-01-30 |

---

## 2. Model Information

### Model Type
- **Task Category**: ETA (Estimated Time of Arrival)
- **Base Class**: `AbstractTrafficStateModel`

### Architecture Overview

DOT (Deep OD-Time) is a two-stage diffusion-based model for travel time estimation:

```
Stage 1: U-Net Denoiser (PiT Generation)
  Input: OD-Time conditions (origin, destination, departure time)
  Output: Pixelated Trajectory (PiT) representation

Stage 2: Transformer Predictor (Travel Time Estimation)
  Input: PiT image representation
  Output: Estimated travel time
```

### Key Components

1. **DiffusionProcess**: Implements forward/backward diffusion sampling with configurable beta schedules (linear, cosine, quadratic, sigmoid)

2. **Unet**: U-Net denoiser architecture with:
   - ConvNeXT or ResNet blocks
   - Linear attention mechanisms
   - Time embedding conditioning
   - Multi-scale feature processing

3. **TransformerPredictor**: Transformer-based ETA predictor with:
   - Grid embeddings
   - Positional encodings
   - Multi-head self-attention

### Parameter Count
- **Total Parameters**: 1,828,828

### Conditioning Modes
- `odt`: Origin-Destination-Time conditioning (5D: o_lng, o_lat, d_lng, d_lat, time)
- `od`: Origin-Destination conditioning only (4D)
- `t`: Time conditioning only (1D)

---

## 3. Files Created/Modified

### New Files Created

| File | Purpose |
|------|---------|
| `Bigscity-LibCity/libcity/model/eta/DOT.py` | Main model implementation (1128 lines) |
| `Bigscity-LibCity/libcity/config/model/eta/DOT.json` | Model configuration |
| `Bigscity-LibCity/libcity/data/dataset/eta_encoder/dot_encoder.py` | Custom PiT encoder |

### Files Modified

| File | Changes |
|------|---------|
| `Bigscity-LibCity/libcity/config/task_config.json` | Added DOT model registration with DOTEncoder |
| `Bigscity-LibCity/libcity/model/eta/__init__.py` | Added DOT import and export |
| `Bigscity-LibCity/libcity/data/dataset/eta_encoder/__init__.py` | Added DOTEncoder import and export |

---

## 4. Issues Fixed

### Issue 1: Missing einops Dependency
**Problem**: The original DOT model uses the `einops` library for tensor rearrangement operations, which is not a standard LibCity dependency.

**Solution**: Added optional import with graceful fallback and clear error messaging:
```python
try:
    from einops import rearrange, repeat
    HAS_EINOPS = True
except ImportError:
    HAS_EINOPS = False
    print("Warning: einops library not installed. DOT requires einops. "
          "Install it with: pip install einops")
```

### Issue 2: BatchPAD Data Format Incompatibility
**Problem**: DOT encoder outputs tensors with `no_pad_float` type, which BatchPAD returns as lists of individual tensors rather than stacked batch tensors.

**Solution**: Implemented `_stack_batch_tensor()` helper method to handle both list and tensor cases:
```python
def _stack_batch_tensor(self, batch, key, fallback_key=None):
    data = batch[key] if key in batch.data else None
    if isinstance(data, list):
        return torch.stack(data, dim=0)
    return data
```

### Issue 3: Pixelated Trajectory (PiT) Encoding
**Problem**: LibCity's standard ETA encoders do not support image-based trajectory representations required by DOT.

**Solution**: Created custom `DOTEncoder` class implementing:
- Grid-based coordinate discretization
- 3-channel PiT image generation (mask, daytime, offset)
- OD-Time feature extraction
- Automatic bounding box normalization

### Issue 4: Multi-stage Loss Computation
**Problem**: DOT requires computing both diffusion loss and prediction loss during training, which differs from standard single-loss LibCity models.

**Solution**: Modified `calculate_loss()` to combine losses with configurable weighting:
```python
total_loss = alpha * diffusion_loss + (1 - alpha) * prediction_loss
```

### Issue 5: Training/Inference Mode Handling
**Problem**: The model behavior differs significantly between training (uses real PiT) and inference (generates PiT via diffusion), requiring careful mode management.

**Solution**: Implemented mode-aware forward pass that:
- Returns `(prediction, diffusion_loss)` tuple during training
- Returns only `prediction` during inference
- Properly handles `calculate_loss()` by temporarily setting training mode

### Issue 6: Output Format Alignment
**Problem**: DOT outputs 1D predictions while LibCity's ETA evaluator expects 2D tensors of shape `(B, 1)`.

**Solution**: Added dimension handling in `predict()` method:
```python
if prediction is not None and prediction.dim() == 1:
    prediction = prediction.unsqueeze(-1)
```

---

## 5. Test Results

### Test Configuration
- **Dataset**: Beijing_Taxi_Sample
- **Epochs**: 3
- **Batch Size**: 128
- **Learning Rate**: 0.001
- **Device**: GPU (CUDA)

### Training Loss Progression

| Epoch | Training Loss |
|-------|---------------|
| 1 | 12.45 |
| 2 | 9.78 |
| 3 | 7.82 |

### Final Evaluation Metrics

| Metric | Value |
|--------|-------|
| **MAE** | 7.00 minutes |
| **RMSE** | 10.34 minutes |
| **MAPE** | 48.33% |
| **R2** | 0.54 |

---

## 6. Usage Instructions

### Basic Usage

```bash
cd Bigscity-LibCity
python run_model.py --task eta --model DOT --dataset Beijing_Taxi_Sample
```

### Custom Configuration

Create a config file (e.g., `dot_config.json`):
```json
{
    "model": "DOT",
    "dataset": "Beijing_Taxi_Sample",
    "split": 20,
    "timesteps": 1000,
    "d_model": 128,
    "num_layers": 2,
    "alpha": 0.5,
    "max_epoch": 100,
    "batch_size": 128
}
```

Run with config:
```bash
python run_model.py --task eta --model DOT --config dot_config.json
```

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `split` | 20 | Grid resolution (split x split) |
| `timesteps` | 1000 | Diffusion timesteps |
| `schedule_name` | "linear" | Beta schedule type |
| `d_model` | 128 | Transformer hidden dimension |
| `num_head` | 8 | Attention heads |
| `num_layers` | 2 | Transformer layers |
| `alpha` | 0.5 | Diffusion/prediction loss balance |
| `condition` | "odt" | Conditioning mode |

---

## 7. Dependencies

### Required Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **einops** | >=0.6.0 | Tensor rearrangement operations |
| **torch** | >=1.10.0 | Deep learning framework |
| **numpy** | >=1.20.0 | Numerical computing |

### Installation

```bash
pip install einops
```

### LibCity Standard Dependencies
The model also requires all standard LibCity dependencies as specified in the framework's requirements.

---

## 8. Recommendations

### Configuration Suggestions

#### Fast Training (Development/Testing)
```json
{
    "split": 16,
    "timesteps": 100,
    "d_model": 64,
    "num_layers": 1,
    "max_epoch": 10,
    "train_diffusion": false
}
```

#### High Accuracy (Production)
```json
{
    "split": 32,
    "timesteps": 1000,
    "d_model": 256,
    "num_layers": 4,
    "num_head": 16,
    "max_epoch": 200,
    "train_diffusion": true,
    "use_generated_pit": true
}
```

#### Memory-Constrained Environments
```json
{
    "split": 16,
    "timesteps": 500,
    "batch_size": 32,
    "d_model": 64,
    "unet_dim_mults": [1, 2]
}
```

### Performance Notes

1. **Training Time**: The diffusion component significantly increases training time. Set `train_diffusion: false` for faster iteration during development.

2. **Memory Usage**: PiT images consume memory proportional to `split^2`. Reduce `split` for memory-constrained environments.

3. **Inference Speed**: Diffusion sampling is slow (1000 steps by default). Consider:
   - Reducing `timesteps` for faster inference
   - Using pre-computed PiT images when available

4. **Dataset Requirements**: Works best with dense GPS trajectory data. Sparse trajectories may produce lower-quality PiT representations.

### Future Enhancements

1. **DDIM Sampling**: Implement DDIM for faster inference without significant quality loss

2. **Conditional Flow Matching**: Alternative to diffusion that may offer faster training

3. **Multi-resolution PiT**: Hierarchical grid representation for varying trajectory scales

4. **Transfer Learning**: Pre-train diffusion on large trajectory datasets for better generalization

---

## 9. Model Architecture Diagram

```
                    DOT Model Architecture
                    =====================

Input: OD-Time Features (origin_lng, origin_lat, dest_lng, dest_lat, time)
                            |
                            v
    +--------------------------------------------------+
    |              Stage 1: Diffusion                   |
    |                                                   |
    |   Noise (T=1000) --> U-Net Denoiser --> PiT      |
    |                           |                       |
    |                    [Conditioning]                 |
    |                           ^                       |
    |                    OD-Time Features               |
    +--------------------------------------------------+
                            |
                            v
                    PiT Image (3 x split x split)
                    - Channel 0: Binary mask
                    - Channel 1: Daytime
                    - Channel 2: Time offset
                            |
                            v
    +--------------------------------------------------+
    |           Stage 2: Transformer Predictor          |
    |                                                   |
    |   PiT --> Linear --> Transformer --> Output      |
    |                           |                       |
    |              [Grid + Position Embed]              |
    +--------------------------------------------------+
                            |
                            v
              Estimated Travel Time (minutes)
```

---

## 10. File Locations Summary

| Component | Absolute Path |
|-----------|---------------|
| Model | `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/DOT.py` |
| Config | `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/eta/DOT.json` |
| Encoder | `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/eta_encoder/dot_encoder.py` |
| Task Config | `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json` |
| Model Registry | `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/__init__.py` |
| Encoder Registry | `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/eta_encoder/__init__.py` |

---

**Migration completed successfully on 2026-01-30**
