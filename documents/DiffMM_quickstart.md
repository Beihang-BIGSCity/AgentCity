# DiffMM Quick Start Guide

## Overview

DiffMM is a flow-based diffusion model for map matching that predicts road segment probabilities for GPS trajectory points.

## Installation

The model is already integrated into LibCity. No additional installation required.

## Model Location

```
Bigscity-LibCity/
└── libcity/
    ├── model/
    │   └── trajectory_loc_prediction/
    │       └── DiffMM.py  ← Model implementation
    └── config/
        └── model/
            └── trajectory_loc_prediction/
                └── DiffMM.json  ← Default configuration
```

## Quick Start

### 1. Basic Usage

```python
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.model import get_model
from libcity.executor import get_executor

# Initialize
config = ConfigParser(model='DiffMM', dataset='your_dataset')
dataset = get_dataset(config)
model = get_model(config, dataset.get_data_feature())

# Train
executor = get_executor(config, model, dataset)
executor.train()

# Evaluate
executor.evaluate()
```

### 2. Data Format

Your dataset should provide batches with the following structure:

```python
batch = {
    'current_loc': torch.Tensor,      # [batch, seq_len, 3] GPS (lng, lat, time)
    'target': torch.LongTensor,       # [batch, seq_len] Ground truth segment IDs
    'candidate_segs': torch.LongTensor,  # [batch, seq_len, max_cand] Candidate IDs
    'candidate_feats': torch.Tensor,  # [batch, seq_len, max_cand, 9] Candidate features
    'candidate_mask': torch.BoolTensor,  # [batch, seq_len, max_cand] Validity mask
    'seq_len': torch.LongTensor       # [batch] Actual sequence lengths
}
```

### 3. Configuration

Edit `DiffMM.json` or override via command line:

```bash
python run_model.py --model DiffMM --dataset my_data \
    --hid_dim 256 --timesteps 2 --sampling_steps 1 \
    --batch_size 32 --learning_rate 0.001
```

Key parameters:
- `hid_dim`: Hidden dimension (default: 256)
- `timesteps`: Training steps (default: 2)
- `sampling_steps`: Inference steps (default: 1)
- `bootstrap_every`: Bootstrap frequency (default: 8)

### 4. Inference

```python
# Load trained model
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

# Predict
with torch.no_grad():
    predictions = model.predict(batch)
    # predictions: [batch, seq_len, num_segments] probabilities

    # Get top segment for each point
    matched_segments = predictions.argmax(dim=-1)  # [batch, seq_len]
```

## Model Architecture

```
Input GPS Points → TrajEncoder → DiT (Flow Matching) → Segment Probabilities
       ↓               ↓              ↓
  [B, L, 3]      [B, L, 512]    [B, L, N_seg]
```

Components:
1. **TrajEncoder**: Encodes GPS + candidate segments
2. **DiT**: Diffusion Transformer with adaptive normalization
3. **ShortCut**: Fast flow matching inference (1-2 steps)

## Expected Performance

- **Inference Speed**: ~10ms per trajectory (100 points, 1 sampling step)
- **Accuracy**: Competitive with traditional HMM-based methods
- **Memory**: ~2GB GPU for batch_size=32

## Common Issues

### Issue 1: Out of Memory
**Solution**: Reduce `batch_size` or `hid_dim`

### Issue 2: Slow Training
**Solution**: Increase `bootstrap_every` (less frequent bootstrapping)

### Issue 3: Poor Accuracy
**Solution**:
- Increase `sampling_steps` (2-4 steps)
- Ensure candidate segments cover ground truth
- Check `search_dist` parameter

### Issue 4: NaN Loss
**Solution**:
- Lower `learning_rate` (try 0.0001)
- Enable gradient clipping (default: 1.0)
- Check for invalid candidate features

## Advanced Usage

### Custom Loss Function

```python
class CustomDiffMM(DiffMM):
    def calculate_loss(self, batch):
        # Add custom loss components
        base_loss = super().calculate_loss(batch)

        # Example: Add temporal smoothness
        predictions = self.predict(batch)
        temporal_loss = ((predictions[:, 1:] - predictions[:, :-1]) ** 2).mean()

        return base_loss + 0.1 * temporal_loss
```

### Multi-GPU Training

```python
config = ConfigParser(
    model='DiffMM',
    dataset='my_data',
    gpu_id='0,1,2,3'  # Use 4 GPUs
)
```

### Hyperparameter Tuning

```python
# Grid search example
for hid_dim in [128, 256, 512]:
    for timesteps in [2, 4, 8]:
        config = ConfigParser(
            model='DiffMM',
            hid_dim=hid_dim,
            timesteps=timesteps
        )
        # Train and evaluate...
```

## Data Preparation

### Required Preprocessing

1. **Road Network**: Load and index road segments
2. **Candidate Generation**: For each GPS point, find nearby road segments
3. **Feature Extraction**: Compute 9-dimensional features per candidate
   - Typical features: distance, angle, speed, heading, etc.

### Example Candidate Generation

```python
def generate_candidates(gps_point, road_network, search_radius=50):
    """
    Generate candidate road segments for a GPS point.

    Args:
        gps_point: (lng, lat, time)
        road_network: Road network object
        search_radius: Search radius in meters

    Returns:
        candidates: List of (segment_id, features)
    """
    candidates = road_network.range_query(gps_point[:2], search_radius)

    features = []
    for seg_id in candidates:
        segment = road_network.get_segment(seg_id)

        # Compute 9-dimensional features
        feat = [
            haversine_distance(gps_point, segment),  # Distance
            bearing_angle(gps_point, segment),       # Angle
            segment.length,                          # Length
            segment.speed_limit,                     # Speed limit
            # ... 5 more features
        ]
        features.append((seg_id, feat))

    return features
```

## Evaluation Metrics

DiffMM supports standard map matching metrics:

- **Accuracy**: Percentage of correctly matched segments
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

## Troubleshooting

### Debugging Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check intermediate outputs
model.eval()
with torch.no_grad():
    batch = next(iter(dataloader))
    enc_out = model.encoder(
        batch['current_loc'],
        batch['seq_len'],
        batch['candidate_segs'],
        batch['candidate_feats'],
        batch['candidate_mask']
    )
    print(f"Encoder output shape: {enc_out.shape}")
```

### Gradient Checking

```python
# Check if gradients flow properly
loss = model.calculate_loss(batch)
loss.backward()

for name, param in model.named_parameters():
    if param.grad is None:
        print(f"No gradient for: {name}")
    else:
        print(f"{name}: grad norm = {param.grad.norm()}")
```

## Best Practices

1. **Start Small**: Test with small batch_size and few epochs first
2. **Validate Candidates**: Ensure candidate generation quality
3. **Monitor Loss**: Loss should decrease smoothly (no spikes)
4. **Use Pretrained**: Fine-tune from pretrained weights if available
5. **Early Stopping**: Monitor validation metrics for early stopping

## Resources

- **Documentation**: `/documents/DiffMM_adaptation.md`
- **Summary**: `/documents/DiffMM_summary.md`
- **Original Code**: `./repos/DiffMM`
- **LibCity Docs**: https://bigscity-libcity-docs.readthedocs.io/

## Support

For questions or issues:
1. Check this quick start guide
2. Review full documentation
3. Examine original DiffMM repository
4. Create an issue with error logs and configuration

---

**Last Updated**: 2026-02-06
**Model Version**: 1.0
**LibCity Compatibility**: 3.0+
