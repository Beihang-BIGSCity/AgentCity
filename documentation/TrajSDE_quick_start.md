# TrajSDE Quick Start Guide

## Installation

No additional installation needed beyond LibCity requirements. The simplified model works with standard PyTorch.

## Basic Usage

### 1. Train with LibCity Pipeline

```python
from libcity.pipeline import run_model

# Run with default config
run_model(task='traj_loc_pred', model_name='TrajSDE', dataset_name='foursquare_tky')
```

### 2. Custom Configuration

Create a config file or pass parameters:

```python
from libcity.pipeline import run_model

# Custom parameters
run_model(
    task='traj_loc_pred',
    model_name='TrajSDE',
    dataset_name='foursquare_tky',
    embed_dim=128,
    hidden_size=256,
    batch_size=64,
    max_epoch=50
)
```

### 3. Programmatic Usage

```python
from libcity.model.trajectory_loc_prediction.TrajSDE import TrajSDE
import torch

# Setup
config = {
    'device': 'cuda',
    'embed_dim': 64,
    'hidden_size': 128,
    'use_native_trajsde': False,
}

data_feature = {
    'loc_size': 1000,
    'uid_size': 500,
    'tim_size': 24,
}

# Create model
model = TrajSDE(config, data_feature).to(config['device'])

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for batch in train_loader:
    optimizer.zero_grad()
    loss = model.calculate_loss(batch)
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model.predict(test_batch)
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embed_dim` | int | 64 | Embedding dimension for locations and times |
| `hidden_size` | int | 128 | Hidden state size for GRU encoder |
| `num_modes` | int | 6 | Number of trajectory modes (future use) |
| `historical_steps` | int | 21 | Historical timesteps (TrajSDE default) |
| `future_steps` | int | 60 | Future timesteps (TrajSDE default) |
| `use_native_trajsde` | bool | false | Use native TrajSDE components (experimental) |
| `learning_rate` | float | 0.001 | Learning rate |
| `batch_size` | int | 32 | Training batch size |
| `max_epoch` | int | 100 | Maximum training epochs |

## Testing

### Run Tests
```bash
cd /home/wangwenrui/shk/AgentCity
python test_trajsde_adapter.py
```

### Expected Output
```
================================================================================
Testing TrajSDE LibCity Adapter
================================================================================

1. Creating TrajSDE model...
   ✓ Model created successfully

2. Creating dummy batch...
   ✓ Batch created

3. Testing forward pass...
   ✓ Forward pass successful
   ✓ Output shape is correct

4. Testing predict method...
   ✓ Predict successful
   ✓ Predictions have correct range (log probabilities)

5. Testing calculate_loss method...
   ✓ Loss calculation successful
   ✓ Loss is a positive scalar

6. Testing backward pass...
   ✓ Backward pass successful
   ✓ Gradients computed successfully

================================================================================
All tests passed! ✓
================================================================================
```

## Troubleshooting

### Import Warnings
**Issue**: Warnings about missing TrajSDE modules
```
Warning: Failed to import TrajSDE components: No module named 'debug_util'
```

**Solution**: This is expected when `use_native_trajsde=False`. The simplified model doesn't require TrajSDE modules.

### torch-geometric Warnings
**Issue**: Warnings about torch-scatter
```
UserWarning: An issue occurred while importing 'torch-scatter'
```

**Solution**: Not critical for simplified model. For native TrajSDE integration, reinstall torch-geometric:
```bash
pip install torch-geometric torch-scatter torch-sparse
```

### CUDA Errors
**Issue**: CUDA out of memory

**Solution**: Reduce batch size or use CPU:
```python
config = {'device': 'cpu', 'batch_size': 16}
```

## Performance Tips

1. **Start Small**: Test with small datasets first
   ```python
   run_model(..., batch_size=32, max_epoch=10)
   ```

2. **Tune Hyperparameters**: Try different embedding sizes
   ```python
   # Smaller for faster training
   embed_dim=32, hidden_size=64

   # Larger for better performance
   embed_dim=128, hidden_size=256
   ```

3. **Monitor Training**: Check loss convergence
   - Loss should decrease over epochs
   - If loss plateaus, try different learning rate

4. **Compare Baselines**: Test against RNN, LSTM models
   ```python
   run_model(task='traj_loc_pred', model_name='RNN', ...)
   run_model(task='traj_loc_pred', model_name='TrajSDE', ...)
   ```

## Next Steps

### For Basic Usage
1. ✅ Model is ready to use with LibCity datasets
2. ✅ Train on standard trajectory datasets (Foursquare, Gowalla, etc.)
3. ✅ Compare with baseline models
4. ✅ Tune hyperparameters for your dataset

### For Advanced Features
1. Add location coordinates to dataset
2. Implement coordinate mapping
3. Enable native TrajSDE components
4. Add multi-modal prediction
5. Integrate uncertainty estimation

## Example Training Session

```bash
# Navigate to LibCity directory
cd /home/wangwenrui/shk/AgentCity/Bigscity-LibCity

# Train TrajSDE on Foursquare dataset
python run_model.py --task traj_loc_pred --model TrajSDE \
    --dataset foursquare_tky --batch_size 32 --max_epoch 50 \
    --embed_dim 64 --hidden_size 128

# Evaluate
python run_model.py --task traj_loc_pred --model TrajSDE \
    --dataset foursquare_tky --load_best_epoch true --evaluate_method full
```

## API Reference

### TrajSDE Class

```python
class TrajSDE(AbstractModel):
    def __init__(self, config: dict, data_feature: dict)
    def forward(self, batch) -> torch.Tensor
    def predict(self, batch) -> torch.Tensor
    def calculate_loss(self, batch) -> torch.Tensor
```

### Input Format (LibCity Batch)

```python
batch = {
    'current_loc': torch.LongTensor,  # [batch_size, seq_len]
    'current_tim': torch.LongTensor,  # [batch_size, seq_len]
    'target': torch.LongTensor,       # [batch_size]
    'uid': torch.LongTensor,          # [batch_size]
}
```

### Output Format

```python
# predict() returns log probabilities
predictions: torch.FloatTensor  # [batch_size, loc_size]

# calculate_loss() returns scalar
loss: torch.FloatTensor  # scalar
```

## FAQs

**Q: Can I use the native TrajSDE model?**
A: Experimental support is available via `use_native_trajsde=True`, but requires additional implementation work for data conversion.

**Q: What datasets are supported?**
A: Any LibCity trajectory_loc_pred dataset (Foursquare, Gowalla, etc.)

**Q: How does this compare to the original TrajSDE?**
A: The simplified model uses similar concepts (temporal encoding, trajectory prediction) but operates on discrete locations instead of continuous coordinates.

**Q: Can I add custom features?**
A: Yes, extend the `_prepare_batch()` method to include additional features from your dataset.

**Q: What loss function is used?**
A: CrossEntropyLoss for discrete location prediction. Native TrajSDE uses L2 + DiffBCE for continuous trajectories.

## Support

For issues or questions:
1. Check the main documentation: `/home/wangwenrui/shk/AgentCity/documentation/TrajSDE_adapter_summary.md`
2. Review the source code: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/TrajSDE.py`
3. Run tests: `python test_trajsde_adapter.py`
4. Check LibCity documentation for general usage

---

**Created**: 2026-02-01
**Version**: 1.0 (Minimal Working Implementation)
