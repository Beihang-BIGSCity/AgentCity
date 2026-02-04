# Config Migration: MVGRL

## Model Information
- **Model Name**: MVGRL (Multi-View Graph Representation Learning)
- **Original Paper**: "Contrastive Multi-View Representation Learning on Graphs" (ICML 2020)
- **Task Type**: traffic_state_pred (Traffic Speed Prediction)
- **Model Path**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_speed_prediction/MVGRL.py`

## Configuration Files Status

### 1. task_config.json
- **Status**: REGISTERED
- **Location**: `Bigscity-LibCity/libcity/config/task_config.json`
- **Added to**: `traffic_state_pred.allowed_model` (line 350)
- **Configuration**:
  ```json
  "MVGRL": {
    "dataset_class": "TrafficStatePointDataset",
    "executor": "TrafficStateExecutor",
    "evaluator": "TrafficStateEvaluator"
  }
  ```

### 2. Model Config JSON
- **Status**: CREATED & UPDATED
- **Location**: `Bigscity-LibCity/libcity/config/model/traffic_state_pred/MVGRL.json`
- **Configuration Complete**: YES

### 3. Model __init__.py
- **Status**: REGISTERED
- **Location**: `Bigscity-LibCity/libcity/model/traffic_speed_prediction/__init__.py`
- **Import Line**: 80
- **Export Line**: 144

## Model Configuration Parameters

### Core Model Architecture (from original paper)
```json
{
  "model_name": "MVGRL",
  "hidden_dim": 512,              # Hidden units (from original paper default)
  "num_gcn_layers": 2,            # Number of GCN layers per view
  "ppr_alpha": 0.2,               # PPR teleport probability
  "use_contrastive_loss": false,  # Use contrastive auxiliary loss
  "contrastive_weight": 0.1,      # Weight for contrastive loss
  "temporal_type": "conv",        # Temporal modeling type: 'conv' or 'gru'
  "dropout": 0.1                  # Dropout probability
}
```

### Dataset Configuration
```json
{
  "bidir_adj_mx": true,           # Bidirectional adjacency matrix
  "scaler": "standard",           # Feature scaler type
  "load_external": false,         # Load external features
  "normal_external": false,       # Normalize external features
  "ext_scaler": "none",           # External feature scaler
  "add_time_in_day": false,       # Add time-of-day features
  "add_day_in_week": false        # Add day-of-week features
}
```

### Training Configuration (adapted for traffic prediction)
```json
{
  "batch_size": 64,               # Batch size (from original graph-level tasks)
  "max_epoch": 100,               # Maximum epochs (adapted from 3000 for node-level)
  "learner": "adam",              # Optimizer
  "learning_rate": 0.001,         # Learning rate (from original paper)
  "weight_decay": 0.0,            # L2 regularization (from original paper)
  "lr_decay": true,               # Enable learning rate decay
  "lr_scheduler": "steplr",       # Learning rate scheduler type
  "lr_decay_ratio": 0.7,          # Learning rate decay ratio
  "step_size": 10,                # Steps between LR decay
  "clip_grad_norm": true,         # Enable gradient clipping
  "max_grad_norm": 5,             # Maximum gradient norm
  "use_early_stop": true,         # Enable early stopping
  "patience": 20                  # Early stopping patience
}
```

### Window Configuration
```json
{
  "input_window": 12,             # Input time steps
  "output_window": 12,            # Output time steps (prediction horizon)
  "output_dim": 1                 # Output feature dimension
}
```

## Parameter Mapping from Original Paper

| Original Parameter | LibCity Parameter | Value | Source |
|-------------------|------------------|-------|--------|
| Hidden units | `hidden_dim` | 512 | Original paper default |
| GNN layers | `num_gcn_layers` | 2 | Typical for MVGRL |
| PPR alpha | `ppr_alpha` | 0.2 | Standard PPR value |
| Learning rate | `learning_rate` | 0.001 | Original paper |
| L2 coefficient | `weight_decay` | 0.0 | Original paper |
| Epochs (node-level) | `max_epoch` | 100 | Adapted for traffic |
| Batch size (graph) | `batch_size` | 64 | From graph-level tasks |
| Dropout | `dropout` | 0.1 | Added for regularization |

## Model Architecture Details

### Multi-View Spatial Encoding
1. **Adjacency View**: GCN layers operating on normalized adjacency matrix
2. **PPR Diffusion View**: GCN layers operating on Personalized PageRank diffusion matrix
3. Both views use identical architecture with separate parameters

### Temporal Modeling
- **Default**: Temporal convolution layers (kernel_size=3)
- **Alternative**: GRU-based temporal modeling (2 layers)
- Choice controlled by `temporal_type` parameter

### Contrastive Learning (Optional)
- Binary cross-entropy loss between local (node) and global (graph) representations
- Discriminator uses bilinear scoring
- Can be enabled with `use_contrastive_loss: true`
- Weight controlled by `contrastive_weight` parameter

## Adaptation Notes

### Changes from Original MVGRL
1. **Supervised Learning**: Adapted from unsupervised to supervised for traffic prediction
2. **Temporal Module**: Added temporal convolution/GRU for time-series data
3. **Prediction Head**: Added multi-layer perceptron for traffic forecasting
4. **Input Format**: Adapted to LibCity's batch format (batch, time, nodes, features)
5. **PPR Computation**: Added automatic PPR matrix computation from adjacency matrix

### LibCity Integration
- Inherits from `AbstractTrafficStateModel`
- Uses LibCity's standard data pipeline (TrafficStatePointDataset)
- Compatible with TrafficStateExecutor and TrafficStateEvaluator
- Supports standard LibCity features: early stopping, learning rate scheduling, gradient clipping

## Dataset Compatibility

### Supported Datasets
All datasets in `traffic_state_pred.allowed_dataset`:
- Point-based traffic datasets (METR_LA, PEMS_BAY, PEMSD3, PEMSD4, PEMSD7, PEMSD8, etc.)
- Grid-based traffic datasets (NYC Taxi, NYC Bike, etc.)
- Subway datasets (Beijing Subway, Shanghai Metro, etc.)

### Required Data Features
- Adjacency matrix (`adj_mx`) - automatically normalized
- Node features (traffic measurements)
- No external features required by default

### PPR Matrix Computation
- Automatically computed from adjacency matrix at initialization
- Uses formula: PPR = alpha * (I - (1-alpha) * A_tilde)^(-1)
- Default alpha = 0.2 (teleport probability)
- Handles singular matrices with pseudo-inverse fallback

## Usage Example

```python
from libcity.pipeline import run_model

# Run with default configuration
run_model(task='traffic_state_pred',
          model_name='MVGRL',
          dataset_name='METR_LA')

# Run with custom configuration
config = {
    'hidden_dim': 256,
    'num_gcn_layers': 3,
    'temporal_type': 'gru',
    'use_contrastive_loss': True,
    'batch_size': 32,
    'learning_rate': 0.0005
}
run_model(task='traffic_state_pred',
          model_name='MVGRL',
          dataset_name='PEMS_BAY',
          config_dict=config)
```

## Validation Checklist

- [x] Model file exists: `MVGRL.py`
- [x] Config file exists: `MVGRL.json`
- [x] Registered in `task_config.json`
- [x] Registered in `__init__.py`
- [x] All model parameters documented
- [x] Dataset compatibility verified
- [x] Training parameters aligned with paper
- [x] PPR computation implemented
- [x] Temporal modeling configured

## Notes and Recommendations

### For Best Performance
1. **Hidden Dimension**: Start with 512 (original paper default), reduce to 256 or 128 for smaller datasets
2. **Contrastive Loss**: Set to `false` initially, enable if supervised loss plateaus
3. **Temporal Type**: Use 'conv' for faster training, 'gru' for potentially better sequence modeling
4. **Batch Size**: Use 64 for graph-level features, adjust based on GPU memory
5. **PPR Alpha**: Default 0.2 works well, increase to 0.3-0.5 for stronger diffusion

### Known Considerations
1. **Memory Usage**: PPR matrix computation requires O(N^2) space where N is number of nodes
2. **Training Time**: Two GCN branches double spatial encoding time compared to single-view models
3. **Contrastive Loss**: Adds computational overhead, use only if beneficial for specific datasets
4. **Matrix Inversion**: Falls back to pseudo-inverse for singular matrices (logged as warning)

### Future Enhancements
1. Support for dynamic adjacency matrices
2. Adaptive PPR alpha learning
3. Multi-scale temporal modeling
4. Attention-based view fusion

## References

- Original Paper: Hassani, K., & Khasahmadi, A. H. (2020). Contrastive Multi-View Representation Learning on Graphs. ICML 2020.
- Original Code: https://github.com/kavehhassani/mvgrl
- LibCity Documentation: https://bigscity-libcity-docs.readthedocs.io/

## Migration Date
2026-02-02

## Migration Status
COMPLETE - All configuration files created and verified.
