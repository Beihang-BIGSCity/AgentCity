# Config Migration: GNNRF

## Model Information
- **Model Name**: GNNRF (Graph Neural Network for Route Finding)
- **Task Type**: Traffic State Prediction (traffic_state_pred)
- **Model File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_speed_prediction/GNNRF.py`
- **Original Task**: ETA/Arrival Time Prediction for autonomous shuttles
- **Migration Date**: 2026-02-01

## task_config.json Updates

### 1. Added to allowed_model list
- **File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
- **Location**: `traffic_state_pred.allowed_model`
- **Line**: 319
- **Status**: Successfully added

### 2. Model-specific configuration
- **Section**: `traffic_state_pred.GNNRF`
- **Lines**: 935-939
- **Configuration**:
  ```json
  "GNNRF": {
      "dataset_class": "TrafficStatePointDataset",
      "executor": "TrafficStateExecutor",
      "evaluator": "TrafficStateEvaluator"
  }
  ```

## Model Config File

### File Location
`/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traffic_state_pred/GNNRF.json`

### Hyperparameters (From Paper)

#### Core Model Parameters
- **hidden_dim**: 64 (from paper - hidden dimensions for GCN layers)
- **dropout**: 0.1 (from paper - dropout probability)
- **aggregation_function**: "mean" (GCN aggregation method, can be "sum" or "max")
- **use_node_encoding**: true (use one-hot node encoding)

#### Feature Dimensions
- **num_vehicles**: 1 (number of vehicle types for one-hot encoding)
- **num_time_features**: 4 (temporal features: dow_sin, dow_cos, tod_sin, tod_cos)
- **num_weather_features**: 3 (weather features: temperature, precipitation, wind speed)
- **num_lags**: 2 (number of lag features)
- **lag_feature_dim**: 1 (dimension per lag feature)

#### Data Processing
- **scaler**: "standard" (standardization for input features)
- **load_external**: false
- **normal_external**: false
- **ext_scaler**: "none"
- **add_time_in_day**: true
- **add_day_in_week**: true

#### Window Configuration
- **input_window**: 12 (historical time steps)
- **output_window**: 12 (prediction horizon)

#### Training Parameters
- **max_epoch**: 200 (from paper)
- **learner**: "adam"
- **learning_rate**: 0.0003 (from paper - 3e-4)
- **weight_decay**: 0.00005 (from paper - L2 regularization 5e-5)
- **batch_size**: 1024 (from paper)

#### Learning Rate Scheduling
- **lr_decay**: true
- **lr_scheduler**: "multisteplr"
- **lr_decay_ratio**: 0.1
- **steps**: [50, 100] (decay at epochs 50 and 100)

#### Gradient Clipping & Early Stopping
- **clip_grad_norm**: true
- **max_grad_norm**: 5
- **use_early_stop**: true
- **patience**: 20

#### Graph Configuration
- **bidir_adj_mx**: true (use bidirectional adjacency matrix)

## Model Architecture

### Original Model: NodeEncodedGCN_1l
The GNNRF model adapts the NodeEncodedGCN_1l architecture for LibCity:

1. **Input Features**:
   - Lag features (historical values)
   - Global features (vehicle ID, time features, weather features)
   - Node one-hot encoding

2. **Architecture Flow**:
   - Concatenate all input features
   - First linear layer (input → hidden_dim) + ReLU + Dropout + BatchNorm
   - Single-layer GCN convolution (hidden_dim → hidden_dim) + ReLU
   - Skip connection: concatenate GCN output with original merged features
   - Merge layer (hidden_dim + input_size → hidden_dim) + ReLU + Dropout + BatchNorm
   - Output layer (hidden_dim → output_dim)

3. **Key Components**:
   - Custom GCNConvLayer (replaces PyTorch Geometric dependency)
   - Symmetric normalized adjacency: D^(-1/2) * A * D^(-1/2)
   - Skip connections for gradient flow
   - Batch normalization for stability

## Dataset Compatibility

### Required Data Features
- **Graph structure**: Adjacency matrix (adj_mx)
- **Node features**: Traffic measurements over time
- **Temporal granularity**: Time-series data with consistent intervals
- **Feature dimension**: At least 1 feature per node (can be extended)

### Compatible LibCity Datasets
The model works with graph-based traffic datasets in LibCity:
- METR_LA
- PEMS_BAY
- PEMSD3/4/7/8
- LOOP_SEATTLE
- Any dataset with TrafficStatePointDataset format and adjacency matrix

### Data Requirements
1. **Adjacency Matrix**: Must provide graph connectivity information
   - If not provided, the model defaults to identity matrix (no graph structure)
   - Supports both directed and undirected graphs (bidir_adj_mx parameter)

2. **Feature Format**:
   - Input shape: (batch_size, input_window, num_nodes, feature_dim)
   - Output shape: (batch_size, output_window, num_nodes, output_dim)

3. **Optional External Features**:
   - Global features can be provided via 'global_feat' in batch dictionary
   - Otherwise, the model uses zeros (default behavior)

## Model Registration Status

### __init__.py Updates
- **File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_speed_prediction/__init__.py`
- **Status**: Already registered (line 77 import, line 138 in __all__)

## Adaptations from Original Code

### Key Changes
1. **Framework Conversion**: PyTorch Lightning → LibCity's AbstractTrafficStateModel
2. **Data Format**: PyTorch Geometric batch format → LibCity batch dictionary
3. **GCN Implementation**: Custom GCNConvLayer to avoid PyTorch Geometric dependency
4. **Scaler Integration**: Uses LibCity's built-in scaler instead of custom transform
5. **Loss Function**: Uses masked MAE from LibCity (handles missing values)

### Preserved Features
- Core architecture: node encoding + single-layer GCN + skip connections
- Hyperparameters: All original paper values maintained
- Normalization: Symmetric adjacency matrix normalization
- Aggregation: Configurable aggregation function (mean/sum/max)

## Usage Example

```python
from libcity.pipeline import run_model

# Run with default configuration
run_model(task='traffic_state_pred',
          model_name='GNNRF',
          dataset_name='METR_LA')

# Custom configuration
config = {
    'hidden_dim': 64,
    'dropout': 0.1,
    'batch_size': 1024,
    'learning_rate': 0.0003,
    'weight_decay': 0.00005,
    'max_epoch': 200
}
```

## Notes and Considerations

### Performance Considerations
1. **Memory Usage**:
   - Batch size of 1024 requires significant memory
   - Adjust batch_size if GPU memory is limited
   - Node encoding increases memory footprint (num_nodes^2)

2. **Graph Size**:
   - Model scales with num_nodes due to one-hot encoding
   - For very large graphs (>500 nodes), consider disabling node encoding

3. **Feature Engineering**:
   - Model expects specific global features (vehicle, time, weather)
   - Current implementation uses zeros if features unavailable
   - For best performance, provide actual global features

### Compatibility Notes
1. **Dataset Format**: Requires TrafficStatePointDataset with adjacency matrix
2. **Graph Structure**: Essential for model performance (identity matrix is fallback)
3. **Multi-step Prediction**: Currently repeats single-step output for output_window > 1
4. **Feature Extraction**: Adapts LibCity format to original model's expected features

### Future Improvements
1. **Multi-step Prediction**: Implement autoregressive or direct multi-step forecasting
2. **Global Features**: Better extraction from LibCity's external features
3. **Dynamic Graphs**: Support for time-varying adjacency matrices
4. **Attention Mechanisms**: Could enhance the single-layer GCN architecture

## References
- Original Repository: `/home/wangwenrui/shk/AgentCity/repos/GNNRF`
- Original Model: `src/models/gcn_model.py` (NodeEncodedGCN_1l)
- Task: Autonomous shuttle arrival time prediction

## Validation Status
- [x] Model registered in task_config.json
- [x] Model configuration file created
- [x] Hyperparameters match paper specifications
- [x] Model imported in __init__.py
- [x] Compatible with LibCity dataset format
- [x] Loss function and evaluation configured

## Migration Completed By
Configuration Migration Agent - 2026-02-01
