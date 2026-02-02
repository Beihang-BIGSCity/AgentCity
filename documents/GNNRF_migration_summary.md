# GNNRF Migration Summary

## Overview

This document describes the migration of the GNNRF (Graph Neural Network for Random Forest) model from the original repository to the LibCity framework.

## Source Information

- **Repository Path**: `/home/wangwenrui/shk/AgentCity/repos/GNNRF`
- **Original Model Class**: `NodeEncodedGCN_1l` in `/home/wangwenrui/shk/AgentCity/repos/GNNRF/src/models/gcn_model.py`
- **Original Base Class**: `BaseModelClass` in `/home/wangwenrui/shk/AgentCity/repos/GNNRF/src/models/model_base_class.py`
- **Task Type**: ETA/Arrival Time Prediction for autonomous shuttles

## Target Files

- **Model File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_speed_prediction/GNNRF.py`
- **Config File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traffic_state_pred/GNNRF.json`
- **Registration**: Updated `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_speed_prediction/__init__.py`

## Model Architecture

The GNNRF model is a single-layer Graph Convolutional Network designed for travel time and dwell time prediction in autonomous shuttle systems.

### Original Architecture (NodeEncodedGCN_1l)

1. **Input Features**:
   - Lag features (historical travel/dwell times per node)
   - Global features (vehicle ID one-hot, temporal features, weather features)
   - Node encoding (one-hot encoding for each node)

2. **Layer Structure**:
   - Linear layer: merged_features -> hidden_dim (ReLU, Dropout)
   - GCN layer: hidden_dim -> hidden_dim (ReLU)
   - Skip connection: concatenate GCN output with original features
   - Merge layer: (hidden_dim + merged_input_size) -> hidden_dim (ReLU, Dropout)
   - Output layer: hidden_dim -> 1

3. **Key Features**:
   - Single-layer GCN with skip connections
   - Node one-hot encoding for node identity
   - Batch normalization
   - Dropout regularization

### LibCity Adapted Architecture (GNNRF)

The adapted model preserves the original architecture while conforming to LibCity conventions:

1. **Base Class**: Inherits from `AbstractTrafficStateModel`

2. **Required Methods**:
   - `__init__(config, data_feature)`: Model initialization
   - `forward(batch)`: Forward pass
   - `predict(batch)`: Prediction (calls forward)
   - `calculate_loss(batch)`: Loss computation

3. **Data Format Adaptation**:
   - Input: LibCity batch dictionary with 'X' and 'y' keys
   - X shape: (batch_size, input_window, num_nodes, feature_dim)
   - y shape: (batch_size, output_window, num_nodes, output_dim)
   - Output shape: (batch_size, output_window, num_nodes, output_dim)

## Key Changes Made

### 1. Framework Conversion

- Converted from PyTorch Lightning to LibCity's AbstractTrafficStateModel
- Replaced PyTorch Geometric's GCNConv with custom GCNConvLayer implementation
- This removes the torch_geometric dependency

### 2. Data Format Transformation

Original format (PyTorch Geometric):
```python
batch.x          # Node features
batch.edge_index # Graph edges
batch.global_feat # Global features
batch.node_encoding # Node one-hot
batch.y          # Target
```

LibCity format:
```python
batch['X']  # (batch, time, nodes, features)
batch['y']  # (batch, time, nodes, features)
```

### 3. Graph Structure Handling

- Original: Uses edge_index from PyTorch Geometric
- Adapted: Uses adjacency matrix from data_feature['adj_mx']
- Builds normalized adjacency matrix internally

### 4. Normalization

- Original: Custom Standardize class for transformations
- Adapted: Uses LibCity's scaler from data_feature['scaler']

### 5. Loss Function

- Original: L1Loss (MAE) by default
- Adapted: Uses masked_mae_torch from libcity.model.loss

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_dim | 64 | Hidden layer dimension |
| dropout | 0.1 | Dropout probability |
| aggregation_function | "mean" | GCN aggregation function |
| num_vehicles | 1 | Number of vehicles (for one-hot encoding) |
| num_time_features | 4 | Time features (dow_sin, dow_cos, tod_sin, tod_cos) |
| num_weather_features | 3 | Weather features (temp, prcp, wspd) |
| use_node_encoding | true | Whether to use node one-hot encoding |
| num_lags | 2 | Number of lag features |
| lag_feature_dim | 1 | Dimension of lag features |

## Assumptions and Limitations

1. **Graph Structure**: The model expects an adjacency matrix in data_feature. If not provided, it falls back to an identity matrix.

2. **Global Features**: The original model uses vehicle ID, time, and weather as global features. In LibCity, these may need to be provided in the batch or extracted from X's additional channels.

3. **Single-Step vs Multi-Step**: The original model predicts a single time step. For output_window > 1, the prediction is repeated (simple baseline).

4. **Feature Extraction**: The adaptation assumes lag features are in the first channel of X. More complex feature extraction may be needed for specific datasets.

## Usage Example

```python
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.model import get_model

# Load configuration
config = ConfigParser(task='traffic_state_pred', model='GNNRF', dataset='METR_LA')

# Get dataset
dataset = get_dataset(config)
train_data, valid_data, test_data = dataset.get_data()
data_feature = dataset.get_data_feature()

# Initialize model
model = get_model(config, data_feature)

# Train and evaluate using LibCity's executor
```

## Dependencies

- torch
- numpy
- logging
- libcity.model.abstract_traffic_state_model
- libcity.model.loss

No external dependencies on torch_geometric required (custom GCN implementation included).

## Testing Notes

After migration, verify:
1. Model imports correctly
2. Forward pass produces correct output shape
3. Loss calculation works with scaler
4. Training loop executes without errors

## References

- Original GNNRF Repository: GNNRF for autonomous shuttle ETA prediction
- LibCity Documentation: https://bigscity-libcity-docs.readthedocs.io/
