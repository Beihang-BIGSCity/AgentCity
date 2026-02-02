# HetETA Model Migration Summary

## Overview

This document describes the migration of the HetETA (Heterogeneous ETA) model from TensorFlow 1.x to PyTorch for the LibCity framework.

## Original Source Files

- **Main Model**: `/home/wangwenrui/shk/AgentCity/repos/HetETA/codes/model/HetETA_model.py`
- **Custom Layers**: `/home/wangwenrui/shk/AgentCity/repos/HetETA/codes/model/cell/cheb_layer.py`
- **Attention Layers**: `/home/wangwenrui/shk/AgentCity/repos/HetETA/codes/model/util/layers.py`

## Adapted Output Location

- **Model File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/HetETA.py`
- **Config File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/eta/HetETA.json`

## Model Architecture

The HetETA model uses Heterogeneous Information Networks for travel time estimation:

1. **Multi-Period Temporal Patterns**: Captures recent, daily, and weekly patterns
2. **Chebyshev Polynomial Graph Filters**: K-hop diffusion on the road network
3. **Heterogeneous Graph Attention**: Multi-head attention for different relation types
4. **Spatio-Temporal Convolutional Blocks**: Combined spatial and temporal convolutions

### Architecture Diagram

```
Input (T=weeks+days+recent, N nodes, F features)
         |
    +----+----+----+
    |    |    |    |
  weeks days recent
    |    |    |    |
  STConvBlock (per time type)
    |    - TemporalConvLayer (GLU)
    |    - SpatioConvLayerCheb (Road + Car networks)
    |    - TemporalConvLayer
    |    - NormLayer + Dropout
    |
  STLastLayer (if seq_len > 1)
    |
  Concatenate outputs
    |
  STPredictLayer
    |
  Sigmoid + Speed scaling
    |
  Output (predicted speed)
```

## Key Components Ported

### 1. NormLayer
- Layer normalization across node and feature dimensions
- TF: `tf.nn.moments` -> PyTorch: `tensor.mean()` and `tensor.var()`

### 2. TemporalConvLayer
- 1D temporal convolution with GLU/ReLU activation
- TF: `tf.nn.conv2d` -> PyTorch: `nn.Conv2d`
- Supports residual connections with dimension alignment

### 3. AttentionHeadCheb
- Attention mechanism for Chebyshev graph convolution
- TF sparse operations -> PyTorch sparse tensor operations
- Computes attention scores for multiple relation types

### 4. MultiAttentionCheb
- Multi-head attention wrapper
- Concatenates or averages head outputs

### 5. SpatioConvLayerCheb
- Spatial graph convolution using Chebyshev polynomials
- Supports heterogeneous networks (road + vehicle)

### 6. STConvBlock
- Combined spatio-temporal convolutional block
- Two temporal convolutions sandwiching spatial convolution

### 7. STLastLayer and STPredictLayer
- Final temporal aggregation and prediction layers

## Key Transformations

### TensorFlow to PyTorch Conversions

| TensorFlow | PyTorch |
|------------|---------|
| `tf.placeholder` | Input batch dict |
| `tf.Variable` | `nn.Parameter` |
| `tf.get_variable` | `nn.Linear`, `nn.Conv2d` |
| `tf.sparse_placeholder` | `torch.sparse_coo_tensor` |
| `tf.sparse_tensor_dense_matmul` | `torch.sparse.mm` |
| `tf.nn.conv2d` | `nn.Conv2d` |
| `tf.nn.moments` | `tensor.mean()`, `tensor.var()` |
| `tf.nn.dropout` | `nn.Dropout` |
| Variable scopes | `nn.Module` hierarchy |

### Sparse Tensor Handling

The original model uses TensorFlow sparse tensors for graph adjacency:
- `tf.sparse_placeholder` replaced with preprocessing to dense tensors
- Sparse attention computation adapted for dense operations with masking
- Option to use PyTorch sparse tensors for memory efficiency

### Graph Support Computation

```python
# Chebyshev polynomial computation (same algorithm)
lap = calculate_scaled_laplacian(adj_mx)
cheb_polys = calculate_cheb_polynomials(lap, K+1)
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_diffusion_step` | 2 | Chebyshev polynomial order (K) |
| `rnn_units` | 11 | Hidden dimension |
| `seq_len` | 4 | Recent time steps |
| `days` | 4 | Daily pattern steps |
| `weeks` | 4 | Weekly pattern steps |
| `road_net_num` | 7 | Road network relation types |
| `car_net_num` | 1 | Vehicle network relation types |
| `heads_num` | 1 | Number of attention heads |
| `dropout` | 0.0 | Dropout rate |
| `regular_rate` | 0.0005 | L2 regularization rate |
| `input_window` | 12 | Total input sequence length |
| `output_window` | 1 | Prediction horizon |

## LibCity Integration

### Required Methods Implemented

1. **`__init__(config, data_feature)`**: Initializes model from config dict
2. **`forward(batch)`**: Forward pass returning predicted speeds
3. **`predict(batch)`**: Multi-step prediction with autoregressive decoding
4. **`calculate_loss(batch)`**: Returns masked MSE loss

### Data Feature Requirements

- `num_nodes`: Number of nodes in the graph
- `feature_dim`: Input feature dimension
- `output_dim`: Output feature dimension
- `adj_mx`: Adjacency matrix (numpy array)
- `scaler`: Data scaler for inverse transform

### Batch Format

```python
batch = {
    'X': tensor,  # [batch_size, input_window, num_nodes, feature_dim]
    'y': tensor,  # [batch_size, output_window, num_nodes, feature_dim]
}
```

## Usage Example

```python
from libcity.model.eta import HetETA

config = {
    'device': torch.device('cuda'),
    'max_diffusion_step': 2,
    'rnn_units': 11,
    'seq_len': 4,
    'days': 4,
    'weeks': 4,
    'road_net_num': 7,
    'car_net_num': 1,
    'heads_num': 1,
    'dropout': 0.1,
}

data_feature = {
    'num_nodes': 100,
    'feature_dim': 5,
    'output_dim': 1,
    'adj_mx': adj_matrix,  # numpy array
    'scaler': StandardScaler(),
}

model = HetETA(config, data_feature)
predictions = model.predict(batch)
loss = model.calculate_loss(batch)
```

## Limitations and Assumptions

1. **Heterogeneous Relations**: The current implementation uses the same adjacency matrix for all relation types. For full heterogeneous support, different adjacency matrices should be provided for each relation type.

2. **Sparse Operations**: Dense tensor operations are used by default. For very large graphs, sparse tensor operations may be needed.

3. **ETA Calculation**: The `calculate_eta` method requires link distances to convert speed predictions to travel time.

4. **Multi-Period Data**: The model expects input data with multi-period patterns (recent + daily + weekly). For standard traffic datasets, the weekly and daily patterns can be disabled by setting `days=0` and `weeks=0`.

## Testing

To verify the model works correctly:

```python
import torch
from libcity.model.eta import HetETA

# Create dummy config and data_feature
config = {'device': torch.device('cpu'), 'num_nodes': 10}
data_feature = {
    'num_nodes': 10,
    'feature_dim': 1,
    'output_dim': 1,
    'adj_mx': np.random.rand(10, 10),
}

model = HetETA(config, data_feature)

# Test forward pass
batch = {
    'X': torch.randn(2, 12, 10, 1),
    'y': torch.randn(2, 1, 10, 1),
}

output = model.forward(batch)
print(f"Output shape: {output.shape}")  # Expected: [2, 1, 10, 1]
```

## References

- Original Paper: "HetETA: Heterogeneous Information Network Embedding for Estimated Time of Arrival"
- Original Repository: `/home/wangwenrui/shk/AgentCity/repos/HetETA`
- LibCity Framework: https://github.com/LibCity/Bigscity-LibCity
