# ProbETA Model Migration Summary

## Overview
This document describes the migration of the ProbETA model from its original implementation to the LibCity framework.

## Original Model Information
- **Location**: `/home/wangwenrui/shk/AgentCity/repos/ProbETA/Model/ProbETA/model.py`
- **Main Class**: `ProbE`
- **Task**: Travel Time Estimation (Probabilistic Link Representation Learning)
- **Architecture**: Dual embeddings + mean/covariance prediction networks

## LibCity Adapted Model
- **Location**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/ProbETA.py`
- **Class Name**: `ProbETA`
- **Base Class**: `AbstractTrafficStateModel`
- **Config File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/eta/ProbETA.json`

## Key Components

### 1. Dual Embedding Layers
- Two separate embedding layers (`embeddings1` and `embeddings2`) for road segments
- Orthogonal initialization with normalization (preserved from original)
- Padding index support added for variable-length sequences

### 2. Mean Prediction Network
- 4 fully connected layers with ReLU activation
- High dropout rate (0.9 by default) for regularization
- Layer sizes: embedding_dim -> 72 -> 64 -> 32 -> 1

### 3. Covariance Estimation Network
- 3 fully connected layers
- Lower dropout rate (0.3 by default)
- Uses softplus activation (log(1 + exp(x))) for positive definiteness
- Layer sizes: embedding_dim -> 32 -> 16 -> 1

### 4. Loss Function
- Multivariate Gaussian Negative Log-Likelihood (NLL) loss
- Considers both mean prediction accuracy and covariance structure
- Fallback to MSE loss if covariance is not positive definite

## Key Adaptations

### 1. Initialization
**Original**:
```python
def __init__(self, road_num, embedding_dim, device):
```

**LibCity**:
```python
def __init__(self, config, data_feature):
    # Parameters extracted from config and data_feature dictionaries
```

### 2. Forward Method
**Original**:
```python
def forward(self, inputs, same_trip):
    # inputs: road segment indices
    # same_trip: device IDs for similarity calculation
```

**LibCity**:
```python
def forward(self, batch):
    # batch: dictionary with 'road_segments', 'device_ids', etc.
    # Also supports LibCity default 'X' key
```

### 3. New Methods Added
- `predict(batch)`: Returns mean travel time predictions
- `predict_with_uncertainty(batch)`: Returns both mean and variance
- `calculate_loss(batch)`: Computes training loss using NLL or MSE
- `compute_crps(batch)`: Computes CRPS metric for probabilistic evaluation

### 4. Data Format Handling
- Supports `road_segments` or `X` keys for input
- Supports `time` or `y` keys for ground truth
- Automatic handling of multi-dimensional inputs
- Padding mask for variable-length sequences

## Configuration Parameters

### Model Parameters (from config)
| Parameter | Default | Description |
|-----------|---------|-------------|
| embedding_dim | 64 | Dimension of road segment embeddings |
| dropout_mean | 0.9 | Dropout rate for mean prediction network |
| dropout_cov | 0.3 | Dropout rate for covariance network |
| hidden_mean_1 | 72 | First hidden layer size for mean network |
| hidden_mean_2 | 64 | Second hidden layer size for mean network |
| hidden_mean_3 | 32 | Third hidden layer size for mean network |
| hidden_cov_1 | 32 | First hidden layer size for covariance network |
| hidden_cov_2 | 16 | Second hidden layer size for covariance network |
| use_device_similarity | true | Whether to use device ID for similarity matrix |
| loss_type | 'nll' | Loss function type ('nll' or 'mse') |

### Data Feature Parameters
| Parameter | Description |
|-----------|-------------|
| road_num | Number of road segments in the dataset |
| time_mean | Mean of travel time for normalization (optional) |
| time_std | Standard deviation of travel time (optional) |

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| max_epoch | 100 | Maximum training epochs |
| batch_size | 128 | Batch size for training |
| learning_rate | 0.001 | Initial learning rate |
| weight_decay | 0.0001 | L2 regularization weight |

## Usage Example

```python
from libcity.model.eta import ProbETA

# Configuration
config = {
    'device': 'cuda',
    'embedding_dim': 64,
    'dropout_mean': 0.9,
    'dropout_cov': 0.3,
    'loss_type': 'nll'
}

# Data features
data_feature = {
    'road_num': 10000,
    'time_mean': 300.0,
    'time_std': 150.0
}

# Initialize model
model = ProbETA(config, data_feature)

# Forward pass
batch = {
    'road_segments': torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),  # Padded sequences
    'device_ids': torch.tensor([0, 1]),
    'time': torch.tensor([120.0, 180.0])
}

# Training
loss = model.calculate_loss(batch)

# Inference
predictions = model.predict(batch)
predictions_with_uncertainty = model.predict_with_uncertainty(batch)
```

## Batch Format Requirements

The model expects a batch dictionary with the following keys:

| Key | Shape | Required | Description |
|-----|-------|----------|-------------|
| road_segments / X | (batch, seq_len) | Yes | Road segment indices (0 for padding) |
| device_ids | (batch,) | No | Device identifiers for similarity matrix |
| time / y | (batch,) | For training | Ground truth travel times |

## Limitations and Assumptions

1. **Covariance Matrix**: The covariance matrix is computed at the batch level, which means the batch size affects the training dynamics. Small batch sizes may lead to unstable covariance estimation.

2. **Numerical Stability**: A small epsilon (1e-4) is added to the covariance diagonal for numerical stability in the multivariate Gaussian.

3. **Device Similarity**: If device IDs are not provided, all samples are treated as coming from different devices (no device-based similarity).

4. **Normalization**: Travel time normalization is optional but recommended for stable training.

## Files Modified/Created

### Created
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/ProbETA.py`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/eta/ProbETA.json`
- `/home/wangwenrui/shk/AgentCity/documents/ProbETA_migration_summary.md`

### Modified
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/__init__.py` (added import and registration)

## Original vs Adapted Architecture Comparison

```
Original ProbE                          LibCity ProbETA
--------------                          ---------------
nn.Module                               AbstractTrafficStateModel
                                              |
__init__(road_num, embedding_dim, device)   __init__(config, data_feature)
                                              |
forward(inputs, same_trip)                  forward(batch)
                                              |
(no predict method)                         predict(batch)
                                              |
(external loss function)                    calculate_loss(batch)
                                              |
outputE(), outputE2()                       output_embeddings1(), output_embeddings2()
                                              |
(no uncertainty method)                     predict_with_uncertainty(batch)
                                              |
(no CRPS method)                            compute_crps(batch)
```

## Testing Notes

To test the model:
1. Ensure the data loader provides batches with the required keys
2. The `road_segments` should be integer indices (0 for padding)
3. For the NLL loss to work properly, batch size should be > 1
4. For evaluation, use both MAE/RMSE and probabilistic metrics (CRPS)
