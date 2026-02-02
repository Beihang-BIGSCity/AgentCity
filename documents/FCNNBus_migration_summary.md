# FCNNBus Migration Summary

## Overview
This document summarizes the migration of the FCNNBus (Fully Convolutional Neural Network for Bus Arrival Time Prediction) model from TensorFlow/Keras to PyTorch and its adaptation to the LibCity framework.

## Original Model Information

### Source Repository
- **Path**: `/home/wangwenrui/shk/AgentCity/repos/FCNNBus`
- **Training Notebook**: `/home/wangwenrui/shk/AgentCity/repos/FCNNBus/Notebooks/Copy_of_CNN_for_bus_sample_2.ipynb`

### Original Keras Architecture
```python
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', padding='same', input_shape=(4, 1)))
model.add(MaxPooling1D(2))
model.add(Conv1D(64, 2, activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1))  # Output for regression
```

### Original Model Specifications
- **Input**: 4 features (timestamp, direction, bus line, stop)
- **Input Shape**: (batch, 4, 1) for Conv1D
- **Output**: Single value (arrival time prediction)
- **Task**: Regression (bus arrival time)
- **Loss**: MSE (Mean Squared Error)
- **Optimizer**: Adam

## Adapted Model Information

### Output Files
1. **Model File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_speed_prediction/FCNNBus.py`
2. **Config File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traffic_state_pred/FCNNBus.json`
3. **Registration**: Updated `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_speed_prediction/__init__.py`

### PyTorch Architecture
```python
class FCNNBus(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        # Conv1d layer 1: in_channels=feature_dim, out_channels=32, kernel_size=3
        self.conv1 = nn.Conv1d(in_channels=feature_dim, out_channels=32, kernel_size=3, padding='same')
        # MaxPool1d
        self.pool = nn.MaxPool1d(kernel_size=2)
        # Conv1d layer 2: in_channels=32, out_channels=64, kernel_size=2
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, padding='same')
        # Fully connected layers
        self.fc1 = nn.Linear(flatten_size, 128)
        self.fc2 = nn.Linear(128, output_window * output_dim)
        self.relu = nn.ReLU()
```

## Key Transformations

### 1. Framework Conversion (TensorFlow/Keras to PyTorch)
| Keras Layer | PyTorch Equivalent |
|-------------|-------------------|
| `Conv1D` | `nn.Conv1d` |
| `MaxPooling1D` | `nn.MaxPool1d` |
| `Flatten` | `tensor.flatten(start_dim=1)` |
| `Dense` | `nn.Linear` |
| `activation='relu'` | `nn.ReLU()` |

### 2. Input Dimension Handling
- **Keras Conv1D**: Input shape is `(batch, timesteps, features)` - channels last
- **PyTorch nn.Conv1d**: Input shape is `(batch, channels, sequence_length)` - channels first
- **Transformation**: Permute dimensions from `(B, T, F)` to `(B, F, T)` before convolution

### 3. LibCity Data Format Adaptation
- **LibCity Input**: `batch['X']` with shape `(batch, time_in, num_nodes, features)`
- **Model Processing**:
  1. Permute to `(batch, num_nodes, time_in, features)`
  2. Reshape to `(batch * num_nodes, time_in, features)`
  3. Permute to `(batch * num_nodes, features, time_in)` for Conv1d
- **Output Transformation**: Reshape and permute back to `(batch, time_out, num_nodes, output_dim)`

### 4. Loss Function Adaptation
- **Original**: Simple MSE loss
- **Adapted**: LibCity's `masked_mse_torch` (with NaN handling) or `masked_mae_torch`
- **Configurable**: Set via `loss_type` config parameter ('mse' or 'mae')

## Configuration Parameters

### Model Architecture
| Parameter | Default | Description |
|-----------|---------|-------------|
| `conv1_filters` | 32 | Number of filters in first Conv1D layer |
| `conv1_kernel` | 3 | Kernel size for first Conv1D layer |
| `conv2_filters` | 64 | Number of filters in second Conv1D layer |
| `conv2_kernel` | 2 | Kernel size for second Conv1D layer |
| `pool_size` | 2 | MaxPool1D pool size |
| `hidden_size` | 128 | Size of hidden dense layer |
| `loss_type` | 'mse' | Loss function type ('mse' or 'mae') |

### Training Parameters (in config JSON)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_window` | 12 | Number of input time steps |
| `output_window` | 12 | Number of output time steps |
| `learning_rate` | 0.001 | Learning rate |
| `max_epoch` | 100 | Maximum training epochs |
| `patience` | 10 | Early stopping patience |

## Assumptions and Limitations

1. **Input Window Size**: The model assumes `input_window` is divisible by `pool_size` (default 2). If not, the pooling layer will truncate the sequence.

2. **Feature Dimension**: The original model was designed for 4 specific bus features. The adapted version uses LibCity's `feature_dim` from data_feature, making it more flexible.

3. **Node Independence**: The model processes each node independently through the CNN layers, then aggregates results. This may not capture inter-node spatial dependencies.

4. **PyTorch Version**: The model uses `padding='same'` which requires PyTorch >= 1.9.

5. **Scaler Dependency**: The model relies on LibCity's scaler for normalization/denormalization in loss calculation.

## Usage Example

```python
# In LibCity config file or script
config = {
    "model": "FCNNBus",
    "dataset": "YOUR_DATASET",
    "input_window": 12,
    "output_window": 12,
    "conv1_filters": 32,
    "conv2_filters": 64,
    "hidden_size": 128,
    "loss_type": "mse"
}
```

## Testing

To test the model import:
```python
from libcity.model.traffic_speed_prediction import FCNNBus
```

## References

- Original FCNNBus Repository: `/home/wangwenrui/shk/AgentCity/repos/FCNNBus`
- LibCity Framework: https://github.com/LibCity/Bigscity-LibCity
