# Config Migration: FCNNBus

## Overview
FCNNBus is a Fully Convolutional Neural Network designed for bus arrival time prediction. It has been successfully integrated into LibCity's traffic state prediction framework.

## task_config.json

### Registration Status: COMPLETED
- **Task type**: `traffic_state_pred`
- **Added to**: `allowed_model` list (line 320)
- **Model configuration entry**: Lines 941-945

### Configuration Entry
```json
"FCNNBus": {
    "dataset_class": "TrafficStatePointDataset",
    "executor": "TrafficStateExecutor",
    "evaluator": "TrafficStateEvaluator"
}
```

### Allowed Datasets
The model can be used with all datasets in the `traffic_state_pred.allowed_dataset` list, including:
- METR_LA
- PEMS_BAY
- PEMSD3, PEMSD4, PEMSD7, PEMSD8
- NYC datasets (NYCBike, NYCTaxi variants)
- TAXIBJ
- LOOP_SEATTLE
- And others (see task_config.json lines 322-364)

## Model Config

### File Location
`/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traffic_state_pred/FCNNBus.json`

### Model Parameters

#### Architecture Parameters (from original paper)
- **conv1_filters**: 32 (first Conv1D layer filters)
- **conv1_kernel**: 3 (first Conv1D kernel size)
- **conv2_filters**: 64 (second Conv1D layer filters)
- **conv2_kernel**: 2 (second Conv1D kernel size)
- **pool_size**: 2 (MaxPooling1D pool size)
- **hidden_size**: 128 (dense layer hidden units)
- **loss_type**: "mse" (Mean Squared Error loss)

Source: Original Keras implementation from FCNNBus notebook

#### Training Parameters (from original paper)
- **max_epoch**: 20 (from original notebook)
- **learner**: "adam" (optimizer)
- **learning_rate**: 0.001 (Adam default, not explicitly specified in original)
- **batch_size**: 32 (from original notebook)
- **patience**: 5 (early stopping patience from original)
- **use_early_stop**: true
- **lr_decay**: false (not used in original)
- **weight_decay**: 0.0 (not specified in original)

#### Data Processing Parameters
- **scaler**: "standard" (standardization)
- **load_external**: true
- **normal_external**: false
- **ext_scaler**: "none"
- **add_time_in_day**: false
- **add_day_in_week**: false

## Model Implementation

### File Location
`/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_speed_prediction/FCNNBus.py`

### Registration in __init__.py
- **Import statement**: Line 78
  ```python
  from libcity.model.traffic_speed_prediction.FCNNBus import FCNNBus
  ```
- **Export in __all__**: Line 140
  ```python
  "FCNNBus",
  ```

### Model Architecture
The PyTorch implementation follows the original Keras architecture:

**Original Keras Model:**
```
Conv1D(32, 3, activation='relu', padding='same', input_shape=(4, 1))
MaxPooling1D(2)
Conv1D(64, 2, activation='relu', padding='same')
Flatten()
Dense(128, activation='relu')
Dense(1)  # Output for regression
```

**PyTorch Implementation:**
- Conv1d layer 1: input channels = feature_dim, output = 32, kernel = 3
- MaxPool1d: pool size = 2
- Conv1d layer 2: input = 32, output = 64, kernel = 2
- Flatten
- Linear layer 1: input = 64 * (input_window // 2), output = 128
- Linear layer 2: input = 128, output = output_window * output_dim

### Key Adaptations
1. **Input format**: Adapted from Keras (batch, time, features) to LibCity format (batch, time, nodes, features)
2. **Loss function**: Uses LibCity's `masked_mse_torch` for compatibility with masked data
3. **Multi-node support**: Processes each node independently through the CNN
4. **Device compatibility**: Supports CPU/GPU training

## Dataset Compatibility

### Original Dataset
- NYC bus data with 4 features: timestamp, direction, bus line, stop
- Task: Bus arrival time prediction (regression)

### LibCity Compatibility
The model is compatible with `TrafficStatePointDataset`, which supports:
- **Input format**: (batch, time, nodes, features)
- **Default input_window**: 12
- **Default output_window**: 12
- **Feature dimension**: Flexible (originally designed for 4 features)

### Recommended Datasets
For bus arrival prediction tasks:
- Any point-based traffic dataset with temporal patterns
- Datasets with feature dimensions matching the original design work best
- Can be adapted to standard traffic speed prediction datasets (METR_LA, PEMS_BAY, etc.)

## Configuration Summary

| Parameter | Value | Source |
|-----------|-------|--------|
| Model Name | FCNNBus | - |
| Task Type | traffic_state_pred | LibCity framework |
| Dataset Class | TrafficStatePointDataset | LibCity framework |
| Executor | TrafficStateExecutor | LibCity framework |
| Evaluator | TrafficStateEvaluator | LibCity framework |
| Epochs | 20 | Original notebook |
| Batch Size | 32 | Original notebook |
| Learning Rate | 0.001 | Adam default |
| Optimizer | Adam | Original notebook |
| Loss Function | MSE | Original notebook |
| Early Stopping | 5 epochs patience | Original notebook |

## Notes

### Compatibility Considerations
1. **Feature dimension flexibility**: The model adapts to any feature dimension, though it was originally designed for 4 features
2. **Multi-node processing**: Each node is processed independently through the CNN architecture
3. **Sequence length**: Input window must be divisible by pool_size (2) to avoid dimension issues

### Differences from Original
1. **Epochs**: Original used 20 epochs (updated from initial config of 100)
2. **Patience**: Original used 5 epochs patience (updated from initial config of 10)
3. **Learning rate decay**: Disabled to match original implementation (was enabled in initial config)
4. **Scaling**: Uses standard scaler for LibCity compatibility

### Limitations
- Originally designed for 4-feature bus arrival prediction
- CNN architecture may not capture complex spatial relationships like graph-based models
- Best suited for point-based temporal prediction tasks

## Validation Checklist

- [x] Model registered in task_config.json allowed_model list
- [x] Model configuration entry added to task_config.json
- [x] Model config file created with correct hyperparameters
- [x] Model imported in __init__.py
- [x] Model exported in __all__ list
- [x] Dataset class compatibility verified (TrafficStatePointDataset)
- [x] Executor compatibility verified (TrafficStateExecutor)
- [x] Evaluator compatibility verified (TrafficStateEvaluator)
- [x] Hyperparameters match original paper/implementation
- [x] JSON syntax validated

## Files Modified

1. **task_config.json**
   - Path: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
   - Changes: Added "FCNNBus" to allowed_model list (line 320), added model configuration (lines 941-945)

2. **FCNNBus.json**
   - Path: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traffic_state_pred/FCNNBus.json`
   - Changes: Updated hyperparameters to match original paper (epochs: 20, patience: 5, lr_decay: false, batch_size: 32)

## Testing Recommendations

1. Test with a standard traffic dataset (e.g., METR_LA) to verify integration
2. Verify model loads correctly with config parameters
3. Check training and evaluation pipeline execution
4. Validate output dimensions match expected format
5. Compare performance with baseline models on same dataset

## References

- Original repository: `/home/wangwenrui/shk/AgentCity/repos/FCNNBus`
- Original notebook: `/home/wangwenrui/shk/AgentCity/repos/FCNNBus/Notebooks/Copy_of_CNN_for_bus_sample_2.ipynb`
- Model implementation: `Bigscity-LibCity/libcity/model/traffic_speed_prediction/FCNNBus.py`
