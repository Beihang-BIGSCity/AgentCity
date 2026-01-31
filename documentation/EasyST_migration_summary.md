# EasyST Configuration Migration Summary

## Model Overview
**EasyST: An Easy and Efficient Spatio-Temporal Learning Framework for Traffic Forecasting**

EasyST implements a Dynamic MLP with Information Bottleneck (DMLP-IB) architecture for traffic state prediction. The model combines spatial and temporal embeddings with a variational information bottleneck to achieve robust and efficient spatio-temporal forecasting.

### Paper Reference
- **Paper**: "EasyST: A Simple Framework for Spatio-Temporal Prediction" (CIKM 2024)
- **Original Repository**: https://github.com/HKUDS/EasyST
- **Task**: Traffic State Prediction (traffic_state_pred)
- **Model Category**: traffic_speed_prediction

## Key Architecture Components

1. **Time Series Embedding**: Embeds input sequences using Conv2d
2. **Spatial Embedding**: Learnable node embeddings for each sensor/location
3. **Temporal Embedding**:
   - Time-of-day embeddings
   - Day-of-week embeddings
4. **Dynamic Node Encoding (DNE)**: Time-varying node representations via temporal indices
5. **Information Bottleneck (IB)**: Variational IB with reparameterization (z = mu + eps * sigma)
6. **Stochastic Inference**: Multiple samples for prediction averaging

## Configuration Migration

### 1. task_config.json Registration
**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`

**Added to**: `traffic_state_pred.allowed_model`
- **Line number**: 259
- **Position**: After CKGGNN

**Model Configuration Section**:
```json
"EasyST": {
    "dataset_class": "TrafficStatePointDataset",
    "executor": "TrafficStateExecutor",
    "evaluator": "TrafficStateEvaluator"
}
```
- **Line numbers**: 820-824

### 2. Model Configuration File
**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traffic_state_pred/EasyST.json`

**Configuration Parameters** (from paper and implementation):

#### Core Model Parameters
- **embed_dim**: 32 (time series embedding dimension)
- **node_dim**: 32 (spatial embedding dimension)
- **temp_dim_tid**: 32 (time-of-day embedding dimension)
- **temp_dim_diw**: 32 (day-of-week embedding dimension)
- **num_layer**: 3 (number of MLP encoder layers)
- **dropout**: 0.15 (dropout rate)

#### Feature Control Flags
- **if_T_i_D**: true (use time-of-day embeddings)
- **if_D_i_W**: true (use day-of-week embeddings)
- **if_spatial**: true (use spatial node embeddings)
- **if_dne**: true (use Dynamic Node Encoding)

#### Dynamic Node Encoding (DNE)
- **mid_dim**: 32 (DNE intermediate dimension)
- **dne_act**: "softplus" (DNE activation function)
  - Options: softplus, relu, leakyrelu, sigmoid, softmax, none

#### Information Bottleneck (IB)
- **beta_ib**: 0.001 (IB loss weight)
- **n_sample_train**: 1 (number of samples during training)
- **n_sample_predict**: 12 (number of samples for stochastic inference during prediction)

#### Window Configuration
- **input_window**: 12 (input sequence length)
- **output_window**: 12 (output sequence length)

#### Data Processing
- **scaler**: "standard" (data normalization method)
- **load_external**: true (load external features)
- **normal_external**: false (normalize external features)
- **ext_scaler**: "none" (external feature scaler)
- **add_time_in_day**: true (add time-in-day features)
- **add_day_in_week**: true (add day-of-week features)

#### Training Configuration
- **max_epoch**: 200
- **learner**: "adam"
- **learning_rate**: 0.001
- **lr_decay**: true
- **lr_scheduler**: "multisteplr"
- **lr_decay_ratio**: 0.5
- **steps**: [50, 100, 150] (learning rate decay steps)
- **clip_grad_norm**: true
- **max_grad_norm**: 5
- **use_early_stop**: true
- **patience**: 20

### 3. Model Implementation
**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_speed_prediction/EasyST.py`

**Status**: ✅ Already implemented

**Registered in**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_speed_prediction/__init__.py`
- Import statement: Line 43
- __all__ export: Line 112

## Dataset Compatibility

### Compatible Datasets
EasyST uses `TrafficStatePointDataset` and is compatible with all standard LibCity traffic datasets:

#### Primary Datasets
- **METR_LA** (Los Angeles Metropolitan Traffic)
- **PEMS_BAY** (Bay Area PeMS Traffic)
- **PEMSD3, PEMSD4, PEMSD7, PEMSD8** (California PeMS datasets)

#### Additional Compatible Datasets
- LOOP_SEATTLE
- LOS_LOOP, LOS_LOOP_SMALL
- Q_TRAFFIC
- SZ_TAXI
- BEIJING_SUBWAY_* (10MIN, 15MIN, 30MIN)
- ROTTERDAM
- HZMETRO, SHMETRO
- PORTO
- And all other traffic_state_pred datasets listed in task_config.json

### Required Data Features
The model expects:
1. **Spatial Structure**: Node/sensor locations (N nodes)
2. **Temporal Features**:
   - Time-of-day (normalized to [0, 1])
   - Day-of-week (one-hot encoded or integer)
3. **Traffic Data**: Speed/flow measurements
4. **Feature Dimension**: Minimum 2 channels (data + time-in-day)
   - Channel 0: Traffic data (speed/flow)
   - Channel 1: Time-in-day
   - Channels 2-8: Day-of-week (one-hot) [optional]

### Data Format
- **Input**: [Batch, Time, Nodes, Features]
- **Output**: [Batch, Time, Nodes, 1]

## Implementation Notes

### Adaptation from Original Code
1. **LibCity Integration**: Converted from standalone script to `AbstractTrafficStateModel`
2. **MultiLayerPerceptron Module**: Included inline (was missing dependency in original)
3. **Temporal Feature Extraction**: Adapted to extract time features from LibCity batch format
4. **Loss Function**: IB loss incorporated in `calculate_loss` method
5. **Stochastic Inference**: Different n_sample for training (1) vs prediction (12)

### Key Features
1. **Information Bottleneck**: Regularizes representations via KL divergence
   - Formula: `KL(q(z|x) || p(z))` where `p(z) = N(0, 1)`
   - Loss: `-0.5 * (1 + 2*log(std) - mu^2 - std^2).sum(1).mean() / log(2)`

2. **Dynamic Node Encoding**: Time-varying spatial representations
   - Uses einsum operations for efficient computation
   - Combines temporal indices with node embeddings

3. **Reparameterization Trick**: For variational inference
   - `z = mu + eps * std` where `eps ~ N(0, 1)`

### Model Strengths
- **Simplicity**: Pure MLP-based, no complex graph convolutions
- **Efficiency**: Fast training and inference
- **Robustness**: Information bottleneck provides regularization
- **Flexibility**: Feature flags allow ablation studies

### Potential Limitations
1. **Spatial Modeling**: Relies on learnable embeddings rather than explicit graph structure
2. **Memory**: DNE parameters grow with time-of-day size and number of nodes
3. **Temporal Resolution**: Fixed time interval discretization (default: 5 minutes)

## Usage Example

```python
from libcity.pipeline import run_model

# Run with default configuration
run_model(
    task='traffic_state_pred',
    model='EasyST',
    dataset='METR_LA',
    config_file=None  # Uses default config
)

# Run with custom configuration
run_model(
    task='traffic_state_pred',
    model='EasyST',
    dataset='PEMS_BAY',
    config_file='custom_config',
    other_args={
        'embed_dim': 64,
        'num_layer': 4,
        'beta_ib': 0.002,
        'input_window': 24,
        'output_window': 12
    }
)
```

## Hyperparameter Tuning Recommendations

### Critical Parameters
1. **beta_ib** (0.0001 - 0.01): Controls IB regularization strength
   - Lower: More expressive representations
   - Higher: More compressed representations

2. **num_layer** (2-5): MLP depth
   - Deeper: More capacity but slower
   - Shallower: Faster but less expressive

3. **embed_dim, node_dim** (16-128): Embedding dimensions
   - Larger datasets: Higher dimensions
   - Smaller datasets: Lower dimensions to prevent overfitting

### For Different Dataset Sizes
- **Small datasets** (< 100 nodes):
  - embed_dim=16-32, num_layer=2-3, beta_ib=0.001-0.01
- **Medium datasets** (100-300 nodes):
  - embed_dim=32-64, num_layer=3-4, beta_ib=0.0001-0.001
- **Large datasets** (> 300 nodes):
  - embed_dim=64-128, num_layer=4-5, beta_ib=0.0001

## Verification Checklist

- [x] Model file exists at correct path
- [x] Model registered in `traffic_speed_prediction/__init__.py`
- [x] Added to `task_config.json` allowed_model list (line 259)
- [x] Model configuration section added to task_config.json (lines 820-824)
- [x] Model configuration file exists with all parameters
- [x] Configuration parameters match paper specifications
- [x] Dataset compatibility verified
- [x] Uses standard TrafficStatePointDataset
- [x] Uses standard TrafficStateExecutor and Evaluator
- [x] Documentation created

## Migration Status: ✅ COMPLETE

All configuration tasks have been successfully completed. EasyST is now fully integrated into LibCity's task configuration system and ready for use with all compatible traffic datasets.

---

**Last Updated**: 2026-01-30
**Migrated by**: Configuration Migration Agent
**LibCity Version**: Latest (shk branch)
