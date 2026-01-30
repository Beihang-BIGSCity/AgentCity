## Config Migration: DutyTTE

### Overview
DutyTTE (Mixture of Experts with Uncertainty Quantification for Travel Time Estimation) has been successfully configured for LibCity framework. This document verifies all configuration components.

### 1. task_config.json Entry
**Status: VERIFIED**

- **Location**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
- **Task Type**: `eta` (Estimated Time of Arrival)
- **Line Number**: 783
- **Entry**:
```json
"DutyTTE": {
    "dataset_class": "ETADataset",
    "executor": "ETAExecutor",
    "evaluator": "ETAEvaluator",
    "eta_encoder": "DutyTTEEncoder"
}
```
- Added to `allowed_model` list at line 783

### 2. Model Configuration
**Status: UPDATED AND VERIFIED**

- **Location**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/eta/DutyTTE.json`

#### Configuration Parameters (with Paper References)

**Model Architecture Parameters:**
- `segment_dims`: 12693 (from dataset - number of road segments)
- `node_dims`: 4601 (from dataset - number of nodes/intersections)
- `id_embed_dim`: 20 (embedding dimension for segment/node IDs)
- `slice_dims`: 145 (number of time slices - 10min granularity)
- `slice_embed_dim`: 20 (embedding dimension for time slices)
- `hidden_size`: 256 (UPDATED from 128 to 256 - matches E_U in paper)
- `num_experts`: 8 (VERIFIED - matches C=8 in paper)
- `top_k`: 4 (UPDATED from 2 to 4 - matches k=4 in paper)
- `n_embed`: 256 (UPDATED from 128 to 256 - embedding dimension for distribution features)
- `m`: 5 (VERIFIED - matches m=5 in paper, number of statistical travel times)
- `alpha`: 0.1 (VERIFIED - matches rho=0.1 in paper, MIS loss parameter)
- `load_balance_weight`: 0.01 (weight for expert load balancing loss)
- `dropout`: 0.1 (dropout rate for expert networks)

**Training Parameters:**
- `max_epoch`: 100
- `batch_size`: 128 (VERIFIED - matches paper)
- `learner`: "adam"
- `learning_rate`: 0.001 (VERIFIED - matches paper)
- `weight_decay`: 0.00001
- `lr_decay`: true
- `lr_scheduler`: "ReduceLROnPlateau"
- `lr_decay_ratio`: 0.5
- `lr_patience`: 5
- `clip_grad_norm`: true
- `max_grad_norm`: 5.0
- `use_early_stop`: true
- `patience`: 20 (VERIFIED - matches early_stop=20 in paper)

**Other Parameters:**
- `model`: "DutyTTE"
- `output_pred`: false

#### Critical Updates Made:
1. **top_k**: Changed from 2 to 4 (paper specifies k=4 experts selected)
2. **hidden_size**: Changed from 128 to 256 (paper specifies E_U=256)
3. **n_embed**: Changed from 128 to 256 (should match hidden_size for consistency)

### 3. Model Registration
**Status: VERIFIED**

- **Model File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/DutyTTE.py`
- **Registered in**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/__init__.py` (line 6)
- **Parent Class**: `AbstractTrafficStateModel`
- **Key Methods Implemented**:
  - `__init__`: Model initialization with all components
  - `forward`: Forward pass with MoE and multi-branch prediction
  - `predict`: Standard prediction interface (returns mean)
  - `predict_with_uncertainty`: Returns prediction with bounds
  - `calculate_loss`: Custom MIS loss + load balancing loss

### 4. Encoder Registration
**Status: VERIFIED**

- **Encoder File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/eta_encoder/dutytte_encoder.py`
- **Registered in**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/eta_encoder/__init__.py` (line 6, 14)
- **Parent Class**: `AbstractETAEncoder`

#### Encoder Features:
The DutyTTEEncoder prepares the following features:
- `segments`: Road segment IDs [seq_len]
- `segment_travel_time`: Distribution features [seq_len, m*2+1]
  - First m values: histogram of travel time distribution
  - Next m values: cumulative distribution
  - Last value: count/normalization factor
- `num_segments`: Number of segments in trajectory [1]
- `start_time`: Start time bucket (10-minute granularity) [1]
- `origin`: Origin node ID [1]
- `destination`: Destination node ID [1]
- `time`: Total travel time (ground truth) [1]
- `traj_len`: Trajectory length [1]
- `traj_id`: Trajectory ID [1]
- `uid`: User ID [1]

### 5. Dataset Compatibility
**Status: VERIFIED**

**Allowed Datasets** (from task_config.json):
- `Chengdu_Taxi_Sample1`
- `Beijing_Taxi_Sample`

**Required Data Features**:
- `coordinates`: GPS coordinates (required)
- `time`: Timestamp (required)
- `segment_id`: Road segment ID (optional - can be computed from coordinates)
- `node_id`: Node/intersection ID (optional - can be computed from coordinates)
- `traj_id`: Trajectory ID (optional)

**Dataset Class**: `ETADataset` (configured in task_config.json)

**Dataset Configuration** (from ETADataset.json):
- `batch_size`: 10 (overridden by model config to 128)
- `min_session_len`: 5
- `max_session_len`: 50
- `cache_dataset`: true
- `train_rate`: 0.7
- `eval_rate`: 0.1

### 6. Model Architecture Summary

**Components**:
1. **Embeddings**:
   - Segment embeddings (segment_dims -> id_embed_dim)
   - Node embeddings (node_dims -> id_embed_dim)
   - Time slice embeddings (slice_dims -> slice_embed_dim)
   - Distribution embeddings (m*2+1 -> n_embed)

2. **Feature Fusion**:
   - Deep pathway: MLP for OD + time features
   - Recurrent pathway: LSTM for sequence modeling

3. **Sparse Mixture of Experts (SparseMoE)**:
   - NoisyTopkRouter: Selects top_k experts with noise
   - Expert networks: 8 experts, each with 4x expansion ratio
   - Load balancing: Encourages even expert utilization

4. **Multi-branch Regressors**:
   - Mean regressor: Point prediction
   - Lower regressor: Lower bound offset
   - Upper regressor: Upper bound offset

5. **Loss Function**:
   - Mean Interval Score (MIS): Balances interval width and coverage
   - Load balancing loss: Prevents expert collapse
   - Point prediction loss: MAE on mean prediction

### 7. Hyperparameter Mapping

| Original Code | Paper Term | LibCity Config | Value |
|---------------|------------|----------------|-------|
| E_U | Embedding dimension | hidden_size | 256 |
| C | Number of experts | num_experts | 8 |
| k | Selected experts | top_k | 4 |
| m | Distribution params | m | 5 |
| batch_size | Batch size | batch_size | 128 |
| lr | Learning rate | learning_rate | 0.001 |
| rho | MIS weight | alpha | 0.1 |
| early_stop | Early stopping | patience | 20 |

### 8. Compatibility Notes

**No Issues Found**:
- All required encoders are properly registered
- Model inherits from correct abstract class
- Dataset class is compatible with ETA task
- Encoder produces all required features for the model
- Loss function is custom implemented (not using standard LibCity losses)

**Special Features**:
1. **Uncertainty Quantification**: Model provides prediction intervals in addition to point predictions
2. **Custom Loss**: Uses Mean Interval Score (MIS) for proper uncertainty calibration
3. **Load Balancing**: MoE includes load balancing loss to prevent expert collapse
4. **Adaptive Encoding**: Encoder can handle datasets with or without segment_id/node_id fields

### 9. Usage Example

```python
# Run with default configuration
python run_model.py --task eta --model DutyTTE --dataset Chengdu_Taxi_Sample1

# Override specific parameters
python run_model.py --task eta --model DutyTTE --dataset Beijing_Taxi_Sample \
    --batch_size 64 --learning_rate 0.0005 --max_epoch 50

# Get predictions with uncertainty
# In model code:
result = model.predict_with_uncertainty(batch)
# Returns: {'prediction': [...], 'lower_bound': [...], 'upper_bound': [...]}
```

### 10. Validation Checklist

- [x] Model added to task_config.json allowed_model list
- [x] Model configuration file created with correct parameters
- [x] Model file implements required methods
- [x] Model registered in __init__.py
- [x] Encoder file created and functional
- [x] Encoder registered in __init__.py
- [x] Dataset compatibility verified
- [x] All hyperparameters match paper specifications
- [x] JSON syntax validated
- [x] No naming conflicts with existing models

### 11. Files Modified/Created

**Created**:
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/eta/DutyTTE.json`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/DutyTTE.py`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/eta_encoder/dutytte_encoder.py`

**Modified**:
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json` (added DutyTTE entry)
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/__init__.py` (added import)
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/eta_encoder/__init__.py` (added import)

### Summary

The DutyTTE model has been successfully configured for LibCity with all necessary components:

1. Configuration file created with paper-accurate hyperparameters
2. Critical parameters (top_k, hidden_size, n_embed) updated to match paper specifications
3. Model properly registered in task configuration
4. Custom encoder created to prepare required input features
5. All components properly registered in __init__ files
6. Dataset compatibility verified for both allowed datasets

The configuration is ready for training and evaluation.
