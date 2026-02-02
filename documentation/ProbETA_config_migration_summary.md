# Config Migration: ProbETA

## Model Information
- Name: ProbETA (Probabilistic Embedding-based Travel Time Estimation)
- Task: eta (Estimated Time of Arrival)
- Paper: Probabilistic approach to ETA prediction using road segment embeddings
- Original Code: /home/wangwenrui/shk/AgentCity/repos/ProbETA/Model/ProbETA/

## Migration Summary

### 1. task_config.json Updates

#### Added to allowed_model list
- Task: `eta`
- Line: 962-972 (allowed_model array)
- Status: COMPLETED

```json
"allowed_model": [
    "DeepTTE",
    "TTPNet",
    "MulT_TTE",
    "LightPath",
    "DOT",
    "DutyTTE",
    "MDTI",
    "HierETA",
    "HetETA",
    "ProbETA"  // ADDED
]
```

#### Added ProbETA Configuration Block
- Location: Lines 1025-1031 (after HetETA)
- Status: COMPLETED

```json
"ProbETA": {
    "dataset_class": "ETADataset",
    "executor": "ETAExecutor",
    "evaluator": "ETAEvaluator",
    "eta_encoder": "ProbETAEncoder"
}
```

### 2. Model Config File

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/eta/ProbETA.json`

**Status**: UPDATED

#### Hyperparameters (from original paper/code)

| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `embedding_dim` | 64 | main.py line 21 (`latent_embedding_dim`) | Dimension of road segment embeddings |
| `dropout_mean` | 0.9 | Original implementation | Dropout rate for mean prediction network |
| `dropout_cov` | 0.3 | Original implementation | Dropout rate for covariance network |
| `hidden_mean_1` | 72 | Model architecture | First hidden layer for mean network |
| `hidden_mean_2` | 64 | Model architecture | Second hidden layer for mean network |
| `hidden_mean_3` | 32 | Model architecture | Third hidden layer for mean network |
| `hidden_cov_1` | 32 | Model architecture | First hidden layer for covariance network |
| `hidden_cov_2` | 16 | Model architecture | Second hidden layer for covariance network |
| `use_device_similarity` | true | Original feature | Enable device ID similarity matrix |
| `loss_type` | "nll" | Model default | Negative log-likelihood loss |
| `enable_time_dis` | true | main.py line 20 | Enable time distance weighting |
| `cov_reg_weight` | 0.1 | Paper default | Covariance regularization weight |

#### Training Parameters

| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `max_epoch` | 100 | Standard | Maximum training epochs |
| `batch_size` | 128 | Adjusted from 64 | Batch size for training |
| `learning_rate` | 0.001 | main.py line 24 | Initial learning rate |
| `lr_decay` | true | main.py line 42 | Enable learning rate decay |
| `lr_decay_ratio` | 0.6 | main.py line 25 (`lr_decay_factor`) | LR decay factor |
| `lr_scheduler` | "reducelronplateau" | LibCity standard | Learning rate scheduler |
| `lr_patience` | 10 | Standard | Patience for LR reduction |
| `learner` | "adam" | main.py line 42 | Optimizer (Adam with amsgrad) |
| `weight_decay` | 0.0001 | Standard | L2 regularization |
| `use_early_stop` | true | Standard | Enable early stopping |
| `patience` | 20 | Standard | Early stopping patience |
| `train_loss` | "none" | Model custom | Use model's calculate_loss() method |

### 3. Model Implementation

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/ProbETA.py`

**Status**: ALREADY IMPLEMENTED

Key features:
- Inherits from `AbstractTrafficStateModel`
- Dual embedding layers for road segments
- Mean prediction network (4 FC layers with dropout)
- Covariance estimation network (3 FC layers with dropout)
- Multivariate Gaussian NLL loss
- Supports uncertainty quantification via `predict_with_uncertainty()`
- CRPS metric computation for probabilistic evaluation

### 4. Data Encoder

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/eta_encoder/probeta_encoder.py`

**Status**: CREATED

**Registration**: Updated `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/eta_encoder/__init__.py`

#### Encoder Features

```python
feature_dict = {
    'road_segments': 'int',   # Road segment ID sequence (PRIMARY INPUT)
    'device_ids': 'int',      # Device ID for similarity matrix
    'time_slots': 'int',      # Time slot ID
    'uid': 'int',             # User ID
    'weekid': 'int',          # Day of week (0-6)
    'timeid': 'int',          # Time of day in minutes
    'time': 'float',          # Travel time (LABEL)
    'traj_len': 'int',        # Trajectory length
    'traj_id': 'int',         # Trajectory ID
    'start_timestamp': 'int', # Start timestamp
    'current_longi': 'float', # Longitude (for visualization)
    'current_lati': 'float',  # Latitude (for visualization)
}
```

#### Padding Configuration
```python
pad_item = {
    'road_segments': 0,      # Pad road segments with 0
    'current_longi': 0.0,
    'current_lati': 0.0,
}
```

#### Data Features
```python
data_feature = {
    'traj_len_idx': <index>,    # Index of trajectory length
    'uid_size': <count>,        # Number of unique users
    'road_num': <count>,        # Number of road segments (CRITICAL)
}
```

### 5. Dataset Compatibility

#### Compatible Datasets
- `Chengdu_Taxi_Sample1`
- `Beijing_Taxi_Sample`

#### Required Data Format

**Input Features (.dyna file)**:
- `location`: Road segment IDs (integer) - **REQUIRED**
- `time`: Timestamp in ISO format - **REQUIRED**
- `entity_id`: User/entity ID - **REQUIRED**
- `traj_id`: Trajectory ID - **REQUIRED**
- `coordinates`: GPS coordinates (for visualization) - Optional
- `device_id`: Device identifier - Optional (uses uid if missing)
- `time_slot`: Time slot ID - Optional (computed if missing)

**Example Data Row**:
```csv
dyna_id,type,time,entity_id,traj_id,location,coordinates,current_dis,...
0,trajectory,2013-10-08T17:45:00Z,254,0,5752,"[116.318726,40.009014]",0.0,...
```

#### Data Processing
1. Encoder extracts road segment sequences from `location` field
2. Computes temporal features (weekid, timeid) from timestamp
3. Travel time computed as time difference between first and last point
4. Device IDs used for similarity matrix (defaults to user ID)
5. Time slots computed from hour (48 slots per day)

### 6. Executor and Evaluator

#### Executor
- Class: `ETAExecutor`
- Location: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/executor/eta_executor.py`
- Loss Function: Uses model's `calculate_loss()` method (train_loss = "none")
- Features: Standard ETA training loop with batch processing

#### Evaluator
- Class: `ETAEvaluator`
- Location: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/evaluator/eta_evaluator.py`
- Supported Metrics:
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)
  - R2 (R-squared score)
  - EVAR (Explained Variance)

**Note**: CRPS (Continuous Ranked Probability Score) is computed in the model but not yet integrated into the standard evaluator.

### 7. Model Registration

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/__init__.py`

**Status**: ALREADY REGISTERED

```python
from libcity.model.eta.ProbETA import ProbETA

__all__ = [
    "DeepTTE", "TTPNet", "MulT_TTE", "LightPath",
    "DOT", "DutyTTE", "MDTI", "HierETA", "HetETA",
    "ProbETA",  # ALREADY PRESENT
]
```

### 8. Configuration Integration Checklist

- [x] Add ProbETA to `task_config.json` allowed_model list
- [x] Add ProbETA configuration block in `task_config.json`
- [x] Create/verify model config file `ProbETA.json`
- [x] Verify model implementation in `ProbETA.py`
- [x] Create ProbETA encoder `probeta_encoder.py`
- [x] Register encoder in `eta_encoder/__init__.py`
- [x] Verify model registration in `eta/__init__.py`
- [x] Document hyperparameter sources
- [x] Verify dataset compatibility
- [x] Document executor/evaluator usage

## Usage Example

### Running ProbETA

```bash
python run_model.py --task eta --model ProbETA --dataset Beijing_Taxi_Sample
```

### Configuration Override

```bash
python run_model.py --task eta --model ProbETA --dataset Beijing_Taxi_Sample \
    --embedding_dim 128 --batch_size 64 --learning_rate 0.0005
```

### Programmatic Usage

```python
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.executor import get_executor
from libcity.model import get_model

# Load config
config = ConfigParser(task='eta', model='ProbETA', dataset='Beijing_Taxi_Sample')

# Load dataset
dataset = get_dataset(config)

# Initialize model
model = get_model(config, dataset.get_data_feature())

# Create executor
executor = get_executor(config, model, dataset.get_data_feature())

# Train
executor.train(dataset)

# Evaluate
executor.evaluate(dataset)
```

## Key Differences from Other ETA Models

### 1. Input Representation
- **Other models**: GPS coordinates (longitude, latitude) sequences
- **ProbETA**: Road segment ID sequences

### 2. Output
- **Other models**: Point estimate of travel time
- **ProbETA**: Mean AND covariance matrix (full uncertainty quantification)

### 3. Loss Function
- **Other models**: MSE, MAE, or simple regression loss
- **ProbETA**: Multivariate Gaussian negative log-likelihood

### 4. Architecture
- **Other models**: Single prediction head
- **ProbETA**: Dual prediction heads (mean + covariance)

### 5. Embeddings
- **Other models**: Single embedding layer or none
- **ProbETA**: Two separate embedding layers for road segments

## Notes and Recommendations

### Critical Parameters
1. **road_num**: Must match the maximum road segment ID in the dataset. This is automatically detected by the encoder.
2. **embedding_dim**: Controls the capacity of road embeddings. Default 64 works well for most datasets.
3. **dropout_mean/dropout_cov**: High dropout (0.9/0.3) is intentional for regularization in this probabilistic model.

### Data Requirements
- Dataset MUST have `location` field containing road segment IDs
- Road segment IDs should be contiguous integers starting from 1 (0 is reserved for padding)
- If your dataset uses GPS coordinates only, you need to perform map matching first

### Performance Considerations
- Batch size affects covariance matrix computation (batch_size x batch_size matrix)
- Large batch sizes may cause memory issues due to covariance matrix
- Consider reducing batch size to 64 or 32 for large road networks

### Potential Issues

1. **Missing road_id field**: The encoder defaults to using `location` field. Ensure your dataset has road segment IDs.

2. **Device similarity**: If device IDs are not available, the encoder uses user IDs as a fallback. This may reduce model performance.

3. **CRPS metric**: The model computes CRPS internally, but it's not yet integrated into the standard evaluator. To use it, you'll need to call `model.compute_crps(batch)` manually.

4. **Normalization**: The model expects time normalization to be handled via `data_feature['time_mean']` and `data_feature['time_std']`. Ensure the encoder's `gen_scalar_data_feature()` is called during dataset preparation.

### Future Enhancements

1. **CRPS Evaluator**: Integrate CRPS metric into ETAEvaluator for proper probabilistic evaluation
2. **Uncertainty Calibration**: Add calibration metrics for predicted uncertainties
3. **Road Network Features**: Incorporate road network topology (currently only uses IDs)
4. **Multi-modal Support**: Extend to support both road IDs and GPS coordinates

## Testing Recommendations

### Basic Functionality Test
```bash
# Test with Beijing_Taxi_Sample dataset
python run_model.py --task eta --model ProbETA --dataset Beijing_Taxi_Sample \
    --max_epoch 5 --batch_size 32
```

### Parameter Sensitivity Test
```bash
# Test different embedding dimensions
for dim in 32 64 128; do
    python run_model.py --task eta --model ProbETA --dataset Beijing_Taxi_Sample \
        --embedding_dim $dim --max_epoch 10
done
```

### Uncertainty Evaluation
```python
# Test uncertainty predictions
model.eval()
with torch.no_grad():
    mean, var = model.predict_with_uncertainty(test_batch)
    crps = model.compute_crps(test_batch)
    print(f"Mean prediction: {mean.mean():.2f}")
    print(f"Avg uncertainty: {var.mean():.2f}")
    print(f"CRPS score: {crps:.4f}")
```

## File Locations Summary

### Configuration Files
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/eta/ProbETA.json`

### Model Implementation
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/ProbETA.py`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/__init__.py`

### Data Processing
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/eta_encoder/probeta_encoder.py`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/eta_encoder/__init__.py`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/eta_dataset.py`

### Execution Components
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/executor/eta_executor.py`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/evaluator/eta_evaluator.py`

## Hyperparameter Mapping Reference

| Original Code | LibCity Config | Notes |
|---------------|---------------|-------|
| `latent_embedding_dim` | `embedding_dim` | Main hyperparameter |
| `num_roads` | Auto-detected | Set via encoder from data |
| `lr` | `learning_rate` | Standard name |
| `lr_decay_factor` | `lr_decay_ratio` | LibCity naming |
| `batch_step` | Not used | LibCity handles batching differently |
| `num_epochs` | `max_epoch` | Standard name |
| N/A | `train_loss` | Set to "none" to use model loss |

## Migration Status: COMPLETE

All components successfully integrated into LibCity framework:
- Configuration: DONE
- Model: DONE (already existed)
- Encoder: DONE (newly created)
- Registration: DONE
- Documentation: DONE

The ProbETA model is now ready for use in the LibCity framework.
