# MVSTM (Multi-View Spatial-Temporal Model) Migration Summary

## Overview

This document summarizes the migration of the MVSTM (Multi-View Spatial-Temporal Model) from its original PyTorch notebook implementation to the LibCity framework for Travel Time Estimation (ETA) tasks.

## Original Implementation

**Source Repository**: `/home/wangwenrui/shk/AgentCity/repos/MVSTM/DIDI_pytorch_code_1252`

**Source File**: `model.ipynb` (Jupyter notebook containing `CombineModel` class)

**Original Architecture**:
- Embedding layers: link (20), driver (20), time slice (20), weekday (3), weather (3)
- LSTM: input_dim=23, hidden_dim=128
- MLP: 175 -> 256 -> 128 -> 1 with LeakyReLU activations
- Loss: L1 Loss (MAE) on log-transformed normalized predictions

## Files Created/Modified

### New Files

1. **Model File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/MVSTM.py`
   - MVSTM class inheriting from `AbstractTrafficStateModel`
   - Implements `forward()`, `predict()`, and `calculate_loss()` methods
   - Handles variable-length link sequences using `pack_padded_sequence`

2. **Encoder File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/eta_encoder/mvstm_encoder.py`
   - MVSTMEncoder class for trajectory data preprocessing
   - Extracts link sequences, order features, and temporal features
   - Generates normalization statistics for training

3. **Config File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/eta/MVSTM.json`
   - Model hyperparameters and default normalization statistics

### Modified Files

1. **Model __init__.py**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/__init__.py`
   - Added MVSTM import and export

2. **Encoder __init__.py**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/eta_encoder/__init__.py`
   - Added MVSTMEncoder import and export

3. **Task Config**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
   - Added MVSTM to allowed models for ETA task
   - Added MVSTM task configuration entry

## Key Transformations

### 1. Class Structure
- Original: Standalone `CombineModel(nn.Module)` class
- Migrated: `MVSTM(AbstractTrafficStateModel)` with LibCity interface

### 2. Data Handling
- Original: Custom collate function with msgpack data format
- Migrated: LibCity batch dictionary format with ETADataset integration

### 3. Feature Processing
The model processes multiple views of data:

**Spatial View (Link Sequences)**:
- Link IDs embedded to 20 dimensions
- Link time, ratio, and current status as additional features
- Variable-length sequences handled with packed LSTM

**Temporal View**:
- Time slice (5-minute intervals, 288 per day) embedded to 20 dimensions
- Weekday embedded to 3 dimensions

**Contextual View**:
- Driver ID embedded to 20 dimensions
- Weather embedded to 3 dimensions (not used in MLP output in original)
- Temperature features normalized and included in numerical features

### 4. Normalization
- Log transformation applied to distance, ETA, and travel time
- Z-score normalization using mean/std statistics
- Link time normalized using min-max scaling

### 5. Loss Function
- L1 Loss (MAE) on normalized log-scale predictions
- Same as original implementation for training stability

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| num_links | 50000 | Maximum number of unique links |
| num_drivers | 10000 | Maximum number of unique drivers |
| num_time_slices | 288 | Time slices per day (5-min intervals) |
| link_emb_dim | 20 | Link embedding dimension |
| driver_emb_dim | 20 | Driver embedding dimension |
| slice_emb_dim | 20 | Time slice embedding dimension |
| weekday_emb_dim | 3 | Weekday embedding dimension |
| weather_emb_dim | 3 | Weather embedding dimension |
| lstm_hidden_dim | 128 | LSTM hidden state dimension |
| lstm_num_layers | 1 | Number of LSTM layers |
| mlp_hidden_dims | [256, 128] | MLP hidden layer dimensions |
| use_log_transform | true | Apply log transformation to targets |

## Usage

```python
# Example usage with LibCity
from libcity.pipeline import run_model

run_model(
    task='eta',
    dataset='Chengdu_Taxi_Sample1',
    model='MVSTM',
    config_file=None  # Uses default config
)
```

## Batch Format

The model expects batches with the following keys:
- `link_ids`: (batch, seq_len) - Link sequence IDs
- `link_time`: (batch, seq_len) - Time spent on each link
- `link_ratio`: (batch, seq_len) - Ratio of link traversed
- `link_current_status`: (batch, seq_len) - Traffic status on link
- `link_len`: (batch,) - Actual sequence lengths
- `dist`: (batch,) - Total trip distance
- `simple_eta`: (batch,) - Simple ETA estimate
- `driver_id`: (batch,) - Driver identifier
- `slice_id`: (batch,) - Time slice identifier
- `weekday`: (batch,) - Day of week
- `weather`: (batch,) - Weather condition
- `high_temp`: (batch,) - High temperature
- `low_temp`: (batch,) - Low temperature
- `eta` or `time`: (batch,) - Ground truth travel time

## Assumptions and Limitations

1. **Link Mapping**: The encoder creates a link-to-ID mapping on the fly. For production use, a pre-built mapping should be provided.

2. **Weather Data**: If weather data is not available in the dataset, default values are used.

3. **Temperature Units**: Assumes Celsius for temperature features.

4. **Distance Calculation**: If link-level distances are not available, geodesic distance is calculated from coordinates.

5. **Original Data Format**: The original implementation used msgpack format with specific feature names. The encoder handles the translation to LibCity format.

## Performance Notes

- The model uses `pack_padded_sequence` for efficient variable-length sequence processing
- Mixed precision training is supported (as in original)
- Exponential LR scheduler with gamma=0.98 recommended
