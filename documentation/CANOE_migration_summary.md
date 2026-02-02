# CANOE Migration Summary

## Overview

**Paper**: Beyond Regularity: Modeling Chaotic Mobility Patterns for Next Location Prediction (WWW)

**Repository**: https://github.com/yuqian2003/CANOE

**Model**: CANOE (Chaotic Attentive Neural Oscillator for Enhanced Location Prediction)

**Task**: Trajectory Location Prediction

**Status**: Successfully Migrated ✓

**Migration Date**: February 2026

---

## 1. Model Description

CANOE is a novel location prediction model that introduces chaotic neural oscillators into attention mechanisms to model irregular human mobility patterns. The model leverages three key innovations:

1. **Chaotic Neural Oscillator**: Uses chaotic dynamics with excitatory and inhibitory signals to enhance attention mechanisms, capturing the chaotic nature of human mobility
2. **Tri-Pair Interaction Encoders**:
   - UserLocationPair: LDA-based topic modeling for user-location preferences
   - TimeUserPair: Attention-based temporal user preference modeling
   - LocationTimePair: Transformer-based spatio-temporal encoding
3. **Cross-Context Attentive Decoder**: Multi-head attention integrating pairwise features
4. **Multi-task Learning**: Simultaneous location, time, and ranking predictions

---

## 2. Migration Process Summary

The migration was completed in 4 phases with iterative debugging:

### Phase 1: Repository Analysis
- Cloned original CANOE repository from GitHub
- Analyzed model architecture and dependencies
- Identified key components to preserve:
  - Chaotic oscillator dynamics
  - Tri-pair interaction encoders
  - Multi-task learning framework

### Phase 2: Model Adaptation
- Created `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/CANOE.py`
- Inherited from `AbstractModel` base class
- Implemented required methods:
  - `__init__()`: Model initialization
  - `forward()`: Forward pass
  - `predict()`: Prediction for evaluation
  - `calculate_loss()`: Multi-task loss computation
- Added `_prepare_batch()` method to adapt LibCity batch format

### Phase 3: Configuration Setup
- Created model configuration: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/CANOE.json`
- Registered model in `__init__.py`
- Set up hyperparameters matching original paper

### Phase 4: Testing and Iteration
- Ran 3 major iterations to fix compatibility issues
- Debugged batch format mismatches
- Fixed tensor dimension issues
- Validated output shapes

---

## 3. Issues Encountered and Fixes

### Issue 1: Batch Key Access with `in` Operator
**Problem**: Code used `if 'key' in batch` which doesn't work with LibCity's Batch object

**Error**:
```
TypeError: argument of type 'Batch' is not iterable
```

**Fix**: Changed to check `batch.data` dictionary:
```python
# Before
if 'current_loc' in batch:

# After
if 'current_loc' in batch.data:
```

### Issue 2: Wrong Batch Key Names
**Problem**: Original code used different key names than LibCity's TrajectoryDataset

**Mapping**:
- `location_x` → `current_loc`
- `location_y` → `target`
- `hour` → `current_tim`
- `timeslot_y` → `target_tim`

**Fix**: Created `_prepare_batch()` method to normalize key names:
```python
def _prepare_batch(self, batch):
    """Convert LibCity batch format to CANOE's expected format"""
    loc_x = batch['current_loc']
    uid = batch['uid']
    hour_x = batch['current_tim']
    target = batch['target']
    # ... normalize and return standardized dictionary
```

### Issue 3: Dimension Mismatch in Combined Tensor
**Problem**: The `combined` tensor concatenation resulted in wrong dimensions for FC layer

**Error**:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (batch*seq, 80) and (64, num_locations)
```

**Root Cause**: When `topic_num > 0`, the combined tensor has 5 components (dim * 5 = 80), but when `topic_num = 0`, it has only 4 components (dim * 4 = 64)

**Fix**: Added conditional dimension calculation:
```python
# In _build_model()
if self.model_type == 'tc':
    if self.topic_num > 0:
        combined_dim = self.base_dim * 5  # 80
        fc_input_dim = self.base_dim * 5
    else:
        combined_dim = self.base_dim * 4  # 64
        fc_input_dim = self.base_dim * 4
```

### Issue 4: Output Shape Mismatch
**Problem**: Model output was sequence-level `[batch*seq, num_locations]` but evaluation expected trajectory-level `[batch, num_locations]`

**Fix**: Modified `predict()` to return last position prediction:
```python
def predict(self, batch):
    loc_output, _, _ = self.forward(batch)

    # Reshape to [batch_size, seq_len, num_locations]
    loc_output = loc_output.view(batch_size, seq_len, -1)

    # Return last position prediction
    last_pred = loc_output[:, -1, :]
    return F.log_softmax(last_pred, dim=-1)
```

---

## 4. Files Created/Modified

### Created Files

1. **Model File** (951 lines)
   - Path: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/CANOE.py`
   - Contains: Complete CANOE implementation with all components

2. **Configuration File**
   - Path: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/CANOE.json`
   - Contains: Model hyperparameters and training settings

### Modified Files

1. **Model Registration**
   - File: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py`
   - Changes:
     ```python
     from libcity.model.trajectory_loc_prediction.CANOE import CANOE

     __all__ = [
         # ... other models
         "CANOE",
         # ...
     ]
     ```

---

## 5. Test Results

### Final Successful Test Run

**Configuration**:
- Model: CANOE
- Dataset: foursquare_tky
- Batch Size: 32
- Max Epochs: 2
- Learning Rate: 0.005
- Embedding Dimension: 16
- Topic Number: 0 (LDA disabled)

**Training Progress**:
- Epoch 0: Loss = 11.97357, Eval Loss = 11.10415, Eval Acc = 0.07145
- Epoch 1: Loss = 10.47334, Eval Loss = 10.51624, Eval Acc = 0.08908

**Evaluation Metrics** (Best Epoch):

| Metric | @1 | @5 | @10 | @20 |
|--------|-------|-------|-------|-------|
| **Recall** | 0.0845 | 0.2545 | 0.3422 | 0.4216 |
| **ACC** | 0.0845 | 0.2545 | 0.3422 | 0.4216 |
| **F1** | 0.0845 | 0.0848 | 0.0622 | 0.0402 |
| **MRR** | 0.0845 | 0.1445 | 0.1562 | 0.1618 |
| **MAP** | 0.0845 | 0.1445 | 0.1562 | 0.1618 |
| **NDCG** | 0.0845 | 0.1718 | 0.2001 | 0.2203 |

**Overall MRR**: 0.1618

---

## 6. Usage Instructions

### Running CANOE in LibCity

1. **Prepare the dataset**:
   ```bash
   # LibCity will automatically process the dataset
   # Ensure dataset is in Bigscity-LibCity/raw_data/foursquare_tky/
   ```

2. **Create configuration file** (or use default CANOE.json):
   ```json
   {
       "dim": 16,
       "bandwidth": 1.0,
       "oscillator_type": "cnoa_tc",
       "at_type": "osc",
       "encoder_type": "trans",
       "topic_num": 0,
       "lambda_loc": 0.9,
       "lambda_time": 0.4,
       "lambda_rank": 0.6,
       "max_epoch": 100,
       "learner": "adam",
       "learning_rate": 0.005,
       "batch_size": 256
   }
   ```

3. **Run training**:
   ```bash
   cd Bigscity-LibCity
   python run_model.py --task traj_loc_pred --model CANOE --dataset foursquare_tky
   ```

4. **Custom configuration**:
   ```bash
   python run_model.py --task traj_loc_pred --model CANOE --dataset foursquare_tky \
       --config_file custom_config.json
   ```

---

## 7. Model Architecture Notes

### Key Components Preserved

1. **MultimodalContextualEmbedding**
   - Location, user, and time embeddings
   - Gaussian kernel smoothing for temporal features with periodicity
   - Bandwidth parameter controls smoothing strength

2. **Oscillator (Chaotic Neural Oscillator)**
   - Core innovation with excitatory (u) and inhibitory (v) signals
   - Parameters: a1, a2, b1, b2 (connection weights), k (decay), n (iterations)
   - Different parameter sets for TC (Traffic Camera) vs MP (Mobile Phone) datasets
   - Modulates attention scores through chaotic dynamics

3. **UserLocationPair**
   - LDA-based topic modeling (optional)
   - MLP with residual connection
   - Input: topic distribution → Output: user-location preference embedding

4. **TimeUserPair**
   - Multi-head attention with oscillator enhancement
   - Models temporal user preferences
   - Produces both attention embeddings and time predictions

5. **LocationTimePair**
   - Transformer encoder with causal masking
   - GELU activation, LayerNorm
   - Captures sequential location-time patterns

6. **CrossContextAttentiveDecoder (CNOA)**
   - Multi-head cross-attention with oscillator
   - Integrates information from different pair encoders
   - Query: user-location features (pre_embedded)
   - Key/Value: combined features without LDA component

7. **NextLocationPrediction**
   - MLP with residual connection and batch normalization
   - Separate heads for location and ranking predictions

8. **PositionalEncoding**
   - Sinusoidal positional embeddings
   - Learnable position parameters

### Multi-task Learning

The model jointly optimizes three objectives:

1. **Location Prediction**: CrossEntropyLoss (weight: λ_loc = 0.9)
2. **Time Prediction**: CrossEntropyLoss (weight: λ_time = 0.4)
3. **Ranking**: MarginRankingLoss (weight: λ_rank = 0.6)

Total Loss: `L = λ_loc × L_loc + λ_time × L_time + λ_rank × L_rank`

---

## 8. Configuration Parameters

### Model Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | int | 16 | Base embedding dimension |
| `bandwidth` | float | 1.0 | Gaussian kernel bandwidth for time smoothing |
| `model_type` | str | 'tc' | Model variant: 'tc' (Traffic Camera) or 'mp' (Mobile Phone) |
| `encoder_type` | str | 'trans' | Encoder type: 'trans' for Transformer |
| `at_type` | str | 'osc' | Attention type: 'osc' for oscillator, 'none' to disable |
| `topic_num` | int | 0 | Number of LDA topics (0 to disable LDA) |
| `max_seq_length` | int | 64 | Maximum sequence length |

### Loss Weights

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lambda_loc` | float | 0.9 | Weight for location prediction loss |
| `lambda_time` | float | 0.4 | Weight for time prediction loss |
| `lambda_rank` | float | 0.6 | Weight for ranking loss |

### Training Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_epoch` | int | 100 | Maximum training epochs |
| `learner` | str | 'adam' | Optimizer type |
| `learning_rate` | float | 0.005 | Initial learning rate |
| `batch_size` | int | 256 | Training batch size |

### Data Requirements

The model expects the following from `data_feature`:

- `loc_size`: Number of unique locations
- `uid_size`: Number of unique users
- `tim_size`: Number of time slots (should be 24 for hourly data)
- `user_topic_loc` (optional): Pre-computed LDA topic distributions `[num_users, topic_num]`

### Batch Format

Input batch should contain:

- `current_loc`: Location sequence `[batch_size, seq_len]`
- `current_tim`: Time sequence `[batch_size, seq_len]`
- `uid`: User IDs `[batch_size]`
- `target`: Target location `[batch_size]`
- `target_tim` (optional): Target time `[batch_size]`

---

## 9. Limitations and Future Work

### Current Limitations

1. **LDA Topic Modeling Disabled**
   - The UserLocationPair encoder using LDA topics is currently disabled (`topic_num = 0`)
   - Reason: LibCity doesn't have built-in LDA preprocessing
   - Impact: Loses one aspect of tri-pair interaction encoding
   - Future: Could add preprocessing script to compute LDA topics using gensim

2. **Time Encoding Differences**
   - Original CANOE uses hour-of-day (0-23)
   - LibCity's time slots may need conversion
   - Current implementation handles this with modulo operation

3. **Dataset-Specific Parameters**
   - Model has separate parameter sets for TC and MP datasets
   - Current default is 'tc' mode
   - May need tuning for optimal performance on different datasets

4. **Computational Cost**
   - Chaotic oscillator runs 50 iterations per forward pass
   - Multi-head attention with oscillator enhancement is computationally intensive
   - May benefit from optimization for large-scale deployment

### Future Improvements

1. **Enable LDA Integration**
   - Add preprocessing script to compute user-location topic distributions
   - Store in data_feature for use during training
   - Expected performance improvement based on original paper

2. **Hyperparameter Tuning**
   - Grid search for optimal λ_loc, λ_time, λ_rank weights
   - Tune oscillator parameters (a1, a2, b1, b2, k, n)
   - Adjust embedding dimension and architecture for different datasets

3. **Performance Optimization**
   - Optimize oscillator iterations (trade-off between chaos and speed)
   - Consider mixed precision training
   - Parallelize multi-head attention computations

4. **Extended Evaluation**
   - Test on more LibCity datasets (Gowalla, Brightkite, etc.)
   - Compare with other trajectory prediction models
   - Ablation studies on different components

5. **Additional Features**
   - Integrate geographical distance information
   - Add category/POI type embeddings
   - Support variable-length sequences more efficiently

---

## 10. Key Differences from Original Implementation

### Similarities (Preserved)

- Complete chaotic oscillator mechanism with all parameters
- Tri-pair interaction encoder architecture
- Multi-task learning with same loss functions
- Transformer encoder with causal masking
- Gaussian kernel smoothing for time embeddings
- All mathematical formulations preserved

### Differences (Adaptations)

1. **Framework Integration**
   - Inherits from LibCity's AbstractModel instead of standalone module
   - Uses LibCity's batch format and data loaders
   - Compatible with LibCity's training pipeline

2. **Batch Preprocessing**
   - Added `_prepare_batch()` method to normalize LibCity batch format
   - Handles different key naming conventions
   - Provides fallback for missing data fields

3. **LDA Preprocessing**
   - Original: Online LDA computation during data loading
   - Adapted: Expects pre-computed topics or disables LDA
   - Reason: Separation of data preprocessing and model training

4. **Output Format**
   - Modified `predict()` to return trajectory-level predictions
   - Added log_softmax for compatibility with NLLLoss evaluation
   - Maintains sequence-level predictions in `forward()` for training

5. **Device Management**
   - Integrated with LibCity's device configuration
   - Automatic GPU/CPU handling through config

---

## 11. References

### Original Paper
```
@inproceedings{yu2024beyond,
  title={Beyond Regularity: Modeling Chaotic Mobility Patterns for Next Location Prediction},
  author={Yu, Qian and others},
  booktitle={Proceedings of The Web Conference (WWW)},
  year={2024}
}
```

### Repository
- Original: https://github.com/yuqian2003/CANOE
- LibCity: https://github.com/LibCity/Bigscity-LibCity

### Related Documentation
- LibCity Documentation: https://bigscity-libcity-docs.readthedocs.io/
- Trajectory Location Prediction Task Guide
- AbstractModel API Reference

---

## 12. Acknowledgments

- Original CANOE implementation by Yu Qian et al.
- LibCity framework by Bigscity Lab
- Migration completed as part of AgentCity project

---

**Document Version**: 1.0
**Last Updated**: February 2, 2026
**Migration Status**: Complete ✓
