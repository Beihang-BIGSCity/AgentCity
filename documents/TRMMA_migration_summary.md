## Repository: TRMMA
- **URL**: https://github.com/derekwtian/TRMMA
- **Cloned to**: /home/wangwenrui/shk/AgentCity/repos/TRMMA
- **Paper**: Efficient Methods for Accurate Sparse Trajectory Recovery and Map Matching (ICDE 2025)

### Key Files

#### Model Files
- **Main Model**: `/repos/TRMMA/models/trmma.py`
  - **Primary Class**: `TrajRecovery` (line 913)
  - **Architecture**: Encoder-Decoder with DualFormer (GPS + Route Transformer)
  - **Components**:
    - `GPSEncoder`: GPS trajectory encoder using GPSFormer
    - `GREncoder`: GPS+Route dual encoder using GRFormer
    - `DecoderMulti`: Multi-task decoder (segment ID + position ratio prediction)
    - `DAPlanner`: Route planning using Direction-Aware algorithm

- **Supporting Model**: `/repos/TRMMA/models/mma.py`
  - **Class**: `GPS2Seg` (Map Matching Attention model)
  - Used for pre-processing/map matching step

- **Layers**: `/repos/TRMMA/models/layers.py`
  - `GPSFormer`: GPS-only transformer encoder
  - `GRFormer`: GPS-Route dual transformer (DualFormer)
  - `Attention`: Custom attention mechanism
  - `MultiHeadAttention`, `FeedForward`, `Norm`: Standard transformer components

### Model Architecture Overview

#### Task Type
**Trajectory Recovery (Map Matching + Route Interpolation)**

The model performs two related tasks:
1. **Map Matching (MMA)**: Match sparse GPS points to road segments
2. **Trajectory Recovery (TRMMA)**: Recover high-sampling trajectories from sparse trajectories

---

## LibCity Migration Summary

### Migration Date: 2026-02-04

### Status: COMPLETED

### Files Created/Modified

1. **Model File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/TRMMA.py`
   - Full adaptation of TrajRecovery model and all supporting classes
   - Inherits from `AbstractModel`
   - Implements `forward()`, `predict()`, `calculate_loss()` methods
   - Self-contained with all required layer implementations

2. **Config File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/TRMMA.json`
   - Default hyperparameters for all model options
   - Training configuration included

3. **Registration**: Model registered in `__init__.py`

---

### Model Architecture

The adapted TRMMA model includes the following components:

#### Supporting Layer Classes

| Class | Description | Original File |
|-------|-------------|---------------|
| `sequence_mask` | Masks irrelevant entries in sequences | layers.py |
| `sequence_mask3d` | Masks irrelevant entries in 3D sequences | layers.py |
| `PositionalEncoder` | Sinusoidal positional encoding | layers.py |
| `MultiHeadAttention` | Multi-head self-attention | layers.py |
| `FeedForward` | Position-wise feed-forward network | layers.py |
| `Norm` | Layer normalization | layers.py |
| `GPSLayer` | GPS transformer layer with self-attention | layers.py |
| `GPSFormer` | GPS sequence transformer encoder | layers.py |
| `RouteLayer` | Route layer with self + cross attention | layers.py |
| `GRLayer` | GPS-Route dual layer | layers.py |
| `GRFormer` | GPS-Route dual transformer | layers.py |
| `Attention` | Bahdanau-style attention for decoder | layers.py |

#### Encoder Classes

| Class | Description | Original File |
|-------|-------------|---------------|
| `GPSEncoder` | GPS-only encoder with temporal features | trmma.py |
| `GREncoder` | GPS-Route dual encoder with temporal features | trmma.py |

#### Decoder Classes

| Class | Description | Original File |
|-------|-------------|---------------|
| `DecoderMulti` | GRU-based decoder with route attention | trmma.py |

#### Main Model Classes

| Class | Description | Original File |
|-------|-------------|---------------|
| `TrajRecoveryModule` | Core trajectory recovery module | trmma.py (TrajRecovery) |
| `TRMMA` | LibCity-compatible wrapper class | New (LibCity adapter) |

---

### Configuration Parameters

#### Model Architecture Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hid_dim` | int | 256 | Hidden dimension for all layers |
| `id_emb_dim` | int | 128 | Road segment embedding dimension |
| `transformer_layers` | int | 2 | Number of transformer layers |
| `heads` | int | 4 | Number of attention heads |
| `dropout` | float | 0.1 | Dropout probability |

#### Feature Flags

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pro_features_flag` | bool | true | Use temporal features (hour of day) |
| `pro_input_dim` | int | 48 | Temporal feature vocabulary size |
| `pro_output_dim` | int | 64 | Temporal embedding dimension |
| `learn_pos` | bool | true | Use learned positional embeddings |
| `da_route_flag` | bool | true | Use dual route encoder (GREncoder) |
| `srcseg_flag` | bool | false | Use source segment features |
| `rid_feats_flag` | bool | false | Use road segment features |
| `rid_fea_dim` | int | 8 | Road segment feature dimension |

#### Decoder Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dest_type` | int | 1 | Destination embedding type (0, 1, or 2) |
| `rate_flag` | bool | false | Predict position rate within segment |
| `prog_flag` | bool | false | Use progressive decoding |

#### Loss Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lambda1` | float | 1.0 | Weight for ID prediction loss |
| `lambda2` | float | 0.0 | Weight for rate prediction loss |
| `teacher_forcing_ratio` | float | 0.5 | Teacher forcing probability during training |

#### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 64 | Training batch size |
| `learning_rate` | float | 0.001 | Initial learning rate |
| `max_epoch` | int | 50 | Maximum training epochs |
| `optimizer` | string | "adam" | Optimizer type |
| `weight_decay` | float | 0.0001 | L2 regularization |
| `lr_scheduler` | string | "reducelronplateau" | Learning rate scheduler |
| `lr_decay` | float | 0.5 | Learning rate decay factor |
| `lr_step` | int | 5 | Learning rate decay step |
| `clip_grad_norm` | bool | true | Enable gradient clipping |
| `max_grad_norm` | float | 1.0 | Maximum gradient norm |
| `use_early_stop` | bool | true | Enable early stopping |
| `patience` | int | 10 | Early stopping patience |

---

### Expected Batch Format

The TRMMA model can accept batch data with the following keys:

#### Primary Input Keys

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `current_loc` OR `src_grid_seq` | (batch, seq_len) OR (batch, seq_len, 3) | long/float | Source GPS sequence |
| `target_loc` OR `trg_rid` | (batch, seq_len) | long | Target road segment IDs |
| `trg_rate` | (batch, seq_len, 1) | float | Target position rates (optional) |
| `da_route` | (batch, route_len) | long | Route candidate IDs |
| `pro_features` OR `current_tim` | (batch,) OR (batch, seq_len) | long | Temporal features |

#### Optional Keys

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `src_seg_seq` | (batch, seq_len) | long | Source segment sequences |
| `src_seg_feat` | (batch, seq_len, 1) | float | Source segment features |
| `rid_features` | (id_size, rid_fea_dim) | float | Road segment feature lookup |
| `label` | (batch, seq_len, route_len) | float | One-hot attention labels |

---

### Usage Example

```python
from libcity.model.trajectory_loc_prediction import TRMMA

config = {
    'device': 'cuda',
    'hid_dim': 256,
    'id_emb_dim': 128,
    'transformer_layers': 2,
    'heads': 4,
    'dropout': 0.1,
    'da_route_flag': True,
    'learn_pos': True,
    'pro_features_flag': True,
    'teacher_forcing_ratio': 0.5,
}

data_feature = {
    'loc_size': 10000,  # Number of road segments
    'loc_pad': 0,       # Padding token ID
}

model = TRMMA(config, data_feature)

# Training
loss = model.calculate_loss(batch)

# Inference
predictions = model.predict(batch)
```

---

### Key Adaptations from Original

1. **Removed External Dependencies**
   - Removed `preprocess` module dependencies (SparseDAM, SegInfo)
   - Removed `utils.spatial_func` dependencies (SPoint, project_pt_to_road, rate2gps)
   - Removed `utils.trajectory_func` dependencies (STPoint)
   - Removed `utils.candidate_point` dependencies (CandidatePoint)
   - Removed `utils.model_utils` dependencies (gps2grid, get_normalized_t)
   - Removed DAPlanner route planning (expects pre-computed routes)

2. **Integrated Layer Classes**
   - All layer classes from `layers.py` integrated into single file
   - All helper functions (sequence_mask, sequence_mask3d) included

3. **LibCity-Compatible Interface**
   - Inherits from `AbstractModel`
   - Implements `predict()` returning log-softmax scores
   - Implements `calculate_loss()` returning combined loss

4. **Flexible Batch Format**
   - Accepts multiple key naming conventions
   - Auto-detects available features
   - Provides sensible defaults for missing optional keys

5. **Device Handling**
   - All tensors properly moved to configured device
   - Uses `config.get('device', 'cpu')` for device configuration

---

### Limitations

1. **Pre-computed Routes Required**: Route planning (DAPlanner) is not included; routes must be pre-computed and provided in batch data.

2. **No Road Network Integration**: External road network preprocessing not included; model works with provided road segment vocabularies.

3. **Simplified Rate Prediction**: Position rate prediction is optional and disabled by default.

4. **No Graph Neural Networks**: Original road network graph features not included.

---

### Testing

To verify the migration works:

```bash
cd /home/wangwenrui/shk/AgentCity/Bigscity-LibCity
python -c "from libcity.model.trajectory_loc_prediction import TRMMA; print('TRMMA import successful')"
```

To run with a dataset:

```bash
python run_model.py --task traj_loc_pred --model TRMMA --dataset <your_dataset>
```

---

### Component Hierarchy

```
TRMMA (AbstractModel)
|
+-- TrajRecoveryModule (nn.Module)
    |
    +-- emb_id (nn.Parameter) - Road segment embeddings
    |
    +-- pos_embedding_gps (nn.Embedding) - GPS positional embeddings
    +-- pos_embedding_route (nn.Embedding) - Route positional embeddings
    |
    +-- fc_in_gps (nn.Linear) - GPS input projection
    +-- fc_in_route (nn.Linear) - Route input projection
    |
    +-- encoder: GREncoder or GPSEncoder
    |   |
    |   +-- transformer: GRFormer or GPSFormer
    |   |   |
    |   |   +-- layers: [GRLayer or GPSLayer]
    |   |       |
    |   |       +-- MultiHeadAttention
    |   |       +-- FeedForward
    |   |       +-- Norm
    |   |
    |   +-- temporal (nn.Embedding) - Temporal feature embedding
    |   +-- fc_hid (nn.Linear) - Hidden state projection
    |
    +-- decoder: DecoderMulti
        |
        +-- rnn (nn.GRU) - Recurrent decoder
        +-- attn_route (Attention) - Route attention
        +-- fc_rate_out (nn.Sequential) - Rate prediction (optional)
```
