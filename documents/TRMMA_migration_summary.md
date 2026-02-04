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

## LibCity Migration - Data Format Fix (2026-02-04)

### Problem Diagnosis

The original TRMMA model expected specialized batch keys designed for trajectory recovery with road network data:
- `src_grid_seq`: Source GPS grid sequence
- `da_routes`: Route segment candidate IDs
- `trg_rid`: Target segment IDs
- `trg_rate`: Target position rates
- `labels`: One-hot encoded target labels over route candidates
- And many more...

**Error encountered**: `KeyError: 'src_grid_seq is not in the batch'`

However, LibCity's `StandardTrajectoryEncoder` provides a simpler format for next-location prediction:
- `current_loc`: Current trajectory location IDs
- `current_tim`: Current trajectory time encodings
- `history_loc`: History location IDs
- `history_tim`: History time encodings
- `target`: Next location to predict
- `target_tim`: Target time encoding
- `uid`: User IDs

### Solution Implemented: Model Modification

**Chosen Approach**: Option 2 - Modify the TRMMA model to accept standard LibCity trajectory batch format.

**Rationale**: Creating a custom encoder would require road network preprocessing and complex route candidate generation that is not available in standard LibCity trajectory datasets.

### Key Changes Made

#### 1. Automatic Batch Format Detection

Added `_is_standard_libcity_batch()` method that detects whether the input batch is in standard LibCity format or original TRMMA format:

```python
def _is_standard_libcity_batch(self, batch):
    """Check if batch is in standard LibCity format."""
    has_current_loc = 'current_loc' in batch
    has_src_grid_seq = 'src_grid_seq' in batch
    return has_current_loc and not has_src_grid_seq
```

#### 2. Batch Format Conversion

Added `_convert_batch_format()` method that transforms standard LibCity batch to TRMMA's internal format:

**Standard LibCity -> TRMMA Mapping**:
| LibCity Key | TRMMA Key | Transformation |
|------------|-----------|----------------|
| `current_loc` | `src_emb` | Location embedding lookup |
| Computed from `current_loc` | `src_len` | Count non-padding positions |
| Generated from batch | `da_routes` | Unique locations + random samples |
| `current_loc[:, -1]` + `target` | `trg_rid` | [last_loc, target, target] |
| Constant 0.5 | `trg_rate` | Dummy rates (not used) |
| `current_tim[:, -1]` | `pro_features` | Last time encoding |
| `target` | `d_rids` | Destination location |
| One-hot over candidates | `labels` | Target in route candidates |

#### 3. Dual Forward Pass Methods

Split the forward method into two implementations:
- `_forward_libcity()`: For standard LibCity format, returns logits for cross-entropy loss
- `_forward_original()`: For TRMMA format with all specialized keys

#### 4. Dual Loss Calculation

Split the loss calculation:
- `_calculate_loss_libcity()`: Standard cross-entropy loss for next-location prediction
- `_calculate_loss_original()`: Combined segment ID + rate loss

#### 5. Updated Architecture

Added new components for LibCity compatibility:
- `loc_embedding`: Learnable location embedding for generating pseudo-GPS features
- `output_layer`: Linear projection from hidden state to vocabulary for next-location prediction

#### 6. Updated Default Configuration

Disabled road-network-specific features by default:
- `srcseg_flag`: `false` (no road segment data)
- `rate_flag`: `false` (no position rate prediction needed)
- `lambda2`: `0.0` (rate loss weight set to zero)

### Files Modified

1. **Model File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/TRMMA.py`
   - Added `_is_standard_libcity_batch()` method
   - Added `_convert_batch_format()` method
   - Added `_forward_libcity()` method
   - Added `_forward_original()` method
   - Added `_calculate_loss_libcity()` method
   - Added `_calculate_loss_original()` method
   - Updated `forward()`, `predict()`, `calculate_loss()` to dispatch based on batch format
   - Added `loc_embedding` and `output_layer` in `_build_model()`

2. **Config File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/TRMMA.json`
   - Updated defaults for LibCity compatibility

### Updated Configuration

```json
{
    "model": "TRMMA",
    "task": "traj_loc_pred",

    "hid_dim": 256,
    "id_emb_dim": 128,
    "transformer_layers": 2,
    "heads": 4,
    "dropout": 0.1,

    "pro_features_flag": true,
    "pro_input_dim": 48,
    "pro_output_dim": 64,
    "learn_pos": true,
    "da_route_flag": true,
    "srcseg_flag": false,
    "rid_feats_flag": false,
    "rid_fea_dim": 8,

    "dest_type": 1,
    "rate_flag": false,
    "prog_flag": false,

    "lambda1": 1.0,
    "lambda2": 0.0,
    "teacher_forcing_ratio": 0.5,

    "candi_size": 50,
    "max_input_length": 500,
    "history_type": "splice",
    "evaluate_method": "popularity"
}
```

### Usage Example

```python
# TRMMA now works with standard LibCity trajectory datasets
from libcity.model.trajectory_loc_prediction import TRMMA

config = {
    'device': 'cuda',
    'hid_dim': 256,
    'transformer_layers': 2,
    'heads': 4,
}

data_feature = {
    'loc_size': 5000,  # Number of locations
    'loc_pad': 4999,   # Padding token ID
    'tim_size': 48,    # Time vocabulary size
}

model = TRMMA(config, data_feature)

# Training with standard LibCity batch
batch = {
    'current_loc': current_loc_tensor,  # (batch, seq_len)
    'current_tim': current_tim_tensor,  # (batch, seq_len)
    'target': target_tensor,            # (batch,)
}

loss = model.calculate_loss(batch)
predictions = model.predict(batch)
```

### Backward Compatibility

The model remains **fully backward compatible** with the original TRMMA format. If a batch contains `src_grid_seq` key, it will use the original TRMMA forward pass. This allows:

1. Using TRMMA with standard LibCity datasets (foursquare_nyc, gowalla, etc.)
2. Using TRMMA with custom preprocessed data in original format
3. Gradual migration of existing pipelines

### Limitations

1. **Simplified Route Candidates**: Route candidates are generated from unique locations in the batch plus random samples, not from actual road network analysis.

2. **No Position Rate Prediction**: The rate prediction (position within segment) is disabled for standard LibCity datasets since there's no road network concept.

3. **Pseudo-GPS Features**: Location embeddings are used instead of actual GPS coordinates for source sequence encoding.

4. **No Road Network Integration**: The model works without road network preprocessing but loses some of the original TRMMA's map-matching capabilities.

### Testing

To verify the fix works:

```bash
cd /home/wangwenrui/shk/AgentCity/Bigscity-LibCity
python run_model.py --task traj_loc_pred --model TRMMA --dataset foursquare_nyc
```

The model should now train without `KeyError` on standard LibCity trajectory datasets.

---

## Original Migration Summary (Preserved for Reference)

### Date: 2026-02-02

### Status: COMPLETED (with Data Format Fix 2026-02-04)

### Files Created/Modified

1. **Model File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/TRMMA.py`
   - Full adaptation of TrajRecovery model
   - Includes all supporting classes (GPSFormer, GRFormer, DecoderMulti, etc.)
   - Inherits from AbstractModel
   - Implements forward(), predict(), calculate_loss() methods
   - **Updated 2026-02-04**: Added LibCity batch format compatibility

2. **Config File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/TRMMA.json`
   - Default hyperparameters
   - All configuration options documented
   - **Updated 2026-02-04**: Changed defaults for LibCity compatibility

3. **Registration**: Model registered in `__init__.py`
