# LoTNext Model Migration to LibCity

## Overview

This document describes the migration of the LoTNext (Long-tail Next POI Prediction) model from its original implementation to the LibCity framework for trajectory location prediction.

## Source Files

**Original Repository:** `/home/wangwenrui/shk/AgentCity/repos/LoTNext`

- **Main Model:** `network.py` - Contains the Flashback class (lines 314-518)
- **Sub-modules:** `model.py` - Contains TransformerModel, Time2Vec, FuseEmbeddings, etc.
- **Utilities:** `utils.py` - Contains sparse matrix operations and distance functions

## Target Location

**LibCity Path:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/LoTNext.py`

## Architecture Components Migrated

### Core Components

1. **Flashback RNN** - Adapted as LoTNext class inheriting from AbstractModel
2. **RnnFactory** - Creates RNN/GRU/LSTM units based on config
3. **Time2Vec** - Temporal encoding using sine/cosine activations
4. **EncoderLayer** - Transformer encoder with multi-head attention
5. **FuseEmbeddings** - Fusion of location and time embeddings
6. **DenoisingGCNNet** - Graph denoising for user-POI interactions

### Supporting Components

- **AttentionLayer** - Edge weight computation for graph
- **DenoisingLayer** - Edge filtering with threshold
- **GCNLayer** - Graph convolution (with torch_geometric fallback)
- **GraphConvolution** - Standard graph convolution
- **MultiHeadAttention** - Self-attention with dynamic masking
- **PositionalEncoding** - Transformer positional encoding
- **FeedForwardNetwork** - FFN for transformer encoder

### Hidden State Strategies

- **H0Strategy** - Base class for hidden state initialization
- **FixNoiseStrategy** - Fixed normal noise initialization
- **LstmStrategy** - LSTM-specific h0/c0 initialization

## Key Adaptations

### 1. Configuration Parameters

Original (command-line args) -> LibCity (config.get()):

```python
# Original
parser.add_argument('--hidden_dim', default=128, type=int)

# LibCity
self.hidden_size = config.get('hidden_size', 128)
```

### 2. Data Feature Extraction

```python
# LibCity data_feature extraction
self.loc_size = data_feature.get('loc_size', 1000)
self.uid_size = data_feature.get('uid_size', 100)
self.loc_pad = data_feature.get('loc_pad', 0)
```

### 3. Batch Format Handling

```python
# LibCity batch format
loc = batch.get('current_loc', batch.get('X', None))
tim = batch.get('current_tim', None)
uid = batch.get('uid', None)
target = batch.get('target', batch.get('y', None))
```

### 4. Required Methods Implementation

```python
def predict(self, batch):
    """Return log-softmax scores for location prediction."""
    y_linear, _, _ = self.forward(batch)
    score = F.log_softmax(y_linear, dim=1)
    return score

def calculate_loss(self, batch):
    """Multi-task loss: location CE + time MSE."""
    y_linear, out_time, _ = self.forward(batch)
    loc_loss = CrossEntropyLoss(y_linear, target)
    time_loss = MSELoss(out_time, target_tim)  # if available
    return loc_loss_weight * loc_loss + time_loss_weight * time_loss
```

### 5. Graph Preprocessing

Added `set_graphs()` method for optional graph initialization:

```python
def set_graphs(self, transition_graph=None, spatial_graph=None, interact_graph=None):
    """Set graph matrices for GCN-enhanced encoding."""
```

### 6. torch_geometric Fallback

Added fallback for environments without torch_geometric:

```python
try:
    from torch_geometric.nn import GCNConv
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    # Simple linear fallback
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_size | 128 | Hidden dimension for embeddings |
| loc_emb_size | 128 | Location embedding size |
| time_emb_size | 6 | Time2Vec output dimension |
| rnn_type | "LSTM" | RNN type: RNN/GRU/LSTM |
| batch_size | 32 | Batch size |
| sequence_length | 20 | Maximum sequence length |
| transformer_nhid | 256 | Transformer FFN hidden size |
| transformer_nhead | 4 | Number of attention heads |
| transformer_dropout | 0.1 | Transformer dropout rate |
| attention_dropout_rate | 0.1 | Attention dropout rate |
| lambda_loc | 1.0 | Graph weight for location |
| lambda_user | 1.0 | Graph weight for user |
| use_graph_user | false | Use GCN for user embeddings |
| use_spatial_graph | false | Use spatial proximity graph |
| loc_loss_weight | 1.0 | Weight for location loss |
| time_loss_weight | 0.1 | Weight for time prediction loss |
| num_time_slots | 168 | Number of time slots (24*7) |
| spatial_decay | 100.0 | Spatial decay parameter (km) |

## Files Modified

1. **Created:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/LoTNext.py`
2. **Modified:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py`
   - Added import for LoTNext
   - Added "LoTNext" to __all__
3. **Created:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/LoTNext.json`
4. **Modified:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
   - Added "LoTNext" to allowed_model list
   - Added LoTNext configuration entry

## Usage Example

```python
from libcity.pipeline import run_model

# Run with default config
run_model(
    task='traj_loc_pred',
    model='LoTNext',
    dataset='foursquare_tky'
)
```

## Limitations and Notes

1. **Graph Features:** The model supports optional graph-enhanced embeddings. If graphs are not provided, the model falls back to standard embeddings.

2. **Coordinate Data:** Spatial attention weighting requires coordinate data (`current_coord` field). Without coordinates, spatial weighting is disabled.

3. **Time Prediction:** The multi-task time prediction is optional and only computed if `target_tim` is provided in the batch.

4. **torch_geometric:** For full GCN functionality, torch_geometric should be installed. A fallback linear layer is used otherwise.

5. **Long-tail Handling:** The Expert/EnsembleModel components for long-tail reweighting are not included in the core model but can be added as extensions.

## Testing

To verify the model import:

```python
from libcity.model.trajectory_loc_prediction import LoTNext
print("LoTNext imported successfully")
```
