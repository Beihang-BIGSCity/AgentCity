# MDTI Model Migration Summary

## Overview
MDTI (Multi-modal Dual Transformer for Travel Time Estimation) has been successfully adapted to the LibCity framework. This model combines grid-based and road network representations for accurate travel time estimation using multi-modal learning.

## Original Source
- **Repository**: `./repos/MDTI`
- **Main model file**: `./repos/MDTI/model/MDTI.py`
- **Supporting modules**: `./repos/MDTI/model/`

## Adapted Files

### Main Model
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/MDTI.py`

### Supporting Modules
Located in `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/mdti_modules/`:
- `__init__.py` - Module exports
- `GridTrm.py` - Grid-based Transformer encoder
- `RoadTrm.py` - Road network Transformer encoder with adaptive attention
- `InterTrm.py` - Cross-modal fusion Transformer
- `RoadGNN.py` - GAT-based road network encoder
- `GridToGraph.py` - Grid image to graph converter
- `GridConv.py` - CNN for grid image processing
- `Date2Vec.py` - Temporal embedding module

### Configuration
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/eta/MDTI.json`

### Registration
- Updated `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/__init__.py`
- Updated `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`

## Key Adaptations

### 1. Base Class Inheritance
The model now inherits from `AbstractTrafficStateModel` following LibCity conventions:
```python
class MDTI(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super(MDTI, self).__init__(config, data_feature)
```

### 2. Configuration Pattern
All hyperparameters use LibCity's `config.get()` pattern:
```python
self.hidden_emb_dim = config.get('hidden_emb_dim', 256)
self.grid_H = config.get('grid_H', 114)
```

### 3. Required Methods
Implemented LibCity-required methods:
- `forward(batch)` - Main forward pass
- `predict(batch)` - Returns predicted travel times
- `calculate_loss(batch)` - Computes training loss

### 4. Dual Mode Support
The model supports two modes:
- **Pre-training mode** (`mode='pretrain'`): Contrastive learning + Masked Language Modeling
- **TTE mode** (`mode='tte'`): Travel time estimation fine-tuning

### 5. Batch Format Adaptation
The model expects batch dictionaries with the following structure:
```python
batch = {
    'grid_data': {
        'grid_image': tensor,      # (B, H, W, C) grid images
        'grid_traj': tensor,       # (B, L) grid trajectory indices
        'grid_time_emb': tensor,   # (B, L, D) time embeddings
        'grid_feature': tensor,    # (B, L, 4) additional features
    },
    'road_data': {
        'g_input_feature': tensor, # (N, F) road node features
        'g_edge_index': tensor,    # (2, E) road graph edges
        'road_traj': tensor,       # (B, L) road trajectory indices
        'road_weeks': tensor,      # (B, L) week indices
        'road_minutes': tensor,    # (B, L) minute indices
        'road_type': tensor,       # (B, L) road type indices
    },
    'travel_time': tensor,         # (B,) or (B, 1) ground truth
}
```

## Model Architecture

### Components
1. **Grid Encoder Path**:
   - GridToGraph: Converts grid images to 8-connectivity graphs
   - GraphEncoder: GAT-based encoding of grid graph
   - GridTrm: Transformer encoder for grid sequences

2. **Road Encoder Path**:
   - RoadGNN: GAT-based encoding of road network
   - RoadTrm: Transformer with road type-aware attention

3. **Fusion Layer**:
   - InterTrm: Cross-attention between road and grid modalities

4. **Task Heads**:
   - Pre-training: Contrastive + MLM heads
   - TTE: MLP regression head

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_emb_dim | 256 | Hidden embedding dimension |
| out_emb_dim | 256 | Output embedding dimension |
| pe_dropout | 0.1 | Positional encoding dropout |
| grid_in_channel | 3 | Grid input channels (RGB) |
| grid_out_channel | 64 | Grid CNN output channels |
| grid_H | 114 | Grid height |
| grid_W | 52 | Grid width |
| grid_trm_head | 4 | Grid transformer heads |
| grid_trm_layer | 2 | Grid transformer layers |
| road_type | 8 | Number of road types |
| road_trm_head | 4 | Road transformer heads |
| road_trm_layer | 4 | Road transformer layers |
| g_num_layers | 3 | Road GNN layers |
| inter_trm_head | 2 | Fusion transformer heads |
| inter_trm_layer | 2 | Fusion transformer layers |
| mode | "tte" | "pretrain" or "tte" |
| mask_ratio | 0.15 | MLM mask ratio |
| w_cl | 1.0 | Contrastive loss weight |
| w_mlm | 0.7 | MLM loss weight |

## Dependencies
- PyTorch
- torchvision (for image normalization)
- torch_geometric (for GAT and graph operations)

## Assumptions and Limitations

1. **Grid Image Format**: Expects grid images in (B, H, W, C) format which is converted to (B, C, H, W) internally
2. **Pre-trained Prompts**: The original model uses pre-computed prompt embeddings from GPT-2. This feature is currently disabled in the LibCity adaptation. To enable it, set the `use_prompt` config parameter and provide the embedding file path.
3. **Data Preprocessing**: The original data preprocessing (including Date2Vec temporal embeddings) should be handled by a custom data encoder (MDTIEncoder) which needs to be implemented separately.
4. **Graph Construction**: Grid graph construction happens during forward pass, which may be computationally expensive for large grids. Consider caching edge indices if grid dimensions are fixed.

## Usage Example

```python
from libcity.model.eta import MDTI

# Initialize with config and data_feature
config = {
    'device': 'cuda',
    'hidden_emb_dim': 256,
    'mode': 'tte',
    # ... other config parameters
}
data_feature = {
    'road_num': 10000,
    'g_fea_size': 64,
}

model = MDTI(config, data_feature)

# Forward pass
pred = model.predict(batch)

# Training
loss = model.calculate_loss(batch)
```
