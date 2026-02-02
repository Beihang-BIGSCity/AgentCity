# HierETA Model Migration Summary

## Overview

HierETA (Hierarchical ETA) is a travel time estimation model that uses a hierarchical architecture to process route information at three levels: Route, Links, and Segments. This document summarizes the migration from the original standalone PyTorch implementation to the LibCity framework.

## Original Source Files

| File | Description |
|------|-------------|
| `/home/wangwenrui/shk/AgentCity/repos/HierETA/models/HierETA.py` | Main model class (HierETA_Net) |
| `/home/wangwenrui/shk/AgentCity/repos/HierETA/models/attrs.py` | Attribute feature extractor |
| `/home/wangwenrui/shk/AgentCity/repos/HierETA/models/segment_encoder.py` | Segment-level encoder with self-attention |
| `/home/wangwenrui/shk/AgentCity/repos/HierETA/models/link_encoder.py` | Link-level encoder |
| `/home/wangwenrui/shk/AgentCity/repos/HierETA/models/decoder.py` | Hierarchy-aware attention decoder |

## Migrated Files

| File | Description |
|------|-------------|
| `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/HierETA.py` | Consolidated model with all components |
| `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/eta/HierETA.json` | Model configuration file |

## Architecture Overview

```
Route Input
    |
    v
+------------------+
| HierETAAttr      |  <-- Embeds categorical and continuous features
| (Feature Extractor)|     for external, segment, and link attributes
+------------------+
    |
    +---> ext (external features)
    +---> seg (segment features)
    +---> cross (link/crossing features)
    |
    v
+------------------+
| SegmentEncoder   |  <-- BiLSTM + Local/Global Self-Attention
| (Segment Level)  |      Processes segments within each link
+------------------+
    |
    v
+------------------+
| LinkEncoder      |  <-- Aggregates segment features + LSTM
| (Link Level)     |      Processes link sequence with crossings
+------------------+
    |
    v
+------------------+
| AttentionDecoder |  <-- Hierarchical attention mechanism
| (Route Level)    |      Combines segment and link attention
+------------------+
    |
    v
Predicted ETA
```

## Key Adaptations

### 1. Base Class Inheritance
- **Original**: Inherits from `nn.Module`
- **Adapted**: Inherits from `AbstractTrafficStateModel`

### 2. Configuration Handling
- **Original**: Uses `FLAGS` argparse object
- **Adapted**: Uses `config` dictionary with defaults

### 3. Device Management
- **Original**: Hardcoded `.cuda()` calls in `segment_encoder.py`
- **Adapted**: Device-agnostic using `self.device` from config

### 4. Module Consolidation
- **Original**: Separate files for each component
- **Adapted**: All modules consolidated in single file for simpler imports

### 5. Required Methods
Added LibCity required methods:
- `predict(batch)`: Returns predictions
- `calculate_loss(batch)`: Returns MAE loss

### 6. Vocabulary Sizes
Made configurable via config/data_feature instead of hardcoded values:
- Driver vocabulary: configurable (default 200141)
- Segment vocabulary: configurable (default 1376567)
- Crossing vocabulary: configurable (default 101009)

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `segment_num` | 50 | Maximum segments per link |
| `link_num` | 31 | Maximum links per route |
| `win_size` | 3 | Window size for local attention |
| `Lambda` | 0.4 | Weight for link-level attention (vs segment) |
| `seg_hidden_dim` | 128 | Segment encoder hidden dimension |
| `link_hidden_dim` | 192 | Link encoder hidden dimension |
| `cross_hidden_dim` | 64 | Crossing encoder hidden dimension |
| `batch_size` | 64 | Training batch size |

## Data Format Requirements

The batch dictionary must contain the following keys:

### External Features
- `weekID`: Day of week (0-6)
- `timeID`: Time slot ID (0-288)
- `driverID`: Driver identifier

### Segment Features
- `segID`: Segment identifier
- `segment_functional_level`: Road functional level
- `roadState`: Traffic state
- `laneNum`: Number of lanes
- `roadLevel`: Road level
- `wid`: Width (continuous)
- `speedLimit`: Speed limit (continuous)
- `time`: Segment time (continuous)
- `len`: Segment length (continuous)

### Link Features
- `crossID`: Crossing/intersection identifier
- `delayTime`: Delay time at crossing (continuous)

### Masks and Lengths
- `link_seg_lens`: Tensor of segment lengths per link
- `road_segment_mask`: Binary mask for valid segments
- `link_lens`: Tensor of link lengths
- `road_link_mask`: Binary mask for valid links

### Target
- `gt_eta_time`: Ground truth travel time

## Assumptions and Limitations

1. **Fixed Maximum Dimensions**: The model assumes fixed maximum values for `segment_num` (50) and `link_num` (31). Sequences longer than these will be truncated.

2. **Normalization Statistics**: The model expects `train_gt_eta_time_mean` and `train_gt_eta_time_std` in `data_feature` for denormalization.

3. **Batch Size**: The batch size must match the configured `batch_size` parameter due to hardcoded reshape operations.

4. **Data Preprocessing**: A custom data encoder may be needed to convert raw trajectory data into the required hierarchical format.

## Loss Function

The model uses Mean Absolute Error (MAE) loss:
```python
loss = torch.mean(torch.abs(pred - label))
```

Both predictions and labels are denormalized before computing the loss.

## Usage Example

```python
from libcity.model.eta import HierETA

# Initialize
config = {
    'device': torch.device('cuda'),
    'batch_size': 64,
    'segment_num': 50,
    'link_num': 31,
    # ... other params
}
data_feature = {
    'train_gt_eta_time_mean': 1200.0,
    'train_gt_eta_time_std': 600.0,
}

model = HierETA(config, data_feature)

# Forward pass
pred = model.predict(batch)

# Training
loss = model.calculate_loss(batch)
```

## References

- Original repository: `repos/HierETA/`
- Paper: HierETA - Hierarchical Estimation of Travel Time with Multi-view Attention
