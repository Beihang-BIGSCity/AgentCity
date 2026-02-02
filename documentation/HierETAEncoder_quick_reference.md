# HierETAEncoder - Quick Reference

## Files Created

### 1. Encoder Implementation
**Path**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/eta_encoder/hiereta_encoder.py`

**Class**: `HierETAEncoder(AbstractETAEncoder)`

**Key Methods**:
- `__init__(config)`: Initialize with config parameters
- `encode(uid, trajectories, dyna_feature_column)`: Transform trajectories to hierarchical format
- `gen_data_feature()`: Generate padding items and data features
- `gen_scalar_data_feature(train_data)`: Generate normalization statistics

### 2. Registration
**Path**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/eta_encoder/__init__.py`

**Changes**:
- Added import: `from .hiereta_encoder import HierETAEncoder`
- Added to `__all__`: `"HierETAEncoder"`

## Output Features

### External (Route-level)
- `weekID`, `timeID`, `driverID`

### Segment (Point-level, padded to link_num×segment_num)
- **Categorical**: `segID`, `segment_functional_level`, `roadState`, `laneNum`, `roadLevel`
- **Continuous**: `wid`, `speedLimit`, `time`, `len`

### Link (Intersection-level, padded to link_num)
- `crossID`, `delayTime`

### Masks & Metadata
- `road_segment_mask`, `road_link_mask`, `link_seg_lens`, `link_lens`, `gt_eta_time`

## Configuration

```json
{
  "eta_encoder": "HierETAEncoder",
  "link_num": 31,
  "segment_num": 50,
  "segments_per_link": 10
}
```

## Real vs Placeholder Features

### Real (Computed from Data)
- ✅ Coordinates and times
- ✅ Segment lengths (haversine distance)
- ✅ Segment times (time deltas)
- ✅ Speed limits (estimated from speed)
- ✅ External temporal features (weekID, timeID)
- ✅ Ground truth ETA

### Placeholder (Need Enhancement)
- ⚠️ Hierarchical link structure (artificial grouping)
- ⚠️ Segment IDs (point index, not road network ID)
- ⚠️ Road categorical attributes (random values)
- ⚠️ Road width (normalized from distance)
- ⚠️ Crossing IDs (link index)
- ⚠️ Delay times (inter-link time gaps)

## Key Design Decisions

1. **Hierarchical Structure**: Groups every `segments_per_link` consecutive points into a link
2. **Pre-padding**: All sequences padded to fixed dimensions during encoding
3. **Index-based IDs**: Uses trajectory indices as segment IDs (no map-matching)
4. **Placeholder Attributes**: Random values for unavailable road network data
5. **Flexible Configuration**: Supports variable link/segment numbers

## Simplifications & Assumptions

1. **No Map Matching**: Assumes trajectory points can serve as "segments"
2. **Uniform Link Length**: All links have ~segments_per_link segments
3. **No Road Network**: Road attributes are placeholders
4. **Sequential Processing**: Links are created sequentially from trajectory
5. **Fixed Dimensions**: All routes padded to same size (link_num × segment_num)

## Enhancement Path

**Phase 1 (Current)**: Working encoder with placeholder features
- ✅ Basic hierarchical structure
- ✅ Real temporal and spatial features
- ✅ Compatible with HierETA model

**Phase 2 (Near-term)**:
- 🔲 Integrate map-matching for real segment IDs
- 🔲 Load road attributes from OSM
- 🔲 Better intersection detection

**Phase 3 (Future)**:
- 🔲 Dynamic road states from traffic APIs
- 🔲 Historical delay patterns
- 🔲 Multi-modal trajectory support

## Usage

```python
# In config file or dict
config = {
    'dataset': 'Chengdu',
    'eta_encoder': 'HierETAEncoder',
    'link_num': 31,
    'segment_num': 50,
    'segments_per_link': 10,
    'batch_size': 64,
}

# Dataset automatically uses encoder
from libcity.data.dataset import ETADataset
dataset = ETADataset(config)
train_data, valid_data, test_data = dataset.get_data()

# Batches contain hierarchical features
for batch in train_data:
    # All HierETA required keys present
    assert 'weekID' in batch
    assert 'segID' in batch
    assert 'crossID' in batch
    assert 'road_segment_mask' in batch
    # ... ready for model.forward(batch)
```

## Testing Checklist

- [ ] Test with short trajectories (< segments_per_link)
- [ ] Test with long trajectories (> link_num × segment_num)
- [ ] Verify mask consistency
- [ ] Check padding correctness
- [ ] Test normalization statistics
- [ ] Integration test with HierETA model
- [ ] Memory profiling with large batches
- [ ] Cache functionality

## Documentation

**Full Documentation**: `/home/wangwenrui/shk/AgentCity/documentation/HierETAEncoder_implementation.md`

Contains:
- Detailed architecture explanation
- Feature derivation logic
- Data flow diagrams
- Enhancement roadmap
- Known limitations
- Testing recommendations
