# LARGEST Dataset - Quick Reference Guide

## Summary

Successfully converted LargeST (California Highway Traffic 2017) to LibCity format.

**Status:** Conversion Complete and Validated ✓
**Dataset Name:** LARGEST
**Ready for Training:** Yes

## Key Statistics

| Metric | Value |
|--------|-------|
| **Nodes (Sensors)** | 8,083 |
| **Edges** | 80,830 |
| **Timesteps** | 105,120 |
| **Total Records** | 849,684,960 |
| **Time Range** | 2017-01-01 to 2017-12-31 |
| **Interval** | 5 minutes (300 seconds) |
| **Feature** | Traffic speed (0-120 mph) |
| **File Size** | 38 GB (.dyna) |

## File Locations

```
/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/raw_data/LARGEST/
├── LARGEST.geo       # 8,083 nodes
├── LARGEST.rel       # 80,830 edges
├── LARGEST.dyna      # 849,684,960 records (38 GB)
├── config.json       # Dataset configuration
└── README.md         # Detailed documentation
```

## Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| **Conversion** | `/home/wangwenrui/shk/AgentCity/preprocess/largest_to_libcity.py` | Convert HDF5 to LibCity |
| **Validation** | `/home/wangwenrui/shk/AgentCity/preprocess/validate_largest.py` | Validate converted files |

## Documentation

| Document | Location | Content |
|----------|----------|---------|
| **Migration Guide** | `/home/wangwenrui/shk/AgentCity/documentation/LARGEST_dataset_migration.md` | Complete migration details |
| **Dataset README** | `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/raw_data/LARGEST/README.md` | Dataset information |

## Quick Usage

### Load in LibCity
```python
from libcity.data import get_dataset

config = {
    "dataset": "LARGEST",
    "task": "traffic_state_pred",
    "batch_size": 32
}

dataset = get_dataset(config)
train_data, val_data, test_data = dataset.get_data()
```

### Train a Model
```bash
python run_model.py \
    --task traffic_state_pred \
    --dataset LARGEST \
    --model DCRNN \
    --batch_size 32
```

## Preprocessing Applied

1. **Sensor Filtering:** Removed 517 sensors with >50% missing data (8,600 → 8,083)
2. **Outlier Capping:** Capped speeds at 120 mph (68% of values affected)
3. **Missing Values:** Converted NaN to 0.0 (0.73% of data)
4. **Graph Construction:** k-NN graph (k=10) based on sensor index proximity

## Important Notes

1. **No spatial coordinates** - Coordinates field is empty in .geo file
2. **Approximate graph** - k-NN based on indices, not true road network
3. **Large file size** - 38 GB .dyna file may require memory management
4. **Missing values** - 0.73% converted to 0.0, consider masked loss

## Comparison with PEMSD7

| Metric | PEMSD7 | LARGEST | Ratio |
|--------|--------|---------|-------|
| Nodes | 883 | 8,083 | 9.2x |
| Timesteps | 16,992 | 105,120 | 6.2x |
| Records | 15.0M | 849.7M | 56.6x |
| Duration | 2 months | 12 months | 6.0x |
| Size | ~15 MB | ~38 GB | 2,533x |

## Validation Results

All checks passed ✓

- File format: Valid
- Schema compliance: Valid
- Data integrity: Valid
- Cross-file consistency: Valid
- Expected record count: Matched (849,684,960)

## Next Steps

1. Test with a small LibCity model (e.g., DCRNN, GRU)
2. Consider subsampling for faster experimentation
3. Monitor memory usage during training
4. Obtain sensor coordinates if possible for better graph structure

---

**Conversion Date:** 2026-02-06
**Conversion Time:** ~27 minutes
**Validation Status:** All checks passed ✓
