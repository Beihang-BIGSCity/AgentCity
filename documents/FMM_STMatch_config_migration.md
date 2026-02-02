## Config Migration: FMM and STMatch

### Migration Date
2026-02-01

### Overview
Successfully configured FMM (Fast Map Matching) and STMatch models in LibCity's task configuration system. Both models are HMM-based map matching algorithms adapted from the FMM C++ library.

---

## Configuration Status

### 1. task_config.json Updates

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`

**Changes Made**:
- Added `FMM` to `map_matching.allowed_model` list (line 1037)
- Added `STMatch` to `map_matching.allowed_model` list (line 1038)
- Added FMM task configuration (lines 1059-1063)
- Added STMatch task configuration (lines 1064-1068)

**Task Configuration**:
```json
"map_matching": {
    "allowed_model": [
        "STMatching",
        "IVMM",
        "HMMM",
        "FMM",
        "STMatch"
    ],
    "allowed_dataset": [
        "global",
        "Seattle"
    ],
    "FMM": {
        "dataset_class": "MapMatchingDataset",
        "executor": "MapMatchingExecutor",
        "evaluator": "MapMatchingEvaluator"
    },
    "STMatch": {
        "dataset_class": "MapMatchingDataset",
        "executor": "MapMatchingExecutor",
        "evaluator": "MapMatchingEvaluator"
    }
}
```

---

## 2. Model Configuration Files

### FMM Model Config
**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/map_matching/FMM.json`

**Status**: ✓ Verified and correct

**Parameters**:
```json
{
  "k": 8,                    // Number of candidate edges per GPS point
  "r": 300,                  // Search radius in meters
  "gps_error": 50,          // GPS measurement error std dev (meters)
  "delta": 3000,            // Upper bound for UBODT precomputation (meters)
  "reverse_tolerance": 0.0, // Allowed proportion of reverse movement
  "use_ubodt": true         // Enable precomputed UBODT table
}
```

**Parameter Sources**:
- `k: 8` - From FMM paper default
- `r: 300` - From FMM paper default
- `gps_error: 50` - From FMM paper default (50m GPS error)
- `delta: 3000` - From FMM paper default (3km upper bound)
- `reverse_tolerance: 0.0` - From FMM implementation
- `use_ubodt: true` - FMM-specific feature

### STMatch Model Config
**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/map_matching/STMatch.json`

**Status**: ✓ Verified and correct

**Parameters**:
```json
{
  "k": 8,                    // Number of candidate edges per GPS point
  "r": 300,                  // Search radius in meters
  "gps_error": 50,          // GPS measurement error std dev (meters)
  "vmax": 30,               // Maximum vehicle speed (m/s)
  "factor": 1.5,            // Search bound factor
  "reverse_tolerance": 0.0  // Allowed proportion of reverse movement
}
```

**Parameter Sources**:
- `k: 8` - From FMM paper default
- `r: 300` - From FMM paper default
- `gps_error: 50` - From FMM paper default
- `vmax: 30` - From STMatch algorithm (30 m/s = 108 km/h)
- `factor: 1.5` - From FMM library default
- `reverse_tolerance: 0.0` - From FMM implementation

**Key Difference from FMM**: STMatch uses on-the-fly shortest path computation with bounded Dijkstra instead of precomputed UBODT, making it more memory-efficient but potentially slower for dense trajectories.

---

## 3. Model Implementation Files

### FMM Model
**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/FMM.py`

**Status**: ✓ Implemented

**Key Features**:
- Inherits from `AbstractTraditionModel`
- Implements UBODT (Upper Bounded Origin Destination Table) precomputation
- Uses HMM-based map matching with Viterbi algorithm
- Supports spatial indexing for efficient candidate search
- Emission probability based on Gaussian distribution
- Transition probability based on network distance vs Euclidean distance

**Algorithm Flow**:
1. Precompute UBODT for shortest paths (if enabled)
2. Find k-nearest candidate edges for each GPS point
3. Build transition graph with emission probabilities
4. Update transition graph using Viterbi algorithm
5. Backtrack to find optimal path

### STMatch Model
**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/STMatch.py`

**Status**: ✓ Implemented

**Key Features**:
- Inherits from `AbstractTraditionModel`
- On-the-fly shortest path computation using bounded Dijkstra
- Composite graph construction with dummy nodes for candidates
- Time-aware search bounds: delta = factor * vmax * delta_time
- More suitable for sparse trajectories with large time gaps

**Algorithm Flow**:
1. Find candidates and build composite graph
2. Build transition graph with emission probabilities
3. Update transitions using bounded Dijkstra for each layer pair
4. Backtrack to find optimal path

---

## 4. Model Registration

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/__init__.py`

**Status**: ✓ Registered

Both models are properly imported and exported:
```python
from libcity.model.map_matching.FMM import FMM
from libcity.model.map_matching.STMatch import STMatch

__all__ = [
    "STMatching",
    "IVMM",
    "HMMM",
    "FMM",
    "STMatch"
]
```

---

## 5. Dataset Compatibility

### MapMatchingDataset
**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/data/MapMatchingDataset.json`

**Required Data Features**:
```json
{
  "delta_time": true
}
```

### Data Requirements

**GPS Trajectories**:
- Format: Array with columns [dyna_id, lon, lat, time, ...]
- Minimum: longitude, latitude, timestamp
- Expected format: NumPy array or similar

**Road Network**:
- Type: NetworkX DiGraph
- Node attributes: `lat`, `lon`
- Edge attributes: `distance` or `weight`, optional `geo_id`
- Must be directed graph for proper path computation

**Allowed Datasets**:
- `global` - General map matching dataset
- `Seattle` - Seattle-specific dataset

---

## 6. Executor and Evaluator

### MapMatchingExecutor
- Handles trajectory preprocessing
- Manages road network loading
- Coordinates model execution
- Produces matched sequences

### MapMatchingEvaluator
- Evaluates matching accuracy
- Common metrics:
  - Precision: Correctly matched points / Total matched points
  - Recall: Correctly matched points / Total GPS points
  - F1-Score: Harmonic mean of precision and recall
  - Route mismatch distance

---

## 7. Usage Example

### Running FMM
```python
from libcity.pipeline import run_model

run_model(task='map_matching',
          model_name='FMM',
          dataset_name='Seattle',
          config_file='FMM.json')
```

### Running STMatch
```python
from libcity.pipeline import run_model

run_model(task='map_matching',
          model_name='STMatch',
          dataset_name='Seattle',
          config_file='STMatch.json')
```

### Custom Configuration
```python
custom_config = {
    'k': 10,           # More candidates
    'r': 500,          # Larger search radius
    'gps_error': 100,  # Higher GPS error tolerance
    'delta': 5000      # Larger UBODT bound (FMM only)
}
```

---

## 8. Algorithm Comparison

| Feature | FMM | STMatch |
|---------|-----|---------|
| Precomputation | UBODT table | None |
| Memory Usage | Higher | Lower |
| Speed (dense) | Faster | Slower |
| Speed (sparse) | Similar | Similar |
| Time-aware | No | Yes (via vmax) |
| Best for | Dense trajectories | Sparse trajectories |

---

## 9. Hyperparameter Tuning Guidelines

### k (Number of Candidates)
- **Default**: 8
- **Range**: 4-16
- **Impact**: Higher k increases robustness but slows computation
- **Recommendation**: 8 for urban roads, 4-6 for highways

### r (Search Radius)
- **Default**: 300m
- **Range**: 100-500m
- **Impact**: Must cover expected GPS error range
- **Recommendation**: 300m for urban, 200m for highways

### gps_error (GPS Error Std Dev)
- **Default**: 50m
- **Range**: 20-100m
- **Impact**: Affects emission probability sensitivity
- **Recommendation**: 50m for typical GPS, 100m for poor reception

### delta (UBODT Bound - FMM only)
- **Default**: 3000m
- **Range**: 1000-5000m
- **Impact**: Larger values increase precomputation time/memory
- **Recommendation**: 3000m for city blocks, 5000m for sparse networks

### vmax (Max Speed - STMatch only)
- **Default**: 30 m/s (108 km/h)
- **Range**: 15-40 m/s
- **Impact**: Affects search bound calculation
- **Recommendation**: 20 m/s for urban, 35 m/s for highways

---

## 10. Known Issues and Limitations

### FMM
1. **Memory intensive**: UBODT table can be large for dense networks
2. **Precomputation time**: Initial UBODT building takes time
3. **Not time-aware**: Doesn't use temporal information for search bounds

### STMatch
1. **Slower for dense trajectories**: On-the-fly Dijkstra for each transition
2. **Requires timestamps**: Needs temporal data for optimal performance
3. **Parameter sensitivity**: vmax and factor need tuning per dataset

### General
1. Both models assume directed road networks
2. Edge weights should represent real distances (meters)
3. GPS trajectories should be pre-sorted by timestamp
4. Performance degrades with very noisy GPS data

---

## 11. Validation Checklist

- ✓ Models registered in `task_config.json`
- ✓ Model configs created with correct parameters
- ✓ Models imported in `__init__.py`
- ✓ Models inherit from `AbstractTraditionModel`
- ✓ Dataset class configured (`MapMatchingDataset`)
- ✓ Executor configured (`MapMatchingExecutor`)
- ✓ Evaluator configured (`MapMatchingEvaluator`)
- ✓ Parameter sources documented
- ✓ JSON syntax validated

---

## 12. Testing Recommendations

### Unit Tests
1. Test candidate search with known GPS points
2. Test UBODT construction on small networks
3. Test emission probability calculation
4. Test transition probability calculation
5. Test Viterbi backtracking

### Integration Tests
1. Run on synthetic trajectory with known ground truth
2. Test with different GPS error levels
3. Test with varying trajectory densities
4. Compare FMM vs STMatch performance
5. Test memory usage with large networks

### Benchmark Tests
1. Seattle dataset validation
2. Comparison with existing STMatching model
3. Processing time vs trajectory length
4. Memory usage vs network size
5. Accuracy metrics on standard datasets

---

## 13. References

### Papers
- Can Yang and Gyozo Gidofalvi. "Fast map matching, an algorithm integrating hidden Markov model with precomputation." International Journal of Geographical Information Science 32.3 (2018): 547-570.

### Implementation
- Original FMM C++ library: https://github.com/cyang-kth/fmm
- LibCity framework: https://github.com/LibCity/Bigscity-LibCity

### Related Models in LibCity
- `STMatching`: Existing spatial-temporal map matching
- `IVMM`: Incremental Viterbi map matching
- `HMMM`: Hidden Markov Model map matching

---

## 14. Configuration Summary

### Files Created/Modified

1. **Modified**: `Bigscity-LibCity/libcity/config/task_config.json`
   - Added FMM and STMatch to map_matching task

2. **Verified**: `Bigscity-LibCity/libcity/config/model/map_matching/FMM.json`
   - Contains FMM hyperparameters

3. **Verified**: `Bigscity-LibCity/libcity/config/model/map_matching/STMatch.json`
   - Contains STMatch hyperparameters

4. **Verified**: `Bigscity-LibCity/libcity/model/map_matching/__init__.py`
   - Both models properly registered

5. **Verified**: `Bigscity-LibCity/libcity/model/map_matching/FMM.py`
   - FMM implementation complete

6. **Verified**: `Bigscity-LibCity/libcity/model/map_matching/STMatch.py`
   - STMatch implementation complete

### Configuration Ready for Testing
✓ All configuration files are in place and validated
✓ Models are registered in the task configuration system
✓ Hyperparameters are set to paper defaults
✓ Dataset compatibility verified
✓ Executor and evaluator properly configured

---

## 15. Next Steps

1. **Testing**: Run integration tests with Seattle dataset
2. **Validation**: Compare results with original FMM implementation
3. **Benchmarking**: Measure performance on standard datasets
4. **Documentation**: Update user documentation with usage examples
5. **Optimization**: Profile and optimize performance bottlenecks

---

## Contact and Support

For issues or questions about this migration:
- Check LibCity documentation
- Review original FMM repository
- Consult map matching literature for algorithm details

---

**Migration Status**: ✓ Complete and Ready for Testing
**Configuration Version**: 1.0
**Last Updated**: 2026-02-01
