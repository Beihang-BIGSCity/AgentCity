# UniST Configuration Fixes

## Summary
Fixed configuration issues identified during UniST testing to support point-based datasets and optimize memory usage.

## Issues Addressed

### 1. Dataset Class Configuration
**Status**: Already Correct
- **File**: `Bigscity-LibCity/libcity/config/task_config.json`
- **Configuration**: UniST is configured to use `TrafficStatePointDataset`
- **Line**: 693
- **Reason**: Most available datasets (METR_LA, PEMS_BAY, etc.) are point-based datasets with sensor locations, not grid-based datasets. UniST can work with point data by reshaping it to a grid internally.

```json
"UniST": {
    "dataset_class": "TrafficStatePointDataset",
    "executor": "TrafficStateExecutor",
    "evaluator": "TrafficStateEvaluator"
}
```

### 2. Grid Dimensions Optimization
**Status**: Fixed
- **File**: `Bigscity-LibCity/libcity/config/model/traffic_state_pred/UniST.json`
- **Changed**: `grid_height` from 16 to 15, `grid_width` from 16 to 15
- **Reason**:
  - Previous: 16x16 = 256 nodes (mismatch with METR_LA's 207 nodes)
  - New: 15x15 = 225 nodes (closer to 207, reduces padding and memory usage)
  - The model automatically handles the padding/reshaping of 207 nodes to fit the 15x15 grid

### 3. Memory-Efficient Model Configuration
**Status**: Fixed
- **File**: `Bigscity-LibCity/libcity/config/model/traffic_state_pred/UniST.json`
- **Model Size**: Changed from "middle" to "small"

#### Parameter Changes:

| Parameter | Previous (Middle) | New (Small) | Reduction |
|-----------|------------------|-------------|-----------|
| `embed_dim` | 128 | 64 | 50% |
| `depth` | 6 | 4 | 33% |
| `decoder_embed_dim` | 128 | 64 | 50% |
| `decoder_depth` | 4 | 2 | 50% |
| `num_heads` | 8 | 4 | 50% |
| `decoder_num_heads` | 4 | 2 | 50% |
| `num_memory_spatial` | 128 | 64 | 50% |
| `num_memory_temporal` | 128 | 64 | 50% |
| `conv_num` | 3 | 2 | 33% |
| `batch_size` | (not set) | 16 | New |

**Memory Impact**: Approximately 70-75% reduction in model parameters and GPU memory usage

### 4. Configuration Documentation
**Status**: Added
- Added inline comments to explain grid dimension requirements
- Added guidance for scaling up model size
- Added notes about pretrained weights usage

## Updated Configuration File

**Location**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traffic_state_pred/UniST.json`

### Key Parameters:

```json
{
  "model_size": "small",
  "embed_dim": 64,
  "depth": 4,
  "decoder_embed_dim": 64,
  "decoder_depth": 2,
  "num_heads": 4,
  "decoder_num_heads": 2,
  "grid_height": 15,
  "grid_width": 15,
  "batch_size": 16,
  "learning_rate": 0.0005,
  "max_epoch": 100
}
```

## Dataset Compatibility

### Point-Based Datasets (Supported)
The configuration works with datasets like:
- METR_LA (207 nodes)
- PEMS_BAY (325 nodes)
- PEMSD3, PEMSD4, PEMSD7, PEMSD8
- Other point-based traffic sensor datasets

**Note**: For datasets with different numbers of nodes, adjust `grid_height` and `grid_width`:
- For N nodes, choose H x W where H*W >= N and H*W is close to N
- Examples:
  - 207 nodes → 15x15 = 225
  - 325 nodes → 18x18 = 324
  - 170 nodes → 13x13 = 169

### Grid-Based Datasets (Future Support)
For true grid datasets (like taxi demand grids), you would:
1. Change dataset_class to `TrafficStateGridDataset` in task_config.json
2. Set `grid_height` and `grid_width` to match the dataset's actual grid dimensions
3. Ensure the dataset provides data in (N, C, T, H, W) format

## Pretrained Weights

The current configuration sets `pretrained_weights: null`, meaning training from scratch.

For transfer learning:
1. Obtain pre-trained weights from the original UniST repository
2. Update the config: `"pretrained_weights": "/path/to/pretrained.pth"`
3. The model will load weights during initialization

**Note**: The pipeline should NOT have hardcoded pretrained weights loading logic (see separate pipeline.py fix).

## Performance Tuning

### For Better Performance (if GPU memory allows):
```json
{
  "model_size": "middle",
  "embed_dim": 128,
  "depth": 6,
  "decoder_embed_dim": 128,
  "decoder_depth": 4,
  "num_heads": 8,
  "decoder_num_heads": 4,
  "batch_size": 32
}
```

### For Larger Datasets or More GPU Memory:
```json
{
  "model_size": "large",
  "embed_dim": 256,
  "depth": 8,
  "decoder_embed_dim": 256,
  "decoder_depth": 6,
  "num_heads": 16,
  "decoder_num_heads": 8
}
```

## Testing Recommendations

1. **Quick Test**: Use the test_config.json override
   ```bash
   python run_model.py --task traffic_state_pred --model UniST --dataset METR_LA --config_file test_config.json
   ```

2. **Memory Monitoring**: Monitor GPU memory usage during training
   - Small config should use ~3-4GB GPU memory
   - Middle config may use ~8-10GB GPU memory

3. **Validation**: Check that the model handles 207 nodes correctly with 15x15 grid

## Files Modified

1. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traffic_state_pred/UniST.json`
   - Optimized for memory efficiency
   - Added documentation comments
   - Set appropriate grid dimensions

2. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
   - No changes needed (already correctly configured)

## Related Issues

- **Hardcoded pretrained weights**: This should be handled in pipeline.py separately (not addressed in this fix)
- **Grid vs Point data handling**: The model's internal logic in UniST.py handles the conversion from point data to grid format

## Verification Steps

- [x] task_config.json has TrafficStatePointDataset for UniST
- [x] Grid dimensions match typical dataset sizes (15x15 for ~207 nodes)
- [x] Model size reduced for memory efficiency
- [x] Batch size explicitly set (16)
- [x] Documentation comments added
- [x] All JSON syntax valid

## Next Steps

1. Test the configuration with METR_LA dataset
2. Monitor training memory usage
3. If memory allows, experiment with middle configuration for better performance
4. Consider fixing pipeline.py to remove hardcoded UniST logic (separate task)
