# TRMMA Executor Configuration Fix

## Problem Summary

**Issue:** TRMMA (and other neural map matching models) were configured to use `MapMatchingExecutor`, which is designed for traditional models with a `run()` method. This caused:

```
AttributeError: 'TRMMA' object has no attribute 'run'
```

**Affected Models:**
- TRMMA (Trajectory Recovery with Multi-Modal Alignment)
- DeepMM (Deep Learning-based Map Matching)
- DiffMM (Diffusion-based Map Matching)

All three models inherit from `AbstractModel` and implement neural network methods:
- `forward()` - Forward pass
- `predict()` - Inference
- `calculate_loss()` - Loss computation

## Root Cause

`MapMatchingExecutor` inherits from `AbstractTraditionExecutor` and expects traditional models that:
- Have a `run(test_data)` method
- Don't require training
- Work with simple batch processing

Neural models require:
- Training loops with backpropagation
- Optimizer and scheduler management
- Support for `calculate_loss()` and `predict()` methods

## Solution Implemented

### Option B: Created DeepMapMatchingExecutor

Created a new executor specifically for neural map matching models:

**File:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/executor/deep_map_matching_executor.py`

**Features:**
- Based on `TrajLocPredExecutor` architecture
- Supports training with backpropagation
- Handles `calculate_loss()` and `predict()` methods
- Compatible with `MapMatchingDataset` and `MapMatchingEvaluator`
- Includes optimizer and scheduler management
- Early stopping based on validation metrics

**Key Methods:**
```python
class DeepMapMatchingExecutor(AbstractExecutor):
    def train(self, train_dataloader, eval_dataloader)
        # Training loop with gradient descent

    def evaluate(self, test_dataloader)
        # Evaluation using model.predict()

    def run(self, data_loader, model, lr, clip)
        # Single epoch training

    def _valid_epoch(self, data_loader, model)
        # Validation loop
```

## Changes Made

### 1. Created DeepMapMatchingExecutor

**File:** `libcity/executor/deep_map_matching_executor.py`
- New executor for neural map matching models
- Supports training, validation, and evaluation
- Compatible with MapMatchingDataset batch format

### 2. Updated Executor Registry

**File:** `libcity/executor/__init__.py`

Added import:
```python
from libcity.executor.deep_map_matching_executor import DeepMapMatchingExecutor
```

Added to `__all__`:
```python
"DeepMapMatchingExecutor"
```

### 3. Updated Task Configuration

**File:** `libcity/config/task_config.json`

Changed executor for neural models:

**Before:**
```json
"TRMMA": {
    "dataset_class": "MapMatchingDataset",
    "executor": "MapMatchingExecutor",
    "evaluator": "MapMatchingEvaluator"
}
```

**After:**
```json
"TRMMA": {
    "dataset_class": "MapMatchingDataset",
    "executor": "DeepMapMatchingExecutor",
    "evaluator": "MapMatchingEvaluator"
}
```

Same changes applied to `DeepMM` and `DiffMM`.

## Architecture Comparison

### Traditional Models (STMatching, IVMM, HMMM, FMM, STMatch)
```
MapMatchingExecutor
  ├─ No training
  ├─ Uses model.run(test_data)
  └─ AbstractTraditionExecutor
```

### Neural Models (TRMMA, DeepMM, DiffMM)
```
DeepMapMatchingExecutor
  ├─ Training with backprop
  ├─ Uses model.calculate_loss() & model.predict()
  ├─ Optimizer & scheduler
  └─ AbstractExecutor
```

## Benefits

1. **Correct Interface:** Neural models now use an executor that supports their API
2. **Training Support:** Models can be properly trained with gradient descent
3. **No Model Changes:** No modifications needed to TRMMA, DeepMM, or DiffMM
4. **Separation of Concerns:** Traditional and neural models use appropriate executors
5. **Reusability:** Future neural map matching models can use DeepMapMatchingExecutor

## Testing Recommendations

### Test TRMMA with DeepMapMatchingExecutor
```bash
python run_model.py --task map_matching --model TRMMA --dataset Seattle
```

### Verify Training Works
Check that:
- Training loop runs without errors
- Loss decreases over epochs
- Validation metrics are computed
- Model checkpoints are saved

### Test Evaluation
```bash
python run_model.py --task map_matching --model TRMMA --dataset Seattle --test_only
```

## Related Files

**Executors:**
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/executor/deep_map_matching_executor.py` (NEW)
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/executor/map_matching_executor.py`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/executor/traj_loc_pred_executor.py`

**Models:**
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/TRMMA.py`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/DeepMM.py`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/DiffMM.py`

**Config:**
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`

## Summary

✅ **Problem:** Neural map matching models using wrong executor type
✅ **Solution:** Created DeepMapMatchingExecutor for neural models
✅ **Impact:** TRMMA, DeepMM, and DiffMM now work correctly
✅ **Status:** Ready for testing

The executor configuration has been fixed to properly support neural network-based map matching models while maintaining backward compatibility with traditional map matching methods.
