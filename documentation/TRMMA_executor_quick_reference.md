# Quick Reference: Neural Map Matching Executor Fix

## What Changed

### Neural Map Matching Models Now Use Correct Executor

**Models Affected:**
- TRMMA
- DeepMM
- DiffMM

**Previous Configuration (BROKEN):**
```json
"TRMMA": {
    "executor": "MapMatchingExecutor"  // ❌ For traditional models only
}
```

**New Configuration (FIXED):**
```json
"TRMMA": {
    "executor": "DeepMapMatchingExecutor"  // ✅ For neural models
}
```

## Why This Fix Was Needed

### The Problem
Neural map matching models inherit from `AbstractModel` with:
- `forward()` method for forward pass
- `predict()` method for inference
- `calculate_loss()` method for training

But they were configured to use `MapMatchingExecutor`, which expects:
- `run()` method (which neural models don't have)
- No training capability
- Traditional model interface

### The Error
```
AttributeError: 'TRMMA' object has no attribute 'run'
```

## The Solution: DeepMapMatchingExecutor

### New Executor Created
**File:** `libcity/executor/deep_map_matching_executor.py`

**Capabilities:**
✅ Training with gradient descent
✅ Validation during training
✅ Early stopping
✅ Model checkpointing
✅ Uses `calculate_loss()` and `predict()`
✅ Compatible with MapMatchingDataset
✅ Works with MapMatchingEvaluator

### Architecture Inheritance
```
AbstractExecutor
  └─ DeepMapMatchingExecutor (NEW)
       ├─ train()
       ├─ evaluate()
       ├─ run() - training loop
       ├─ _valid_epoch()
       └─ optimizer/scheduler management
```

## How to Use

### Training Example
```python
# Configuration automatically uses DeepMapMatchingExecutor
python run_model.py \
    --task map_matching \
    --model TRMMA \
    --dataset Seattle \
    --learning_rate 0.001 \
    --max_epoch 100
```

### Evaluation Example
```python
python run_model.py \
    --task map_matching \
    --model TRMMA \
    --dataset Seattle \
    --test_only
```

## Comparison: Traditional vs Neural Executors

| Feature | MapMatchingExecutor | DeepMapMatchingExecutor |
|---------|-------------------|------------------------|
| Training | ❌ No | ✅ Yes |
| Model Type | Traditional | Neural Network |
| Required Methods | `run()` | `forward()`, `predict()`, `calculate_loss()` |
| Base Class | AbstractTraditionExecutor | AbstractExecutor |
| Optimizer | ❌ No | ✅ Yes |
| Backpropagation | ❌ No | ✅ Yes |
| Models | STMatching, IVMM, HMMM, FMM, STMatch | TRMMA, DeepMM, DiffMM |

## Files Modified

1. **Created:** `libcity/executor/deep_map_matching_executor.py`
2. **Updated:** `libcity/executor/__init__.py`
3. **Updated:** `libcity/config/task_config.json`

## Testing Checklist

- [ ] TRMMA trains without errors
- [ ] DeepMM trains without errors
- [ ] DiffMM trains without errors
- [ ] Validation metrics are computed
- [ ] Model checkpoints are saved
- [ ] Evaluation works correctly
- [ ] No regression on traditional models (STMatching, etc.)

## Key Insight

**The Fundamental Issue:**
Map matching has TWO types of models:
1. **Traditional:** Rule-based, no training → MapMatchingExecutor
2. **Neural:** Deep learning, requires training → DeepMapMatchingExecutor

The fix ensures each type uses the appropriate executor.

## Support for Future Models

Any new neural map matching model should use:
```json
"YourNeuralMapMatchingModel": {
    "dataset_class": "MapMatchingDataset",
    "executor": "DeepMapMatchingExecutor",  // ← Use this for neural models
    "evaluator": "MapMatchingEvaluator"
}
```

Requirements:
- Inherit from `AbstractModel`
- Implement `forward()`, `predict()`, `calculate_loss()`
- Use MapMatchingDataset format

## Additional Notes

- Traditional map matching models (STMatching, IVMM, etc.) continue to use `MapMatchingExecutor`
- No changes needed to existing model implementations
- The executor handles all training/evaluation logic
- Compatible with LibCity's hyperparameter tuning framework
