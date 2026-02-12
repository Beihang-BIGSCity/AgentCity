# DeepMM Model Fixes Summary

## File Location
**Path**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/DeepMM.py`

## Issues Fixed

### Issue 1: BatchPAD Object Access Error

**Problem**: The `BatchPAD` object does not support the `.get()` method, causing `AttributeError`.

**Locations Fixed**:

#### Line 283-288 (forward method)
**Before**:
```python
input_src = batch['current_loc']  # [batch, seq_len]
input_trg = batch.get('target_seg', batch.get('target', input_src))  # [batch, seq_len]
```

**After**:
```python
input_src = batch['current_loc']  # [batch, seq_len]
# Fix: BatchPAD object doesn't support .get() method - use dictionary access with try-except
try:
    input_trg = batch['target']
except KeyError:
    input_trg = input_src  # Fallback to source if target not available
```

#### Line 360-364 (calculate_loss method)
**Before**:
```python
decoder_logit = self.forward(batch)

# Get target sequence
target = batch.get('target', batch.get('target_seg'))
```

**After**:
```python
decoder_logit = self.forward(batch)

# Get target sequence
# Fix: BatchPAD object doesn't support .get() method - use direct access
target = batch['target']
```

### Issue 2: Vocabulary Size Mismatch

**Problem**: The model was configured for map matching (GPS → road segments) but is being used for trajectory location prediction (locations → locations). Both source and target should use the same location vocabulary.

**Location Fixed**: Lines 180-187 (__init__ method)

**Before**:
```python
# Get vocabulary sizes from data_feature
self.src_loc_vocab_size = data_feature.get('loc_size', 10000)
self.trg_seg_vocab_size = data_feature.get('road_num', 10000)

# Get padding indices
self.pad_token_src_loc = data_feature.get('loc_pad', 0)
self.pad_token_trg = data_feature.get('road_pad', 0)
```

**After**:
```python
# Get vocabulary sizes from data_feature
# For trajectory location prediction, both source and target use location vocabulary
self.src_loc_vocab_size = data_feature.get('loc_size', 10000)
self.trg_seg_vocab_size = data_feature.get('loc_size', 10000)  # Use loc_size instead of road_num

# Get padding indices
self.pad_token_src_loc = data_feature.get('loc_pad', 0)
self.pad_token_trg = data_feature.get('loc_pad', 0)  # Use loc_pad instead of road_pad
```

## Changes Summary

### 1. Batch Access Pattern
- **Changed from**: `.get()` method calls on BatchPAD object
- **Changed to**: Direct dictionary access with try-except blocks
- **Reason**: BatchPAD objects do not implement the `.get()` method from the dict interface

### 2. Vocabulary Configuration
- **Changed from**: Using `road_num` and `road_pad` from data_feature
- **Changed to**: Using `loc_size` and `loc_pad` for both source and target
- **Reason**: Trajectory location prediction task uses location IDs for both input and output, not road segment IDs

## Impact

These fixes resolve:
1. **AttributeError**: 'BatchPAD' object has no attribute 'get'
2. **Vocabulary mismatch**: Model now correctly uses location vocabulary for trajectory location prediction
3. **Padding index consistency**: Both source and target use the same padding index

## Testing Recommendations

1. Test with trajectory location prediction dataset
2. Verify that batch dictionary contains 'current_loc' and 'target' keys
3. Check that vocabulary sizes match between model initialization and data
4. Confirm loss calculation works with proper padding masking

## Configuration Requirements

The model expects the following in `data_feature`:
- `loc_size`: Vocabulary size for location IDs
- `loc_pad`: Padding index for location sequences

The model expects the following in batch dictionary:
- `current_loc`: Source location sequence [batch, seq_len]
- `target`: Target location sequence [batch, seq_len]

## Date
2026-02-06
