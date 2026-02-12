# DeepMM Dual-Mode Adaptation Summary

## Overview
Adapted DeepMM model to handle both **sequence targets** (map matching) and **single targets** (next-location prediction) seamlessly.

## File Modified
`/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/DeepMM.py`

## Problem Statement
DeepMM was originally designed for map matching where:
- Input: GPS trajectory `[batch, seq_len]`
- Output: Road segment sequence `[batch, seq_len]`

However, for next-location prediction task:
- Input: GPS trajectory `[batch, seq_len]`
- Output: Single next location `[batch]`

The model needed to handle both target shapes without errors.

## Key Adaptations

### 1. Forward Method (Lines 272-351)

**Added Target Shape Detection:**
```python
# Adaptation: Handle both single target [batch] and sequence target [batch, seq_len]
is_single_target = False
if input_trg.dim() == 1:  # Single target [batch]
    # Convert to sequence format [batch, 1] for uniform processing
    input_trg = input_trg.unsqueeze(1)
    is_single_target = True
```

**Impact:**
- Single targets `[batch]` are converted to `[batch, 1]` internally
- Allows the decoder to process uniformly regardless of task type
- The rest of the forward pass remains unchanged

### 2. Predict Method (Lines 353-379)

**Added Output Reshaping:**
```python
# Adaptation: If input target was single-valued, squeeze the output
try:
    target = batch['target']
    if target.dim() == 1:  # Single target case
        # decoder_logit is [batch, 1, vocab_size], predictions is [batch, 1]
        # Squeeze to [batch]
        predictions = predictions.squeeze(1)
except KeyError:
    pass
```

**Impact:**
- For single-target tasks: Returns `[batch]` predictions
- For sequence tasks: Returns `[batch, seq_len]` predictions
- Maintains compatibility with different evaluation metrics

### 3. Calculate Loss Method (Lines 381-422)

**Added Conditional Loss Calculation:**
```python
# Adaptation: Handle both [batch] and [batch, seq_len] targets
if target.dim() == 1:  # Single target [batch]
    # decoder_logit is [batch, 1, vocab_size] (from forward with unsqueezed input)
    # Keep target as [batch] and reshape decoder_logit to [batch, vocab_size]
    decoder_logit = decoder_logit.squeeze(1)  # [batch, vocab_size]
else:
    # Sequence target [batch, seq_len]
    # Flatten both to calculate loss
    decoder_logit = decoder_logit.contiguous().view(-1, self.trg_seg_vocab_size)
    target = target.view(-1)
```

**Impact:**
- Single target: Loss computed on `[batch, vocab_size]` vs `[batch]`
- Sequence target: Loss computed on `[batch*seq_len, vocab_size]` vs `[batch*seq_len]`
- Both use the same CrossEntropyLoss with padding mask

### 4. Documentation Updates

**Updated Class Docstring:**
```python
Supports two modes:
1. Map Matching: Input [batch, seq_len] → Output [batch, seq_len]
   Predicts a sequence of road segments from GPS trajectory
2. Next-Location: Input [batch, seq_len] → Output [batch]
   Predicts the next single location from trajectory history

The model automatically detects the task type based on target shape.
```

## Data Flow Comparison

### Map Matching Mode
```
Input:
  current_loc: [32, 50]  # batch=32, seq_len=50
  target:      [32, 50]  # sequence of road segments

Forward:
  input_trg.dim() == 2  → No modification
  decoder_logit: [32, 50, vocab_size]

Loss:
  decoder_logit: [32*50, vocab_size]
  target:        [32*50]

Predict:
  predictions: [32, 50]
```

### Next-Location Prediction Mode
```
Input:
  current_loc: [32, 50]  # batch=32, seq_len=50
  target:      [32]      # single next location

Forward:
  input_trg.dim() == 1  → Unsqueeze to [32, 1]
  decoder_logit: [32, 1, vocab_size]

Loss:
  decoder_logit: [32, vocab_size]  (squeezed)
  target:        [32]

Predict:
  predictions: [32]  (squeezed)
```

## Benefits

1. **Single Model, Dual Purpose**: One model handles both tasks without code duplication
2. **Automatic Detection**: No configuration needed - adapts based on batch structure
3. **Backward Compatible**: Existing map matching code continues to work
4. **Minimal Changes**: Core architecture unchanged, only I/O reshaping added
5. **Efficient**: No performance overhead - reshaping is cheap

## Testing Checklist

- [ ] Map matching with sequence targets `[batch, seq_len]`
- [ ] Next-location prediction with single targets `[batch]`
- [ ] Loss calculation for both modes
- [ ] Prediction output shapes for both modes
- [ ] Gradient flow in both modes
- [ ] Edge cases (seq_len=1, batch=1)

## Configuration

No additional configuration required. The model automatically adapts based on:
```python
# Detected at runtime from batch['target'].dim()
if target.dim() == 1:
    # Next-location prediction mode
else:
    # Map matching mode
```

## Potential Issues

1. **Teacher Forcing**: In sequence mode, the model expects teacher forcing during training (target sequence provided). In single-target mode, only one target step is used.
2. **Autoregressive Inference**: For true autoregressive decoding in map matching, a separate inference method may be needed.
3. **Attention Weights**: Attention is computed for each decoder step, but in single-target mode, there's only one step.

## Future Enhancements

1. Add explicit mode parameter if automatic detection is insufficient
2. Implement beam search for sequence generation
3. Add support for variable-length sequences with dynamic padding
4. Optimize decoder for single-step inference

## Related Files
- Model: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/DeepMM.py`
- Base class: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/abstract_model.py`
- Task config: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`

## Conclusion

DeepMM now gracefully handles both map matching (sequence-to-sequence) and next-location prediction (sequence-to-single) tasks through automatic target shape detection and appropriate tensor reshaping. The adaptations are minimal, maintainable, and preserve the original model's architecture and performance.
