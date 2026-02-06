# DiffMM Batch Format Mismatch Fix

## Issue Description

The DiffMM model's `forward()`, `infer()`, and `predict()` methods expected different batch keys than those provided by the `DiffMMDataset`.

### Original Model Expected Keys
- `'src'` - GPS points
- `'src_len'` - Sequence lengths
- `'src_segs'` - Candidate segment IDs
- `'target'` - Target one-hot distribution
- `'segs_feat'` - Segment features
- `'segs_mask'` - Validity mask

### Actual Dataset Batch Keys (from `diffmm_collate_fn`)
- `'norm_gps_seq'` - GPS points (batch, seq_len, 3)
- `'lengths'` - Sequence lengths (batch,)
- `'segs_id'` - Candidate segment IDs (batch, seq_len, num_cands)
- `'trg_onehot'` - Target one-hot over all roads (batch, seq_len, num_roads)
- `'trg_rid'` - Target road segment indices (batch, seq_len)
- `'segs_feat'` - Segment features (batch, seq_len, num_cands, 9)
- `'segs_mask'` - Validity mask (batch, seq_len, num_cands)

## Additional Issue: Target Dimension Mismatch

The dataset's `trg_onehot` has shape `(batch, seq_len, num_roads)` representing a one-hot vector over the entire road vocabulary. However, the DiffMM model's DiT architecture outputs `(batch, seq_len, num_cands)` representing predictions over candidate segments only.

## Solution

### 1. Fixed Batch Key Mappings

Updated all three methods to use the correct dataset keys:

| Old Key | New Key |
|---------|---------|
| `'src'` | `'norm_gps_seq'` |
| `'src_len'` | `'lengths'` |
| `'src_segs'` | `'segs_id'` |
| `'target'` | Created dynamically from `'trg_rid'` and `'segs_id'` |

### 2. Added `_create_candidate_target()` Method

Created a new helper method that converts the target road segment ID to a per-candidate target distribution:

```python
def _create_candidate_target(self, trg_rid, segs_id, segs_mask):
    """Create per-candidate target distribution from target road IDs.

    For each GPS point, creates a one-hot vector over candidates indicating
    which candidate matches the ground truth road segment.
    """
    # Expand trg_rid for comparison: (batch, seq_len, 1)
    trg_rid_expanded = trg_rid.unsqueeze(-1)

    # Find which candidate matches target: (batch, seq_len, num_cands)
    matches = (segs_id == trg_rid_expanded).float()

    # Apply mask and normalize
    matches = matches * segs_mask
    match_sum = torch.clamp(matches.sum(dim=-1, keepdim=True), min=1e-8)
    target = matches / match_sum

    return target
```

### 3. Updated `predict()` Method

The predict method now returns a dictionary with:
- `'pred_rid'` - Predicted road segment indices (batch, seq_len)
- `'pred_cand_idx'` - Predicted candidate indices (batch, seq_len)
- `'probs'` - Probability distributions over candidates (batch, seq_len, num_cands)

## Modified File

**File:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/DiffMM.py`

**Changes:**
1. Added `_create_candidate_target()` method (lines 966-997)
2. Updated `forward()` method (lines 999-1047) - uses correct batch keys
3. Updated `infer()` method (lines 1049-1080) - uses correct batch keys
4. Updated `predict()` method (lines 1082-1118) - returns dict with road IDs

## Testing

After this fix, the model should:
1. Accept batches from `DiffMMDataset` without KeyError
2. Correctly train on per-candidate targets matching model output dimensions
3. Return actual road segment IDs in predictions for evaluation

## Date
2026-02-04
