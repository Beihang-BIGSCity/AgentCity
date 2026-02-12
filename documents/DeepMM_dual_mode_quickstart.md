# DeepMM Dual-Mode Quick Reference

## Model Location
`Bigscity-LibCity/libcity/model/trajectory_loc_prediction/DeepMM.py`

## Supported Tasks

### 1. Map Matching (Sequence-to-Sequence)
Predict a sequence of road segments from GPS trajectory.

**Batch Structure:**
```python
batch = {
    'current_loc': tensor([batch, seq_len]),  # GPS locations
    'target':      tensor([batch, seq_len])   # Road segments
}
```

**Example:**
```python
batch_size = 32
seq_len = 50
vocab_size = 5000

batch = {
    'current_loc': torch.randint(0, vocab_size, (batch_size, seq_len)),
    'target':      torch.randint(0, vocab_size, (batch_size, seq_len))
}

# Forward pass
logits = model.forward(batch)  # [32, 50, 5000]

# Prediction
preds = model.predict(batch)   # [32, 50]

# Loss
loss = model.calculate_loss(batch)  # scalar
```

### 2. Next-Location Prediction (Sequence-to-Single)
Predict the next single location from trajectory history.

**Batch Structure:**
```python
batch = {
    'current_loc': tensor([batch, seq_len]),  # Historical trajectory
    'target':      tensor([batch])            # Next location (single value)
}
```

**Example:**
```python
batch_size = 32
seq_len = 50
vocab_size = 5000

batch = {
    'current_loc': torch.randint(0, vocab_size, (batch_size, seq_len)),
    'target':      torch.randint(0, vocab_size, (batch_size,))  # Note: 1D tensor
}

# Forward pass (internally converts to [batch, 1])
logits = model.forward(batch)  # [32, 1, 5000]

# Prediction (automatically squeezes output)
preds = model.predict(batch)   # [32]

# Loss (handles 1D target)
loss = model.calculate_loss(batch)  # scalar
```

## Key Differences

| Aspect | Map Matching | Next-Location |
|--------|--------------|---------------|
| **Target Shape** | `[batch, seq_len]` | `[batch]` |
| **Forward Output** | `[batch, seq_len, vocab]` | `[batch, 1, vocab]` |
| **Predict Output** | `[batch, seq_len]` | `[batch]` |
| **Loss Shape** | Flattened sequence | Single step |
| **Use Case** | GPS→Road alignment | Trajectory→Next POI |

## Configuration

**Model Config (same for both tasks):**
```json
{
    "src_loc_emb_dim": 256,
    "trg_seg_emb_dim": 256,
    "src_hidden_dim": 512,
    "trg_hidden_dim": 512,
    "bidirectional": true,
    "n_layers_src": 2,
    "dropout": 0.5,
    "rnn_type": "LSTM",
    "attn_type": "dot"
}
```

**Data Feature Requirements:**
```python
data_feature = {
    'loc_size': 5000,      # Vocabulary size
    'loc_pad': 0           # Padding token index
}
```

## Automatic Mode Detection

The model automatically detects which mode to use:

```python
# In forward method
if input_trg.dim() == 1:
    # Single-target mode (next-location prediction)
    input_trg = input_trg.unsqueeze(1)  # [batch] → [batch, 1]
else:
    # Sequence-target mode (map matching)
    # Process as-is
```

## Usage Examples

### Training Loop
```python
model = DeepMM(config, data_feature)
optimizer = torch.optim.Adam(model.parameters())

for batch in dataloader:
    # Works for both map matching and next-location prediction
    loss = model.calculate_loss(batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Evaluation Loop
```python
model.eval()
all_predictions = []

with torch.no_grad():
    for batch in dataloader:
        # Returns [batch, seq_len] or [batch] automatically
        preds = model.predict(batch)
        all_predictions.append(preds)

# Concatenate results
predictions = torch.cat(all_predictions, dim=0)
```

## Common Pitfalls

### 1. Wrong Target Shape
```python
# WRONG: Using 2D tensor for next-location
batch = {
    'current_loc': tensor([32, 50]),
    'target':      tensor([32, 1])  # Should be [32]
}

# CORRECT
batch = {
    'current_loc': tensor([32, 50]),
    'target':      tensor([32])     # 1D tensor
}
```

### 2. Mismatched Vocabulary Sizes
```python
# Ensure vocab size matches data
data_feature = {
    'loc_size': actual_vocab_size,  # Must match actual data
    'loc_pad': 0
}
```

### 3. Device Mismatch
```python
# Ensure batch and model are on same device
batch = {k: v.to(device) for k, v in batch.items()}
model = model.to(device)
```

## Integration with LibCity

### Task Configuration
```json
{
    "task": "traj_loc_pred",
    "model": "DeepMM",
    "dataset": "your_dataset"
}
```

### Dataset Requirements
Your dataset should return batches with:
- `current_loc`: Source trajectory
- `target`: Either sequence or single value

The encoder will handle both formats automatically.

## Performance Considerations

1. **Sequence Length**: Longer sequences require more memory in map matching mode
2. **Batch Size**: Single-target mode is more memory-efficient (seq_len=1 in decoder)
3. **Attention**: Computed for all encoder steps regardless of target length
4. **Teacher Forcing**: Used in training for both modes (target provided to decoder)

## Troubleshooting

### Dimension Errors
```python
# Check target dimensions
print(f"Target shape: {batch['target'].shape}")
print(f"Target dim: {batch['target'].dim()}")

# Expected:
# Map matching: Target dim: 2, shape: [batch, seq_len]
# Next-location: Target dim: 1, shape: [batch]
```

### Loss is NaN
- Check vocabulary size matches data
- Ensure padding token is set correctly
- Verify learning rate is appropriate

### Wrong Output Shape
```python
# After prediction
preds = model.predict(batch)
print(f"Predictions shape: {preds.shape}")

# Should match target shape
assert preds.shape == batch['target'].shape
```

## Related Documentation
- Full adaptation details: `documents/DeepMM_dual_mode_adaptation.md`
- Original DeepMM paper: KDD 2018
- LibCity AbstractModel: `libcity/model/abstract_model.py`

## Contact
For issues or questions about this adaptation, check the LibCity documentation or model source code comments.
