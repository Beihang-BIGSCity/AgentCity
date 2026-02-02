# DeepMM Testing Configuration Guide

## Quick Reference

**Model**: DeepMM
**Task**: map_matching
**Status**: Configured, needs custom dataset encoder and executor for testing

---

## Current Configuration Summary

### 1. task_config.json Registration
```json
"map_matching": {
    "allowed_model": ["STMatching", "IVMM", "HMMM", "FMM", "STMatch", "DeepMM"],
    "allowed_dataset": ["global", "Seattle"],
    ...
    "DeepMM": {
        "dataset_class": "MapMatchingDataset",
        "executor": "MapMatchingExecutor",
        "evaluator": "MapMatchingEvaluator"
    }
}
```

**Location**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
- Line 1040-1046: Added to allowed_model list
- Line 1071-1078: Model-specific configuration

### 2. Model Configuration

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/map_matching/DeepMM.json`

**Key Parameters**:
- Encoder: 2-layer BiLSTM, 512 hidden units, dropout 0.5
- Decoder: 1-layer LSTM + attention, 512 hidden units
- Embeddings: 256-dim location, 256-dim road segment
- Batch size: 128
- Learning rate: 0.001 (Adam optimizer)
- Attention type: dot
- Time encoding: NoEncoding (location only, no time features)

### 3. Model Implementation

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/DeepMM.py`

**Class**: `DeepMM(AbstractModel)`

**Key Methods**:
- `forward(batch)`: Teacher forcing forward pass
- `predict(batch)`: Prediction with argmax
- `calculate_loss(batch)`: CrossEntropyLoss with padding ignored
- `greedy_decode(batch, max_len, sos_idx, eos_idx)`: Inference without teacher forcing

---

## Data Requirements

### Input Batch Format

DeepMM expects batches with the following keys:

```python
batch = {
    'input_src': torch.LongTensor,  # Shape: (batch_size, src_len)
                                    # Tokenized GPS locations

    'input_trg': torch.LongTensor,  # Shape: (batch_size, trg_len)
                                    # Tokenized road segments (teacher forcing)

    'target': torch.LongTensor,     # Shape: (batch_size, trg_len)
                                    # Ground truth target (input_trg shifted)

    'input_time': Optional,         # Only if time_encoding != "NoEncoding"
                                    # Time features for each GPS point
}
```

### Data Features Required

```python
data_feature = {
    'src_loc_vocab_size': int,      # Number of GPS location tokens
    'trg_seg_vocab_size': int,      # Number of road segments + special tokens
    'pad_token_src_loc': int,       # Padding token for source (default: 0)
    'pad_token_trg': int,           # Padding token for target (default: 0)

    # Optional for time encoding
    'src_tim_vocab_size': int or list,  # Time discretization vocabulary
    'pad_token_src_tim1': int,          # Padding for OneEncoding
    'pad_token_src_tim2': list,         # Padding for TwoEncoding
}
```

---

## Critical Implementation Gaps

### Gap 1: Data Preprocessing (BLOCKING)

**Issue**: Current `MapMatchingDataset` provides raw GPS coordinates and road network, not tokenized sequences.

**Needed**:
1. GPS location discretization (e.g., grid-based or learned clustering)
2. Road segment vocabulary mapping
3. Sequence tokenization and padding
4. Batch creation with proper tensor shapes

**Solution Options**:
- Create `DeepMMEncoder` class
- Create `DeepMMDataset` subclass
- Extend `MapMatchingDataset` with neural model support

### Gap 2: Training Executor (BLOCKING)

**Issue**: Current `MapMatchingExecutor` is for traditional (non-trainable) models.

**Needed**:
1. Training loop with batching
2. Teacher forcing support
3. Gradient updates and optimization
4. Model checkpointing

**Solution**: Create `DeepMMExecutor(AbstractExecutor)` with:
```python
class DeepMMExecutor(AbstractExecutor):
    def train(self, train_dataloader, eval_dataloader):
        # Implement training loop
        pass

    def evaluate(self, test_dataloader):
        # Implement evaluation with greedy/beam decoding
        pass
```

---

## Testing Checklist

### Phase 1: Data Pipeline (TODO)
- [ ] Implement GPS location tokenization
- [ ] Build location vocabulary from training data
- [ ] Build road segment vocabulary
- [ ] Create batch collation function
- [ ] Validate tensor shapes and dtypes
- [ ] Test with small sample batch

### Phase 2: Executor (TODO)
- [ ] Implement DeepMMExecutor class
- [ ] Add training loop with teacher forcing
- [ ] Add evaluation with greedy decoding
- [ ] Implement checkpoint saving/loading
- [ ] Configure optimizer and learning rate scheduler
- [ ] Add gradient clipping

### Phase 3: Integration (TODO)
- [ ] Update task_config.json with DeepMMExecutor
- [ ] Register DeepMMEncoder if created
- [ ] Test with small dataset sample
- [ ] Verify loss computation
- [ ] Check prediction format
- [ ] Validate evaluation metrics

### Phase 4: Full Testing (TODO)
- [ ] Train on Seattle or global dataset
- [ ] Monitor training loss and metrics
- [ ] Evaluate on test set
- [ ] Compare with traditional baselines
- [ ] Analyze attention weights
- [ ] Test inference speed

---

## Example Test Configuration

```json
{
    "task": "map_matching",
    "model": "DeepMM",
    "dataset": "Seattle",
    "batch_size": 128,
    "learning_rate": 0.001,
    "max_epoch": 100,
    "gpu": true,
    "gpu_id": 0,

    "src_loc_emb_dim": 256,
    "trg_seg_emb_dim": 256,
    "src_hidden_dim": 512,
    "trg_hidden_dim": 512,
    "nlayers_src": 2,
    "dropout": 0.5,
    "attn_type": "dot",

    "use_early_stop": true,
    "patience": 10,
    "clip_grad_norm": true,
    "max_grad_norm": 5.0
}
```

---

## Expected Workflow

### 1. Dataset Preparation
```python
# Pseudocode for what needs to be implemented

dataset = MapMatchingDataset(config)
train_data, val_data, test_data = dataset.get_data()

# Convert raw data to tokenized sequences
encoder = DeepMMEncoder(config, dataset)
encoder.build_vocabularies(train_data)

# Create dataloaders
train_loader = encoder.create_dataloader(train_data, batch_size=128, shuffle=True)
val_loader = encoder.create_dataloader(val_data, batch_size=128, shuffle=False)
```

### 2. Model Training
```python
# Pseudocode for executor

model = DeepMM(config, data_feature)
executor = DeepMMExecutor(config, model, data_feature)

# Train model
executor.train(train_loader, val_loader)
# -> Trains for max_epoch with teacher forcing
# -> Validates on val_loader each epoch
# -> Saves best checkpoint based on validation loss

# Evaluate model
executor.evaluate(test_loader)
# -> Loads best checkpoint
# -> Runs greedy decoding on test set
# -> Computes map matching metrics
```

### 3. Inference
```python
# For new GPS trajectory

gps_trajectory = [[lon1, lat1], [lon2, lat2], ...]
tokenized_input = encoder.encode_trajectory(gps_trajectory)

batch = {
    'input_src': tokenized_input,
    # No input_trg needed for inference
}

# Greedy decoding
predicted_route = model.greedy_decode(batch, max_len=54, sos_idx=1, eos_idx=2)
road_segments = encoder.decode_segments(predicted_route)
```

---

## Troubleshooting

### Common Issues

**1. Vocabulary Size Mismatch**
- **Error**: "index out of range in self"
- **Cause**: Embedding layer size doesn't match vocabulary
- **Fix**: Ensure `src_loc_vocab_size` and `trg_seg_vocab_size` in data_feature match actual vocabularies

**2. Sequence Length Issues**
- **Error**: "shape mismatch" in attention or LSTM
- **Cause**: Unpadded or incorrectly padded sequences
- **Fix**: Pad all sequences to `input_max_len` (source) and `output_max_len` (target)

**3. Teacher Forcing Format**
- **Error**: Loss is NaN or extremely high
- **Cause**: `input_trg` and `target` are not properly aligned
- **Fix**: Ensure `target = input_trg[:, 1:]` (shifted by 1) and add EOS token

**4. Memory Issues**
- **Error**: CUDA out of memory
- **Cause**: Large batch size or long sequences
- **Fix**: Reduce batch_size, use gradient accumulation, or shorter max lengths

**5. No Training Progress**
- **Error**: Loss not decreasing
- **Cause**: Learning rate too high/low, wrong loss function, or data issues
- **Fix**: Check learning rate, verify loss function (CrossEntropyLoss), inspect data

---

## Hyperparameter Tuning Guide

### Critical Parameters

**1. Embedding Dimensions**
- `src_loc_emb_dim`: 128-512 (depends on location vocabulary size)
- `trg_seg_emb_dim`: 128-512 (depends on road network size)
- Larger = more capacity but slower

**2. Hidden Dimensions**
- `src_hidden_dim`: 256-1024
- `trg_hidden_dim`: 256-1024
- Should match or be close to embedding dims
- Larger = better capacity but more memory

**3. Dropout**
- `dropout`: 0.1-0.5
- Higher for larger models or overfitting
- 0.5 recommended for encoder (from config_best.json)

**4. Attention Type**
- `dot`: Fast, works well (default)
- `general`: More parameters, potentially better
- `mlp`: Most parameters, slowest, best for complex patterns

**5. Learning Rate**
- `learning_rate`: 0.0001-0.01
- 0.001 is a good default (Adam)
- Use lr_scheduler for decay

**6. Batch Size**
- `batch_size`: 32-256
- Larger = faster training but more memory
- 128 recommended (from config_best.json)

### Tuning Strategy

1. Start with default configuration (config_best.json values)
2. Train for 10 epochs, monitor validation loss
3. If overfitting: increase dropout, add regularization
4. If underfitting: increase model capacity (hidden_dim, nlayers)
5. If unstable: reduce learning_rate, increase grad clipping
6. Fine-tune attention_type and time_encoding based on data

---

## Performance Expectations

### Training Time (Estimates)

**Small Dataset (1000 trajectories)**:
- ~5-10 minutes per epoch (GPU)
- ~30-60 minutes per epoch (CPU)

**Medium Dataset (10000 trajectories)**:
- ~30-60 minutes per epoch (GPU)
- ~5-10 hours per epoch (CPU)

**Large Dataset (100000+ trajectories)**:
- ~5-10 hours per epoch (GPU)
- Not recommended on CPU

### Convergence

- Expect loss to stabilize after 20-50 epochs
- Early stopping typically triggers around epoch 30-70
- Validation loss should be close to training loss (within 10-20%)

### Accuracy Metrics

- **Route Accuracy**: % of exactly matched routes (strict)
- **Precision@k**: % of top-k predictions containing correct segment
- **Edit Distance**: Sequence similarity between predicted and ground truth
- **Graph Distance**: Shortest path distance on road network

---

## Next Steps Summary

### Immediate Actions Required

1. **Implement DeepMMEncoder** (2-3 days):
   - GPS tokenization logic
   - Vocabulary building
   - Batch collation
   - Data validation

2. **Implement DeepMMExecutor** (1-2 days):
   - Training loop
   - Teacher forcing
   - Evaluation logic
   - Checkpointing

3. **Update Configuration** (1 hour):
   - Modify task_config.json to use DeepMMExecutor
   - Add encoder registration if needed
   - Create test config file

4. **Integration Testing** (1-2 days):
   - Small sample test
   - Verify all components work together
   - Debug data pipeline issues
   - Validate outputs

### Long-term Enhancements

- Beam search decoding
- Attention visualization
- Multi-modal inputs (speed, heading, etc.)
- Graph-aware constraints
- Transfer learning from pretrained embeddings

---

## Files Reference

### Configuration Files
- **Task config**: `Bigscity-LibCity/libcity/config/task_config.json`
- **Model config**: `Bigscity-LibCity/libcity/config/model/map_matching/DeepMM.json`

### Model Files
- **Model implementation**: `Bigscity-LibCity/libcity/model/map_matching/DeepMM.py`
- **Model registration**: `Bigscity-LibCity/libcity/model/map_matching/__init__.py`

### Dataset Files
- **Dataset class**: `Bigscity-LibCity/libcity/data/dataset/map_matching_dataset.py`
- **Dataset config**: `Bigscity-LibCity/libcity/config/data/MapMatchingDataset.json`

### Executor Files
- **Executor class**: `Bigscity-LibCity/libcity/executor/map_matching_executor.py` (traditional)
- **Executor config**: `Bigscity-LibCity/libcity/config/executor/MapMatchingExecutor.json`

### Documentation
- **Full migration summary**: `documentation/DeepMM_config_migration_summary.md`
- **This guide**: `documentation/DeepMM_testing_guide.md`

---

**Last Updated**: 2026-02-02
**Status**: Configuration complete, awaiting encoder and executor implementation
