# RNTrajRec Configuration Migration Summary

**Model:** RNTrajRec (Road Network-aware Trajectory Recovery)
**Task:** Trajectory Location Prediction (traj_loc_pred)
**Date:** 2026-02-02
**Status:** Configuration Complete

---

## 1. Model Registration

### task_config.json Updates

**Location:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`

#### Added to allowed_model list (Line 3-34)
```json
"traj_loc_pred": {
    "allowed_model": [
        ...
        "TrajSDE",
        "RNTrajRec"  // Added at line 34
    ]
}
```

#### Model Configuration Entry (Lines 210-215)
```json
"RNTrajRec": {
    "dataset_class": "TrajectoryDataset",
    "executor": "TrajLocPredExecutor",
    "evaluator": "TrajLocPredEvaluator",
    "traj_encoder": "StandardTrajectoryEncoder"
}
```

---

## 2. Model Configuration File

**Location:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/RNTrajRec.json`

### Complete Configuration

```json
{
    "model": "RNTrajRec",
    "task": "traj_loc_pred",

    // Model Architecture Parameters
    "hid_dim": 512,                    // Hidden dimension for encoder/decoder
    "loc_emb_dim": 512,                // Location embedding dimension
    "transformer_layers": 2,            // Number of transformer encoder layers
    "num_heads": 8,                    // Number of attention heads
    "dropout": 0.1,                    // Dropout probability
    "use_attention": true,             // Use attention in decoder
    "use_time": true,                  // Use temporal embeddings
    "tim_emb_dim": 64,                 // Time embedding dimension
    "teacher_forcing_ratio": 0.5,      // Teacher forcing ratio during training
    "max_output_len": 128,             // Maximum output sequence length

    // Training Parameters
    "batch_size": 64,                  // Training batch size
    "learning_rate": 0.0001,           // Initial learning rate (Adam)
    "max_epoch": 50,                   // Maximum training epochs
    "optimizer": "adam",               // Optimizer type
    "clip": 5.0,                       // Gradient clipping value
    "lr_step": 10,                     // LR scheduler step size
    "lr_decay": 0.5,                   // LR decay factor
    "lr_scheduler": "steplr",          // Learning rate scheduler
    "log_every": 1,                    // Logging frequency
    "load_best_epoch": true,           // Load best model after training
    "hyper_tune": false                // Hyperparameter tuning mode
}
```

---

## 3. Hyperparameter Details

### Architecture Parameters (from Paper/Original Implementation)

| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `hid_dim` | 512 | Paper default | Hidden dimension for all components |
| `loc_emb_dim` | 512 | Paper default | Location embedding dimension |
| `transformer_layers` | 2 | Paper default | Number of transformer encoder layers |
| `num_heads` | 8 | Paper default | Multi-head attention heads |
| `dropout` | 0.1 | Paper default | Dropout rate for regularization |
| `tim_emb_dim` | 64 | Paper default | Temporal embedding dimension |
| `teacher_forcing_ratio` | 0.5 | Common practice | Probability of using ground truth |
| `max_output_len` | 128 | Dataset-dependent | Maximum trajectory length |

### Training Parameters (LibCity Standard)

| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `batch_size` | 64 | LibCity standard | Training batch size |
| `learning_rate` | 0.0001 | Paper/common | Adam learning rate |
| `max_epoch` | 50 | LibCity standard | Maximum training epochs |
| `optimizer` | "adam" | Paper standard | Adam optimizer |
| `clip` | 5.0 | Common practice | Gradient clipping threshold |
| `lr_step` | 10 | LibCity standard | StepLR step size |
| `lr_decay` | 0.5 | LibCity standard | LR decay factor |
| `lr_scheduler` | "steplr" | LibCity standard | Step-based LR scheduler |

---

## 4. Model Components

### Encoder: TransformerEncoder
- **Input:** Location sequence (+ optional time sequence)
- **Embedding:** Separate embeddings for location and time
- **Processing:** Multi-layer transformer with positional encoding
- **Output:** Contextualized sequence representations

### Decoder: TrajDecoder
- **Type:** GRU-based autoregressive decoder
- **Attention:** Bahdanau-style attention over encoder outputs
- **Input:** Previous location + attended context
- **Output:** Next location prediction

### Key Features
- Multi-head self-attention for sequence modeling
- Positional encoding for sequence order
- Teacher forcing during training
- Attention mechanism for decoder context

---

## 5. Dataset Compatibility

### Supported Datasets

RNTrajRec works with **TrajectoryDataset** class datasets:

| Dataset | Status | Notes |
|---------|--------|-------|
| foursquare_tky | Compatible | Tokyo check-in data |
| foursquare_nyc | Compatible | NYC check-in data |
| gowalla | Compatible | Gowalla check-in data |
| foursquare_serm | Compatible | SERM dataset |
| Proto | Compatible | Custom trajectory data |

### Required Data Features

From `data_feature` dictionary:
- `loc_size`: Number of unique locations/POIs
- `tim_size`: Number of time slots (default: 48 for half-hour slots)
- `loc_pad`: Padding index for locations (usually 0)

### Input Data Format

The model expects LibCity trajectory batches with:

**Required Keys:**
- `current_loc`: Input location sequence (batch, seq_len) - LongTensor
- `target`: Target location for next-step prediction (batch,) - LongTensor

**Optional Keys:**
- `current_tim`: Input time sequence (batch, seq_len) - LongTensor
- `target_loc`: Full target sequence for trajectory recovery (batch, trg_len)

---

## 6. Differences from Original Implementation

### Simplified Components

| Original Feature | LibCity Adaptation | Impact |
|------------------|-------------------|--------|
| DGL-based Road Graph | Learnable embeddings | No explicit graph structure |
| Graph Neural Network | Removed | Simplified spatial modeling |
| Road network constraints | Removed | No topology constraints |
| Graph refinement layers | Removed | Simpler architecture |
| Sub-segment rate prediction | Simplified | Only location prediction |

### Retained Core Features
- Transformer encoder architecture
- Multi-head self-attention
- Positional encoding
- GRU decoder with attention
- Teacher forcing mechanism
- Autoregressive generation

### Limitations
1. **No Road Network:** Cannot leverage explicit road topology
2. **No Graph Constraints:** Predictions not constrained by connectivity
3. **Simplified Recovery:** Cannot predict sub-road positions
4. **Embedding-based:** Spatial relationships learned implicitly

---

## 7. Usage Examples

### Basic Training
```bash
python run_model.py --task traj_loc_pred --model RNTrajRec --dataset foursquare_tky
```

### Custom Configuration
```bash
python run_model.py --task traj_loc_pred --model RNTrajRec --dataset gowalla \
    --hid_dim 256 --transformer_layers 3 --learning_rate 0.0005
```

### With Specific Parameters
```bash
python run_model.py --task traj_loc_pred --model RNTrajRec --dataset foursquare_nyc \
    --batch_size 32 --max_epoch 100 --teacher_forcing_ratio 0.7
```

---

## 8. Model Performance Considerations

### Computational Requirements
- **Memory:** Moderate (transformer encoder + GRU decoder)
- **Training Speed:** Medium (attention mechanism overhead)
- **Inference Speed:** Fast (efficient autoregressive decoding)

### Recommended Settings

**For Small Datasets (< 10K trajectories):**
```json
{
    "hid_dim": 256,
    "transformer_layers": 2,
    "batch_size": 32,
    "max_epoch": 100
}
```

**For Large Datasets (> 100K trajectories):**
```json
{
    "hid_dim": 512,
    "transformer_layers": 3,
    "batch_size": 128,
    "max_epoch": 50
}
```

---

## 9. Evaluation Metrics

RNTrajRec uses **TrajLocPredEvaluator** with standard metrics:

- **Accuracy@K:** Top-K location prediction accuracy (K=1, 5, 10)
- **MRR:** Mean Reciprocal Rank
- **NDCG:** Normalized Discounted Cumulative Gain
- **Macro-F1:** Macro-averaged F1 score
- **Micro-F1:** Micro-averaged F1 score

Evaluation method controlled by `evaluate_method` parameter:
- `"all"`: Predict over entire location vocabulary
- `"sample"`: Use negative sampling for efficiency

---

## 10. Configuration Validation

### Required Parameters Checklist
- [x] `hid_dim` - Model hidden dimension
- [x] `loc_emb_dim` - Location embedding size
- [x] `transformer_layers` - Number of encoder layers
- [x] `num_heads` - Attention heads
- [x] `dropout` - Regularization
- [x] `learning_rate` - Optimizer LR
- [x] `max_epoch` - Training epochs
- [x] `optimizer` - Optimization algorithm

### Optional Parameters
- [x] `use_time` - Enable temporal modeling
- [x] `use_attention` - Enable decoder attention
- [x] `teacher_forcing_ratio` - Training strategy
- [x] `max_output_len` - Trajectory length limit
- [x] `lr_scheduler` - Learning rate scheduling

---

## 11. Troubleshooting

### Common Issues

**Issue 1: Out of Memory**
- Reduce `batch_size` to 32 or 16
- Reduce `hid_dim` to 256 or 128
- Reduce `transformer_layers` to 1

**Issue 2: Slow Training**
- Increase `batch_size` to 128 or 256
- Reduce `max_output_len` if not needed
- Consider disabling `use_time` for simpler model

**Issue 3: Poor Performance**
- Increase `hid_dim` to 512 or 768
- Add more `transformer_layers` (3-4)
- Adjust `teacher_forcing_ratio` (try 0.7)
- Increase training epochs

**Issue 4: Overfitting**
- Increase `dropout` to 0.2 or 0.3
- Reduce `hid_dim`
- Use learning rate scheduling
- Early stopping with validation

---

## 12. Related Models

Models with similar architecture/purpose:

| Model | Similarity | Difference |
|-------|-----------|------------|
| PLMTrajRec | Transformer-based | Uses pre-trained language model |
| GETNext | Attention mechanism | Graph-enhanced transformer |
| LoTNext | Location prediction | Long-tail distribution handling |
| JGRM | Graph-based | Joint graph representation |
| TrajSDE | Neural ODE | Stochastic differential equations |

---

## 13. References

### Original Paper
"RNTrajRec: Road Network Enhanced Trajectory Recovery with Spatial-Temporal Transformer"

### Original Repository
https://github.com/WenMellors/RNTrajRec

### Key Dependencies
- PyTorch >= 1.7.0
- LibCity framework
- Standard trajectory encoders

---

## 14. Future Enhancements

Potential improvements for LibCity integration:

1. **Add Road Network Support:**
   - Integrate with LibCity's road network data structures
   - Implement simplified GNN layers without DGL dependency

2. **Enhanced Spatial Features:**
   - Add distance-based embeddings
   - Incorporate POI category information

3. **Improved Temporal Modeling:**
   - Sinusoidal time embeddings
   - Relative time intervals

4. **Training Improvements:**
   - Scheduled sampling for teacher forcing
   - Curriculum learning strategies

---

## Summary

RNTrajRec has been successfully configured for LibCity framework:

- **Model registered:** ✅ Added to traj_loc_pred.allowed_model
- **Config created:** ✅ `/config/model/traj_loc_pred/RNTrajRec.json`
- **Task mapping:** ✅ Uses TrajectoryDataset, TrajLocPredExecutor
- **Parameters:** ✅ All hyperparameters documented and validated
- **Datasets:** ✅ Compatible with standard trajectory datasets

The model can now be used for trajectory location prediction tasks in LibCity following standard workflows.
