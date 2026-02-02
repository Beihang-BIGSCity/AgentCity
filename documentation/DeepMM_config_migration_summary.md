# DeepMM Configuration Migration Summary

**Date**: 2026-02-02
**Model**: DeepMM (Deep Learning-based Map Matching)
**Task Type**: map_matching
**Phase**: 3 - Framework Registration and Testing Configuration

---

## 1. Task Config Registration

### File: `libcity/config/task_config.json`

**Changes Made:**

1. **Added to allowed_model list** (Line 1040-1046):
   ```json
   "allowed_model": [
       "STMatching",
       "IVMM",
       "HMMM",
       "FMM",
       "STMatch",
       "DeepMM"
   ]
   ```

2. **Added model-specific configuration** (Line 1071-1078):
   ```json
   "DeepMM": {
       "dataset_class": "MapMatchingDataset",
       "executor": "MapMatchingExecutor",
       "evaluator": "MapMatchingEvaluator"
   }
   ```

**Note**: DeepMM is the first deep learning model in the map_matching task. All other models (STMatching, IVMM, HMMM, FMM, STMatch) are traditional geometric/probabilistic algorithms.

---

## 2. Model Configuration

### File: `libcity/config/model/map_matching/DeepMM.json`

**Complete Hyperparameter Configuration:**

```json
{
  "src_loc_emb_dim": 256,
  "src_tim_emb_dim": 64,
  "trg_seg_emb_dim": 256,
  "src_hidden_dim": 512,
  "trg_hidden_dim": 512,
  "bidirectional": true,
  "nlayers_src": 2,
  "dropout": 0.5,
  "time_encoding": "NoEncoding",
  "rnn_type": "LSTM",
  "attn_type": "dot",
  "batch_size": 128,
  "learning_rate": 0.001,
  "max_epoch": 100,
  "optimizer": "Adam",
  "input_max_len": 40,
  "output_max_len": 54,
  "learner": "adam",
  "lr_decay": false,
  "lr_scheduler": "multisteplr",
  "lr_decay_ratio": 0.1,
  "steps": [20, 40, 60],
  "clip_grad_norm": true,
  "max_grad_norm": 5.0,
  "use_early_stop": true,
  "patience": 10,
  "log_every": 1,
  "saved": true,
  "save_mode": "best",
  "train_loss": "CrossEntropyLoss"
}
```

### Hyperparameter Mapping (Original → LibCity)

| Original Parameter | LibCity Parameter | Value | Source |
|-------------------|-------------------|-------|--------|
| `loc_emb_dim` | `src_loc_emb_dim` | 256 | config_best.json |
| `time_emb_dim` | `src_tim_emb_dim` | 64 | config_best.json |
| `seg_emb_dim` | `trg_seg_emb_dim` | 256 | config_best.json |
| `hid_dim` | `src_hidden_dim` | 512 | config_best.json |
| `hid_dim` | `trg_hidden_dim` | 512 | config_best.json |
| `bidirectional` | `bidirectional` | true | config_best.json |
| `nlayers` | `nlayers_src` | 2 | config_best.json |
| `dropout` | `dropout` | 0.5 | config_best.json |
| `time_encoding` | `time_encoding` | "NoEncoding" | config_best.json |
| `rnn_type` | `rnn_type` | "LSTM" | config_best.json |
| `attn_type` | `attn_type` | "dot" | config_best.json |
| `batch_size` | `batch_size` | 128 | config_best.json |
| `lr` | `learning_rate` | 0.001 | config_best.json |
| N/A | `max_epoch` | 100 | default |
| `optimizer` | `optimizer` / `learner` | "Adam" / "adam" | config_best.json |

### Model Architecture Parameters

**Encoder (GPS Trajectory → Hidden Representation)**:
- Type: 2-layer Bidirectional LSTM
- Hidden size: 512 (256 per direction)
- Dropout: 0.5 (between layers)
- Input embedding: 256-dim location + optional 64-dim time

**Decoder (Hidden Representation → Road Segment Sequence)**:
- Type: 1-layer LSTM with attention
- Hidden size: 512
- Attention: Soft dot attention
- Output embedding: 256-dim road segment

**Attention Mechanism**:
- Type: `dot` (default, best performance)
- Options: `dot`, `general`, `mlp`
- Reference: http://www.aclweb.org/anthology/D15-1166

**Time Encoding Options**:
- `NoEncoding`: Location only (default)
- `OneEncoding`: Location + single time feature
- `TwoEncoding`: Location + two time features (e.g., hour + day)

**Training Configuration**:
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 128
- Max epochs: 100
- Learning rate scheduler: MultiStepLR (decay at steps 20, 40, 60)
- LR decay ratio: 0.1
- Gradient clipping: Max norm 5.0
- Early stopping: Enabled (patience=10)
- Loss function: CrossEntropyLoss (ignores padding tokens)

---

## 3. Dataset Compatibility

### Current Dataset Class: `MapMatchingDataset`

**Data Format Requirements**:
- **Input**: GPS trajectory points (lon, lat, optional time)
- **Output**: Road segment IDs (ground truth route)
- **Road Network**: NetworkX DiGraph with geo_id, distance, coordinates

**Key Features Required by DeepMM**:

1. **Vocabularies** (Not currently in MapMatchingDataset):
   - `src_loc_vocab_size`: Number of unique GPS location bins/tokens
   - `trg_seg_vocab_size`: Number of road segments + special tokens
   - `src_tim_vocab_size`: Time discretization vocabulary (if using time encoding)

2. **Special Tokens**:
   - `pad_token_src_loc`: Padding token for source sequences (default: 0)
   - `pad_token_trg`: Padding token for target sequences (default: 0)
   - SOS (Start of Sequence) token: typically 1
   - EOS (End of Sequence) token: typically 2

3. **Sequence Format**:
   - `input_src`: Tokenized GPS locations, shape (batch, src_len)
   - `input_trg`: Tokenized road segments for teacher forcing, shape (batch, trg_len)
   - `target`: Shifted target sequence (input_trg shifted by 1)
   - `input_time`: Optional time features (if time_encoding != "NoEncoding")

### Available Datasets

From `task_config.json`:
```json
"allowed_dataset": [
    "global",
    "Seattle"
]
```

**Issue**: The current `MapMatchingDataset` class is designed for traditional map matching algorithms, not deep learning models. It provides:
- Raw GPS coordinates
- Road network graph
- Ground truth routes

**Missing for DeepMM**:
- Location tokenization/discretization
- Road segment vocabulary
- Sequence padding and batching
- Special token handling

---

## 4. Executor and Evaluator

### Current Configuration

**Executor**: `MapMatchingExecutor`
- Type: `AbstractTraditionExecutor` (no training)
- Designed for traditional algorithms
- Only implements `evaluate()`, not `train()`

**Evaluator**: `MapMatchingEvaluator`
- Metrics for map matching quality
- Likely uses geometric/graph-based metrics

### Requirements for DeepMM

**DeepMM Needs**:
1. A **trainable executor** (e.g., extending `AbstractExecutor`)
   - Implements training loop with batching
   - Handles teacher forcing
   - Supports gradient updates
   - Checkpoint saving/loading

2. Custom **data encoder** or dataset class:
   - GPS location discretization/tokenization
   - Road segment ID vocabulary management
   - Sequence padding to max lengths
   - Batch preparation

---

## 5. Critical Implementation Gaps

### Gap 1: Dataset Encoder
**Status**: Missing
**Required**: Custom encoder or dataset subclass

**Needed Functionality**:
- Convert GPS coordinates to discrete tokens
- Build vocabularies for locations and road segments
- Handle sequence padding and special tokens
- Support batched data loading

**Potential Solutions**:
1. Create `DeepMMEncoder` class (similar to `StandardTrajectoryEncoder`)
2. Create `DeepMMDataset` subclass of `MapMatchingDataset`
3. Extend `MapMatchingDataset` to support neural models

### Gap 2: Executor
**Status**: Configured to use `MapMatchingExecutor` (traditional)
**Required**: Neural training executor

**Options**:
1. Create `DeepMMExecutor` extending `AbstractExecutor`
2. Modify task_config.json to use `TrafficStateExecutor` or similar
3. Create generic `NeuralMapMatchingExecutor`

**Recommended**: Create custom `DeepMMExecutor` since:
- Map matching has unique data structure (graphs + sequences)
- Needs special handling for route evaluation
- Different from traffic state prediction or trajectory tasks

### Gap 3: Data Preprocessing
**Status**: Not implemented
**Required**: Tokenization pipeline

**Needed**:
- Grid-based or learned location discretization
- Road segment ID mapping to sequential indices
- Handling variable-length trajectories
- Vocabulary building from training data

---

## 6. Testing Workflow

### Prerequisites Before Testing

1. **Dataset Preparation**:
   - Choose a dataset (Seattle or global)
   - Implement tokenization pipeline
   - Build vocabularies for locations and segments
   - Ensure data_feature dictionary contains:
     - `src_loc_vocab_size`
     - `trg_seg_vocab_size`
     - `pad_token_src_loc`
     - `pad_token_trg`

2. **Executor Implementation**:
   - Create trainable executor for DeepMM
   - Implement batch processing
   - Add teacher forcing support
   - Configure evaluation metrics

3. **Configuration Updates**:
   - Update task_config.json with correct executor
   - Add dataset configuration if needed
   - Verify all paths and class names

### Test Command (Once Ready)

```bash
python run_model.py --task map_matching --model DeepMM --dataset Seattle
```

### Expected Output

**Training Phase**:
- Batch processing with teacher forcing
- Loss decreasing over epochs
- Gradient clipping applied
- Early stopping if no improvement

**Evaluation Phase**:
- Predicted road segment sequences
- Comparison with ground truth routes
- Map matching metrics (accuracy, precision@k, route similarity)

---

## 7. Comparison with Other Neural Seq2Seq Models in LibCity

### Similar Models

**ETA Task (Seq2Seq structure)**:
- `DeepTTE`, `TTPNet`, etc.
- Use `ETAExecutor` and `ETADataset`
- Handle trajectory sequences

**Trajectory Location Prediction**:
- `LSTM`, `GRU`, `DeepMove`, etc.
- Use `TrajLocPredExecutor` and `TrajectoryDataset`
- Predict next location from sequence

### Key Differences for Map Matching

1. **Input**: GPS coordinates (continuous) vs. location IDs (discrete)
2. **Output**: Road segment sequence (graph-constrained) vs. free location
3. **Evaluation**: Route matching accuracy vs. location prediction accuracy
4. **Data Structure**: Road network graph + trajectories vs. trajectories only

---

## 8. Recommended Next Steps

### Immediate Actions

1. **Create Dataset Encoder** (Priority: HIGH):
   ```python
   class DeepMMEncoder:
       - discretize_gps_coordinates()
       - build_location_vocabulary()
       - build_segment_vocabulary()
       - encode_trajectory()
       - create_batches()
   ```

2. **Create Executor** (Priority: HIGH):
   ```python
   class DeepMMExecutor(AbstractExecutor):
       - train() with teacher forcing
       - evaluate() with greedy/beam search
       - save/load model checkpoints
   ```

3. **Update task_config.json** (Priority: MEDIUM):
   ```json
   "DeepMM": {
       "dataset_class": "DeepMMDataset",
       "executor": "DeepMMExecutor",
       "evaluator": "MapMatchingEvaluator",
       "encoder": "DeepMMEncoder"
   }
   ```

4. **Create Test Dataset** (Priority: MEDIUM):
   - Prepare small Seattle subset
   - Build vocabularies
   - Validate data format

5. **Integration Testing** (Priority: LOW):
   - Test with small batch
   - Verify loss computation
   - Check prediction format
   - Validate evaluation metrics

### Long-term Enhancements

1. **Beam Search Decoding**: Improve prediction quality
2. **Pretrained Embeddings**: Use road2vec or similar
3. **Multi-task Learning**: Combine with ETA prediction
4. **Attention Visualization**: Debug model behavior
5. **Transfer Learning**: Pre-train on large dataset

---

## 9. Files Modified

### Configuration Files
1. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
   - Added "DeepMM" to map_matching.allowed_model
   - Added DeepMM-specific configuration block

2. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/map_matching/DeepMM.json`
   - Updated with complete hyperparameter configuration
   - Changed dropout from 0.1 to 0.5 (matching config_best.json)
   - Added training configuration parameters

### Model Files (Already Completed in Phase 1-2)
3. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/DeepMM.py`
   - Implemented model architecture
   - Added LibCity interface methods

4. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/__init__.py`
   - Registered DeepMM model

---

## 10. Special Considerations

### Unique Challenges

1. **First Neural Model in map_matching Task**:
   - All existing infrastructure assumes traditional algorithms
   - Need new data pipeline for neural training
   - Executor and dataset not designed for gradient-based learning

2. **Graph-Constrained Output**:
   - Road segments form a graph, not free vocabulary
   - Could add graph-aware loss or constraints
   - Current implementation treats as sequence generation

3. **Variable Length Handling**:
   - GPS trajectories have different lengths
   - Road routes have different lengths
   - Need dynamic padding and masking

4. **Vocabulary Management**:
   - Location vocabulary can be large (grid discretization)
   - Road segment vocabulary size = num_roads + special tokens
   - Need efficient embedding tables

### Performance Optimization

1. **Batching Strategy**:
   - Sort by sequence length
   - Pack sequences to minimize padding
   - Use DataLoader with collate_fn

2. **Memory Efficiency**:
   - Encoder hidden states can be large (BiLSTM)
   - Attention requires O(src_len * trg_len) memory
   - Consider gradient checkpointing for long sequences

3. **Inference Speed**:
   - Greedy decoding is fast but suboptimal
   - Beam search improves quality but slower
   - Consider cached attention for multiple beams

---

## 11. Contact and References

### Original Code Reference
- Repository: repos/DeepMM
- Main file: DeepMM/model.py (Seq2SeqAttention class, lines 825-1034)
- Config: config_best.json

### LibCity Documentation
- Framework: Bigscity-LibCity
- Task: map_matching
- Model directory: libcity/model/map_matching/

### Paper Reference
- Title: "Deep Learning Based Map Matching"
- Model: Seq2Seq with Attention for GPS-to-Road matching

---

## Summary

**Configuration Status**: Complete ✓
**Registration Status**: Complete ✓
**Testing Status**: Blocked (needs custom dataset encoder and executor)

**Key Achievements**:
- DeepMM registered in task_config.json
- Model config file updated with all hyperparameters
- Model implementation completed and registered in __init__.py

**Blocking Issues**:
1. MapMatchingDataset lacks neural model support (vocabularies, tokenization)
2. MapMatchingExecutor is for traditional models only (no training loop)
3. Need data preprocessing pipeline for sequence generation

**Immediate Next Step**: Implement DeepMMEncoder and DeepMMExecutor to enable end-to-end training and testing.
