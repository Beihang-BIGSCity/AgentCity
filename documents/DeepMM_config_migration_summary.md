# Config Migration: DeepMM

## Overview
DeepMM (Deep Learning-based Map Matching) is a sequence-to-sequence model with attention mechanism for map matching tasks. This document summarizes the configuration migration for integrating DeepMM into the LibCity framework.

## Model Information
- **Model Name**: DeepMM
- **Model Class**: DeepMM
- **Location**: `Bigscity-LibCity/libcity/model/trajectory_loc_prediction/DeepMM.py`
- **Task Types**:
  - `map_matching` (primary task)
  - `traj_loc_pred` (trajectory location prediction - newly added)
- **Paper Reference**: "DeepMM: Deep Learning based Map Matching with Heterogeneous Trajectory Data"

## Architecture Summary
DeepMM uses a sequence-to-sequence architecture consisting of:
1. **Source Location Embedding Layer**: Embeds GPS trajectory points (discretized to grid cells)
2. **Optional Time Embedding Layers**: Supports NoEncoding, OneEncoding, or TwoEncoding
3. **Bidirectional LSTM Encoder**: Processes input trajectory sequences
4. **LSTM Decoder with Attention**: Generates matched road segment sequences
5. **Output Projection Layer**: Maps decoder outputs to road segment vocabulary

## Configuration Files Created/Updated

### 1. task_config.json Updates

#### Added to traj_loc_pred Task
- **File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
- **Line**: 37 (in allowed_model list)
- **Configuration Entry** (Lines 238-243):
```json
"DeepMM": {
    "dataset_class": "DeepMMSeq2SeqDataset",
    "executor": "TrajLocPredExecutor",
    "evaluator": "TrajLocPredEvaluator",
    "traj_encoder": "StandardTrajectoryEncoder"
}
```

#### Already Registered in map_matching Task
- **Line**: 1099 (in allowed_model list)
- **Configuration Entry** (Lines 1138-1142):
```json
"DeepMM": {
    "dataset_class": "DeepMMSeq2SeqDataset",
    "executor": "DeepMapMatchingExecutor",
    "evaluator": "MapMatchingEvaluator"
}
```

### 2. Model Configuration Files

#### A. Map Matching Configuration (Pre-existing)
- **File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/map_matching/DeepMM.json`
- **Status**: Already exists with comprehensive parameters

#### B. Trajectory Location Prediction Configuration (Newly Created)
- **File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/DeepMM.json`
- **Status**: Newly created

## Model Hyperparameters

### Core Model Architecture Parameters

| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `src_loc_emb_dim` | 256 | Original paper | Source location embedding dimension |
| `src_tim_emb_dim` | 64 | Original paper | Time embedding dimension |
| `trg_seg_emb_dim` | 256 | Original paper | Target segment embedding dimension |
| `src_hidden_dim` | 512 | Original paper | Encoder hidden dimension |
| `trg_hidden_dim` | 512 | Original paper | Decoder hidden dimension |
| `bidirectional` | true | Original paper | Use bidirectional encoder |
| `nlayers_src` | 2 | Original paper | Number of encoder layers |
| `nlayers_trg` | 1 | Original paper | Number of decoder layers |
| `dropout` | 0.5 | Original paper | Dropout probability |
| `time_encoding` | "NoEncoding" | Original paper | Time encoding type |
| `rnn_type` | "LSTM" | Original paper | RNN type for encoder |
| `attn_type` | "dot" | Original paper | Attention mechanism type |
| `teacher_forcing_ratio` | 1.0 | Original paper | Teacher forcing ratio during training |
| `max_src_length` | 40 | Original paper | Maximum source sequence length |
| `max_trg_length` | 54 | Original paper | Maximum target sequence length |

### Training Parameters

| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `batch_size` | 128 | Original config_best.json | Training batch size |
| `learning_rate` | 0.001 | Original config_best.json | Adam optimizer learning rate |
| `max_epoch` | 100 | Default | Maximum training epochs |
| `optimizer` | "adam" | Original paper | Optimizer type |
| `weight_decay` | 0.0001 | Original paper | L2 regularization weight |
| `lr_scheduler` | "exponentiallr" | Original paper | Learning rate scheduler |
| `lr_decay` | 0.5 | Original paper | LR decay factor |
| `lr_step` | 5 | Original paper | Decay every N epochs |
| `clip_grad_norm` | true | Original paper | Enable gradient clipping |
| `max_grad_norm` | 5.0 | Original paper | Maximum gradient norm |
| `use_early_stop` | true | Original paper | Enable early stopping |
| `patience` | 3 | Original paper | Early stopping patience |

### Data Processing Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `input_max_len` | 40 | Maximum input trajectory length |
| `output_max_len` | 54 | Maximum output segment sequence length |
| `grid_size` | 0.001 | Grid size for discretization |
| `train_rate` | 0.7 | Training set ratio |
| `eval_rate` | 0.15 | Validation set ratio |
| `cache_dataset` | true | Cache preprocessed dataset |
| `num_workers` | 0 | Number of data loading workers |

### Segmentation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `enable_segmentation` | true | Enable trajectory segmentation |
| `segment_window_src` | 40 | Source segment window size |
| `segment_window_trg` | 54 | Target segment window size |
| `segment_stride_ratio` | 0.5 | Stride ratio for sliding window |
| `min_segment_length` | 10 | Minimum segment length |

## Time Encoding Options

DeepMM supports three time encoding modes:

1. **NoEncoding** (default): No time information encoded
2. **OneEncoding**: Single time embedding (e.g., timestamp)
3. **TwoEncoding**: Dual time embeddings (e.g., hour and minute separately)

### Configuration Examples

For OneEncoding:
```json
{
  "time_encoding": "OneEncoding",
  "src_tim_emb_dim": 64,
  "src_tim_vocab_size": 1000
}
```

For TwoEncoding:
```json
{
  "time_encoding": "TwoEncoding",
  "src_tim_emb_dim": [64, [32, 32]],
  "src_tim_vocab_size": [1000, [24, 60]]
}
```

## Attention Mechanism Types

DeepMM supports three attention types:

1. **dot** (default): Simple dot product attention
2. **general**: Linear transformation before dot product
3. **mlp**: Multi-layer perceptron attention

Reference: http://www.aclweb.org/anthology/D15-1166

## Dataset Requirements

### Required Data Features

The model expects the following features from the dataset:

| Feature | Type | Description |
|---------|------|-------------|
| `src_loc_vocab_size` | int | Size of source location vocabulary |
| `trg_seg_vocab_size` | int | Size of target segment vocabulary |
| `pad_token_src_loc` | int | Padding token ID for source locations |
| `pad_token_trg` | int | Padding token ID for target segments |
| `src_tim_vocab_size` | int or list | Time vocabulary size(s) (if using time encoding) |

### Input Data Format

Expected batch format:
```python
batch = {
    'input_src': torch.Tensor,      # (batch_size, src_seq_len) - source location IDs
    'input_trg': torch.Tensor,      # (batch_size, trg_seq_len) - target segment IDs for teacher forcing
    'output_trg': torch.Tensor,     # (batch_size, trg_seq_len) - ground truth target segments
    'input_time': torch.Tensor or list  # (optional) time information
}
```

### Compatible Datasets

- **Map Matching Task**: global, Seattle, Neftekamsk, Valky, Ruzhany, Santander, Spaichingen, NovoHamburgo
- **Traj Loc Pred Task**: foursquare_tky, foursquare_nyc, gowalla, foursquare_serm, Proto

## Executor and Evaluator Configuration

### Map Matching Task
- **Dataset Class**: `DeepMMSeq2SeqDataset`
- **Executor**: `DeepMapMatchingExecutor`
- **Evaluator**: `MapMatchingEvaluator`
- **Metrics**: RMF (Route Mismatch Fraction), AN (Average Number of matched segments), AL (Average Length)

### Trajectory Location Prediction Task
- **Dataset Class**: `DeepMMSeq2SeqDataset`
- **Executor**: `TrajLocPredExecutor`
- **Evaluator**: `TrajLocPredEvaluator`
- **Traj Encoder**: `StandardTrajectoryEncoder`

## Model Interface Methods

The DeepMM model implements the following key methods:

### 1. forward(batch)
Performs forward pass through the encoder-decoder architecture.
- **Input**: Batch dictionary with source and target sequences
- **Output**: Decoder logits of shape (batch, trg_seq_len, vocab_size)

### 2. predict(batch)
Generates predictions for evaluation.
- **Input**: Batch dictionary
- **Output**: Predicted segment IDs of shape (batch, trg_seq_len)

### 3. calculate_loss(batch)
Computes cross-entropy loss for training.
- **Input**: Batch dictionary with ground truth targets
- **Output**: Scalar loss tensor
- **Loss Function**: Cross-entropy with padding token ignored

## Training Configuration

### Original DeepMM Training Settings
Based on the original implementation:
- **Optimizer**: Adam with learning rate 0.001
- **LR Scheduler**: Exponential decay (factor 0.5 every 5 epochs)
- **Batch Size**: 128
- **Early Stopping**: Enabled with patience of 3 epochs
- **Gradient Clipping**: Max norm of 5.0

### LibCity Integration Notes
Most training parameters are now controlled by:
1. **Model Config**: Model-specific hyperparameters (embedding dims, hidden dims, etc.)
2. **Executor Config**: Training procedures (lr scheduling, early stopping, etc.)
3. **Task Config**: Dataset, executor, and evaluator assignments

## Key Adaptations from Original Code

The model was adapted from the original DeepMM repository with the following changes:

1. **Removed deprecated torch.autograd.Variable wrappers**
2. **Replaced hardcoded .cuda() calls with device abstraction**
3. **Updated deprecated functions** (F.sigmoid() → torch.sigmoid())
4. **Implemented LibCity AbstractModel interface**
5. **Added predict() and calculate_loss() methods for LibCity compatibility**
6. **Maintained original architecture**: Seq2SeqAttention, LSTMAttentionDot, SoftDotAttention

## Usage Example

### Running with Map Matching Task
```bash
python run_model.py --task map_matching --model DeepMM --dataset Seattle
```

### Running with Trajectory Location Prediction Task
```bash
python run_model.py --task traj_loc_pred --model DeepMM --dataset foursquare_nyc
```

### Custom Configuration
```bash
python run_model.py --task map_matching --model DeepMM --dataset Seattle \
    --config_file custom_deepmm.json
```

## Configuration Compatibility Notes

### Key Differences Between Tasks

| Aspect | map_matching | traj_loc_pred |
|--------|--------------|---------------|
| Executor | DeepMapMatchingExecutor | TrajLocPredExecutor |
| Evaluator | MapMatchingEvaluator | TrajLocPredEvaluator |
| Dataset Class | DeepMMSeq2SeqDataset | DeepMMSeq2SeqDataset |
| Primary Use Case | GPS-to-road matching | General trajectory prediction |

### Shared Configuration Parameters
Both tasks use the same core model parameters (embedding dims, hidden dims, attention type, etc.) but may differ in:
- Dataset preprocessing
- Evaluation metrics
- Executor-specific training procedures

## Validation Checklist

- [x] DeepMM added to `traj_loc_pred.allowed_model` list
- [x] DeepMM already registered in `map_matching.allowed_model` list
- [x] Task configuration created for `traj_loc_pred.DeepMM`
- [x] Task configuration exists for `map_matching.DeepMM`
- [x] Model config created at `config/model/traj_loc_pred/DeepMM.json`
- [x] Model config exists at `config/model/map_matching/DeepMM.json`
- [x] All hyperparameters documented with sources
- [x] Dataset requirements specified
- [x] JSON syntax validated

## References

1. **Original Repository**: repos/DeepMM/
2. **Original Model Implementation**: repos/DeepMM/DeepMM/model.py
3. **Original Config**: repos/DeepMM/DeepMM/config_best.json
4. **Paper**: "DeepMM: Deep Learning based Map Matching with Heterogeneous Trajectory Data"
5. **Attention Mechanism**: http://www.aclweb.org/anthology/D15-1166

## Notes and Recommendations

### Dataset Compatibility
- DeepMM requires tokenized GPS locations and road segments
- Both source and target vocabularies must be provided by the dataset
- Sequence lengths should be capped at configured max lengths (40 for source, 54 for target)

### Memory Considerations
- Bidirectional encoder doubles memory usage
- Attention mechanism scales quadratically with sequence length
- Batch size may need adjustment based on GPU memory

### Performance Tuning
- `teacher_forcing_ratio` can be decreased for more robust generation
- `segment_window_src/trg` affects how long trajectories are split
- `segment_stride_ratio` controls overlap between segments

### Future Enhancements
- Beam search decoding for inference
- Scheduled sampling for teacher forcing
- Attention visualization utilities

## Change Log

### 2026-02-03
- Added DeepMM to `traj_loc_pred` task allowed models
- Created DeepMM task configuration for `traj_loc_pred`
- Created model config file at `config/model/traj_loc_pred/DeepMM.json`
- Documented all configuration parameters and dataset requirements
- Verified existing `map_matching` task configuration
