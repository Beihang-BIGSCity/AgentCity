# DeepMM Repository Analysis - Phase 1

## Repository: DeepMM (Deep Map Matching)
- **URL**: https://github.com/vonfeng/DeepMapMatching
- **Cloned to**: /home/wangwenrui/shk/AgentCity/repos/DeepMM
- **Paper**: Deep Learning Based Map Matching with Data Augmentation (IEEE TMC 2020 / SIGSPATIAL 2019)
- **Task**: Map Matching for GPS trajectories to road networks

---

## Executive Summary

DeepMM is a deep learning-based map matching model that uses sequence-to-sequence learning with attention mechanism to map sparse and noisy GPS trajectories onto accurate road networks. The model treats map matching as a trajectory-to-road segment translation problem, leveraging the seq2seq framework with LSTM encoders and decoders.

**Key Innovation**: Unlike traditional HMM-based methods, DeepMM performs matching in latent space with high tolerance to noise and incorporates mobility patterns learned from trajectory big data.

---

## Repository Structure

```
DeepMM/
├── DeepMM/              # Main model implementation
│   ├── model.py         # Core seq2seq model architectures
│   ├── seq2seq.py       # Training script (main entry point)
│   ├── data_utils.py    # Data loading and preprocessing utilities
│   ├── evaluate.py      # Evaluation metrics and inference
│   ├── opt.py           # Configuration generator
│   ├── demonstration.py # Demo/visualization script
│   └── configs/         # JSON configuration files
│       ├── config_best.json
│       ├── config_vanilla.json
│       ├── noise/       # Noise level experiments
│       ├── time_gap/    # Temporal sampling experiments
│       ├── attention_tune_*/  # Hyperparameter tuning
│       └── ...
├── preprocess/          # Data preprocessing pipeline
├── postprocess/         # Result processing and visualization
├── baselines/           # FMM and ST-Matching baselines
├── TraceGen/            # Java code for trajectory generation
└── data/                # Sample data and maps
```

---

## Key Files Analysis

### 1. Model Definition Files

#### `/home/wangwenrui/shk/AgentCity/repos/DeepMM/DeepMM/model.py` (1374 lines)
**Main Model Classes:**

1. **`Seq2Seq`** (lines 446-657)
   - Basic encoder-decoder without attention
   - Bidirectional LSTM encoder
   - LSTM decoder
   - Optional time encoding (NoEncoding/OneEncoding/TwoEncoding)

2. **`Seq2SeqAttention`** (lines 825-1034) - **PRIMARY MODEL**
   - **Encoder**: Bidirectional LSTM/GRU
   - **Decoder**: LSTM with attention (LSTMAttentionDot)
   - **Attention**: Soft dot-product attention (SoftDotAttention)
   - Supports location + temporal embeddings
   - Configurable attention types: dot, general, mlp

3. **`Seq2SeqFastAttention`** (lines 1182-1373)
   - Fast attention using batch matrix multiplication
   - Efficient for longer sequences

4. **Supporting Components:**
   - `LSTMAttentionDot` (lines 391-443): LSTM cell with soft attention
   - `SoftDotAttention` (lines 303-388): Attention mechanism
   - `StackedAttentionLSTM` (lines 10-59): Multi-layer attention LSTM
   - `DeepBidirectionalLSTM` (lines 62-129): Deep bidirectional encoder

**Input/Output:**
- **Input**: 
  - GPS location sequence (embedded)
  - Optional: Time intervals (1 or 2 encodings)
- **Output**: Road segment sequence

---

### 2. Training Script

#### `/home/wangwenrui/shk/AgentCity/repos/DeepMM/DeepMM/seq2seq.py` (420 lines)
**Main Training Logic:**
- Entry point: `if __name__ == '__main__':`
- Uses MLflow for experiment tracking
- Configurable via JSON config files
- Supports 3 model variants: vanilla, attention, fastattention
- Training features:
  - Adam/Adadelta/SGD optimizers
  - Learning rate decay (exponential: 0.5 every 5 epochs)
  - Early stopping (patience=3, after epoch 10)
  - Batch processing
  - Train/valid/test accuracy monitoring

**Key Training Parameters:**
```python
- batch_size: 128 (default)
- learning_rate: 0.001
- max_epochs: 100
- optimizer: adam
- dropout: 0.5
- n_layers_src: 2 (encoder)
- n_layers_trg: 1 (decoder)
- hidden_dim: 512
- embedding_dim: 256
```

---

### 3. Configuration Files

#### `/home/wangwenrui/shk/AgentCity/repos/DeepMM/DeepMM/configs/config_best.json`
```json
{
  "model": {
    "seq2seq": "attention",
    "src_hidden_dim": 512,
    "trg_hidden_dim": 512,
    "dim_loc_src": 256,      // Location embedding
    "dim_seg_trg": 256,      // Road segment embedding
    "max_src_length": 40,    // Max GPS points
    "max_trg_length": 54,    // Max road segments
    "n_layers_src": 2,
    "n_layers_trg": 1,
    "bidirectional": true,
    "dropout": 0.5,
    "time_encoding": "NoEncoding"  // or OneEncoding/TwoEncoding
  },
  "training": {
    "optimizer": "adam",
    "batch_size": 128,
    "lrate": 0.001
  },
  "data": {
    "folder": "../timegap-60_noise-gaussian_sigma-100/dup-10_sl-100/trace_700000/with_real_train/",
    "train": {
      "src_loc": "train.block",
      "src_tim1": "train.time1",
      "src_tim2": "train.time2",
      "trg_seg": "train.seg"
    }
  }
}
```

**Multiple Config Variants:**
- Noise levels: 10, 20, 40, 60, 80, 100, 120 meters
- Time gaps: 30, 40, 60, 80, 100, 120 seconds
- Data augmentation: different duplication factors
- Hyperparameter tuning: embedding/hidden dimensions

---

### 4. Data Loading

#### `/home/wangwenrui/shk/AgentCity/repos/DeepMM/DeepMM/data_utils.py` (335 lines)
**Key Functions:**

1. **`read_nmt_data(dataset, config)`** (lines 108-147)
   - Reads train/valid/test data
   - Supports 3 time encoding modes:
     - `NoEncoding`: Location only
     - `OneEncoding`: Location + time interval
     - `TwoEncoding`: Location + hour + minute
   - Returns: src, trg, vocab_sizes

2. **`construct_vocab(lines)`** (lines 31-80)
   - Builds vocabulary with special tokens: `<s>`, `<pad>`, `</s>`, `<unk>`
   
3. **`get_minibatch(...)`** (lines 205-253)
   - Batches sequences with padding
   - Adds start/end tokens
   - Returns PyTorch Variables (GPU)

**Data Format:**
- **Source (GPS)**: 
  - `train.block`: GPS grid block IDs
  - `train.time1`: Time intervals (single encoding)
  - `train.time2`: Hour-minute pairs (two encoding)
- **Target (Road)**: 
  - `train.seg`: Road segment IDs

---

### 5. Evaluation

#### `/home/wangwenrui/shk/AgentCity/repos/DeepMM/DeepMM/evaluate.py` (517 lines, first 100 shown)
**Metrics:**
- Primary: Sequence accuracy (exact match)
- BLEU score (n-gram overlap)
- Custom `evaluate_accuracy()` function

**Inference:**
- Greedy decoding
- Auto-regressive generation
- Max length: 54 tokens

---

## Preprocessing Pipeline

Located in `/home/wangwenrui/shk/AgentCity/repos/DeepMM/preprocess/`:

1. **Step 1**: `preprocess_step1_cut.py` - Cut trajectories into segments
2. **Step 2**: `preprocess_step2_filter_*.py` - Filter valid trajectories (gen/real)
3. **Step 3**: `preprocess_step3_traintest_split.py` - Train/test split
4. **Step 4**: `preprocess_step4_addnoise_*.py` - Add Gaussian noise to GPS
5. **Step 5**: `preprocess_step5_construct_*_data.py` - Convert to model input format
6. **Step 6**: `preprocess_step6_combine.py` - Combine generated + real data

**Key Preprocessing Features:**
- Gaussian noise simulation (configurable sigma: 10-120m)
- Temporal downsampling (configurable time gaps)
- Data augmentation through duplication
- OSM trajectory utilities

---

## Dependencies

### Core Dependencies (Inferred from code):
```
Python: 3.x (likely 3.6+)
PyTorch: 1.x (uses torch.autograd.Variable, older API)
  - torch
  - torch.nn
  - torch.optim
  - torch.nn.functional

MLflow: For experiment tracking
setproctitle: Process naming
numpy
json (stdlib)
argparse (stdlib)
```

### Additional Tools:
- GraphHopper (Java): Trajectory generation
- FMM library: Baseline map matching
- OSM utilities: Map data processing

**Note**: No `requirements.txt` or `setup.py` found. Dependencies must be inferred.

---

## Model Architecture Details

### Seq2SeqAttention (Primary Model)

```
INPUT: GPS Sequence
  └─> Location Embedding (256-dim)
  └─> Time Embedding (optional, 16/32-dim)
  └─> Concatenate → [batch, seq_len, emb_dim]
  
ENCODER: Bidirectional LSTM
  └─> Hidden: 512-dim (256 each direction)
  └─> Layers: 2
  └─> Output: Context vectors [batch, src_len, 512]
  
ATTENTION DECODER: LSTMAttentionDot
  └─> Hidden: 512-dim
  └─> Layers: 1
  └─> Attention: Soft dot-product over encoder outputs
  └─> Output: [batch, trg_len, 512]
  
OUTPUT: Road Segment Prediction
  └─> Linear: 512 → vocab_size
  └─> Softmax → Road segment probabilities
```

### Attention Mechanism Types:
1. **Dot**: Direct dot product (default)
2. **General**: Learned linear transformation
3. **MLP**: Multi-layer perceptron attention

---

## Input/Output Data Formats

### Input Format (GPS Trajectory):
```python
# Example: train.block (location)
"123 456 789 234 567"  # Grid block IDs

# Example: train.time1 (one encoding)
"30 60 45 30"  # Time intervals in seconds

# Example: train.time2 (two encoding)
"14-30 14-35 14-40 14-45"  # Hour-Minute pairs
```

### Output Format (Road Network):
```python
# Example: train.seg (road segments)
"seg_001 seg_002 seg_003 seg_002 seg_004"  # Road segment IDs
```

### Vocabulary Structure:
- Special tokens: `<s>` (start), `</s>` (end), `<pad>` (padding), `<unk>` (unknown)
- Location vocab: ~thousands of grid blocks
- Time vocab: 
  - OneEncoding: time intervals (0-max_interval)
  - TwoEncoding: hours (0-23) + minutes (0-59)
- Road segment vocab: ~thousands of road segments

---

## Training Process

### Command Example:
```bash
python seq2seq.py --gpu=0 --config=configs/config_best.json \
    --epoch=100 --batch_size=128 --lr=0.001 \
    --seq2seq=attention --rnn_type=LSTM --attn_type=dot
```

### Training Flow:
1. Load data (train/valid/test)
2. Build vocabularies
3. Initialize model (Seq2SeqAttention)
4. Training loop:
   - Forward pass
   - CrossEntropyLoss (ignore padding)
   - Backprop + Adam optimizer
   - LR decay every 5 epochs
   - Evaluate on valid set every epoch
   - Early stopping (patience=3)
5. Save final model
6. Evaluate on test set

### Output:
- Model checkpoint: `models/{experiment_name}/final.model`
- Logs: `logs/{experiment_name}_lr-decay.log`
- Samples: `samples/{experiment_name}/train_epoch_*.samp`
- MLflow artifacts

---

## Baseline Methods

Located in `/home/wangwenrui/shk/AgentCity/repos/DeepMM/baselines/`:
- **FMM**: Fast Map Matching (IJGIS 2018)
- **ST-Matching**: Spatio-temporal map matching (SIGSPATIAL 2009)
- Implementation based on: https://github.com/cyang-kth/fmm

---

## Data Augmentation Strategy

DeepMM's key contribution is data augmentation for map matching:

1. **Generated Trajectories**:
   - Use GraphHopper to generate synthetic trajectories
   - Follow shortest path + perturbation
   - Guaranteed ground truth road segments

2. **Real Trajectory Augmentation**:
   - Duplicate real trajectories
   - Add varying levels of Gaussian noise
   - Different temporal downsampling rates

3. **Combination**:
   - Mix generated + real data
   - Configurable ratios (e.g., 700k generated + 10x real)

---

## Structure Notes

### Code Organization:
- **Well-structured**: Clear separation of model/training/data/evaluation
- **Modular**: Multiple seq2seq variants in single file
- **Configurable**: JSON-based hyperparameter management
- **Production-ready**: MLflow integration, logging, checkpointing

### Potential Issues for Migration:
1. **PyTorch API**: Uses older `Variable` (PyTorch 0.4-1.0 style)
   - Modern PyTorch: tensors are autograd by default
2. **Hard-coded CUDA**: `.cuda()` calls throughout
   - Need device-agnostic code
3. **No requirements.txt**: Dependencies unclear
4. **Absolute paths**: Some hardcoded paths in code
5. **MLflow dependency**: May conflict with LibCity's workflow

---

## Migration Priorities for LibCity

### High Priority:
1. **Model Architecture** (`Seq2SeqAttention`):
   - Core attention-based seq2seq
   - Clean implementation, well-documented
   
2. **Data Format**:
   - Understand GPS→Road segment mapping
   - Vocabulary construction
   - Padding/masking strategy

3. **Training Loop**:
   - Loss function (CrossEntropyLoss with weight mask)
   - Learning rate schedule
   - Early stopping logic

### Medium Priority:
1. **Preprocessing Pipeline**:
   - Noise simulation
   - Data augmentation
   - OSM trajectory utilities

2. **Evaluation Metrics**:
   - Sequence accuracy
   - BLEU score adaptation

3. **Configuration System**:
   - Map JSON configs to LibCity's config format

### Low Priority:
1. **Baselines** (FMM, ST-Matching): Can reference separately
2. **TraceGen** (Java): Not needed for model migration
3. **Postprocessing**: Visualization/analysis tools

---

## Next Steps for Phase 2 (Migration)

1. **Create LibCity model class**: `DeepMM` in `libcity/model/map_matching/`
2. **Adapt model architecture**:
   - Inherit from `AbstractModel`
   - Update PyTorch API (remove `Variable`)
   - Device-agnostic code
3. **Create data executor**: `DeepMMDataExecutor`
   - Convert LibCity trajectory format to DeepMM input
   - Vocabulary management
4. **Create config**: `DeepMM.json` in LibCity format
5. **Implement evaluator**: Map matching metrics
6. **Test on sample data**: Verify correctness

---

## Key Hyperparameters Reference

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `src_hidden_dim` | 512 | 128-1024 | Encoder LSTM hidden size |
| `trg_hidden_dim` | 512 | 128-1024 | Decoder LSTM hidden size |
| `dim_loc_src` | 256 | 64-1024 | Location embedding dim |
| `dim_seg_trg` | 256 | 64-1024 | Road segment embedding dim |
| `n_layers_src` | 2 | 1-4 | Encoder layers |
| `n_layers_trg` | 1 | 1-2 | Decoder layers |
| `dropout` | 0.5 | 0.0-0.7 | Dropout rate |
| `batch_size` | 128 | 32-256 | Batch size |
| `learning_rate` | 0.001 | 0.0001-0.01 | Initial LR |
| `max_src_length` | 40 | 20-100 | Max GPS points |
| `max_trg_length` | 54 | 20-120 | Max road segments |

---

## Citation

```bibtex
@article{feng2020deepmm,
  title={DeepMM: Deep learning based map matching with data augmentation},
  author={Feng, Jie and Li, Yong and Zhao, Kai and Xu, Zhao and Xia, Tong and Zhang, Jinglin and Jin, Depeng},
  journal={IEEE Transactions on Mobile Computing},
  volume={21},
  number={7},
  pages={2372--2384},
  year={2020},
  publisher={IEEE}
}
```

---

## Contact & Notes

- **Original Authors**: Jie Feng, Yong Li, et al.
- **GitHub**: https://github.com/vonfeng/DeepMapMatching
- **Clone Status**: ✓ Successfully cloned to `./repos/DeepMM`
- **Analysis Date**: 2026-02-02
- **Analysis Status**: Phase 1 Complete - Ready for Migration

---

## Appendix: File Tree

```
repos/DeepMM/
├── DeepMM/
│   ├── configs/
│   │   ├── config_best.json
│   │   ├── config_vanilla.json
│   │   ├── noise/*.json (7 files)
│   │   ├── time_gap/*.json (6 files)
│   │   ├── attention_tune_emb/*.json (7 files)
│   │   ├── attention_tune_hidden/*.json (4 files)
│   │   ├── dup_num/*.json (4 files)
│   │   ├── gen_num/*.json (3 files)
│   │   ├── real/*.json (6 files)
│   │   └── shortest/*.json (4 files)
│   ├── model.py (1374 lines)
│   ├── seq2seq.py (420 lines)
│   ├── data_utils.py (335 lines)
│   ├── evaluate.py (517 lines)
│   ├── opt.py (56 lines)
│   ├── demonstration.py (285 lines)
│   └── run_*.sh (9 scripts)
├── preprocess/ (13 Python files)
├── postprocess/ (5 Python files)
├── baselines/ (fmm_matching.py)
├── TraceGen/ (Java files)
└── data/
    ├── map/ (OSM Beijing data)
    ├── osm_traj/ (Sample trajectories)
    ├── real/ (Real trajectory data)
    └── tencent/ (Additional data)
```

---

**End of Phase 1 Analysis**
