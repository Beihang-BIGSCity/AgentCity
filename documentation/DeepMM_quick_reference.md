# DeepMM Quick Reference Guide

## Repository Info
- **URL**: https://github.com/vonfeng/DeepMapMatching  
- **Local Path**: `/home/wangwenrui/shk/AgentCity/repos/DeepMM`
- **Task**: Map Matching (GPS → Road Network)
- **Approach**: Seq2Seq with Attention

---

## Main Model Class

**Location**: `/home/wangwenrui/shk/AgentCity/repos/DeepMM/DeepMM/model.py`

**Class Name**: `Seq2SeqAttention` (lines 825-1034)

**Architecture**:
```
GPS Sequence → Location Embedding (256) + Time Embedding (optional)
             → Bidirectional LSTM Encoder (2 layers, 512 hidden)
             → Attention Decoder (1 layer, 512 hidden)
             → Road Segment Logits → Softmax
```

**Key Parameters**:
- Encoder: Bidirectional LSTM, 2 layers, 512 hidden
- Decoder: LSTM + Attention, 1 layer, 512 hidden
- Embeddings: 256-dim (location), 256-dim (road segment)
- Attention: Dot-product / General / MLP
- Max sequence length: 40 (input), 54 (output)

---

## Training Script

**Location**: `/home/wangwenrui/shk/AgentCity/repos/DeepMM/DeepMM/seq2seq.py`

**Command**:
```bash
python seq2seq.py --gpu=0 --config=configs/config_best.json
```

**Key Arguments**:
- `--config`: JSON config file path
- `--gpu`: GPU device ID
- `--batch_size`: Default 128
- `--lr`: Learning rate (0.001)
- `--seq2seq`: Model type (vanilla/attention/fastattention)
- `--rnn_type`: LSTM or GRU
- `--attn_type`: dot/general/mlp

---

## Configuration

**Best Config**: `/home/wangwenrui/shk/AgentCity/repos/DeepMM/DeepMM/configs/config_best.json`

**Key Settings**:
```json
{
  "model": {
    "seq2seq": "attention",
    "src_hidden_dim": 512,
    "trg_hidden_dim": 512,
    "dim_loc_src": 256,
    "dim_seg_trg": 256,
    "n_layers_src": 2,
    "n_layers_trg": 1,
    "bidirectional": true,
    "dropout": 0.5,
    "max_src_length": 40,
    "max_trg_length": 54
  },
  "training": {
    "optimizer": "adam",
    "lrate": 0.001,
    "batch_size": 128
  }
}
```

---

## Data Format

**Input Files**:
- `train.block`: GPS grid block IDs (space-separated)
- `train.time1`: Time intervals in seconds (optional)
- `train.time2`: Hour-minute pairs (optional)
- `train.seg`: Road segment IDs (ground truth)

**Example**:
```
# train.block (GPS)
123 456 789 234 567

# train.seg (Road)
seg_001 seg_002 seg_003 seg_002 seg_004
```

**Vocabulary**:
- Special tokens: `<s>`, `</s>`, `<pad>`, `<unk>`
- Dynamic vocab built from training data

---

## Data Loading

**File**: `/home/wangwenrui/shk/AgentCity/repos/DeepMM/DeepMM/data_utils.py`

**Key Function**: `read_nmt_data(dataset, config)`
- Returns: `src, trg, src_loc_vocab_size, src_tim_vocab_size, trg_seg_vocab_size`
- Supports 3 time encodings: NoEncoding, OneEncoding, TwoEncoding

**Batching**: `get_minibatch(lines, word2ind, index, batch_size, max_len, ...)`

---

## Dependencies

```
Python >= 3.6
PyTorch >= 1.0 (uses older Variable API)
MLflow
setproctitle
numpy
```

**Note**: No requirements.txt provided; inferred from imports.

---

## Model Variants

1. **Seq2Seq** (vanilla): Basic encoder-decoder, no attention
2. **Seq2SeqAttention** (best): LSTM + soft attention ⭐
3. **Seq2SeqFastAttention**: Efficient batch attention
4. **Seq2SeqAttentionSharedEmbedding**: Shared embeddings
5. **Seq2SeqAutoencoder**: Autoencoding variant

---

## Preprocessing Pipeline

Located in `/home/wangwenrui/shk/AgentCity/repos/DeepMM/preprocess/`:

1. `preprocess_step1_cut.py`: Segment trajectories
2. `preprocess_step2_filter_*.py`: Filter valid data
3. `preprocess_step3_traintest_split.py`: Split datasets
4. `preprocess_step4_addnoise_*.py`: Add Gaussian noise
5. `preprocess_step5_construct_*_data.py`: Format conversion
6. `preprocess_step6_combine.py`: Combine gen + real data

---

## Evaluation

**File**: `/home/wangwenrui/shk/AgentCity/repos/DeepMM/DeepMM/evaluate.py`

**Metrics**:
- Sequence accuracy (exact match)
- BLEU score
- Custom accuracy function

**Decoding**: Greedy (auto-regressive)

---

## Migration Checklist

- [ ] Extract `Seq2SeqAttention` class
- [ ] Update PyTorch API (remove Variable)
- [ ] Create LibCity executor for data loading
- [ ] Map config to LibCity format
- [ ] Implement evaluation metrics
- [ ] Add device-agnostic code (remove hardcoded .cuda())
- [ ] Create vocab builder compatible with LibCity
- [ ] Test on sample data

---

## Key File Paths

```
Model:       repos/DeepMM/DeepMM/model.py (line 825)
Training:    repos/DeepMM/DeepMM/seq2seq.py
Data Utils:  repos/DeepMM/DeepMM/data_utils.py
Evaluation:  repos/DeepMM/DeepMM/evaluate.py
Config:      repos/DeepMM/DeepMM/configs/config_best.json
```

---

## Citation

```bibtex
@article{feng2020deepmm,
  title={DeepMM: Deep learning based map matching with data augmentation},
  author={Feng, Jie and Li, Yong and others},
  journal={IEEE Transactions on Mobile Computing},
  year={2020}
}
```

---

**Status**: ✓ Cloned and Analyzed  
**Date**: 2026-02-02  
**Next**: Phase 2 Migration to LibCity
