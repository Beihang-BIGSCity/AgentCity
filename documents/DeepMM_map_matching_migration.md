# DeepMM Map Matching Model - Migration Summary

## Overview

The DeepMM model has been adapted from the original standalone PyTorch implementation
to the LibCity framework for the **map matching** task.

## File Locations

### Source (Original)
- `/home/wangwenrui/shk/AgentCity/repos/DeepMM/DeepMM/model.py`
  - Primary class: `Seq2SeqAttention` (lines 825-1035)
  - Supporting classes: `SoftDotAttention` (lines 303-388), `LSTMAttentionDot` (lines 391-443)

### Target (Adapted)
- Model: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/DeepMM.py`
- Registration: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/__init__.py`
- Config: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/map_matching/DeepMM.json`
- Task Config: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json` (map_matching section)

## Architecture

```
GPS Trajectory (input_src)        Road Segments (input_trg, teacher forcing)
        |                                    |
  [src_embedding]                     [trg_embedding]
        |                                    |
  [time encoding] (optional)                 |
        |                                    |
  [Bidirectional LSTM Encoder]               |
        |                                    |
        +-- encoder2decoder bridge ------->  |
        |                                    |
        +-- attention context -----------> [LSTMAttentionDot Decoder]
                                             |
                                      [decoder2vocab]
                                             |
                                      logits (trg_seg_vocab_size)
```

## Key Changes from Original

| Aspect | Original | Adapted |
|--------|----------|---------|
| Base class | `nn.Module` | `AbstractModel` |
| Constructor | 18 explicit parameters | `config` dict + `data_feature` dict |
| Forward signature | `forward(input_src, input_trg, input_time, ctx_mask, trg_mask)` | `forward(batch)` - extracts from dict |
| Device management | Hardcoded `.cuda()` calls | Uses `self.device` from config |
| Variable usage | `torch.autograd.Variable` (deprecated) | Direct tensor creation with `device=` |
| Activation functions | `F.sigmoid()`, `F.tanh()` (deprecated) | `torch.sigmoid()`, `torch.tanh()` |
| decoder2vocab | `nn.Linear(...).cuda()` | `nn.Linear(...)` (device handled by framework) |
| Loss computation | External (in training script) | `calculate_loss(batch)` method |
| Prediction | External (in training script) | `predict(batch)` method |

## Required Config Parameters

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
  "attn_type": "dot"
}
```

## Required data_feature Keys

| Key | Type | Description |
|-----|------|-------------|
| `src_loc_vocab_size` | int | Number of unique source location tokens |
| `trg_seg_vocab_size` | int | Number of unique target road segment tokens |
| `pad_token_src_loc` | int | Padding index for source locations |
| `pad_token_trg` | int | Padding index for target segments |
| `src_tim_vocab_size` | int/list | (Optional) Time vocabulary size(s) |
| `pad_token_src_tim1` | int | (Optional) Padding for OneEncoding time |
| `pad_token_src_tim2` | list[int] | (Optional) Padding for TwoEncoding time |

## Expected Batch Format

The model expects a dictionary-like batch object with these keys:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `input_src` | (B, src_len) | LongTensor | Source GPS location token IDs |
| `input_trg` | (B, trg_len) | LongTensor | Target segment IDs (shifted for teacher forcing) |
| `output_trg` | (B, trg_len) | LongTensor | Ground truth target segment IDs |
| `input_time` | (B, src_len) | LongTensor | (Optional) Time token IDs |

## Executor and Dataset

- **Executor**: `DeepMapMatchingExecutor` - handles training, validation, and evaluation
- **Dataset**: `DeepMMSeq2SeqDataset` - prepares seq2seq batches with input_src, input_trg, output_trg
- **Evaluator**: `MapMatchingEvaluator` - computes RMF, AN, AL metrics

## Assumptions and Limitations

1. Teacher forcing is used during training (decoder receives ground truth shifted input).
2. During inference (`predict`), the model still requires `input_trg` -- for autoregressive
   generation, the caller should provide start-of-sequence tokens or implement beam search externally.
3. Time encoding is disabled by default (`NoEncoding`). Enable with `OneEncoding` or `TwoEncoding`
   and provide the corresponding vocabulary sizes and embedding dimensions.
4. The model preserves the original attention mechanism exactly, including support for
   dot, general, and mlp attention types.
