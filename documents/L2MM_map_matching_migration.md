# L2MM Migration Summary

## Overview

L2MM (Latent-to-Map Matching) has been adapted from the original repository
at `repos/L2MM/mapmatching/` to the LibCity framework as a map matching model.

## File Locations

### Created Files

- **Model**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/L2MM.py`
- **Config**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/map_matching/L2MM.json`

### Modified Files

- **Map Matching Init**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/__init__.py`
  - Added `from libcity.model.map_matching.L2MM import L2MM`
  - Added `"L2MM"` to `__all__`
- **Task Config**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
  - Added `"L2MM"` to `map_matching.allowed_model`
  - Added `L2MM` task entry with dataset/executor/evaluator

### Original Source Files

| Original File | Purpose |
|---|---|
| `repos/L2MM/mapmatching/model.py` | EncoderDecoder, Encoder, Decoder, LatentDistribution, StackingGRUCell, GlobalAttention |
| `repos/L2MM/mapmatching/util.py` | DenseLoss (masked cross-entropy) |
| `repos/L2MM/mapmatching/train.py` | Training loop and loss combination logic |
| `repos/L2MM/mapmatching/train_with_bucket.py` | Bucket training, evaluation, pretrain pipeline |
| `repos/L2MM/mapmatching/evaluate.py` | Autoregressive decoding for inference |
| `repos/L2MM/mapmatching/begin.py` | Default hyperparameters (Args class) |
| `repos/L2MM/mapmatching/data_util.py` | DataLoader with bucketing |
| `repos/L2MM/mapmatching/init_latent.py` | KMeans-based latent cluster initialization |

## Architecture

The L2MM model has the following pipeline:

1. **Encoder**: Bidirectional GRU over grid cell embeddings
2. **LatentDistribution**: VAE-style module that maps encoder hidden state to latent z
   - Pretrain mode: sample z, no latent loss
   - Train mode: sample z with Gaussian mixture KL loss + category loss
   - Test mode: deterministic mu_z (no sampling)
3. **Decoder**: Stacked GRU cells with GlobalAttention over encoder outputs
4. **Output Projection**: Linear + LogSoftmax over road vocabulary

## Key Adaptations

### 1. Model Merging
The original code splits the model into two modules:
- `m0` (EncoderDecoder): encoder + latent distribution + decoder
- `m1` (nn.Sequential): Linear + LogSoftmax output layer

These are merged into a single `L2MM(AbstractModel)` class with `output_projection`
as a submodule, enabling standard LibCity single-model training.

### 2. Deprecated PyTorch Syntax
- `torch.autograd.Variable` removed; direct tensors used instead
- `nn.Softmax()` replaced with `nn.Softmax(dim=-1)`
- `nn.LogSoftmax()` replaced with `nn.LogSoftmax(dim=-1)`
- `clip_grad_norm` updated to `clip_grad_norm_` (handled by executor)

### 3. Device Handling
- All hardcoded `.cuda()` and `torch.cuda.is_available()` checks removed
- Device inferred from `config.get('device')` and tensor devices
- Random tensors (eps_z) created directly on the correct device

### 4. Data Format
- Original: separate src tensor, lengths tensor, target tensor (seq-first format)
- Adapted: LibCity batch dict with keys `input_src`, `src_lengths`, `input_trg`,
  `output_trg`, `trg_mask` (batch-first, transposed internally)

### 5. Loss Integration
- Original DenseLoss class from util.py is integrated into `calculate_loss()`
- Supports both CE (CrossEntropyLoss) and NLL (NLLLoss) modes
- Mask handling via `trg_mask` or automatic PAD-based masking
- Latent losses (Gaussian KL + category) combined with configurable weights

### 6. Pretrained Weights
- Original encoder loads pretrained weights from hardcoded `sparse2dense.pt`
- Original latent module loads cluster centers from `init_latent.pt`
- Adapted version initializes randomly; external loading can be done via config

### 7. pack_padded_sequence
- Added `enforce_sorted=False` for robustness when batch is not pre-sorted

## Config Parameters

| Parameter | Default | Description |
|---|---|---|
| `embedding_size` | 256 | Embedding dim for grid cells and road segments |
| `hidden_size` | 256 | GRU hidden state dimension |
| `num_layers` | 2 | Encoder GRU layers |
| `de_layer` | 1 | Decoder stacked GRU layers |
| `dropout` | 0.1 | Dropout rate |
| `bidirectional` | true | Bidirectional encoder |
| `cluster_size` | 10 | Number of Gaussian mixture clusters |
| `max_length` | 300 | Max decoding length for inference |
| `criterion_name` | "CE" | Loss type: "CE" or "NLL" |
| `latent_weight` | 1.0 | Weight for Gaussian latent loss |
| `cate_weight` | 0.1 | Weight for category loss |
| `training_mode` | "train" | Operating mode: "pretrain" or "train" |
| `BOS` | 1 | Beginning-of-sequence token |
| `EOS` | 2 | End-of-sequence token |
| `PAD` | 0 | Padding token |

## Required data_feature Keys

| Key | Type | Description |
|---|---|---|
| `input_vocab_size` | int | Grid cell vocabulary size (including PAD) |
| `output_vocab_size` | int | Road segment vocabulary size (including BOS, EOS, PAD) |

## Assumptions and Limitations

1. The pretrain stage (sparse-to-dense encoder pretraining) is not included
   in this adaptation. It should be handled separately or via checkpoint loading.
2. KMeans-based cluster center initialization (`init_latent.py`) is not
   integrated. Cluster centers are randomly initialized; for better results,
   users should pretrain and then load cluster centers.
3. The model uses `DeepMMSeq2SeqDataset` as its dataset class, which should
   provide batch dicts with the expected keys.
4. Grid conversion (GPS to grid cells) is assumed to be handled by the dataset.
