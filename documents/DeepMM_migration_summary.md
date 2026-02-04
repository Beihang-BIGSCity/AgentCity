# DeepMM Migration Summary

## Overview

DeepMM is a deep learning-based map matching model that uses a sequence-to-sequence architecture with attention. The model takes GPS trajectory data (location IDs with optional time information) as input and outputs a sequence of road segments.

## Source Files

- **Original Model**: `/home/wangwenrui/shk/AgentCity/repos/DeepMM/DeepMM/model.py`
  - `Seq2SeqAttention` class: lines 825-1034
  - `LSTMAttentionDot` class: lines 391-443
  - `SoftDotAttention` class: lines 303-388

## Target Files

- **Adapted Model**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/DeepMM.py`
- **Configuration**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/map_matching/DeepMM.json`
- **Registry**: Updated `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/map_matching/__init__.py`

## Architecture

### Encoder
- Bidirectional LSTM (default)
- 2 layers, 512 hidden units (256 per direction)
- Supports LSTM or GRU via `rnn_type` config

### Decoder
- Custom LSTM with soft attention at each timestep
- 512 hidden units
- Attention types: `dot`, `general`, `mlp`

### Embeddings
- Source location embedding: 256 dimensions
- Target segment embedding: 256 dimensions
- Optional time embeddings for temporal encoding

## Key Adaptations

### 1. PyTorch Modernization
- Removed deprecated `torch.autograd.Variable` usage
- Replaced `F.sigmoid(x)` with `torch.sigmoid(x)`
- Replaced `F.tanh(x)` with `torch.tanh(x)`
- Removed hardcoded `.cuda()` calls for device-agnostic operation

### 2. LibCity Interface
- Inherits from `AbstractModel` (neural network model, not tradition model)
- Uses `config.get()` pattern for hyperparameters with defaults
- Extracts vocabulary sizes and padding tokens from `data_feature`
- Implements required methods: `forward()`, `predict()`, `calculate_loss()`

### 3. Device Handling
- Device obtained from config: `self.device = config.get('device', torch.device('cpu'))`
- Tensors created with explicit device parameter
- No hardcoded CUDA calls

### 4. Loss Calculation
- Uses `nn.CrossEntropyLoss` with weight mask for padding tokens
- Supports both `output_trg` and `target` keys for ground truth sequences

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `src_loc_emb_dim` | 256 | Source location embedding dimension |
| `src_tim_emb_dim` | 64 | Source time embedding dimension |
| `trg_seg_emb_dim` | 256 | Target segment embedding dimension |
| `src_hidden_dim` | 512 | Encoder hidden dimension |
| `trg_hidden_dim` | 512 | Decoder hidden dimension |
| `bidirectional` | true | Use bidirectional encoder |
| `nlayers_src` | 2 | Number of encoder layers |
| `dropout` | 0.5 | Dropout rate |
| `time_encoding` | "NoEncoding" | Time encoding: NoEncoding, OneEncoding, TwoEncoding |
| `rnn_type` | "LSTM" | RNN type: LSTM or GRU |
| `attn_type` | "dot" | Attention type: dot, general, mlp |
| `learning_rate` | 0.001 | Learning rate |
| `max_epoch` | 100 | Maximum training epochs |
| `clip` | 5.0 | Gradient clipping threshold |

## Required Data Features

| Feature | Description |
|---------|-------------|
| `src_loc_vocab_size` | Number of GPS location tokens |
| `trg_seg_vocab_size` | Number of road segment tokens |
| `pad_token_src_loc` | Padding token index for source |
| `pad_token_trg` | Padding token index for target |
| `src_tim_vocab_size` | Time vocabulary size (if using time encoding) |
| `pad_token_src_tim1` | Time padding token (if using OneEncoding) |
| `pad_token_src_tim2` | Time padding tokens (if using TwoEncoding) |

## Expected Batch Format

```python
batch = {
    'input_src': torch.LongTensor,  # (batch, src_len) - GPS locations
    'input_trg': torch.LongTensor,  # (batch, trg_len) - Target segments (teacher forcing)
    'output_trg': torch.LongTensor, # (batch, trg_len) - Ground truth (alternative: 'target')
    'input_time': torch.LongTensor, # Optional: time features for time encoding
}
```

## Methods

### `forward(batch)`
Performs the full encoder-decoder forward pass with teacher forcing.
Returns logits over target vocabulary: `(batch, trg_len, vocab_size)`

### `predict(batch)`
Generates predictions by taking argmax of forward pass output.
Returns predicted token indices: `(batch, seq_len)`

### `calculate_loss(batch)`
Computes CrossEntropyLoss between predictions and targets.
Ignores padding tokens in loss computation.
Supports both `output_trg` and `target` keys.

### `greedy_decode(batch, max_len=100, sos_idx=0, eos_idx=2)`
Inference-time greedy decoding without teacher forcing.
Returns generated sequences: `(batch, decoded_len)`

## Task Configuration

The model is registered under `map_matching` task in `task_config.json`:
- **Dataset class**: `MapMatchingDataset`
- **Executor**: `DeepMapMatchingExecutor`
- **Evaluator**: `MapMatchingEvaluator`

## Assumptions and Limitations

1. **Teacher Forcing**: Training uses teacher forcing (ground truth target as decoder input)
2. **Special Tokens**: Assumes SOS=0, EOS=2, PAD=1 for greedy decoding (configurable)
3. **Time Encoding**: Complex time encoding (TwoEncoding) requires properly structured data features
4. **Bidirectional**: Hidden dimension is split between forward and backward directions

## Usage Example

```python
from libcity.model.map_matching import DeepMM

config = {
    'device': torch.device('cuda'),
    'src_loc_emb_dim': 256,
    'src_hidden_dim': 512,
    'attn_type': 'dot'
}

data_feature = {
    'src_loc_vocab_size': 10000,
    'trg_seg_vocab_size': 5000,
    'pad_token_src_loc': 1,
    'pad_token_trg': 1
}

model = DeepMM(config, data_feature)
```

## Notes

- DeepMM is the first deep learning-based map matching model in LibCity's map_matching module
- Previous map_matching models (HMMM, STMatching, etc.) are tradition/rule-based models using `AbstractTraditionModel`
- This model uses `AbstractModel` as it requires gradient-based training
- The model has been updated on 2026-02-03 to include `greedy_decode` method for inference
