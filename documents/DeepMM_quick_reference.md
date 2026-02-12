# DeepMM Quick Reference Guide

## Model Information
- **Task**: Trajectory Location Prediction
- **File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/DeepMM.py`
- **Base Class**: `AbstractModel`
- **Architecture**: Seq2Seq with Bidirectional LSTM Encoder + Attention-based LSTM Decoder

## Fixed Issues (2026-02-06)

### 1. Batch Access Methods
- Replaced `.get()` calls with direct dictionary access and try-except blocks
- Affects: `forward()` and `calculate_loss()` methods

### 2. Vocabulary Configuration
- Changed from road segment vocabulary to location vocabulary
- Both source and target now use `loc_size` and `loc_pad`

## Required Configuration Parameters

### In `data_feature` dict:
```python
{
    'loc_size': 10000,      # Vocabulary size for location IDs
    'loc_pad': 0,           # Padding index for locations
}
```

### In `config` dict:
```python
{
    'dim_loc_src': 256,          # Source location embedding dimension
    'dim_seg_trg': 256,          # Target segment embedding dimension
    'src_hidden_dim': 512,       # Encoder hidden dimension
    'trg_hidden_dim': 512,       # Decoder hidden dimension
    'bidirectional': True,       # Use bidirectional encoder
    'n_layers_src': 2,           # Number of encoder layers
    'dropout': 0.5,              # Dropout rate
    'rnn_type': 'LSTM',          # RNN type (LSTM or GRU)
    'attn_type': 'dot',          # Attention type (dot, general, or mlp)
    'device': torch.device('cuda')  # Device
}
```

## Batch Dictionary Format

The model expects a batch dictionary with:

```python
batch = {
    'current_loc': torch.LongTensor,  # Shape: [batch_size, seq_len]
                                      # GPS location indices (source sequence)

    'target': torch.LongTensor,       # Shape: [batch_size, seq_len]
                                      # Target location indices for prediction
}
```

## Usage Example

```python
from libcity.model.trajectory_loc_prediction import DeepMM

# Initialize model
config = {
    'dim_loc_src': 256,
    'dim_seg_trg': 256,
    'src_hidden_dim': 512,
    'trg_hidden_dim': 512,
    'bidirectional': True,
    'n_layers_src': 2,
    'dropout': 0.5,
    'rnn_type': 'LSTM',
    'attn_type': 'dot',
    'device': torch.device('cuda')
}

data_feature = {
    'loc_size': 10000,
    'loc_pad': 0
}

model = DeepMM(config, data_feature)

# Forward pass
batch = {
    'current_loc': torch.randint(0, 10000, (32, 50)),  # [batch=32, seq_len=50]
    'target': torch.randint(0, 10000, (32, 50))        # [batch=32, seq_len=50]
}

# Get predictions
predictions = model.predict(batch)  # Returns: [batch, seq_len]

# Calculate loss
loss = model.calculate_loss(batch)  # Returns: scalar loss tensor
```

## Model Architecture

1. **Source Embedding Layer**
   - Input: Location indices [batch, seq_len]
   - Output: Embeddings [batch, seq_len, dim_loc_src]

2. **Bidirectional LSTM Encoder**
   - Input: Source embeddings
   - Output: Encoder hidden states [batch, seq_len, src_hidden_dim * 2]

3. **Attention-based LSTM Decoder**
   - Input: Target embeddings + encoder context
   - Attention: Dot-product / General / MLP attention
   - Output: Decoder hidden states [batch, seq_len, trg_hidden_dim]

4. **Output Projection**
   - Input: Decoder hidden states
   - Output: Logits over vocabulary [batch, seq_len, vocab_size]

## Key Features

- **Bidirectional Encoding**: Captures both past and future context
- **Attention Mechanism**: Allows decoder to focus on relevant encoder states
- **Flexible Attention**: Supports dot, general, and MLP attention types
- **Padding Masking**: Automatically ignores padding tokens in loss calculation

## Notes

- The model uses teacher forcing during training (requires target sequence)
- During inference, predictions are made via argmax over logits
- Padding tokens are masked in loss calculation using weighted cross-entropy
- Both source and target use the same location vocabulary for trajectory prediction

## Related Files

- Model: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/DeepMM.py`
- Config: Should be defined in LibCity config files
- Documentation: `/home/wangwenrui/shk/AgentCity/documents/DeepMM_fixes_summary.md`
