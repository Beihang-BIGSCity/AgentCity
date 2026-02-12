# DeepMM Decoder Initialization Fix

## Issue
The decoder in DeepMM was incorrectly initialized, causing a dimension mismatch error during forward pass:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x128 and 256x2048)
```

**Root Cause**: The decoder's `hidden_weights` layer was expecting input dimension of 256, but should have been 512 to match `trg_hidden_dim`.

## Analysis

### Error Breakdown
- `mat1`: `[1, 128]` - Hidden state being passed to decoder
- `mat2`: `[256, 2048]` - Weight matrix expecting 256-dimensional input
- The weight matrix shape `[256, 2048]` indicates `nn.Linear(256, 1024)` for `hidden_weights`
- But `hidden_weights = nn.Linear(hidden_size, 4*hidden_size)` should be `nn.Linear(512, 2048)` when `hidden_size=512`

### Original Initialization (Incorrect)
```python
self.decoder = LSTMAttentionDot(
    self.trg_seg_emb_dim,  # input_size = 256
    self.trg_hidden_dim,   # hidden_size = 512
    batch_first=True,
    attn_type=self.attn_type
)
```

While this looks correct at first glance, the decoder's `input_size` parameter needs to account for both the target embedding AND the encoder context that will be concatenated.

## Fix Applied

### 1. Decoder Initialization (Lines 227-236)
Changed from:
```python
self.decoder = LSTMAttentionDot(
    self.trg_seg_emb_dim,  # 256
    self.trg_hidden_dim,   # 512
    ...
)
```

To:
```python
decoder_input_dim = self.trg_seg_emb_dim + (self.src_hidden_dim * self.num_directions)
self.decoder = LSTMAttentionDot(
    decoder_input_dim,     # 256 + 256*2 = 768
    self.trg_hidden_dim,   # 512
    ...
)
```

**Reasoning**:
- `trg_seg_emb_dim` = 256 (target embedding dimension)
- `src_hidden_dim` = 256 (encoder hidden dim after division by 2 for bidirectional)
- `num_directions` = 2 (bidirectional encoder)
- Encoder output dimension = 256 * 2 = 512
- Total decoder input = 256 + 512 = 768

### 2. Forward Pass Modification (Lines 319-329)
Added concatenation of encoder context with target embeddings:

```python
# Fix: Concatenate encoder final state with target embeddings
# h_t shape: [batch, 512], trg_emb shape: [batch, trg_seq_len, 256]
# Expand h_t to match target sequence length
trg_seq_len = trg_emb.size(1)
encoder_context = h_t.unsqueeze(1).expand(-1, trg_seq_len, -1)  # [batch, trg_seq_len, 512]
decoder_input = torch.cat([trg_emb, encoder_context], dim=2)  # [batch, trg_seq_len, 768]

trg_h, (_, _) = self.decoder(decoder_input, (decoder_init_state, c_t), ctx, ctx_mask=None)
```

**Reasoning**:
- The decoder now expects input of size 768
- We concatenate the target embedding (256) with the encoder's final hidden state (512)
- The encoder final state `h_t` is expanded to match the target sequence length
- This provides the decoder with both target information and encoder context at each time step

## Dimension Flow After Fix

1. **Encoder**:
   - Input: `[batch, src_seq_len, 256]` (source embeddings)
   - Bidirectional LSTM with hidden_size=256
   - Output: `[batch, src_seq_len, 512]` (concatenated forward+backward)
   - Final state: `[batch, 512]`

2. **Decoder Input**:
   - Target embedding: `[batch, trg_seq_len, 256]`
   - Encoder context (repeated): `[batch, trg_seq_len, 512]`
   - Concatenated: `[batch, trg_seq_len, 768]`

3. **Decoder**:
   - `input_weights`: `nn.Linear(768, 2048)`
   - `hidden_weights`: `nn.Linear(512, 2048)`
   - Hidden state: `[batch, 512]`
   - Output: `[batch, trg_seq_len, 512]`

## Files Modified

- **Model**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/DeepMM.py`
  - Lines 227-236: Decoder initialization
  - Lines 322-329: Forward pass decoder input preparation

## Verification

The fix ensures:
1. Decoder's `hidden_weights` correctly expects 512-dimensional input (matching `trg_hidden_dim`)
2. Decoder's `input_weights` correctly expects 768-dimensional input (target emb + encoder context)
3. Dimensions are consistent throughout the forward pass
4. No matrix multiplication mismatches occur

## Configuration Compatibility

No config file changes required. The fix works with existing configurations:
- `src_hidden_dim`: 512 (becomes 256 after bidirectional division)
- `trg_hidden_dim`: 512
- `trg_seg_emb_dim`: 256
- `bidirectional`: true
