# DeepMM Dual-Mode Architecture Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DeepMM Model                            │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    Input Processing                      │  │
│  │                                                           │  │
│  │  Batch Input:                                            │  │
│  │    current_loc: [batch, seq_len]                        │  │
│  │    target: [batch, seq_len] OR [batch]  ← Dual Input   │  │
│  │                                                           │  │
│  │  Automatic Detection:                                    │  │
│  │    if target.dim() == 1:                                │  │
│  │      target = target.unsqueeze(1)  # [batch] → [batch,1]│  │
│  │      mode = "next-location"                             │  │
│  │    else:                                                 │  │
│  │      mode = "map-matching"                              │  │
│  └─────────────────────────────────────────────────────────┘  │
│                            ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    Embeddings                            │  │
│  │                                                           │  │
│  │  src_embedding(current_loc) → [batch, seq_len, 256]    │  │
│  │  trg_embedding(target) → [batch, trg_len, 256]         │  │
│  │                          where trg_len = seq_len or 1   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                            ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Bidirectional LSTM Encoder                  │  │
│  │                                                           │  │
│  │  Input:  [batch, seq_len, 256]                          │  │
│  │  Output: [batch, seq_len, 512] (concatenated BiLSTM)   │  │
│  │  State:  h_t [batch, 512], c_t [batch, 512]            │  │
│  └─────────────────────────────────────────────────────────┘  │
│                            ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              LSTM Decoder with Attention                 │  │
│  │                                                           │  │
│  │  For each target position (trg_len steps):              │  │
│  │    1. Compute attention over encoder outputs            │  │
│  │    2. Decode with attended context                      │  │
│  │                                                           │  │
│  │  Output: [batch, trg_len, 512]                          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                            ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Vocabulary Projection                       │  │
│  │                                                           │  │
│  │  Linear(512 → vocab_size)                               │  │
│  │  Output: [batch, trg_len, vocab_size]                   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                            ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                   Output Processing                      │  │
│  │                                                           │  │
│  │  If mode == "next-location":                            │  │
│  │    Squeeze: [batch, 1, vocab] → [batch, vocab]         │  │
│  │    Predict: [batch, 1] → [batch]                       │  │
│  │  Else:                                                   │  │
│  │    Keep: [batch, seq_len, vocab]                        │  │
│  │    Predict: [batch, seq_len]                            │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Forward Pass Flow

### Map Matching Mode (Sequence-to-Sequence)

```
Input Batch:
┌──────────────────────────────────────┐
│ current_loc: [32, 50]                │  GPS trajectory
│ target:      [32, 50]                │  Road segments
└──────────────────────────────────────┘
            ↓
┌──────────────────────────────────────┐
│ Target Check: target.dim() = 2      │  No modification
└──────────────────────────────────────┘
            ↓
┌──────────────────────────────────────┐
│ Embeddings:                          │
│   src_emb: [32, 50, 256]            │
│   trg_emb: [32, 50, 256]            │
└──────────────────────────────────────┘
            ↓
┌──────────────────────────────────────┐
│ Encoder (BiLSTM):                   │
│   Output: [32, 50, 512]             │
│   State: h_t [32, 512]              │
└──────────────────────────────────────┘
            ↓
┌──────────────────────────────────────┐
│ Decoder (LSTM + Attention):         │
│   50 steps, each attends to encoder │
│   Output: [32, 50, 512]             │
└──────────────────────────────────────┘
            ↓
┌──────────────────────────────────────┐
│ Projection:                          │
│   Logits: [32, 50, 5000]            │
└──────────────────────────────────────┘
            ↓
┌──────────────────────────────────────┐
│ Prediction:                          │
│   argmax: [32, 50]                  │
└──────────────────────────────────────┘
            ↓
┌──────────────────────────────────────┐
│ Loss:                                │
│   Flatten: [1600, 5000] vs [1600]  │
│   CrossEntropy → scalar             │
└──────────────────────────────────────┘
```

### Next-Location Prediction Mode (Sequence-to-Single)

```
Input Batch:
┌──────────────────────────────────────┐
│ current_loc: [32, 50]                │  GPS trajectory
│ target:      [32]                    │  Next location (1D!)
└──────────────────────────────────────┘
            ↓
┌──────────────────────────────────────┐
│ Target Check: target.dim() = 1      │
│ Action: unsqueeze(1) → [32, 1]      │  Convert to sequence
└──────────────────────────────────────┘
            ↓
┌──────────────────────────────────────┐
│ Embeddings:                          │
│   src_emb: [32, 50, 256]            │
│   trg_emb: [32, 1, 256]  ← seq=1   │
└──────────────────────────────────────┘
            ↓
┌──────────────────────────────────────┐
│ Encoder (BiLSTM):                   │
│   Output: [32, 50, 512]             │
│   State: h_t [32, 512]              │
└──────────────────────────────────────┘
            ↓
┌──────────────────────────────────────┐
│ Decoder (LSTM + Attention):         │
│   1 step only, attends to encoder   │
│   Output: [32, 1, 512]              │
└──────────────────────────────────────┘
            ↓
┌──────────────────────────────────────┐
│ Projection:                          │
│   Logits: [32, 1, 5000]             │
└──────────────────────────────────────┘
            ↓
┌──────────────────────────────────────┐
│ Prediction:                          │
│   argmax: [32, 1]                   │
│   squeeze(1) → [32]  ← Restore 1D   │
└──────────────────────────────────────┘
            ↓
┌──────────────────────────────────────┐
│ Loss:                                │
│   Squeeze: [32, 5000] vs [32]       │
│   CrossEntropy → scalar             │
└──────────────────────────────────────┘
```

## Attention Mechanism

```
┌─────────────────────────────────────────────────────────┐
│                   Soft Dot Attention                     │
│                                                          │
│  Encoder Outputs:    [batch, src_len, hidden]          │
│  Decoder State:      [batch, hidden]                    │
│                                                          │
│  1. Compute Scores:                                     │
│     scores = decoder_state @ encoder_outputs.T          │
│     shape: [batch, src_len]                             │
│                                                          │
│  2. Softmax:                                            │
│     attn_weights = softmax(scores, dim=-1)              │
│     shape: [batch, src_len]                             │
│                                                          │
│  3. Weighted Sum:                                       │
│     context = attn_weights @ encoder_outputs            │
│     shape: [batch, hidden]                              │
│                                                          │
│  4. Combine:                                            │
│     output = tanh(Linear([context; decoder_state]))     │
│     shape: [batch, hidden]                              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Shape Transformation Table

| Stage | Map Matching | Next-Location | Notes |
|-------|--------------|---------------|-------|
| **Input** |
| current_loc | `[32, 50]` | `[32, 50]` | Same |
| target | `[32, 50]` | `[32]` | Different! |
| **After Detection** |
| target_internal | `[32, 50]` | `[32, 1]` | Normalized |
| **Embeddings** |
| src_emb | `[32, 50, 256]` | `[32, 50, 256]` | Same |
| trg_emb | `[32, 50, 256]` | `[32, 1, 256]` | Different length |
| **Encoder** |
| encoder_out | `[32, 50, 512]` | `[32, 50, 512]` | Same |
| encoder_state | `[32, 512]` | `[32, 512]` | Same |
| **Decoder** |
| decoder_out | `[32, 50, 512]` | `[32, 1, 512]` | Different length |
| **Projection** |
| logits | `[32, 50, 5000]` | `[32, 1, 5000]` | Different length |
| **Prediction** |
| predictions | `[32, 50]` | `[32]` | Matches input target |
| **Loss** |
| logits_flat | `[1600, 5000]` | `[32, 5000]` | Different |
| target_flat | `[1600]` | `[32]` | Different |

## Code Flow Decision Tree

```
forward(batch)
    │
    ├─ Extract target from batch
    │
    ├─ Check target.dim()
    │   │
    │   ├─ dim() == 1? (Single target)
    │   │   │
    │   │   ├─ Yes: unsqueeze(1) → [batch, 1]
    │   │   │       set is_single_target = True
    │   │   │
    │   │   └─ No:  keep as [batch, seq_len]
    │   │           set is_single_target = False
    │   │
    │   └─ Continue with normalized shape
    │
    ├─ Encode source sequence
    │
    ├─ Decode target sequence (1 or seq_len steps)
    │
    └─ Return logits [batch, trg_len, vocab]

predict(batch)
    │
    ├─ logits = forward(batch)
    │
    ├─ predictions = argmax(logits, dim=-1)
    │
    ├─ Check original target.dim()
    │   │
    │   ├─ dim() == 1?
    │   │   │
    │   │   ├─ Yes: squeeze(1) → [batch]
    │   │   │
    │   │   └─ No:  keep [batch, seq_len]
    │   │
    │   └─ Return predictions
    │
    └─ Shape matches input target

calculate_loss(batch)
    │
    ├─ logits = forward(batch)  # [batch, trg_len, vocab]
    │
    ├─ target = batch['target']
    │
    ├─ Check target.dim()
    │   │
    │   ├─ dim() == 1? (Single target)
    │   │   │
    │   │   ├─ Yes: squeeze logits → [batch, vocab]
    │   │   │       keep target as [batch]
    │   │   │
    │   │   └─ No:  flatten logits → [batch*seq_len, vocab]
    │   │           flatten target → [batch*seq_len]
    │   │
    │   └─ Both ready for CrossEntropyLoss
    │
    ├─ loss = CrossEntropyLoss(logits, target)
    │
    └─ Return scalar loss
```

## Memory Comparison

### Map Matching (seq_len=50)
```
Total Decoder Memory:
  Decoder steps: 50
  Hidden states: 50 × [batch, 512]
  Attention: 50 × [batch, src_len]
  Memory: ~High
```

### Next-Location (seq_len=1)
```
Total Decoder Memory:
  Decoder steps: 1  ← Much fewer!
  Hidden states: 1 × [batch, 512]
  Attention: 1 × [batch, src_len]
  Memory: ~Low (50× less decoder memory)
```

## Key Innovation

```
┌─────────────────────────────────────────────────────────┐
│                   Key Insight                            │
│                                                          │
│  Treat single targets as sequences of length 1          │
│                                                          │
│  Benefits:                                              │
│  ✓ Unified code path                                   │
│  ✓ No architecture changes                             │
│  ✓ Minimal performance overhead                        │
│  ✓ Automatic mode detection                            │
│  ✓ Output shapes always match input                    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

**Created**: 2026-02-06
**Purpose**: Visual guide to DeepMM dual-mode adaptation
