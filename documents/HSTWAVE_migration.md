# HSTWAVE Model Migration Documentation

## Overview

This document describes the migration of the HSTWAVE (Hierarchical Spatial-Temporal Weaving Attention and Heterogeneous Graph Network) model from the original PyTorch Lightning implementation to the LibCity framework.

## Original Model Information

- **Source Repository:** `/home/wangwenrui/shk/AgentCity/repos/HSTWAVE`
- **Main Model File:** `/home/wangwenrui/shk/AgentCity/repos/HSTWAVE/model.py`
- **Original Framework:** PyTorch Lightning
- **Task:** Traffic Flow Prediction

## Adapted Model Location

- **Model File:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_flow_prediction/HSTWAVE.py`
- **Config File:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traffic_state_pred/HSTWAVE.json`

## Key Components Migrated

### 1. CausalConv2d
- Causal 2D convolution with left-side padding for temporal causality
- Ensures the model only uses past information for predictions

### 2. Align
- Channel dimension alignment module
- Uses 1x1 convolution for reduction and zero-padding for expansion

### 3. GTU (Gated Temporal Unit)
- Gated temporal convolution for multi-scale temporal modeling
- Uses tanh/sigmoid gating mechanism

### 4. MSWT (Multi-Scale Weaving Transformer)
- Multi-scale temporal feature extraction with transformer encoder
- Combines multiple GTUs with different kernel sizes
- Cross-scale attention mechanism

### 5. CHGANSimplified
- Simplified version of Coupled Heterogeneous Graph Attention Network
- Adapted to work with LibCity's standard adjacency matrix format
- Uses multi-head attention with learnable node type embeddings

### 6. MSDTHGTEncoderSimplified
- Main encoder combining MSWT and CHGAN
- Performs joint spatial-temporal feature learning

### 7. SequenceAugmentor
- Data augmentation for contrastive learning
- Supports flip, mask, shift, and noise addition operations

### 8. HSTWAVE (Main Class)
- Inherits from `AbstractTrafficStateModel`
- Implements required LibCity methods: `__init__`, `forward`, `predict`, `calculate_loss`

## Key Adaptations

### 1. PyTorch Lightning to Standard PyTorch

**Original:**
```python
class HSTWAVE(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        ...
    def validation_step(self, batch, batch_idx):
        ...
```

**Adapted:**
```python
class HSTWAVE(AbstractTrafficStateModel):
    def forward(self, batch):
        ...
    def predict(self, batch):
        ...
    def calculate_loss(self, batch):
        ...
```

### 2. Heterogeneous Graph to Standard Format

**Original:**
The original model used PyTorch Geometric's heterogeneous graph with two node types:
- `hw` (highway nodes)
- `para` (parallel road nodes)

**Adapted:**
- Simplified to work with LibCity's standard adjacency matrix
- Node type information is simulated through learnable embeddings
- Edge types are handled through a simplified attention mechanism

### 3. Data Format Transformation

**Original:**
```python
# Input from PyG DataLoader
hgs, hw_x, hw_y, para_x = batch
x_dict = hgs.x_dict  # {'hw': tensor, 'para': tensor}
```

**Adapted:**
```python
# Input from LibCity batch dict
x = batch['X']  # (B, T, N, F)
y = batch['y']  # (B, T_out, N, F)
```

### 4. Scaler Integration

**Original:**
```python
self.scalar = scaler  # Custom scaler passed to constructor
pre, label = self._inverse_transform([pre, hw_y[:,:,:,0]], self.scalar)
```

**Adapted:**
```python
self._scaler = self.data_feature.get('scaler')  # LibCity's scaler
y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
```

### 5. Loss Function

**Original:**
```python
loss1 = torch.mean(torch.abs(pre - label))  # MAE
loss2 = self.SimCLRLoss(out2, out3, hw_x.shape[0])  # Contrastive
loss = 0.8*loss1 + 0.2*loss2
```

**Adapted:**
```python
mae_loss = loss.masked_mae_torch(y_predicted, y_true, 0)  # LibCity's masked MAE
contrastive_loss = self._simclr_loss(out1, out2, batch_size)
total_loss = (1 - self.contrastive_weight) * mae_loss + self.contrastive_weight * contrastive_loss
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 64 | Hidden dimension for encoder layers |
| `num_layers` | 3 | Number of encoder layers |
| `num_scales` | 3 | Number of temporal scales in MSWT |
| `num_heads` | 4 | Number of attention heads in CHGAN |
| `dropout` | 0.3 | Dropout rate |
| `use_contrastive` | true | Enable contrastive learning |
| `contrastive_weight` | 0.2 | Weight for contrastive loss |
| `contrastive_temp` | 500 | Temperature for SimCLR loss |
| `noise_std` | 0.05 | Standard deviation for augmentation noise |
| `input_window` | 12 | Input sequence length |
| `output_window` | 12 | Output prediction length |

## Usage Example

```python
# Run with LibCity
python run_model.py --task traffic_state_pred --model HSTWAVE --dataset METR_LA
```

## Limitations and Assumptions

1. **Graph Structure:** The original heterogeneous graph with separate highway and parallel road nodes is simplified to a homogeneous graph. Node type information is approximated through learnable embeddings.

2. **Large Label Filtering:** The original model had special handling for "large label" nodes. This feature is not implemented in the adapted version.

3. **Training Mode Selection:** The original model had different augmentation strategies based on `trainmode`. The adapted version uses a simplified approach with configurable noise standard deviation.

4. **External Dataset Loading:** The original model loaded additional training data during forward pass for certain training modes. This is removed as LibCity handles all data loading.

## Dependencies

- PyTorch >= 1.10
- LibCity framework
- NumPy

## Original Paper Reference

Please cite the original HSTWAVE paper if you use this implementation:

```
@article{hstwave2024,
  title={HSTWAVE: Hierarchical Spatial-Temporal Weaving Attention and Heterogeneous Graph Network for Traffic Flow Prediction},
  author={...},
  journal={...},
  year={2024}
}
```
