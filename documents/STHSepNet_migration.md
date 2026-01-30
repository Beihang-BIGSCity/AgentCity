# STHSepNet Model Migration Documentation

## Overview

This document describes the migration of the STHSepNet (Spatio-Temporal Hypergraph Separation Network) model to the LibCity framework.

## Source Information

- **Original Repository Path**: `/home/wangwenrui/shk/AgentCity/repos/STHSepNet`
- **Main Model File**: `/home/wangwenrui/shk/AgentCity/repos/STHSepNet/models/ST_SepNet.py`
- **Original Model Class**: `Model` (inherits from nn.Module)
- **Task Type**: Traffic Flow/Demand Forecasting

## Target Location

- **LibCity Model Path**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_flow_prediction/STHSepNet.py`
- **Model Class Name**: `STHSepNet`
- **Base Class**: `AbstractTrafficStateModel`

## Architecture Overview

### Core Components

1. **Temporal Module**:
   - Original: LLM-based (BERT, GPT-2, LLAMA variants) with LoRA layers
   - Adapted: Lightweight Transformer-based encoder (no external LLM dependencies)

2. **Spatial Module**:
   - Adaptive hypergraph neural networks (preserved from original)
   - Graph constructor for first-order interactions
   - Hypergraph constructor for high-order interactions

3. **Fusion Mechanisms**:
   - Learnable gating parameters (alpha, beta, gamma, theta)
   - Multiple fusion gate options: adaptive, attentiongate, lstmgate, hyperstgnn

### Included Helper Classes

All necessary helper classes are included in the single file:

- `GraphConstructor` - Adaptive graph construction
- `HypergraphConstructor` - Adaptive hypergraph construction
- `HypergraphConvolution` - Hypergraph convolution layer
- `HypergraphAttention` - Hypergraph attention layer
- `HypergraphSAGE` - Hypergraph SAGE aggregation
- `NConv`, `Linear`, `MixProp` - Graph convolution utilities
- `DilatedInception` - Temporal convolution module
- `STHGNNLayerNorm` - Custom layer normalization
- `AttentionFusion`, `LSTMGate` - Fusion mechanisms
- `STHGNN` - Full spatio-temporal hypergraph module
- `LightweightTemporalEncoder` - Alternative to LLM encoder

## Key Adaptations

### 1. Base Class Change
```python
# Original
class Model(nn.Module):

# Adapted
class STHSepNet(AbstractTrafficStateModel):
```

### 2. Constructor Signature
```python
# Original
def __init__(self, configs, patch_len=16, stride=8):

# Adapted
def __init__(self, config, data_feature):
```

### 3. Data Format
```python
# Original: Custom CSV loading with specific tensor shapes
x_enc = torch.squeeze(x_enc)  # (B, T, N)

# Adapted: LibCity batch format
x = batch['X']  # (B, T, N, F)
```

### 4. Device Handling
```python
# Original: Hardcoded CUDA
.to(device='cuda:0', dtype=torch.float16)

# Adapted: Config-based
self.device = config.get('device', torch.device('cpu'))
```

### 5. Adjacency Matrix
```python
# Original: CSV file loading
self.adj = torch.tensor(pd.read_csv(...))

# Adapted: From data_feature
adj_mx = data_feature.get('adj_mx', None)
```

### 6. LLM Removal
The original model heavily depends on HuggingFace transformers (BERT, GPT-2, LLAMA). The adapted version provides a lightweight Transformer-based encoder as an alternative, removing external LLM dependencies.

### 7. Required Methods Implementation
```python
def forward(self, batch):
    """Process batch and return predictions."""

def predict(self, batch):
    """Return predictions (calls forward)."""

def calculate_loss(self, batch):
    """Calculate training loss using LibCity's loss functions."""
```

## Configuration Parameters

### Data Features (from data_feature)
| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_nodes` | Number of nodes/sensors | 1 |
| `feature_dim` | Input feature dimension | 1 |
| `output_dim` | Output feature dimension | 1 |
| `adj_mx` | Adjacency matrix | None |
| `scaler` | Data scaler for normalization | None |

### Model Config Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_window` | Input sequence length | 12 |
| `output_window` | Output prediction length | 12 |
| `hidden_dim` | Hidden dimension size | 64 |
| `d_model` | Model dimension | 32 |
| `d_ff` | Feed-forward dimension | 32 |
| `n_heads` | Number of attention heads | 4 |
| `num_layers` | Number of encoder layers | 2 |
| `dropout` | Dropout rate | 0.1 |
| `gcn_true` | Enable GCN | True |
| `hgcn_true` | Enable Hypergraph GCN | True |
| `hgat_true` | Enable Hypergraph Attention | False |
| `buildA_true` | Build adaptive graph | True |
| `buildH_true` | Build adaptive hypergraph | True |
| `adaptive_hyperhgnn` | Hypergraph type: 'hgcn', 'hgat', 'hsage' | 'hgcn' |
| `temporl_true` | Enable temporal gating | True |
| `static` | Use static adjacency | False |
| `scale_hyperedges` | Number of KNN neighbors | 10 |
| `alpha` | Fusion parameter | 0.1 |
| `beta` | Fusion parameter | 0.3 |
| `gamma` | Fusion parameter | 0.5 |
| `theta` | Fusion parameter | 0.2 |
| `fusion_gate` | Fusion type: 'adaptive', 'hyperstgnn', 'attentiongate', 'lstmgate' | 'adaptive' |

## Usage Example

```python
# In LibCity config file or command line
config = {
    'model': 'STHSepNet',
    'dataset': 'METR_LA',
    'input_window': 12,
    'output_window': 12,
    'hidden_dim': 64,
    'gcn_true': True,
    'hgcn_true': True,
    'adaptive_hyperhgnn': 'hgcn',
    'fusion_gate': 'adaptive'
}
```

## Dependencies

### Required
- torch
- numpy
- logging (standard library)

### Optional (for advanced hypergraph construction)
- scikit-learn (for KMeans, NearestNeighbors)

If scikit-learn is not available, the hypergraph constructor falls back to a similarity-based approach.

## Limitations

1. **No LLM Support**: The original model's LLM-based temporal encoding is replaced with a lightweight Transformer encoder. For LLM integration, additional work would be needed.

2. **Memory Efficiency**: The hypergraph construction may be memory-intensive for very large graphs due to KNN computation.

3. **Batch Hypergraph**: Hypergraphs are constructed per-forward pass which may slow training. Consider caching for production use.

## Files Modified

1. **Created**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_flow_prediction/STHSepNet.py`

2. **Modified**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_flow_prediction/__init__.py`
   - Added import statement
   - Added to __all__ list

## Testing

To test the model:
```bash
cd /home/wangwenrui/shk/AgentCity/Bigscity-LibCity
python run_model.py --task traffic_state_pred --model STHSepNet --dataset METR_LA
```

## Original Files Reference

| Original File | Adapted Location |
|--------------|------------------|
| `models/ST_SepNet.py` | Integrated into STHSepNet.py |
| `layer/STHGNN.py` | STHGNN class in STHSepNet.py |
| `layer/HyperGNN.py` | HypergraphConvolution, HypergraphAttention, HypergraphSAGE |
| `layer/FusionGate.py` | AttentionFusion, LSTMGate |
| `layer/GraphGCN.py` | Part of GraphConstructor |
| `layer/Embed.py` | Simplified in LightweightTemporalEncoder |
| `layer/AdaGNN.py` | GraphConstructor, HypergraphConstructor |

## Author

Adapted for LibCity framework by Model Migration Agent.
