# GNPRSID Model Migration Documentation

## Model Overview

**Original Name:** CRQVAE (Cosine Residual Quantized VAE)
**LibCity Name:** GNPRSID (Graph-based Next POI Recommendation with Semantic ID)
**Task Type:** Trajectory Location Prediction
**Original Repository:** https://github.com/wds1996/GNPR-SID

## Source Files

| Original File | Description |
|--------------|-------------|
| `/home/wangwenrui/shk/AgentCity/repos/GNPRSID/V2/SID/CRQVAE/crqvae.py` | Main CRQVAE model class |
| `/home/wangwenrui/shk/AgentCity/repos/GNPRSID/V2/SID/CRQVAE/rq.py` | Residual Vector Quantizer |
| `/home/wangwenrui/shk/AgentCity/repos/GNPRSID/V2/SID/CRQVAE/cvq_ema.py` | Cosine Vector Quantizer with EMA |
| `/home/wangwenrui/shk/AgentCity/repos/GNPRSID/V2/SID/CRQVAE/mlp.py` | MLP layers and utility functions |
| `/home/wangwenrui/shk/AgentCity/repos/GNPRSID/V2/POIembedding/POI2emb.py` | POI embedding generation (reference) |

## Created/Modified Files

| File Path | Action |
|-----------|--------|
| `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/GNPRSID.py` | Created |
| `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/GNPRSID.json` | Created |
| `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py` | Modified (added import) |
| `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json` | Modified (added model entry) |

## Model Architecture

```
Input (POI Embeddings)
       |
       v
+------------------+
|   Encoder MLP    |
| [in -> 512 ->    |
|  256 -> 128 -> 64]|
+------------------+
       |
       v
+------------------+
| Residual Vector  |
|   Quantizer      |
| (3 layers x 64   |
|    codebook)     |
+------------------+
       |
       v
+------------------+
|   Decoder MLP    |
| [64 -> 128 ->    |
|  256 -> 512 -> in]|
+------------------+
       |
       v
Output (Reconstructed Embedding + Semantic IDs)
```

## Key Adaptations

### 1. Class Inheritance
- Changed from `nn.Module` to `AbstractModel` (LibCity base class)
- Added `config` and `data_feature` parameters to constructor

### 2. Data Handling
- Implemented `_get_poi_embeddings()` to extract POI embeddings from batch
- Supports three modes:
  - Pre-computed embeddings in `data_feature['poi_embeddings']`
  - External embeddings in `batch['poi_embeddings']`
  - Learnable embeddings using `nn.Embedding`

### 3. Required Methods
- `predict(batch)`: Returns semantic IDs for POIs
- `calculate_loss(batch)`: Returns total loss (reconstruction + quantization)
- `forward(batch)`: Returns reconstructed embeddings, quantization loss, and codes

### 4. Module Integration
- Combined all sub-modules into a single file:
  - `MLPLayers`: Multi-layer perceptron
  - `CosineVectorQuantizer`: Single-layer cosine similarity VQ with EMA
  - `ResidualVectorQuantizer`: Multi-layer residual quantization
  - `GNPRSID`: Main model class

## Configuration Parameters

### Model Architecture
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | int | 79 | POI embedding dimension |
| `e_dim` | int | 64 | Latent/codebook dimension |
| `num_emb_list` | list | [64, 64, 64] | Codebook sizes for each RQ layer |
| `encoder_layers` | list | [512, 256, 128] | Hidden layer sizes |
| `dropout_prob` | float | 0.0 | Dropout probability |
| `use_bn` | bool | false | Use batch normalization |

### Loss Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_type` | str | "mse" | Reconstruction loss type ("mse" or "l1") |
| `quant_loss_weight` | float | 0.25 | Weight for quantization loss |
| `beta` | float | 0.25 | Commitment loss weight |

### Quantization Settings
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kmeans_init` | bool | false | Use K-means initialization |
| `kmeans_iters` | int | 100 | K-means iterations |
| `sk_epsilons` | list | [0.05, 0.05, 0.05] | Sinkhorn epsilon per layer |
| `sk_iters` | int | 100 | Sinkhorn iterations |
| `use_linear` | int | 0 | Use linear projection for codebook |
| `use_sk` | bool | true | Use Sinkhorn during training |

### Data Feature Mappings
| data_feature Key | Usage |
|------------------|-------|
| `loc_size` | Number of POI locations (for learnable embeddings) |
| `poi_embeddings` | Pre-computed POI embeddings (optional) |

## Usage Example

```python
from libcity.model.trajectory_loc_prediction import GNPRSID

config = {
    'device': 'cuda',
    'input_dim': 79,
    'e_dim': 64,
    'num_emb_list': [64, 64, 64],
    'encoder_layers': [512, 256, 128],
    'dropout_prob': 0.0,
    'loss_type': 'mse',
    'quant_loss_weight': 0.25,
    'beta': 0.25
}

data_feature = {
    'loc_size': 10000
}

model = GNPRSID(config, data_feature)

# Training
batch = {'current_loc': torch.randint(0, 10000, (32, 10))}
loss = model.calculate_loss(batch)
loss.backward()

# Inference
semantic_ids = model.predict(batch)  # [batch_size, seq_len, num_quantizers]
```

## Differences from Original Implementation

1. **Modularization**: All sub-modules are combined into a single file for easier deployment.

2. **POI Embedding Handling**: Added flexible POI embedding options (pre-computed, external, or learnable).

3. **Batch Format**: Adapted to LibCity's batch dictionary format instead of direct tensor input.

4. **Logging**: Added logging using Python's logging module.

5. **Skipped Components**:
   - LLM fine-tuning component (focus on SID generation only)
   - POI embedding generation script (POI2emb.py) - kept as reference

## Limitations and Assumptions

1. **POI Embeddings**: The model assumes POI embeddings are either:
   - Pre-computed and provided in data_feature
   - Provided in each batch
   - Learned from scratch using location indices

2. **Batch Keys**: Expects `current_loc` or `loc` keys in batch for POI indices.

3. **Evaluation**: The model generates semantic IDs, which may require custom evaluation metrics for semantic ID quality.

## Testing

To test the model import:
```bash
cd /home/wangwenrui/shk/AgentCity/Bigscity-LibCity
python -c "from libcity.model.trajectory_loc_prediction import GNPRSID; print('Import successful')"
```
