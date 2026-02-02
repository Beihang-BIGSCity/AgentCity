# Config Migration: GNPRSID

## Summary

GNPRSID (Generative Next POI Recommendation with Semantic ID) configuration has been successfully created and integrated into LibCity. The model uses a Cosine Residual Quantized Variational Autoencoder (CRQVAE) to learn semantic IDs for POIs combined with a Transformer-based sequence model for next POI prediction.

---

## Task Configuration

### task_config.json
- **Status**: Already registered
- **Task type**: `traj_loc_pred` (trajectory location prediction)
- **Location**: Line 23 in allowed_model list
- **Dataset class**: TrajectoryDataset
- **Executor**: TrajLocPredExecutor
- **Evaluator**: TrajLocPredEvaluator
- **Trajectory encoder**: StandardTrajectoryEncoder

---

## Model Configuration

### File Location
`/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/GNPRSID.json`

### Configuration Parameters

#### POI Embeddings
- **loc_emb_size**: 128 (POI location embedding dimension)
- **uid_emb_size**: 64 (User embedding dimension)

#### CRQVAE Architecture
- **encoder_layers**: [512, 256, 128] (MLP encoder hidden layers)
- **e_dim**: 64 (Latent embedding dimension for quantization)
- **num_codebooks**: 64 (Number of codebook entries per RQ level)
- **num_rq_layers**: 3 (Number of residual quantization layers)
- **dropout_prob**: 0.1 (Dropout probability in MLP layers)
- **use_bn**: true (Use batch normalization in MLP)
- **loss_type**: "mse" (Reconstruction loss type: MSE or L1)

#### Vector Quantization Parameters
- **quant_loss_weight**: 0.5 (Weight for quantization loss)
- **beta**: 0.25 (Commitment cost for VQ)
- **kmeans_init**: true (Initialize codebooks with k-means)
- **kmeans_iters**: 100 (K-means iterations for initialization)
- **sk_epsilon**: 0.1 (Sinkhorn epsilon for optimal transport)
- **sk_iters**: 50 (Sinkhorn iterations)
- **use_ema**: true (Use exponential moving average for codebook updates)
- **ema_decay**: 0.99 (EMA decay rate - updated to match paper)
- **use_linear**: 1 (Use linear projection for codebook)

#### Loss Weights
- **pred_loss_weight**: 1.0 (Weight for prediction loss)
- **recon_loss_weight**: 0.1 (Weight for reconstruction loss)
- **quant_loss_weight**: 0.5 (Weight for quantization loss)

#### Transformer Sequence Encoder
- **transformer_nhead**: 4 (Number of attention heads)
- **transformer_nhid**: 512 (Feedforward dimension - updated to match paper)
- **transformer_nlayers**: 2 (Number of transformer encoder layers)
- **transformer_dropout**: 0.1 (Dropout in transformer)

#### Training Parameters
- **batch_size**: 128
- **learning_rate**: 0.001 (1e-3)
- **max_epoch**: 100
- **L2**: 0.0001 (Weight decay / L2 regularization)
- **optimizer**: "adamw" (AdamW optimizer)
- **lr_step**: 20 (Learning rate decay step)
- **lr_decay**: 0.9 (Learning rate decay factor)

---

## Model Registration

### __init__.py
- **Status**: Already registered
- **Import statement**: Line 25
- **Export list**: Line 51
- **Location**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py`

---

## Compatible Datasets

### Primary Datasets (from task_config.json)
The model is compatible with all trajectory location prediction datasets:

1. **foursquare_tky** - Foursquare Tokyo check-in data
2. **foursquare_nyc** - Foursquare New York City check-in data
3. **gowalla** - Gowalla location-based social network data
4. **foursquare_serm** - Foursquare SERM dataset
5. **Proto** - Prototype trajectory dataset

### Dataset Requirements
Based on the model implementation, datasets must provide:
- **uid**: User IDs
- **current_loc**: Location sequence (trajectory)
- **target**: Target POI for next location prediction
- **loc_size**: Number of unique POI locations (vocabulary size)
- **uid_size**: Number of unique users
- **loc_pad**: Padding index for locations

All standard LibCity trajectory datasets using `TrajectoryDataset` class are compatible.

---

## Configuration Updates Made

1. **ema_decay**: Updated from 0.95 to 0.99 to match paper specifications
2. **transformer_nhid**: Updated from 256 to 512 to match paper's feedforward dimension

---

## Model Architecture Details

### Key Components

1. **POI Embedding Layer**
   - Converts location IDs to dense embeddings
   - Dimension: loc_emb_size (128)

2. **CRQVAE (Cosine Residual Quantized VAE)**
   - Encoder: MLP [128 -> 512 -> 256 -> 128 -> 64]
   - Residual Vector Quantizer: 3 layers with 64 codebooks each
   - Decoder: MLP [64 -> 128 -> 256 -> 512 -> 128]
   - Uses cosine similarity for vector quantization
   - EMA updates for stable codebook learning

3. **User Embedding Layer**
   - Dimension: uid_emb_size (64)

4. **Transformer Sequence Encoder**
   - Input: Concatenated [quantized POI embedding (64) + user embedding (64)]
   - Architecture: 2-layer transformer with 4 attention heads
   - Feedforward dimension: 512
   - Output: Next POI predictions

### Loss Function
Total Loss = pred_loss_weight × prediction_loss + quant_loss_weight × quantization_loss + recon_loss_weight × reconstruction_loss

---

## Validation Status

### JSON Syntax
- **Status**: Valid
- All parameters properly formatted
- No syntax errors detected

### Parameter Mapping
All model hyperparameters from the implementation are included:
- POI and user embeddings: Mapped
- CRQVAE architecture: Complete
- Vector quantization: All parameters configured
- Transformer: All parameters configured
- Training: Standard LibCity parameters included

### Model-Config Compatibility
- All `config.get()` calls in model implementation have corresponding entries
- Default values in model code align with config file
- No missing required parameters

---

## Usage Example

```bash
# Run GNPRSID on Foursquare Tokyo dataset
python run_model.py --task traj_loc_pred --model GNPRSID --dataset foursquare_tky

# With custom config
python run_model.py --task traj_loc_pred --model GNPRSID --dataset gowalla \
  --batch_size 256 --learning_rate 0.0005 --max_epoch 200
```

---

## Notes

1. **Semantic ID Learning**: The model learns discrete semantic IDs for POIs through the CRQVAE component, which can be extracted using the `get_semantic_ids()` method.

2. **Two-Stage Operation**: The model can be trained end-to-end or with a pre-trained CRQVAE using `load_pretrained_crqvae()`.

3. **Evaluation Methods**: Supports both full evaluation (`evaluate_method: 'all'`) and negative sampling (`evaluate_method: 'sample'`).

4. **Memory Considerations**: With num_codebooks=64 and num_rq_layers=3, the model maintains 192 total codebook vectors. Larger values may improve representation but increase memory usage.

5. **EMA vs Non-EMA**: The model uses EMA updates by default (use_ema=true) for stable codebook learning. This is recommended for better convergence.

6. **Sinkhorn Regularization**: The sk_epsilon parameter controls the temperature in optimal transport-based quantization. Set to 0 or None to disable.

---

## Configuration Validation Checklist

- [x] Model registered in task_config.json
- [x] Model config file created
- [x] All hyperparameters from paper included
- [x] JSON syntax valid
- [x] Model registered in __init__.py
- [x] Compatible with standard trajectory datasets
- [x] Parameter names match model implementation
- [x] Default values align with paper
- [x] Training parameters configured
- [x] Dataset compatibility verified

---

## References

**Model Implementation**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/GNPRSID.py`

**Configuration File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/GNPRSID.json`

**Task Config**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json` (Line 23, 150-155)

---

## Migration Status: COMPLETE

All configuration files have been successfully created and validated. The GNPRSID model is ready for training and evaluation in LibCity.
