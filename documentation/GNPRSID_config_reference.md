# GNPRSID Configuration Quick Reference

## Model Overview
GNPRSID combines a Cosine Residual Quantized VAE (CRQVAE) with Transformer-based sequence modeling for next POI recommendation.

---

## Parameter Categories

### 1. Embedding Dimensions
| Parameter | Value | Description | Source |
|-----------|-------|-------------|--------|
| loc_emb_size | 128 | POI embedding size | Paper default |
| uid_emb_size | 64 | User embedding size | Paper default |
| e_dim | 64 | CRQVAE latent dimension | Paper default |

### 2. CRQVAE Architecture
| Parameter | Value | Description | Source |
|-----------|-------|-------------|--------|
| encoder_layers | [512, 256, 128] | MLP hidden layers | Paper default |
| num_codebooks | 64 | Codebook entries per level | Paper default |
| num_rq_layers | 3 | Residual quantization layers | Paper default |
| dropout_prob | 0.1 | MLP dropout | Paper default |
| use_bn | true | Batch normalization | Common practice |
| loss_type | "mse" | Reconstruction loss | Paper default |

### 3. Vector Quantization
| Parameter | Value | Description | Source |
|-----------|-------|-------------|--------|
| beta | 0.25 | Commitment cost | Paper: commitment_cost |
| kmeans_init | true | K-means initialization | Best practice |
| kmeans_iters | 100 | K-means iterations | Standard |
| sk_epsilon | 0.1 | Sinkhorn temperature | Paper default |
| sk_iters | 50 | Sinkhorn iterations | Standard |
| use_ema | true | EMA codebook updates | Paper default |
| ema_decay | 0.99 | EMA decay rate | Paper specification |
| use_linear | 1 | Linear codebook projection | Enabled |

### 4. Loss Weights
| Parameter | Value | Description | Source |
|-----------|-------|-------------|--------|
| pred_loss_weight | 1.0 | Prediction loss weight | Paper default |
| quant_loss_weight | 0.5 | Quantization loss weight | Paper default |
| recon_loss_weight | 0.1 | Reconstruction loss weight | Paper default |

### 5. Transformer Configuration
| Parameter | Value | Description | Source |
|-----------|-------|-------------|--------|
| transformer_nhead | 4 | Attention heads | Paper: nhead |
| transformer_nhid | 512 | Feedforward dimension | Paper: dim_feedforward |
| transformer_nlayers | 2 | Encoder layers | Paper: nlayers |
| transformer_dropout | 0.1 | Transformer dropout | Paper default |

### 6. Training Parameters
| Parameter | Value | Description | Source |
|-----------|-------|-------------|--------|
| batch_size | 128 | Batch size | Paper default |
| learning_rate | 0.001 | Learning rate | Paper: 1e-3 |
| max_epoch | 100 | Training epochs | Paper default |
| L2 | 0.0001 | Weight decay | Standard |
| optimizer | "adamw" | Optimizer type | Best practice |
| lr_step | 20 | LR decay step | Standard |
| lr_decay | 0.9 | LR decay factor | Standard |

---

## Parameter Tuning Guide

### For Better Accuracy
- Increase `num_codebooks` (64 → 128)
- Increase `num_rq_layers` (3 → 4)
- Increase `transformer_nlayers` (2 → 3)
- Increase `max_epoch` (100 → 200)

### For Faster Training
- Increase `batch_size` (128 → 256)
- Decrease `num_rq_layers` (3 → 2)
- Decrease `transformer_nlayers` (2 → 1)
- Set `kmeans_init` to false

### For Better Generalization
- Increase `dropout_prob` (0.1 → 0.2)
- Increase `L2` (0.0001 → 0.001)
- Increase `transformer_dropout` (0.1 → 0.2)
- Lower `learning_rate` (0.001 → 0.0005)

### For Different Dataset Sizes
**Small datasets** (< 10K check-ins):
- Decrease `loc_emb_size` (128 → 64)
- Decrease `num_codebooks` (64 → 32)
- Increase `dropout_prob` (0.1 → 0.2)

**Large datasets** (> 100K check-ins):
- Increase `batch_size` (128 → 256)
- Increase `encoder_layers` ([512,256,128] → [1024,512,256,128])
- Increase `max_epoch` (100 → 200)

---

## Common Issues & Solutions

### Issue: NaN in training
**Solutions:**
- Lower `learning_rate` (0.001 → 0.0001)
- Lower `sk_epsilon` (0.1 → 0.05)
- Set `use_ema` to true
- Check for dead codebooks

### Issue: Poor quantization
**Solutions:**
- Increase `quant_loss_weight` (0.5 → 1.0)
- Enable `kmeans_init` = true
- Increase `num_codebooks` (64 → 128)
- Adjust `beta` (0.25 → 0.5)

### Issue: Overfitting
**Solutions:**
- Increase `dropout_prob` (0.1 → 0.3)
- Increase `L2` (0.0001 → 0.001)
- Decrease model capacity
- Add early stopping

### Issue: Slow convergence
**Solutions:**
- Increase `learning_rate` (0.001 → 0.002)
- Adjust `lr_decay` schedule
- Increase `batch_size` (128 → 256)
- Reduce `sk_iters` (50 → 20)

---

## Advanced Features

### Semantic ID Extraction
After training, extract semantic IDs:
```python
model.eval()
semantic_ids = model.get_semantic_ids(location_indices)
# Returns: (batch, num_rq_layers) discrete codes
```

### Pre-trained CRQVAE
Load pre-trained CRQVAE weights:
```python
model.load_pretrained_crqvae('path/to/crqvae_checkpoint.pth')
```

### Two-Stage Training
1. **Stage 1**: Train CRQVAE only
   - High `recon_loss_weight` (0.1 → 1.0)
   - High `quant_loss_weight` (0.5 → 1.0)
   - Zero `pred_loss_weight` (1.0 → 0.0)

2. **Stage 2**: Fine-tune with prediction
   - Normal loss weights
   - Optionally freeze CRQVAE

---

## LibCity Integration

### Command Line
```bash
# Standard run
python run_model.py --task traj_loc_pred --model GNPRSID --dataset foursquare_tky

# Custom parameters
python run_model.py --task traj_loc_pred --model GNPRSID --dataset gowalla \
    --learning_rate 0.0005 --batch_size 256 --max_epoch 150
```

### Config File Override
Create a custom config JSON and pass via `--config_file`:
```json
{
    "model": "GNPRSID",
    "dataset": "foursquare_nyc",
    "num_codebooks": 128,
    "learning_rate": 0.0005,
    "batch_size": 256
}
```

---

## Performance Expectations

### Typical Metrics (Foursquare datasets)
- **Acc@1**: 0.05 - 0.15
- **Acc@5**: 0.15 - 0.30
- **Acc@10**: 0.25 - 0.45
- **MRR**: 0.10 - 0.25

### Training Time (estimates)
- **Small dataset** (10K check-ins): ~30 min
- **Medium dataset** (100K check-ins): ~3 hours
- **Large dataset** (1M check-ins): ~24 hours

*Based on single GPU (V100/A100)*

---

## File Locations

- **Model**: `libcity/model/trajectory_loc_prediction/GNPRSID.py`
- **Config**: `libcity/config/model/traj_loc_pred/GNPRSID.json`
- **Task Config**: `libcity/config/task_config.json` (line 23, 150-155)

---

## Citation

If you use GNPRSID, please cite the original paper:
```
[Paper citation to be added]
```
