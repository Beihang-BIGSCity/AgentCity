# DCHL Quick Start Guide

## Model Overview
**DCHL** (Disentangled Contrastive Hypergraph Learning) is a next POI recommendation model from SIGIR 2024 that learns disentangled representations using three graph views:
- Multi-view hypergraph (collaborative patterns)
- Geographic graph (spatial patterns)
- Directed hypergraph (sequential patterns)

## Quick Run

```bash
# Basic usage
python run_model.py --task traj_loc_pred --model DCHL --dataset foursquare_nyc

# With GPU
python run_model.py --task traj_loc_pred --model DCHL --dataset foursquare_nyc --device cuda:0

# Custom hyperparameters
python run_model.py --task traj_loc_pred --model DCHL --dataset foursquare_nyc \
    --emb_dim 256 \
    --num_mv_layers 4 \
    --lambda_cl 0.2 \
    --max_epoch 50
```

## Supported Datasets
- `foursquare_tky` - Foursquare Tokyo
- `foursquare_nyc` - Foursquare New York City
- `gowalla` - Gowalla check-ins
- `foursquare_serm` - Foursquare SERM
- `Proto` - Prototype dataset

## Key Hyperparameters

| Parameter | Default | Description | Tuning Range |
|-----------|---------|-------------|--------------|
| `emb_dim` | 128 | Embedding dimension | 64-256 |
| `num_mv_layers` | 3 | Multi-view hypergraph layers | 2-5 |
| `num_geo_layers` | 3 | Geographic graph layers | 2-5 |
| `num_di_layers` | 3 | Directed hypergraph layers | 2-5 |
| `lambda_cl` | 0.1 | Contrastive loss weight | 0.01-0.5 |
| `temperature` | 0.1 | InfoNCE temperature | 0.05-0.2 |
| `dropout` | 0.3 | Dropout rate | 0.1-0.5 |
| `distance_threshold` | 2.5 | Geo distance (km) | 1.0-5.0 |
| `learning_rate` | 0.001 | Learning rate | 0.0001-0.01 |
| `batch_size` | 200 | Batch size | 64-512 |

## Example Configurations

### High Performance (Large Dataset)
```bash
python run_model.py --task traj_loc_pred --model DCHL --dataset gowalla \
    --emb_dim 256 \
    --num_mv_layers 4 \
    --num_geo_layers 4 \
    --num_di_layers 4 \
    --batch_size 512 \
    --max_epoch 50
```

### Fast Training (Small Dataset)
```bash
python run_model.py --task traj_loc_pred --model DCHL --dataset foursquare_serm \
    --emb_dim 64 \
    --num_mv_layers 2 \
    --num_geo_layers 2 \
    --num_di_layers 2 \
    --batch_size 128 \
    --max_epoch 20
```

### Strong Regularization
```bash
python run_model.py --task traj_loc_pred --model DCHL --dataset foursquare_nyc \
    --lambda_cl 0.3 \
    --dropout 0.5 \
    --keep_rate 0.9 \
    --keep_rate_poi 0.9 \
    --weight_decay 0.001
```

## Data Requirements

### Minimum Requirements
- User sessions with POI sequences
- At least 2 sessions per user
- At least 3 check-ins per user
- Session length: 5-50 POIs

### Optional (Recommended)
- POI coordinates (latitude, longitude) for geographic graph
- Without coordinates, spatial view will be disabled

## Troubleshooting

### Out of Memory
- Reduce `emb_dim` (e.g., 64 or 128)
- Reduce `batch_size`
- Reduce number of layers
- Enable gradient checkpointing

### Poor Performance
- Check if POI coordinates are available (important!)
- Increase `lambda_cl` for stronger disentanglement
- Tune `distance_threshold` based on city size
- Increase `num_layers` for all views

### Slow Training
- Precompute and cache hypergraph structures
- Reduce number of layers
- Increase `batch_size` if memory allows
- Use GPU acceleration

## Output Files

Results are saved to:
```
result/traj_loc_pred/<dataset>/<model_name>/
├── model_checkpoints/
├── training_log.txt
├── evaluation_results.json
└── best_model.pth
```

## Evaluation Metrics

DCHL is evaluated using:
- Recall@K (K=1, 5, 10, 20)
- NDCG@K (K=5, 10, 20)
- MRR (Mean Reciprocal Rank)
- Precision@K

## Citation

If you use DCHL, please cite:
```
@inproceedings{lai2024dchl,
  title={Disentangled Contrastive Hypergraph Learning for Next POI Recommendation},
  author={Lai, Yantong and others},
  booktitle={SIGIR},
  year={2024}
}
```

## Additional Resources

- Full documentation: `/home/wangwenrui/shk/AgentCity/documents/DCHL_configuration_summary.md`
- Original paper: SIGIR 2024
- Original code: https://github.com/icmpnorequest/SIGIR2024_DCHL
- LibCity docs: https://bigscity-libcity-docs.readthedocs.io/
