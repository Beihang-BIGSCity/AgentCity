# GraphMM Quick Reference

## Basic Usage

### 1. Install Dependencies
```bash
pip install torch_geometric torch_sparse
```

### 2. Run GraphMM
```bash
python run_model.py --task map_matching --model GraphMM --dataset Seattle
```

## Configuration Files

### Model Config
**Location**: `Bigscity-LibCity/libcity/config/model/map_matching/GraphMM.json`

### Key Parameters
```json
{
  "emb_dim": 256,           // Embedding dimension
  "dropout": 0.5,           // Dropout probability
  "use_crf": true,          // Use CRF layer
  "use_attention": true,    // Use attention in decoder
  "bidirectional": true,    // Use bidirectional encoder
  "teacher_forcing_ratio": 0.5,  // Teacher forcing ratio
  "layer": 4,               // Adjacency polynomial hops
  "gamma": 10000,           // Unreachable road penalty
  "topn": 5,                // Top-N candidates for decoding
  "neg_nums": 800,          // Negative samples for CRF
  "batch_size": 32,         // Batch size (32 with CRF, 256 without)
  "learning_rate": 0.0001,  // Learning rate
  "optimizer": "AdamW",     // Optimizer type
  "max_grad_norm": 5.0      // Gradient clipping
}
```

## Custom Configuration

### Create custom config file
```json
{
  "emb_dim": 128,
  "batch_size": 64,
  "use_crf": false,
  "learning_rate": 0.001
}
```

### Run with custom config
```bash
python run_model.py --task map_matching --model GraphMM \
    --dataset Seattle --config_file my_config.json
```

## Dataset Requirements

GraphMM requires datasets with:
1. Road network graph (nodes, edges, features)
2. Trace graph (grid cells, transitions)
3. Grid-to-road mapping
4. GPS trajectories with road labels

### Supported Datasets
- Seattle
- Neftekamsk
- Valky
- Ruzhany
- Santander
- Spaichingen
- NovoHamburgo

## Model Architecture

```
Input GPS Trajectory
    ↓
Grid Discretization → TraceGCN → Grid Embeddings
    ↓
Road Network → RoadGIN → Road Embeddings
    ↓
Seq2Seq Decoder (GRU + Attention)
    ↓
CRF Layer (optional)
    ↓
Predicted Road Sequence
```

## Hyperparameter Tuning

### For Small Road Networks (< 1000 roads)
```json
{
  "emb_dim": 128,
  "layer": 2,
  "neg_nums": 500
}
```

### For Large Road Networks (> 10000 roads)
```json
{
  "emb_dim": 256,
  "layer": 4,
  "neg_nums": 1000,
  "batch_size": 16
}
```

### For Fast Training (No CRF)
```json
{
  "use_crf": false,
  "batch_size": 256,
  "learning_rate": 0.001
}
```

## Common Issues

### Issue: Missing Dependencies
```
ImportError: GraphMM requires torch_geometric and torch_sparse
```
**Solution**: Install dependencies
```bash
pip install torch_geometric torch_sparse
```

### Issue: Out of Memory
**Solution**: Reduce batch size or embedding dimension
```json
{
  "batch_size": 16,
  "emb_dim": 128
}
```

### Issue: Dataset Missing Graph Structures
**Solution**: Ensure dataset provides:
- `num_roads`, `num_grids`
- `road_x`, `road_adj`
- `trace_in_edge_index`, `trace_out_edge_index`, `trace_weight`
- `map_matrix`, `A_matrix`

## Performance Tips

1. **Use CRF for Better Accuracy**: Set `"use_crf": true`
2. **Use Attention**: Set `"use_attention": true`
3. **Tune Teacher Forcing**: Adjust `"teacher_forcing_ratio"` (0.0 to 1.0)
4. **Gradient Clipping**: Keep `"max_grad_norm": 5.0`
5. **Early Stopping**: Enable with `"use_early_stop": true`

## File Locations

### Model
```
Bigscity-LibCity/libcity/model/map_matching/GraphMM.py
```

### Configuration
```
Bigscity-LibCity/libcity/config/model/map_matching/GraphMM.json
```

### Dataset
```
Bigscity-LibCity/libcity/data/dataset/deep_map_matching_dataset.py
```

## Example Command with All Options

```bash
python run_model.py \
    --task map_matching \
    --model GraphMM \
    --dataset Seattle \
    --emb_dim 256 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --use_crf True \
    --max_epoch 200 \
    --gpu True \
    --gpu_id 0
```

## Monitoring Training

### Check Logs
```bash
tail -f logs/GraphMM_Seattle.log
```

### Tensorboard (if enabled)
```bash
tensorboard --logdir libcity_cache/GraphMM
```

## Evaluation Metrics

GraphMM is evaluated using MapMatchingEvaluator:
- Accuracy: % of correctly matched road segments
- Precision: Precision of matched segments
- Recall: Recall of matched segments
- F1-Score: Harmonic mean of precision and recall

## Citation

If you use GraphMM in your research, please cite:
```
"GraphMM: Graph-based Vehicular Map Matching by Leveraging
Trajectory and Road Correlations"
```
