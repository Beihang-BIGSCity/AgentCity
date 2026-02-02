# GraphMM Quick Reference

## Model Configuration

### Basic Usage
```bash
python run_model.py --task traj_loc_pred --model GraphMM --dataset your_dataset
```

### Task Config Entry
- **Task**: `traj_loc_pred`
- **Model Name**: `GraphMM`
- **Line in allowed_model**: 35
- **Config Block**: Lines 224-229

### Key Hyperparameters

| Category | Parameter | Default | Range | Description |
|----------|-----------|---------|-------|-------------|
| **Core** | emb_dim | 256 | 128-512 | Embedding dimension |
| | layer | 4 | 2-6 | K-hop adjacency polynomial |
| | use_crf | true | bool | Use CRF for decoding |
| **Training** | learning_rate | 0.0001 | 1e-5 to 1e-3 | Adam/AdamW learning rate |
| | batch_size | 32 | 16-128 | Batch size (important for CRF) |
| | tf_ratio | 0.5 | 0.0-1.0 | Teacher forcing ratio |
| | drop_prob | 0.5 | 0.0-0.7 | Dropout probability |
| **CRF** | gamma | 10000 | 1000-100000 | Unreachable road penalty |
| | topn | 5 | 3-10 | Top-N Viterbi candidates |
| | neg_nums | 800 | 200-2000 | Negative samples for CRF |
| **Architecture** | gin_depth | 3 | 2-5 | RoadGIN layers |
| | gin_mlp_layers | 2 | 1-3 | MLP layers per GIN |
| | digcn_depth | 2 | 1-4 | TraceGCN layers |

### Required Data Features

**Graph Data** (in data_feature):
- `num_roads`: Number of road segments
- `road_x`: Road features [num_roads, 28]
- `road_adj`: Road adjacency (PyG format)
- `A`: Adjacency matrix for A^k
- `num_grids`: Number of grid cells
- `trace_in_edge_index`: Trace graph incoming edges
- `trace_out_edge_index`: Trace graph outgoing edges
- `trace_weight`: Trace graph edge weights
- `map_matrix`: Grid-to-road mapping [num_grids, num_roads]

**Batch Data**:
- `grid_traces`: Grid IDs [batch, seq_len]
- `tgt_roads`: Road IDs [batch, road_len]
- `traces_gps`: GPS coords [batch, seq_len, 2]
- `sample_Idx`: Sample indices [batch, seq_len]
- `traces_lens`: Trajectory lengths (list)
- `road_lens`: Road sequence lengths (list)

### Architecture Components

1. **RoadGIN**: Graph Isomorphism Network for road encoding
2. **TraceGCN**: Bidirectional GCN for trace graph encoding
3. **Seq2Seq**: GRU-based encoder-decoder with attention
4. **CRF**: Conditional Random Field for structured decoding (optional)

### File Locations

```
Bigscity-LibCity/
├── libcity/
│   ├── model/
│   │   └── trajectory_loc_prediction/
│   │       ├── GraphMM.py                    # Model implementation
│   │       └── __init__.py                   # Model registration
│   └── config/
│       ├── task_config.json                  # Task configuration (lines 35, 224-229)
│       └── model/traj_loc_pred/
│           └── GraphMM.json                  # Model hyperparameters
```

### Dependencies

```bash
pip install torch-geometric torch-sparse
```

### Common Issues

1. **Missing Graph Data**: Ensure all graph structures are in data_feature
2. **Shape Mismatches**: Check grid_traces, tgt_roads dimensions
3. **CRF Memory**: Reduce neg_nums or batch_size if OOM
4. **PyG Import Errors**: Install torch-geometric and torch-sparse

### Performance Tuning

- **For larger road networks**: Increase `neg_nums`, `topn`
- **For faster training**: Disable CRF (`use_crf: false`), reduce `gin_depth`
- **For better accuracy**: Increase `emb_dim`, `layer`, enable CRF
- **For stable training**: Use gradient clipping (`grad_clip_norm: 5.0`)

### Example Config Override

```json
{
    "task": "traj_loc_pred",
    "model": "GraphMM",
    "dataset": "custom_map_matching",
    "emb_dim": 128,
    "use_crf": false,
    "batch_size": 64,
    "learning_rate": 0.001
}
```

## Status

- Configuration: ✓ Complete
- Registration: ✓ Complete
- Documentation: ✓ Complete
- Testing: ⚠ Requires custom dataset with graph data
