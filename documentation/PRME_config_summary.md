# PRME Configuration Summary

## Model Information

**Model Name**: PRME (Personalized Ranking Metric Embedding for Next New POI Recommendation)
**Paper**: Shanshan Feng, Xutao Li, Yifeng Zeng, Gao Cong, Yeow Meng Chee, Quan Yuan. "Personalized Ranking Metric Embedding for Next New POI Recommendation", IJCAI 2015
**Task Type**: Trajectory Location Prediction (traj_loc_pred)
**Original Implementation**: https://github.com/flaviovdf/prme

---

## Configuration Files

### 1. task_config.json Registration

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`

**Added to**: `traj_loc_pred.allowed_model` (line 29)

```json
{
  "traj_loc_pred": {
    "allowed_model": [
      ...
      "PLSPL",
      "PRME"
    ],
    ...
    "PRME": {
      "dataset_class": "TrajectoryDataset",
      "executor": "TrajLocPredExecutor",
      "evaluator": "TrajLocPredEvaluator",
      "traj_encoder": "StandardTrajectoryEncoder"
    }
  }
}
```

---

### 2. Model Configuration

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/PRME.json`

```json
{
    "model": "PRME",
    "dataset_class": "TrajectoryDataset",
    "executor": "TrajLocPredExecutor",
    "evaluator": "TrajLocPredEvaluator",

    "embedding_dim": 50,
    "alpha": 0.5,
    "tau": 3.0,
    "num_negative": 10,
    "regularization": 0.03,

    "learning_rate": 0.001,
    "max_epoch": 100,
    "batch_size": 64,
    "learner": "adam",
    "lr_decay": false,
    "lr_scheduler": "multisteplr",
    "lr_decay_ratio": 0.1,
    "steps": [50, 80],
    "clip_grad_norm": false,
    "use_early_stop": true,
    "patience": 10,

    "device": "cpu",
    "gpu": true,
    "gpu_id": 0
}
```

---

## Model Parameters

### PRME-Specific Hyperparameters

| Parameter | Default | Source | Description |
|-----------|---------|--------|-------------|
| `embedding_dim` | 50 | IJCAI 2015 paper | Dimension of geographic and personalized embeddings |
| `alpha` | 0.5 | IJCAI 2015 paper | Balance between personalized distance and geographic distance (0-1) |
| `tau` | 3.0 | IJCAI 2015 paper | Time threshold in hours for alpha adjustment |
| `num_negative` | 10 | IJCAI 2015 paper | Number of negative samples per positive sample for BPR loss |
| `regularization` | 0.03 | IJCAI 2015 paper | L2 regularization coefficient for embeddings |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.001 | Learning rate for Adam optimizer |
| `max_epoch` | 100 | Maximum number of training epochs |
| `batch_size` | 64 | Training batch size |
| `learner` | "adam" | Optimizer type |
| `use_early_stop` | true | Enable early stopping |
| `patience` | 10 | Early stopping patience (epochs) |

---

## Model Architecture

### Key Concepts

1. **Dual Embedding Spaces**:
   - **Geographic embeddings (XG_ok)**: Captures sequential POI transitions
   - **Personalized POI embeddings (XP_ok)**: Captures POI-specific features for personalized recommendations
   - **User embeddings (XP_hk)**: Captures user-specific preferences

2. **Distance Metric**:
   ```
   distance = alpha * personalized_dist + (1-alpha) * geographic_dist

   where:
   - personalized_dist = ||XP_ok[destination] - XP_hk[user]||^2
   - geographic_dist = ||XG_ok[destination] - XG_ok[source]||^2
   ```

3. **Time-Aware Alpha Adjustment**:
   - If time_delta > tau: alpha = 1.0 (purely personalized, long time gap)
   - Otherwise: use configured alpha (mixed mode)

4. **Pairwise Ranking Loss (BPR-style)**:
   - Prefers actual destination over negative samples
   - Loss = -log(sigmoid(neg_distance - pos_distance))

---

## Dataset Compatibility

### Required Data Features

The model expects the following keys in `data_feature`:

| Feature | Description | Required |
|---------|-------------|----------|
| `loc_size` | Number of POI locations | Yes |
| `uid_size` | Number of users | Yes |
| `loc_pad` | Padding index for locations | Optional (default: 0) |

### Input Batch Format

The model expects batches with:

| Key | Shape | Description |
|-----|-------|-------------|
| `current_loc` | (batch_size, seq_len) | POI indices in trajectory |
| `uid` | (batch_size,) | User indices |
| `target` | (batch_size,) | Target next POI indices |
| `current_tim` | (batch_size, seq_len) | Time values (optional, for alpha adjustment) |
| `target_tim` | (batch_size,) | Target time (optional, for time delta calculation) |

### Compatible Datasets

PRME is designed for POI recommendation and works with trajectory datasets that include:
- User check-ins at POIs
- Temporal information
- Sequential trajectories

**Allowed datasets** (from task_config.json):
- `foursquare_tky` - Foursquare check-ins in Tokyo
- `foursquare_nyc` - Foursquare check-ins in New York City
- `gowalla` - Gowalla check-ins
- `foursquare_serm` - Foursquare dataset for SERM
- `Proto` - Prototype dataset

---

## Model Registration

### Python Module Registration

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/__init__.py`

The model is already registered (line 23):

```python
from libcity.model.trajectory_loc_prediction.PRME import PRME

__all__ = [
    ...
    "PLSPL",
    "PRME"
]
```

---

## Usage Example

### Running PRME on Foursquare Dataset

```bash
cd /home/wangwenrui/shk/AgentCity/Bigscity-LibCity

python run_model.py --task traj_loc_pred --model PRME --dataset foursquare_tky
```

### Custom Configuration

Create a custom config file (e.g., `prme_custom.json`):

```json
{
    "model": "PRME",
    "dataset": "foursquare_nyc",
    "embedding_dim": 64,
    "alpha": 0.6,
    "tau": 2.0,
    "num_negative": 15,
    "regularization": 0.05,
    "learning_rate": 0.0005,
    "max_epoch": 150,
    "batch_size": 128
}
```

Run with custom config:

```bash
python run_model.py --config_file prme_custom.json
```

---

## Model Features

### Advantages

1. **Dual Embedding Spaces**: Captures both geographic transitions and personalized preferences
2. **Time-Aware**: Adjusts recommendation strategy based on time intervals
3. **Efficient Training**: Uses BPR-style ranking loss with negative sampling
4. **Scalable**: Embedding-based approach scales well with large POI sets

### Limitations

1. **Cold Start**: Requires sufficient user history for personalized embeddings
2. **Sequential Dependency**: Primarily models transitions from last POI
3. **Fixed Alpha**: Uses a global alpha parameter (though adjustable by time)

---

## Dataset Requirements

### Minimum Requirements

For PRME to work effectively:

1. **User Information**: Each trajectory must have associated user IDs
2. **POI Information**: POI IDs for each check-in
3. **Temporal Information**: Timestamps for check-ins (optional but recommended for time-aware features)
4. **Sequential Structure**: Check-ins organized in temporal sequences

### Data Preprocessing

The `TrajectoryDataset` class handles:
- Filtering sessions by length (min_session_len, max_session_len)
- Filtering users by number of sessions (min_sessions)
- Filtering users by total check-ins (min_checkins)
- Trajectory encoding using `StandardTrajectoryEncoder`

---

## Evaluation Metrics

PRME uses `TrajLocPredEvaluator` which computes:

- **Accuracy**: Top-k accuracy for next POI prediction
- **Precision@k**: Precision at different k values
- **Recall@k**: Recall at different k values
- **F1@k**: F1 score at different k values
- **MRR**: Mean Reciprocal Rank
- **NDCG@k**: Normalized Discounted Cumulative Gain

---

## Technical Notes

### Embedding Initialization

- All embeddings initialized with normal distribution (mean=0.0, std=0.01)
- Padding embeddings set to zeros

### Loss Function

Total loss = Pairwise Ranking Loss + L2 Regularization

```python
ranking_loss = -log(sigmoid(neg_distance - pos_distance))
reg_loss = regularization * sum(||embeddings||^2)
total_loss = ranking_loss + reg_loss
```

### Negative Sampling

- Negative samples drawn uniformly from all POIs
- Excludes the positive target and padding index
- Default: 10 negative samples per positive

---

## Configuration Checklist

- [x] Model added to `task_config.json` allowed_model list
- [x] Model configuration entry added to `task_config.json`
- [x] Model config file created at `config/model/traj_loc_pred/PRME.json`
- [x] All hyperparameters from original paper included
- [x] Training hyperparameters added
- [x] Model registered in `__init__.py`
- [x] Dataset compatibility verified
- [x] Documentation created

---

## Status

**Configuration Complete**: Ready for testing

The PRME model is now fully configured in LibCity and ready for training and evaluation on trajectory location prediction tasks.

### Next Steps

1. Test on a small dataset (e.g., `foursquare_tky`)
2. Verify training convergence
3. Compare results with baseline models (FPMC, STRNN, etc.)
4. Tune hyperparameters if needed
5. Run full experiments on all compatible datasets

---

## References

1. Shanshan Feng, Xutao Li, Yifeng Zeng, Gao Cong, Yeow Meng Chee, Quan Yuan. "Personalized Ranking Metric Embedding for Next New POI Recommendation". IJCAI 2015.
2. Original implementation: https://github.com/flaviovdf/prme
3. LibCity documentation: https://bigscity-libcity-docs.readthedocs.io/
