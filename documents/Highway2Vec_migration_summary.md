# Config Migration: Highway2Vec

## Migration Status: VERIFIED AND COMPLETE ✓

---

## 1. task_config.json Registration

**Status:** ✓ VERIFIED - Correctly registered

**Location:** Line 1131 in `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`

**Task Type:** `road_representation`

**Configuration:**
```json
"road_representation": {
    "allowed_model": [
        "ChebConv",
        "LINE",
        "GAT",
        "Node2Vec",
        "DeepWalk",
        "GeomGCN",
        "SARN",
        "CCASSG",
        "Highway2Vec"  // ✓ Added at line 1131
    ],
    "allowed_dataset": [
        "BJ_roadmap"
    ],
    "Highway2Vec": {
        "dataset_class": "ChebConvDataset",
        "executor": "ChebConvExecutor",
        "evaluator": "RoadRepresentationEvaluator"
    }
}
```

---

## 2. Model Configuration File

**Status:** ✓ VERIFIED AND UPDATED

**File Path:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/road_representation/Highway2Vec.json`

**Parameters:**

| Parameter | Value | Source |
|-----------|-------|--------|
| `model` | "Highway2Vec" | Added for metadata |
| `hidden_dim` | 64 | Original paper/repo |
| `output_dim` | 3 | Original paper (3D embedding for visualization) |
| `latent_dim` | 3 | Alias for output_dim |
| `scaler` | "none" | ChebConvDataset default |
| `max_epoch` | 2000 | Road representation models default |
| `learner` | "adam" | Original paper |
| `learning_rate` | 0.001 | Standard Adam default |
| `lr_decay` | true | Best practice for convergence |
| `lr_scheduler` | "reducelronplateau" | Adaptive learning rate |
| `lr_decay_ratio` | 0.7 | Standard decay ratio |
| `clip_grad_norm` | true | Prevent gradient explosion |
| `max_grad_norm` | 5 | Standard clipping threshold |
| `use_early_stop` | true | Prevent overfitting |
| `patience` | 100 | Suitable for 2000 epochs |
| `log_every` | 10 | Added for logging frequency |
| `saved` | true | Added for model checkpointing |
| `load_best_epoch` | true | Added to load best model |

**Complete Configuration:**
```json
{
  "model": "Highway2Vec",
  "hidden_dim": 64,
  "output_dim": 3,
  "latent_dim": 3,
  "scaler": "none",
  "max_epoch": 2000,
  "learner": "adam",
  "learning_rate": 0.001,
  "lr_decay": true,
  "lr_scheduler": "reducelronplateau",
  "lr_decay_ratio": 0.7,
  "clip_grad_norm": true,
  "max_grad_norm": 5,
  "use_early_stop": true,
  "patience": 100,
  "log_every": 10,
  "saved": true,
  "load_best_epoch": true
}
```

---

## 3. Model Implementation

**Status:** ✓ VERIFIED

**File Path:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/road_representation/Highway2Vec.py`

**Model Registration:** ✓ Added to `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/road_representation/__init__.py`

**Architecture:**
- **Encoder:** input_dim → 64 (ReLU) → 3
- **Decoder:** 3 → 64 (ReLU) → input_dim
- **Loss:** MSE reconstruction loss
- **Optimizer:** Adam (learning_rate=0.001)

**Model Features:**
- Inherits from `AbstractTrafficStateModel`
- Implements `forward()`, `predict()`, and `calculate_loss()` methods
- Compatible with ChebConvExecutor training pipeline
- Saves embeddings during prediction (follows LibCity pattern)
- Supports masked training/evaluation

---

## 4. Dataset Compatibility

**Status:** ✓ VERIFIED - Fully Compatible

**Dataset Class:** `ChebConvDataset`
**File Path:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/dataset_subclass/chebconv_dataset.py`

**Dataset Configuration:**
```json
{
  "cache_dataset": true,
  "train_rate": 0.7,
  "eval_rate": 0.1,
  "scaler": "none"
}
```

**Data Features Provided:**
- `num_nodes`: Number of road segments in network
- `feature_dim`: Dimension of input features (one-hot encoded OSM attributes)
- `adj_mx`: Adjacency matrix (sparse format)
- `scaler`: Data normalization scaler
- `node_features`: Road segment feature vectors
- `mask`: Train/valid/test split masks

**Input Data Format:**
- **Source:** OSM road network features (one-hot encoded)
- **Dimension:** 100-200 (varies by dataset, processed from .geo file)
- **Preprocessing:** One-hot encoding for categorical features (highway, lanes), normalization for continuous features (length, maxspeed, width)

**Compatibility Verification:**
✓ Highway2Vec expects `node_features` - ChebConvDataset provides this
✓ Highway2Vec uses `feature_dim` from data_feature - ChebConvDataset provides this
✓ Highway2Vec supports masking - ChebConvDataset provides train/valid/test masks
✓ Highway2Vec is an autoencoder (input=output) - ChebConvDataset clones features as labels

---

## 5. Executor Configuration

**Status:** ✓ VERIFIED - Fully Compatible

**Executor Class:** `ChebConvExecutor`
**File Path:** `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/executor/chebconv_executor.py`

**Executor Features:**
- Inherits from `TrafficStateExecutor`
- Handles node feature-based training (not time-series)
- Implements `train()`, `evaluate()`, and `_train_epoch()` methods
- Supports model checkpointing and early stopping
- Saves embeddings to cache directory during evaluation
- Detects embedding models (output_dim ≠ input_dim) and skips reconstruction metrics

**Training Pipeline:**
1. Loads node features from dataset
2. Applies train/valid/test masks
3. Calls `model.calculate_loss()` for training
4. Supports gradient clipping and learning rate scheduling
5. Saves best model based on validation loss

**Evaluation Pipeline:**
1. Calls `model.predict()` to generate embeddings
2. Saves embeddings to `./libcity/cache/{exp_id}/evaluate_cache/embedding_{model}_{dataset}_{output_dim}.npy`
3. Calls evaluator to compute downstream task metrics
4. Skips reconstruction metrics for embedding-only models (Highway2Vec outputs 3D embeddings)

**Compatibility Verification:**
✓ ChebConvExecutor calls `model.calculate_loss()` - Highway2Vec implements this
✓ ChebConvExecutor calls `model.predict()` - Highway2Vec implements this
✓ ChebConvExecutor passes `{'node_features': ..., 'node_labels': ..., 'mask': ...}` - Highway2Vec handles this
✓ ChebConvExecutor detects embedding models - Highway2Vec has output_dim (3) ≠ input_dim (~100-200)

---

## 6. Evaluator Configuration

**Status:** ✓ VERIFIED

**Evaluator Class:** `RoadRepresentationEvaluator`

**Evaluation Strategy:**
- Highway2Vec is an unsupervised model (autoencoder)
- Embeddings are saved during prediction
- Downstream tasks (clustering, visualization) can be performed on saved embeddings
- ChebConvExecutor skips reconstruction metrics for embedding-only models

---

## 7. Original Paper Parameters

**Paper:** "Highway2Vec: Learning representations for road network elements"
**Repository:** https://github.com/sarm/highway2vec

| Parameter | Original Value | LibCity Config |
|-----------|---------------|----------------|
| Input dimension | 100-200 (OSM features) | Dynamic (from dataset) |
| Hidden dimension | 64 | `hidden_dim: 64` ✓ |
| Latent dimension | 3 | `output_dim: 3` ✓ |
| Loss function | MSE | Implemented in `calculate_loss()` ✓ |
| Optimizer | Adam | `learner: adam` ✓ |
| Learning rate | Not specified (default 1e-3) | `learning_rate: 0.001` ✓ |

---

## 8. Configuration Summary

### ✓ All Configurations Complete

| Component | Status | File Path |
|-----------|--------|-----------|
| task_config.json | ✓ Registered | `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json` (line 1131, 1176-1180) |
| Model config | ✓ Complete | `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/road_representation/Highway2Vec.json` |
| Model implementation | ✓ Complete | `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/road_representation/Highway2Vec.py` |
| Model registration | ✓ Complete | `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/road_representation/__init__.py` (line 9, 20) |
| Dataset | ✓ Compatible | ChebConvDataset |
| Executor | ✓ Compatible | ChebConvExecutor |
| Evaluator | ✓ Compatible | RoadRepresentationEvaluator |

---

## 9. Recommendations and Notes

### ✓ No Issues Found

All configurations are properly set up and verified. The model is ready for training and evaluation.

### Additional Notes:

1. **Input Dimension:** The `input_dim` is automatically inferred from `data_feature['feature_dim']` during model initialization, matching the ChebConvDataset's processed features.

2. **Embedding Output:** Highway2Vec outputs 3D embeddings (suitable for visualization), not reconstructions. The ChebConvExecutor correctly detects this and skips reconstruction metrics.

3. **Training Strategy:**
   - Autoencoder training: Minimize reconstruction loss (MSE)
   - Early stopping with patience=100 prevents overfitting
   - ReduceLROnPlateau scheduler adapts learning rate based on validation loss

4. **Evaluation Strategy:**
   - Embeddings are automatically saved during evaluation
   - Path: `./libcity/cache/{exp_id}/evaluate_cache/embedding_Highway2Vec_{dataset}_3.npy`
   - Can be used for downstream tasks: clustering, similarity search, visualization

5. **Dataset Requirements:**
   - Requires a `.geo` file with road segment features (OSM attributes)
   - Requires a `.rel` file with road network topology
   - Default dataset: BJ_roadmap

6. **Scalability:**
   - The model handles variable input dimensions (100-200 features)
   - Suitable for road networks of different sizes
   - Sparse adjacency matrix support for large networks

---

## 10. Example Usage

```python
# Run Highway2Vec on BJ_roadmap dataset
python run_model.py --task road_representation --model Highway2Vec --dataset BJ_roadmap

# Or with custom config
python run_model.py --task road_representation --model Highway2Vec --dataset BJ_roadmap \
    --hidden_dim 64 --output_dim 3 --learning_rate 0.001 --max_epoch 2000
```

**Expected Output:**
- Trained model saved to: `./libcity/cache/{exp_id}/model_cache/`
- Embeddings saved to: `./libcity/cache/{exp_id}/evaluate_cache/embedding_Highway2Vec_BJ_roadmap_3.npy`
- Training logs: `./libcity/log/{exp_id}/`

---

## 11. Migration Checklist

- [x] Highway2Vec added to `task_config.json` under `road_representation.allowed_model`
- [x] Highway2Vec configuration added to `task_config.json` with correct dataset/executor/evaluator
- [x] Model config file created: `config/model/road_representation/Highway2Vec.json`
- [x] All hyperparameters documented with sources
- [x] Model implementation verified: `libcity/model/road_representation/Highway2Vec.py`
- [x] Model registered in `__init__.py`
- [x] Dataset compatibility verified: ChebConvDataset
- [x] Executor compatibility verified: ChebConvExecutor
- [x] Evaluator compatibility verified: RoadRepresentationEvaluator
- [x] Training pipeline verified
- [x] Evaluation pipeline verified
- [x] Embedding save mechanism verified
- [x] Config parameters updated with best practices
- [x] Documentation complete

---

## Verification Date: 2026-02-02
## Verified By: LibCity Configuration Migration Agent
## Status: COMPLETE AND READY FOR TRAINING ✓
