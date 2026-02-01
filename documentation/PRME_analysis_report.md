## Repository: PRME (Personalized Ranking Metric Embedding)

### Overview
- **URL**: https://github.com/flaviovdf/prme
- **Cloned to**: /home/wangwenrui/shk/AgentCity/repos/PRME
- **Paper**: "Personalized Ranking Metric Embedding for Next New POI Recommendation" - IJCAI 2015
- **Authors**: Shanshan Feng, Xutao Li, Yifeng Zeng, Gao Cong, Yeow Meng Chee, Quan Yuan
- **Implementation**: Python/Cython (optimized for performance)

---

## Key Files

### Model Implementation
- **Main Model**: `/home/wangwenrui/shk/AgentCity/repos/PRME/prme/prme.pyx`
  - Cython implementation for performance
  - Core functions: `sgd()`, `do_iter()`, `compute_dist()`, `compute_cost()`
  - Model class: No explicit class, functional implementation

### Training Scripts
- **Training**: `/home/wangwenrui/shk/AgentCity/repos/PRME/main.py`
  - Main entry point for model training
  - Uses `learn()` function from `prme/__init__.py`
- **Cross-validation**: `/home/wangwenrui/shk/AgentCity/repos/PRME/cross_val.py`
  - Grid search over hyperparameters

### Evaluation
- **MRR Computation**: `/home/wangwenrui/shk/AgentCity/repos/PRME/mrr.py`
  - Mean Reciprocal Rank evaluation
  - Cython implementation: `/home/wangwenrui/shk/AgentCity/repos/PRME/prme/mrr.pyx`

### Data Loading
- **Data I/O**: `/home/wangwenrui/shk/AgentCity/repos/PRME/prme/dataio.py`
  - `initialize_trace()`: Loads trajectory data
  - `save_model()`: Saves model to HDF5 format

### Configuration
- **Setup**: `/home/wangwenrui/shk/AgentCity/repos/PRME/setup.py`
  - Build configuration for Cython extensions
  - No separate requirements.txt file

---

## Dependencies

### Core Dependencies
- **Python**: 2.7+ (legacy code, needs porting to Python 3)
- **Cython**: For compiled extensions
- **NumPy**: Numerical operations
- **Pandas**: HDF5 storage and data handling

### Optional Dependencies
- **plac**: For command-line argument parsing in mrr.py

### Build Requirements
- **GCC**: For compiling Cython/C code
- **OpenMP**: For parallel processing
- **Compiler flags**: SSE, SSE2 optimizations

---

## Model Architecture

### Embedding Matrices
1. **XG_ok**: Geographic/sequential embeddings (objects × latent_factors)
   - Captures sequential transitions between POIs
2. **XP_ok**: Personalized embeddings for objects (POIs) (objects × latent_factors)
   - POI-specific features
3. **XP_hk**: Personalized embeddings for users (users × latent_factors)
   - User-specific preferences

### Key Hyperparameters
- **nk (num_latent_factors)**: Dimensionality of embeddings
- **rate**: Learning rate (default: 0.005)
- **regularization**: L2 regularization (default: 0.03)
- **alpha**: Balance between personalized and geographic distance (default: 0.02)
- **tau**: Time threshold for transition types (default: 3 hours = 10800 seconds)

### Distance Computation
The model computes distance as:
```
distance = α * ||XP_ok[d] - XP_hk[h]||² + (1-α) * ||XG_ok[d] - XG_ok[s]||²
```
where:
- h = user (hyper)
- s = source POI
- d = destination POI
- α adjusts based on time threshold τ

---

## Data Format

### Input Format (Tab-separated)
```
dt \t user \t from_poi \t to_poi
```
- **dt**: Time spent at `from_poi` before transitioning to `to_poi`
- **user**: User identifier
- **from_poi**: Source POI ID
- **to_poi**: Destination POI ID

### Example
```
1800.5  user_123  poi_A  poi_B
3600.0  user_456  poi_C  poi_D
```

### Data Processing
- `hyper2id`: Maps user strings to integer IDs
- `obj2id`: Maps POI strings to integer IDs
- `seen`: Dictionary tracking (user, source_poi) → set of visited destination POIs
- `Trace`: Numpy array of [user_id, source_poi_id, dest_poi_id]
- `dts`: Numpy array of time deltas

---

## Training Process

### SGD Algorithm (Stochastic Gradient Descent)
1. **Negative Sampling**: For each transition (h, s, d_old):
   - Sample random POI `d_new` not in seen[(h, s)]
   
2. **Ranking Objective**: Minimize pairwise ranking loss
   - Prefer `d_old` (actual destination) over `d_new` (random negative)
   
3. **Gradient Updates**: Update all three embedding matrices
   - XP_hk[h]: User embedding
   - XP_ok[d_new], XP_ok[d_old]: POI embeddings
   - XG_ok[s], XG_ok[d_new], XG_ok[d_old]: Geographic embeddings
   
4. **Time-based Alpha Adjustment**:
   - If dt > τ: α = 1.0 (purely personalized, long gap)
   - If dt ≤ τ: α = configured value (mixed personalized + geographic)

5. **Fixed Iterations**: 1000 iterations

---

## Model Output

### Saved Model (HDF5 format via Pandas)
Keys in the HDF5 store:
- `XG_ok`: Geographic embeddings
- `XP_ok`: POI embeddings  
- `XP_hk`: User embeddings
- `hyper2id`: User ID mapping
- `obj2id`: POI ID mapping
- `num_topics`: Number of latent factors
- `rate`, `regularization`, `alpha`, `tau`: Hyperparameters
- `cost_train`, `cost_val`: Training/validation costs
- `training_time`: Total training time

---

## Evaluation

### Mean Reciprocal Rank (MRR)
- Computes distance to all candidate POIs
- Ranks actual destination among all POIs
- MRR = 1 / rank
- Parallelized with Cython/OpenMP

---

## Implementation Challenges for LibCity Migration

### 1. **Cython Dependencies**
- Core model logic in `.pyx` files (Cython)
- Needs conversion to pure PyTorch for LibCity
- Custom random number generator in C

### 2. **Python 2.7 Legacy Code**
- Uses `xrange`, print as statement
- Needs Python 3 compatibility updates

### 3. **Data Format Mismatch**
- Custom tab-separated format (dt, user, from, to)
- LibCity uses different trajectory formats
- Need adapter for LibCity's data loaders

### 4. **No Explicit Model Class**
- Functional programming style
- LibCity expects nn.Module subclass
- Need to encapsulate in PyTorch model

### 5. **Custom Training Loop**
- Fixed 1000 iterations
- No early stopping or checkpointing
- LibCity has standardized training framework

### 6. **HDF5 Storage**
- Uses Pandas HDFStore for model persistence
- LibCity typically uses PyTorch `.pth` or pickled models

### 7. **Negative Sampling**
- Custom negative sampling per user-source pair
- Need to integrate with LibCity's batching

### 8. **Time-based Features**
- Uses time deltas (dt) and threshold (τ)
- Need to ensure LibCity datasets provide temporal info

---

## Structure Notes

### Code Organization
```
repos/PRME/
├── prme/                    # Main package
│   ├── __init__.py         # learn() function
│   ├── prme.pyx            # Core SGD implementation (Cython)
│   ├── dataio.py           # Data loading/saving
│   ├── mrr.pyx             # MRR evaluation (Cython)
│   └── myrandom/           # Custom RNG
│       ├── random.pyx      # Cython random wrapper
│       └── randomkit.c     # C random implementation
├── main.py                 # Training entry point
├── cross_val.py            # Hyperparameter tuning
├── mrr.py                  # Evaluation script
├── setup.py                # Build configuration
└── Readme.md              # Documentation
```

### Key Algorithmic Insights
1. **Dual Embeddings**: Separate geographic (XG) and personalized (XP) embeddings
2. **Time-aware**: Adjusts α based on time threshold to handle short vs long gaps
3. **Pairwise Ranking**: Uses BPR-like objective (positive vs negative)
4. **User Context**: Tracks seen POIs per (user, source) pair for negative sampling

---

## Migration Recommendations

### Phase 1: Pure Python Port
1. Convert Cython (.pyx) to pure Python
2. Replace custom RNG with PyTorch random
3. Update Python 2 → 3 syntax

### Phase 2: PyTorch Implementation
1. Create `PRME(nn.Module)` class
2. Implement embeddings as `nn.Embedding`
3. Rewrite distance computation in PyTorch
4. Implement custom loss function (pairwise ranking)

### Phase 3: LibCity Integration
1. Adapt data loader for LibCity trajectory format
2. Integrate with LibCity executor/trainer
3. Add configuration JSON for hyperparameters
4. Implement LibCity-compatible predict() method

### Phase 4: Testing
1. Validate against original implementation
2. Ensure MRR scores match
3. Test on LibCity benchmark datasets

---

## Additional Notes

- **No GPU Support**: Original implementation is CPU-only (Cython/OpenMP)
- **No Batching**: Processes one transition at a time
- **Scalability**: May need mini-batching for large datasets in LibCity
- **Reproducibility**: Uses custom RNG, need to ensure deterministic behavior with PyTorch

---

## References
Paper: Shanshan Feng et al., "Personalized Ranking Metric Embedding for Next New POI Recommendation", IJCAI 2015
