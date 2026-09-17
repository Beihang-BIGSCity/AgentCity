I'll begin the STBP model migration workflow. Starting with **Phase 1: Clone** — delegating to the `repo-cloner` agent to clone and analyze the repository.
Clone the STBP model repository and analyze its structure for migration to the LibCity framework.

Repository URL: https://github.com/panzijie825/STBP
Target directory: ./repos/STBP

Please:
1. Clone the repository to ./repos/STBP
2. Analyze the file structure and identify:
   - Main model class name(s) and file(s)
   - Training script(s)
   - Configuration files
   - Key dependencies (requirements.txt, imports used)
   - Dataset handling code
   - Any adjacency matrix / graph construction code
3. Report the model architecture at a high level (layers, inputs/outputs)
4. Identify what task this model performs (traffic speed/flow/demand prediction?)
5. Note input/output shapes and any special data format requirements

This model is "STBP" from the paper "Continual Spatio-Temporal Forecasting" (ICLR). Report everything needed for the model-adapter agent to port it to LibCity.
The repo-cloner has completed a thorough analysis. Now moving to **Phase 2: Adapt** — delegating to the `model-adapter` agent to create the LibCity-compatible model class.
Adapt the STBP model from the cloned repository at ./repos/STBP to the LibCity framework. The model must be saved to Bigscity-LibCity/libcity/model/traffic_speed_prediction/STBP.py and registered in the appropriate __init__.py.

## Repository Analysis Summary

**Task**: Continual spatio-temporal traffic speed/flow forecasting  
**Model class**: `STBP_Model` in `repos/STBP/src/model/model.py`  
**Supporting classes (all in same file)**: `LayerNorm`, `Mlp`, `Conv`, `DLGA`, `Encoder`, `FConv`, `FreNetwork`

### Architecture Overview
- **FConv layer 1**: Linear + frequency-domain filter (FFT-based), in_channel → hidden_channel
- **Encoder / STModule**: DLGA (Dynamic Linear Graph Attention, O(N) via kernel trick) + FFW (Conv2d MLP)
- **FConv layer 2**: hidden_channel → out_channel  
- **Residual connection**: adds original input  
- **Output projection fc**: nn.Linear(out_channel, y_len) → 12-step predictions
- **Pattern bank**: nn.Parameter [num_nodes, hidden_channel*3] — the continual learning component (gate, scale, memory)

### Input/Output
- **Original forward signature**: `forward(self, data, adj)` where `data` is a `torch_geometric.data.Data` object with `data.x` shape `[B*N, T_in]`
- **LibCity input**: `batch['X']` with shape `[B, T, N, C]`
- **Original output**: `[B*N, 12]`  
- **LibCity expected output**: `[B, T_out, N, C]`

### Key Parameters (from PEMS config)
- `in_channel` (x_len): 12
- `hidden_channel`: 192 (PEMS/CA), 128 (AIR)
- `out_channel`: 12
- `y_len`: 12
- `dropout`: 0.0
- Loss: Masked MAE (null_val=0.0)

### Adapter Notes
1. **Remove torch_geometric dependency**: Replace `torch_geometric.data.Data` batching with standard `[B, N, T]` tensor operations. The graph batching (`to_dense_batch`) is no longer needed.
2. **Continual learning simplification**: For a standard LibCity benchmark, initialize `pattern_bank` once with the full node count (from `data_feature['num_nodes']`). The `expand_adaptive_params` logic can be kept as a method but called once during `__init__`.
3. **Forward signature**: Accept `batch` dict → extract `batch['X']` of shape `[B, T, N, C]` → reshape to `[B, N, T]` (take first channel if C>1 or squeeze).
4. **Output reshape**: Original `[B*N, 12]` → reshape to `[B, N, 12]` → permute/unsqueeze to `[B, T_out=12, N, C=1]` for LibCity compatibility.
5. **adj not used**: The `adj` parameter is passed to `forward()` but not used in any actual computation (DLGA is purely attention-based). You can pass `None` or just ignore it in the LibCity wrapper.
6. **args replacement**: The original `__init__(self, args)` uses an `args` object. In LibCity, use `config` dict and `data_feature` dict instead. Map:
   - `args.model["in_channel"]` → `config.get('in_channel', 12)`
   - `args.model["hidden_channel"]` → `config.get('hidden_channel', 192)`
   - `args.model["out_channel"]` → `config.get('out_channel', 12)`
   - `args.y_len` → `config.get('output_window', 12)`
   - `args.base_node_size` → `data_feature.get('num_nodes', 1)` (use the actual node count, not a base)
   - `args.dropout` → `config.get('dropout', 0.0)`
   - `args.logger` → use `getLogger()`
7. **Scaler**: Get `self._scaler = data_feature.get('scaler')` and use it in `calculate_loss` and `predict`.

### Files to Read
- `repos/STBP/src/model/model.py` — full model source
- `Bigscity-LibCity/libcity/model/traffic_speed_prediction/__init__.py` — where to register STBP
- `Bigscity-LibCity/libcity/model/abstract_traffic_state_model.py` — base class to inherit from

### What to Create
1. `Bigscity-LibCity/libcity/model/traffic_speed_prediction/STBP.py`:
   - Copy all supporting classes (`LayerNorm`, `Mlp`, `Conv`, `DLGA`, `Encoder`, `FConv`, `FreNetwork`) verbatim (removing torch_geometric imports and usage)
   - Adapt `STBP_Model` class to inherit from `AbstractTrafficStateModel`
   - Implement `__init__(self, config, data_feature)`, `predict(self, batch)`, `calculate_loss(self, batch)`, and `forward(self, batch)`
   - Use masked MAE loss: `torch.nn.functional.l1_loss` with masking for null_val=0.0, or implement the mask manually
   
2. Register in `Bigscity-LibCity/libcity/model/traffic_speed_prediction/__init__.py`:
   - Add `from libcity.model.traffic_speed_prediction.STBP import STBP` to the imports

Please read the source files first, then write the adapted model. Make sure the code is clean, well-commented, and follows existing LibCity model conventions (look at another model file like GRU.py or GWNET.py for reference).
Excellent! The model-adapter has also handled the configuration setup. Moving directly to **Phase 4: Test** — delegating to the `migration-tester` agent to run the benchmark test.
Test the STBP model that was just migrated to LibCity. Run a test migration and report results.

## Model Details
- **Model name**: STBP
- **Location**: Bigscity-LibCity/libcity/model/traffic_speed_prediction/STBP.py
- **Task**: traffic_state_pred
- **Config**: Bigscity-LibCity/libcity/config/model/traffic_state_pred/STBP.json
- **Config entry**: Already added to Bigscity-LibCity/libcity/config/task_config.json

## Test Instructions
1. Run test_migration for model STBP using a standard traffic speed dataset (METR_LA or PEMSD7_M or PeMS04 — whichever is available in Bigscity-LibCity/raw_data/)
2. Capture all output, including any errors
3. If the test fails, provide a detailed error diagnosis with:
   - The exact error message and traceback
   - Which file and line caused the error
   - Your diagnosis of the root cause
   - Specific recommended fix (what code to change and how)
4. If the test succeeds, report the metrics (MAE, RMSE, MAPE)

The LibCity framework is at Bigscity-LibCity/. The test_migration tool should be used with model_name="STBP".

Please run the test and report results.
The test passed on the first attempt! Now documenting the migration in the required location.
Create a migration summary document for the STBP model that was successfully migrated to LibCity. Save it to `./documentation/STBP_migration_summary.md`.

## Migration Details

### Model
- **Name**: STBP (Spatio-Temporal Backpropagation for Continual Forecasting)
- **Paper**: "Continual Spatio-Temporal Forecasting" (ICLR 2026)
- **Original repo**: https://github.com/panzijie825/STBP
- **Task**: Traffic speed/flow prediction (traffic_state_pred)

### Architecture
- FConv (Linear + FFT-based frequency filter) layers for temporal encoding
- DLGA (Dynamic Linear Graph Attention) — O(N) linearized graph attention
- Pattern Bank: learnable `nn.Parameter [num_nodes, hidden_channel*3]` for continual learning
- Two FConv layers + Encoder + residual + FC output projection
- Total parameters on METR_LA (207 nodes): 351,352

### Key Adaptations Made
1. Removed torch_geometric dependency — replaced with standard tensor batching
2. Simplified continual learning: pattern_bank initialized once for full node count
3. Forward signature: `batch['X']` of shape `[B, T_in, N, F_in]` → `[B, N, T_in]`
4. Output reshaped: `[B*N, T_out]` → `[B, T_out, N, C]`
5. Inherited from `AbstractTrafficStateModel`
6. Used `masked_mae_torch` loss with `null_val=0.0`

### Files Created/Modified
- `Bigscity-LibCity/libcity/model/traffic_speed_prediction/STBP.py` — main model file
- `Bigscity-LibCity/libcity/model/traffic_speed_prediction/__init__.py` — registered STBP
- `Bigscity-LibCity/libcity/config/model/traffic_state_pred/STBP.json` — model hyperparameters
- `Bigscity-LibCity/libcity/config/task_config.json` — added STBP to allowed_model list

### Test Results (METR_LA, 2 epochs)
| Horizon | masked_MAE | masked_MAPE | masked_RMSE |
|---------|-----------|------------|-------------|
| 3       | 2.9478    | 7.82%      | 5.7924      |
| 6       | 3.4356    | 9.81%      | 6.8784      |
| 12      | 4.0457    | 12.27%     | 8.1584      |
| Average | 3.4014    | 9.73%      | 6.7413      |

### Configuration Defaults
- hidden_channel: 192
- dropout: 0.0
- input_window: 12
- output_window: 12
- batch_size: 64
- max_epoch: 100
- lr: 0.01 (AdamW)
- Dataset class: TrafficStatePointDataset
- Executor: TrafficStateExecutor
- Evaluator: TrafficStateEvaluator

### Status: ✅ SUCCESS

Please create the markdown document at `./documentation/STBP_migration_summary.md` with a well-structured summary of all the above information.
The STBP model migration is complete. Here is the final summary:

---

## Migration Summary

### ✅ STBP — Successful

**Model**: STBP (Continual Spatio-Temporal Forecasting, ICLR 2026)  
**Repository**: https://github.com/panzijie825/STBP  
**Task**: Traffic Speed Prediction (`traffic_state_pred`)

#### Files Created / Modified
| File | Action |
|------|--------|
| `libcity/model/traffic_speed_prediction/STBP.py` | Created |
| `libcity/model/traffic_speed_prediction/__init__.py` | Updated |
| `libcity/config/model/traffic_state_pred/STBP.json` | Created |
| `libcity/config/task_config.json` | Updated |
| `documentation/STBP_migration_summary.md` | Created |

#### Key Adaptations
1. **Removed torch_geometric** — replaced graph-batched `Data` objects with standard `[B, T, N, C]` tensors
2. **Simplified continual learning** — `pattern_bank` initialized once for the full node count at construction
3. **Inherited from `AbstractTrafficStateModel`** with `predict()`, `calculate_loss()`, and `forward()` properly implemented
4. **Loss**: `masked_mae_torch` with `null_val=0.0` and scaler inverse-transform
5. **`adj` parameter dropped** — confirmed unused in any computation (DLGA is purely attention-based)

#### Test Results on METR_LA (2-epoch smoke test)
| Horizon | masked_MAE | masked_MAPE | masked_RMSE |
|---------|-----------|------------|-------------|
| 3 | 2.9478 | 7.82% | 5.7924 |
| 6 | 3.4356 | 9.81% | 6.8784 |
| 12 | 4.0457 | 12.27% | 8.1584 |
| **Avg** | **3.4014** | **9.73%** | **6.7413** |

Model is learning correctly (loss decreasing), architecture verified end-to-end with 351,352 parameters.

---

### No Datasets to Migrate
No datasets were selected for this migration run.

### Recommendations
- Run full 100-epoch training for publication-quality benchmarks
- Test on additional datasets (PeMS04, PeMS08) for broader validation
- Consider porting the multi-year continual learning schedule as a separate LibCity executor extension for complete paper reproduction
