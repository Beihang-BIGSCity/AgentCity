# STBP Migration Summary

## Model Overview

| Field | Value |
|-------|-------|
| **Name** | STBP (Spatio-Temporal Backpropagation for Continual Forecasting) |
| **Paper** | "Continual Spatio-Temporal Forecasting" (ICLR 2026) |
| **Original Repository** | https://github.com/panzijie825/STBP |
| **Task** | Traffic speed/flow prediction (`traffic_state_pred`) |
| **Status** | SUCCESS |

---

## Architecture

The STBP model is designed for continual spatio-temporal forecasting and consists of the following components:

- **FConv layers** (x2): Each FConv block combines a standard Linear layer with an FFT-based frequency filter for temporal encoding. Captures both time-domain and frequency-domain patterns.
- **DLGA (Dynamic Linear Graph Attention)**: O(N) linearized graph attention mechanism for efficient spatial dependency modeling across nodes.
- **Pattern Bank**: A learnable `nn.Parameter` of shape `[num_nodes, hidden_channel * 3]` that serves as a continual learning memory, storing per-node contextual patterns across time.
- **Encoder**: Processes the temporally encoded features from FConv.
- **Residual connection**: Stabilizes training by adding the input directly to the encoded output.
- **FC output projection**: Final fully-connected layer projecting to the desired output horizon.

**Total parameters on METR-LA (207 nodes):** 351,352

---

## Key Adaptations for LibCity

| # | Adaptation | Details |
|---|-----------|---------|
| 1 | Removed `torch_geometric` dependency | Replaced with standard PyTorch tensor batching |
| 2 | Simplified continual learning | `pattern_bank` initialized once for the full node count |
| 3 | Input signature | `batch['X']` of shape `[B, T_in, N, F_in]` → reshaped to `[B, N, T_in]` |
| 4 | Output reshaping | `[B*N, T_out]` → `[B, T_out, N, C]` |
| 5 | Base class | Inherited from `AbstractTrafficStateModel` |
| 6 | Loss function | `masked_mae_torch` with `null_val=0.0` |

---

## Files Created / Modified

### Added
- `Bigscity-LibCity/libcity/model/traffic_speed_prediction/STBP.py` — main model implementation
- `Bigscity-LibCity/libcity/config/model/traffic_state_pred/STBP.json` — model hyperparameter defaults

### Modified
- `Bigscity-LibCity/libcity/model/traffic_speed_prediction/__init__.py` — registered STBP in the model registry
- `Bigscity-LibCity/libcity/config/task_config.json` — added STBP to the `allowed_model` list for `traffic_state_pred`

---

## Configuration Defaults

| Parameter | Value |
|-----------|-------|
| `hidden_channel` | 192 |
| `dropout` | 0.0 |
| `input_window` | 12 |
| `output_window` | 12 |
| `batch_size` | 64 |
| `max_epoch` | 100 |
| `learning_rate` | 0.01 (AdamW) |
| `dataset_class` | `TrafficStatePointDataset` |
| `executor` | `TrafficStateExecutor` |
| `evaluator` | `TrafficStateEvaluator` |

---

## Test Results

**Dataset:** METR-LA | **Epochs:** 2 (smoke test)

| Horizon | masked_MAE | masked_MAPE | masked_RMSE |
|---------|-----------|------------|-------------|
| 3       | 2.9478    | 7.82%      | 5.7924      |
| 6       | 3.4356    | 9.81%      | 6.8784      |
| 12      | 4.0457    | 12.27%     | 8.1584      |
| **Average** | **3.4014** | **9.73%** | **6.7413** |

> Note: These results are from a 2-epoch smoke test and are not intended as a full reproduction benchmark. Full training (100 epochs) is expected to yield significantly improved metrics.

---

## Known Limitations

- The original repo's continual-learning mechanisms (multi-stage pattern bank expansion and freezing schedules) are **not fully ported**; the current implementation supports single-dataset forecasting only.
- Validation was performed as a quick smoke test (`max_epoch=2`); full reproduction benchmarks remain pending.
- MAPE values may show instability (`inf`) on datasets with zero or near-zero ground-truth targets; `masked_mae_torch` with `null_val=0.0` mitigates this for MAE/RMSE.

---

## Recommended Next Steps

1. Run full training (100 epochs) and evaluate across multiple seeds for stable benchmark numbers.
2. Extend evaluation to additional datasets (e.g., `PEMS_BAY`, `PEMSD7`).
3. Port the continual-learning training logic (pattern bank expansion/freezing) into a LibCity-compatible executor if continual forecasting is a target use case.
4. Add robust masked/percentage metric handling to prevent `MAPE=inf` from dominating evaluation reports.
