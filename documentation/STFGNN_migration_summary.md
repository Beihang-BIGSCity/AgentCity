# STFGNN — LibCity Migration Summary

**Model**: STFGNN — *Spatial-Temporal Fusion Graph Neural Networks for Traffic Flow Forecasting*
**Paper**: AAAI 2021
**Original repository**: https://github.com/MengzhangLI/STFGNN (commit `a02feee`, depth=1, no submodules)
**Migration date**: 2026-09-20
**Task**: `traffic_state_pred` (traffic flow prediction, 5-min interval, 12-step → 12-step)

---

## Test status: ✅ PASS (2-epoch smoke test)

| Metric (mean over horizons 1–12) | Value |
|---|---|
| MAE | **29.93** |
| RMSE | **43.95** |
| masked-MAPE | **15.01 %** |
| R² (horizon 1 → 12) | **0.970 → 0.913** |

- **Dataset**: `PEMSD7` (883 nodes), `traffic_state_pred`.
- **Command** (run from `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity`):
  ```bash
  /home/wangwenrui/opt/miniconda3/envs/autogen/bin/python run_model.py \
      --task traffic_state_pred --model STFGNN --dataset PEMSD7 --max_epoch 2
  ```
- **Log**: `/home/wangwenrui/shk/AgentCity/logs/STFGNN_test.log`
- **Caveat**: this is a **2-epoch smoke test proving pipeline correctness, not a full-convergence result** (the shipped config uses `max_epoch=100`). Loss decreased monotonically (train 62.62 → 31.26; val 31.43 → 29.49); metrics are sane but are not the paper's accuracy.

> ⚠️ **Key migration caveat — the original code is MXNet symbolic, not PyTorch.**
> `models/stsgcn_4n_res.py` is written in the Apache MXNet symbolic API (`mx.sym.*`). There is **no `nn.Module` and no exportable weights**, so this migration is a **full PyTorch re-authoring** of the architecture, **not a weight port**. The MXNet repo is not runnable in the target environment; only the architecture (and its graph-construction utilities) were reproduced.

---

## 1. What was migrated & final file paths

| Artifact | Action | Absolute path |
|---|---|---|
| Model source (PyTorch re-authoring) | **Created** | `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_speed_prediction/STFGNN.py` |
| Model registration | **Modified** | `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_speed_prediction/__init__.py` (`from ...STFGNN import STFGNN`; `"STFGNN"` added to `__all__`) |
| Model config (hyperparameters) | **Created** | `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traffic_state_pred/STFGNN.json` |
| Task wiring | **Modified** | `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json` |
| Smoke test | **Created** | `/home/wangwenrui/shk/AgentCity/tests/stfgnn_smoke.py` (project root; run `python tests/stfgnn_smoke.py`) |

- **Registered class name**: `STFGNN`
- **Base class**: `libcity.model.abstract_traffic_state_model.AbstractTrafficStateModel`
- **Resolution**: `libcity.utils.utils.get_model` looks up `config['model']` on `libcity.model.traffic_flow_prediction`, then falls back to `libcity.model.traffic_speed_prediction` (where `STFGNN` lives). Set `"model": "STFGNN"` in the task config.
- **`task_config.json` wiring** (inside the `"traffic_state_pred"` object):
  1. `"STFGNN"` appended to `allowed_model` (after `"MVGRL"`).
  2. Model block added:
     ```json
     "STFGNN": {
         "dataset_class": "TrafficStatePointDataset",
         "executor": "TrafficStateExecutor",
         "evaluator": "TrafficStateEvaluator"
     }
     ```
  No dataset-list change was needed — `PEMSD7` is already in `traffic_state_pred.allowed_dataset`.

---

## 2. Architecture summary (as implemented in PyTorch)

Faithful re-authoring of the MXNet symbol graph, mirroring the already-migrated predecessor `libcity/model/traffic_flow_prediction/STSGCN.py` (LibCity batch-first `(B, T, N, C)` convention) and extended with STFGNN's two innovations (the temporal-conv residual branch and the `(4N, 4N)` fusion graph).

```
(B, 12, N, C)                              # input; only channel 0 (flow) is used
  └─ First_FC: Linear(C -> 64) + ReLU                  # first_layer_embedding
  └─ STSGCL × 3   (filters [[64,64,64]] × 3, 'individual'):
        • PositionEmbedding: learnable temporal (1,T,1,C) + spatial (1,1,N,C)
        • Gated dilated temporal conv (GLU), kernel (1,2), dilation (1, strides-1)
          → residual branch (B, T-3, N, C')
        • for each sliding window of `strides` (=4) time steps:
              reshape → (B, 4N, C) → STSGCM(fusion-GCN) → (B, N, C')
        • residual add:  STSGCM-concat  +  temporal-conv branch
  └─ Output head (per horizon step): Linear(T*C -> 128) + ReLU -> Linear(128 -> 1)
  └─ concat over 12 steps → (B, 12, N, output_dim)
```

- **STSGCM** = `len(filters)` stacked GCN ops; each GCN op is `A @ X` then GLU (`lhs * sigmoid(rhs)`); outputs are cropped to the **middle N vertices** `[N:2N]` and combined with element-wise **max**.
- **Temporal conv**: `Conv2d(kernel=(1,2), dilation=(1, strides-1))` on the `(B, C, N, T)` layout → cuts `T` by `strides-1 = 3` per layer, matching the STSGCM window reduction (12 → 9 → 6 → 3).
- **Loss**: Huber, `delta = rho` (paper `rho = 1`), computed on scaler-inverse-transformed predictions/targets (LibCity convention).

### Fusion graph / adjacency (`(4N, 4N)`) — the defining innovation
`construct_adj_fusion(a_spatial, a_dtw, steps=4)` is a verbatim reproduction of `utils_4n0_3layer_12T_res.py::construct_adj_fusion`:

```
        blk0   blk1   blk2   blk3
blk0    A_dtw   I      I      A_dtw
blk1    I       A_sp   I      I
blk2    I       I      A_sp   I
blk3    A_dtw   I      I      A_dtw      (I = identity band; self-loops forced = 1)
```

- **Spatial graph `A_sp`**: taken from LibCity `data_feature['adj_mx']`, binarized to connectivity `(adj_mx != 0)` (paper uses a binary connectivity graph; inf/nan dropped). Falls back to identity if `adj_mx` is missing/mis-shaped — never hard-fails.
- **Temporal (DTW) graph `A_dtw`**: resolved from `config['adj_dtw']` / `adj_dtw_filename` / `adj_dtw_file`, else `adj_dtw_default` (`'identity'` default → `np.eye(N)`, or `'spatial'`). A helper `build_dtw_temporal_adjacency(...)` (DTW order=1, lag=12, top-1 % neighbours, symmetrized, self-loops) matching `data/fastDTW_adj_gen.py` is provided but **not called by default**.
- **`use_mask=True`** makes edge strengths learnable: an `nn.Parameter` initialized to the binary support `(fusion != 0)` is multiplied with the fixed fusion graph inside `forward` (no `retain_graph=True` needed).

### Layout / I/O
- **Input** `batch['X']` `(B, input_window, N, feature_dim)`; extra channels cropped to `feature_dim` (paper: channel 0 = flow).
- **Internal permutations**: temporal conv on `(B, C, N, T)`; each sliding window reshaped to `(B, 4N, C)` for the `(4N, 4N)` GCN; output head reshapes `(B, T, N, C) → (B, N, T, C) → (B, N, T*C)`.
- **Output** `forward`/`predict` return `(B, output_window, N, output_dim)`; a single forward pass produces all 12 steps (no autoregression).
- **Normalization**: model works in the dataset's z-scored scale; `calculate_loss` applies `scaler.inverse_transform` before Huber loss; the evaluator de-normalizes for metrics.

Parameter count for N=20, T=12, `filters=[[64,64,64]]×3`: **808,908** (smoke config); the full PEMSD7 run (N=883) reports **13,443,228** parameters.

---

## 3. Resolved hyperparameters (`STFGNN.json`)

Every model-specific key below is read by `STFGNN.py` via `config.get(...)` and defaults reproduce the paper.

| Key | Value | Source |
|---|---|---|
| `input_window` | `12` | Paper (T_in = 12) |
| `output_window` | `12` | Paper (T_out = 12) |
| `first_layer_embedding_size` | `64` | Paper / adapter default |
| `filters` | `[[64,64,64],[64,64,64],[64,64,64]]` | 3 STSGCL layers × 3 GCN filters of 64 |
| `act_type` | `"GLU"` | Paper (gated activation) |
| `module_type` | `"individual"` | Only mode the MXNet source implements |
| `use_mask` | `true` | Paper (learnable adjacency mask) |
| `temporal_emb` / `spatial_emb` | `true` / `true` | Paper |
| `strides` | `4` | Paper; **must be 4** — the `(4N,4N)` fusion graph is defined for 4 time steps |
| `rho` | `1.0` | Paper (Huber `delta`) |
| `out_layer_dim` | `128` | Paper / adapter default |
| `adj_dtw_default` | `"identity"` | Adapter default this round (no DTW graph supplied) |

**Standard LibCity pipeline keys** (consumed by dataset/executor, not invented model params; mirrored from `STGCN.json`/`STSGCN.json`): `batch_size=64`, `scaler="standard"`, `load_external=false`, `normal_external=false`, `ext_scaler="none"`, `add_time_in_day=false`, `add_day_in_week=false`, `max_epoch=100`, `learner="adam"`, `learning_rate=0.001`, `lr_decay=false`, `clip_grad_norm=false`, `use_early_stop=false`. Only `learner="adam"` follows the paper directly.

**Intentionally omitted**: `dropout` (the model has no dropout anywhere — GLU gating + residuals only); unrelated architecture keys (`Ks`/`Kt`/`blocks`, `nb_block`/`K`, `bidir_adj_mx`, …); `device` (injected at runtime by `ConfigParser._init_device()`); `adj_dtw`/`adj_dtw_filename` (left unset → documented identity fallback).

---

## 4. Dataset choice

**Chosen dataset: `PEMSD7`** (`Bigscity-LibCity/raw_data/PEMSD7/` — atomic files `.geo`, `.rel`, `.dyna`, `config.json` present).
Its own `config.json` already delivers exactly what STFGNN expects: `data_col=["traffic_flow"]` (single-channel **flow**), `output_dim=1`, `set_weight_link_or_dist="link"` + `init_weight_inf_or_zero="zero"` (binary connectivity adjacency — what `_build_spatial_adj` binarizes to), `time_intervals=300` (5-min). **No dataset override was needed.**
*Fallback:* `METR_LA` (single-channel **speed**, the `test_migration` tool default) also loads cleanly as a smoke-test alternative, but the paper's task is flow.

---

## 5. Test evidence

**Verdict: PASS** — the model loads, trains 2 full epochs, saves checkpoints, and produces valid metrics. No model-code, config, or dataset defect was found (the only defect was environmental — see §6).

| Item | Result |
|---|---|
| Dataset loads (atomic → arrays) | 883 nodes; train `(19741,12,883,1)`, eval `(2820,12,883,1)`, test `(5640,12,883,1)` |
| Model constructs & runs on GPU | 13,443,228 params, `cuda:0` |
| Training epochs | 2/2 (`max_epoch=2`) |
| Convergence signal | train_loss 62.62 → 31.26; val_loss 31.43 → 29.49 (decreasing) |
| Test evaluation | per-horizon MAE/RMSE/R² produced; CSV written |
| Checkpoint | `STFGNN_PEMSD7.m` (211 MB) |

**Per-horizon metrics (best model = epoch 1), PEMSD7:**

| H | MAE | RMSE | masked_MAPE | R² |
|---|---|---|---|---|
| 1 | 21.77 | 32.51 | 0.1046 | 0.9696 |
| 2 | 23.84 | 35.54 | 0.1118 | 0.9637 |
| 3 | 25.46 | 37.81 | 0.1240 | 0.9589 |
| 4 | 26.93 | 39.72 | 0.1324 | 0.9547 |
| 5 | 27.91 | 41.62 | 0.1414 | 0.9502 |
| 6 | 29.58 | 43.60 | 0.1486 | 0.9454 |
| 7 | 30.49 | 44.91 | 0.1502 | 0.9421 |
| 8 | 31.99 | 46.63 | 0.1681 | 0.9376 |
| 9 | 33.45 | 48.49 | 0.1760 | 0.9325 |
| 10 | 33.80 | 49.43 | 0.1636 | 0.9299 |
| 11 | 35.70 | 52.00 | 0.1761 | 0.9224 |
| 12 | 38.21 | 55.08 | 0.2048 | 0.9130 |
| **mean** | **29.93** | **43.95** | **0.1501 (15.01 %)** | 0.9696 → 0.9130 |

Notes: MAE/RMSE are the plain means over horizons. `MAPE` is reported as `inf` by the evaluator (PEMSD7 flow contains true zeros → undefined relative error); the zero-safe **masked_MAPE** is the meaningful metric (15.01 %). These are **2-epoch smoke numbers** — the point of the test is pipeline correctness, not convergence.

**Artifacts:**
| Kind | Path |
|---|---|
| Raw run log | `/home/wangwenrui/shk/AgentCity/logs/STFGNN_test.log` |
| Eval metrics CSV | `Bigscity-LibCity/libcity/cache/STFGNN_test/evaluate_cache/2026_09_20_12_30_12_STFGNN_PEMSD7.csv` |
| Predictions | `Bigscity-LibCity/libcity/cache/STFGNN_test/evaluate_cache/2026_09_20_12_29_01_STFGNN_PEMSD7_predictions.npz` |
| Checkpoints | `Bigscity-LibCity/libcity/cache/STFGNN_test/model_cache/STFGNN_PEMSD7.m` (+ `_epoch0.tar`, `_epoch1.tar`) |

---

## 6. Limitations & follow-ups

1. **Framework gap — MXNet → PyTorch re-authoring (architectural, inherent).** The source is MXNet symbolic with no `nn.Module` and no exportable weights; weights **cannot** be ported. Correctness relies on a faithful re-implementation of the op order, not on numerical equivalence to the original trained model. This is a **round-1 (architecture) migration**; no accuracy parity against the paper was claimed or tested.
2. **Environment blocker (found during testing; NOT a model/config/dataset defect).** In the `autogen` env, `torch_sparse==0.6.18+pt24cu121` (built for torch 2.4 / CUDA 12.1) sits against `torch 2.6.0+cu118`, so importing it raises an **`OSError` (ABI mismatch)**. `libcity/data/dataset/__init__.py` imports `deep_map_matching_dataset` eagerly, and that module guards the import with `except ImportError` only — an `OSError` propagates and **aborts dataset loading for every model**. It was worked around by pre-registering an empty `torch_sparse` stub (safe: `torch_sparse` is map-matching-only, irrelevant to `traffic_state_pred`). **Recommended permanent fixes** (for the Lead to route): (a) reinstall a torch-2.6/cu118-matching `torch_sparse`, or remove `torch_sparse` from the `autogen` env; and/or (b) broaden the guard in `libcity/data/dataset/deep_map_matching_dataset.py` to `except (ImportError, OSError)`. Until fixed, the model only runs with the stub.
3. **Smoke run, not full convergence.** The verified result is a 2-epoch run; the shipped config sets `max_epoch=100`. Paper-accurate training curves / final accuracy were **not** produced this round.
4. **No dataset in scope this round → temporal graph defaults to identity.** With no DTW file supplied, `A_dtw` falls back to `np.eye(N)`, so the fusion graph is not the full paper graph (which uses a train-derived DTW temporal graph, top-1 % neighbours, order=1, lag=12). Supply `adj_dtw`/`adj_dtw_filename` (or run `build_dtw_temporal_adjacency`) for the paper-faithful fusion graph.
5. **`strides` is coupled and fixed at 4.** The `(4N,4N)` fusion builder is defined for `steps=4`; changing `strides` without re-deriving the fusion-graph block layout will break the model.
6. **`module_type` limited to `'individual'`.** The MXNet source only implements `individual`; any other value logs a warning and falls back to `individual`.
7. **Sequence split/normalization delegated** to LibCity's dataset pipeline (chronological 6:2:2, z-score from train split), not reproduced inside the model — consistent with framework convention.

---

## 7. Upstream artifacts (source of this summary)

| Doc | Author |
|---|---|
| `documentation/STFGNN_repo_analysis.md` | repo-cloner |
| `documentation/STFGNN_adapter_notes.md` | model-adapter |
| `documentation/STFGNN_config_notes.md` | config-migrator |
| `documentation/STFGNN_test_report.md` | migration-tester |

*This summary was verified against the actual migrated files (`STFGNN.py`, `traffic_speed_prediction/__init__.py`, `STFGNN.json`, `task_config.json`) and the eval CSV; all metrics were recomputed and match the test report.*
