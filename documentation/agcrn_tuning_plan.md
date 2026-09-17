# AGCRN Hyperparameter Tuning Plan

**Model:** AGCRN — *Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting* (NeurIPS 2020)
**Original repository:** <https://github.com/LeiBAI/AGCRN>
**Task:** `traffic_state_pred` (12-step traffic-flow forecasting)
**Prepared for:** LibCity migration validation / Stage 1 execution
**Status:** Plan (persisted by the tuning executor; the planner agent had no write tool)

---

## 1. Target artifacts

| Artifact | Path |
|---|---|
| Model config to tune | `Bigscity-LibCity/libcity/config/model/traffic_state_pred/AGCRN.json` |
| Model config override (fixed / data alignment) | `Bigscity-LibCity/agcrn_flow12.json` |
| Hyperparameter file (search space) | `Bigscity-LibCity/agcrn_hyper.txt` |
| Hyper-tuning entry point | `Bigscity-LibCity/run_hyper.py` |
| Tuning console log | `Bigscity-LibCity/agcrn_stage1.log` |
| Tuning result dump | `Bigscity-LibCity/hyper.result` |

**Runtime model class:** `Bigscity-LibCity/libcity/model/traffic_flow_prediction/AGCRN.py`
(shadows the functionally-equivalent `traffic_speed_prediction/AGCRN.py`, because
`get_model()` resolves `traffic_flow_prediction` first for `traffic_state_pred`).

---

## 2. Dataset

- **PEMSD4** — the only verified dataset. Data present in `raw_data/PEMSD4/`
  (`PEMSD4.dyna` ≈ 285 MB), 307 sensors, 5-minute interval.
- **PEMSD8** — *not runnable*: `raw_data/PEMSD8/` is absent and would additionally
  require `embed_dim = 2`.

---

## 3. Paper defaults (baseline, from the reference repo)

| Hyperparameter | Value |
|---|---|
| `embed_dim` | 10 (PEMSD4; PEMSD8 uses 2) |
| `rnn_units` | 64 |
| `num_layers` | 2 |
| `cheb_order` | 2 |
| `batch_size` | 64 |
| `learning_rate` | 0.003 |
| `weight_decay` | 0 |
| `max_epoch` | 100 |
| `patience` | 15 |
| `clip_grad_norm` | false |

**Reference target:** PEMSD4 converged **MAE ≈ 19.2** (12-horizon average).

---

## 4. Search space (33 trials total)

### Stage 1 — coarse grid (27 trials)

Full factorial over the three dominant knobs:

| Axis | Values |
|---|---|
| `learning_rate` | 0.001, 0.003, 0.005 |
| `rnn_units` | 32, 64, 128 |
| `embed_dim` | 2, 10, 20 |

→ 3 × 3 × 3 = **27 trials** (`grid_search` / exhaustive search).

### Stage 2 — refinement (6 trials)

Around the Stage-1 best combination:

- `batch_size` ∈ {32, 64} × `weight_decay` ∈ {0.0, 1e-4} (2 × 2 grid = 4 trials)
- `cheb_order` ∈ {2, 3} (one-factor-at-a-time = 2 trials)

---

## 5. Fixed hyperparameters (not tuned)

| Parameter | Value | Notes |
|---|---|---|
| `input_window` | 12 | paper `lag` |
| `output_window` | 12 | paper `horizon` |
| `num_layers` | 2 | paper default |
| `learner` | adam | paper optimizer |
| `lr_epsilon` | 1e-8 | Adam epsilon |
| `train_loss` | masked_mae | equals model's own `calculate_loss` |
| `clip_grad_norm` | false | paper setting |
| `scaler` | standard | z-score, per paper |
| `data_col` | `["traffic_flow"]` | single-channel flow |
| `output_dim` | 1 | single-channel flow |
| `seed` | 10 | run_hyper.py seed |

---

## 6. Rationale

1. `learning_rate`, `embed_dim`, and `rnn_units` are the **dominant knobs** for AGCRN
   accuracy / capacity.
2. `embed_dim` is the only **dataset-specific** hyperparameter reported in the paper
   (10 for PEMSD4, 2 for PEMSD8) — so it deserves explicit coverage.
3. The **staged design keeps the total trial count < 50**, which keeps the wall-clock
   budget tractable while still covering the most impactful axes first.

---

## 7. Execution command (Stage 1)

```bash
cd /home/wangwenrui/shk/AgentCity/Bigscity-LibCity
python run_hyper.py --task traffic_state_pred --model AGCRN --dataset PEMSD4 \
  --config_file agcrn_flow12 --params_file agcrn_hyper.txt \
  --hyper_algo grid_search --max_evals 27 --seed 10
```

### Environment workaround (required in this container)

`libcity/data/dataset/__init__.py` eagerly imports `DeepMapMatchingDataset`, which
imports the ABI-broken `torch_sparse` (`libcudart.so.12` missing **and**
`undefined symbol ...VariableInfo`). Mitigation:

- prepend `LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib`, **and**
- install a `sys.modules['torch_sparse']` stub before `libcity.data.dataset` is imported
  (same technique as the verified `tests/agcrn_e2e_runner.py`).

A 1-epoch end-to-end run already passed (`tests/agcrn_e2e.log`), so the harness path is
known-good.
