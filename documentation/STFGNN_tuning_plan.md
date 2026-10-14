# STFGNN — Hyperparameter Tuning Plan (PEMSD7)

- **Model**: STFGNN (AAAI 2021) — migrated PyTorch re-authoring in Bigscity-LibCity
- **Task / Dataset**: `traffic_state_pred` / `PEMSD7` (883 nodes, 5-min flow, 12→12)
- **Author**: tuning-planner
- **Date**: 2026-09-20
- **Deliverables**:
  - `documentation/STFGNN_search_space.json` — machine-readable search space + metadata
  - `documentation/STFGNN_search_space.run_hyper.txt` — native `run_hyper.py` params file
  - this plan

---

## 1. Inputs reviewed

| Source | Path | What it gave us |
|---|---|---|
| Migrated model code | `Bigscity-LibCity/libcity/model/traffic_speed_prediction/STFGNN.py` | Which keys are read via `config.get(...)`; loss = Huber(rho); no dropout |
| Current model config | `Bigscity-LibCity/libcity/config/model/traffic_state_pred/STFGNN.json` | Current defaults (lr=1e-3, bs=64, emb=64, filters 64×3, etc.) |
| Repo/paper analysis | `documentation/STFGNN_repo_analysis.md` §5 | Paper hyperparameters from the original MXNet configs |
| Config notes | `documentation/STFGNN_config_notes.md` | Which params the model actually reads; paper-vs-framework gaps |
| Only prior run | `documentation/STFGNN_test_report.md` | 2-epoch smoke: MAE 29.93 / RMSE 43.95 / masked-MAPE 15.01% (NOT converged) |
| LibCity tuning infra | `run_hyper.py`, `libcity/executor/hyper_tuning.py`, `libcity/pipeline/pipeline.py` | Params-file format (`param type value`), grid_search semantics, `max_epoch`/`gpu_id` CLI args |
| LibCity defaults | `libcity/config/executor/TrafficStateExecutor.json`, `libcity/config/data/TrafficStatePointDataset.json` | Supported training keys (weight_decay, lr_scheduler…) and their defaults |
| Paper benchmark (verified) | STFGNN AAAI 2021 Table 2 (arXiv:2012.09641v2), via result-analyzer | PEMSD7 avg-h1–12: MAE 22.07 / RMSE 35.80 / MAPE 9.21% |

---

## 2. Paper vs. current config (the tuning anchor)

| Parameter | Paper (original repo config) | Migrated `STFGNN.json` | Note |
|---|---|---|---|
| `learning_rate` | **1e-3** | 1e-3 | Same base value, **but paper uses Adam + PolyScheduler(pwr=2, warmup≈1 epoch); LibCity runs constant lr (`lr_decay=false`)** |
| `batch_size` | **32** | **64** | **Genuine divergence** → tune |
| `epochs` | 200 | 100 | Tuning budget ≪ 100 (see §4) |
| `filters` | `[[64,64,64]]×3` | same | Paper architecture → freeze |
| `first_layer_embedding_size` | 64 | 64 | Capacity probe → tune |
| `out_layer_dim` | 128 | 128 | Head hidden dim → freeze (low leverage) |
| `act_type` / `module_type` | GLU / individual | same | Freeze (only modes implemented) |
| `strides` | 4 | 4 | Freeze (hard-coupled to fusion graph) |
| Huber `rho` | 1 | 1.0 | Freeze |
| Param init | Xavier(0.0003) | (PyTorch default in re-authoring) | Freeze |

**Why the search targets what it does.** The 2-epoch smoke is ~+35.6% above the paper MAE (29.93 vs 22.07) — almost all of that is *non-convergence* (2 of 100 epochs), not a hyperparameter defect. The two most plausible residual sources are (a) the **training schedule** (constant lr vs. the paper's warm-started poly decay) and (b) **optimization/capacity** settings. With a short per-trial budget, `learning_rate`, `batch_size`, and `first_layer_embedding_size` are the highest-leverage, cheap-to-tune knobs.

---

## 3. Tuned parameters (3) — grid + justification

| Parameter | Grid | Paper / config value | Why this range |
|---|---|---|---|
| `learning_rate` | `[0.0005, 0.001, 0.002, 0.005]` | paper = **1e-3**, config = 1e-3 | Highest-leverage knob. Brackets the verified paper anchor 1e-3 with one value below (0.0005 → stability / higher-batch) and two above (0.002, 0.005 → offset the *short* constant-lr budget, since the paper's warm-up means its early effective lr is much lower than 1e-3). 4 values ≈ one decade. |
| `batch_size` | `[32, 64]` | paper = **32**, config = **64** | Direct paper-vs-config divergence; strongly affects steps/epoch and optimization. Both ends tested. |
| `first_layer_embedding_size` | `[32, 64, 128]` | paper = config = **64** | Input-projection width; the model is already heavy (~13.4M params on 883 nodes), so probing smaller (32) and larger (128) capacity around the verified 64 center is informative. |

**Frozen architecture/optimizer params** (not tuned) are listed with chosen values in §4 and in `STFGNN_search_space.json → frozen_params`.

### Search-space size
`4 × 2 × 3 = **24 trials**` (well under the 50-trial cap). Priority parameters: `learning_rate` first, then `first_layer_embedding_size`, then `batch_size`.

---

## 4. Budget, selection metric, and frozen values

### 4.1 Per-trial budget
- **`per_trial_max_epoch = 5`** (NOT the shipped 100).
- Measured cost ≈ 433 s for a steady-state epoch (epoch 2 of the smoke test) ≈ **~7–8 min/epoch** on PEMSD7 → **~35–50 min/trial** (5 epochs + ~2–3 min test eval; dataset cache is warm).
- **Total ≈ 24 × ~40 min ≈ 14–20 h.** This is the binding constraint and the reason the grid is deliberately coarse.

> ⚠️ **Short-budget caveat.** Tuning at 5 epochs favours faster-converging settings; the winner must be **re-trained at the full epoch budget** (≤100, or until early-stop) before the final paper comparison. If the winning lr is a high value (0.005), re-confirm it does not destabilize over the longer run; the paper anchor (lr=1e-3, bs=32) is the fallback if tuning is inconclusive.

### 4.2 Selection metric
`best_valid_score` = LibCity's **mean de-normalized validation Huber loss** (`STFGNN.calculate_loss()` returns `huber_loss(delta=rho=1)` on scaler-inverse-transformed values — it is NOT masked-MAE; lower is better). Report test MAE / RMSE / masked-MAPE alongside every trial, plus the full 1–12-horizon curve for the best trial. Baseline anchors: paper PEMSD7 avg = MAE 22.07 / RMSE 35.80; **5% stop target = MAE ≤ 23.17, RMSE ≤ 37.59**.

### 4.3 Frozen parameters (fixed for every trial)
| Key | Value | Why frozen |
|---|---|---|
| `input_window` / `output_window` | 12 / 12 | Paper T_in = T_out = 12 |
| `filters` | `[[64,64,64],[64,64,64],[64,64,64]]` | Paper architecture; changing layer count/width is high-risk and coupled to the shrinking time axis (12→9→6→3) |
| `act_type` | `"GLU"` | Paper (gated activation) |
| `module_type` | `"individual"` | Only mode the code implements |
| `use_mask` | `true` | Paper (learnable fusion-graph mask) |
| `temporal_emb` / `spatial_emb` | `true` / `true` | Paper |
| `strides` | `4` | **Hard-coupled** to `construct_adj_fusion(steps=4)` — changing it breaks the fusion graph |
| `rho` | `1.0` | Paper Huber delta |
| `out_layer_dim` | `128` | Paper / adapter; low-leverage head dim |
| `adj_dtw_default` | `"identity"` | No DTW file supplied this round (documented fallback) |
| `scaler` | `"standard"` | Paper (z-score); matches the adapter's loss/runtime assumption |
| `learner` | `"adam"` | Paper optimizer |
| `weight_decay` | `0` | Executor default; regularization deferred to a later stage (model has **no dropout**) |
| `lr_decay` | `false` | Current config; the paper's PolyScheduler is a *documented framework gap* (see §6), deferred to avoid confounding the lr sweep |
| `clip_grad_norm` | `false` | Current config |
| `use_early_stop` | `false` | Fixed 5-epoch budget → deterministic, comparable trials |
| `max_epoch` | `5` | Per-trial tuning budget (see §4.1) |
| `train_rate` / `eval_rate` | `0.7` / `0.1` | LibCity defaults (chronological 7:1:2) |
| `load_external` / `normal_external` / `ext_scaler` | `false` / `false` / `"none"` | PEMSD7 has no external features |
| `add_time_in_day` / `add_day_in_week` | `false` / `false` | STFGNN uses no external time features |

---

## 5. How the executor runs it (directly consumable)

**Option A — `run_hyper.py` (hyperopt, grid_search).** The 24-trial grid equals the exhaustive space size, so `--hyper_algo grid_search` enumerates all combos. Use the provided params file:

```bash
cd /home/wangwenrui/shk/AgentCity/Bigscity-LibCity
# torch_sparse stub workaround required in the `autogen` env (see STFGNN_test_report.md §4)
/home/wangwenrui/opt/miniconda3/envs/autogen/bin/python -c "
import sys, types
sys.modules['torch_sparse'] = types.ModuleType('torch_sparse')
sys.argv = ['run_hyper.py','--task','traffic_state_pred','--model','STFGNN','--dataset','PEMSD7',
            '--params_file','/home/wangwenrui/shk/AgentCity/documentation/STFGNN_search_space.run_hyper.txt',
            '--hyper_algo','grid_search','--max_epoch','5','--gpu','True','--gpu_id','1',
            '--exp_id','stfgnn_tune']
import runpy; runpy.run_path('run_hyper.py', run_name='__main__')
"
```

`documentation/STFGNN_search_space.run_hyper.txt` (native format, `param type value`):
```
learning_rate choice [0.0005, 0.001, 0.002, 0.005]
batch_size choice [32, 64]
first_layer_embedding_size choice [32, 64, 128]
```

**Option B — framework tool.** Pass `search_space = json.load(open('documentation/STFGNN_search_space.json'))['search_space']` to `tune_migration_model(...)`, with `max_epoch=5`.

**Mechanics to note.**
- `run_hyper.py` → `HyperTuning(grid_search)` sets `max_evals = spacesize = 24` automatically.
- `--max_epoch 5` is passed via `other_args`; `learning_rate`/`batch_size`/`first_layer_embedding_size` come from `hyper_config_dict` (higher priority), so they override per-trial correctly.
- **GPU**: recommend `--gpu_id 1` — the executor's recon shows GPU 0 is busy (~53% util, held by an unrelated `fire-shap` job) while GPU 1 is idle. GPU choice does not affect the search space; Lead may override.
- **batch_size caveat**: it is part of LibCity's dataset-cache key, so the first `bs=32` trial (and first `bs=64` reuse) may regenerate the cached arrays once (~1.5 min).

---

## 6. Explicit assumptions & deferred (Stage-2) parameters

**Assumptions (not fabricated):**
1. Paper hyperparameters are those recorded from the original repo configs in `STFGNN_repo_analysis.md` §5; they were not re-read from a paper PDF in this task.
2. Paper PEMSD7 metrics (MAE 22.07 / RMSE 35.80 / MAPE 9.21%) are taken from STFGNN Table 2 as verified by result-analyzer.
3. The 5-epoch per-trial budget is a cost compromise; final numbers require a full-budget re-train.

**Deferred to a later iteration (documented, NOT counted in the 24):**
- **LR schedule**: the paper trains with a `PolyScheduler` (warmup + pwr=2 decay); LibCity currently runs constant lr. Adding a schedule (`lr_decay=true` + `lr_scheduler`) is the most principled next step if the lr sweep alone plateaus. Kept out of Stage 1 to avoid confounding the lr comparison.
- **`weight_decay`** ∈ {0, 1e-5, 1e-4}: the model has **no dropout**, so weight decay is the only regularization lever — relevant once training runs long enough to overfit (13.4M params vs. ~19.7k train windows).
- **`out_layer_dim`** ∈ {64, 128, 256} and filter width/depth: secondary capacity knobs.
- **`adj_dtw_default='spatial'` / a real DTW temporal graph**: this is the paper-faithful fusion graph and could be the single largest accuracy lever beyond training; it is out of scope this round (no DTW file) but worth a dedicated experiment.

**Out of scope**: training/execution (owned by tuning-executor); result analysis & iteration decisions (result-analyzer / Lead).

---

## 7. Summary

| Item | Value |
|---|---|
| Tuned params | `learning_rate`, `batch_size`, `first_layer_embedding_size` |
| Grids | `[0.0005,0.001,0.002,0.005]` × `[32,64]` × `[32,64,128]` |
| **Exact trial count** | **24** (< 50) |
| Per-trial `max_epoch` | **5** |
| Selection metric | val Huber loss (`best_valid_score`; lower better) |
| Stop target (5%) | PEMSD7 test MAE ≤ 23.17 / RMSE ≤ 37.59 (paper 22.07 / 35.80) |
| Est. wall-clock | ~14–20 h (≈40 min/trial) |
