# STFGNN Migration Test — Verdict Report

- **Task**: `#stfgnn-test` — end-to-end verification that the migrated **STFGNN** model loads and trains in LibCity.
- **Author**: migration-tester
- **Date**: 2026-09-20
- **Model / Dataset / Task**: `STFGNN` / `PEMSD7` / `traffic_state_pred`
- **Verdict**: ✅ **PASS** — the migrated model loads, trains for 2 full epochs, and completes test evaluation with sane metrics.

---

## 1. Result summary

| Item | Result |
|---|---|
| Dataset loads (atomic files → arrays) | **PASS** — 883 nodes; train `(19741,12,883,1)`, eval `(2820,12,883,1)`, test `(5640,12,883,1)` |
| Model constructs & runs on GPU | **PASS** — 13,443,228 params, cuda:0 |
| Training epochs completed | **PASS** — 2/2 (`max_epoch=2`) |
| Convergence signal | **PASS** — train_loss 62.62 → 31.26; val_loss 31.43 → 29.49 (decreasing) |
| Test evaluation produced metrics | **PASS** — per-horizon MAE/RMSE/R², CSV written |
| Model checkpoint saved | **PASS** — `STFGNN_PEMSD7.m` (211 MB) |

No model-code, config, or dataset defects were found. One **environment** defect (unrelated to STFGNN) had to be worked around — see §4.

---

## 2. Exact command(s)

Environment: the `autogen` conda interpreter (has torch). Run from the LibCity root.

> ⚠️ A pre-registered empty `torch_sparse` stub is injected at process start (see §4 for why).
> `torch_sparse` is used **only** by the map-matching dataset and is irrelevant to `traffic_state_pred`.

```bash
cd /home/wangwenrui/shk/AgentCity/Bigscity-LibCity
/home/wangwenrui/opt/miniconda3/envs/autogen/bin/python -c "
import sys, types
sys.modules['torch_sparse'] = types.ModuleType('torch_sparse')   # env workaround, see §4
sys.argv=['run_model.py','--task','traffic_state_pred','--model','STFGNN','--dataset','PEMSD7',
          '--gpu','True','--gpu_id','0','--max_epoch','2','--exp_id','STFGNN_test']
import runpy; runpy.run_path('run_model.py', run_name='__main__')
" 2>&1 | tee /home/wangwenrui/shk/AgentCity/logs/STFGNN_test.log
```

Equivalent "plain" command (fails in this env only because of the §4 blocker, not because of STFGNN):

```bash
cd /home/wangwenrui/shk/AgentCity/Bigscity-LibCity
/home/wangwenrui/opt/miniconda3/envs/autogen/bin/python run_model.py \
    --task traffic_state_pred --model STFGNN --dataset PEMSD7 --gpu True --gpu_id 0 --max_epoch 2
```

Note: the framework `test_migration(...)` tool runs a bare `python`, which is **not** the torch interpreter, so it cannot be used here; the `autogen` interpreter is required.

---

## 3. Key log lines (raw log: `./logs/STFGNN_test.log`)

```
12:03:20 INFO - Loaded file PEMSD7.geo, num_nodes=883
12:03:20 INFO - Loaded file PEMSD7.rel, shape=(883, 883)
12:03:20 INFO - Loading file PEMSD7.dyna
12:03:55 INFO - Loaded file PEMSD7.dyna, shape=(28224, 883, 1)
12:04:10 INFO - Dataset created
12:04:10 INFO - train x: (19741, 12, 883, 1), y: (19741, 12, 883, 1)
12:04:10 INFO - eval  x: (2820, 12, 883, 1),  y: (2820, 12, 883, 1)
12:04:10 INFO - test  x: (5640, 12, 883, 1),  y: (5640, 12, 883, 1)
12:10:14 INFO - Total parameter numbers: 13443228
12:10:14 INFO - You select `adam` optimizer.
12:10:14 INFO - Start training ...
12:10:14 INFO - num_batches:309
12:20:31 INFO - Epoch [0/2] train_loss: 62.6151, val_loss: 31.4282, lr: 0.001000, 616.90s
12:20:31 INFO - Val loss decrease from inf to 31.4282, saving to .../STFGNN_PEMSD7_epoch0.tar
12:27:44 INFO - Epoch [1/2] train_loss: 31.2607, val_loss: 29.4863, lr: 0.001000, 432.99s
12:27:44 INFO - Trained totally 2 epochs, average train time is 488.975s, average eval time is 35.968s
12:27:45 INFO - Start evaluating ...
12:30:12 INFO - Note that you select the single mode to evaluate!
12:30:12 INFO - Evaluate result is saved at .../2026_09_20_12_30_12_STFGNN_PEMSD7.csv
```

---

## 4. Environment blocker (found & diagnosed) — component: **environment, NOT model/config/dataset**

**Symptom (first, un-worked-around attempt):**
```
libcity/data/dataset/deep_map_matching_dataset.py:43: from torch_sparse import SparseTensor
... torch_sparse/__init__.py: torch.ops.load_library(spec.origin)
OSError: libcudart.so.12: cannot open shared object file: No such file or directory
```
After adding the env's own `nvidia/cuda_runtime/lib` to `LD_LIBRARY_PATH` (where `libcudart.so.12` actually lives), it then failed with a torch ABI error:
```
OSError: .../torch_sparse/_spmm_cuda.so: undefined symbol:
        _ZN5torch8autograd12VariableInfoC1ERKN2at6TensorE
```

**Root cause:** in the `autogen` env, `torch_sparse==0.6.18+pt24cu121` (built for **torch 2.4 / CUDA 12.1**) was installed against **torch 2.6.0+cu118** (CUDA 11.8). The `.so` is both CUDA- and torch-ABI-incompatible.

**Why it blocks everything:** `libcity/data/dataset/__init__.py` imports `deep_map_matching_dataset` eagerly. That module wraps the import as
```python
try:
    from torch_sparse import SparseTensor
    HAS_TORCH_SPARSE = True
except ImportError:          # <-- catches ImportError only
    HAS_TORCH_SPARSE = False
```
but the broken install raises **`OSError`**, which is *not* an `ImportError`, so it propagates and aborts dataset import → **no model of any kind can run in this env.**

**Workaround used (no repo code changed):** pre-register an empty `torch_sparse` module so the guarded `from torch_sparse import SparseTensor` raises `ImportError` and the code cleanly falls back to `HAS_TORCH_SPARSE=False`. This is safe for STFGNN/traffic-state (torch_sparse is map-matching-only).

**Recommended permanent fixes (Leader to route / decide):**
1. *Environment fix (preferred):* in the `autogen` env, install a CUDA/torch-matching `torch_sparse` (e.g. built for torch 2.6 + cu118 / `pt26cu118`), or remove `torch_sparse` from `autogen` entirely (the guarded import then works and no run is affected). Other conda envs on this box have **no** `torch_sparse` but also lack pandas/gensim, so none is a drop-in alternative.
2. *Robustness fix (optional, LibCity upstream):* broaden the guard in `libcity/data/dataset/deep_map_matching_dataset.py` to `except Exception:` (or `except (ImportError, OSError):`) so a broken optional dependency degrades gracefully instead of aborting all dataset loading.

---

## 5. Test metrics (PEMSD7, best model = epoch 1)

Evaluated in LibCity `single` mode, de-normalized (standard scaler). Horizon = 1..12 (5-min steps). Full table: `Bigscity-LibCity/libcity/cache/STFGNN_test/evaluate_cache/2026_09_20_12_30_12_STFGNN_PEMSD7.csv`.

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
| **mean** | **29.93** | **43.95** | **0.1501 (15.01%)** | — (0.9696 → 0.9130) |

- **MAE** (mean over horizons): **29.93**
- **RMSE** (mean over horizons): **43.95**
- **MAPE**: reported as `inf` by the evaluator (PEMSD7 traffic-flow contains true zeros → undefined relative error); the zero-safe **masked_MAPE** mean is **15.01%**.
- These are 2-epoch smoke numbers (config `max_epoch=100` for a real run); the point of this task is pipeline correctness, not convergence to paper accuracy. Metrics are nevertheless sane and the loss/val-loss decrease monotonically.

---

## 6. Artifacts

| Kind | Path |
|---|---|
| Raw run log | `/home/wangwenrui/shk/AgentCity/logs/STFGNN_test.log` (299 lines) |
| Eval metrics CSV | `Bigscity-LibCity/libcity/cache/STFGNN_test/evaluate_cache/2026_09_20_12_30_12_STFGNN_PEMSD7.csv` |
| Predictions | `Bigscity-LibCity/libcity/cache/STFGNN_test/evaluate_cache/2026_09_20_12_29_01_STFGNN_PEMSD7_predictions.npz` |
| Checkpoints | `Bigscity-LibCity/libcity/cache/STFGNN_test/model_cache/STFGNN_PEMSD7.m` (+ `_epoch0.tar`, `_epoch1.tar`) |
| Dataset cache (reuse) | `Bigscity-LibCity/libcity/cache/dataset_cache/point_based_PEMSD7_12_12_0.7_0.1_standard_64_False_False_False_True.npz` |
| Model code | `Bigscity-LibCity/libcity/model/traffic_speed_prediction/STFGNN.py` |
| Model config | `Bigscity-LibCity/libcity/config/model/traffic_state_pred/STFGNN.json` |
| Wiring | `Bigscity-LibCity/libcity/config/task_config.json` (`traffic_state_pred` → `STFGNN`) |

Wall-clock: total ≈ 26.9 min (≈0.8 min data load, ≈1.5 min cache write, ≈16.3 min training 2 epochs, ≈2.5 min test eval). Dataset cache is now warm, so re-runs skip the load/encode step.

---

## 7. Conclusion

The migrated **STFGNN** model is **verified working end-to-end in LibCity** on `PEMSD7 / traffic_state_pred`: it constructs on GPU, trains 2 epochs with decreasing loss, saves checkpoints, and produces valid MAE/RMSE/R² metrics. **No fix is required from model-adapter or config-migrator.** The only outstanding item is the `autogen`-env `torch_sparse` incompatibility (§4), which is an environment concern the Leader should route/fix independently.
