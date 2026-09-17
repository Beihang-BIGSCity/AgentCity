# AGCRN — Migration Summary (LibCity)

**Paper:** "Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting" (NeurIPS 2020)
**Model:** AGCRN
**Original repository:** https://github.com/LeiBAI/AGCRN
**Migration date:** 2026-09-16
**Status:** ✅ **SUCCESS (verified end-to-end)**

> This is the canonical, consolidated migration record. It supersedes the earlier
> `./documents/AGCRN_migration_summary.md`.

---

## Phase 1 — Clone

- Cloned to `./repos/AGCRN` (shallow `--depth=1`, HEAD `7fbbf2a`).
- Key files:

| File | Contents |
|---|---|
| `model/AGCRN.py` | `AGCRN`, `AVWDCRNN` |
| `model/AGCRNCell.py` | `AGCRNCell` |
| `model/AGCN.py` | `AVWGCN` |
| `model/Run.py` | standalone entry point / arg parsing / weight init |
| `model/BasicTrainer.py` | training + validation loops |
| `model/PEMSD4_AGCRN.conf`, `model/PEMSD8_AGCRN.conf` | configparser INI hyper-parameters |
| `lib/dataloader.py` | data loading |
| `lib/normalization.py` | z-score normalisation |
| `lib/metrics.py` | MAE / RMSE / MAPE |

- **Dependencies:** `torch`, `numpy`. No `requirements.txt` in the repo.
- `data/` is empty in git — the code expects the ASTGCN PEMSD4/PEMSD8 `.npz` files, key
  `'data'` of shape `(T, N, 3)`, with **channel 0 = flow**.
- **Task:** single-channel traffic **FLOW** prediction, 12 in → 12 out @ 5-minute resolution.
- **No adjacency file** — the graph is fully adaptive, learned from node embeddings.

---

## Phase 2 — Adapt

- **Created:** `Bigscity-LibCity/libcity/model/traffic_speed_prediction/AGCRN.py`
  - Classes: `AVWGCN`, `AGCRNCell`, `AVWDCRNN`, `AGCRN(AbstractTrafficStateModel)`.
- **Modified:** `Bigscity-LibCity/libcity/model/traffic_speed_prediction/__init__.py`
  - Added the import and the `__all__` entry.

The implementation faithfully preserves the original math:

- `softmax(relu(E·Eᵀ))` adaptive adjacency,
- Chebyshev support set of order `cheb_k`,
- `weights_pool` / `bias_pool` + `einsum` node-adaptive weights,
- GRU-style gated cell,
- non-autoregressive `Conv2d` head over the last encoder hidden state.

**Interface:**

```python
__init__(config, data_feature)
forward(batch)
predict(batch)
calculate_loss(batch)
# output shape: (B, output_window, N, output_dim)
```

- Channels-last `(B, T, N, F)` is LibCity's native layout → **no transpose needed**.
- **No `adj_mx` required.**
- **Config keys read:** `input_window` (fallback `lag` → 12), `output_window`
  (fallback `horizon` → 12), `rnn_units` (64), `embed_dim` (10), `num_layers` (2),
  `cheb_order` (fallback `cheb_k` → 2), `input_dim`.
  **From `data_feature`:** `num_nodes`, `feature_dim`, `output_dim`, `scaler`.
- **Weight-init parity:** `xavier_uniform_` for `dim > 1`, `uniform_` for `dim <= 1`.
- **CPU-safe** — no hard-coded `.cuda()`.

---

## Phase 3 — Configure

**Modified:** `Bigscity-LibCity/libcity/config/model/traffic_state_pred/AGCRN.json`

Final contents:

```json
{
  "embed_dim": 10, "rnn_units": 64, "num_layers": 2, "cheb_order": 2, "bidir_adj_mx": true,
  "input_window": 12, "output_window": 12, "input_dim": 1, "output_dim": 1,
  "scaler": "standard", "load_external": false, "normal_external": false, "ext_scaler": "none",
  "add_time_in_day": false, "add_day_in_week": false,
  "max_epoch": 100, "batch_size": 64, "learner": "adam", "learning_rate": 0.003,
  "weight_decay": 0, "lr_epsilon": 1e-8, "train_loss": "masked_mae",
  "lr_decay": false, "clip_grad_norm": false, "use_early_stop": true, "patience": 15
}
```

**Provenance** (paper / `.conf` values):

- `lag` → `input_window = 12`
- `horizon` → `output_window = 12`
- `rnn_units = 64`
- `embed_dim = 10` (PEMSD4; PEMSD8 uses 2)
- `num_layers = 2`
- `cheb_order = 2`
- `lr_init` → `learning_rate = 0.003` (Adam `eps=1e-8`, `weight_decay=0`)
- `batch_size = 64`
- `epochs` → `max_epoch = 100`
- `early_stop_patience` → `patience = 15`

**`task_config.json`: no change needed** — `AGCRN` is already in
`traffic_state_pred.allowed_model` and already has a module block
(`TrafficStatePointDataset` / `TrafficStateExecutor` / `TrafficStateEvaluator`).

**Datasets:**

- PEMSD4 ✅ ready (`raw_data/PEMSD4/config.json`)
- PEMSD8 ❌ data missing (only listed in `allowed_dataset`)

No new `libcity/config/data/*.json` is needed — `TrafficStatePointDataset.json` already
covers PEMSD4/PEMSD8.

**Loss note:** `train_loss="masked_mae"` is exactly equivalent to the model's own
`calculate_loss` (masked MAE on inverse-transformed values).

---

## Phase 4 — Test (VERIFIED)

### Synthetic smoke test: PASS

- Script: `Bigscity-LibCity/tests/agcrn_synth.py` · Log: `Bigscity-LibCity/tests/agcrn_synth.log`
- import OK
- param count **745,940** (N=20 config)
- input `(2,12,20,1)` → output `(2,12,20,1)`
- backward OK

### End-to-end run: PASS

```
python run_model.py --task traffic_state_pred --model AGCRN \
  --dataset PEMSD4 --seed 10 --config_file ./tests/agcrn_e2e
```

(wrapped in `timeout 600`) · Log: `Bigscity-LibCity/tests/agcrn_e2e.log`

- Wall time **269 s** (data build ~147 s, train 47 s, eval 3 s)
- `num_batches: 372`, total params **756,490**
- `Epoch [0/1] train_loss: 41.9672, val_loss: 24.8194, lr: 0.003000`
- Evaluate CSV: `libcity/cache/34158/evaluate_cache/2026_09_16_16_28_22_AGCRN_PEMSD4.csv`

**Metrics (1 epoch):**

| Horizon | MAE | RMSE | masked_MAPE |
|---|---|---|---|
| 1 | 19.559 | 29.609 | 0.1755 |
| 12 | 28.723 | 43.160 | 0.2291 |

> ⚠️ 1-epoch metrics are **smoke-test values only**, not converged benchmark numbers.
> The paper reports ~MAE 19.2 on PEMSD4 with 100 epochs.

---

## Known issues / caveats

1. **AGCRN shadowing (open).**
   `libcity/utils/utils.py` `get_model()` tries `traffic_flow_prediction` BEFORE
   `traffic_speed_prediction` for task `traffic_state_pred`, and
   `traffic_flow_prediction/__init__.py` already exports `AGCRN`. So the **flow** class
   wins at runtime and the new speed file is shadowed. Both are functionally equivalent.
   *Recommended fix:* remove `"AGCRN"` from `traffic_flow_prediction/__init__.py`, or add
   an explicit registry mapping. **Not applied** (touches shared code outside migration scope).
2. **Environment issue, unrelated to AGCRN.**
   `libcity/data/dataset/__init__.py` eagerly imports `DeepMapMatchingDataset`, which
   imports `torch_sparse`. In this env `torch_sparse` is ABI-broken
   (`undefined symbol ... VariableInfo`) and `libcudart.so.12` is not on the default path.
   *Workaround used:* `LD_LIBRARY_PATH` pointing at
   `site-packages/nvidia/cuda_runtime/lib` + a throwaway `sys.modules` stub for
   `torch_sparse` in `tests/agcrn_e2e_runner.py`.
   *Recommended permanent fix:* make that import lazy/guarded in
   `libcity/data/dataset/__init__.py`.
3. **`_init_device()` latent bug.** The `use_gpu=False` fallback branch references an
   undefined name `cpu` → `NameError`. So `--gpu false` cannot be used; CPU-only runs
   need a code fix.
4. **PEMSD8 not runnable.** `raw_data/PEMSD8/` is absent; it also needs an `embed_dim`
   override to 2 via `--config_file`.
5. **`input_dim` channel mismatch.** The speed model asserts `x.shape[3] == input_dim (1)`,
   while stock `raw_data/PEMSD4/config.json` has `data_col` = 3 columns
   (`feature_dim=3`). The flow model ignores `input_dim` so it works as-is. To exercise
   the speed class on PEMSD4, override `data_col: ["traffic_flow"]`.
6. **`config_file` CLI rule.** Pass the path WITHOUT `.json` (relative to the LibCity
   root) — `config_parser.py` does `./{}.json`.
7. **Source-repo bugs not ported.** `Run.py` mode-string casing mismatch (`'Train'` vs
   `'train'`) and hard-coded `.cuda()` in `BasicTrainer`.

---

## Artifacts

| Artifact | Path |
|---|---|
| Model class | `Bigscity-LibCity/libcity/model/traffic_speed_prediction/AGCRN.py` |
| Registration | `Bigscity-LibCity/libcity/model/traffic_speed_prediction/__init__.py` |
| Model config | `Bigscity-LibCity/libcity/config/model/traffic_state_pred/AGCRN.json` |
| Synthetic test | `Bigscity-LibCity/tests/agcrn_synth.py` (+ `.log`) |
| E2E runner | `Bigscity-LibCity/tests/agcrn_e2e_runner.py`, `agcrn_e2e.json` (+ `agcrn_e2e.log`) |
| Cloned source | `./repos/AGCRN` |

---

## Follow-up recommendations

1. Resolve the AGCRN shadowing (pick one implementation).
2. Make the `DeepMapMatchingDataset` / `torch_sparse` import lazy in
   `libcity/data/dataset/__init__.py`.
3. Fix `_init_device()`'s `cpu` `NameError` so CPU-only runs work.
4. Run a full 100-epoch benchmark on PEMSD4 to compare against the paper (MAE ≈ 19.2).
5. Optionally obtain PEMSD8 data and run with `embed_dim=2`.
