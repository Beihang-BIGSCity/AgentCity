# STFGNN — LibCity Config Migration Notes

**Model**: STFGNN — *Spatial-Temporal Fusion Graph Neural Networks for Traffic Flow Forecasting* (AAAI 2021)
**Task**: `traffic_state_pred`
**Author**: config-migrator
**Companion docs**: `documentation/STFGNN_adapter_notes.md` (adapter), `documentation/STFGNN_repo_analysis.md` (repo-cloner)

---

## 1. Files created / modified (exact paths)

| Action | Absolute path |
|---|---|
| **Created** | `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traffic_state_pred/STFGNN.json` |
| **Modified** | `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json` |

No dataset config file was modified.

---

## 2. `task_config.json` wiring

All changes are inside the `"traffic_state_pred"` object.

1. **Registered in `allowed_model`** — `"STFGNN"` appended after `"MVGRL"` (the previous last entry):

   ```json
   "Garner",
   "MVGRL",
   "STFGNN"
   ],
   ```

2. **Model block** — added immediately after the `"STGCN"` block (mirrors how existing
   point-based traffic-state models are declared; `GWNET`/`AGCRN`/`TGCN` use the same trio):

   ```json
   "STFGNN": {
       "dataset_class": "TrafficStatePointDataset",
       "executor": "TrafficStateExecutor",
       "evaluator": "TrafficStateEvaluator"
   },
   ```

   Rationale for each field:
   - `dataset_class = TrafficStatePointDataset` — STFGNN consumes `data_feature['adj_mx']`
     (the N×N point adjacency from the `.rel` file), which this dataset supplies
     (`traffic_state_point_dataset.py::get_data_feature`). Same class used by STGCN/GWNET.
   - `executor = TrafficStateExecutor` — standard train/eval loop for
     `traffic_state_pred`; STFGNN needs no custom executor.
   - `evaluator = TrafficStateEvaluator` — standard MAE/RMSE/MAPE evaluator.

`"STFGNN"` is already present in `traffic_state_pred.allowed_dataset` for the dataset chosen
below, so **no dataset-list change was required**.

*Verification of resolution*: `ConfigParser._load_default_config`
(`libcity/config/config_parser.py`) looks the model up in `task_config[task]['allowed_model']`
and reads `task_config[model]['dataset_class'/'executor'/'evaluator']`. With the two edits above
it resolves to `STFGNN` → `TrafficStatePointDataset` / `TrafficStateExecutor` / `TrafficStateEvaluator`.
The model class itself is registered at
`libcity/model/traffic_speed_prediction/__init__.py` (import + `__all__`), which
`libcity.utils.utils.get_model` finds via its `traffic_flow_prediction` → `traffic_speed_prediction`
fallback.

---

## 3. Resolved hyperparameters (`STFGNN.json`)

Every model-specific key below is read by `STFGNN.py` via `config.get(...)`
(cross-checked line by line against `documentation/STFGNN_adapter_notes.md` §4).

| Key | Value | Read at | Source |
|---|---|---|---|
| `input_window` | `12` | `STFGNN.py:364` | Paper (T_in = 12); also LibCity default |
| `output_window` | `12` | `STFGNN.py:365` | Paper (T_out = 12) |
| `first_layer_embedding_size` | `64` | `STFGNN.py:377` | Paper / adapter default |
| `filters` | `[[64,64,64],[64,64,64],[64,64,64]]` | `STFGNN.py:408` | Paper: 3 STSGCL layers × 3 GCN filters of 64 |
| `act_type` | `"GLU"` | `STFGNN.py:370` | Paper (gated activation) |
| `module_type` | `"individual"` | `STFGNN.py:369` | Only mode the code implements |
| `use_mask` | `true` | `STFGNN.py:373` | Paper (learnable adjacency mask) |
| `temporal_emb` | `true` | `STFGNN.py:371` | Paper |
| `spatial_emb` | `true` | `STFGNN.py:372` | Paper |
| `strides` | `4` | `STFGNN.py:374` | Paper; **must be 4** — the (4N,4N) fusion graph is defined for 4 time steps |
| `rho` | `1.0` | `STFGNN.py:375` | Paper (Huber `delta`) |
| `out_layer_dim` | `128` | `STFGNN.py:376` | Paper / adapter default |
| `adj_dtw_default` | `"identity"` | `STFGNN.py:463` | Adapter default for this round (no DTW graph supplied) |

Standard LibCity pipeline keys (mirroring `STGCN.json` / `STSGCN.json`; consumed by the
dataset/executor, **not** invented model parameters):

| Key | Value | Purpose |
|---|---|---|
| `batch_size` | `64` | training batch size (LibCity dataset default) |
| `scaler` | `"standard"` | z-score, matching the paper and the adapter's loss/runtime assumption |
| `load_external` | `false` | PEMSD7 has no `.ext` file |
| `normal_external` | `false` | — |
| `ext_scaler` | `"none"` | — |
| `add_time_in_day` | `false` | STFGNN uses no external time features |
| `add_day_in_week` | `false` | — |
| `max_epoch` | `100` | training epochs |
| `learner` | `"adam"` | paper optimiser (Adam) |
| `learning_rate` | `0.001` | mirrors STGCN/STSGCN model configs |
| `lr_decay` | `false` | no LR decay |
| `clip_grad_norm` | `false` | — |
| `use_early_stop` | `false` | run the full epoch budget (mirrors STGCN.json) |

### Keys intentionally NOT included

Per the requirement "only include keys the STFGNN model code actually reads":

- **`dropout`** — `STFGNN.py` has *no* dropout anywhere (the block uses GLU gating plus
  residual connections); it never calls `config.get('dropout')`. Omitting it is deliberate.
- Keys from neighbouring architectures that STFGNN ignores, e.g. `Ks`/`Kt`/`blocks`
  (STGCN), `nb_block`/`K`/`nb_chev_filter` (ASTGCN), `bidir_adj_mx`/`distance_inverse`
  (STSGCN).
- `device` — injected at runtime by `ConfigParser._init_device()`; never set in a config file.
- `adj_dtw` / `adj_dtw_filename` — optional; left unset so the model takes the documented
  identity fallback (no DTW graph is in scope this round).

---

## 4. Dataset choice and compatibility

**Chosen dataset: `PEMSD7`** (`Bigscity-LibCity/raw_data/PEMSD7/`).

Verified present in LibCity atomic format: `PEMSD7.geo`, `PEMSD7.rel`, `PEMSD7.dyna`,
`config.json` (and `PEMSD7.zip`).

Rationale (`raw_data/PEMSD7/config.json`):

| Property | PEMSD7 value | Why it fits STFGNN |
|---|---|---|
| `data_col` | `["traffic_flow"]` | **Single-channel traffic FLOW** — exactly the task STFGNN was designed for (and what `STFGNN.py` expects: channel 0 / `feature_dim = 1`) |
| `output_dim` | `1` | single-channel output, matches the paper |
| `set_weight_link_or_dist` | `"link"` | adjacency is a **binary connectivity matrix**, which is precisely what `STFGNN.py::_build_spatial_adj` binarises (`adj_mx != 0`) |
| `init_weight_inf_or_zero` | `"zero"` | required pairing with `"link"` (per the lead's guidance and the dataset loader) |
| `time_intervals` | `300` | 5-minute sampling, matches the paper's setting |
| `rel_file` | `"PEMSD7"` | a real `.rel` file exists → `data_feature['adj_mx']` is a meaningful N×N graph |

This means **no dataset override of `data_col` / `output_dim` / adjacency flags was needed** —
PEMSD7's own `config.json` already delivers the single-channel flow task *and* the binary
adjacency that STFGNN requires. (By contrast `PEMSD4` is 3-channel flow+occupancy+speed and
`METR_LA`/`PEMS_BAY` are single-channel *speed*.)

`PEMSD7` is already listed in `traffic_state_pred.allowed_dataset`, so the wiring is complete
as-is.

**Fallback dataset: `METR_LA`** — also present under `raw_data/` and single-channel (speed),
and it is the default dataset of the `test_migration` tool. STFGNN also loads it cleanly
(its `traffic_speed` column → `feature_dim = 1`, `output_dim = 1`); the only caveat is that the
paper's task is *flow*, not speed. Use PEMSD7 for the paper-faithful run; METR_LA is a valid
smoke-test alternative.

---

## 5. Loadable (model, dataset) pair + run command

**Pair**: model = `STFGNN`, dataset = `PEMSD7` (task `traffic_state_pred`).

Command (run from the LibCity root `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity`):

```bash
cd /home/wangwenrui/shk/AgentCity/Bigscity-LibCity
python run_model.py --task traffic_state_pred --model STFGNN --dataset PEMSD7
```

Or, via the team's migration-test tool:

```
test_migration(model_name="STFGNN", dataset="PEMSD7", task="traffic_state_pred")
```

(That tool defaults to `dataset="METR_LA"`; pass `PEMSD7` explicitly for the flow run.)

---

## 6. Notes / assumptions

- **Sequence-length feasibility**: with `input_window = 12` and 3 STSGCL layers at `strides = 4`,
  the internal length shrinks `12 → 9 → 6 → 3` — every layer stays positive
  (`STFGNN.py:421` raises only if it would go `≤ 0`). The chosen windows are safe.
- **`strides` is not free**: it is coupled to the `construct_adj_fusion(..., steps=4)` builder;
  changing it without re-deriving the fusion graph will break the model.
- **Adjacency**: PEMSD7's `link`/`zero` settings make `data_feature['adj_mx']` a binary
  connectivity matrix, so `_build_spatial_adj()` binarises to itself. If a future dataset uses
  `"dist"`/`"inf"`, the loader still binarises (after dropping inf/nan), but `"link"`/`"zero"`
  is the intended pairing.
- **Paper-vs-framework training keys**: the STFGNN class reads only the model keys in §3.
  `batch_size`, `max_epoch`, `learner`, `learning_rate`, etc. are consumed by the LibCity
  dataset/executor; values were mirrored from the sibling model configs
  (`STGCN.json` / `STSGCN.json`) since the adapter notes do not pin optimiser hyperparameters.
  Only `learner="adam"` follows the paper directly. These are safe to tune later.
- No files under `./documents/` or `./tests/` were created by this task.
