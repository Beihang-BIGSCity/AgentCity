# STFGNN LibCity Adapter Notes

**Model**: STFGNN — *Spatial-Temporal Fusion Graph Neural Networks for Traffic Flow Forecasting* (AAAI 2021)
**Original repo**: https://github.com/MengzhangLI/STFGNN (MXNet symbolic — `models/stsgcn_4n_res.py`)
**Migration type**: re-authoring (the source is `mx.sym.*`; there is no `nn.Module` and no exportable weights)
**LibCity task**: `traffic_state_pred` (traffic speed/flow prediction)

---

## 1. Registered class name & file path

| Item | Value |
|---|---|
| **Registered class name** | `STFGNN` |
| **Model file** | `Bigscity-LibCity/libcity/model/traffic_speed_prediction/STFGNN.py` |
| **Registered in** | `Bigscity-LibCity/libcity/model/traffic_speed_prediction/__init__.py` (`from ...STFGNN import STFGNN`, added to `__all__`) |
| **Base class** | `libcity.model.abstract_traffic_state_model.AbstractTrafficStateModel` |
| **Resolved by** | `libcity.utils.utils.get_model` (looks up `config['model']` on `libcity.model.traffic_flow_prediction`, then falls back to `libcity.model.traffic_speed_prediction`) |
| **Smoke test** | `tests/stfgnn_smoke.py` (run: `python tests/stfgnn_smoke.py`) |

Set `"model": "STFGNN"` in the task config.

---

## 2. Architecture as implemented (PyTorch)

Faithful re-authoring of the MXNet symbol graph, mirroring the already-migrated
predecessor `libcity/model/traffic_flow_prediction/STSGCN.py` (which uses the
LibCity batch-first `(B, n, C)` convention), extended with STFGNN's two
innovations (temporal-conv residual branch + `(4N, 4N)` fusion graph).

```
(B, 12, N, C)                          # input; only channel 0 is used
  └─ First_FC: Linear(C -> 64) + ReLU              (first_layer_embedding)
  └─ STSGCL x3   (filters [[64,64,64]] x3, 'individual'):
        • PositionEmbedding: learnable temporal (1,T,1,C) + spatial (1,1,N,C)
        • Gated dilated temporal conv (GLU), kernel (1,2), dilation (1, strides-1)
          → residual branch (B, T-3, N, C')
        • for each sliding window of `strides`(=4) time steps:
              reshape → (B, 4N, C) → STSGCM(fusion-GCN) → (B, N, C')
        • residual add:  STSGCM-concat  +  temporal-conv branch
  └─ Output head (per horizon step): Linear(T*C -> 128) + ReLU -> Linear(128 -> 1)
  └─ concat over 12 steps → (B, 12, N, output_dim)
```

- **STSGCM** = `len(filters)` stacked GCN ops; each GCN op is `A @ X` then GLU
  (`lhs * sigmoid(rhs)`); outputs cropped to the **middle N vertices** `[N:2N]`
  and combined with element-wise **max**.
- **Temporal conv**: `Conv2d(kernel=(1,2), dilation=(1, strides-1))` on the
  `(B, C, N, T)` layout → cuts `T` by `strides-1 = 3` per layer, matching the
  STSGCM window reduction (12 → 9 → 6 → 3).
- **Loss**: Huber, `delta = rho` (paper `rho = 1`), computed on
  scaler-inverse-transformed predictions/targets (LibCity convention).

Parameter count for N=20, T=12, filters `[[64,64,64]]x3`: **808,908**.

---

## 3. Fusion graph / adjacency construction

Implemented as a module-side utility `construct_adj_fusion(a_spatial, a_dtw, steps=4)`
— a verbatim reproduction of `utils_4n0_3layer_12T_res.py::construct_adj_fusion`:

```
        blk0   blk1   blk2   blk3
blk0    A_dtw   I      I      A_dtw
blk1    I       A_sp   I      I
blk2    I       I      A_sp   I
blk3    A_dtw   I      I      A_dtw      (I = identity band, self-loops forced = 1)
```
Output shape `(4N, 4N)`. `use_mask=True` makes the edge strengths learnable:
a `nn.Parameter` mask is initialized to the binary support `(fusion != 0)` and
multiplied with the fixed fusion graph **inside `forward`** (so no
`retain_graph=True` is required).

**Spatial graph `A_sp`** — taken from LibCity's `data_feature['adj_mx']`, then
binarized to connectivity `(adj_mx != 0)` (the paper uses a binary connectivity
graph; `inf`/`nan` from distance matrices are dropped). If `adj_mx` is missing or
mis-shaped, it falls back to an identity (self-loop) matrix — **never hard-fails**.

**Temporal (DTW) graph `A_dtw`** — resolved in this order (all optional; no
dataset is in scope this round, so the safe default is used):
1. `config['adj_dtw']` — an `(N, N)` array;
2. `config['adj_dtw_filename']` / `config['adj_dtw_file']` — path to `.npy`/`.csv`;
3. `config['adj_dtw_default']` — `'identity'` (default) → `np.eye(N)`, or
   `'spatial'` → reuse `A_sp` (the repo README's "spatial-information-free" mode).

A helper `build_dtw_temporal_adjacency(series, adj_percent=0.01, order=1, lag=12)`
is also provided to pre-compute `A_dtw` from a `(L, N)` z-scored train signal
matrix (DTW order=1, lag=12, top-1% neighbours, symmetrized, self-loops), matching
`data/fastDTW_adj_gen.py`. It is **not** called by the model's default path.

> To make `adj_mx` a binary connectivity matrix in the dataset config, set
> `"set_weight_link_or_dist": "link"` (and `"init_weight_inf_or_zero": "zero"`).

---

## 4. Config keys read via `config.get(...)`

Standard LibCity keys:

| Key | Default | Notes |
|---|---|---|
| `input_window` | 12 | T_in (paper = 12) |
| `output_window` | 12 | T_out (paper = 12) |
| `device` | `torch.device('cpu')` | guarded as required |

STFGNN-specific keys (all optional; defaults reproduce the paper):

| Key | Default | Meaning |
|---|---|---|
| `first_layer_embedding_size` | 64 | input projection dim (0/None disables) |
| `filters` | `[[64,64,64],[64,64,64],[64,64,64]]` | 3 STSGCL layers × 3 GCN of 64 |
| `act_type` | `'GLU'` | GCN activation (`'GLU'` or `'relu'`) |
| `module_type` | `'individual'` | only `'individual'` implemented (warns otherwise) |
| `use_mask` | `True` | learnable adjacency mask |
| `temporal_emb` | `True` | learnable temporal embedding |
| `spatial_emb` | `True` | learnable spatial embedding |
| `strides` | 4 | time steps per local spatio-temporal graph; must be 4 for the fusion graph |
| `rho` | 1.0 | Huber loss delta |
| `out_layer_dim` | 128 | output-head hidden FC dim |
| `adj_dtw` / `adj_dtw_filename` / `adj_dtw_file` | — | optional DTW temporal graph (array or path) |
| `adj_dtw_default` | `'identity'` | fallback temporal graph (`'identity'` or `'spatial'`) |

Read from `data_feature` (not config): `num_nodes`, `feature_dim`, `output_dim`, `scaler`, `adj_mx`.

---

## 5. Data / layout transformations

- **Input**: `batch['X']` is `(B, input_window, N, feature_dim)`. Only the first
  `feature_dim` (channel 0 = flow) channels are used; extra external channels are
  cropped: `X[:, :, :, 0:feature_dim]`.
- **Internal permutations**:
  - temporal conv operates on `(B, C, N, T)` (`permute(0,3,2,1)`);
  - each sliding window is reshaped to `(B, strides*N, C) = (B, 4N, C)` for the
    `(4N, 4N)` fusion GCN (batch-first, `torch.matmul(adj, data)` broadcasting);
  - output head reshapes `(B, T, N, C) -> (B, N, T, C) -> (B, N, T*C)`.
- **Output**: `forward`/`predict` return `(B, output_window, N, output_dim)`
  (single forward pass produces all `output_window` steps — no autoregression).
- **Normalization**: the model works in the dataset's (z-scored) scale;
  `calculate_loss` applies `scaler.inverse_transform` before the Huber loss, and
  the LibCity evaluator de-normalizes `predict` output for metrics.

---

## 6. Smoke check result (CPU, dummy data)

`tests/stfgnn_smoke.py` (run under the `autogen` conda env, torch 2.6.0):

```
[ok] model constructed; parameters = 808908
[ok] fusion adj shape = (80, 80)                  # N=20 -> 4N=80
[ok] forward out shape = (4, 12, 20, 1)
[ok] predict out shape = (4, 12, 20, 1)
[ok] calculate_loss = ...
[ok] backward ok
[ok] multi-channel input forward out shape = (4, 12, 20, 1)
[ok] adj_mx=None fallback adj shape = (80, 80)    # identity fallback, no hard fail
[ok] get_model("STFGNN") resolved -> STFGNN
SMOKE CHECK PASSED
```

---

## 7. Assumptions & limitations

- `module_type != 'individual'` is not implemented (the MXNet source only implements
  `individual`); a warning is emitted and it falls back to `individual`.
- No dataset/DTW graph is in scope this round, so the default temporal branch of
  the fusion graph is identity (self-loops). Supply `adj_dtw`/`adj_dtw_filename`
  for the full paper fusion graph.
- `strides` is fixed at 4 (the fusion-graph builder is defined for `steps=4`);
  changing it would also require re-deriving the fusion-graph block layout.
- Sequence split/normalization (chronological 6:2:2, z-score from train split) is
  delegated to LibCity's dataset pipeline, not reproduced inside the model.
