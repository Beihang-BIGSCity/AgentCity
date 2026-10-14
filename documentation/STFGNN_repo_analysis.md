# STFGNN Repository Analysis Report

**Paper**: Spatial-Temporal Fusion Graph Neural Networks for Traffic Flow Forecasting (AAAI 2021)
**Repo URL**: https://github.com/MengzhangLI/STFGNN
**Cloned to**: `./repos/STFGNN` (project root `/home/wangwenrui/shk/AgentCity`)
**Clone date / HEAD**: commit `a02feee` ("newreadme"), depth=1
**Submodules**: none (no `.gitmodules`)
**Task**: traffic flow prediction (5-min interval, 12-step input → 12-step output)

> ⚠️ **CRITICAL MIGRATION NOTE — this repo is MXNet, NOT PyTorch.**
> The entire model is written in **Apache MXNet symbolic API** (`mx.sym.*`), not `torch.nn`.
> There is **no `nn.Module` / model class** — the "model" is a set of *symbol-definition functions*
> (`stsgcn(...)`, `stsgcm(...)`, `stsgcl(...)`, etc.) that build a symbolic computation graph.
> model-adapter must **re-implement the architecture in PyTorch** (there is no direct port path for
> weights). The architecture description below is complete enough to reconstruct it 1:1 in PyTorch.

---

## 1. Key Files (exact paths)

| Role | Exact path | Notes |
|---|---|---|
| **Main model definition** | `models/stsgcn_4n_res.py` | SOLE model *source* file. Contains function `stsgcn(...)` and all layers. |
| **Model builder / data utils** | `utils_4n0_3layer_12T_res.py` | `construct_model(config)`, `generate_data`, adjacency builders, metrics. |
| **Training / eval entry script** | `main_4n0_3layer_12T_res.py` | CLI: `python main_4n0_3layer_12T_res.py --config <cfg.json>` |
| **Model package init** | `models/__init__.py` | Empty file. |
| **Configs** | `config/PEMS03/individual_3layer_12T.json`, `config/PEMS04/individual_3layer_12T.json`, `config/PEMS07/individual_3layer12T.json`, `config/PEMS08/individual_3layer_12T.json`, `config/newdataset/individual_3layer_12T.json` | JSON per dataset. |
| **Requirements** | `requirements.txt` | `mxnet-cu100`, `pytest`, `graphviz` |
| **README** | `README.md` | Notes MXNet, dataset download (Baidu), notes PEMS-BAY/METR-LA results withdrawn. |
| **Graph construction (DTW)** | `data/fastDTW_adj_gen.py`, `data/stsgcn-dtw.py`, `data/Temporal_Graph_gen.py` | Build the temporal (DTW) adjacency. |
| **Data preprocessing** | `data/STSGCN_data_preprocession.py`, `data/STSGCN_data_preprocession2.py` | Build `.npz` from raw CSV (PeMSD7 example). |
| **Data (present)** | `data/PEMS08/PEMS08.npz`, `data/PEMS08/PEMS08.csv`; `data/adj_PEMS0X*.csv`; `data/sensor_graph/*`; `data/stsgcn/*.npy` | See §4. |

> **Only PEMS08 raw data ships in the repo.** PEMS03/PEMS04/PEMS07 raw datasets are **NOT** included
> (must be downloaded from Baidu — `STSGCN_data.tar.gz`, pw `p72z`). The pre-computed adjacency CSVs
> for all four datasets *are* present under `data/`.

### 1b. Dead / bytecode-only variants (do NOT rely on these)
`models/__pycache__/` contains compiled `.pyc` for `stsgcn`, `stsgcn_4n`, `stsgcn_4n0`, `stsgcn_4n1`,
`stsgcn_4n_res`, `stsgcn_1nFusion`, `stsgcn_3nFusion`, `stsgcn_dtw` — **only `stsgcn_4n_res.py` has
source**. All others are bytecode-only. The training script uses **`models.stsgcn_4n_res`** exclusively.
`stsgcn_4n_res` = the "4N, residual, 3-layer, 12T" variant = the STFGNN paper model.

---

## 2. Model architecture (from `models/stsgcn_4n_res.py`)

All functions are MXNet-symbol builders; shapes are symbolic tensor shapes.

### Entry: `stsgcn(data, adj, label, input_length, num_of_vertices, num_of_features, filter_list, module_type, activation, use_mask, mask_init_value, temporal_emb, spatial_emb, prefix, rho=1, predict_length=12)`
- `data` shape `(B, T, N, C)` = `(B, 12, N, 1)`
- `adj` shape `(4N, 4N)` (the STFGNN **fusion graph**, see §4)
- `label` shape `(B, 12, N)`
- Optional learnable `mask` of shape `(4N, 4N)` multiplies `adj` (sparsity mask, `use_mask=True`).
- Loops over `filter_list` (one entry per STSGCL layer); each layer calls `stsgcl`; after each layer
  `input_length -= 3` and `num_of_features = filters[-1]`.
- Builds the output by calling `output_layer(...)` **`predict_length` times with `predict_length=1`**
  and `mx.sym.concat(*, dim=1)` → final pred `(B, 12, N)`.
- Returns `mx.sym.Group([huber_loss(data, label, rho), mx.sym.BlockGrad(data, name='pred')])`.
  → **outputs = [loss, prediction]**; `mod.predict(...)[1]` is the prediction.

### Supporting layer functions (names exact)
| Function | Purpose | In→Out shape |
|---|---|---|
| `position_embedding(data, input_length, num_of_vertices, embedding_size, temporal, spatial, init, prefix)` | Learnable temporal `(1,T,1,C)` and spatial `(1,1,N,C)` embeddings, broadcast-added. | `(B,T,N,C)` → `(B,T,N,C)` |
| `gcn_operation(data, adj, num_of_filter, num_of_features, num_of_vertices, activation, prefix)` | Graph conv: `dot(adj, data)` then FC; `GLU` = split→`lhs*sigmoid(rhs)`; `relu` alt. | `(4N,B,C)` → `(4N,B,C')` |
| `stsgcm(data, adj, filters, num_of_features, num_of_vertices, activation, prefix)` | **STSGCM**: stack `len(filters)` GCN ops, crop each to the **middle N vertices** (`slice` begin=`N` end=`2N`), then element-wise **max** over the stacked outputs. | `(4N,B,C)` → `(N,B,C')` |
| `stsgcl(data, adj, T, num_of_vertices, num_of_features, filters, module_type, activation, temporal_emb, spatial_emb, prefix)` | STSGCL dispatcher; only `module_type=='individual'` implemented. | `(B,T,N,C)` → `(B,T-3,N,C')` |
| `sthgcn_layer_individual(...)` | The actual STSGCL body (see below). | `(B,T,N,C)` → `(B,T-3,N,C')` |
| `output_layer(data, num_of_vertices, input_length, num_of_features, num_of_filters=128, predict_length=12)` | Output head: `swapaxes`→`(B,N,T,C)`→reshape `(B,N,T*C)`→FC(128)+relu→FC(predict_length)→`swapaxes`→`(B,T',N)`. | `(B,T,N,C)` → `(B,T',N)` |
| `huber_loss(data, label, rho=1)` | `where(|d|>rho, |d|-0.5rho, 0.5/rho*d^2)` wrapped in `MakeLoss`. | `(B,T',N),(B,T',N)` → loss |
| `weighted_loss(data, label, input_length, rho=1)` | Huber loss re-weighted by descending horizon weights. **Defined but NOT used** by `construct_model`/`stsgcn`. | — |

### `sthgcn_layer_individual` — the core STFGNN block (exact op order)
1. `position_embedding` (temporal+spatial) on `(B,T,N,C)`.
2. **Gated temporal convolution ("TCN"/dilated causal conv)**:
   - `transpose` → `(B,C,N,T)`
   - `data_left  = sigmoid(Convolution(kernel=(1,2), stride=(1,1), dilate=(1,3)))`
   - `data_right = tanh(   Convolution(kernel=(1,2), stride=(1,1), dilate=(1,3)))`
   - `data_time_axis = data_left * data_right` (GLU gate)
   - `data_res = transpose(...)` → `(B, T-3, N, C)`  ← **residual branch**
   - Effective temporal receptive field `= (2-1)*3+1 = 4` ⇒ output time steps `T-3`.
3. **Spatial-temporal fusion block**: for `i in range(T-3)`:
   - slice 4 consecutive time steps → `(B,4,N,C)`
   - `reshape` → `(B, 4N, C)` → `transpose` → `(4N, B, C)`
   - `stsgcm(t, adj, filters, ...)` → `(N, B, C')` (uses the **4N×4N fusion adj**)
   - `swapaxes` → `(B, N, C')` → `expand_dims(axis=1)` → `(B,1,N,C')`
   - `concat` over 1..T-3 → `(B, T-3, N, C')`
4. **Residual add**: `layer_out = stsgcm_concat + data_res` → `(B, T-3, N, C')`.

### Shape flow through the full net (config values)
```
Input data            (B, 12, N, 1)
first_layer_embedding FullyConnected(64)+relu   -> (B, 12, N, 64)
STSGCL #1 (filters [64,64,64])                  -> (B,  9, N, 64)   # T-3
STSGCL #2                                       -> (B,  6, N, 64)   # T-3
STSGCL #3                                       -> (B,  3, N, 64)   # T-3
output_layer (x12, each predict_length=1) concat-> (B, 12, N)
```
`construct_model` asserts `net.infer_shape(...)[1][1] == (B, num_for_predict, num_of_vertices)`.

### `construct_model(config)` (in `utils_4n0_3layer_12T_res.py`) — exact wiring
1. `adj = get_adjacency_matrix(config['adj_filename'], N, id_filename=config['id_filename'])` → spatial `(N,N)`.
2. `adj_dtw = np.array(pd.read_csv(config['adj_dtw_filename'], header=None))` → temporal `(N,N)`.
3. `adj_mx = construct_adj_fusion(adj, adj_dtw, 4)` → **`(4N, 4N)` fusion graph**.
4. `adj` symbolic var initialized as `Constant(adj_mx)` and wrapped in `BlockGrad` (fixed, non-learnable).
5. `mask_init_value = (adj_mx != 0).astype('float32')` → binary sparsity mask.
6. If `first_layer_embedding_size`: `FullyConnected(data, hidden=first_layer_embedding_size)+relu`.
7. Call `stsgcn(...)` with `predict_length=12`, `rho=1`.

---

## 3. Training / eval entry & forward signature

**Entry script**: `main_4n0_3layer_12T_res.py`
- CLI: `python main_4n0_3layer_12T_res.py --config config/PEMS08/individual_3layer_12T.json [--test] [--plot] [--save]`
- Uses `mx.mod.Module` (symbolic), `data_names=['data']`, `label_names=['label']`, `context=ctx` (from `config['ctx']` GPU id(s)).
- `mod.bind`:
  - **data** `('data', (batch_size, points_per_hour=12, num_of_vertices, 1))`
  - **label** `('label', (batch_size, num_for_predict=12, num_of_vertices))`
- `mod.init_params(mx.init.Xavier(magnitude=0.0003))`.
- **LR schedule**: `mx.lr_scheduler.PolyScheduler(max_update=global_train_steps*epochs*max_update_factor, base_lr=lr, pwr=2, warmup_steps=global_train_steps)`.
- Optimizer `adam`. Metric `['RMSE','MAE']`, output name `'pred_output'`.
- `mx.io.NDArrayIter` data loaders (train shuffled, val/test not). `y` is `squeeze(axis=-1)` → `(samples, 12, N)`.
- Early model selection on **validation masked MAE** (`masked_mae_np`); on improvement it evaluates the
  test set and records MAE/MAPE/RMSE for horizons 1..12.

**Forward-pass signature (LibCity target analogue)**:
- **Input tensor**: `x` of shape `(B, input_window=12, num_nodes=N, input_dim=1)`, channel = **flow**.
- **Output tensor**: `y_pred` of shape `(B, output_window=12, num_nodes=N)` (label/drop the last dim).
- Note the layout: LibCity typically uses `(B, T, N, C)` for input and `(B, T, N, C)`/`(B, T, N)` for
  output — this matches STFGNN's `(B, T', N)` output, **but** STFGNN internally reorders to `(4N, B, C)`
  for the graph conv and `(B, N, T, C)` in the output head. Adapter must reproduce these permutations.

**Loss**: `huber_loss(rho=1)` inside the symbol graph (train), metrics computed in NumPy.
(`weighted_loss` is available in `utils_...` but unused.)

---

## 4. Datasets, data handling & graph construction (KEY DETAIL)

### 4.1 Datasets used by the paper
| Dataset | Nodes (N) | Interval | Config `ctx` | Raw data in repo? |
|---|---|---|---|---|
| PEMS03 | 358 | 5 min | 2 | ❌ (download) |
| PEMS04 | 307 | 5 min | 3 | ❌ |
| PEMS07 | 883 | 5 min | 1 | ❌ |
| PEMS08 | 170 | 5 min | 0 | ✅ `data/PEMS08/` |
| METR-LA / PEMS-BAY | 207 / 325 | 5 min | — | sensor data present under `data/sensor_graph/` but **results withdrawn** (README §Update) |

Notation: "`3layer_12T`" = 3 STSGCL layers, **12 input time steps** (=1 hour). Output = **12 steps**.

### 4.2 Data files present & their formats
- `data/PEMS08/PEMS08.npz` → single key **`data`**, shape **`(17856, 170, 3)`** float64
  (`17856 = 62 days × 288`; channels = `[flow, occupy, speed]`; **only channel 0 / flow is used**, see `generate_seq` `[..., 0:1]`).
- `data/PEMS08/PEMS08.csv` → spatial edge list with header `from,to,cost` (296 lines = 295 edges).
  This is what `config['adj_filename']` points to.
- `data/adj_PEMS0X.csv` → **pre-built DTW temporal adjacency** as a plain `N×N` matrix CSV (no header),
  rows = N. (These are the `adj_dtw_filename`? No — see below: configs point to `adj_PEMS0X_001.csv`.)
- `data/adj_PEMS0X_001.csv` → **the temporal-graph adjacency actually used** (`adj_dtw_filename`),
  binary `N×N`, 0/1 values, **self-loops = 1**.
  - `adj_PEMS03_001.csv` rows=358, `adj_PEMS04_001.csv`=307, `adj_PEMS07_001.csv`=883, `adj_PEMS08_001.csv`=170.
  - `adj_PEMS03_005.csv` is the 5%-sparsity variant (unused by default configs).
- `data/stsgcn/PEMS0X-dtw-288-1-.npy` → **raw (non-thresholded) DTW distance matrices** per dataset.
- `data/sensor_graph/` → METR-LA/PEMS-BAY sensor graph + distance CSVs + `adj_mx.pkl` (pickle, non-ascii → py2 pickle).

### 4.3 Spatial adjacency (`get_adjacency_matrix`)
- Reads edge CSV `(from,to,cost)`; if `id_filename` given, maps original sensor IDs → contiguous indices
  (PEMS03 uses `data/PEMS03/PEMS03.txt`, others `null`).
- `type_='connectivity'` (default) → binary symmetric `A[i,j]=A[j,i]=1`.
- **Spatial graph = connectivity (binary), NOT distance-weighted.**

### 4.4 Temporal / DTW adjacency construction (the "fusion graph" detail)
Reference scripts: `data/fastDTW_adj_gen.py`, `data/stsgcn-dtw.py`, `data/Temporal_Graph_gen.py`.
Pipeline (identical logic, `fastDTW_adj_gen.py` is the PEMS08 concrete version):
1. Load `.npz` `data` → use channel 0 → reshape to `(days, 288, N)`; take **train portion only (first 60%)**.
2. **Normalize** each node series (z-score, per-row mean/std).
3. **Fast DTW** between every node pair `(i,j)`:
   - `compute_dtw(a,b, order=1, Ts=12)`: local cost `d[i,j]=||a_i−b_j||_order` (order `o=1` → L1);
     DP accumulation with a **Sakoe-Chiba–style band** limited to `j ∈ (i−Ts, i+Ts]`, `Ts=12` (=lag);
     returns `D[-1,-1]**(1/o)` = path cost.
   - Result matrix `d`; symmetrize: `adj = d + d.T`.
4. **Thresholding / sparsification**: for each row keep the **top `int(N * adj_percent)` smallest-distance**
   neighbors, `adj_percent = 0.01` (1%). Entries set to `1`.
5. **Symmetrize the binary graph** (make it undirected) and set **self-loops `w_adj[i,i]=1`**.
6. Save to CSV (`adj_PEMS08_001.csv` etc.) — this is the **temporal graph** `A_dtw`.

### 4.5 The STFGNN **fusion graph** `construct_adj_fusion(A_spatial, A_dtw, steps=4)` → `(4N, 4N)`
Builds a `4N×4N` block matrix (steps=4) that fuses spatial + temporal (DTW) graphs, block pattern:
```
        blk0   blk1   blk2   blk3
blk0    A_dtw   I      0      A_dtw
blk1    I       A_sp   I      0
blk2    0       I      A_sp   I
blk3    A_dtw   0      I      A_dtw
```
Rules implemented in code:
- **Diagonal blocks**: block1, block2 = `A_spatial`; block0, block3 = `A_dtw` (`for i in range(4): adj[blk,blk] = A if i in (1,2) else A_dtw`).
- **Temporal band**: adjacent blocks connected by **identity (diagonal) links** `adj[k*N+i,(k+1)*N+i]=1` (bidirectional), for `k=0..2`.
- **Corners**: `adj[3N:4N, 0:N] = A_dtw` and `adj[0:N, 3N:4N] = A_dtw` (the big `T` at the four corners of the diagram).
- **Off-diagonal "1" blocks** (blk(2,0), blk(0,2), blk(1,3), blk(3,1)) = the identity blocks created by the band step.
- Finally force `adj[i,i] = 1` (self-loops) for all rows.

Docstring diagram (from the code):
```
[T, 1, 1, T
 1, S, 1, 1
 1, 1, S, 1
 T, 1, 1, T]      # T = temporal/DTW graph, S = spatial graph
```
Also present: `construct_adj(A, steps)` (simpler degenerate builder, **commented out / unused**).

The `use_mask` option multiplies the fusion `adj` by a **learnable mask** initialized to `(adj_mx != 0)`
(binary), so the graph topology is fixed but edge strengths can adapt.

### 4.6 Sequence generation & normalization (`generate_data` / `generate_seq` / `generate_from_data`)
- Loads `.npz`. If keys `train/val/test` exist → use them; else if key `data` exists →
  **chronological 60% / 20% / 20% split** (`train_line=int(0.6*L)`, `val_line=int(0.8*L)`).
- `generate_seq(data, train_length=12, pred_length=12)`: sliding window over time, concat then split into
  `x,y`, each **`(num_samples, 12, N, 1)`**; only channel `0:1` (flow) kept.
- **Z-score normalization** using the **train split's** mean/std (`generate_from_data` computes mean/std
  from the *first* (train) batch and applies the same to val/test — i.e. train statistics only).
- Metrics: `masked_mae_np`, `masked_mape_np`, `masked_mse_np` (mask = `!= 0` for null_val=0).

---

## 5. Hyperparameters (paper/config defaults)

From `config/*/individual_3layer_12T.json` (identical across PEMS03/04/07/08 except `num_of_vertices`,
`adj_filename`, `graph_signal_matrix_filename`, `ctx`, and PEMS03's `id_filename`):

| Parameter | Value | Notes |
|---|---|---|
| `module_type` | `"individual"` | each STSGCL separate; only this mode implemented |
| `act_type` | `"GLU"` | gated linear unit in GCN |
| `temporal_emb` / `spatial_emb` | `true` / `true` | learnable embeddings (Xavier 0.0003) |
| `use_mask` | `true` | learnable sparsity mask on adj |
| `first_layer_embedding_size` | `64` | input-projection dim |
| `filters` | `[[64,64,64],[64,64,64],[64,64,64]]` | **3 STSGCL layers**, each 3 GCN sub-layers of 64 |
| `batch_size` | `32` | |
| `optimizer` | `"adam"` | MXNet adam, PolyScheduler(pwr=2, warmup=global_train_steps) |
| `learning_rate` | `1e-3` | `base_lr` |
| `epochs` | `200` | |
| `max_update_factor` | `1` | PolyScheduler horizon multiplier |
| `points_per_hour` | `12` | **input window T_in = 12** |
| `num_for_predict` | `12` | **output window T_out = 12** |
| `num_of_features` | `1` | input channels (flow) |
| param init | `Xavier(magnitude=0.0003)` | |
| loss | Huber, `rho=1` | inside symbol graph |
| `num_of_vertices` | 358 / 307 / 883 / 170 | PEMS03/04/07/08 |
| `ctx` | 2 / 3 / 1 / 0 | GPU index per dataset |

Output head has `num_of_filters=128` (the 128-dim FC before the final predict FC).

---

## 6. Dependencies

`requirements.txt`:
```
mxnet-cu100
pytest
graphviz
```
Third-party imports across the code:
- **`mxnet`** (`mx.sym`, `mx.init`, `mx.mod.Module`, `mx.io.NDArrayIter`, `mx.metric`, `mx.lr_scheduler`) — CUDA 10.0 build.
- `numpy`, `pandas`
- standard lib: `os`, `json`, `time`, `argparse`, `csv`
- README says **Python 3**.

No PyTorch, no DGL/PyG. Graph ops are raw `mx.sym.dot` with dense `4N×4N` adjacency (small N).

---

## 7. Structure notes & migration implications

1. **Framework = MXNet symbolic.** adapter must re-author the model in PyTorch (`AbstractTrafficStateModel`).
   No class exists; port the *functions* to `nn.Module` submodules:
   - `STSGCM` (n stacked GCN + crop-middle-N + max)
   - `STSGCL`/`sthgcn_layer_individual` (embedding + gated dilated temporal conv + residual + STSGCM loop)
   - `output_layer` (reshape + FC128 relu + FC→12)
   - top-level `STFGNN` (embedding FC + 3 × STSGCL + output head, loss = Huber).
2. **Fusion graph is the defining innovation** — must be reproduced exactly:
   `get_adjacency_matrix` (binary spatial) + DTW temporal graph (top-1% threshold, self-loops) →
   `construct_adj_fusion(..., steps=4)` → `(4N,4N)`, plus learnable binary-init mask.
   In LibCity this can be baked into the model's config/data step (no dataset is in scope this round,
   so the adapter should implement the fusion-graph builder as a model-side utility).
3. **Permutation-heavy forward**: `(B,T,N,C)` → `(4N,B,C)` for GCN and → `(B,N,T,C)` in the head;
   the temporal conv reduces `T` by 3 per layer (last layer output `T=3`).
4. **Only PEMS08 raw data ships**; other datasets' raw `.npz` must be fetched separately. Pre-built
   adjacency CSVs for all four datasets are available in-repo.
5. **Data layout for LibCity**: input `(B,12,N,1)`, output `(B,12,N)`; normalization z-score from train
   split; chronological 6:2:2 split.
6. `models/__pycache__` holds several alternative variants as bytecode only — ignore; production model
   is `stsgcn_4n_res`.

---

## 8. Quick-reference summary for downstream tasks

```
Model file        : models/stsgcn_4n_res.py   (fn: stsgcn, stsgcm, stsgcl, sthgcn_layer_individual, output_layer, gcn_operation, position_embedding)
Builder           : utils_4n0_3layer_12T_res.py  (construct_model, get_adjacency_matrix, construct_adj_fusion, generate_data, metrics)
Entry             : main_4n0_3layer_12T_res.py --config config/<DS>/individual_3layer_12T.json
Forward           : x (B,12,N,1)  ->  y_pred (B,12,N)
Layers            : 3 × STSGCL, each 3 × GCN(64), module_type=individual, GLU, temporal+spatial emb, mask on 4N×4N fusion adj
Temporal conv     : kernel (1,2), dilate (1,3) -> receptive field 4, out T-3 per layer
Fusion adj        : construct_adj_fusion(spatial_binary, dtw_temporal, steps=4) -> (4N,4N); blocks: [DTW,I,0,DTW; I,S,I,0; 0,I,S,I; DTW,0,I,DTW]
Temporal graph    : DTW(order=1, lag=12) on train split, top-1% neighbors, symmetrize, self-loops=1  (data/fastDTW_adj_gen.py, data/stsgcn-dtw.py)
Hyperparams       : bs=32, lr=1e-3(Adam, Poly pwr2, warmup=steps), epochs=200, filters [[64,64,64]]x3, first_emb=64, head FC=128, Huber rho=1, T_in=12, T_out=12, C=1
Data file         : data/PEMS08/PEMS08.npz  key 'data' shape (17856,170,3) [flow,occupy,speed]; only channel 0 used
Datasets          : PEMS03(358) PEMS04(307) PEMS07(883) PEMS08(170); 5-min; only PEMS08 raw in repo
Deps              : Python3, mxnet-cu100 (MXNet symbolic!), numpy, pandas, graphviz, pytest
Submodules        : none
```
