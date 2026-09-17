# AGCRN Migration Summary (LibCity)

## Overview

Migration of **AGCRN** — *Adaptive Graph Convolutional Recurrent Network for Traffic
Forecasting* (NeurIPS 2020) — from the original standalone repository into the LibCity
framework as a traffic state (speed) prediction model.

- **Paper**: "Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting"
  (https://arxiv.org/abs/2007.02842)
- **Original repository**: https://github.com/LeiBAI/AGCRN (cloned at `./repos/AGCRN`)

## Source Information

Original files (all under `./repos/AGCRN/model/`):

| Original file | Class(es) | Role |
|---------------|-----------|------|
| `AGCN.py` | `AVWGCN` | Adaptive Vertex-Weighted Graph Convolution (core layer) |
| `AGCRNCell.py` | `AGCRNCell` | GRU-style recurrent cell built from two `AVWGCN` layers |
| `AGCRN.py` | `AVWDCRNN`, `AGCRN` | Encoder stack + Conv2d multi-step head |
| `Run.py` | — | standalone training script (argparse, weight init, optimizer) |
| `BasicTrainer.py` | — | training/validation loop |
| `PEMSD4_AGCRN.conf`, `PEMSD8_AGCRN.conf` | — | configparser INI hyper-parameters |

Original task: short-term **traffic flow** prediction, multi-step (12 in → 12 out @ 5-min),
single channel (`input_dim = output_dim = 1`).

## Target Files

- **Created**: `Bigscity-LibCity/libcity/model/traffic_speed_prediction/AGCRN.py`
  - Consolidated `AVWGCN`, `AGCRNCell`, `AVWDCRNN` and the `AGCRN` model class into one file.
- **Modified**: `Bigscity-LibCity/libcity/model/traffic_speed_prediction/__init__.py`
  - Added `from libcity.model.traffic_speed_prediction.AGCRN import AGCRN`
  - Added `"AGCRN"` to `__all__`
- **Not modified** (already present, reused):
  - `libcity/config/model/traffic_state_pred/AGCRN.json` (default hyper-parameters)
  - `libcity/config/task_config.json` (`AGCRN` → `TrafficStatePointDataset` /
    `TrafficStateExecutor` / `TrafficStateEvaluator`, already listed in
    `traffic_state_pred.allowed_model`)

## Model Architecture (unchanged logic)

1. **Adaptive adjacency from node embeddings** — a single learnable matrix
   `node_embeddings = nn.Parameter(torch.randn(N, embed_dim))` is shared by every
   `AVWGCN` call:
   ```python
   supports = F.softmax(F.relu(node_embeddings @ node_embeddings.T), dim=1)  # (N, N)
   support_set = [I_N, supports]
   for k in range(2, cheb_k):
       support_set.append(2 * supports @ support_set[-1] - support_set[-2])
   supports = torch.stack(support_set, dim=0)                                 # (cheb_k, N, N)
   ```
2. **Node-adaptive weights** (weight pooling + `einsum`):
   `weights = einsum('nd,dkio->nkio', node_embeddings, weights_pool)`,
   `bias = node_embeddings @ bias_pool`,
   `x_gconv = einsum('bnki,nkio->bno', x_g, weights) + bias`.
3. **`AGCRNCell`** — GRU-style cell using a `gate` (`2*hidden` out) and an `update`
   (`hidden` out) `AVWGCN`.
4. **`AVWDCRNN`** — `num_layers` stacked cells with an explicit Python time loop;
   layer 0 maps `input_dim → hidden`, deeper layers `hidden → hidden`.
5. **Head** — `nn.Conv2d(1, output_window * output_dim, kernel_size=(1, hidden_dim))`
   applied to the last hidden state → **non-autoregressive** multi-step prediction.

## Key Changes / Adaptations

| Aspect | Original | LibCity adaptation |
|--------|----------|--------------------|
| Entry point | `Run.py` instantiates `Network(args)` from an `argparse.Namespace` | `AGCRN(config, data_feature)` inherits `AbstractTrafficStateModel` |
| Input | `forward(source, targets, teacher_forcing_ratio)` | `forward(batch)` reads `batch['X']` (channels-last `(B, T_in, N, F)`, **no transpose**) |
| Output | `(B, horizon, N, output_dim)` | same `(B, output_window, N, output_dim)` |
| Horizon / windows | `args.lag`, `args.horizon` | `config['input_window']` / `config['output_window']` (with `lag`/`horizon` fallbacks) |
| Node count / channels | `args.num_nodes`, `args.input_dim` | `data_feature['num_nodes']`, `data_feature['feature_dim']` |
| Output channels | `args.output_dim` | `data_feature['output_dim']` |
| Graph | `default_graph` flag (adjacency was unused anyway) | not required — graph is fully learned; `adj_mx` intentionally ignored |
| Loss | `torch.nn.L1Loss` / masked MAE on inverse-transformed values | `loss.masked_mae_torch(y_pred, y_true, null_val=0.0)` after `self._scaler.inverse_transform` |
| Weight init | `Run.py`: `xavier_uniform_` if `dim > 1` else `uniform_` | reproduced verbatim in `AGCRN._init_parameters()` |
| Optimizer | `Adam(lr=0.003, eps=1e-8, weight_decay=0)` | provided by LibCity config (`learning_rate = 0.003` in `AGCRN.json`) |
| Device | hard-coded `cuda:0` | CPU-safe: no hard-coded `.cuda()`; LibCity's executor moves the module |

## Config Keys Read by the Model

Read from `config` (LibCity `ConfigParser`, all with defensive defaults):

| Key | Default | Meaning |
|-----|---------|---------|
| `input_window` | `lag` or `12` | encoder input length `T_in` |
| `output_window` | `horizon` or `12` | prediction horizon `T_out` |
| `lag` | `12` | fallback for `input_window` (original repo naming) |
| `horizon` | `12` | fallback for `output_window` (original repo naming) |
| `rnn_units` | `64` | hidden dimension of the recurrent cells |
| `embed_dim` | `10` | node-embedding dimension (drives the adaptive graph) |
| `num_layers` | `2` | number of stacked `AGCRNCell`s |
| `cheb_order` | `cheb_k` or `2` | Chebyshev polynomial order `K` |
| `cheb_k` | `2` | fallback for `cheb_order` |
| `input_dim` | `feature_dim` | encoder input channels (original repo naming) |
| `device` | `torch.device('cpu')` | torch device |

Read from `data_feature`:

| Key | Default | Meaning |
|-----|---------|---------|
| `num_nodes` | `1` | number of road sensors `N` |
| `feature_dim` | `1` | input channels |
| `output_dim` | `1` | predicted channels |
| `scaler` | `None` | LibCity scaler used to inverse-transform for the loss |

The pre-existing `libcity/config/model/traffic_state_pred/AGCRN.json` already supplies
`embed_dim`, `rnn_units`, `num_layers`, `cheb_order`, `learning_rate=0.003`, etc.

## Public Interface

```python
class AGCRN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature)
    def forward(self, batch)         # batch['X'] -> (B, output_window, N, output_dim)
    def predict(self, batch)         # == forward (non-autoregressive)
    def calculate_loss(self, batch)  # masked MAE in original value space
```

Helper modules in the same file: `AVWGCN`, `AGCRNCell`, `AVWDCRNN`.

## Usage

```bash
cd Bigscity-LibCity
python run_model.py --task traffic_state_pred --model AGCRN --dataset PEMSD4
```

## Assumptions and Limitations

1. **Duplicate registration / name shadowing (important).**
   An AGCRN port already exists at
   `libcity/model/traffic_flow_prediction/AGCRN.py` and is registered there.
   LibCity's `get_model` (`libcity/utils/utils.py`) resolves the `traffic_state_pred`
   task by trying `traffic_flow_prediction` **first** and only falling back to
   `traffic_speed_prediction`. Consequently, at runtime the *flow* module's `AGCRN`
   takes precedence and this new speed-prediction class is shadowed unless the flow
   registration is removed/renamed. Both implementations are functionally equivalent.
   If the speed-prediction variant must be used, drop `AGCRN` from
   `traffic_flow_prediction/__init__.py`, or resolve it explicitly.
2. **No adjacency matrix.** The model learns the graph from node embeddings, so
   `adj_mx` is never used. `bidir_adj_mx` in `AGCRN.json` is harmless but unused.
3. **Loss parity.** The original repo used plain `L1Loss` on inverse-transformed values
   (`loss_func = mae`). This port uses `masked_mae_torch(..., null_val=0.0)`, matching
   the existing LibCity AGCRN port and standard LibCity traffic-state models; zeros are
   masked out, which slightly differs from the original unmasked MAE.
4. **External features.** `input_dim` defaults to `feature_dim`; the default AGCRN
   config disables external features (`load_external=false`, `add_time_in_day=false`,
   `add_day_in_week=false`). Enabling them changes `feature_dim` and thus the encoder's
   input channels, which is supported but untested here.
5. **No temporal embedding.** AGCRN does not consume time-of-day / day-of-week
   embeddings, so those flags can stay off.
6. `horizon`/`output_window` and `num_node`/`num_nodes` are both exposed as attributes
   for source-fidelity with the original code.

## Testing Notes

The model file was added and registered; import was checked statically (no execution
sandbox available in this environment). Verify with:

```python
from libcity.model.traffic_speed_prediction.AGCRN import AGCRN  # imports cleanly
```

and a smoke forward pass with `X` of shape `(B, input_window, N, feature_dim)`
expecting output `(B, output_window, N, output_dim)`.
