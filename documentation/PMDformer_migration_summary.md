# PMDformer Migration Summary (Phase 3: Configure)

## Migrated Files

1. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
   - Added `PMDformer` to `traffic_state_pred.allowed_model`.
   - Added `PMDformer` task mapping:
     - `dataset_class`: `TrafficStatePointDataset`
     - `executor`: `TrafficStateExecutor`
     - `evaluator`: `TrafficStateEvaluator`

2. `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traffic_state_pred/PMDformer.json`
   - Uses PMDformer traffic defaults aligned to upstream run scripts.

## Config Defaults Chosen

### Source references
- Upstream argument defaults:
  - `/home/wangwenrui/shk/AgentCity/repos/PMDformer/run.py`
- Upstream traffic experiment script:
  - `/home/wangwenrui/shk/AgentCity/repos/PMDformer/scripts/PMDformer/traffic.sh`

### Adopted defaults in LibCity model config
- `model_name`: `PMDformer`
- `input_window`: `720`
  - from `traffic.sh` `seq_len=720`
- `output_window`: `96`
  - from first traffic run in `traffic.sh` (`pred_len=96`) as baseline horizon
- `input_dim`: `1`
- `output_dim`: `1`
  - consistent with LibCity traffic speed one-target setup and current PMDformer adapter
- `d_model`: `512`
  - from `traffic.sh` for the 96-step traffic run
- `n_heads`: `8`
  - from `traffic.sh` for the 96-step traffic run
- `e_layers`: `3`
  - from `traffic.sh`
- `v_layers`: `1`
  - from `traffic.sh`
- `d_ff`: `512`
  - from `traffic.sh`
- `dropout`: `0.1`
  - from `run.py` default and `traffic.sh`
- `patch_size`: `72`
  - from `traffic.sh`
- `use_norm`: `true`
  - from `run.py` default
- `max_epoch`: `50`
  - from `traffic.sh` (`train_epochs=50`)
- `batch_size`: `32`
  - from 96-step traffic run in `traffic.sh`
- `learning_rate`: `0.001`
  - from 96-step traffic run in `traffic.sh`
- `patience`: `5`
  - from `traffic.sh`

### LibCity-compatible training extras
- `learner`: `adam`
- `weight_decay`: `0`
- `lr_epsilon`: `1e-8`
- `lr_decay`: `true`
- `lr_scheduler`: `multisteplr`
- `lr_decay_ratio`: `0.5`
- `steps`: `[25, 40]`
- `clip_grad_norm`: `true`
- `max_grad_norm`: `5`
- `use_early_stop`: `true`

## Known Constraints / Compatibility Notes

1. Task placement
   - LibCity uses task `traffic_state_pred` with model configs under:
     `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traffic_state_pred/PMDformer.json`
   - This is correct even though the implementation module lives under `traffic_speed_prediction`.

2. Dataset assumption
   - Registered with `TrafficStatePointDataset`, so it is intended for standard point-based traffic-state forecasting datasets such as `METR_LA`, `PEMS_BAY`, and related PEMS point datasets.

3. Feature-channel behavior
   - The current adapter uses the first `output_dim` channels from LibCity input and predicts `output_dim` channels back.
   - Therefore `output_dim <= feature_dim` must hold.
   - For the default traffic-speed setup, `output_dim=1` is the safest choice.

4. Patch/window constraints
   - Must satisfy `input_window >= patch_size`.
   - For clean non-overlapping patching, choosing `input_window` divisible by `patch_size` is recommended; the default `720/72` satisfies this exactly.

5. Attention dimension constraint
   - Must satisfy `d_model % n_heads == 0`; the default `512 % 8 == 0` is valid.

6. Practical runtime note
   - `input_window=720` and `d_model=512` are heavy defaults copied from the upstream traffic benchmark and may require smaller batch sizes on limited-memory GPUs.
