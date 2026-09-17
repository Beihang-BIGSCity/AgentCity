# CPiRi Migration Summary

## Overview
CPiRi has been migrated into LibCity for traffic speed prediction. The model integration, configuration registration, and a basic execution test were completed in the target workspace.

## Source/Target
- Source repository: `/home/wangwenrui/shk/AgentCity/repos/CPiRi`
- Target framework: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity`
- Target task type: `traffic_speed_prediction`
- Migrated model name: `CPiRi`

## Files changed
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_speed_prediction/CPiRi.py`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/traffic_speed_prediction/__init__.py`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
- `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traffic_state_pred/CPiRi.json`

## Configuration choices
Initial test encountered CUDA OOM. Configuration was tuned to reduce memory use:
- `batch_size: 8`
- `hidden_size: 256`
- `heads: 8`
- `intermediate: 1024`

## Testing results
- Dataset: `METR_LA`
- Runtime: GPU
- Validation run: `max_epoch=1`
- Result: run completed successfully after configuration tuning

Sample metrics:
- Horizon 1: MAE `14.225272`, RMSE `12.978363`, R2 `-0.181990`, EVAR `0.063821`
- Horizon 12: MAE `14.546106`, RMSE `13.419259`, R2 `-0.213248`, EVAR `0.039145`
- MAPE: reported as `inf`

## Known limitations/caveats
- Reported MAPE is `inf`, indicating instability when true values are near zero.
- Current validation evidence is limited to a short run (`max_epoch=1`), so metrics are not final convergence quality.
- Performance metrics (negative R2) indicate room for further tuning.

## Next steps
- Run longer training with multiple seeds for stable evaluation.
- Investigate and mitigate `inf` MAPE (e.g., masking/epsilon strategies in evaluation).
- Tune model and optimization hyperparameters for improved predictive accuracy.
- Validate on additional datasets to confirm generalization and robustness.
