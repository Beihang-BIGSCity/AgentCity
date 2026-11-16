# Runtime configuration

The task JSON is the source of truth. CLI `--max-epoch`, `--max-rounds` and
`--batch-size` explicitly override it. Shell activation does not set training budgets.

| Setting | Adaptation loops | Final retraining / test |
| --- | --- | --- |
| Epochs | 35 per baseline and trial; no early stopping | LibCity TrafficStateExecutor default: max 100, early stopping enabled, patience 50 |
| Train / validation / test | 0.7 / 0.1 / 0.2 | Same |
| Split algorithm | Native LibCity sample-window split | Same |
| Hyperparameter trials | 5 per loop | Frozen winning hyperparameters |
| Loops | 6, even if the threshold is reached early | One frozen-model retraining |
| Model selection | Validation metrics only | Test partition evaluated after retraining |
| Composition | One encoder, spatial, temporal, fusion and head | Freeze the selected composition |

Default ratios and final epoch/early-stopping settings are read from the vendored
LibCity JSON defaults. All six task presets use this protocol. Historical names
containing `smoke` or dates remain for compatibility; they no longer imply a short pilot.
`candidate_scope` still controls candidate categories/models. There is no implicit
0.1M parameter ceiling. Capacity follows validation evidence and actual resources.

Each loop trains its own composition, then runs a five-trial custom grid search.
Provide at least five distinct combinations. Epoch and split ratios cannot be tuned
or overridden by tool-call arguments. A failed trial consumes one trial and cannot
win selection. The fixed grid backend avoids a second five-trial fallback search.
No baseline is reused across potentially different compositions.

```bash
source scripts/activate_local.sh
python main.py --task tasks/metr_la_speed_smoke.json --gpu on
```

After all six loops, select and freeze the winning composition and its parameter
JSON using validation evidence. The final command verifies the loop history before
training; `--dry-run` prints the command without training or test evaluation.

```bash
python scripts/run_final_evaluation.py \
  --task tasks/metr_la_speed_smoke.json --model SelectedModel \
  --config-file /absolute/path/to/frozen-winning-config.json --gpu on --dry-run
```

Remove `--dry-run` to retrain and evaluate. Final configuration explicitly applies
LibCity's epoch and early-stopping defaults, so a generated model's old 1/35-epoch
value cannot leak into final training. The final test result must not feed back into
composition or tuning.

`AGENTCITY_EPOCHS` and `AGENTCITY_FIXED_ROUNDS` are no longer configuration sources.
Use task `train.max_epoch`, `max_rounds`, and `fixed_rounds`. `AGENTCITY_SERIAL_ROLES`
and `AGENTCITY_RESUME` remain optional execution controls. CLI orchestration scopes
validation-only/native-split settings to the run and restores the caller environment.
GPU-off takes precedence over a lingering `AGENTCITY_DEVICE` setting.
