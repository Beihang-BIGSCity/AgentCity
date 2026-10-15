# metr_la_speed_smoke 模型适配总结

## 任务规格

- 业务场景数据集: METR_LA
- 预测任务类型: traffic_state_pred
- 适配目标: masked_mae ≤ 35.0
- 效率偏好: lightweight
- 训练预算: 15 epochs / batch 32 / GPU False（round 1 为 1 epoch 冒烟；见文末《追加实验》）
- 适配模型: STDiffGRU
- 迭代轮次: 1 / 1
- 最终结论: 达标

## 数据特征画像

- CVglobal: 0.21588565575666047
- CVnode: 0.17629988621669912
- 空间异质性 SH: 0.10945428458618764
- 一阶自相关 rho1: 0.7907617056752304
- 采样间隔(分钟): 5.0
- 节点数 / 时间步: 207 / 34272

## 模块组合方案

- hidden_dim: 32, depth: {'spatial_layers': 1, 'temporal_layers': 1, 'diffusion_steps': 2, 'encoder_layers': 1, 'head_layers': 1}, loss: masked_mae
- encoder: feature_embedding_encoder
- spatial: diffusion_gcn
- temporal: gru
- fusion: concat
- head: linear_head

## 迭代轨迹

- 第 1 轮: masked_mae=3.8778148094813027 (阈值 35.0) -> 达标
  - 判定理由: The achieved value is tuning.json.best_metrics.masked_mae = 3.8778148094813027, the mean of the per-horizon masked_MAE over the 12 prediction horizons on the held-out test split, quoted exactly as recorded. With maximize=false (lower-is-better) and threshold=35.0, 3.8778148094813027 is STRICTLY below 35.0, so the target is met and the round PASSES. The best trial is t1_lr0.005_h32 (learning_rate 0.01 -> 0.005 at hidden_dim=32), the minimum-masked_mae trial among the three recorded trials (t0=3.9
  - 下一步方向: Target is met (PASS, margin 31.12). No further tuning round is required for task success. If a further round were allowed and better quality were wanted (the smoke run is heavily undertrained at max_epoch=1, train_rate=0.1, r2=0.3055): (1) keep the winning direction - learning_rate=0.005 with hidden_dim=32 (the t2 width-64 probe confirmed width 64 buys nothing on 3425 training windows); (2) the strongest remaining lever is training budget, not capacity - raise max_epoch (e.g. 5-10) at the fixed 
- 末次运行: status=success exp_id=metr_la_speed_smoke_stdiffgru_runtester_r1 metrics_source=evaluate_cache metrics={"mae": 8.673194169998169, "mse": 365.57873280843097, "rmse": 19.110634009043377, "masked_mae": 3.901829024155935, "masked_mape": 0.10913111828267574, "masked_mse": 61.22883860270182, "masked_rmse": 7.687232335408528, "r2": 0.13592013891413104, "evar": 0.19900463024775186}

## 阶段摘要

### task_planning

**Plan (round 1, METR_LA traffic_state_pred smoke test)**

- **Understanding**: 12-step-ahead (1 h) traffic-speed forecasting on METR_LA (207 sensors, 5-min sampling); this run is a CPU smoke test of the full chain (data integration → composition → run → tuning → evaluation). "Adapted model" = a model assembled from the atomic module library and registered into LibCity, whose test-split masked_mae (mean over 12 horizons, inverse-scaled to km/h) is strictly below 35.0.
- **Data contract**: input_window 12, output_window 12, null_value 0, standard (z-score) scaling with metrics computed after inverse transform, no imputation — zeros carry through as the LibCity mask (null_ratio ≈ 8.0%); C = 2 features (traffic_speed + time_in_day); METR_LA.rel graph; chronological split 3425 / 1712 / 29112.
- **Composition hint**: one shallow `diffusion_gcn` (or `cheb_gcn`) spatial layer; `gru` temporal with num_layers = 1; cheap concat/add fusion (no cross-attention); single shared linear head over the 12-step horizon; total params ~10⁴ (width 32 preferred).
- **Tuning directions**: lr is the primary knob ({0.01, 0.005}); keep capacity small (width 32 over 64); freeze depth/dropout/diffusion steps/batch 32/1 epoch; ≤ 2 extra knob-isolating trials beyond baseline, one knob per trial; selection = lowest masked_mae from the evaluate cache; trial spread < 2% is noise — don't over-read it; avoid depth, attention, large dilation.
- **Success criterion**: composed model with max_epoch = 1, batch_size = 32, `--gpu false` reaches test masked_mae < 35.0 km/h (mean over 12 horizons), consistent between tuning.json and the LibCity evaluate cache.
- **Risks**: ~10–15 min per CPU run limits trials; interpreter/ray defect may break the native `run_hyper.py` hook (fall back to the custom grid via bash); 1-epoch undertraining on 3425 windows; masked-metric degeneracy; unmasked MAPE = inf is expected null-mask behaviour; test_rate 0.85 dominates runtime; lr-induced divergence (mitigate with grad clipping).
- **Corrections**: `[]` (round 1).

File written: `/home/wangwenrui/shk/AgentCity/data/adaptation/metr_la_speed_smoke/plan.json`

### execution

I'll start by assembling the team, as this is my required opening move.Now let me read the plan and inspect the current team/task state in parallel.The plan is present and valid. The task board is empty but the dataset profile exists. Let me check the task-state files to understand the current state.The plan is valid. The task board is empty and several artifacts are missing, so I need to run the pipeline. The dataset profile exists, and `runs.json` confirms the environment (framework interpreter = `/home/wangwenrui/opt/miniconda3/envs/autogen/bin/python`; base conda python lacks `ray`). Now I'll build the task DAG.Let me construct the task DAG properly.The DAG is correct — a clean sequential chain. Now I'll give the first member the context its task text cannot carry (plan path, pre-existing profile, interpreter caveat) and let it execute.The team is assembled and the execution DAG is in place. Here's the current state:

**Round 1 (of 1) — task `metr_la_speed_smoke` — execution started**

**Pre-flight validation (done by me):**
- `plan.json` — ✅ exists, valid JSON, consistent with the task spec (target `masked_mae < 35.0`, `max_epoch=1`, `batch_size=32`, `train_rate=0.1`, `eval_rate=0.05`, `max_trials=2`, CPU-only).
- `dataset_profile.json` — ✅ exists already (207 nodes, 5-min, null_ratio ≈0.080, split 3425/1712/29112); the data-integrator will confirm/refresh and add the integration contract.
- `runs.json` — confirms the known env pitfall: framework interpreter `/home/wangwenrui/opt/miniconda3/envs/autogen/bin/python` works; base conda python lacks `ray`.
- Missing and to be produced this round: `data_integration.json`, `composition.json`, `run_report.json`, `tuning.json`.

**Delegated task DAG (sequential, one owner each):**
1. `#data-integration` → **data-integrator** (pending, now launched) — dataset profile + integration contract.
2. `#model-composition` → **model-composer** (blocked by #1) — compose encoder/spatial/temporal/fusion/head modules, generate + register the LibCity model, write `composition.json`.
3. `#model-run` → **run-tester** (blocked by #2) — execute on METR_LA, record real evaluate-cache metrics in `run_report.json`.
4. `#model-tuning` → **run-tester** (blocked by #3) — ≤2 knob-isolating trials (lr 0.01/0.005 × hidden 64/32), write `tuning.json`.

I've handed the data-integrator the absolute paths, the plan location, the interpreter caveat and the CPU/gpu=false constraint. Members execute autonomously; I'll validate each artifact (`exists → parses → consistent with plan/profile`) as reports come in, and will re-delegate with a concrete correction if any artifact is missing or inconsistent. I'll report the run status, metrics and the threshold comparison once the chain completes.
The team is assembled and the execution DAG is in place. Here's the current state:

**Round 1 (of 1) — task `metr_la_speed_smoke` — execution started**

**Pre-flight validation (done by me):**
- `plan.json` — ✅ exists, valid JSON, consistent with the task spec (target `masked_mae < 35.0`, `max_epoch=1`, `batch_size=32`, `train_rate=0.1`, `eval_rate=0.05`, `max_trials=2`, CPU-only).
- `dataset_profile.json` — ✅ exists already (207 nodes, 5-min, null_ratio ≈0.080, split 3425/1712/29112); the data-integrator will confirm/refresh and add the integration contract.
- `runs.json` — confirms the known env pitfall: framework interpreter `/home/wangwenrui/opt/miniconda3/envs/autogen/bin/python` works; base conda python lacks `ray`.
- Missing and to be produced this round: `data_integration.json`, `composition.json`, `run_report.json`, `tuning.json`.

**Delegated task DAG (sequential, one owner each):**
1. `#data-integration` → **data-integrator** (pending, now launched) — dataset profile + integration contract.
2. `#model-composition` → **model-composer** (blocked by #1) — compose encoder/spatial/temporal/fusion/head modules, generate + register the LibCity model, write `composition.json`.
3. `#model-run` → **run-tester** (blocked by #2) 

### tuning

I'll start by assembling the team.Team assembled. Let me establish the current state: check the roster and inspect the round artifacts on disk before planning.The disk state differs from the feedback (best trial is now T1 lr=0.005/h32, timestamps ~19:14), and I notice `verdict.json` is **absent** from the working directory. Let me inspect the rest of the artifacts and the previous-run folder.`rounds/previous_run/` holds a full archived round (including a `verdict.json`), while the current working directory has fresh `tuning.json`/`run_report.json` but **no `verdict.json`**. Let me read the archived verdict and the remaining current artifacts.I now have a clear picture. Let me confirm the trial evaluate-caches referenced by the current `tuning.json` actually exist (runtime output, not framework source), so my validation is grounded.The trial caches for the current `tuning.json` all exist (`..._runtester_r1`, `..._tune_lr0005_r1`, `..._tune_h64_r1`). State is clear:

- **Present & consistent**: dataset_profile.json, plan.json, composition.json, data_integration.json, run_report.json (status=success, masked_mae=3.9018), tuning.json (3 trials, best T1 lr=0.005/h32 masked_mae=3.8778).
- **Missing**: `verdict.json` in the working directory — the evaluation stage has not been recorded for this attempt.

Let me inspect the task board before creating tasks.Board is empty. Now I'll create the round-1 task DAG: a read-only audit of the tuning record (no re-run — the CPU budget is spent and the target is met), then the missing evaluation verdict that depends on it.Both tasks created. Let me verify the board.Board is correct: `t1` pending for hyper-searcher, `t2` blocked by `t1` for effect-evaluator. Let me give both owners the situational context their task text can't fully carry (that this is a close-out round with pre-existing artifacts), so nobody burns CPU re-running.Round 1 is dispatched. Current status:

**Assessment of round state:** the previous pipeline stages already completed on disk — `dataset_profile.json`, `plan.json`, `composition.json`, `data_integration.json`, `run_report.json` (status=success, masked_mae=3.9018) and `tuning.json` (3 trials, best T1 lr=0.005/h32, masked_mae=3.8778) are all present and internally consistent, with the trial evaluate-caches verified to exist. The one **missing deliverable is `verdict.json`** in the working directory.

**Delegated DAG (round 1, CPU-only, budget-respecting):**
- `t1-tuning-audit` → **hyper-searcher** (pending): read-only audit that `tuning.json` contains the real search space, trials and best params/metrics — explicitly **no re-run** (the `max_trials=2` budget is spent and the target is already met).
- `t2-eval-verdict` → **effect-evaluator** (blocked by t1): produce `/home/wangwenrui/shk/AgentCity/data/adaptation/metr_la_speed_smoke/verdict.json` with a real numeric comparison against the 35.0 threshold, a strict boolean pass/fail, consistency checks, and the round lesson fed back to the knowledge base/skills.
- `perf-debugger`: intentionally not engaged — the last run succeeded with no NaN/divergence.

I'll report the verdict and wrap up once the members' results land.
I'll start by claiming and reviewing the assigned task.This is a read-only verification task assigned to me. Let me claim it and gather the evidence.The referenced CSV files exist. Now let me read each evaluate-cache CSV to verify the masked_MAE values (mean over the 12 horizons).
Round 1 is dispatched. Current status:

**Assessment of round state:** the previous pipeline stages already completed on disk — `dataset_profile.json`, `plan.json`, `composition.json`, `data_integration.json`, `run_report.json` (status=success, masked_mae=3.9018) and `tuning.json` (3 trials, best T1 lr=0.005/h32, masked_mae=3.8778) are all present and internally consistent, with the trial evaluate-caches verified to exist. The one **missing deliverable is `verdict.json`** in the working directory.

**Delegated DAG (round 1, CPU-only,

## 产物

- 共享执行上下文: /home/wangwenrui/shk/AgentCity/data/adaptation/metr_la_speed_smoke/context.json
- 运行记录: /home/wangwenrui/shk/AgentCity/data/adaptation/metr_la_speed_smoke/runs.json
- 组合方案: /home/wangwenrui/shk/AgentCity/data/adaptation/metr_la_speed_smoke/composition.json
- 评估结论: /home/wangwenrui/shk/AgentCity/data/adaptation/metr_la_speed_smoke/verdict.json
- 适配模型代码: Bigscity-LibCity/libcity/model/traffic_speed_prediction/STDiffGRU.py

---

## 追加实验：15 epoch 对比 + 双分支消融 + 15 epoch 调参（2026-10-15）

### 目的

1 epoch 只能证明链路可用（`masked_mae` 3.90，训练明显不足）。本轮回答三个问题：

1. 公平预算（15 epoch）下组合模型到底能到多少；
2. 组合模型能不能打过它的**单个分支**（消融），以及能不能打过原生单模块基线；
3. 调参阶段在 15 epoch 预算下是否还能找到更好的配置。

### 方法

- **消融位置**：`STDiffGRU.py` 新增 config 驱动的分支开关 `spatial_module` / `temporal_module`
  （取值 `"none"` 表示禁用该分支）。被禁用的分支以**同形状全零张量**参与 fusion，
  因此 fusion 投影与 head 的形状、参数量完全不变，唯一差别是“没有该分支提供的信息”。
  默认值仍为 `diffusion_gcn` / `gru`，即完整组合，行为不变。
- **消融配置**（通过 `--config_file` 传入，模型文件不复制、不派生）：
  - `libcity/config/model/traffic_state_pred/STDiffGRU_abl_temporal_only.json` → 关掉空间分支（只留 GRU 时序分支）
  - `libcity/config/model/traffic_state_pred/STDiffGRU_abl_spatial_only.json` → 关掉时序分支（只留 DiffusionGCN 空间分支）
- **参数规模自检**：完整 18284 / 仅时序 13132 / 仅空间 11948，逐项等于
  `composition.json` 记录的模块参数量（18284−5152=13132，18284−6336=11948），
  说明消融确实只摘掉了目标分支。
- **运行方式**：`run_adaptation`（工具层，记录进 `data/adaptation/metr_la_speed_smoke/runs.json`），
  15 epoch / batch 32 / train_rate 0.1 / eval_rate 0.05 / CPU / `--gpu false`，seed 0、1、2。
  另跑原生 LibCity `RNN`（单模块基线）15 epoch 作为对照。

### 结果

test split，`masked_mae` 为 12 个预测 horizon 的均值（越低越好）：

| 变体 | 参数 | seed 0 | seed 1 | seed 2 | 均值 |
| --- | --- | --- | --- | --- | --- |
| **完整组合** encoder→DiffGCN‖GRU→fusion→head | 18284 | 3.6236 | 3.6270 | 3.6433 | **3.6313** |
| 消融：仅时序（空间分支置零） | 13132 | 3.6699 | 3.6747 | 3.6827 | 3.6757 |
| 消融：仅空间（时序分支置零） | 11948 | 3.7310 | 3.7251 | 3.7347 | 3.7302 |
| 原生单模块基线：LibCity RNN（单个 GRU） | — | 4.6083 | | | 4.6083 |

差值（相对完整组合均值）：

- 完整组合 vs 仅时序：**−0.0444（好 1.21%）**
- 完整组合 vs 仅空间：**−0.0989（好 2.65%）**
- 完整组合 vs 原生 RNN 单模块：**−0.9770（好 21.2%）**

逐 horizon 看（seed 0），完整组合在 12 个 horizon 上**全部**优于两个单分支变体，
说明增益不是某个 horizon 的偶然。同变体跨 seed 的极差 ≤ 0.02，小于变体间差值，
且三个 seed 下的排序完全一致（完整 < 仅时序 < 仅空间 < 原生 RNN）。

1 epoch → 15 epoch 的提升：组合模型 3.9018 → 3.6313（−6.9%），
原生 RNN 5.6829 → 4.6083（−18.9%）；预算增加对两者都有收益，组合模型仍然领先。

### 15 epoch 调参（`tune_adaptation`，custom_grid）

按 1 epoch 证据缩小搜索空间：`learning_rate=[0.005]`、`hidden_dim=[32, 64]`（2 trials）。

| trial | learning_rate | hidden_dim | masked_mae |
| --- | --- | --- | --- |
| t2 | 0.005 | 32 | 3.6648 |
| **t1（best）** | **0.005** | **64** | **3.6134** |

- 最佳值 3.6134 与 15 epoch 基线 3.6313 相差 0.018，落在跨 seed 波动（≤0.02）之内，
  即 15 epoch 预算下**容量的收益已经饱和**；1 epoch 时“宽度 32 更好”的结论在 15 epoch 下反转，
  说明小预算下的调参结论不能外推到更大预算。
- 该轮同时修掉一个真实缺陷（见下），调参阶段现在能真正跑通。

### 本轮修掉的缺陷

`run_custom_grid_search` 把搜索空间里的**每个**键都当作 `run_model.py` 的命令行参数转发。
`run_model.py` 只声明了固定的一组 flag（`learning_rate`、`batch_size`、`max_epoch`、`train_rate`、
`eval_rate`、`seed`、`exp_id`、`gpu`、`gpu_id`、`dataset_class`、`executor`、`evaluator`、`task`、
`model`、`dataset`、`config_file`、`saved_model`、`train`），
模型超参（`hidden_dim`、`num_layers`、`dropout`、`num_diffusion_steps` …）**不在其中**：
argparse 直接以 `unrecognized arguments: --hidden_dim 32` 终止进程，两个 trial 都在训练前失败（
`tuning.json` 因此记录 `status=error`；round 1 是靠手写 `tuning_stdiffgru_h64.json` + `--config_file` 绕开的）。

修复：

- `agents/adaptation/tuning.py` 新增 `RUN_MODEL_CLI_FLAGS`、`write_trial_config()`、`trial_command()`：
  非 CLI 键写入 `tuning_configs/` 下的 trial 配置并通过 `--config_file` 传入（与已有 `config_file` 合并），
  CLI 键仍走命令行（保持 LibCity “命令行 > 配置文件 > 默认”的优先级）。
- `agents/tools/impl.py` 的 `run_adaptation` 增加 fail-loud 守卫：`extra_cli_args` 里出现模型超参时
  直接给出可读错误，而不是让 argparse 抛出噪声。
- `scripts/validate_adaptation.py` 新增 `check_tuning_cli_contract()`（15 项断言）：
  把 `RUN_MODEL_CLI_FLAGS` 与 vendored LibCity 源码（`run_model.py` + `argument_list.py`）交叉校验，
  并断言 trial 命令不会把模型超参当 flag 转发、base config 合并后在、CLI 键不写进配置文件。

### 产物

- 运行记录（含消融、seed 重复、调参）：`data/adaptation/metr_la_speed_smoke/runs.json`
- 调参结论：`data/adaptation/metr_la_speed_smoke/tuning.json`（best `learning_rate=0.005`, `hidden_dim=64`, masked_mae 3.6134）
- 消融配置：`Bigscity-LibCity/libcity/config/model/traffic_state_pred/STDiffGRU_abl_temporal_only.json`、`STDiffGRU_abl_spatial_only.json`
- trial 配置（自动生成，作为证据保留）：`Bigscity-LibCity/tuning_configs/`
- 原始日志：`Bigscity-LibCity/libcity/log/metr_la_speed_smoke_stdiffgru_ep15_*.log`

### 已知限制

- 单机 CPU、无 GPU；三个 seed 只用于估计波动，不是严格的显著性检验。
- 消融保留 fusion 层（零张量走同一 Linear），衡量的是“去掉该分支的信息”，不是“去掉该分支的算力”。
- 15 epoch 是本次实验预算，不是收敛预算；更长训练下的排序未验证。
