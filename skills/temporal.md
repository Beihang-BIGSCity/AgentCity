# 时间表征调参 Skills

面向时空交通预测的「数据特征画像 → 时间模块与超参方向」经验库（时间表征维度）。
配套 `skills/index.json` 中的 `dimension=temporal` 条目；检索按 `feature_profile` 做确定性数值最近邻。
所有方向都对齐 LibCity 的既有旋钮（`input_window` / `output_window` / `rnn_units` / `filters` /
`dilation` / `dropout` / `learning_rate` / `lr_scheduler` / `clip_grad_norm`），并注明 CPU-only 预算。

## 高持续性数据用门控循环（GRU/LSTM）

**适用数据特征**：`rho1 >= 0.85`、`cv_global <= 0.25`，序列近似平滑、不存在强事件激增。本仓库四个真实数据集
实测 rho1 分别为 METR_LA 0.791、PEMS_BAY 0.917、PEMSD4 0.952、PEMSD7 0.966，除 METR-LA 外全部落在该区间。
`node_count` 与采样间隔影响不大，`sampling_interval_min` 越大越倾向 LSTM 而非 GRU。

**调参方向**
- `rnn_units 64 -> 96`：DCRNN 默认 `rnn_units=64`；rho1 > 0.9 时加宽隐层通常比加深更划算。
- `num_rnn_layers 2 -> 3`：DCRNN 与 AGCRN 默认均为 2；超过 3 层收益递减。
- 先 GRU，末端 horizon 误差偏大再换 LSTM（ST-LEGO §App.B：GRU 更快更紧凑，LSTM 长序列更好但更耗资源）。
- `clip_grad_norm=true` + `max_grad_norm=5`（DCRNN 默认）；`batch_size=64` 保持。
- CPU-only：`max_epoch` 按任务规格（冒烟 <=5），循环结构逐时刻计算，是 CPU 上最吃亏的时间模块，不要堆层数。

**依据/证据**
- ST-LEGO §3.1：`rho1` 越高说明序列连续性越强，倾向 GRU 一类门控循环结构。
- LibCity `DCRNN.json`：`rnn_units=64`、`num_rnn_layers=2`、`learning_rate=1e-2`、`clip_grad_norm=true(max_grad_norm=5)`。
- LibCity `AGCRN.json`：`rnn_units=64`、`num_layers=2`、`learning_rate=3e-3`。

**参考来源**：ST-LEGO §3.1 / §App.B；LibCity `config/model/traffic_state_pred/DCRNN.json`、`AGCRN.json`。

## 多尺度周期波动用膨胀门控 TCN

**适用数据特征**：`cv_global >= 0.35` 且存在早晚高峰多尺度周期；`rho1` 中等（0.4–0.7）说明既有周期性又有
非周期性抖动。PEMSD7 实测 cv_global 0.609 属典型大波动；PEMSD4 虽然 cv_global 只有 0.132，但其日周期
结构稳定，同样可从膨胀卷积中获益。

**调参方向**
- `dilation` 按 2^n 展开覆盖输入窗：`T'=12` 用 1,2,4,8；`T'=24` 时把 1,2,4,8 堆叠两层。
- `kernel_size=2` 保持（STGCN / STNorm 默认），指数膨胀即可达到满感受野，不要加宽卷积核。
- `dropout 0 -> 0.1..0.3`（GWNET / MTGNN 默认 `0.3` 为上界）；大波动数据过拟合更快。
- `learning_rate 1e-3` 起步；若 2 个 epoch 内 valid 分数噪声很大则降到 `5e-4`。
- CPU-only：TCN 可并行，是 CPU 上性价比最高的时间模块；`max_epoch` 仍受任务预算约束。

**依据/证据**
- ST-LEGO §3.1：`CVglobal` 越大波动越强，鼓励 TCN 或 Transformer 一类更具表达力的时间编码器。
- ST-LEGO §App.B：DilatedTCN「指数膨胀、快速可并行、适合长依赖」；GLU-TCN 为其门控版本。
- ST-LEGO §4.6：效率导向 prompt 同样偏好 DilatedTCN（兼顾效率与表达力）。
- LibCity `STGCN.json`：`Kt=3`、`blocks=[[1,32,64],[64,32,128]]`；`MTGNN.json`：`dilation_exponential=1`、`layers=3`。

**参考来源**：ST-LEGO §3.1 / §App.B / §4.6；LibCity `STGCN.json`、`MTGNN.json`、`STNorm.json`。

## 长视界与弱持续性用时间 Transformer

**适用数据特征**：`rho1 <= 0.6`（序列记忆弱，需要直接对齐远距离时刻），或 `horizon` 很长需要长上下文；
`cv_global` 偏高且 `node_count` 中等（<=600）以便承担 O(T²) 开销。

**调参方向**
- `num_heads 4 -> 8`，`hidden_dim 64 -> 128`；表达力随宽度增长，但必须核算 CPU 成本。
- `dropout 0.1 -> 0.2/0.3`（GWNET / MTGNN 默认 `0.3`）：注意力在大波动数据上最易过拟合。
- `learning_rate 5e-4` 起（注意力对 lr 敏感），配 warmup + `lr_scheduler=steplr`、`lr_decay_ratio=0.7`、`step_size=5`。
- `input_window 12 -> 24/48`：只有足够长上下文，注意力才优于卷积；短窗下不要选用。
- CPU-only：注意力时间开销随 T² 增长，绝不要与 N² 的注意力空间模块叠加。

**依据/证据**
- ST-LEGO §3.1：`CVglobal` 大 → 需要更强时间编码器；§4.6 表达力导向 prompt 引导选择 Transformer 与 GLU-TCN。
- ST-LEGO §4.8：METR-LA 上去掉统计特征使 GPT-4o MAE 上升 0.69，说明结构-任务匹配直接决定精度。
- LibCity `STGCN.json`：`lr_scheduler=steplr`、`lr_decay_ratio=0.7`、`step_size=5` 是现成的衰减配方。
- LibCity `STNorm.json`：`learning_rate=1e-4`、`patience=5` —— 注意力类结构更小步长的参考。

**参考来源**：ST-LEGO §3.1 / §4.6 / §4.8；LibCity `STGCN.json`、`STNorm.json`。

## CPU 单轮冒烟：高持续性数据上 diffusion+GRU 组合优先调 learning_rate/hidden_size 而非深度

- 来源: AgentCity adaptation tuning stage, round 1, task metr_la_speed_smoke (model STDiffGRU: encoder->diffusion_gcn||gru->concat->linear_head; deliverable tuning.json)
- 数据特征画像: {"dataset": "METR_LA", "task": "traffic_state_pred", "data_col": "traffic_speed", "node_count": 207, "has_graph": true, "cv_global": 0.21588565575666047, "cv_node": 0.17629988621669912, "spatial_heterogeneity": 0.10945428458618764, "rho1": 0.7907617056752304, "null_ratio": 0.08037800466402342, "cv_window_mean": 0.2156654647391049, "cv_window_std": 0.009248575660567906, "input_window": 12, "output_window": 12, "budget": {"max_epoch": 1, "batch_size": 32, "train_rate": 0.1, "eval_rate": 0.05, "max_trials": 2, "gpu": false}}
- 调参方向:
  - 极高持续性(cv_global~0.216, rho1~0.79, 窗口CV稳定±0.009) 但训练样本仅3425 -> CPU 1-epoch 预算下只调 learning_rate 与 hidden_size(隐藏宽度)，不加深度；lr 试 {0.01, 0.005}，hidden_size 试 {64, 32}，batch_size/nulls/scaler/loss/lr_decay 全部固定以做 apples-to-apples
  - hidden_size 64 -> 32: 参数量 63180 -> 18284, masked_MAE 3.9528 -> 3.9018(本回合最低)，验证‘小容量’方向
  - learning_rate 0.01 -> 0.005 @ hidden=64: masked_MAE 基本持平(3.9528->3.9515) 但 masked_RMSE/R2 明显改善(7.675->7.435, 0.133->0.278) —— 1-epoch 下 lr 对 MAE 的边际效应很小
- 策略:
  - 适用：CPU-only、1-epoch、小 train_rate 的冒烟/门禁轮次，目标是打通调参链并给出可复现的 best_params，而非收敛精度
  - 适用：diffusion+GRU 组合、强时间持续性数据；只调 learning_rate 与隐藏宽度这两个‘每单位时间收益最高’的 knob，其余固定
  - 不适用：充足 GPU/多 epoch——那时才值得扩到 depth/层数网格(如 env 先验 layers 2->3)
  - 复用数据缓存：hidden_dim/lr 不改变 LibCity npz 缓存名，可在同一缓存上连跑多个 trial，显著省时
- 证据: ["T0 baseline (lr=0.01, hidden=64, 63180 params): masked_mae=3.95281, masked_rmse=7.67532, masked_mape=0.114852, r2=0.133143 (source: libcity/cache/metr_la_speed_smoke_stdiffgru/evaluate_cache/2026_10_14_18_18_20_STDiffGRU_METR_LA.csv)", "T1 (lr=0.005, hidden=64): masked_mae=3.95146, masked_rmse=7.43505, masked_mape=0.110057, r2=0.277805 (real run exp_id metr_la_speed_smoke_stdiffgru_lr0005)", "T2 (lr=0.01, hidden=32, 18284 params): masked_mae=3.90183, masked_rmse=7.68723, masked_mape=0.109131, r2=0.135920 (real run exp_id metr_la_speed_smoke_stdiffgru_h32, config_file tune_hidden32.json {hidden_dim:32})", "All three trials PASS masked_mae<=35.0 with huge margin; best = T2 (lr=0.01, hidden=32) masked_mae=3.9018, consistent with the small-capacity prior for 3425 training windows", "Environment defect: tune_adaptation hook errored ('LibCity tuner skipped: LibCity tuner returned error', no artifacts written) because the hook selected the base-conda interpreter lacking ray; trials were run via bash with the framework interpreter /home/wangwenrui/opt/miniconda3/envs/autogen/bin/python (ray 2.24.0)"]

## METR_LA 1-epoch CPU 冒烟评估：阈值判断稳健，hidden_size 64->32 降 masked_mae，lr 0.01->0.005 改善 masked_rmse/R2/EVAR 但 masked_mae 持平

- 来源: run:metr_la_speed_smoke#r1
- 数据特征画像: {"budget": {"batch_size": 32, "gpu": false, "max_epoch": 1, "max_trials": 2, "train_rate": 0.1, "eval_rate": 0.05}, "cv_global": 0.21588565575666047, "cv_node": 0.17629988621669912, "cv_window_mean": 0.2156654647391049, "cv_window_std": 0.009248575660567906, "data_col": "traffic_speed", "dataset": "METR_LA", "input_window": 12, "node_count": 207, "null_ratio": 0.08037800466402342, "output_window": 12, "rho1": 0.7907617056752304, "spatial_heterogeneity": 0.10945428458618764}
- 调参方向:
  - hidden_size (STDiffGRU hidden_dim): at lr=0.01, 64 -> 32 lowered masked_mae 3.9528136054674783 -> 3.901829024155935 and halved parameters 63180 -> 18284; small capacity is the right direction with only 3425 training windows
  - learning_rate: at hidden=64, 0.01 -> 0.005 left masked_mae essentially unchanged (3.9528136054674783 -> 3.9514551361401877) but clearly improved masked_rmse 7.675324360529582 -> 7.43504798412323, R2 0.13314296218270413 -> 0.2778045750828821 and EVAR 0.2326603283484777 -> 0.3281637380520503
  - keep fixed: num_layers=1, dropout=0.0, num_diffusion_steps=2, batch_size=32, max_epoch=1, scaler=standard, loss_type=masked_mae, clip_grad_norm (max_grad_norm=5); do NOT tune architectural depth or add cross-attention fusion on this CPU 1-epoch budget
- 策略:
  - 适用：CPU-only、1-epoch、小 train_rate 的冒烟/门禁轮次——判 pass/fail 用阈值，不要把 trial 排序当强结论
  - 适用：diffusion+GRU 组合在高持续性数据上，只调 learning_rate 与隐藏宽度(hidden_size)，其余固定做 apples-to-apples
  - 不适用：充足 GPU/多 epoch——那时 masked_mae 差异才会显著，可扩到 depth/层数网格
- 证据: [{"knob": "hidden_size", "effect": "64->32 at lr=0.01: masked_mae 3.9528136054674783 -> 3.901829024155935 (lower is better, improved); params 63180 -> 18284", "trials_compared": ["T0", "T2"]}, {"knob": "learning_rate", "effect": "0.01->0.005 at hidden=64: masked_mae 3.9528136054674783 -> 3.9514551361401877 (flat, -0.001 over noise); masked_rmse 7.675324360529582 -> 7.43504798412323 (better); R2 0.13314296218270413 -> 0.2778045750828821 (better); EVAR 0.2326603283484777 -> 0.3281637380520503 (better)", "trials_compared": ["T0", "T1"]}, {"selection_rule": "best = lowest masked_mae; winner T2 (lr=0.01, hidden=32) masked_mae=3.901829024155935", "all_trials_pass": true, "target": {"metric": "masked_mae", "threshold": 35.0, "maximize": false}, "verdict": "PASS margin 31.098", "caveat": "the 3 trials span 3.9018-3.9528 (<2%), within run-to-run noise, so the ranking of T2 over T0/T1 is not statistically strong; all comfortably clear 35.0"}, {"evaluation_lesson": "with max_epoch=1 the masked_mae differences between lr/hidden settings are minute, so judge pass/fail against the (loose) threshold and treat trial ranking as directional only; masked_rmse/R2/EVAR can move a lot (e.g. lr 0.005 improved R2 0.133->0.278) even when masked_mae is flat"}, {"environment_defect": "run_adaptation and tune_adaptation selected the base-conda interpreter (no ray) instead of envs/autogen (ray 2.24.0): 'ModuleNotFoundError: No module named ray'; trials had to be run directly via bash with the framework interpreter"}]

## METR_LA CPU 1-epoch smoke: halve lr (0.01->0.005) at hidden_dim=32 lowers masked_mae; widening to 64 hurts

- 来源: run:metr_la_speed_smoke#r1
- 数据特征画像: {"cv_global": 0.21588565575666047, "cv_node": 0.17629988621669912, "spatial_heterogeneity": 0.10945428458618764, "rho1": 0.7907617056752304, "sampling_interval_min": 5.0, "node_count": 207.0}
- 调参方向:
  - learning_rate 0.01 -> 0.005 with hidden_dim fixed at 32 lowered masked_mae 3.901829 -> 3.877815 (-0.62%) and sharply raised r2 0.1359 -> 0.3055 and evar 0.1990 -> 0.3723; on CPU 1-epoch METR_LA smoke keep hidden_dim small (32) and prefer the lower lr step
  - hidden_dim 32 -> 64 at lr=0.01 degraded masked_mae 3.901829 -> 3.952814 (+1.31%) and r2 0.1359 -> 0.1331 on the small 3425-window train split: do NOT widen the hidden dimension for this scenario
  - at max_epoch=1 the training budget (not capacity/depth) is the binding constraint; the next quality lever is more epochs at lr=0.005/hidden_dim=32, not a larger model
- 证据: [{"observed": "learning_rate 0.01->0.005 at hidden_dim=32", "masked_mae_before": 3.901829024155935, "masked_mae_after": 3.8778148094813027, "delta": -0.0240142146746323, "r2_before": 0.13592013891413104, "r2_after": 0.3054652796568308, "evar_before": 0.19900463024775186, "evar_after": 0.37231869002183277}, {"observed": "hidden_dim 32->64 at lr=0.01", "masked_mae_before": 3.901829024155935, "masked_mae_after": 3.9528136054674783, "delta": 0.0509845813115433, "r2_before": 0.13592013891413104, "r2_after": 0.13314296218270416}, {"verdict": "passed=true, value=3.8778148094813027 < threshold=35.0 (lower-is-better), margin 31.122, best_trial_id=t1_lr0.005_h32", "all_trials_pass": true}, {"evaluation_lesson": "at max_epoch=1 the training budget dominates: moving lr 0.01->0.005 changed masked_mae only -0.62% but moved r2 0.1359->0.3055 and evar 0.1990->0.3723 a lot; judge pass/fail strictly against the threshold and treat trial ranking as directional only"}]
