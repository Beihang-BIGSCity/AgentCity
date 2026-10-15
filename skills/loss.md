# 损失函数调参 Skills

面向时空交通预测的「数据特征画像 → 损失函数选择与掩码口径」经验库（损失维度）。
配套 `skills/index.json` 中的 `dimension=loss` 条目。所有方向都以 `libcity/model/loss.py` 的
`masked_mae_torch` / `masked_mse_torch` / `masked_mape_torch` 与 LibCity 默认 `null_value=0.0` 为基准，
并引用本仓库真实跑出的指标（`documentation/STFGNN_tune_results.csv`、`STFGNN_migration_summary.md`）。

## 掩码 MAE（Huber δ=1）作为默认鲁棒损失

**适用数据特征**：速度类连续量预测（METR_LA、PEMS_BAY 实测 `output_dim=1`），误差近似拉普拉斯分布；
数据含缺失/掉线（METR_LA 实测 `null_ratio=8.04%`）需要掩码，且存在轻量离群。对应模块：`linear_head`
（`DEFAULT_LOSS=masked_mae`）。

**调参方向**
- `loss_type=masked_mae` 起步（STFGNN 与 `linear_head` 的默认值），先跑通再比较其他损失。
- Huber `δ`/`rho` 1.0 -> 0.5：STFGNN 用 `huber_loss(delta=rho)`，是选择 `best_valid_score` 的度量；波动大的数据收到 0.5 更抗离群。
- `null_value` 保持 `0.0` 以复用 `libcity/model/loss.py` 的 `masked_*` 实现；改 `null_value` 会同时改变掩码语义。
- 指标口径：`test_MAPE` 在含真实 0 的数据上为 `inf`，必须同时汇报 `masked_MAPE`。
- CPU-only：损失选择不改变计算量，直接按画像选即可，无需为损失预算做额外取舍。

**依据/证据**
- `data/modlib/index.json`：`linear_head` 的 `DEFAULT_LOSS=masked_mae`（对齐 `libcity/model/loss.py` 的 `masked_mae_torch`）。
- `documentation/STFGNN_search_space.json`：`selection_metric` = 反归一化验证集 Huber loss（`STFGNN.calculate_loss = huber_loss(delta=rho=1)`，越小越好）。
- `documentation/STFGNN_tune_results.csv`：`best_valid_score` 落在 26.76–31.37（lr=2e-3/embed=32 最低 26.76；lr=5e-4/embed=64 为 27.10）。
- `documentation/STFGNN_migration_summary.md`：2-epoch 冒烟 `test_MAPE=inf`、`masked_MAPE=15.01%`。

**参考来源**：`data/modlib/index.json`（`linear_head`）；`documentation/STFGNN_search_space.json`、`STFGNN_tune_results.csv`、`STFGNN_migration_summary.md`。

## 含真实零值的流量数据用掩码 MAPE，禁用裸 MAPE

**适用数据特征**：流量/订单类非负计数数据，存在真实零时段（深夜），且业务方以相对误差为验收指标。
PEMSD7 实测 `cv_global=0.609`、`rho1=0.966`、`node_count=883`，非负流量的大波动会显著放大相对误差。
对应模块：`horizon_head`（`DEFAULT_LOSS=masked_mape`）。

**调参方向**
- 不要使用裸 `MAPE`：流量含真实 0 时相对误差发散；改用 `masked_mape` 或 `masked_mae`。
- `horizon_head` 默认 `masked_mape`，适合分步相对精度；`mlp_head` 默认 `masked_mse`，按业务指标二选一。
- 若必须报告 MAPE，标注 `inf` 并只统计非零时刻，避免误读（调参结果 CSV 中 `test_MAPE` 整列为 `inf`）。
- `learning_rate 1e-3 -> 5e-4`：掩码后有效样本数下降，等效学习率偏高。
- CPU-only：无额外开销，但注意 `masked_mape` 需要在反归一化域计算，避免尺度相关的伪高精度。

**依据/证据**
- `documentation/STFGNN_tune_results.csv`：24 组 trial 的 `test_MAPE` 全部为 `inf`，而 `masked_MAPE` 稳定在 0.125–0.172。
- `documentation/STFGNN_migration_summary.md`：MAPE 由 evaluator 报 `inf`（PEMSD7 流量含真实 0），`masked_MAPE=15.01%` 才是有意义指标。
- 本仓库实测 PEMSD7 `cv_global=0.609`（`agents/adaptation/utils.py::compute_dataset_profile`）。
- LibCity `STSGCN.json`：`use_mask=false`、`first_layer_embedding_size=64`、`distance_inverse=true`。

**参考来源**：`documentation/STFGNN_tune_results.csv`、`STFGNN_migration_summary.md`；`data/modlib/index.json`（`horizon_head`）。

## 掉线与极端事件下切 MSE 并裁剪梯度

**适用数据特征**：高缺失率（METR_LA 实测 `null_ratio=8.04%`，可作为掉线率量级校准点）或存在极端事件离群点，
且验收指标为 RMSE。`cv_node` 偏高（节点个体动态差异大）时同样适用。对应模块：`mlp_head`（`DEFAULT_LOSS=masked_mse`）。

**调参方向**
- `loss_type=masked_mse`：掉线恢复期与事件极值误差大，MSE 优先修正大偏差。
- `clip_grad_norm=true` + `max_grad_norm=5`（DCRNN / GWNET / MTGNN 默认）：缺失恢复期的离群梯度必须裁剪。
- `weight_decay 1e-4`（MTGNN 默认）：MSE 对离群更敏感，需要正则约束。
- 仍不稳时退回 Huber `δ=0.5`（比 MSE 抗离群、比 MAE 平滑且处处可导）。
- CPU-only：MSE 与 MAE 的计算开销相同，改损失不增加预算；但需要用 `max_epoch` 冒烟先确认不发散。

**依据/证据**
- `data/modlib/index.json`：`mlp_head` 的 `DEFAULT_LOSS=masked_mse`（与 `linear_head` 的 `masked_mae` 形成对照）。
- 本仓库实测 METR_LA `null_ratio=8.04%`，说明掉线是该数据集常态。
- LibCity `MTGNN.json`：`weight_decay=1e-4`、`clip_grad_norm=true(max_grad_norm=5)`。
- LibCity `STNorm.json`：`clip_grad_norm=true`、`max_grad_norm=10`（更宽松的裁剪阈值）。

**参考来源**：`data/modlib/index.json`（`mlp_head`）；`agents/adaptation/utils.py` 实测 METR_LA；LibCity `MTGNN.json`、`STNorm.json`。
