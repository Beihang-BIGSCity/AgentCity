# 预测模块调参 Skills

面向时空交通预测的「数据特征画像 → 输出头选择与超参方向」经验库（预测模块维度）。
配套 `skills/index.json` 中的 `dimension=head` 条目。三个候选头均来自 `data/modlib/index.json`：
`linear_head`（DEFAULT_LOSS `masked_mae`）、`horizon_head`（`masked_mape`）、`mlp_head`（`masked_mse`），
统一接口 `forward(x:(B,T',N,d)) -> (B,H,N,output_dim)` 与 `loss(pred, target, null_val=0.0, loss_type=None)`。

## 固定视界用直接多步线性头

**适用数据特征**：预测步数固定（LibCity 默认 `output_window=12`）、追求最简实现、不需要自回归；
`samples` 规模中等、CPU 预算紧张。`output_dim` 由数据集决定：METR_LA / PEMS_BAY / PEMSD7 为 1，PEMSD4 为 3。

**调参方向**
- `linear_head`：一次 `Linear(T'·d -> H·output_dim)`，参数量 `T'·d·H·output_dim`，H 或 d 翻倍参数即翻倍。
- `input_len` 必须与 LibCity `input_window` / `output_window` 一致（默认 12/12），否则 reshape 维度不匹配报错。
- `output_dim 1 -> 3`：跟随 `raw_data/PEMSD4/config.json` 的 `output_dim=3`（流量/占有率/速度），其余数据集保持 1。
- `loss_type` 保持 `masked_mae`，并逐 horizon 汇报 MAE 定位末端步退化。
- CPU-only：单层线性头是最省的输出模块，是效率型任务的默认选择。

**依据/证据**
- `data/modlib/index.json`：`linear_head` 的 source 为 STFGNN `OutputLayer`，把 `(B,N,T,C)` 展平后接
  `nn.Linear(num_of_filters, predict_length*output_dim)`，属直接多步、非自回归。
- ST-LEGO §3.2.2 benchmark 先验条目固定 `horizon=12`，与 LibCity `output_window=12` 一致。
- LibCity `config/data/TrafficStateDataset.json`：`input_window=12`、`output_window=12`；`STFGNN.json` 同为 12/12。
- LibCity `raw_data/PEMSD4/config.json`：`output_dim=3`。

**参考来源**：`data/modlib/index.json`（`linear_head`）；ST-LEGO §3.2.2；LibCity `config/data/TrafficStateDataset.json`、`raw_data/PEMSD4/config.json`。

## 各预测步差异明显时用分步输出头

**适用数据特征**：各 horizon 模式差异显著（峰值步、平台步、衰退步混在一起），典型如拥堵回流与事件激增；
配合细粒度采样（`sampling_interval_min` 为 1–2）收益最大。对应模块：`horizon_head`。

**调参方向**
- `horizon_head`：每步独立 `Linear(T'·d -> output_dim)`，参数量 `H·T'·d·output_dim`；H=12 时约为线性头的 12 倍，必须核算预算。
- `loss_type=masked_mape`（`horizon_head` 默认），分步相对精度与分步参数更贴合。
- 配合细粒度 Δt（1–2 min）：回流/激增场景的分步差异最显著。
- `learning_rate 1e-3 -> 5e-4`：分步参数更多，H 步 loss 求和后梯度更大，先降步长。
- CPU-only：参数量随 H 线性增长，H 较大时退回线性头或「共享隐层 + 轻量分步投影」。

**依据/证据**
- `data/modlib/index.json`：`horizon_head` 用按预测步组织的 `ModuleList`，每步一个 `OutputLayer` 后沿 horizon 轴拼接
  （STSGCN / STFGNN 结构），`DEFAULT_LOSS=masked_mape`。
- `documentation/STFGNN_migration_summary.md`：按 H 分步输出 `MAE/RMSE/masked_MAPE/R²` 表，说明分步诊断是既有实践。
- ST-LEGO §3.1：Δt 从时间戳提取并指导时间粒度；细粒度场景的分步差异更显著。
- LibCity `STSGCN.json`：`module_type=individual`、四层 `filters`，每步独立输出层是该结构的现实配置。

**参考来源**：`data/modlib/index.json`（`horizon_head`）；`documentation/STFGNN_migration_summary.md`；ST-LEGO §3.1。

## 历史-未来强非线性时给输出头加一层 MLP

**适用数据特征**：`cv_global` 高、历史-未来映射明显非线性（事件驱动的激增-回落）；主干容量已足够但输出层成为
瓶颈（训练与验证 loss 同步偏高）。对应模块：`mlp_head`（`DEFAULT_LOSS=masked_mse`）。

**调参方向**
- `mlp_head`：展平后先过一层隐藏 `Linear+ReLU`（默认宽度 `2×hidden_dim`）再输出；参数量 `T'·d·mlp_hidden + mlp_hidden·H·output_dim`。
- `hidden_dim 64 -> 32` 同时保留 MLP 隐藏层：总参数降到线性头量级而保住非线性，是 CPU 友好折中。
- `dropout 0 -> 0.2/0.3`：额外隐藏层最易过拟合（GWNET / MTGNN 默认 `0.3`）。
- `loss_type` 默认 `masked_mse`；若验收指标是 MAE 则显式改为 `masked_mae`。
- CPU-only：额外隐藏层的开销相对主干可忽略，但会吃掉效率型任务的 0.1M 参数预算。

**依据/证据**
- `data/modlib/index.json`：`mlp_head` 用一层隐藏 `Linear+ReLU` 拟合历史到未来的复杂映射，`DEFAULT_LOSS=masked_mse`。
- ST-LEGO §3.2.2 benchmark 先验中 METR-LA 参数仅 `0.39M`，说明输出头相对主干是小头但可成为瓶颈。
- ST-LEGO §3.2.3：模块深度上限 4 以避免显存溢出，输出头加深同样受此约束。
- LibCity `GWNET.json` / `MTGNN.json`：`dropout=0.3` 是主流正则强度。

**参考来源**：`data/modlib/index.json`（`mlp_head`）；ST-LEGO §3.2.2 / §3.2.3；LibCity `GWNET.json`、`MTGNN.json`。
