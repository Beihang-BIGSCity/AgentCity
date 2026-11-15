# 空间表征调参 Skills

面向时空交通预测的「数据特征画像 → 空间模块与超参方向」经验库（空间表征维度）。
配套 `skills/index.json` 中的 `dimension=spatial` 条目。核心判据来自 ST-LEGO §3.1 的空间异质性 `SH`
（`SH = std(μ_i)/mean(μ_i)`，`μ_i` 为节点 i 的时间均值）与邻接可得性；方向对齐 LibCity 的
`Ks`/`K`、`cheb_order`、`embed_dim`/`node_dim`、`max_diffusion_step`、`filter_type`、`adjtype`、
`bidir_adj_mx` 等旋钮。原子模块 id 取自 `data/modlib/index.json`。

## 拓扑可靠且同质时用固定邻接切比雪夫卷积

**适用数据特征**：`spatial_heterogeneity <= 0.08` 且邻接矩阵可靠（距离/连通权重稳定），跨节点分布近似同质。
本仓库实测 PEMS_BAY `SH=0.041`、PEMSD4 `SH=0.048`、METR_LA `SH=0.109`，前两者是该 skill 的典型命中者。
对应模块：`cheb_gcn`（接口 `forward(x:(B,T,N,d), adj:(N,N)|None)`，内部用 `T_k = 2L̃T_{k-1} - L̃T_{k-2}` 展开谱卷积）。

**调参方向**
- `Ks`/`K` 3 -> 2：ASTGCN 默认 `K=3`、STGCN 默认 `Ks=3`；SH 低时二阶邻域已覆盖主要依赖，可省一阶传播算力。
- `hidden_dim`/`filters` 64 -> 32：低异质性数据上冗余容量会过拟合（STGCN `blocks=[[1,32,64],[64,32,128]]` 采用 32 → 64 递增）。
- `learning_rate 1e-3`（STGCN / STFGNN 默认）或 `1e-4`（ASTGCN 默认）；`lr_decay=steplr` + `lr_decay_ratio=0.7` + `step_size=5`。
- `bidir_adj_mx` 必须与邻接构造一致：STGCN 默认 `true`、DCRNN 默认 `false`，设错会静默降低精度。
- CPU-only：谱卷积的 K 阶传播是 N² 级稀疏乘法，比注意力省一个数量级，适合大 N 场景。

**依据/证据**
- ST-LEGO §3.1：SH 越大越需要可学习邻接或注意力式空间编码器 —— 反推 SH 低时固定邻接谱卷积已足够。
- ST-LEGO §App.B：ChebGCN「基于切比雪夫多项式，在固定邻接下高效且可扩展」。
- 本仓库实测 PEMS_BAY SH=0.041、PEMSD4 SH=0.048（`agents/adaptation/utils.py::compute_dataset_profile`）。
- LibCity `ASTGCN.json`：`K=3`、`nb_chev_filter=64`、`learning_rate=1e-4`；`STGCN.json`：`Ks=3`、`graph_conv_type=chebconv`。

**参考来源**：ST-LEGO §3.1 / §App.B；LibCity `ASTGCN.json`、`STGCN.json`；`data/modlib/index.json`（`cheb_gcn`）。

## 高异质性或邻接缺失时用可学习邻接

**适用数据特征**：`spatial_heterogeneity >= 0.25`，或数据集 `has_adjacency=false`（无先验路网）。PEMSD7 实测
`SH=0.365` 是典型高异质场景。对应模块：`adaptive_gcn`（节点嵌入生成邻接 + Neumann 近似，`forward(x:(B,T,N,d), adj:(N,N)|None)`，
`adj=None` 时完全依赖学到的图）。

**调参方向**
- node embedding 维度 `10 -> 40`：AGCRN 默认 `embed_dim=10`、MTGNN 默认 `node_dim=40`；节点越多/越异构越需要更大嵌入。
- `learning_rate 1e-3 -> 3e-3`（AGCRN 默认）：可学习邻接从随机初始化出发需要更大步长才能收敛。
- 邻接冷启：Graph WaveNet 默认 `randomadj=true` + `aptonly=true` + `adjtype=doubletransition`，无先验图时先用随机支撑。
- Neumann 阶数 / `cheb_order=2` 保持：阶数过大在无约束邻接上会放大量纲导致训练不稳。
- CPU-only：节点嵌入 + 邻接是 `N·d + N²` 级开销，N 很大时优先退回 `cheb_gcn`。

**依据/证据**
- ST-LEGO §3.1：SH 大 → 需要可学习邻接或注意力式空间编码器，以自适应捕捉非均匀空间模式。
- ST-LEGO §App.B：AdaptiveGCN「学习邻接、无可预定义支撑、适合动态结构、灵活性更高」。
- ST-LEGO §4.6 图 4：PeMS03 上 GAT 与 GumbelGraph（均为可学习/注意力结构）主导模块选择。
- LibCity `GWNET.json`：`adjtype=doubletransition`、`randomadj=true`、`aptonly=true`、`dropout=0.3`；`AGCRN.json`：`embed_dim=10`、`cheb_order=2`。

**参考来源**：ST-LEGO §3.1 / §App.B / §4.6；LibCity `GWNET.json`、`AGCRN.json`；`data/modlib/index.json`（`adaptive_gcn`）。

## 有向路网的方向性传播用扩散卷积

**适用数据特征**：高速、单行路网等方向性显著的场景，上下游传播不对称。METR_LA 实测
`cv_global 0.216 / SH 0.109 / rho1 0.791`，ST-LEGO §4.6 明确该数据集上 ChebGCN 与 DiffusionGCN 被优先选中。
对应模块：`diffusion_gcn`（有向 K 阶扩散 + 前向/后向随机游走两套权重）。

**调参方向**
- `max_diffusion_step=2` 保持（DCRNN 默认）：阶数每 +1 增加前后向各一套权重，收益递减而参数翻倍。
- `filter_type=dual_random_walk`（DCRNN 默认）：前向/后向随机游走各一套，适配有向图。
- `learning_rate=1e-2`（DCRNN 默认）+ `clip_grad_norm=true` / `max_grad_norm=5`：扩散 + 循环组合梯度易爆。
- `bidir_adj_mx=false`；不要与 STGCN 的双向归一化邻接混用（两者语义不同，混用会静默降精度）。
- CPU-only：K 阶扩散的 N² 乘法在 N > 1000 时成为瓶颈，需降 `batch_size` 或阶数。

**依据/证据**
- ST-LEGO §App.B：DiffusionGCN「基于扩散过程、方向感知的传播」。
- ST-LEGO §4.6 图 4：METR-LA 上 ChebGCN 与 DiffusionGCN 被优先选中，说明有向高速数据偏爱扩散卷积。
- ST-LEGO §3.2.2 benchmark 条目：METR-LA + DiffusionGCN(layers=3) + GRU(layers=3) + add，`MAE 2.88 / RMSE 5.91 / MAPE 7.83% / 0.39M 参数`。
- LibCity `DCRNN.json`：`filter_type=dual_random_walk`、`max_diffusion_step=2`、`learning_rate=1e-2`。

**参考来源**：ST-LEGO §3.2.2 / §App.B / §4.6；LibCity `DCRNN.json`；`data/modlib/index.json`（`diffusion_gcn`）。

## PEMS_BAY low-SH pilot: cheb_ks 3 -> 2 over the fixed 325-node adjacency is cheap and stable at 1 epoch; the binding constraint is the epoch budget, not the graph order or lr=0.01

- 来源: run:compose_pems_bay_20261027#r1
- 数据特征画像: {"cv_global": 0.14923154268507874, "cv_node": 0.13369922253347344, "spatial_heterogeneity": 0.041481823815304464, "rho1": 0.9734767212972786, "sampling_interval_min": 5.0, "node_count": 325.0}
- 调参方向:
  - cheb_ks 3 -> 2 confirmed for spatial_heterogeneity=0.0415 with node_count=325: the second-order spectral support over the 4129-nnz symmetric adjacency (bidir_adj_mx=true) trained without OOM at batch_size=128 x T=12, so do not raise Ks before the 15-epoch benchmark shows the graph branch is under-expressive.
  - Keep learning_rate=0.01 and clip_grad_norm=5 for this rho1=0.9735 / cv_window_std=0.0101 profile: 245 steps produced finite train/val loss (2.2556 / 2.2266) with no NaN/Inf, so there is zero evidence that a smaller lr is needed for stability; spend the next round on max_epoch 1 -> 15 with seeds {0,1,2} instead of re-tuning lr under the same 1-epoch budget.
  - Do not read the pilot's masked_mae 2.0775 as a tuning signal about the spatial branch: at 1 epoch the metric is bounded by the optimiser budget, so the only way to price the graph branch is the spatial_module='none' ablation run at the same 15 epochs and the same three seeds.
  - hidden_dim stays 32 (task-fixed) even though the composed model then holds only 15212 parameters, well below the 0.1-0.5M band: widen capacity only after the benchmark shows underfitting, and never to compensate a spec band in a plumbing round.
  - Always pass --gpu_id 0 on the CLI: the model config stores gpu_id as the JSON string '0', which libcity ConfigParser._init_device cannot feed to torch.cuda.set_device.
- 证据: [{"effect": "cheb_gcn at Ks=2 over the 325x325 symmetric adjacency (4129 nnz) ran without OOM at batch_size=128 x T=12: 1 epoch in 5.5 s, whole pilot 54.16 s wall clock, test masked_mae 2.0774722496668496 (mean over 12 horizons) vs threshold 10.0 -> PASS", "branch_params": 2080, "total_params": 15212, "max_epoch": 1, "seed": 0}, {"effect": "numerical stability: train_loss 2.2556 and val_loss 2.2266 after 245 steps at lr=0.01, no NaN/Inf, clip_grad_norm=5 - so neither the spectral branch nor lr=0.01 needs damping in this profile", "blocking_constraint": "the 1-epoch budget, not the spatial support or the learning rate", "scope_limit": "plumbing evidence only: no convergence, no ablation delta, no seed variance was measured"}, {"artifacts": ["/home/wangwenrui/shk/AgentCity/data/adaptation/compose_pems_bay_20261027/verdict.json", "/home/wangwenrui/shk/AgentCity/data/adaptation/compose_pems_bay_20261027/run_report.json", "/home/wangwenrui/shk/AgentCity/data/adaptation/compose_pems_bay_20261027/tuning.json"], "untested_next": "spatial_module='none' ablation at 15 epochs x 3 seeds is what would price the graph branch"}]

## PEMS_BAY low-SH pilot: cheb_ks 3 -> 2 over the fixed 325-node adjacency is cheap and stable at 1 epoch; the binding constraint is the epoch budget, not the graph order or lr=0.01

- 来源: run:compose_pems_bay_20261027#r1
- 数据特征画像: {"cv_global": 0.14923154268507874, "cv_node": 0.13369922253347344, "spatial_heterogeneity": 0.041481823815304464, "rho1": 0.9734767212972786, "sampling_interval_min": 5.0, "node_count": 325.0}
- 调参方向:
  - cheb_ks 3 -> 2 confirmed for spatial_heterogeneity=0.0415 with node_count=325: the second-order spectral support over the 4129-nnz symmetric adjacency (bidir_adj_mx=true) trained without OOM at batch_size=128 x T=12, so do not raise Ks before the 15-epoch benchmark shows the graph branch is under-expressive.
  - Keep learning_rate=0.01 and clip_grad_norm=5 for this rho1=0.9735 / cv_window_std=0.0101 profile: 245 steps produced finite train/val loss (2.2556 / 2.2266) with no NaN/Inf, so there is no evidence a smaller lr is needed for stability; spend the next round on max_epoch 1 -> 15 with seeds {0,1,2} instead of re-tuning lr under the same 1-epoch budget.
  - Do not read the pilot's masked_mae 2.0775 as a tuning signal about the spatial branch: at 1 epoch the metric is bounded by the optimiser budget, so the only way to price the graph branch is the spatial_module='none' ablation at the same 15 epochs and the same three seeds.
  - hidden_dim stays 32 (task-fixed) even though the composed model then holds only 15212 parameters, below the 0.1-0.5M band: widen capacity only after the benchmark shows underfitting.
  - Always pass --gpu_id 0 on the CLI: the model config stores gpu_id as the JSON string '0', which libcity ConfigParser._init_device cannot feed to torch.cuda.set_device.
- 证据: [{"effect": "cheb_gcn at Ks=2 over the 325x325 symmetric adjacency (4129 nnz) ran without OOM at batch_size=128 x T=12 on the verified run exp_id compose_pems_bay_20261027_acbaycompose_r2: 1 epoch in 6.78 s, whole pilot 52.941 s wall clock, test masked_mae 2.0774722496668496 (mean over 12 horizons) vs threshold 10.0 -> PASS (plumbing)", "branch_params": 2080, "total_params": 15212, "max_epoch": 1, "seed": 0}, {"effect": "numerical stability on the verified run: train_loss 2.2556 and val_loss 2.2266 after 245 steps at lr=0.01, no NaN/Inf under clip_grad_norm=5 - neither the spectral branch nor lr=0.01 needs damping at this profile", "blocking_constraint": "the 1-epoch budget, not the spatial support order or the learning rate", "reproducibility": "bit-identical to the 13:38Z execution under the same seed/config/source, so the value is deterministic rather than replicated across seeds"}, {"artifacts": ["/home/wangwenrui/shk/AgentCity/data/adaptation/compose_pems_bay_20261027/verdict.json", "/home/wangwenrui/shk/AgentCity/data/adaptation/compose_pems_bay_20261027/run_report.json", "/home/wangwenrui/shk/AgentCity/data/adaptation/compose_pems_bay_20261027/tuning.json", "/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/cache/compose_pems_bay_20261027_acbaycompose_r2/evaluate_cache/2026_10_27_13_47_28_ACBayCompose_PEMS_BAY.csv"], "untested_next": "spatial_module='none' ablation at 15 epochs x 3 seeds is what would price the graph branch"}, {"operational_note": "LibCity run_hyper.py aborted after logger bootstrap on this attempt and contributed zero trials; the deterministic single-trial runner executed the one recorded trial - when a single fixed point is all the contract allows, calling the tuner first costs a wasted attempt but does not change the budget"}]
