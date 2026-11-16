# AgentCity

AgentCity 基于仓库内的 LibCity 和 openJiuwen，实现时空预测模型的多智能体适配流程：读取任务与数据画像，检索知识和调参经验，组合原子模块生成模型，训练、调参，再将验证结果反馈到下一轮。

当前预设采用 **6 轮迭代、每轮 baseline 与调参训练各 35 epoch、每轮 5 次调参**。迭代只使用验证集进行模型选择；最终冻结模型与参数后，通过独立入口重新训练并评估测试集。

## 执行流程

```text
任务 JSON → main.py
    │
    ├─ task_planning：任务规划 → plan.json
    ├─ execution：数据集成 → 模型组合 → baseline 训练
    └─ tuning：超参数搜索 → 性能调试 → 效果评估与经验写入
            │
            ├─ 尚未完成 6 轮 → 将本轮结果反馈给下一轮规划
            └─ 完成迭代 → 保存汇总
                            │
                            └─ 根据验证结果选定并冻结模型与参数
                                → run_final_evaluation.py
                                → 最终重新训练与测试集评估
```

`fixed_rounds=true` 时，提前达到指标阈值也会继续执行剩余轮次；设为 `false` 才会在达标后提前结束。运行异常仍可能中断流程。最终训练不是 `main.py` 自动执行的阶段，需要单独启动。

| 层级 | 角色 | 职责 |
| --- | --- | --- |
| 决策层 | `task-planner` | 制定本轮计划；默认团队模式下领导执行层和调参层 |
| 执行层 | `data-integrator` | 分析训练数据画像，明确数据契约 |
| 执行层 | `model-composer` | 选择模块、生成模型代码与配置、完成 LibCity 注册 |
| 执行层 | `run-tester` | 运行 baseline，记录训练与验证指标 |
| 调参层 | `hyper-searcher` | 根据检索到的经验设计搜索空间并调用调参工具 |
| 调参层 | `perf-debugger` | 诊断运行与模型问题 |
| 调参层 | `effect-evaluator` | 比较指标与阈值，写入 verdict、知识和技能经验 |

默认使用 openJiuwen 团队模式；`AGENTCITY_SERIAL_ROLES=1` 可切换为角色顺序执行。模型组合遵循 `encoder → 并行 spatial/temporal → fusion → head`，五类各选一个模块，没有默认的 0.1M 参数量上限。

## 训练与评估配置

| 配置 | 迭代阶段 | 最终训练 / 测试阶段 |
| --- | --- | --- |
| epoch | 35，关闭早停 | LibCity 默认最多 100，早停开启，patience=50 |
| train / validation / test | 70% / 10% / 20% | 相同 |
| 划分方式 | LibCity 原生样本窗口划分 | 相同 |
| 调参 | 每轮 5 次 custom grid trial | 使用已冻结的参数，不再搜索 |
| loop | 固定 6 轮 | 单次最终训练 |
| 选择依据 | 验证集 MAE | 测试集仅用于最终报告 |

默认划分比例来自 `TrafficStatePointDataset.json`；最终 epoch 和早停配置来自 `TrafficStateExecutor.json`。这不是将所有参数都恢复为 LibCity 默认：迭代预算按本项目设置，模型结构和超参数由组合与调参决定。

每轮训练当前组合的 baseline，不跨轮复用旧模型的 baseline。默认 custom grid 要求搜索空间至少包含 5 个组合，单次调参调用执行 5 个 trial；失败 trial 也占用次数，不能成为最佳结果。角色提示要求每轮调用一次调参工具。代码还保留显式选择 `tuning.backend="libcity"` 的兼容路径，但当前任务预设均使用 `custom_grid`，不走该路径的失败回退搜索。

迭代阶段强制验证模式，以训练过程中最佳验证 epoch 的 MAE 作为选择指标，不运行测试集评估。划分采用 LibCity 原生的“先生成样本窗口，再切分样本”，不是原始时间段完全隔离的严格划分。

详细配置说明见 [runtime_configuration.md](documentation/runtime_configuration.md)。

## 环境准备

从项目根目录执行。仓库提供 Python 3.12 的双环境依赖清单：编排环境运行 openJiuwen / FastMCP，训练环境运行 LibCity / PyTorch。训练依赖清单以 macOS Apple Silicon 本机环境为基础；其他平台，尤其 CUDA 环境，需要按设备调整 PyTorch 等依赖。

```bash
python3.12 -m venv .venv
python3.12 -m venv .venv-libcity
.venv/bin/python -m pip install -r requirements.local-agent.txt
.venv-libcity/bin/python -m pip install -r requirements.local-libcity.txt
source scripts/activate_local.sh
```

激活脚本只选择解释器，不设置 epoch 或强制 MPS。原 `requirements.txt` 是历史环境导出，当前本机安装入口是上述两份清单。也可以使用已有环境，通过 `AGENTCITY_LIBCITY_PYTHON` 或 `main.py --libcity-python` 指定训练解释器。补充说明见 [LOCAL_ENV.md](LOCAL_ENV.md)。

准备对应任务的数据集，将 LibCity 格式文件放入 `Bigscity-LibCity/raw_data/<dataset>/`，包含该数据集所需的 `config.json`、动态数据与图结构等文件。原始数据不随 Git 提交；数据下载不由启动命令自动完成。

### 模型服务凭据

提交的模板位于 `agents/core/harness_profiles/deepagent.yaml`，其中 API key 为空。可使用环境变量：

```bash
export AGENTCITY_API_KEY="<your-api-key>"
# 如需覆盖模板中的服务地址：
export AGENTCITY_API_BASE="<your-api-base>"
```

也可复制模板为同目录的 `deepagent.local.yaml`，在本地文件中配置 provider、模型映射和 key。该文件被 `.gitignore` 忽略，不应提交真实凭据。

配置文件选择顺序为 `AGENTCITY_HARNESS_PROFILE` 指定的文件 → 本地 `deepagent.local.yaml` → 提交的模板。选定文件中的 `api_key_env` / `api_base_env` 指向的非空环境变量优先于文件内的值。

## 运行迭代

```bash
source scripts/activate_local.sh

# 查看任务列表，不调用模型服务
python main.py --list-tasks

# 显式启用 GPU：LibCity 自动选择可用的 CUDA 或 MPS
python main.py --task tasks/metr_la_speed_smoke.json --gpu on

# CPU 模式
python main.py --task tasks/metr_la_speed_smoke.json --gpu off
```

`--gpu on` 表示启用 GPU 策略，不保证设备一定可用；未指定具体设备时 LibCity 仍可能回退 CPU。需要指定设备时设置 `AGENTCITY_DEVICE=mps` 或 `cuda:0`，不可用会报错。`--gpu off` 优先于该环境变量。`--gpu auto` 会结合任务设置与设备探测决定是否启用；Apple GPU 上建议显式使用 `--gpu on`。

六份任务预设都使用当前协议。文件名中的 `smoke` 或历史日期仅为兼容保留，不代表短时冒烟任务。新的独立实验应使用不同的 `task_id`，以区分历史和恢复状态。

### 任务配置示例

```json
{
  "task_id": "metr_la_speed_smoke",
  "dataset": "METR_LA",
  "task": "traffic_state_pred",
  "model_name": null,
  "efficiency": "lightweight",
  "target": { "metric": "masked_mae", "threshold": 35.0, "maximize": false },
  "train": {
    "max_epoch": 35,
    "batch_size": 32,
    "gpu": false,
    "profile_train_only": true,
    "extra_args": { "train_rate": 0.7, "eval_rate": 0.1 }
  },
  "max_rounds": 6,
  "fixed_rounds": true,
  "tuning": { "max_trials": 5, "backend": "custom_grid" },
  "candidate_scope": {
    "models": [],
    "categories": ["encoder", "spatial", "temporal", "fusion", "head"]
  }
}
```

任务 JSON 决定预算；显式 CLI 参数 `--max-epoch`、`--batch-size`、`--max-rounds` 可覆盖本次迭代。工具调用中的旧 epoch 或划分参数不能覆盖任务预算，调参搜索空间也不能搜索 epoch 和划分比例。`AGENTCITY_EPOCHS` 和 `AGENTCITY_FIXED_ROUNDS` 不再是配置入口。

`AGENTCITY_RESUME=1` 启用恢复：根据已完成轮次和进行中轮次标记继续运行；串行模式下还可跳过已存在的阶段产物。`AGENTCITY_FREEZE_EXPERIENCE=1` 用于冻结经验的对照实验，阻止经验写入，并检查知识与技能文件是否发生变化。

正常结束时，只要有一轮达到目标，CLI 返回 `0`；未达标或执行异常返回 `1`。任务文件缺失或训练解释器预检失败返回 `2`。固定轮数不改变这一达标判定。

## 最终训练与测试

完成迭代后，根据验证结果选定模型，保留对应模型源码与注册，并准备冻结的参数 JSON。当前脚本不会自动从六轮历史中挑选最佳模型，也不会替你恢复被覆盖的模型源码。

```bash
python scripts/run_final_evaluation.py \
  --task tasks/metr_la_speed_smoke.json \
  --model SelectedModel \
  --config-file /absolute/path/to/frozen-winning-config.json \
  --gpu on --dry-run
```

将示例模型名和路径替换为实际产物。`--dry-run` 生成配置并打印命令，不启动训练；去掉它才会检查任务历史中的已完成轮数，再执行最终训练与测试。

最终脚本从任务 JSON 重新读取配置，不继承之前 `main.py` 的 CLI 覆盖值。epoch、早停、patience 使用 LibCity 默认；划分比例和 batch size 使用任务配置，其余模型参数来自冻结 JSON。如果选中的 trial 改过 batch size，应同步任务中的 batch size，避免最终脚本覆盖它。最终测试结果不应再用于组合或调参。

## 代码与产物

| 路径 | 内容 |
| --- | --- |
| `main.py` | 任务入口、CLI 覆盖、解释器预检 |
| `agents/core/` | 配置、openJiuwen 后端、阶段执行与多轮编排 |
| `agents/adaptation/` | 角色提示、模块检索、调参、指标读取与任务上下文 |
| `agents/tools/` | FastMCP 业务工具服务 |
| `Bigscity-LibCity/` | 仓库内的 LibCity 源码与配置 |
| `data/modlib/` | 纳入 Git 的原子模块库 |
| `knowledge/`、`skills/` | 结构先验、调参策略与可更新的经验 |
| `tasks/` | 任务预设 |
| `data/adaptation/<task_id>/` | 本轮产物、`context.json`、轮次归档和经验变更快照 |
| `Bigscity-LibCity/libcity/cache/` | 模型缓存、训练证据与评估结果 |
| `documentation/` | 配置说明与适配汇总 |

主要阶段产物为 `dataset_profile.json`、`plan.json`、`data_integration.json`、`composition.json`、`run_report.json`、`tuning.json` 和 `verdict.json`。进入下一轮前，上一轮阶段产物归档到 `rounds/round_<n>/`；数据画像保留。MCP 子进程通过活动任务文件和环境快照关联任务，CLI 默认使用 `data/runs/active_task.<pid>.json`。

虚拟环境、原始数据、运行缓存、`local-env/`、日志、实验结果及本地汇报文件已被忽略。知识和技能文件仍受 Git 跟踪，真实运行中的经验更新可能产生待提交改动。

本仓库包含 LibCity 适配：CUDA/MPS/CPU 设备选择、模型与 executor 的按需导入、验证模式与训练证据记录、`num_workers` 参数透传，以及显式开启时的严格时间划分支持。当前迭代和最终入口均选择原生划分。

## 验证与开发

```bash
# 离线结构、任务配置和工具契约检查，不调用 LLM
python scripts/validate_adaptation.py

# 增加原子模块导入与张量形状检查，需要训练解释器
python scripts/validate_adaptation.py --import-modules

# 配置、提示及指标来源回归测试，不启动真实训练
python -m pytest -q \
  tests/test_runtime_policy.py \
  tests/test_adaptation_compute_prompts.py \
  tests/test_validation_experiment.py
```

`python scripts/validate_adaptation.py --tool-smoke` 会运行真实训练工具冒烟检查，应与上述离线检查区分。离线检查通过不代表完整六轮实验或最终训练已执行成功。
