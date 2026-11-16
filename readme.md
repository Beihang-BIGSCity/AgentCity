# AgentCity

AgentCity 是基于 openJiuwen 和 LibCity 的时空预测模型适配工具。输入任务后，自动完成数据分析、模型组合、训练、调参和效果评估，并通过多轮反馈改进模型。

默认运行 **6 轮**，每轮训练 **35 epoch**、调参 **5 次**；训练 / 验证 / 测试集比例为 **70% / 10% / 20%**。迭代使用验证集选择模型，最终单独重新训练并评估测试集。

## 环境准备

需要 Python 3.12。以下命令均在项目根目录执行：

```bash
python3.12 -m venv .venv
python3.12 -m venv .venv-libcity
.venv/bin/python -m pip install -r requirements.local-agent.txt
.venv-libcity/bin/python -m pip install -r requirements.local-libcity.txt
source scripts/activate_local.sh
```

依赖清单以 macOS Apple Silicon 环境为基础；CUDA 等其他环境需调整对应的 PyTorch 版本。

运行前准备：

- 将 LibCity 格式的数据集放入 `Bigscity-LibCity/raw_data/<数据集名称>/`。
- 配置模型服务 API key。可设置 `AGENTCITY_API_KEY`，也可复制 `agents/core/harness_profiles/deepagent.yaml` 为同目录下的 `deepagent.local.yaml`，填写服务地址、模型和 key。本地配置不会提交到 Git。

## Quick Start

```bash
source scripts/activate_local.sh

# 查看可用任务
python main.py --list-tasks

# 使用 GPU 运行任务
python main.py --task tasks/metr_la_speed_smoke.json --gpu on
```

需要 CPU 时，将 `--gpu on` 改为 `--gpu off`。任务名中的 `smoke` 是历史命名，该任务当前同样执行完整的默认迭代。

## 运行方式

通过 `--task` 选择任务文件；需要修改本次运行预算时，可显式指定：

```bash
python main.py --task tasks/metr_la_speed_smoke.json \
  --gpu on --max-rounds 6 --max-epoch 35
```

迭代结束后，根据验证结果选定模型和参数，再启动最终训练与测试。将下面的模型名和配置路径替换为实际产物：

```bash
python scripts/run_final_evaluation.py \
  --task tasks/metr_la_speed_smoke.json \
  --model SelectedModel \
  --config-file /absolute/path/to/frozen-winning-config.json \
  --gpu on
```

最终训练采用 LibCity 默认最多 **100 epoch**，保留默认早停。添加 `--dry-run` 可先预览命令。最终阶段重新读取任务文件，不继承之前运行时的 CLI 覆盖值。

## 结果查看

| 位置 | 内容 |
| --- | --- |
| 终端输出 | 是否达标、完成轮数、结果文件路径 |
| `documentation/<task_id>_adaptation_summary.md` | 任务汇总 |
| `data/adaptation/<task_id>/` | 模型组合、训练结果、调参结果及逐轮记录 |
| `Bigscity-LibCity/libcity/cache/` | 模型权重、训练证据与评估结果 |
| `logs/adaptation.log` | 运行日志 |

运行前可执行离线检查，不调用模型服务：

```bash
python scripts/validate_adaptation.py
```

更多配置说明见 [运行配置](documentation/runtime_configuration.md)。
