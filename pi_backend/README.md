# pi_backend — pi-mono 执行层

本目录承载框架的 pi([badlogic/pi-mono](https://github.com/badlogic/pi-mono),现 earendil-works/pi）运行时。Python 编排层(`agents/core/`)通过 `pi-py-sdk`(PyPI）驱动 `pi --mode rpc` 子进程。

## 组成

- `bin/pi` — 启动脚本。pi 0.79.9 要求 Node >= 22.19，而系统 Node 是 v20，因此用项目内 `node_modules/node`(Node 22）执行 pi 的 CLI 入口。
- `agent/models.json` — 自定义 provider(`agentcity`）配置，指向 `ANTHROPIC_BASE_URL` 代理，key 引用 `$ANTHROPIC_AUTH_TOKEN` 环境变量。pi 进程通过 `PI_CODING_AGENT_DIR` 指向本目录。
- `extensions/agentcity-tools.ts` — pi extension，注册业务工具(`catalog_article` / `search_paper` / `evaluate_paper_relevance` / `test_migration` / `tune_migration_model` / `web_search` / `web_fetch`，桥接到 `python -m agents.tools.runner`）和 `task` 子代理工具（spawn 子 pi 进程）。
- `package.json` — 锁定 `@earendil-works/pi-coding-agent@0.79.9` 与 `node@22`。

## 安装 / 重建

```bash
cd pi_backend
npm install --ignore-scripts
node node_modules/node/installArchSpecificPackage.js   # node 包的二进制在 postinstall 中下载
```

## 运行前提

- Python 侧 `pip install pi-py-sdk==0.2.1`（已在 requirements.txt)。
- 环境变量：`ANTHROPIC_AUTH_TOKEN`（代理密钥）；可选 `SERPER_API_KEY`(web_search/search_paper)、`AGENTCITY_PYTHON`(默认 `python`)、`AGENTCITY_TASK_TIMEOUT_MS`（子代理超时，默认 30 分钟）。

## 结构说明

编排层(`agents/core/orchestrator.py` → `pipeline.py`）每个 stage 启动一个 pi RPC 进程(`agents/core/pi_backend.py` 的 `PiStageSession`)。多 agent stage 中 lead 只挂 `task` 工具；子代理定义（工具 allowlist、模型 tier、system prompt）由 pipeline 序列化到 `data/runs/.subagents_<stage>.json`，由 extension 的 `task` 工具读取并 spawn 子 pi。
