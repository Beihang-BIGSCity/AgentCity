# AgentCity 迁移方案：Claude Code SDK → openJiuwen 平台

> 本文档是完整的迁移实施计划，供新会话直接执行。
> 分支：`openjiuwen`（基于 `shk` 创建）。基线架构见"现状"一节。

## 0. 背景结论（调研已确认的事实）

- openJiuwen 是华为开源的 Agent 平台（Apache 2.0），核心 SDK 为 `agent-core`（PyPI 包名 `openjiuwen`，Python 3.11–3.13）。
- 执行层抽象为 `openjiuwen.harness_protocol.HarnessProtocol`（typing.Protocol 鸭子类型 SPI），内置 provider：`native`（自研 DeepAgent）、`claudecode`、`codex`、`dsh`（DeepSeek Harness），位于 `openjiuwen/harness_providers/`。**不支持 pi**。
- 外部 SDK 为可选依赖：`openjiuwen[claude]`（claude-agent-sdk）、`openjiuwen[codex]`（openai-codex）、`openjiuwen[dsh]`（deepseek-harness-sdk）。
- 多代理协同由 `openjiuwen.agent_teams` 提供，自带团队 MCP server（入口 `openjiuwen-team-mcp` = `openjiuwen.agent_teams.mcp.server:main`）。leader 通过 MCP 工具**自主**委派 member；框架负责 spawn/生命周期/消息路由/观测，不写死流程。member 可以是外部 harness 实例（`agent_teams.external.member_runtime`）。
- 事件模型：HarnessEvent（序号严格递增、每 turn 恰好一个终止事件）、TurnResult、HarnessCheckpoint；HarnessCard 声明能力，不支持的能力抛 `UnsupportedHarnessCapabilityError`。
- 代码参考：https://github.com/openJiuwen-ai/agent-core （develop 分支比 main 新）。

## 1. 现状（shk 分支基线）

| 层 | 文件 | 职责 |
|---|---|---|
| 执行层 | `agents/core/orchestrator.py` | 全局共享一个 `ClaudeSDKClient`；`ClaudeAgentOptions`：allowed_tools 白名单、`permission_mode="bypassPermissions"`、`setting_sources=["project"]`（加载 `.claude/skills/pdf` 与 `.claude/ralph-wiggum`） |
| 流水线 | `agents/core/pipeline.py` | 逐 stage 执行；`_run_multi_agent_stage` 为 lead 单独建 client，`agents=SDKAgentDefinition` 注册子代理，lead 只持 `Task` 工具 |
| 业务工具 | `claude_client.py` | 5 个 `@tool` 进程内 MCP 工具：`catalog_article` / `search_paper` / `evaluate_paper_relevance` / `test_migration` / `tune_migration_model` |
| 工作流定义 | `agents/{literature,migration,tuning}/workflow.py`、`agents/core/agent_registry.py` | **三个 workflow 均为"单 MultiAgentStage + lead 自主分发"**：literature 3 个子代理、migration 6 个、tuning 3 个。`StageDefinition` 单 agent 路径目前未被使用 |
| 持久化/服务 | `agents/core/storage.py`、`server.py`、`agents/migration/storage.py` | stage log JSON、MigrationResultStore、FastAPI + 前端 |

关键含义：三条 workflow 吃同一套机制，只需改 `pipeline.py` 的 `_run_multi_agent_stage` 一处集成点，三线同时完成迁移。

## 2. 目标架构（修正后的方案）

核心原则：**子代理协同从 harness 私有机制上提到框架层；业务工具走 MCP；harness 相关配置抽成 profile**。

```text
agents/core/orchestrator.py + pipeline.py   （编排层，只面向 openJiuwen API 编程）
        │
        ├─ leader = harness 实例（初期 claudecode provider）
        │     └─ 挂 openjiuwen-team-mcp 的委派工具 → leader 自主分发
        │
        ├─ members = 12 个子代理（agent_teams member，各自独立 harness 实例）
        │     配置来自 agent_registry.py 的 AgentDefinition（框架中立，原样保留）
        │
        ├─ 业务工具 = FastMCP server（agents/tools/），同时挂给 leader 和 members
        │
        └─ harness_profiles/*.yaml  （per-harness 的工具命名/模型档位映射）
```

leader 的委派决策完全自主（与现状语义一致）；将来换 harness（codex/dsh/pi）只换 profile，编排与花名册不动。

## 3. 实施步骤

### Phase 1：依赖与配置

1. `pip install "openjiuwen[claude]"`（>=0.1.18，注意核对 agent_teams API 是否在所装版本中；必要时从 gitcode develop 分支安装）。
2. `requirements.txt` 去掉 `claude_agent_sdk`，加 `openjiuwen`。
3. 新建 `agents/core/harness_profiles/`：为 claudecode 建 profile，包含：
   - 逻辑角色 → 模型映射（如 `fast=haiku`、`balanced=sonnet`、`strong=opus`；为将来 dsh 的 flash/pro 预留）
   - 工具命名映射（逻辑工具名 → 该 harness 的实际工具名，如 `Read/Write/Bash/Grep/Glob/WebSearch`）
4. `agent_registry.py` 中 `model="sonnet"` 等字面量改为逻辑角色名，由 profile 解析。

### Phase 2：业务工具 MCP 化

5. 把 `claude_client.py` 的 5 个 `@tool` 函数迁移到 `agents/tools/server.py`，用 `fastmcp` 注册为 MCP server（openjiuwen 依赖已含 fastmcp）。
6. `test_migration` / `tune_migration_model` 内的 `subprocess.run` 同步阻塞调用改为 `asyncio.create_subprocess_exec` 或包 `asyncio.to_thread`，避免阻塞 openjiuwen 的异步执行引擎。
7. `search_paper` 中硬编码的 serper API key（`claude_client.py:220`）改为读环境变量 `SERPER_API_KEY`。

### Phase 3：执行引擎替换（核心）

8. 新建 `agents/core/openjiuwen_backend.py`：
   - 封装 openjiuwen 初始化、team 构建（从 `AgentDefinition` dict 生成 member 配置）、事件流 → stage messages 的翻译（替代 `_serialize_message` 的 TextBlock/ToolUseBlock 解析，改消费 HarnessEvent/TurnResult）。
9. 改 `agents/core/pipeline.py::_run_multi_agent_stage`：
   - leader = harness 实例（system prompt = `lead_agent_prompt_builder(context)`，初期 provider=claudecode），挂 openjiuwen-team-mcp 委派工具 + 业务 MCP server；
   - members = `agent_registry` 的 12 个子代理定义 → agent_teams member 配置；
   - 删除 `SDKAgentDefinition` 转换与 `ClaudeSDKClient` 专用逻辑。
10. 改 `agents/core/orchestrator.py`：去掉 `ClaudeAgentOptions`/`ClaudeSDKClient`；allowed_tools 白名单改由 harness profile + member 配置承载；`bypassPermissions` 改为 openjiuwen 的工具审批策略（interaction handler 自动放行或 HarnessCard 配置）。
11. `claude_client.py` 的 `main()` 保留为 CLI 入口，但只组装 orchestrator；`@tool` 定义全部移走后此文件应显著瘦身或改名为 `main.py`（注意 `server.py`、`batch_migration.py` 的 import 同步更新）。

### Phase 4：生态能力替换

12. `.claude/skills/pdf`：openJiuwen 无 Claude 原生 PDF 读取。方案 A：paper-analyzer 增加一个基于 `pdfplumber`（SDK 自带依赖）的文本提取工具；方案 B：转成 openJiuwen Skill。先做 A，简单可控。
13. `.claude/ralph-wiggum`：评估是否仍需要；openjiuwen 有 `harness/task_loop`、`harness/goal` 等迭代执行模块可替代。默认移除，若 literature workflow 行为退化再回补。
14. `WebSearch`/`WebFetch`：Claude 内置工具，其他 harness 没有。在业务 MCP server 中补 `web_search`（serper）和 `web_fetch` 两个工具，并在 paper-searcher / dataset-downloader 的工具白名单中用逻辑名引用（pi 分支 `pi_backend/extensions/agentcity-tools.ts` 有现成实现可参考）。

### Phase 5：观测与持久化

15. `StageStorage`（stage log JSON）保留，数据源从 SDK message 改为 openjiuwen 事件流；保证 `server.py` 的 `/stages` 接口和前端展示不回归。
16. `stage_callback`（`agents/migration/hooks.py` 的 `create_migration_stage_callback`）与 `MigrationResultStore.record_run` 保持不变——它们消费的是 summary 字符串，与执行层解耦。
17. 可选：接入 `openjiuwen[observability]`（OpenTelemetry）替代部分 logger；断点恢复用 `HarnessCheckpoint` 替代 `preserve_stage_log` 的粗粒度续跑（可作为后续增强，不在本期必须项）。

### Phase 6：验证

18. 冒烟顺序：literature（只读，最安全）→ tuning → migration（写文件最多）。
19. 对照测试：同一篇论文分别用新旧执行层跑 migration，对比 stage log 结构、迁移产物（`documentation/*_migration_summary.md`）与 `result/traffic_state_pred/*.csv` 指标。
20. 回归：`python claude_client.py`（或新入口）全流程、`server.py` 启动 + 前端、`batch_migration.py` 批跑。

## 4. 风险点

- **PDF 阅读**：脱离 Claude 后需 tools 层自行提取（见步骤 12）。
- **模型档位**：haiku/sonnet/opus 三档若换 DeepSeek 只有 flash/pro 两档，`opus` 角色（model-adapter、migration-tester、tuning-planner、result-analyzer）需重新校准。
- **SDK 成熟度**：openjiuwen 0.1.x Beta，agent_teams/harness_providers API 可能变动；安装后先核对 `openjiuwen.agent_teams` 与 `openjiuwen.harness_providers` 的实际接口，与本文档描述（基于 develop 分支调研）有出入时以实际代码为准并更新本文档。
- **委派工具形态**：openjiuwen-team-mcp 的具体工具名/参数以安装版本源码为准（`openjiuwen/agent_teams/mcp/`）。

## 5. 新会话开工提示词（供 /clear 后使用）

```text
阅读 documentation/openjiuwen_migration_plan.md，按其中的 Phase 1-6 执行
AgentCity 从 Claude Code SDK 到 openJiuwen 的完整迁移。当前在 openjiuwen 分支。
先核对已安装 openjiuwen 版本的 agent_teams / harness_providers / harness_protocol
实际接口与文档描述是否一致，不一致时以源码为准并更新文档，然后逐 Phase 实施。
```
