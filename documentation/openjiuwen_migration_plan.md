# AgentCity 迁移方案：Claude Code SDK → openJiuwen 平台

> 本文档是完整的迁移实施计划，供新会话直接执行。
> 分支：`openjiuwen`（基于 `shk` 创建）。基线架构见"现状"一节。
> **已按 openjiuwen 0.1.18 实际安装源码核对并修正**（2026-09-20）。原文档基于 develop
> 分支调研的若干假设与 0.1.18 不符，修正点见 §0.1。

## 0. 背景结论（调研已确认的事实）

- openJiuwen 是华为开源的 Agent 平台（Apache 2.0），核心 SDK 为 `agent-core`（PyPI 包名 `openjiuwen`，Python 3.11–3.13）。本仓运行环境：conda env `jiuwen`（Python 3.11.4），已装 `openjiuwen==0.1.18` + `claude-agent-sdk==0.2.154` + `fastmcp 2.14.7` + `pdfplumber`。
- 代码参考：https://github.com/openJiuwen-ai/agent-core 。

## 0.1 与 0.1.18 实际源码核对后的修正（以源码为准）

| 原文档假设 | 0.1.18 实际情况 |
|---|---|
| `openjiuwen.harness_protocol.HarnessProtocol` 顶层模块 | **不存在**。外部 harness 协议在 `openjiuwen.agent_teams.external.protocol`：`HarnessEvent`（events.py:288）、`TurnResult`（results.py:149）、`HarnessCheckpoint`（checkpoints.py:29）、`UnsupportedHarnessCapabilityError`（errors.py:17）、`ExternalHarnessProtocol`（protocol.py:22） |
| `HarnessCard` | 实际类名 `ExternalHarnessCard`（external/protocol/models.py:132） |
| `openjiuwen.harness_providers/` 含 native/claudecode/codex/dsh provider | **不存在**。内置执行体只有一个：自研 `DeepAgent`（`openjiuwen.harness`，ReAct loop + `core.foundation.llm` 的 Model/ModelClient，内置 `AnthropicModelClient`/`OpenAI`/`DeepSeek` 等 client）。`claude-agent-sdk` 仅用于 `agent_teams/external/cli_agent/claude/` —— 把 Claude Code CLI 当**外部成员**拉起，不是 harness 后端 |
| leader 挂 `openjiuwen-team-mcp` 获得委派工具 | `openjiuwen-team-mcp`（`agent_teams/mcp/server.py:main`，stdio）是给**外部 CLI 成员**注入的入口（身份取自环境变量 `OPENJIUWEN_TEAM_JOIN`），不是给 DeepAgent leader 用的。DeepAgent leader 的协同工具由框架在进程内注册（`agent_teams/tools/tool_factory.py` 的 `create_team_tools`）：leader 独有 `build_team`/`spawn_teammate`/`shutdown_member`/`approve_*`/`checkpoint` 等，共享 `create_task`/`update_task`/`view_task`/`claim_task`/`verify_task`/`send_message`/`list_members` 等 |
| member 可以是外部 harness 实例（`agent_teams.external.member_runtime`） | 属实：`external/member_runtime.py:63 ExternalHarnessMemberRuntime`；另有 `EXTERNAL_CLI` 角色经 `TeamAgentSpec.external_cli_agents=[ExternalCliAgentSpec(cli_agent="claude", ...)]` 声明。本迁移**不使用**外部 harness，leader 与 member 全用原生 DeepAgent |

其他已核实要点：

- **编程入口**：`await Runner.start()`（`core/runner/runner.py`）→ 单 agent 用 `create_deep_agent(model, system_prompt=..., mcps=[McpServerConfig(...)], workspace=..., language=..., **config_kwargs)`（`harness/factory.py:455`）+ `await agent.invoke({"query": ...})`。团队**不要**手动 `spec.build()`/`team.stream()`（session 动态表不会建，报 `no such table: team_task_*`）；走 `Runner.run_agent_team_streaming(spec, {"query": ...})`（`core/runner/team_runner.py:934` classmethod），runtime manager 负责 activation（session 创建、member spawn、建表）与 finalize。结束 `await Runner.stop()`。
- **stream chunk schema**（实测）：对象含 `type`/`index`/`payload`/`source_member`/`role`；`type` ∈ `llm_output`（token 增量，payload.content）、`tool_call`（payload: name/arguments/tool_call_id）、`controller_output`（如 TASK_COMPLETION，data[0].data.output）、`answer`（最终 payload.output）、`llm_usage`/`context.usage`（跳过）、`team.*` 标记（跳过）。
- **团队存储**：`StorageSpec(type="memory")` 在 0.1.18 有 bug（读写引擎各自独立 :memory:，DDL 不可见）；用文件 sqlite：`StorageSpec(type="sqlite", params={"connection_string": <绝对路径>})`。
- **模型**：`init_model(provider="Anthropic", model_name="claude-sonnet-4-5", api_key=..., api_base=...)`；api_key 显式传入，**包不读环境变量**。本机环境变量 `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN`（代理）由我们的后端代码读取后显式注入。**没有 sonnet/opus/haiku 抽象**，model_name 即 API 模型名。
- **团队**：`TeamAgentSpec(agents={"leader": DeepAgentSpec(...), "teammate": DeepAgentSpec(...)}, team_name=..., leader=LeaderSpec(...), predefined_members=[TeamMemberSpec(...)], spawn_mode="inprocess", dispatch_mode="autonomous", model_pool=[ModelPoolEntry(...)], model_pool_strategy="by_model_name", storage=StorageSpec(type="sqlite", params={...})).build()` → `TeamAgent`；`await team.invoke({"query": ...})` / `team.stream(...)`（`agent_teams/agent/team_agent.py`）。
- **`agents` dict 的 key 是 TeamRole 值（"leader"/"teammate"），同角色成员共享一份 DeepAgentSpec**；成员级定制只有 `TeamMemberSpec` 的 `desc`（公开，进 roster）/`prompt`（私有 system prompt）/`model_name`（配合 model_pool 按名分配）。**成员级工具白名单不被 TeamMemberSpec 直接支持**——只能：a) teammate 角色挂工具并集 + 在成员 prompt 中声明分工（软约束，本期采用）；b) leader 用 `spawn_teammate(permissions={...})` 动态 spawn 时收紧（`spawn_member` 的 `permissions_override` 只支持收紧）。
- **MCP**：`McpServerConfig(server_name=..., client_type="stdio", server_path="", params={"command": ..., "args": [...], "cwd": ..., "env": {...}})`（`core/foundation/tool/mcp/base.py:106`；stdio 参数见 `client/stdio_client.py`）。首次 invoke 时注册。leader/member 的 DeepAgentSpec 均可挂。
- **内置工具名（snake_case，与 Claude 不同）**：`read_file`/`write_file`/`edit_file`/`glob`/`grep`/`list_files`/`bash`/`fetch_webpage`/`free_search`/`paid_search`。**无 `WebSearch`**；`web_search` 需由业务 MCP server 自补。
- **工具审批**：`DeepAgentConfig.permissions`（TypedDict：`{enabled, tools: {name: "allow"|"ask"|"deny"}, ...}`），经 `create_deep_agent(**config_kwargs)` 转发；DeepAgentSpec 另有 `approval_required_tools`。默认不弹审批即可跑（无需等价 bypassPermissions 的特殊配置）。
- **skill**：`SkillUseRail(skills_dir=[...])`（`harness/rails/skills/skill_use_rail.py`），skill = 含 SKILL.md 的目录。
- **task_loop/goal**：`harness/task_loop`（TaskLoopController，外层多轮任务循环，`enable_task_loop=True`；DeepAgentSpec 默认 True）与 `harness/goal`（GoalManager 目标持久化/评估）可替代 ralph-wiggum 式自循环。
- **环境修复**：0.1.18 需补装 `opentelemetry-sdk` 与 `opentelemetry-exporter-otlp`，否则 `import openjiuwen.harness` 失败（已装）。

## 1. 现状（shk 分支基线）

| 层 | 文件 | 职责 |
|---|---|---|
| 执行层 | `agents/core/orchestrator.py` | 全局共享一个 `ClaudeSDKClient`；`ClaudeAgentOptions`：allowed_tools 白名单、`permission_mode="bypassPermissions"`、`setting_sources=["project"]`（加载 `.claude/skills/pdf` 与 `.claude/ralph-wiggum`） |
| 流水线 | `agents/core/pipeline.py` | 逐 stage 执行；`_run_multi_agent_stage` 为 lead 单独建 client，`agents=SDKAgentDefinition` 注册子代理，lead 只持 `Task` 工具 |
| 业务工具 | `claude_client.py` | 5 个 `@tool` 进程内 MCP 工具：`catalog_article` / `search_paper` / `evaluate_paper_relevance` / `test_migration` / `tune_migration_model`。**注意：这 5 个工具从未经 `create_sdk_mcp_server` 真正接线，等于待重新接线** |
| 工作流定义 | `agents/{literature,migration,tuning}/workflow.py`、`agents/core/agent_registry.py` | **三个 workflow 均为"单 MultiAgentStage + lead 自主分发"**：literature 3 个子代理、migration 6 个、tuning 3 个。`StageDefinition` 单 agent 路径目前未被使用 |
| 持久化/服务 | `agents/core/storage.py`、`server.py`、`agents/migration/storage.py` | stage log JSON、MigrationResultStore、FastAPI + 前端 |

关键含义：三条 workflow 吃同一套机制，只需改 `pipeline.py` 的 `_run_multi_agent_stage` 一处集成点，三线同时完成迁移。

## 2. 目标架构（已按 0.1.18 修正）

核心原则：**子代理协同从 harness 私有机制上提到框架层（agent_teams）；业务工具走 MCP；harness 相关配置抽成 profile**。

```text
agents/core/orchestrator.py + pipeline.py   （编排层，只面向 openJiuwen API 编程）
        │
        └─ agents/core/openjiuwen_backend.py
              ├─ Runner.start/stop 生命周期管理（每 run_workflows 一次）
              ├─ TeamAgentSpec 构建：
              │    leader = DeepAgent（system prompt = lead_agent_prompt_builder(context)，
              │                       协同工具由框架进程内注册，自主委派）
              │    members = 12 个子代理 → predefined_members（TeamMemberSpec:
              │              member_name/display_name/desc/prompt/model_name）
              │    spawn_mode="inprocess"，dispatch_mode="autonomous"
              │    model_pool + by_model_name → 成员级模型（fast/balanced/strong 三档）
              ├─ 业务工具 = FastMCP stdio server（agents/tools/server.py），
              │    挂进 teammate 角色 DeepAgentSpec.mcps
              └─ harness_profiles/deepagent.yaml（逻辑角色→模型名、逻辑工具名→内置工具名）
```

leader 的委派决策完全自主（与现状语义一致）；将来换后端（DeepSeek 等）只换 profile，编排与花名册不动。

已知简化（0.1.18 限制）：同角色成员共享 DeepAgentSpec，成员级工具白名单改为「teammate 挂工具并集 + 成员 prompt 声明各自分工」的软约束。

## 3. 实施步骤

### Phase 1：依赖与配置

1. ~~`pip install "openjiuwen[claude]"`~~ 实际：`pip install openjiuwen==0.1.18 fastmcp pdfplumber opentelemetry-sdk opentelemetry-exporter-otlp aiosqlite fastapi uvicorn`（jiuwen 环境已完成）。
2. `requirements.txt` 原为 conda 全量 dump、本就不含 `claude_agent_sdk`；追加 `openjiuwen` 等新依赖条目。（已做）
3. `agents/core/harness_profiles/deepagent.yaml`（已建）：
   - provider：DeepSeek 的 Anthropic 兼容端点（`https://api.deepseek.com/anthropic`，provider 仍为 `Anthropic`）。提交的 profile 不含凭证，使用 `AGENTCITY_API_KEY`/`AGENTCITY_API_BASE` 环境变量覆盖；也可使用已被 Git 忽略的 `deepagent.local.yaml`。未显式指定 profile 时，优先读取本地 profile。
   - 逻辑角色 → 模型映射：`fast`/`balanced`/`strong` → 当前均 `deepseek-flash`（该端点实测 `deepseek-flash`/`deepseek-chat` 可用；原 Anthropic 代理余额不足且不认裸别名）
   - 工具命名映射：`Read→read_file`、`Write→write_file`、`Edit→edit_file`、`Bash→bash`、`Grep→grep`、`Glob→glob`、`ListDir→list_files`、`WebFetch→web_fetch`、`WebSearch→web_search`（均实测存在）
4. `agent_registry.py` 的 `haiku/sonnet/opus` 已改为 `fast`/`balanced`/`strong`（含 types.py、三个 workflow 的 lead_model）。（已做）

### Phase 2：业务工具 MCP 化

5. 新建 `agents/tools/server.py`（FastMCP，stdio），迁移 `claude_client.py` 的 5 个工具。函数体以 `pi-agent-migration` 分支的 `agents/tools/impl.py` 为底本（`git show pi-agent-migration:agents/tools/impl.py`，已是去 SDK 化的 `async def tool(args: dict) -> str` 形态）。
6. `test_migration` / `tune_migration_model` 内的 `subprocess.run` 同步阻塞调用改为 `asyncio.create_subprocess_exec`（pi 分支 impl.py 已是 async 形态，直接沿用）。
7. `search_paper` 中硬编码的 serper API key 改为只读环境变量 `SERPER_API_KEY`，不再提交硬编码缺省凭证。使用搜索工具前需设置该环境变量。

### Phase 3：执行引擎替换（核心，已实现）

8. `agents/core/openjiuwen_backend.py`（已建）：
   - `HarnessProfile`（profile 加载、逻辑角色/工具解析、env 覆盖）；
   - `_team_model_config` / `_model_pool`（ModelPoolEntry + `by_model_name` 按成员分配模型）；
   - `build_team_spec`：lead prompt → `LeaderSpec`，`AgentDefinition` dict → `predefined_members`（prompt 末尾追加「designated tools」软约束），teammate 角色挂业务 MCP；每 stage 一个文件 sqlite（`data/runs/agent_teams/<stage_key>.db`，跑前删除保证全新团队）；
   - `run_team_stage`：走 `Runner.run_agent_team_streaming`，`StreamTranslator` 把 `llm_output` token 增量按 `source_member` 聚合成 text 记录、`tool_call` → `tool_use`、`answer`/`controller_output` → 最终 text，schema 与旧 SDK 序列化一致；
   - `run_single_prompt`：单 agent stage 路径（`create_deep_agent` + `invoke`）。
9. `agents/core/pipeline.py::_run_multi_agent_stage` 已改为调用 backend；`_run_stage` 非 runner 路径改走 `run_single_prompt`；SDK 相关 import 与 `_serialize_message` 全部移除。（已做）
10. `agents/core/orchestrator.py`：`ClaudeAgentOptions`/`ClaudeSDKClient` 删除，改为 `HarnessProfile.load()` + `ensure_runner_started()/shutdown_runner()` 包住 `pipeline.run()`。权限无需等价物（openjiuwen 默认非交互放行）。（已做）
11. `claude_client.py` 已删除，新增 `main.py` 作为 CLI 入口；`server.py`/`batch_migration.py` 本就不 import 它，无需改动。（已做）

### Phase 4：生态能力替换（已实现）

12. `.claude/skills/pdf`：采用方案 A——业务 MCP server 的 `read_pdf_text` 工具（pdfplumber）；paper-analyzer 的工具列表与 prompt 已改用 `read_pdf_text`。（已做）
13. `.claude/ralph-wiggum`：代码中已无任何引用（`setting_sources` 随 SDK 一起移除）；`.claude/` 目录文件保留但失效（不影响新栈；如确认不再需要可手工删除）。DeepAgentSpec 的 `enable_task_loop`（默认 True）+ 团队 idle nudge 覆盖迭代语义。（已做）
14. `web_search`/`web_fetch` 已进业务 MCP server（`agents/tools/impl.py`）；prompt 中的 Claude 工具名（Task/WebSearch/Read/Write/Bash 等）已全部改为 openjiuwen 名（create_task/send_message/web_search/read_file/write_file/bash）。（已做）

### Phase 5：观测与持久化（已实现）

15. `StageStorage` 未动；消息来源换成 `StreamTranslator` 输出，schema 不变，`server.py /api/stages` 与前端零改动。（已做）
16. `stage_callback` 与 `MigrationResultStore.record_run` 未动。（已做）
17. 可选增强（未做）：`openjiuwen[observability]`、`HarnessCheckpoint` 断点恢复。

### Phase 6：验证

18. 冒烟顺序：literature（只读，最安全）→ migration → tuning。
    - literature 冒烟（已做，通过）：search→evaluate→analyze 依序被正确成员认领完成，`catalog.json` 落盘，stage log 1 stage/34 消息/4 角色，exit=0。
    - migration 冒烟（已做，通过）：STFGNN（AAAI 2021，catalog 中唯一有 repo 且未迁移的论文）全链路跑通——clone(`repos/STFGNN`)→ 适配（`libcity/model/traffic_speed_prediction/STFGNN.py`，原仓库为 MXNet 符号式代码，系 PyTorch 重写）→ 配置（`STFGNN.json` + `task_config.json` 注册）→ 测试（PEMSD7，2-epoch smoke PASS，MAE 29.93/RMSE 43.95/masked-MAPE 15.01%）→ 文档（`documentation/STFGNN_migration_summary.md` + 4 份成员过程文档）。stage log 96 条消息；迁移目录经 stage_callback 落盘。耗时约 55 分钟（7 成员团队）。
    - tuning 冒烟（已做，通过但有边界说明）：同一篇 STFGNN。planner 产出搜索空间文档（`STFGNN_tuning_plan.md`/`STFGNN_search_space.json`），executor 双卡并行跑真实 trial（5 epoch/组，约 355s/epoch），2 组完成并落盘 `STFGNN_tune_results.csv`（最佳 lr=0.0005/emb=64：test MAE 27.58、RMSE 41.48、masked-MAPE 12.51%，优于 migration 的 2-epoch 基线 29.93/43.95）。stage 最终由 7200s 超时兜底收尾（团队还在等"最佳配置收敛复训"任务，属预期的长任务场景；超时强制关流后流水线正常推进、进程干净退出）。注意：executor 同样绕过 `tune_migration_model` MCP 工具直接用 bash 起训练（工具只支持串行单次调用，agent 为双卡并行选择绕行），调参结果因此未经 `MigrationResultStore.record_run` 落盘。
    - 后续调参产物更新：当前 `STFGNN_tune_results.csv` 共 24 组记录，12 组成功，12 组因 `BadZipFile` 失败。按验证损失选择 trial 7（lr=0.002、batch_size=64、embedding=32），5 epoch 的 test MAE=27.3940、RMSE=40.9667、masked-MAPE=12.6448%。上述 2 组结果是早期团队冒烟记录，不代表当前全部 trial；完整收敛复训尚未验证。
19. 对照测试：同一篇论文分别用新旧执行层跑 migration，对比 stage log 结构、迁移产物（`documentation/*_migration_summary.md`）与 `result/traffic_state_pred/*.csv` 指标。（未做：原 Anthropic 代理无余额，旧栈暂时跑不了）
20. 回归：`python main.py`（原 claude_client.py）全流程、`server.py` 启动 + 前端、`batch_migration.py` 批跑。

### Phase 6 冒烟暴露的问题与修复（均已落地）

- **LibCity 子进程解释器**：orchestrator 跑在 jiuwen env（无 torch），而 `test_migration`/`tune_migration_model` 原来硬编码 `python run_model.py`，会解析到错误的解释器。migration 冒烟中 tester 被迫绕过 MCP 工具直接用 autogen env 跑测试。修复：新增 `AGENTCITY_LIBCITY_PYTHON` 环境变量（`agents/migration/utils.py::libcity_python()`，默认 `python` 保持向后兼容），`agents/tools/impl.py` 与 `agents/tuning/tuning.py` 的三处子进程调用全部改走它。部署时将其指向带 torch 的 env（本机 `/home/wangwenrui/opt/miniconda3/envs/autogen/bin/python`）。
- **进程退出悬挂**：openjiuwen 0.1.18 teardown 会泄漏非守护线程（aiosqlite 连接线程 + 成员 TaskScheduler 收尾不干净），`asyncio.run()` 结束后解释器永不退出。修复：`main.py` 入口改为 `try/except + flush + os._exit(code)`；`shutdown_runner()` 对 `Runner.stop()` 的 teardown 异常（如 sqlite `no active connection` 的 rollback 报错）容错吞掉。server.py 为常驻进程不受影响。

## 4. 风险点

- **PDF 阅读**：脱离 Claude 后需 tools 层自行提取（见步骤 12）。
- **模型档位**：现统一用 DeepSeek `deepseek-flash`（`https://api.deepseek.com/anthropic`），fast/balanced/strong 三档同名。若将来分档，改 `harness_profiles/deepagent.yaml` 即可；原 Anthropic 代理实测余额不足且不接受裸别名（`claude-sonnet-4-5` → model_not_found，带日期的名字可用但无余额）。
- **SDK 成熟度**：openjiuwen 0.1.x Beta，agent_teams API 可能变动；本文档已按 0.1.18 实际源码修正，后续升级版本时需重新核对。
- **成员级工具白名单**：0.1.18 的 predefined member 不支持独立工具集（共享 teammate DeepAgentSpec），本期采用并集+prompt 软约束；若实际运行中成员越权调用严重，再评估改动态 spawn + permissions_override 收紧。
- **autonomous 委派语义**：任务上板 + 成员认领 + leader 广播，与 Claude `Task` 工具的"点对点同步调用"语义不完全相同（异步、经 mailbox）；冒烟已验证 3 阶段委派按预期执行（phase1/2/3 分别被正确成员认领完成）。
- **团队流终止**：当前代码采用 `persistent` 生命周期，等待框架的 `team.completed` marker 自动收尾，`stage_timeout_s`（默认 7200s）兜底；不依赖 leader 主动调用 `clean_team`。leader 的 `answer` chunk 是每轮都有，不能当终态。persistent 仍有阶段间空窗被误判完成的风险；超时会保留部分输出，不等同于业务任务全部成功。

## 5. 新会话开工提示词（供 /clear 后使用）

```text
阅读 documentation/openjiuwen_migration_plan.md，按其中的 Phase 1-6 执行
AgentCity 从 Claude Code SDK 到 openJiuwen 的完整迁移。当前在 openjiuwen 分支。
先核对已安装 openjiuwen 版本的 agent_teams / harness / external.protocol
实际接口与文档描述是否一致，不一致时以源码为准并更新文档，然后逐 Phase 实施。
```
