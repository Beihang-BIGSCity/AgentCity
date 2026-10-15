# AgentCity

AgentCity is an LLM-Agent-driven automation platform for urban computing, built on top of
[LibCity](https://github.com/LibCity/Bigscity-LibCity). It implements a single **spatio-temporal model adaptation
pipeline (模型适配)**: given a business-scenario task specification, a layered multi-agent system profiles the data,
composes a runnable LibCity model out of atomic modules, runs it, tunes it, evaluates the effect, and iterates until
the target metric is met.

The design follows 技术点三 (多智能体框架) and takes ST-LEGO (*Automated Spatio-Temporal Model Generation via LEGO-style
Composition*, `3774904.3792132.pdf`) as its technical reference: the statistics↔structure mapping, the atomic-module
library, the closed self-correction loop and the tuning framework all trace back to it.

## Framework Mapping (技术点三)

| 技术点三 | AgentCity implementation |
| --- | --- |
| 决策层 / 任务规划 | `task-planner` — plans the round, defines success criteria, and leads **both** team stages, validating every intermediate artifact |
| 执行层 / 数据集成 | `data-integrator` — `analyze_dataset` computes CVglobal / CVnode / SH / rho1 / Δt and writes the data contract |
| 执行层 / 模型组合 | `model-composer` — selects one atomic module per category, generates `libcity/model/.../<Model>.py` + config + registration |
| 执行层 / 运行测试 | `run-tester` — `run_adaptation` executes LibCity `run_model.py` on the scenario dataset and records the metrics |
| 调参层 / 超参数搜索 | `hyper-searcher` — designs a small search space from the retrieved skills and calls `tune_adaptation` |
| 调参层 / 性能调试 | `perf-debugger` — diagnoses OOM/NaN/shape/data-contract failures, repairs and re-runs |
| 调参层 / 效果评估 | `effect-evaluator` — strict threshold verdict + `record_experience` back into the knowledge base and skills |
| 专家知识库 | `knowledge/scenario_strategies.json` — 20 hand-authored "scenario feature → structural strategy" entries in three families (环境态势 / 事件因果链路 / 业务场景约束); a round's lesson is appended as an `evolved` entry instead of replacing them |
| Agent Skills | `skills/index.json` + `skills/{temporal,spatial,loss,head}.md` — "data feature profile → tuning direction" experience |
| 五类原子化模块 | `data/modlib/` — 19 curated modules: encoder / spatial / temporal / fusion / head, each with a unified `forward` interface |
| 闭环自纠错 | `agents/core/orchestrator.py` — metric threshold + max-round budget, previous round's verdict is fed back into planning |

## Pipeline

```
main.py --task tasks/xxx.json
        │
        ▼
 round 1..N  (closed loop: verdict → feedback → replan)
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ task_planning   [single agent, role=strong]      → plan.json             │
 │ execution       [team: data-integrator,          → dataset_profile.json  │
 │                  model-composer, run-tester]       data_integration.json │
 │                                                    composition.json      │
 │                                                    run_report.json       │
 │ tuning          [team: hyper-searcher,           → tuning.json           │
 │                  perf-debugger, effect-evaluator]  verdict.json          │
 └───────────────────────────────────────────────────────────────────────────┘
        │  verdict.passed?  ── no ──▶ next round with feedback_note()
        └── yes ──▶ documentation/<task_id>_adaptation_summary.md
```

The decision layer (`task-planner`) is the lead of both team stages, so it validates the artifacts of the execution and
tuning layers instead of doing the work itself. All state is shared through `data/adaptation/<task_id>/context.json`.

## Quick Start

```bash
# 1. orchestration env needs openjiuwen (no torch required)
export AGENTCITY_API_KEY=...                                   # optional: provider credentials
export AGENTCITY_LIBCITY_PYTHON=/path/to/env/bin/python        # interpreter that can import LibCity + torch
#   (or pass it per run: python main.py --task ... --libcity-python /path/to/env/bin/python)

# 2. list the available task specifications
python main.py --list-tasks

# 3. adapt one scenario (CPU smoke task: 1 epoch, 10% train / 10% eval)
python main.py --task tasks/metr_la_speed_smoke.json

# 4. override the loop / budget from the CLI
python main.py --task tasks/pemsd4_speed_multichannel.json --max-rounds 2 --max-epoch 3

# 5. offline plumbing validation (no LLM calls)
python scripts/validate_adaptation.py --import-modules
```

Exit code is `0` when the adaptation met the target threshold, `1` when it did not (or instead failed), `2` for a
missing task file or a LibCity interpreter that cannot import LibCity (`--skip-preflight` disables that check).

## Environment Notes

- **Two interpreters.** The CLI/orchestrator runs in the openJiuwen environment (no torch); every LibCity training,
  tuning and profiling step runs through the interpreter in `AGENTCITY_LIBCITY_PYTHON` (or `--libcity-python`).
  `main.py` preflights that interpreter with `import libcity, torch` and exits `2` with a hint instead of failing
  halfway through a round.
- **Provider credentials.** `agents/core/harness_profiles/deepagent.yaml` is the committed profile (no key); put a real
  key in the git-ignored `agents/core/harness_profiles/deepagent.local.yaml` (preferred automatically) or export
  `AGENTCITY_API_KEY` / `AGENTCITY_API_BASE`.
- **Proxies.** httpx prefers `ALL_PROXY` over `HTTP(S)_PROXY`. If `ALL_PROXY` is a `socks*` URL and the environment has
  no `socksio` installed, every LLM call fails; `main.py` drops that variable for the run (keeping `HTTP(S)_PROXY`) and
  prints a note. Install `httpx[socks]` if you actually need the SOCKS proxy.
- **Wall clock.** A full round is one planning call, a team stage that composes the model, one LibCity run
  (~10–20 min CPU, see the task-spec section) plus up to `tuning.max_trials` trial runs, then the tuning team. Budget
  ~30–60 minutes per round on this host.
- **Absolute paths.** Delegated team members run in their own isolated workspace
  (`~/.openjiuwen/.agent_teams/...`), not in the repository, so every member prompt is rendered with the absolute
  LibCity root, task directory, module-library, knowledge-base and skills paths, and the lead is told to delegate with
  absolute paths. Keep it that way: a relative path makes a member go looking for the filesystem instead of working.
- **Teardown.** openJiuwen leaves asyncio tasks and executor work behind, and `asyncio.run` waits for them without a
  timeout, so a finished run could print its result and then idle for minutes. Once the run summary is printed the CLI
  arms an exit watchdog (`AGENTCITY_TEARDOWN_GRACE_S`, default 30 s; `AGENTCITY_NO_EXIT_WATCHDOG` disables it) that
  force-exits with the run's code and reports whatever was still running.
- **Run records.** The MCP tool server is a separate process, so the CLI publishes the active task to
  `data/runs/active_task.<pid>.json` (handed over through `AGENTCITY_ACTIVE_TASK_FILE`) before the harness starts. Round
  artifacts and run records are filed under that task id, never under the model name; if the pointer is lost, `run_adaptation`
  and `tune_adaptation` fail loudly instead of writing `data/adaptation/<model_name>/`. Two pipelines in parallel from the
  same checkout need their own `AGENTCITY_ACTIVE_TASK_FILE`, and the validation/smoke probes already use a private one.

## Task Specification

```json
{
  "task_id": "metr_la_speed_smoke",
  "dataset": "METR_LA",
  "task": "traffic_state_pred",
  "model_name": null,
  "efficiency": "lightweight",
  "target": { "metric": "masked_mae", "threshold": 35.0, "maximize": false },
  "train": { "max_epoch": 1, "batch_size": 32, "gpu": false, "extra_args": { "train_rate": 0.1, "eval_rate": 0.05 } },
  "max_rounds": 2,
  "candidate_scope": { "models": [], "categories": ["encoder", "spatial", "temporal", "fusion", "head"] },
  "notes": "..."
}
```

- `target` — the acceptance criterion of the closed loop (`maximize: false` means "lower is better").
- `train` — the CPU budget handed to every LibCity subprocess. `gpu` defaults to `false`; `--gpu auto` keeps it off unless
  `nvidia-smi` reports a device. `train.extra_args` is forwarded verbatim as extra LibCity CLI flags, so a smoke run can
  shrink the workload (`"extra_args": {"train_rate": 0.1, "eval_rate": 0.05}`) without touching the framework. LibCity
  splits samples as `train_rate` / `eval_rate` / `1 - train_rate - eval_rate`, and one epoch always walks the whole
  dataset once, so the split mostly decides *which phase* gets the samples, not the total cost. Measured on this
  CPU-only host (METR_LA, small recurrent model, batch 32, one epoch): a single model run takes ~12 minutes, of which
  training is ~3 and the validation/test passes make up the rest. Budget ~10–20 minutes per model per round, and check
  `data/adaptation/<task_id>/run_report.json` for what actually ran.
- `tuning` — optional budget for the search stage: `max_trials` / `max_evals` cap every trial and
  `search_space` pre-seeds the space (the hyperparameter agent may still narrow it).
- `max_rounds` — the self-correction budget (1–10).
- Sample specs: `tasks/metr_la_speed_smoke.json`, `tasks/pemsd4_speed_multichannel.json`, `tasks/pems_bay_flow_event.json`.

Each round writes `plan.json`, `data_integration.json`, `composition.json`, `run_report.json`, `tuning.json` and
`verdict.json` into `data/adaptation/<task_id>/`. When a round fails the target, those files are archived to
`data/adaptation/<task_id>/rounds/round_<n>/` before the next round starts, and whatever an earlier CLI run left behind
is stashed under `data/adaptation/<task_id>/rounds/previous_run/` before round 1. A round therefore never inherits a
stale verdict; the dataset profile is a data asset and is deliberately kept.

## Business Tools

Exposed as a FastMCP stdio server (`python -m agents.tools.server`) and mounted on every agent role:

| Tool | Purpose |
| --- | --- |
| `analyze_dataset` | CVglobal / CVnode / SH / rho1 / Δt / node count → `dataset_profile.json` |
| `retrieve_knowledge` | deterministic nearest-neighbour retrieval over the knowledge base + skills |
| `record_experience` | write a round's lesson back into the knowledge base / skills; the store keeps only the six retrieval features, requires all five module families (knowledge) or a tuning direction (skill), and never overwrites a hand-authored entry |
| `list_modules`, `get_module`, `register_module` | read and extend the atomic module library |
| `run_adaptation` | run LibCity `run_model.py` on CPU and extract the metrics → `run_report.json` |
| `tune_adaptation` | LibCity `run_hyper.py` search with a deterministic grid-search fallback → `tuning.json` |
| `web_search`, `web_fetch`, `read_pdf_text` | optional external references |

## Data Assets

- `data/modlib/index.json` + `data/modlib/code/<category>/<id>.py` — the atomic module library (a data asset, versioned
  in git). Every module follows one unified interface:

  | category | interface |
  | --- | --- |
  | encoder | `forward(x: (B,T,N,C)) -> (B,T,N,d)` |
  | spatial | `forward(x: (B,T,N,d), adj: (N,N)|None) -> (B,T,N,d)` |
  | temporal | `forward(x: (B,T,N,d)) -> (B,T',N,d)` |
  | fusion | `forward(h_sp, h_tm) -> (B,T,N,d)` |
  | head | `forward(x: (B,T',N,d)) -> (B,H,N,output_dim)` + `loss(...)` |

- `knowledge/scenario_strategies.json` — expert knowledge base (three prior families).
- `skills/` — Agent Skills index plus one Markdown document per tuning dimension.
- `data/adaptation/<task_id>/` — per-task shared context, round artifacts and run records (runtime output).
- `documentation/<task_id>_adaptation_summary.md` — the written-up result of each adaptation.

## Development

```bash
python scripts/validate_adaptation.py                 # layout / library / knowledge / specs / workflow / tools / imports
python scripts/validate_adaptation.py --import-modules # + torch shape probe for all 19 modules
python scripts/validate_adaptation.py --tool-smoke     # + real CPU run through the tool layer
```

## Local LibCity Patches

Two upstream defects block CPU-only execution and are patched in this checkout (see `git -C Bigscity-LibCity diff`):

1. `libcity/config/config_parser.py::_init_device` — referenced an undefined `cpu` name; now resolves the device properly
   and only touches CUDA when it is actually available.
2. `libcity/data/dataset/deep_map_matching_dataset.py` — the optional `torch_sparse` import only caught `ImportError`,
   while ABI/`libcudart` failures raise `OSError`; the guard now covers both so CPU-only installs can import LibCity.

## Requirements

- Orchestration env: `openjiuwen`, `fastmcp`, `pyyaml` (see `requirements.txt`); no torch needed.
- LibCity runtime env (pointed at by `AGENTCITY_LIBCITY_PYTHON`): `torch`, `pandas`, `numpy`, LibCity's own dependencies.
