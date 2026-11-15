#!/usr/bin/env python
"""Structural validation for the AgentCity model-adaptation pipeline.

The checks are deterministic and offline: they prove the *plumbing* (module
library, knowledge base, skills, task specs, workflow/role roster, tool
contract, artifact contract, import health) before any LLM call happens.

Usage::

    python scripts/validate_adaptation.py                 # static checks
    python scripts/validate_adaptation.py --import-modules # + torch shape checks
    python scripts/validate_adaptation.py --tool-smoke     # + real CPU tool run

``--import-modules`` and ``--tool-smoke`` need the LibCity interpreter
(``AGENTCITY_LIBCITY_PYTHON``), so re-exec them under that interpreter.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.adaptation import knowledge as knowledge_store  # noqa: E402
from agents.adaptation import modlib as module_library  # noqa: E402
from agents.adaptation.workflow import get_adaptation_workflow  # noqa: E402
from agents.core.config import AppPaths, load_task_spec  # noqa: E402
from agents.core.orchestrator import ROUND_ARTIFACTS  # noqa: E402
from agents.core.openjiuwen_backend import HarnessProfile  # noqa: E402

REMOVED_PATHS = (
    "agents/literature",
    "agents/migration",
    "agents/tuning",
    "agents/core/agent_registry.py",
    "agents/core/jobs.py",
    "server.py",
    "frontend",
    "batch_migration.py",
    "batch_runner.py",
)
REMOVED_MODULES = (
    "agents.literature",
    "agents.migration",
    "agents.tuning",
    "agents.core.agent_registry",
    "agents.core.jobs",
)

# artifact -> (producer, expected keys)
ARTIFACT_CONTRACT: Dict[str, tuple[str, Sequence[str]]] = {
    "dataset_profile.json": ("analyze_dataset", ("cv_global", "cv_node", "rho1")),
    "plan.json": ("task planner (decision layer)", ("target", "data_requirements", "success_criteria")),
    "data_integration.json": ("data-integrator", ("dataset", "input_window", "output_window")),
    "composition.json": ("model-composer", ("model_name", "modules", "hidden_dim")),
    "run_report.json": ("run-adaptation tool + run-tester", ("status", "metrics", "command")),
    "tuning.json": ("tune-adaptation tool + hyper-searcher", ("search_space", "best_params")),
    "verdict.json": ("effect-evaluator", ("passed", "metric", "value")),
}

EXPECTED_STAGES = ("task_planning", "execution", "tuning")
EXPECTED_EXECUTION_AGENTS = {"data-integrator", "model-composer", "run-tester"}
EXPECTED_TUNING_AGENTS = {"hyper-searcher", "perf-debugger", "effect-evaluator"}
BUSINESS_TOOLS = (
    "analyze_dataset",
    "retrieve_knowledge",
    "record_experience",
    "list_modules",
    "get_module",
    "register_module",
    "run_adaptation",
    "tune_adaptation",
)

FAILURES: List[str] = []
NOTES: List[str] = []

# Child program for the "lost pointer" check: set the task in this process, delete
# the pointer it wrote, then resolve it from a fresh process that inherited only the
# environment.
_ENV_FALLBACK_PROBE = """
import json, os, subprocess, sys
from agents.adaptation.runtime import set_active_task, task_file
set_active_task({'task_id': 'env_fallback_probe'})
task_file().unlink()
child = subprocess.run(
    [sys.executable, '-c',
     'import json; from agents.tools.impl import _active_task; print(json.dumps(_active_task()))'],
    capture_output=True, text=True, check=False,
)
print(child.stdout.strip())
print('WARN' if 'falling back to the environment' in child.stderr else 'NO-WARN')
"""

# Child program for the run-registration check: with no active task anywhere, the
# run/tune tools must refuse instead of filing results under the model name.
_UNRESOLVED_TASK_PROBE = """
import asyncio, json, os
os.environ.pop('AGENTCITY_ACTIVE_TASK_ID', None)
os.environ.pop('AGENTCITY_ACTIVE_TASK_SPEC', None)
from agents.tools import impl
print(json.dumps({
    'active_task': impl._active_task(),
    'from_spec': impl._resolve_task_id({}, {'task_id': 'real_task'}),
    'from_args': impl._resolve_task_id({'task_id': 'arg_task'}, {'task_id': 'spec_task'}),
    'unresolved': impl._resolve_task_id({'model_name': 'STDiffGRU'}, {}),
    'run': asyncio.run(impl.run_adaptation({'model_name': 'STDiffGRU'})),
    'tune': asyncio.run(impl.tune_adaptation({'model_name': 'STDiffGRU'})),
    'analyze': asyncio.run(impl.analyze_dataset({'dataset': 'METR_LA'})),
}))
"""


def check(condition: bool, message: str) -> bool:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        FAILURES.append(message)
    return condition


def section(title: str) -> None:
    print(f"\n== {title} ==")


# ---------------------------------------------------------------------------


def check_repo_layout(paths: AppPaths) -> None:
    section("repository layout")
    for relative in REMOVED_PATHS:
        check(not (ROOT / relative).exists(), f"legacy path removed: {relative}")
    check((ROOT / "main.py").exists(), "CLI entry point exists: main.py")
    check((paths.modlib_dir / "index.json").exists(), "module library index exists")
    check(paths.knowledge_dir.is_dir(), "knowledge/ exists")
    check(paths.skills_dir.is_dir(), "skills/ exists")
    check((ROOT / "readme.md").exists(), "readme.md exists")
    check((ROOT / "scripts" / "validate_adaptation.py").exists(), "validation script exists")


def check_module_library(paths: AppPaths, import_modules: bool) -> None:
    section("atomic module library")
    problems = module_library.validate_index(paths.modlib_dir)
    check(not problems, f"index.json validates ({len(problems)} problem(s))")
    for problem in problems:
        NOTES.append(f"modlib: {problem}")

    index = module_library.load_index(paths.modlib_dir)
    modules = index.get("modules") or []
    check(len(modules) >= 15, f"module count >= 15 (found {len(modules)})")
    counts: Dict[str, int] = {}
    for module in modules:
        counts[module["category"]] = counts.get(module["category"], 0) + 1
    check(
        set(counts) == set(module_library.CATEGORIES),
        f"all five categories populated: {dict(sorted(counts.items()))}",
    )
    for category in module_library.CATEGORIES:
        check(counts.get(category, 0) >= 3, f"{category}: >= 3 modules (found {counts.get(category, 0)})")

    # every module source must expose the unified interface and stay framework-free
    missing_forward = []
    for module in modules:
        source_path = paths.modlib_dir / module["file"]
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        classes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        target = classes.get(module["class"])
        if target is None:
            missing_forward.append(f"{module['id']}: class missing")
            continue
        if not any(isinstance(node, ast.FunctionDef) and node.name == "forward" for node in target.body):
            missing_forward.append(f"{module['id']}: no forward()")
        if "libcity" in source and "import libcity" in source:
            missing_forward.append(f"{module['id']}: imports libcity")
    check(not missing_forward, "every module declares its unified forward() interface")
    for item in missing_forward:
        NOTES.append(f"modlib: {item}")

    if not import_modules:
        return
    check(
        "torch" in _libcity_interpreter_probe(),
        "LibCity interpreter can import torch (--import-modules)",
    )
    failures = _module_shape_probe(paths)
    check(not failures, "all modules build and run a CPU forward pass")
    for item in failures:
        NOTES.append(f"modlib: {item}")


def _libcity_interpreter_probe() -> str:
    from agents.adaptation.utils import libcity_python

    proc = subprocess.run(
        [libcity_python(), "-c", "import torch; print('torch', torch.__version__)"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip()


def _module_shape_probe(paths: AppPaths) -> List[str]:
    """Re-enter under the LibCity interpreter and exercise every module.

    The probe needs torch, which the orchestration interpreter deliberately
    does not have, so it always runs in a child process.
    """

    from agents.adaptation.utils import libcity_python

    proc = subprocess.run(
        [libcity_python(), str(Path(__file__).resolve()), "--module-probe"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "AGENTCITY_MODULE_PROBE": "1"},
        check=False,
    )
    failures = [
        line[len("MODULE-FAIL: ") :]
        for line in proc.stdout.splitlines()
        if line.startswith("MODULE-FAIL: ")
    ]
    if proc.returncode != 0 and not failures:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:]
        failures.append(f"module probe crashed: {tail}")
    if not failures and "MODULE-PROBE OK" not in proc.stdout:
        failures.append("module probe produced no verdict")
    return failures


def _module_shape_probe_inner(paths: AppPaths) -> List[str]:
    """Runs under the LibCity interpreter: build and forward every module.

    Modules are instantiated from the ``ctr_args`` declared in the index (with
    a smaller node/hidden width so the probe stays cheap), which also verifies
    that the declared constructor contract is the real one.
    """

    import torch

    batch, steps, nodes, width = 2, 12, 8, 16
    encoded = torch.randn(batch, steps, nodes, 3)
    hidden = torch.randn(batch, steps, nodes, width)
    adjacency = torch.eye(nodes)

    failures: List[str] = []
    for module in module_library.list_modules(paths.modlib_dir, include_code=False):
        kwargs = dict(module.get("ctr_args") or {})
        if "num_nodes" in kwargs:
            kwargs["num_nodes"] = nodes
        if "hidden_dim" in kwargs:
            kwargs["hidden_dim"] = width
        category = module["category"]
        inputs = {
            "encoder": (encoded,),
            "spatial": (hidden, adjacency),
            "temporal": (hidden,),
            "fusion": (hidden, hidden.clone()),
            "head": (hidden,),
        }[category]
        namespace: Dict[str, Any] = {}
        code = (paths.modlib_dir / module["file"]).read_text(encoding="utf-8")
        exec(compile(code, module["file"], "exec"), namespace)  # noqa: S102
        try:
            instance = namespace[module["class"]](**kwargs)
            instance.eval()
            with torch.no_grad():
                output = instance(*inputs)
            if not torch.isfinite(torch.as_tensor(output)).all():
                failures.append(f"{module['id']}: forward produced non-finite values")
            if category == "spatial":
                with torch.no_grad():
                    instance(hidden, None)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{module['id']}: forward failed ({type(exc).__name__}: {exc})")
    return failures


def check_knowledge_and_skills(paths: AppPaths) -> None:
    section("expert knowledge base + agent skills")
    knowledge_payload = knowledge_store.load_knowledge(paths.knowledge_dir)
    entries = knowledge_payload.get("entries") or []
    check(len(entries) >= 20, f"knowledge entries >= 20 (found {len(entries)})")
    families = [str(entry.get("family") or "") for entry in entries]
    for family in knowledge_store.FAMILIES:
        check(families.count(family) >= 6, f"family {family}: >= 6 entries (found {families.count(family)})")
    check(
        all(set(("id", "family", "title", "feature_profile", "strategy", "tuning_direction", "source")) <= set(entry) for entry in entries),
        "every knowledge entry carries the required keys",
    )
    check(
        all(
            all(key in knowledge_store.FEATURE_KEYS for key in (entry.get("feature_profile") or {}))
            for entry in entries
        ),
        "knowledge feature profiles use the retrieval feature keys",
    )
    check(
        all(
            isinstance(entry.get("strategy"), dict)
            and set(entry["strategy"]) >= {"encoder", "spatial", "temporal", "fusion", "head"}
            for entry in entries
        ),
        "every knowledge entry names a strategy for all five module families",
    )
    known_ids = {module["id"] for module in module_library.list_modules(paths.modlib_dir)}
    for entry in entries:
        named = _module_ids_in(entry.get("strategy") or {}, sorted(known_ids))
        if not named:
            NOTES.append(f"knowledge {entry.get('id')}: strategy names no atomic module id")

    skills_payload = knowledge_store.load_skills(paths.skills_dir)
    skills = skills_payload.get("skills") or []
    check(len(skills) >= 12, f"skills >= 12 (found {len(skills)})")
    per_dimension: Dict[str, int] = {}
    for skill in skills:
        dimension = str(skill.get("dimension") or "")
        per_dimension[dimension] = per_dimension.get(dimension, 0) + 1
    for dimension in knowledge_store.SKILL_DIMENSIONS:
        check(per_dimension.get(dimension, 0) >= 3, f"skill dimension {dimension} >= 3 (found {per_dimension.get(dimension, 0)})")
        document = paths.skills_dir / f"{dimension}.md"
        check(document.exists() and document.stat().st_size > 400, f"skill document populated: {document.name}")
    check(
        all(
            skill.get("tuning_direction") and skill.get("feature_profile") and skill.get("source")
            for skill in skills
        ),
        "every skill carries tuning directions, a feature profile and a source",
    )

    ids = [entry.get("id") for entry in entries] + [skill.get("id") for skill in skills]
    check(len(ids) == len(set(ids)), "knowledge/skill ids are unique")

    query = {
        "cv_global": 0.31,
        "cv_node": 0.44,
        "spatial_heterogeneity": 0.58,
        "rho1": 0.62,
        "sampling_interval_min": 5.0,
        "node_count": 207,
    }
    first = knowledge_store.retrieve(
        knowledge_dir=paths.knowledge_dir, skills_dir=paths.skills_dir, features=query, top_k=3
    )
    second = knowledge_store.retrieve(
        knowledge_dir=paths.knowledge_dir, skills_dir=paths.skills_dir, features=query, top_k=3
    )
    check(first == second, "retrieval is deterministic for a fixed feature profile")
    check(len(first.get("results") or []) == 3, "retrieval returns top_k results")
    check(
        all(result["distance"] >= 0 for result in first["results"]),
        "retrieval returns finite distances",
    )
    kinds = {result["kind"] for result in first["results"]}
    check(kinds <= {"knowledge", "skill"}, f"retrieval returns knowledge/skill only: {sorted(kinds)}")


def check_feedback_write_path() -> None:
    """The back-edge must not corrupt the shared knowledge/skill data assets.

    Observed live: a round recorded its raw ``dataset_profile.json`` plus a prose
    architecture string under the id of the curated prior it had just retrieved,
    which both replaced the authored entry and left the store with an entry that
    neither retrieval nor the schema could reason about.
    """

    section("feedback write path")
    families = list(knowledge_store.MODULE_FAMILIES)
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        knowledge_dir = base / "knowledge"
        skills_dir = base / "skills"
        knowledge_dir.mkdir()
        skills_dir.mkdir()
        knowledge_store.save_knowledge(
            knowledge_dir,
            {
                "version": 1,
                "entries": [
                    {
                        "id": "env-probe-prior",
                        "family": "environment",
                        "title": "curated probe prior",
                        "feature_profile": {
                            "cv_global": 0.1,
                            "cv_node": 0.1,
                            "spatial_heterogeneity": 0.1,
                            "rho1": 0.9,
                            "sampling_interval_min": 5.0,
                            "node_count": 100,
                        },
                        "strategy": {family: f"probe_{family}" for family in families},
                        "tuning_direction": ["keep the CPU budget"],
                        "source": "probe",
                    }
                ],
            },
        )

        recorded = knowledge_store.record_experience(
            knowledge_dir=knowledge_dir,
            skills_dir=skills_dir,
            kind="knowledge",
            entry={
                "id": "env-probe-prior",
                "title": "round probe lesson",
                "family": "traffic_speed_pred",
                "feature_profile": {
                    "dataset": "METR_LA",
                    "cv_global": 0.22,
                    "node_count": 207,
                    "split": {"train_rate": 0.1},
                },
                "strategy": {family: {"id": f"mod_{family}"} for family in families},
                "tuning_direction": ["learning_rate 0.01 -> 0.005"],
            },
        )
        stored = {
            entry.get("id"): entry
            for entry in knowledge_store.load_knowledge(knowledge_dir).get("entries") or []
        }
        check(recorded["id"] != "env-probe-prior", "runtime feedback never reuses a curated entry id")
        check(
            stored.get("env-probe-prior", {}).get("title") == "curated probe prior",
            "the curated entry survives a round that records its lesson",
        )
        check(
            set(recorded["feature_profile"]) <= set(knowledge_store.FEATURE_KEYS),
            "a recorded feature profile keeps only the retrieval features",
        )
        check(
            set(recorded["strategy"]) == set(families),
            "a recorded strategy names all five module families",
        )
        check(
            recorded["family"] in knowledge_store.FAMILIES,
            "a recorded entry is filed under a curated prior family",
        )

        for entry, why in (
            (
                {"id": "probe-incomplete", "feature_profile": {"cv_global": 0.2}, "strategy": {"spatial": "x"}},
                "without a module id for every family",
            ),
            (
                {"id": "probe-no-profile", "strategy": {family: "x" for family in families}},
                "without a numeric feature profile",
            ),
        ):
            try:
                knowledge_store.record_experience(
                    knowledge_dir=knowledge_dir, skills_dir=skills_dir, kind="knowledge", entry=entry
                )
                check(False, f"a knowledge entry {why} is rejected instead of stored")
            except ValueError:
                check(True, f"a knowledge entry {why} is rejected instead of stored")

        for entry, why in (
            (
                {"id": "probe-skill-a", "dimension": "temporal", "feature_profile": {"cv_global": 0.2}},
                "without a tuning direction",
            ),
            (
                {
                    "id": "probe-skill-b",
                    "dimension": "temporal",
                    "tuning_direction": ["learning_rate 0.01 -> 0.005"],
                    "feature_profile": {"dataset": "METR_LA"},
                },
                "without a numeric feature profile",
            ),
        ):
            try:
                knowledge_store.record_experience(
                    knowledge_dir=knowledge_dir, skills_dir=skills_dir, kind="skill", entry=entry
                )
                check(False, f"a tuning skill {why} is rejected")
            except ValueError:
                check(True, f"a tuning skill {why} is rejected")

        skill = knowledge_store.record_experience(
            knowledge_dir=knowledge_dir,
            skills_dir=skills_dir,
            kind="skill",
            entry={
                "id": "probe-skill-ok",
                "dimension": "temporal",
                "title": "probe skill",
                "tuning_direction": ["learning_rate 0.01 -> 0.005"],
                "feature_profile": {"cv_global": 0.22, "node_count": 207, "split": {"train_rate": 0.1}},
                "source": "run:probe-task#r1",
            },
        )
        check(
            set(skill["feature_profile"]) <= set(knowledge_store.FEATURE_KEYS),
            "a recorded skill keeps only the retrieval features",
        )
        knowledge_same_source = knowledge_store.record_experience(
            knowledge_dir=knowledge_dir,
            skills_dir=skills_dir,
            kind="knowledge",
            entry={
                "title": "probe knowledge from the same round",
                "family": "environment",
                "feature_profile": {"cv_global": 0.22, "node_count": 207},
                "strategy": {family: f"mod_{family}" for family in families},
                "tuning_direction": ["learning_rate 0.01 -> 0.005"],
                "source": "run:probe-task#r1",
            },
        )
        check(
            knowledge_same_source["id"] != skill["id"],
            "the knowledge and skill stores keep separate id namespaces",
        )

    evaluator_text = (ROOT / "agents/adaptation/prompts/effect_evaluator.py").read_text(encoding="utf-8")
    check(
        "composition.json" in evaluator_text and "cv_global" in evaluator_text,
        "the effect-evaluator prompt states the knowledge contract (retrieval features + measured composition)",
    )
    check(
        knowledge_store.strategy_module_ids(
            {"modules": {family: {"id": f"mod_{family}"} for family in families}}
        )
        == {family: f"mod_{family}" for family in families},
        "a composition.json modules block is accepted verbatim as the strategy",
    )


def _module_ids_in(strategy: Dict[str, Any], known_ids: Sequence[str]) -> set:
    """Module ids the knowledge entry actually names in its strategy."""

    text = json.dumps(strategy, ensure_ascii=False)
    return {module_id for module_id in known_ids if module_id in text}


def check_cli_teardown() -> None:
    """The CLI must exit once the run summary is printed.

    Observed live: the run finished, printed its JSON and then stayed alive for
    minutes, because ``asyncio.run`` waits without a timeout for the tasks and
    default-executor work openjiuwen leaves behind. The watchdog is proven by
    running it against an interpreter that blocks forever afterwards.
    """

    section("cli teardown")
    main_text = (ROOT / "main.py").read_text(encoding="utf-8")
    check("_arm_exit_watchdog(" in main_text, "main.py arms an exit watchdog")
    armed_at = main_text.find("_arm_exit_watchdog(0 if passed else 1")
    summary_at = main_text.find('"artifact_dir": result.get("artifact_dir"),')
    check(
        armed_at > summary_at >= 0,
        "the watchdog is armed only after the run summary is printed",
    )
    check("AGENTCITY_TEARDOWN_GRACE_S" in main_text, "the teardown grace is configurable")

    snippet = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from main import _arm_exit_watchdog\n"
        "_arm_exit_watchdog(7)\n"
        "time.sleep(600)\n"
    )
    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "AGENTCITY_TEARDOWN_GRACE_S": "2"},
            check=False,
            timeout=90,
        )
    except subprocess.TimeoutExpired:
        check(False, "a blocked interpreter teardown is bounded")
        return
    elapsed = time.time() - started
    check(proc.returncode == 7, f"the watchdog exits with the run's code (got {proc.returncode})")
    check(elapsed < 45, f"a blocked interpreter teardown is bounded (exited after {elapsed:.1f}s)")
    check("forcing exit 7" in proc.stderr, "the watchdog reports why it forced the exit")


def check_task_specs(paths: AppPaths) -> None:
    section("task specifications")
    specs = sorted((ROOT / "tasks").glob("*.json"))
    check(len(specs) >= 2, f"sample task specs present (found {len(specs)})")
    for spec_path in specs:
        try:
            spec = load_task_spec(spec_path)
        except Exception as exc:  # noqa: BLE001
            check(False, f"{spec_path.name} loads ({exc})")
            continue
        check(
            bool(spec["task_id"]) and bool(spec["dataset"]) and bool(spec["task"]),
            f"{spec_path.name}: required keys present",
        )
        check(isinstance(spec["train"]["gpu"], bool), f"{spec_path.name}: explicit CPU/GPU budget")
        check(
            spec["target"]["threshold"] is not None,
            f"{spec_path.name}: numeric target threshold set",
        )
        check(1 <= spec["max_rounds"] <= 10, f"{spec_path.name}: max_rounds in [1, 10]")


def check_workflow_and_registry(paths: AppPaths) -> None:
    section("workflow, layers and role roster")
    workflow = get_adaptation_workflow()
    keys = tuple(stage.key for stage in workflow.stages)
    check(keys == EXPECTED_STAGES, f"workflow stages are {EXPECTED_STAGES} (found {keys})")

    from agents.core.types import AgentContext
    from agents.adaptation.registry import build_execution_agents, build_tuning_agents

    context = AgentContext(repo_path=".", task_id="probe", task_slug="probe", task_dir=".", task_spec={})
    execution = build_execution_agents(context)
    tuning = build_tuning_agents(context)
    check(set(execution) == EXPECTED_EXECUTION_AGENTS, f"execution layer roster: {sorted(execution)}")
    check(set(tuning) == EXPECTED_TUNING_AGENTS, f"tuning layer roster: {sorted(tuning)}")

    all_agents = {**execution, **tuning}
    check(len(all_agents) == 6, "six layer agents registered")

    profile = HarnessProfile.load()
    for name, definition in all_agents.items():
        mapped = profile.resolve_tools(list(definition.tools))
        check(len(mapped) == len(definition.tools) and all(mapped), f"{name}: all tool names resolve")
        check(bool(definition.prompt.strip()), f"{name}: system prompt is populated")

    multi_stages = [stage for stage in workflow.stages if hasattr(stage, "lead_agent_prompt_builder")]
    check(len(multi_stages) == 2, "two team stages (execution, tuning)")
    check(
        all(stage.lead_model == "strong" for stage in multi_stages),
        "the decision layer is the lead of both team stages",
    )
    planning = workflow.stages[0]
    check(planning.model == "strong", "planning stage runs on the strong role")

    # The planning stage runs as a single DeepAgent; without an explicit
    # sys_operation rail it only receives the business MCP tools and cannot
    # persist plan.json (observed live: the planner reported "no write_file").
    import inspect

    from agents.core.openjiuwen_backend import run_single_prompt

    single_agent_source = inspect.getsource(run_single_prompt)
    check(
        "SysOperationRail" in single_agent_source and "rails=" in single_agent_source,
        "the single-agent stage mounts the filesystem toolset (read_file/write_file/bash)",
    )

    # The MCP stdio client defaults to a *minimal* environment when env is not
    # supplied, so the business tool server would lose AGENTCITY_LIBCITY_PYTHON
    # and launch LibCity with a bare "python" (observed live: ray/torch import
    # errors from the wrong interpreter).
    from agents.core.openjiuwen_backend import _business_mcp_config

    sentinel = "AGENTCITY_VALIDATOR_SENTINEL"
    os.environ[sentinel] = "forwarded"
    try:
        forwarded = _business_mcp_config().params.get("env") or {}
    finally:
        os.environ.pop(sentinel, None)
    check(
        isinstance(forwarded, dict) and bool(forwarded),
        "the business MCP server receives an explicit environment",
    )
    check(
        forwarded.get(sentinel) == "forwarded",
        "the business MCP server inherits the orchestrator environment "
        "(so AGENTCITY_LIBCITY_PYTHON reaches the LibCity subprocesses)",
    )
    check(
        all(forwarded.get(key) for key in ("PATH", "HOME")),
        "the forwarded environment still carries PATH/HOME for the tool subprocesses",
    )


def check_member_prompt_paths(paths: AppPaths) -> None:
    """Member prompts must carry absolute paths.

    Team members run inside their own isolated workspace, so a prompt that only
    says ``Bigscity-LibCity/raw_data/...`` leaves the model to guess the
    location. Observed live: a member ran ``find / -type d -name METR_LA`` and
    burned minutes before failing.
    """

    section("member prompt paths")
    from agents.adaptation.registry import build_execution_agents, build_tuning_agents
    from agents.core.config import build_context

    context = build_context(paths, {"task_id": "prompt_probe", "dataset": "PROBE"})
    members = {**build_execution_agents(context), **build_tuning_agents(context)}
    repo = str(paths.repo_path)
    task_dir = str(paths.task_dir("prompt_probe"))
    # Delegated members run in isolated workspaces, so the paths handed to them
    # (and to their shell commands) must be absolute.
    check(Path(repo).is_absolute(), f"the repository root handed to members is absolute ({repo})")
    check(Path(task_dir).is_absolute(), f"the task directory handed to members is absolute ({task_dir})")

    for name, definition in members.items():
        prompt = definition.prompt
        check(isinstance(prompt, str) and bool(prompt.strip()), f"{name}: prompt is a rendered string")
        check(repo in prompt, f"{name}: prompt carries the absolute repository root")
        check(task_dir in prompt, f"{name}: prompt carries the absolute task directory")
        check("<task_dir>" not in prompt, f"{name}: no unsubstituted <task_dir> placeholder")
        check(
            "__REPO__" not in prompt and "__TASK_DIR__" not in prompt,
            f"{name}: every path token is substituted",
        )
        check("Absolute Paths" in prompt, f"{name}: prompt lists the absolute paths")
        check(
            "Never search the filesystem" in prompt,
            f"{name}: prompt forbids filesystem-wide searching",
        )
        for line in prompt.splitlines():
            if "Bigscity-LibCity" in line:
                check(
                    repo in line,
                    f"{name}: LibCity references are absolute: {line.strip()[:70]!r}",
                )

    lead_prompts = {
        stage.key: stage.lead_agent_prompt_builder(context)
        for stage in get_adaptation_workflow().stages
        if hasattr(stage, "lead_agent_prompt_builder")
    }
    check(len(lead_prompts) == 2, "both team stages build a lead prompt")
    for key, prompt in lead_prompts.items():
        check(
            "Delegate with absolute paths only" in prompt,
            f"{key}: the lead is told to delegate with absolute paths",
        )
        check("<task_dir>" not in prompt, f"{key}: lead prompt has no unsubstituted placeholder")


def check_round_loop() -> None:
    """Lock the closed-loop bookkeeping: a new round must never read stale artifacts."""

    section("closed self-correction loop")
    from agents.adaptation.context import ExecutionContext
    from agents.core.orchestrator import ROUND_ARTIFACTS, ROUND_CARRYOVER, AgentOrchestrator

    check(len(ROUND_ARTIFACTS) == 7, f"seven round artifacts are collected (found {len(ROUND_ARTIFACTS)})")
    check("verdict.json" in ROUND_ARTIFACTS, "the round verdict is part of the collected artifacts")

    with tempfile.TemporaryDirectory() as tmp:
        store = ExecutionContext.load("loop_probe", base_dir=Path(tmp))
        for name in ROUND_ARTIFACTS:
            (store.dir / name).write_text(json.dumps({"round": 1}), encoding="utf-8")

        # Round 1 has no earlier round inside this run, but the artifact
        # directory is reused across CLI invocations, so pre-existing files are
        # leftovers of an earlier session rather than this run's results.
        AgentOrchestrator._archive_previous_round(store, 1)
        check(
            not (store.dir / "verdict.json").exists(),
            "a fresh run cannot inherit the previous session's verdict",
        )
        check(
            (store.dir / "dataset_profile.json").exists(),
            "the dataset profile survives into a fresh run (it is a reused data asset)",
        )
        carried = [name for name in ROUND_ARTIFACTS if name not in ROUND_CARRYOVER]
        check(
            all((store.dir / "rounds" / "previous_run" / name).exists() for name in carried),
            "round 1 stashes an earlier session's artifacts under rounds/previous_run",
        )
        for name in ROUND_ARTIFACTS:
            (store.dir / name).write_text(json.dumps({"round": 1}), encoding="utf-8")

        AgentOrchestrator._archive_previous_round(store, 2)
        moved = [name for name in ROUND_ARTIFACTS if name not in ROUND_CARRYOVER]
        archive_dir = store.dir / "rounds" / "round_1"
        check(
            all((archive_dir / name).exists() for name in moved),
            f"previous round artifacts archived to {archive_dir.name}/",
        )
        check(
            all(not (store.dir / name).exists() for name in moved),
            "a new round starts without the previous round's verdict/plan",
        )
        check(
            all((store.dir / name).exists() for name in ROUND_CARRYOVER),
            "the dataset profile (a data asset, not a round result) survives the round boundary",
        )


def check_metric_extraction() -> None:
    """Lock the metric contract: LibCity's evaluate_cache CSV is authoritative."""

    section("metric extraction contract")
    from agents.adaptation.utils import collect_run_metrics, extract_standard_metrics

    check(
        extract_standard_metrics("masked_MAE: 1.5") == {"masked_mae": 1.5},
        "stdout scraping still works as a fallback",
    )

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        cache = repo / "libcity" / "cache" / "12345" / "evaluate_cache"
        cache.mkdir(parents=True)
        (cache / "2026_01_01_00_00_00_RNN_METR_LA.csv").write_text(
            "MAE,MAPE,MSE,masked_MAE,masked_MAPE\n10,inf,100,2,0.1\n20,inf,400,4,0.3\n",
            encoding="utf-8",
        )

        metrics, detail = collect_run_metrics(
            repo, model_name="RNN", dataset="METR_LA", stdout="no metrics in this stdout"
        )
        check(detail.get("source_kind") == "evaluate_cache", "metrics come from the evaluate_cache CSV")
        check(
            math.isclose(float(metrics.get("masked_mae", -1)), 3.0),
            f"masked_mae is the mean over prediction horizons (got {metrics.get('masked_mae')})",
        )
        check(
            math.isclose(float(metrics.get("mae", -1)), 15.0),
            f"every metric is averaged over the horizons (got {metrics.get('mae')})",
        )
        check(
            math.isclose(float(metrics.get("masked_mape", -1)), 0.2),
            "non-finite horizons (MAPE=inf) are dropped instead of poisoning the mean",
        )

        _, stale = collect_run_metrics(
            repo, model_name="RNN", dataset="METR_LA", since=time.time() + 600
        )
        check(stale.get("source_kind") == "none", "a stale evaluate cache cannot be attributed to a new run")

        _, renamed = collect_run_metrics(repo, model_name="LSTM", dataset="METR_LA")
        check(
            renamed.get("source_kind") == "evaluate_cache",
            "RNN-family aliases still find the cache (LibCity renames LSTM/GRU to RNN)",
        )


def check_tuning_cli_contract() -> None:
    """Lock how trial knobs reach LibCity: CLI flags only where run_model.py has them."""

    section("tuning CLI contract")
    from agents.adaptation.tuning import (
        RUN_MODEL_CLI_FLAGS,
        trial_command,
        write_trial_config,
    )
    from agents.tools.impl import _unsupported_override_keys

    libcity = ROOT / "Bigscity-LibCity"
    declared = set()
    for line in (libcity / "run_model.py").read_text(encoding="utf-8").splitlines():
        marker = "add_argument('--"
        if marker in line:
            declared.add(line.split(marker, 1)[1].split("'", 1)[0])
    general = set()
    argument_list = libcity / "libcity" / "utils" / "argument_list.py"
    tree = ast.parse(argument_list.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "general_arguments":
            general = {key.value for key in node.value.keys}
    check(
        declared | general == RUN_MODEL_CLI_FLAGS,
        "RUN_MODEL_CLI_FLAGS matches run_model.py + libcity general_arguments",
    )
    check(bool(general), "run_model.py flag list was parsed from the vendored LibCity source")

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        (repo / "libcity" / "config" / "model" / "traffic_state_pred").mkdir(parents=True)
        base = repo / "libcity" / "config" / "model" / "traffic_state_pred" / "STDiffGRU_abl.json"
        base.write_text(json.dumps({"spatial_module": "none", "hidden_dim": 32}), encoding="utf-8")
        base_arg = "libcity/config/model/traffic_state_pred/STDiffGRU_abl"

        cmd, used = trial_command(
            repo_dir=repo,
            model_name="STDiffGRU",
            dataset_name="METR_LA",
            task="traffic_state_pred",
            params={"hidden_dim": 64, "learning_rate": 0.005},
            config_file=base_arg,
            extra_cli_args={"max_epoch": 1, "num_layers": 2},
            gpu=False,
            gpu_id="0",
        )
        check("--hidden_dim" not in cmd, "a model-config knob is not forwarded as a run_model.py flag")
        check("--learning_rate" in cmd and "0.005" in cmd, "a real CLI flag (learning_rate) still goes on the command line")
        check("--num_layers" not in cmd, "extra_cli_args model knobs are not forwarded as flags either")
        check("--max_epoch" in cmd, "extra_cli_args CLI knobs keep going on the command line")
        check(bool(used) and used.startswith("tuning_configs/"), f"trial config file is passed via --config_file (got {used!r})")
        written = repo / f"{used}.json"
        check("--config_file" in cmd and str(used) in cmd, "--config_file points at the generated trial config")
        if written.exists():
            payload = json.loads(written.read_text(encoding="utf-8"))
            check(payload.get("hidden_dim") == 64, "trial knob overrides the base config (hidden_dim 64)")
            check(payload.get("num_layers") == 2, "extra_cli_args model knobs are written to the trial config")
            check(payload.get("spatial_module") == "none", "the inherited base config survives the merge")
            check("learning_rate" not in payload, "CLI flags are left out of the config file (command line keeps precedence)")
        else:
            check(False, f"generated trial config exists: {written}")

        check(
            write_trial_config(repo, model_name="STDiffGRU", params={"learning_rate": 0.005, "batch_size": 32}) is None,
            "no config file is written when every knob is already a CLI flag",
        )

    check(_unsupported_override_keys({"hidden_dim": 64}) == ["hidden_dim"], "run_adaptation refuses a model knob instead of letting argparse fail")
    check(_unsupported_override_keys({"train_rate": 0.1, "max_epoch": 15, "exp_id": "x"}) == [], "run_adaptation accepts the flags run_model.py declares")


def check_tool_contract(paths: AppPaths) -> None:
    section("business tool + artifact contract")
    from agents.tools import server as tool_server

    registered = set(asyncio.run(_tool_names(tool_server)))
    for tool in BUSINESS_TOOLS:
        check(tool in registered, f"MCP tool registered: {tool}")

    from agents.adaptation.registry import build_execution_agents, build_tuning_agents
    from agents.core.types import AgentContext

    context = AgentContext(repo_path=".", task_id="probe", task_slug="probe", task_dir=".", task_spec={})
    referenced = set()
    for definition in {**build_execution_agents(context), **build_tuning_agents(context)}.values():
        referenced.update(name for name in definition.tools if name in BUSINESS_TOOLS)
    missing = referenced - registered
    check(not missing, f"every tool referenced by an agent is registered ({sorted(missing) or 'none missing'})")

    # The MCP tool server is spawned by the harness, which does not necessarily
    # forward our environment, so the LibCity interpreter must also be
    # recoverable from the CLI-written record before falling back to bare
    # `python` (observed live: the tool ran LibCity with a torch/ray-less
    # interpreter and the run-tester had to work around it by hand).
    from agents.adaptation import utils as adaptation_utils

    original_file = adaptation_utils.LIBCITY_PYTHON_FILE
    original_env = os.environ.pop(adaptation_utils.LIBCITY_PYTHON_ENV, None)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            adaptation_utils.LIBCITY_PYTHON_FILE = Path(tmp) / "libcity_python.txt"
            check(
                adaptation_utils.libcity_python() == "python",
                "without env or record the tool layer falls back to python on PATH",
            )
            adaptation_utils.record_libcity_python("/probe/bin/python")
            check(
                adaptation_utils.libcity_python() == "/probe/bin/python",
                "the recorded LibCity interpreter is used when the environment does not carry it",
            )
    finally:
        adaptation_utils.LIBCITY_PYTHON_FILE = original_file
        if original_env is not None:
            os.environ[adaptation_utils.LIBCITY_PYTHON_ENV] = original_env
    check(
        callable(adaptation_utils.record_libcity_python),
        "the CLI can persist the resolved LibCity interpreter",
    )
    main_text = (ROOT / "main.py").read_text(encoding="utf-8")
    check(
        "record_libcity_python" in main_text,
        "main.py records the LibCity interpreter for harness-spawned tool servers",
    )

    prompt_text = "\n".join(
        (ROOT / "agents/adaptation/prompts" / name).read_text(encoding="utf-8")
        for name in sorted(os.listdir(ROOT / "agents/adaptation/prompts"))
        if name.endswith(".py")
    )
    impl_text = (ROOT / "agents/tools/impl.py").read_text(encoding="utf-8")
    orchestrator_text = (ROOT / "agents/core/orchestrator.py").read_text(encoding="utf-8")
    for artifact, (producer, keys) in ARTIFACT_CONTRACT.items():
        check(artifact in ROUND_ARTIFACTS, f"round artifact declared: {artifact}")
        produced = artifact.replace(".json", "") in producer or artifact in impl_text or artifact in prompt_text
        check(produced, f"{artifact} has a producer ({producer})")
        check(
            all(key in prompt_text or key in impl_text for key in keys),
            f"{artifact} keys documented: {', '.join(keys)}",
        )
        check(artifact in prompt_text or artifact in orchestrator_text, f"{artifact} is consumed by a stage")
    check("tuning_result.json" not in impl_text, "no stale tuning_result.json artifact name")

    # Prompt/tool-schema alignment: the role prompts instruct agents to call a
    # tool with specific arguments, so a tool whose schema rejects those
    # arguments burns a turn per call (observed live: tune_adaptation rejected
    # `extra_cli_args`, and the error surfaced as "Extra inputs are not
    # permitted" before the agent retried without it).
    for tool_name in ("run_adaptation", "tune_adaptation"):
        tool = getattr(tool_server, tool_name)
        declared = set((getattr(tool, "parameters", None) or {}).get("properties") or {})
        check(
            "extra_cli_args" in declared,
            f"{tool_name} accepts extra_cli_args (the prompts forward train.extra_args through it)",
        )
    check(
        "extra_cli_args" in prompt_text,
        "the tuning/run prompts name extra_cli_args, so the schema and the prompts agree",
    )
    # run_hyper.py defaults to 100 evaluations; on a CPU host one evaluation is
    # a full training run, so the trial budget must also cap the LibCity tuner.
    check(
        "libcity_max_evals" in impl_text,
        "the trial budget also bounds LibCity's run_hyper.py evaluation count",
    )


async def _tool_names(tool_server) -> List[str]:
    tools = tool_server.mcp
    for attribute in ("get_tools", "_tool_manager"):
        value = getattr(tools, attribute, None)
        if attribute == "get_tools" and value is not None:
            result = value()
            if hasattr(result, "__await__"):
                result = await result
            return sorted(result)
        if attribute == "_tool_manager" and value is not None:
            return sorted(value._tools)  # noqa: SLF001
    return []


def check_no_stale_imports() -> None:
    section("import health")
    tracked = _tracked_python_files()
    stale: List[str] = []
    for path in tracked:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            stale.append(f"{path}: syntax error ({exc})")
            continue
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(REMOVED_MODULES):
                        stale.append(f"{path}: imports removed module {alias.name}")
                continue
            if module and any(module == removed or module.startswith(removed + ".") for removed in REMOVED_MODULES):
                stale.append(f"{path}: imports removed module {module}")
    check(not stale, f"no imports of removed modules ({len(stale)} found)")
    for item in stale:
        NOTES.append(item)

    missing = []
    for module in ("agents", "agents.core", "agents.adaptation", "agents.tools", "agents.adaptation.workflow"):
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            missing.append(f"{module}: {proc.stderr.strip().splitlines()[-1:]}")
    check(not missing, f"core packages import cleanly ({missing or 'all ok'})")

    # The business tools run inside a separate FastMCP stdio server process, so the
    # active task must be resolvable across the process boundary. These probes use a
    # private pointer file: the default one is shared repository state, and clearing it
    # from here would break a pipeline that is currently in flight, which then files
    # its runs under the model name (observed live).
    pointer = ROOT / "data" / "runs" / "active_task_validator.json"
    pointer.unlink(missing_ok=True)
    probe_env = {**os.environ, "AGENTCITY_ACTIVE_TASK_FILE": str(pointer)}
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "from agents.adaptation.runtime import set_active_task, clear_active_task, task_file;"
            "set_active_task({'task_id': 'xprocess_probe'});"
            "print(task_file())",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=probe_env,
        check=False,
    )
    check(
        probe.returncode == 0 and probe.stdout.strip() == str(pointer) and pointer.exists(),
        "active-task pointer is writable (AGENTCITY_ACTIVE_TASK_FILE override)",
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, os, pathlib;"
            "from agents.tools.impl import _active_task;"
            "print(json.dumps(_active_task()))",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=probe_env,
        check=False,
    )
    child_ok = '"task_id": "' in child.stdout and "xprocess_probe" in child.stdout
    check(child_ok, "a child process (the MCP tool server) resolves the active task")
    for item in (probe.stderr, child.stderr):
        if item.strip():
            NOTES.append(f"probe stderr: {item.strip().splitlines()[-1][:200]}")

    # If the shared pointer is lost (a concurrent smoke run, an unrelated probe),
    # the environment snapshot taken by set_active_task must still resolve the task
    # instead of silently falling back to the model name.
    fallback = subprocess.run(
        [sys.executable, "-c", _ENV_FALLBACK_PROBE],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=probe_env,
        check=False,
    )
    fallback_lines = fallback.stdout.split()
    check(
        fallback.returncode == 0 and "env_fallback_probe" in fallback.stdout and "WARN" in fallback_lines,
        "a lost pointer falls back to the environment snapshot (with a warning)",
    )
    if fallback.stderr.strip():
        NOTES.append(f"fallback probe stderr: {fallback.stderr.strip().splitlines()[-1][:200]}")
    pointer.unlink(missing_ok=True)

    main_text = (ROOT / "main.py").read_text(encoding="utf-8")
    check("agents.migration" not in main_text and "literature" not in main_text, "main.py has no legacy workflow import")
    check("--task" in main_text, "main.py exposes the --task CLI flag")


def check_run_record_filing(paths: AppPaths) -> None:
    """Run records must be filed under the adaptation task, never under the model."""

    section("run registration (task id resolution)")
    from agents.adaptation import runtime

    check(
        str(runtime.task_file().name).endswith("active_task.json"),
        "the shared active-task pointer keeps its default location",
    )

    pointer = paths.run_dir / "active_task_missing.json"
    pointer.unlink(missing_ok=True)
    probe = subprocess.run(
        [sys.executable, "-c", _UNRESOLVED_TASK_PROBE],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "AGENTCITY_ACTIVE_TASK_FILE": str(pointer)},
        check=False,
    )
    payload: Dict[str, Any] = {}
    try:
        payload = json.loads(probe.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        payload = {}
    if not payload:
        NOTES.append(f"filing probe output: {(probe.stdout or probe.stderr).strip()[-300:]}")

    check(payload.get("from_spec") == "real_task", "the active task id is used for the run record")
    check(payload.get("from_args") == "arg_task", "an explicit task_id argument wins over the active task")
    check(not payload.get("unresolved"), "an unresolved task yields no id instead of the model name")
    refused = {"run": "run_adaptation", "tune": "tune_adaptation", "analyze": "analyze_dataset"}
    for key, tool in refused.items():
        output = str(payload.get(key) or "")
        check(
            output.startswith(f"{tool} failed:") and "active adaptation task" in output,
            f"{tool} refuses to run without a resolvable active task",
        )
    check(
        not (paths.adaptation_dir / "stdiffgru").exists(),
        "no run record is filed under data/adaptation/<model_name>",
    )


def _tracked_python_files() -> List[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    files = [
        line.strip()
        for line in proc.stdout.splitlines()
        if line.strip() and not line.startswith("Bigscity-LibCity/")
    ]
    if not files:
        files = [str(path.relative_to(ROOT)) for path in ROOT.rglob("*.py") if "Bigscity-LibCity" not in path.parts]
    return [ROOT / item for item in files]


def check_tool_smoke(paths: AppPaths) -> None:
    section("tool layer CPU smoke (data integration -> run -> tuning)")
    if os.environ.get("AGENTCITY_TOOL_SMOKE") != "1":
        from agents.adaptation.utils import libcity_python

        # Same reasoning as the contract probes: the smoke run sets and clears the
        # active task, so keep it on a private pointer and leave a live run alone.
        smoke_pointer = paths.run_dir / "active_task_smoke.json"
        proc = subprocess.run(
            [libcity_python(), str(Path(__file__).resolve()), "--tool-smoke-inner"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "AGENTCITY_TOOL_SMOKE": "1",
                "AGENTCITY_ACTIVE_TASK_FILE": str(smoke_pointer),
            },
            check=False,
        )
        smoke_pointer.unlink(missing_ok=True)
        print(proc.stdout.strip()[-4000:])
        if proc.returncode != 0:
            print(proc.stderr.strip()[-2000:])
        check(proc.returncode == 0 and "TOOL-SMOKE OK" in proc.stdout, "tool layer smoke run succeeded")
        return

    from agents.adaptation.runtime import clear_active_task, set_active_task
    from agents.tools import impl

    spec = load_task_spec(ROOT / "tasks" / "metr_la_speed_smoke.json")
    spec["train"]["max_epoch"] = int(os.environ.get("AGENTCITY_SMOKE_EPOCHS", "1"))
    spec["train"]["gpu"] = False
    model_name = os.environ.get("AGENTCITY_SMOKE_MODEL", "RNN")
    print(
        f"smoke budget: model={model_name} dataset={spec['dataset']} "
        f"max_epoch={spec['train']['max_epoch']} batch_size={spec['train']['batch_size']} "
        f"extra_args={spec['train']['extra_args']}"
    )
    print("note: one LibCity pass over a real CPU dataset takes ~10-25 min; this is expected.")
    set_active_task(spec)
    try:
        profile_output = asyncio.run(impl.analyze_dataset({"dataset": spec["dataset"], "task": spec["task"]}))
        print(profile_output.strip()[:1500])
        run_output = asyncio.run(
            impl.run_adaptation(
                {
                    "model_name": model_name,
                    "dataset": spec["dataset"],
                    "task": spec["task"],
                    "max_epoch": spec["train"]["max_epoch"],
                    "batch_size": spec["train"]["batch_size"],
                    "gpu": False,
                }
            )
        )
        print(run_output.strip()[:2500])
        task_dir = paths.task_dir(spec["task_id"])
        report = json.loads((task_dir / "run_report.json").read_text(encoding="utf-8"))
        if report.get("status") == "success" and report.get("metrics"):
            print("TOOL-SMOKE OK", json.dumps(report["metrics"], ensure_ascii=False))
        else:
            print("TOOL-SMOKE FAILED", report.get("status"), report.get("error_signature"))
    finally:
        clear_active_task()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the adaptation pipeline plumbing.")
    parser.add_argument("--import-modules", action="store_true", help="torch shape probe for every module")
    parser.add_argument("--tool-smoke", action="store_true", help="run the tool layer on CPU end to end")
    parser.add_argument("--tool-smoke-inner", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--module-probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    paths = AppPaths()
    paths.ensure()

    if args.module_probe or os.environ.get("AGENTCITY_MODULE_PROBE") == "1":
        problems = _module_shape_probe_inner(paths)
        for problem in problems:
            print(f"MODULE-FAIL: {problem}")
        print("MODULE-PROBE OK" if not problems else f"MODULE-PROBE FAILED ({len(problems)})")
        return 1 if problems else 0

    inner = os.environ.get("AGENTCITY_TOOL_SMOKE") == "1"
    if inner or args.tool_smoke_inner:
        check_tool_smoke(paths)
    else:
        check_repo_layout(paths)
        check_module_library(paths, import_modules=args.import_modules)
        check_knowledge_and_skills(paths)
        check_task_specs(paths)
        check_workflow_and_registry(paths)
        check_member_prompt_paths(paths)
        check_feedback_write_path()
        check_round_loop()
        check_cli_teardown()
        check_metric_extraction()
        check_tuning_cli_contract()
        check_tool_contract(paths)
        check_run_record_filing(paths)
        check_no_stale_imports()
        if args.tool_smoke:
            check_tool_smoke(paths)

    print("")
    if FAILURES:
        print(f"RESULT: FAILED ({len(FAILURES)} check(s))")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("RESULT: PASSED (all checks)")
    if NOTES:
        print(f"notes ({len(NOTES)}):")
        for note in NOTES:
            print(f"  - {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
