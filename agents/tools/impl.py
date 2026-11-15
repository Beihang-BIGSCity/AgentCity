"""Business tool implementations for the AgentCity MCP server.

Every entry point is ``async def tool(args: dict) -> str`` so the FastMCP
server (server.py) can delegate without blocking openjiuwen's async engine.
Tools that belong to one adaptation run resolve the active task through
``agents.adaptation.runtime`` and write their structured output into that
task's shared execution context directory.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup

from agents import AppPaths
from agents.adaptation import knowledge as knowledge_store
from agents.adaptation import modlib as module_library
from agents.adaptation.runtime import get_active_task
from agents.adaptation.storage import AdaptationResultStore
from agents.adaptation.tuning import (
    RUN_MODEL_CLI_FLAGS,
    ensure_search_space,
    is_libcity_tuning_allowed,
    run_custom_grid_search,
    run_libcity_tuning,
    write_params_file,
)
from agents.adaptation.utils import (
    collect_run_metrics,
    compute_dataset_profile,
    libcity_python,
    sanitize_slug,
)

paths = AppPaths()
paths.ensure()

result_store = AdaptationResultStore(
    root=paths.root,
    base_dir=paths.adaptation_dir,
    index_path=paths.adaptation_index,
)

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")

RUN_TIMEOUT = 3600
TUNE_TIMEOUT = 7200


# ---------------------------------------------------------------------------
# argument coercion helpers
# ---------------------------------------------------------------------------


def _as_bool(value, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _as_int(value, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_project_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = paths.root / candidate
    return candidate


def _truncate_text(value: str, limit: int = 12000) -> str:
    text = (value or "").strip()
    if len(text) > limit:
        return f"{text[:limit]}...[truncated]"
    return text


def _libcity_root() -> Path:
    """Resolve the LibCity checkout the pipeline adapts into."""

    repo = paths.repo_path
    return repo if repo.is_absolute() else (paths.root / repo).resolve()


NO_ACTIVE_TASK = (
    "the tool server could not resolve the active adaptation task "
    "(the active-task pointer is missing and no environment snapshot is available)"
)


def _active_task() -> Dict[str, Any]:
    return dict(get_active_task() or {})


def _resolve_task_id(args: Dict[str, Any], task_spec: Dict[str, Any]) -> str:
    """Pick the adaptation task a tool call belongs to.

    Run records and the round artifacts are filed under this id, so it must come
    from the pipeline's active task (or an explicit argument). Falling back to
    the model name - the previous behaviour - scattered records into
    ``data/adaptation/<model>/`` and left the real task directory stale.
    """

    for candidate in (args.get("task_id"), task_spec.get("task_id")):
        resolved = str(candidate or "").strip()
        if resolved:
            return resolved
    return ""


def _unresolved_task_error(action: str) -> str:
    return (
        f"{action} failed: {NO_ACTIVE_TASK}, so the result cannot be filed under a task. "
        "Re-run through `python main.py --task <spec>` (or point AGENTCITY_ACTIVE_TASK_FILE at the "
        "pointer) instead of filing the record under the model name."
    )


def _require_task_dir() -> Path | None:
    task = _active_task()
    task_id = str(task.get("task_id") or "").strip()
    if not task_id:
        return None
    task_dir = paths.task_dir(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def _write_artifact(name: str, payload: Any) -> Path | None:
    task_dir = _require_task_dir()
    if task_dir is None:
        print(
            f"warning: {NO_ACTIVE_TASK}; {name} was not written to disk",
            file=sys.stderr,
            flush=True,
        )
        return None
    target = task_dir / name
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def _update_context(section: str, payload: Any) -> None:
    """Mirror a structured result into the shared execution context."""

    task = _active_task()
    task_id = str(task.get("task_id") or "").strip()
    if not task_id:
        return
    from agents.adaptation.context import ExecutionContext

    store = ExecutionContext.load(task_id, base_dir=paths.adaptation_dir)
    store.update(section, payload)


def _append_context(section: str, payload: Any) -> None:
    """Append a structured result to a growing list in the shared context."""

    task = _active_task()
    task_id = str(task.get("task_id") or "").strip()
    if not task_id:
        return
    from agents.adaptation.context import ExecutionContext

    store = ExecutionContext.load(task_id, base_dir=paths.adaptation_dir)
    store.append(section, payload)


# ---------------------------------------------------------------------------
# data integration
# ---------------------------------------------------------------------------


async def analyze_dataset(args: dict) -> str:
    """Compute the task-level statistical signals for one LibCity dataset."""

    task_spec = _active_task()
    train_budget = task_spec.get("train") or {}
    dataset = args.get("dataset") or task_spec.get("dataset")
    task = args.get("task") or task_spec.get("task") or "traffic_state_pred"
    windows = _as_int(args.get("windows"), 10) or 10
    max_timesteps = _as_int(args.get("max_timesteps"), train_budget.get("profile_max_timesteps", 20000)) or 20000
    data_col = args.get("data_col") or train_budget.get("data_col")
    if isinstance(data_col, list):
        if len(data_col) != 1:
            return "analyze_dataset failed: profiling requires one explicit target column"
        data_col = data_col[0]
    profile_train_rate = args.get("train_rate")
    if profile_train_rate is None and train_budget.get("profile_train_only"):
        profile_train_rate = (train_budget.get("extra_args") or {}).get("train_rate")
    if profile_train_rate is None:
        profile_train_rate = 1.0
    if not dataset:
        return "analyze_dataset failed: provide dataset"
    # dataset_profile.json is part of the round's artifact contract, so resolve the
    # task before spending the scan: an unresolvable task is a broken run, not a
    # reason to silently skip the file (the caller cannot see it was not written).
    if not _resolve_task_id(args, task_spec):
        return _unresolved_task_error("analyze_dataset")

    raw_data_dir = _libcity_root() / "raw_data"
    try:
        profile = await asyncio.to_thread(
            compute_dataset_profile,
            dataset=str(dataset),
            raw_data_dir=raw_data_dir,
            task=str(task),
            windows=windows,
            max_timesteps=max_timesteps,
            data_col=data_col,
            train_rate=float(profile_train_rate),
        )
    except Exception as exc:
        return f"analyze_dataset failed: {exc}"

    _write_artifact("dataset_profile.json", profile)
    _update_context("dataset_profile", profile)
    summary = {
        key: profile.get(key)
        for key in (
            "dataset",
            "data_col",
            "node_count",
            "timesteps",
            "sampling_interval_min",
            "has_graph",
            "null_ratio",
            "cv_global",
            "cv_node",
            "spatial_heterogeneity",
            "rho1",
        )
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# knowledge base / skills
# ---------------------------------------------------------------------------


async def retrieve_knowledge(args: dict) -> str:
    """Retrieve the closest knowledge entries and skills for a feature profile."""

    features = args.get("features")
    if not isinstance(features, dict):
        features = {}
    if not features:
        task_dir = _require_task_dir()
        profile_path = (task_dir / "dataset_profile.json") if task_dir else None
        if profile_path and profile_path.exists():
            features = json.loads(profile_path.read_text(encoding="utf-8"))
    if not features:
        return "retrieve_knowledge failed: provide features (or run analyze_dataset first)"

    kinds = args.get("kinds")
    if isinstance(kinds, str):
        kinds = [kinds]
    if not isinstance(kinds, list) or not kinds:
        kinds = ["knowledge", "skill"]

    try:
        payload = knowledge_store.retrieve(
            knowledge_dir=paths.knowledge_dir,
            skills_dir=paths.skills_dir,
            features=features,
            top_k=_as_int(args.get("top_k"), 3) or 3,
            kinds=kinds,
            include_documents=bool(_as_bool(args.get("include_documents"), False)),
        )
    except Exception as exc:
        return f"retrieve_knowledge failed: {exc}"
    return json.dumps(payload, ensure_ascii=False, indent=2)


async def record_experience(args: dict) -> str:
    """Feed a finished round back into the expert knowledge base or the skills."""

    kind = args.get("kind")
    entry = args.get("entry")
    if not isinstance(entry, dict):
        entry = {key: value for key, value in args.items() if key != "kind"}
    task = _active_task()
    if task.get("task_id") and not entry.get("source"):
        entry["source"] = f"run:{task.get('task_id')}"
    try:
        recorded = knowledge_store.record_experience(
            knowledge_dir=paths.knowledge_dir,
            skills_dir=paths.skills_dir,
            kind=str(kind or ""),
            entry=entry,
        )
    except Exception as exc:
        return f"record_experience failed: {exc}"
    return f"Experience recorded as {recorded.get('id')} (kind={kind})."


# ---------------------------------------------------------------------------
# atomic module library
# ---------------------------------------------------------------------------


async def list_modules(args: dict) -> str:
    """List atomic modules, optionally filtered by category and tags."""

    tags = args.get("tags")
    if isinstance(tags, str):
        tags = [tags]
    try:
        modules = module_library.list_modules(
            paths.modlib_dir,
            category=args.get("category"),
            tags=tags if isinstance(tags, list) else None,
            include_code=bool(_as_bool(args.get("include_code"), False)),
        )
    except Exception as exc:
        return f"list_modules failed: {exc}"
    if not modules:
        return "list_modules: no module matched the filter."
    return json.dumps(modules, ensure_ascii=False, indent=2)


async def get_module(args: dict) -> str:
    """Return one module entry together with its source code."""

    module_id = str(args.get("module_id") or "").strip()
    if not module_id:
        return "get_module failed: provide module_id"
    try:
        entry = module_library.get_module(
            paths.modlib_dir,
            module_id,
            include_code=bool(_as_bool(args.get("include_code"), True)),
        )
    except KeyError as exc:
        return f"get_module failed: {exc}"
    except Exception as exc:
        return f"get_module failed: {exc}"
    return json.dumps(entry, ensure_ascii=False, indent=2)


async def register_module(args: dict) -> str:
    """Add a newly extracted atomic module to the library (idempotent by id)."""

    try:
        entry = module_library.register_module(
            paths.modlib_dir,
            module_id=str(args.get("module_id") or ""),
            category=str(args.get("category") or ""),
            class_name=str(args.get("class_name") or args.get("class") or ""),
            code=str(args.get("code") or ""),
            name=args.get("name"),
            desc=str(args.get("desc") or ""),
            ctr_args=args.get("ctr_args") if isinstance(args.get("ctr_args"), dict) else None,
            io=str(args.get("io") or ""),
            tags=args.get("tags") if isinstance(args.get("tags"), list) else None,
            source=args.get("source") if isinstance(args.get("source"), dict) else None,
        )
    except Exception as exc:
        return f"register_module failed: {exc}"
    return f"Module {entry['id']} registered under {entry['category']} ({entry['file']})."


# ---------------------------------------------------------------------------
# LibCity execution
# ---------------------------------------------------------------------------


async def _run_subprocess(cmd: List[str], cwd: Path, timeout: int) -> tuple[int, str, str]:
    """Run a subprocess without blocking the event loop."""

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "", f"process killed after {timeout}s timeout"
    return proc.returncode, stdout_b.decode(errors="replace"), stderr_b.decode(errors="replace")


def _libcity_cmd(
    *,
    script: str,
    model_name: str,
    dataset: str,
    task: str,
    gpu: bool,
    gpu_id: str,
    config_file: str | None = None,
    overrides: Dict[str, Any] | None = None,
) -> List[str]:
    cmd = [
        libcity_python(),
        script,
        "--task",
        task,
        "--model",
        model_name,
        "--dataset",
        dataset,
        "--gpu",
        "true" if gpu else "false",
    ]
    if gpu:
        cmd.extend(["--gpu_id", str(gpu_id)])
    if config_file:
        cmd.extend(["--config_file", config_file])
    for key, value in (overrides or {}).items():
        if value is None or key in {"gpu", "gpu_id", "task", "model", "dataset", "config_file"}:
            continue
        cmd.extend([f"--{key}", str(value)])
    return cmd


def _unsupported_override_keys(overrides: Dict[str, Any]) -> List[str]:
    """Flag override keys that ``run_model.py`` cannot accept as command-line flags.

    ``run_model.py`` declares a fixed set of flags; anything else is a model
    config key (``hidden_dim``, ``num_layers``, ``dropout``, ...). argparse
    rejects the whole command with "unrecognized arguments", so the run dies
    before training instead of reporting a usable error. Those knobs belong in a
    ``--config_file``.
    """

    return sorted(
        str(key)
        for key, value in (overrides or {}).items()
        if value is not None and str(key) not in RUN_MODEL_CLI_FLAGS
    )


def _interpreter_hint(interpreter: str, output: str) -> str:
    """Explain the classic wrong-interpreter failure instead of leaving a bare ModuleNotFoundError."""

    missing = re.search(r"No module named '([^']+)'", output or "")
    if not missing:
        return ""
    module = missing.group(1)
    if module not in {"torch", "ray", "libcity", "tensorflow"}:
        return ""
    if interpreter != "python":
        return (
            f"{interpreter!r} cannot import {module!r}; point AGENTCITY_LIBCITY_PYTHON at an "
            f"interpreter that has LibCity's dependencies installed."
        )
    return (
        f"LibCity was launched with a bare 'python' from PATH and cannot import {module!r}. Set "
        f"AGENTCITY_LIBCITY_PYTHON (or pass --libcity-python) to the interpreter that has LibCity installed; "
        f"the MCP tool server only sees it when the harness forwards its environment."
    )


async def run_adaptation(args: dict) -> str:
    """Train/evaluate the composed model on the scenario dataset."""

    model_name = args.get("model_name")
    task_spec = _active_task()
    train_budget = task_spec.get("train") or {}
    dataset = args.get("dataset") or task_spec.get("dataset") or "METR_LA"
    task = args.get("task") or task_spec.get("task") or "traffic_state_pred"
    gpu = bool(_as_bool(args.get("gpu"), bool(train_budget.get("gpu", False))))
    gpu_id = str(args.get("gpu_id", "0"))
    max_epoch = _as_int(args.get("max_epoch"), _as_int(train_budget.get("max_epoch"), 2))
    batch_size = _as_int(args.get("batch_size"), _as_int(train_budget.get("batch_size"), 16))
    config_file = args.get("config_file")
    seed = _as_int(args.get("seed"))
    saved_model = _as_bool(args.get("saved_model"))
    train_flag = _as_bool(args.get("train"))
    extra_cli_args = args.get("extra_cli_args")
    overrides: Dict[str, Any] = dict(train_budget.get("extra_args") or {})
    if isinstance(extra_cli_args, dict):
        overrides.update(extra_cli_args)
    overrides.setdefault("max_epoch", max_epoch)
    overrides.setdefault("batch_size", batch_size)
    if seed is not None:
        overrides.setdefault("seed", seed)
    if saved_model is not None:
        overrides.setdefault("saved_model", "true" if saved_model else "false")
    if train_flag is not None:
        overrides.setdefault("train", "true" if train_flag else "false")
    if not model_name:
        return "run_adaptation failed: provide model_name"
    unsupported = _unsupported_override_keys(overrides)
    if unsupported:
        return (
            "run_adaptation failed: run_model.py has no flag for "
            + ", ".join(f"--{key}" for key in unsupported)
            + "; pass model hyperparameters through a config_file "
            "(libcity/config/model/<task>/<name>.json) instead of extra_cli_args."
        )
    task_id = _resolve_task_id(args, task_spec)
    if not task_id:
        return _unresolved_task_error("run_adaptation")

    repo_dir = _libcity_root()
    # Deterministic experiment id: it is the cache directory the metrics are
    # read back from, so the run and the metric reader must agree on it.
    exp_id = str(args.get("exp_id") or f"{sanitize_slug(task_id)}_{sanitize_slug(str(model_name))}")
    overrides.setdefault("exp_id", exp_id)
    cmd = _libcity_cmd(
        script="run_model.py",
        model_name=str(model_name),
        dataset=str(dataset),
        task=str(task),
        gpu=gpu,
        gpu_id=gpu_id,
        config_file=str(config_file) if config_file else None,
        overrides=overrides,
    )
    started_at = time.time()
    returncode, stdout, stderr = await _run_subprocess(cmd, repo_dir, RUN_TIMEOUT)
    status = "success" if returncode == 0 else "error"
    metrics, metrics_detail = collect_run_metrics(
        repo_dir, model_name=str(model_name), dataset=str(dataset), stdout=stdout, since=started_at
    )
    error_signature = _error_signature(stdout, stderr) if status == "error" else ""

    interpreter = libcity_python()
    report = {
        "model_name": str(model_name),
        "dataset": str(dataset),
        "task": str(task),
        "status": status,
        "metrics": metrics,
        "metrics_source": metrics_detail.get("source_kind", "none"),
        "metrics_detail": metrics_detail,
        "exp_id": exp_id,
        "interpreter": interpreter,
        "command": " ".join(cmd),
        "returncode": returncode,
        "stdout_excerpt": _truncate_text(stdout, 4000),
        "error_signature": error_signature,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    if status == "error":
        report["interpreter_hint"] = _interpreter_hint(interpreter, stdout + stderr)
    _write_artifact("run_report.json", report)

    result_store.record_run(
        task_id,
        {
            "model_name": str(model_name),
            "dataset": str(dataset),
            "task": str(task),
            "status": status,
            "metrics": metrics,
            "metrics_detail": metrics_detail,
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
            "command": " ".join(cmd),
        },
    )
    _append_context("runs", report)

    lines = [f"Status: {status}", f"Command: {' '.join(cmd)}"]
    if metrics:
        lines.append(
            f"Metrics ({metrics_detail.get('source_kind')}): "
            + ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        )
    else:
        lines.append("Metrics: none reported by LibCity")
    if error_signature:
        lines.append(f"Error signature: {error_signature}")
    lines.append(_truncate_text(stdout or stderr, 2000))
    return "\n".join(lines)


async def tune_adaptation(args: dict) -> str:
    """Run hyperparameter search for the composed model."""

    model_name = args.get("model_name")
    if not model_name:
        return "tune_adaptation failed: provide model_name"

    task_spec = _active_task()
    task_id = _resolve_task_id(args, task_spec)
    if not task_id:
        return _unresolved_task_error("tune_adaptation")
    train_budget = task_spec.get("train") or {}
    dataset = args.get("dataset") or task_spec.get("dataset") or "METR_LA"
    task = args.get("task") or task_spec.get("task") or "traffic_state_pred"
    gpu = bool(_as_bool(args.get("gpu"), bool(train_budget.get("gpu", False))))
    gpu_id = str(args.get("gpu_id", "0"))
    max_epoch = _as_int(args.get("max_epoch"), _as_int(train_budget.get("max_epoch"), 2))
    batch_size = _as_int(args.get("batch_size"), _as_int(train_budget.get("batch_size"), 16))
    config_file = args.get("config_file")
    hyper_algo = args.get("hyper_algo")
    tuning_budget = task_spec.get("tuning") or {}
    raw_search_space = args.get("search_space") or tuning_budget.get("search_space")
    search_space = ensure_search_space(raw_search_space if isinstance(raw_search_space, dict) else None)
    params_file_arg = _resolve_project_path(args.get("params_file"))
    if not hyper_algo:
        hyper_algo = tuning_budget.get("hyper_algo")
    max_evals = _as_int(args.get("max_evals"), _as_int(tuning_budget.get("max_evals")))
    max_trials = _as_int(args.get("max_trials"), _as_int(tuning_budget.get("max_trials")))
    if max_evals is not None and max_evals <= 0:
        max_evals = None
    if max_trials is not None and max_trials <= 0:
        max_trials = None
    repo_dir = _libcity_root()
    extra_cli_args: Dict[str, Any] = dict(train_budget.get("extra_args") or {})
    forwarded = args.get("extra_cli_args")
    if isinstance(forwarded, dict):
        extra_cli_args.update(forwarded)
    extra_cli_args.setdefault("max_epoch", max_epoch)
    if batch_size:
        extra_cli_args.setdefault("batch_size", batch_size)

    result: Dict[str, Any] | None = None
    libcity_failure = ""
    libcity_attempted = is_libcity_tuning_allowed()
    # LibCity's run_hyper.py defaults to 100 evaluations. On this CPU host one
    # evaluation is a full training run, so an unbounded default would never
    # finish: the trial budget also caps the LibCity tuner.
    libcity_max_evals = max_evals if max_evals is not None else max_trials
    if libcity_attempted:
        try:
            params_file = (
                params_file_arg
                if params_file_arg and params_file_arg.exists()
                else write_params_file(paths.run_dir, search_space)
            )
            candidate = await asyncio.wait_for(
                asyncio.to_thread(
                    run_libcity_tuning,
                    repo_dir=repo_dir,
                    run_dir=paths.run_dir,
                    model_name=str(model_name),
                    dataset_name=str(dataset),
                    params_file=params_file,
                    task=str(task),
                    config_file=str(config_file) if config_file else None,
                    hyper_algo=hyper_algo,
                    max_evals=libcity_max_evals,
                    gpu=gpu,
                    gpu_id=gpu_id,
                    extra_cli_args=extra_cli_args,
                ),
                timeout=TUNE_TIMEOUT,
            )
            if candidate.get("status") == "success":
                result = candidate
            else:
                libcity_failure = f"LibCity tuner returned {candidate.get('status')}"
        except Exception as exc:
            libcity_failure = f"LibCity tuner failed: {exc}"

    if result is None:
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    run_custom_grid_search,
                    repo_dir=repo_dir,
                    model_name=str(model_name),
                    dataset_name=str(dataset),
                    search_space=search_space,
                    task=str(task),
                    config_file=str(config_file) if config_file else None,
                    max_trials=max_trials,
                    gpu=gpu,
                    gpu_id=gpu_id,
                    extra_cli_args=extra_cli_args,
                ),
                timeout=TUNE_TIMEOUT,
            )
        except Exception as exc:
            return f"tune_adaptation encountered an error: {exc}"

    metrics = result.get("best_metrics") or {}
    payload: Dict[str, Any] = {
        "strategy": result.get("strategy"),
        "status": result.get("status"),
        "search_space": search_space,
        "best_params": result.get("best_params") or {},
        "best_metrics": metrics,
        "artifacts": result.get("artifacts") or {},
        "libcity_failure": libcity_failure,
        "interpreter": libcity_python(),
        "debug_log": [],
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    if result.get("trials"):
        payload["trials"] = result["trials"]
    _write_artifact("tuning.json", payload)
    _update_context("tuning", payload)

    result_store.record_run(
        task_id,
        {
            "model_name": str(model_name),
            "dataset": str(dataset),
            "task": str(task),
            "status": str(result.get("status") or "error"),
            "kind": "tuning",
            "metrics": metrics,
            "best_params": payload["best_params"],
            "stdout": (result.get("stdout") or "").strip(),
            "stderr": (result.get("stderr") or "").strip(),
        },
    )

    lines = [f"Tuning strategy: {payload['strategy']}", f"Status: {payload['status']}"]
    if libcity_failure:
        lines.append(f"LibCity tuner skipped: {libcity_failure}")
    if payload["best_params"]:
        lines.append(f"Best parameters: {json.dumps(payload['best_params'], ensure_ascii=False)}")
    if metrics:
        lines.append("Best metrics: " + ", ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
    if payload.get("trials"):
        lines.append(f"Trials run: {len(payload['trials'])}")
    return _truncate_text("\n".join(lines))


def _error_signature(stdout: str, stderr: str) -> str:
    """Extract the shortest distinctive failure line from LibCity output."""

    candidates: List[str] = []
    for line in (stderr or "").splitlines() + (stdout or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("Error", "Traceback", "RuntimeError", "ValueError", "KeyError", "AssertionError")):
            candidates.append(stripped)
        elif "Error:" in stripped and len(stripped) < 300:
            candidates.append(stripped)
    if candidates:
        return candidates[-1][:500]
    tail = (stderr or stdout or "").strip().splitlines()
    return tail[-1][:500] if tail else "unknown failure"


# ---------------------------------------------------------------------------
# generic web / document tools (kept for module research)
# ---------------------------------------------------------------------------


def _web_search_sync(query: str, num: int) -> str:
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": num, "page": 1})
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload, timeout=30)
    if response.status_code != 200:
        return f"web_search failed: HTTP {response.status_code}: {response.text[:500]}"

    results = json.loads(response.text)
    organic = [
        {
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        }
        for item in results.get("organic", [])
    ]
    return json.dumps(organic, ensure_ascii=False, indent=2)


async def web_search(args: dict) -> str:
    """Search the web via the Serper Google Search API."""
    query = args.get("query")
    num = _as_int(args.get("num"), 10)

    if not query:
        return "web_search failed: query is required"

    return await asyncio.to_thread(_web_search_sync, query, num)


def _web_fetch_sync(url: str, max_chars: int) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return _truncate_text(text, limit=max_chars)


async def web_fetch(args: dict) -> str:
    """Fetch a URL and return its plain-text content, truncated to max_chars."""
    url = args.get("url")
    max_chars = _as_int(args.get("max_chars"), 20000)

    if not url:
        return "web_fetch failed: url is required"

    try:
        return await asyncio.to_thread(_web_fetch_sync, url, max_chars)
    except Exception as exc:
        return f"web_fetch failed: {exc}"


def _read_pdf_text_sync(pdf_path: Path, max_pages: int, max_chars: int) -> str:
    import pdfplumber

    chunks: List[str] = []
    total = 0
    with pdfplumber.open(str(pdf_path)) as pdf:
        for index, page in enumerate(pdf.pages):
            if index >= max_pages:
                chunks.append(f"...[truncated after {max_pages} pages]")
                break
            text = page.extract_text() or ""
            remaining = max_chars - total
            if remaining <= 0:
                chunks.append("...[truncated by max_chars]")
                break
            chunks.append(text[:remaining])
            total += len(text[:remaining])
    return "\n\n".join(chunks).strip()


async def read_pdf_text(args: dict) -> str:
    """Extract plain text from a PDF file (kept for module research)."""
    raw_path = args.get("pdf_path") or args.get("path")
    max_pages = _as_int(args.get("max_pages"), 40)
    max_chars = _as_int(args.get("max_chars"), 60000)

    pdf_path = _resolve_project_path(raw_path)
    if pdf_path is None or not pdf_path.exists():
        return f"read_pdf_text failed: file not found: {raw_path}"

    try:
        text = await asyncio.to_thread(_read_pdf_text_sync, pdf_path, max_pages, max_chars)
    except Exception as exc:
        return f"read_pdf_text failed: {exc}"
    if not text:
        return f"read_pdf_text: no extractable text in {pdf_path}"
    return text
