"""CLI entry point for AgentCity's spatio-temporal model adaptation pipeline.

The pipeline is the single ``任务规划 + 数据集成/模型组合/运行测试 + 超参数搜索/性能调试/效果评估``
workflow implemented under ``agents/adaptation`` and executed by
``agents/core/orchestrator.py`` on the openJiuwen backend.

Usage::

    python main.py --task tasks/metr_la_speed_smoke.json
    python main.py --task tasks/metr_la_speed_smoke.json --max-rounds 2
    python main.py --list-tasks

Environment
-----------
``AGENTCITY_LIBCITY_PYTHON``
    Interpreter able to import LibCity + torch (this process only orchestrates).
``AGENTCITY_API_KEY`` / ``AGENTCITY_API_BASE``
    Override the provider credentials in ``agents/core/harness_profiles/``.
``AGENTCITY_TEARDOWN_GRACE_S``
    Seconds the exit watchdog waits for the interpreter to finish on its own
    once the run summary is printed (default 30). ``AGENTCITY_NO_EXIT_WATCHDOG``
    disables the watchdog.
``AGENTCITY_ACTIVE_TASK_FILE``
    Where the run publishes its active-task pointer for the MCP tool server.
    Defaults to ``data/runs/active_task.<pid>.json`` so concurrent runs cannot
    steal each other's run records.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from agents import AppPaths, setup_logger
from agents.core.config import EFFICIENCY_CHOICES, load_task_spec
from agents.core.orchestrator import AgentOrchestrator

GPU_CHOICES = ("auto", "on", "off")


def _normalise_proxy_env() -> None:
    """Drop a SOCKS ``ALL_PROXY`` that httpx cannot use without `socksio`.

    httpx prefers ``all_proxy`` over ``http_proxy``/``https_proxy``, so a SOCKS
    entry there makes every LLM call fail with "Using SOCKS proxy, but the
    'socksio' package is not installed" even when a working HTTP proxy is set.
    The HTTP(S) proxy variables are left untouched.
    """

    socks = [
        name
        for name in ("ALL_PROXY", "all_proxy")
        if str(os.environ.get(name, "")).lower().startswith("socks")
    ]
    if not socks:
        return
    try:
        import socksio  # noqa: F401
    except ImportError:
        for name in socks:
            os.environ.pop(name, None)
        print(
            "note: ignoring the SOCKS ALL_PROXY (httpx has no 'socksio' installed); "
            "HTTP(S)_PROXY is still honoured. Install httpx[socks] to use it.",
            file=sys.stderr,
        )


def _cuda_available() -> bool:
    """Cheap GPU probe so ``--gpu auto`` never asks LibCity for a missing device."""

    if os.environ.get("AGENTCITY_DEVICE", "").startswith("mps"):
        probe_python = os.environ.get("AGENTCITY_LIBCITY_PYTHON", sys.executable)
        try:
            probe = subprocess.run([probe_python, "-c", "import torch; raise SystemExit(0 if torch.backends.mps.is_available() else 1)"], timeout=15, capture_output=True)
            return probe.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False
    executable = shutil.which("nvidia-smi")
    if not executable:
        return False
    try:
        probe = subprocess.run(
            [executable, "-L"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0 and "GPU" in (probe.stdout or "")


def _pending_tasks(loop) -> list:
    """Names of the asyncio tasks still alive on ``loop`` (best effort)."""

    try:
        return sorted(task.get_name() for task in asyncio.all_tasks(loop) if not task.done())
    except Exception:  # noqa: BLE001 - diagnostics must never break the exit path
        return []


def _live_threads() -> list:
    """Names of the non-daemon threads that would keep the interpreter alive."""

    return sorted(
        thread.name
        for thread in threading.enumerate()
        if not thread.daemon and thread is not threading.main_thread()
    )


def _arm_exit_watchdog(exit_code: int, *, loop=None) -> None:
    """Force the process to exit once the run summary has been printed.

    openjiuwen's team/session runtime leaves asyncio tasks and default-executor
    work behind, and ``asyncio.run`` waits for them with no timeout, so a
    finished run can idle instead of exiting (observed live: the CLI printed its
    result and then stayed alive ~10 min). The work is already persisted at this
    point, so the watchdog bounds the wait; the normal path reaches the
    ``os._exit`` in ``__main__`` long before the deadline fires, and the
    diagnostics it prints name whatever was still running.
    """

    if os.environ.get("AGENTCITY_NO_EXIT_WATCHDOG"):
        return
    # The watchdog's contract is "the result is already readable", so make that
    # true even when stdout is block-buffered (a redirected pipe or file).
    sys.stdout.flush()
    try:
        grace = float(os.environ.get("AGENTCITY_TEARDOWN_GRACE_S", "30"))
    except ValueError:
        grace = 30.0
    grace = max(1.0, grace)

    def _bail() -> None:
        time.sleep(grace)
        print(
            f"note: teardown still busy {grace:.0f}s after the run summary was printed; "
            f"forcing exit {exit_code} (pending tasks: {_pending_tasks(loop) or 'none'}; "
            f"non-daemon threads: {_live_threads() or 'none'})",
            file=sys.stderr,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)

    threading.Thread(target=_bail, name="agentcity-exit-watchdog", daemon=True).start()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run the AgentCity spatio-temporal model adaptation pipeline for one task.",
    )
    parser.add_argument(
        "--task",
        help="Path to the JSON task specification (see tasks/ for examples).",
    )
    parser.add_argument("--max-rounds", type=int, help="Override the closed-loop round budget.")
    parser.add_argument(
        "--efficiency",
        choices=list(EFFICIENCY_CHOICES),
        help="Override the efficiency preference declared in the task specification.",
    )
    parser.add_argument("--metric", help="Override target.metric (e.g. masked_mae).")
    parser.add_argument("--threshold", type=float, help="Override target.threshold.")
    parser.add_argument(
        "--maximize",
        dest="maximize",
        action="store_true",
        default=None,
        help="Treat the target metric as 'higher is better'.",
    )
    parser.add_argument("--max-epoch", type=int, help="Override the per-run epoch budget.")
    parser.add_argument("--batch-size", type=int, help="Override the training batch size.")
    parser.add_argument(
        "--gpu",
        choices=GPU_CHOICES,
        default="auto",
        help="GPU policy; 'auto' enables the GPU only when one is actually present.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    parser.add_argument("--repo", default="./Bigscity-LibCity", help="LibCity checkout to adapt into.")
    parser.add_argument(
        "--libcity-python",
        help=(
            "Interpreter that can import LibCity + torch for the training subprocesses; "
            "sets AGENTCITY_LIBCITY_PYTHON (default: that env var, else `python`)."
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Do not verify that the LibCity interpreter can import LibCity before adapting.",
    )
    parser.add_argument(
        "--summary-json",
        help="Also write the run summary (the returned dict) to this path.",
    )
    parser.add_argument(
        "--list-tasks", action="store_true", help="List the task specifications in tasks/ and exit."
    )
    return parser


def _apply_overrides(spec: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI overrides on top of the task specification."""

    target = dict(spec.get("target") or {})
    if args.metric:
        target["metric"] = args.metric
    if args.threshold is not None:
        target["threshold"] = args.threshold
    if args.maximize is not None:
        target["maximize"] = bool(args.maximize)
    spec["target"] = target

    train = dict(spec.get("train") or {})
    if args.max_epoch is not None:
        train["max_epoch"] = args.max_epoch
    if args.batch_size is not None:
        train["batch_size"] = args.batch_size
    if args.gpu == "off":
        train["gpu"] = False
    elif args.gpu == "on":
        train["gpu"] = True
    elif not _cuda_available():
        train["gpu"] = False
    spec["train"] = train

    if args.max_rounds is not None:
        spec["max_rounds"] = max(1, min(int(args.max_rounds), 10))
    if args.efficiency:
        spec["efficiency"] = args.efficiency
    return spec


def _list_tasks(paths: AppPaths) -> int:
    task_dir = paths.root / "tasks"
    specs = sorted(task_dir.glob("*.json")) if task_dir.exists() else []
    if not specs:
        print(f"no task specifications found in {task_dir}")
        return 1
    for spec_path in specs:
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"{spec_path.name}: invalid JSON ({exc})")
            continue
        target = spec.get("target") or {}
        print(
            f"{spec_path.name}: task_id={spec.get('task_id')} dataset={spec.get('dataset')} "
            f"task={spec.get('task')} target={target.get('metric')}"
            f"{'≥' if target.get('maximize') else '≤'}{target.get('threshold')}"
        )
    return 0


def _resolve_repo(repo: Path, root: Path) -> Path:
    """Resolve ``--repo`` to an absolute path.

    Delegated agents run in isolated workspaces, so the checkout path is handed
    to prompts and subprocesses as an absolute path rather than a cwd-relative one.
    """

    repo = repo.expanduser()
    return repo if repo.is_absolute() else (root / repo).resolve()


def _preflight(paths: AppPaths) -> bool:
    """Fail fast when the LibCity interpreter cannot actually import LibCity."""

    from agents.adaptation.utils import libcity_python

    interpreter = libcity_python()
    try:
        probe = subprocess.run(
            [interpreter, "-c", "import libcity, torch; print('ok')"],
            cwd=str(paths.repo_path),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"LibCity interpreter {interpreter!r} is unusable: {exc}", file=sys.stderr)
        return False
    if probe.returncode == 0:
        return True
    print(
        f"LibCity interpreter {interpreter!r} cannot import libcity/torch:\n"
        f"{(probe.stderr or probe.stdout).strip()[-800:]}\n"
        "Set AGENTCITY_LIBCITY_PYTHON (or pass --libcity-python) to an interpreter that has "
        "LibCity's dependencies installed, e.g.\n"
        "  python main.py --task tasks/metr_la_speed_smoke.json "
        "--libcity-python /path/to/env/bin/python",
        file=sys.stderr,
    )
    return False


async def _run(args: argparse.Namespace, paths: AppPaths) -> int:
    spec_path = Path(args.task).expanduser()
    if not spec_path.exists():
        print(f"task specification not found: {spec_path}", file=sys.stderr)
        return 2

    spec = load_task_spec(spec_path)
    spec["repo_path"] = str(paths.repo_path)
    spec = _apply_overrides(spec, args)

    # Keep this run's active-task pointer private (it is published to the tool
    # server through the environment before the harness starts). The default path
    # in data/runs is shared repository state: a concurrent CLI run or a smoke
    # probe that cleared it made this run's tool calls unresolvable, and the runs
    # were then filed under the model name. Parallel runs already need this
    # override, so make it the default.
    os.environ.setdefault(
        "AGENTCITY_ACTIVE_TASK_FILE", str(paths.run_dir / f"active_task.{os.getpid()}.json")
    )

    logger = setup_logger(paths.log_file, level=getattr(args, "log_level", "INFO"))
    if not spec["train"].get("gpu"):
        logger.info("GPU disabled: LibCity will run on CPU (device policy: %s)", args.gpu)

    orchestrator = AgentOrchestrator(paths=paths, logger=logger)
    result = await orchestrator.run_adaptation(spec)

    passed = bool(result.get("passed"))
    print(
        json.dumps(
            {
                "task_id": result.get("task_id"),
                "passed": passed,
                "rounds": len(result.get("rounds") or []),
                "verdict": result.get("verdict"),
                "summary_path": result.get("summary_path"),
                "artifact_dir": result.get("artifact_dir"),
                "context_path": result.get("context_path"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    # Everything this run produces is on disk now, so bound the interpreter
    # teardown: openjiuwen leaves tasks/executor work behind and asyncio.run()
    # waits for them forever (observed live: ~10 min of idle after the summary).
    _arm_exit_watchdog(0 if passed else 1, loop=asyncio.get_running_loop())
    return 0 if passed else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _normalise_proxy_env()
    paths = AppPaths()
    paths.repo_path = _resolve_repo(Path(args.repo), paths.root)
    paths.ensure()

    if args.list_tasks:
        return _list_tasks(paths)
    if not args.task:
        parser.error("--task is required (or use --list-tasks)")
    if args.libcity_python:
        os.environ["AGENTCITY_LIBCITY_PYTHON"] = args.libcity_python
    # Persist the resolved interpreter: the MCP tool server is spawned by the
    # harness and may not inherit our environment, so the tool layer reads this
    # file as a fallback before falling back to `python` on PATH.
    from agents.adaptation.utils import libcity_python, record_libcity_python

    record_libcity_python(libcity_python())
    if not args.skip_preflight and not _preflight(paths):
        return 2

    return asyncio.run(_run(args, paths))


if __name__ == "__main__":
    # openjiuwen teardown leaks non-daemon threads (aiosqlite connections),
    # which would hang the interpreter at exit; force-exit after cleanup.
    import traceback

    exit_code = 0
    try:
        exit_code = main()
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        traceback.print_exc()
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
