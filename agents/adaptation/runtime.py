"""Tracks the adaptation task currently being executed.

The business tools (``agents.tools``) are exposed by a *separate* FastMCP
stdio server process, so an in-process registry alone is not enough: the
mounted server would see no active task and would silently misfile artifacts.
The active task is therefore mirrored to a small JSON file that any process can
read; ``AGENTCITY_ACTIVE_TASK_FILE`` overrides its location (used by the
validation script and by parallel runs).

The same snapshot is exported to ``AGENTCITY_ACTIVE_TASK_ID`` /
``AGENTCITY_ACTIVE_TASK_SPEC`` because the pointer file is shared repository
state: a concurrent smoke run or an unrelated probe that clears it would leave
the tool server unable to resolve the task at all (observed live: run records
were then filed under the model name). Children spawned after
``set_active_task`` - openjiuwen forks the tool server out of the CLI process -
inherit the snapshot, so the task still resolves (with a warning) instead.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

ACTIVE_TASK_ENV = "AGENTCITY_ACTIVE_TASK_FILE"
ACTIVE_TASK_ID_ENV = "AGENTCITY_ACTIVE_TASK_ID"
ACTIVE_TASK_SPEC_ENV = "AGENTCITY_ACTIVE_TASK_SPEC"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOKEN_KEY = "_active_task_token"

_ACTIVE_TASK: Optional[Dict[str, Any]] = None
_ACTIVE_TOKEN: Optional[str] = None


def task_file() -> Path:
    """Location of the cross-process active-task pointer."""

    override = os.environ.get(ACTIVE_TASK_ENV)
    if override:
        return Path(override).expanduser()
    return PROJECT_ROOT / "data" / "runs" / "active_task.json"


def _publish_env(task_spec: Dict[str, Any] | None) -> None:
    """Mirror the active task into the environment for children spawned later."""

    if task_spec is None:
        os.environ.pop(ACTIVE_TASK_ID_ENV, None)
        os.environ.pop(ACTIVE_TASK_SPEC_ENV, None)
        return
    task_id = str(task_spec.get("task_id") or "").strip()
    if task_id:
        os.environ[ACTIVE_TASK_ID_ENV] = task_id
    else:
        os.environ.pop(ACTIVE_TASK_ID_ENV, None)
    os.environ[ACTIVE_TASK_SPEC_ENV] = json.dumps(task_spec, ensure_ascii=False)


def set_active_task(task_spec: Dict[str, Any] | None) -> None:
    """Track which task spec is currently being adapted (in-process + on disk)."""

    global _ACTIVE_TASK, _ACTIVE_TOKEN
    path = task_file()
    if not task_spec:
        # Only remove a pointer this process wrote. A second process sharing the
        # default path used to delete the live run's pointer here, and the tool
        # server then filed that run's records under the model name.
        existing = _read_pointer(path) if path.exists() else None
        if existing is not None and existing.get(TOKEN_KEY) != _ACTIVE_TOKEN:
            print(
                f"warning: keeping active-task pointer {path}; it was published by another run "
                f"(task_id={existing.get('task_id')!r})",
                file=sys.stderr,
                flush=True,
            )
        else:
            path.unlink(missing_ok=True)
        _ACTIVE_TASK = None
        _ACTIVE_TOKEN = None
        _publish_env(None)
        return

    _ACTIVE_TOKEN = uuid.uuid4().hex
    _ACTIVE_TASK = dict(task_spec)
    _ACTIVE_TASK[TOKEN_KEY] = _ACTIVE_TOKEN
    _publish_env(_ACTIVE_TASK)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: a reader in the tool-server process must never observe a
    # half-written pointer, because a truncated file reads as "no active task".
    staged = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    staged.write_text(json.dumps(_ACTIVE_TASK, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(staged, path)


def _read_pointer(path: Path) -> Optional[Dict[str, Any]]:
    """Read the on-disk pointer, retrying once in case of a concurrent rewrite."""

    for attempt in range(2):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError):
            if attempt == 0:
                time.sleep(0.2)
                continue
            return None
        if isinstance(payload, dict) and str(payload.get("task_id") or "").strip():
            return payload
        return None
    return None


def _env_task() -> Optional[Dict[str, Any]]:
    """Recover the active task from the environment snapshot when the pointer is lost."""

    raw = os.environ.get(ACTIVE_TASK_SPEC_ENV)
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and str(payload.get("task_id") or "").strip():
            return payload
    task_id = os.environ.get(ACTIVE_TASK_ID_ENV, "").strip()
    return {"task_id": task_id} if task_id else None


def get_active_task() -> Optional[Dict[str, Any]]:
    """Return the adaptation task in flight, reading the pointer when needed."""

    if _ACTIVE_TASK is not None:
        return _ACTIVE_TASK
    path = task_file()
    if path.exists():
        payload = _read_pointer(path)
        if payload is not None:
            return payload
    payload = _env_task()
    if payload is not None:
        print(
            f"warning: active-task pointer {path} is unusable; falling back to the environment "
            f"snapshot (task_id={payload.get('task_id')!r})",
            file=sys.stderr,
            flush=True,
        )
        return payload
    return None


def clear_active_task() -> None:
    """Clear the active task pointer."""

    set_active_task(None)


__all__ = [
    "ACTIVE_TASK_ENV",
    "TOKEN_KEY",
    "ACTIVE_TASK_ID_ENV",
    "ACTIVE_TASK_SPEC_ENV",
    "clear_active_task",
    "get_active_task",
    "set_active_task",
    "task_file",
]
