"""Paths, task-spec loading and context construction for the pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from .types import AgentContext

DEFAULT_TARGET = {"metric": "masked_mae", "threshold": None, "maximize": False}
DEFAULT_TRAIN = {"max_epoch": 2, "batch_size": 16, "gpu": False}
DEFAULT_MAX_ROUNDS = 3
EFFICIENCY_CHOICES = ("lightweight", "balanced", "accuracy")


@dataclass
class AppPaths:
    """Centralizes important repository paths for the agents."""

    root: Path = Path(__file__).resolve().parents[2]
    # Absolute on purpose: team members run in their own isolated workspaces,
    # so a cwd-relative checkout path does not resolve for them.
    repo_path: Path = root / "Bigscity-LibCity"
    data_dir: Path = root / "data"
    run_dir: Path = data_dir / "runs"
    stage_log: Path = run_dir / "latest_stage_log.json"
    adaptation_dir: Path = data_dir / "adaptation"
    adaptation_index: Path = adaptation_dir / "index.json"
    modlib_dir: Path = data_dir / "modlib"
    knowledge_dir: Path = root / "knowledge"
    skills_dir: Path = root / "skills"
    documentation_dir: Path = root / "documentation"
    log_file: Path = root / "logs" / "adaptation.log"

    def ensure(self) -> None:
        for folder in (
            self.log_file.parent,
            self.data_dir,
            self.run_dir,
            self.adaptation_dir,
            self.modlib_dir,
            self.knowledge_dir,
            self.skills_dir,
            self.documentation_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)
        if not self.stage_log.exists():
            self.stage_log.write_text('{"stages": []}\n', encoding="utf-8")
        if not self.adaptation_index.exists():
            self.adaptation_index.write_text("[]\n", encoding="utf-8")

    def task_dir(self, task_id: str) -> Path:
        from agents.adaptation.utils import sanitize_slug

        return self.adaptation_dir / sanitize_slug(task_id)


def load_task_spec(path: Path) -> Dict[str, Any]:
    """Load and normalize an adaptation task specification."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("task specification must be a JSON object")

    missing = [key for key in ("task_id", "dataset", "task") if not payload.get(key)]
    if missing:
        raise ValueError(f"task specification is missing required keys: {', '.join(missing)}")

    target = dict(DEFAULT_TARGET)
    target.update(payload.get("target") or {})
    if target.get("threshold") is not None:
        target["threshold"] = float(target["threshold"])
    target["maximize"] = bool(target.get("maximize", False))

    train = dict(DEFAULT_TRAIN)
    train.update(payload.get("train") or {})
    train["max_epoch"] = int(train.get("max_epoch") or DEFAULT_TRAIN["max_epoch"])
    train["batch_size"] = int(train.get("batch_size") or DEFAULT_TRAIN["batch_size"])
    train["gpu"] = bool(train.get("gpu", False))
    extra_args = {
        str(key): value
        for key, value in dict(train.get("extra_args") or {}).items()
        if key not in {"gpu", "gpu_id"} and value is not None
    }
    train["extra_args"] = extra_args

    efficiency = str(payload.get("efficiency") or "balanced").strip().lower()
    if efficiency not in EFFICIENCY_CHOICES:
        efficiency = "balanced"

    max_rounds = int(payload.get("max_rounds") or DEFAULT_MAX_ROUNDS)
    max_rounds = max(1, min(max_rounds, 10))

    tuning_payload = dict(payload.get("tuning") or {})
    tuning: Dict[str, Any] = {
        "max_trials": _positive_int(tuning_payload.get("max_trials")),
        "max_evals": _positive_int(tuning_payload.get("max_evals")),
        "hyper_algo": tuning_payload.get("hyper_algo") or None,
        "search_space": dict(tuning_payload.get("search_space") or {}) or None,
    }

    scope = payload.get("candidate_scope") or {}
    normalized = {
        "task_id": str(payload["task_id"]),
        "dataset": str(payload["dataset"]),
        "task": str(payload["task"]),
        "model_name": payload.get("model_name") or None,
        "target": target,
        "efficiency": efficiency,
        "train": train,
        "max_rounds": max_rounds,
        "tuning": tuning,
        "candidate_scope": {
            "models": [str(item) for item in (scope.get("models") or [])],
            "categories": [str(item) for item in (scope.get("categories") or [])],
        },
        "notes": str(payload.get("notes") or ""),
    }
    return normalized


def _positive_int(value: Any) -> int | None:
    """Coerce a positive integer budget, returning None when unset."""

    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def build_context(
    paths: AppPaths,
    task_spec: Dict[str, Any],
    *,
    round_index: int = 1,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    feedback: str = "",
    dataset_profile: Dict[str, Any] | None = None,
    stage_notes: Dict[str, str] | None = None,
) -> AgentContext:
    """Build the agent context for one round of one adaptation task."""

    from agents.adaptation.utils import sanitize_slug

    task_id = str(task_spec.get("task_id") or "task")
    return AgentContext(
        repo_path=str(paths.repo_path),
        task_id=task_id,
        task_slug=sanitize_slug(task_id),
        task_dir=str(paths.task_dir(task_id)),
        task_spec=dict(task_spec),
        round_index=round_index,
        max_rounds=max_rounds,
        modlib_dir=str(paths.modlib_dir),
        knowledge_dir=str(paths.knowledge_dir),
        skills_dir=str(paths.skills_dir),
        documentation_dir=str(paths.documentation_dir),
        feedback=feedback,
        dataset_profile=dict(dataset_profile or {}),
        stage_notes=dict(stage_notes or {}),
    )


__all__ = [
    "AppPaths",
    "DEFAULT_MAX_ROUNDS",
    "DEFAULT_TARGET",
    "DEFAULT_TRAIN",
    "EFFICIENCY_CHOICES",
    "build_context",
    "load_task_spec",
]
