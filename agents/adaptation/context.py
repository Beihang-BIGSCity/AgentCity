"""Shared execution context passed between adaptation stages and rounds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .utils import sanitize_slug


class ExecutionContext:
    """JSON-backed execution context shared by every stage of one task.

    Layout of ``data/adaptation/<task_id>/context.json``::

        {
          "task_id": "...",
          "round": 1,
          "dataset_profile": {...},
          "plan": {...},
          "composition": {...},
          "runs": [...],
          "tuning": {...},
          "verdict": {...},
          "history": [...]
        }
    """

    def __init__(self, task_id: str, *, base_dir: Path) -> None:
        self.task_id = task_id
        self.slug = sanitize_slug(task_id)
        self.dir = base_dir / self.slug
        self.path = self.dir / "context.json"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = self._load()

    @classmethod
    def load(cls, task_id: str, *, base_dir: Path) -> "ExecutionContext":
        return cls(task_id, base_dir=base_dir)

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    payload.setdefault("task_id", self.task_id)
                    payload.setdefault("history", [])
                    return payload
            except json.JSONDecodeError:
                pass
        return {"task_id": self.task_id, "round": 0, "history": []}

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def update(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.save()

    def append(self, key: str, value: Any) -> None:
        bucket = self.data.setdefault(key, [])
        if not isinstance(bucket, list):
            bucket = [bucket]
            self.data[key] = bucket
        bucket.append(value)
        self.save()

    def start_round(self) -> int:
        """Increment and persist the round counter."""

        self.data["round"] = int(self.data.get("round") or 0) + 1
        self.save()
        return int(self.data["round"])

    def record_round(self, payload: Mapping[str, Any]) -> None:
        """Archive a finished round so the next planning round can learn from it."""

        self.append("history", dict(payload))

    def feedback_note(self) -> str:
        """Render the previous round result for the next task-planning round."""

        history = self.data.get("history") or []
        if not history:
            return ""
        last = history[-1]
        lines: List[str] = []
        verdict = last.get("verdict") or {}
        lines.append(f"- round {last.get('round')} verdict: {json.dumps(verdict, ensure_ascii=False)}")
        composition = last.get("composition") or {}
        if composition:
            lines.append(
                f"- round {last.get('round')} composition: "
                f"{json.dumps(composition.get('plan') or composition, ensure_ascii=False)[:1200]}"
            )
        tuning = last.get("tuning") or {}
        if tuning:
            lines.append(f"- round {last.get('round')} tuning: {json.dumps(tuning, ensure_ascii=False)[:1200]}")
        failure = last.get("failure") or last.get("error")
        if failure:
            lines.append(f"- round {last.get('round')} failure: {str(failure)[:1500]}")
        runs = last.get("runs") or []
        for run in runs[-2:]:
            metrics = run.get("metrics") or {}
            lines.append(
                f"- round {last.get('round')} run {run.get('model_name')} on {run.get('dataset')}: "
                f"status={run.get('status')} metrics={json.dumps(metrics, ensure_ascii=False)}"
            )
        return "\n".join(lines)


__all__ = ["ExecutionContext"]
