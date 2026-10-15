"""Per-task run bookkeeping for the adaptation pipeline."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .utils import sanitize_slug


class AdaptationResultStore:
    """Persists run records per adaptation task and exposes an aggregate index."""

    def __init__(self, *, root: Path, base_dir: Path, index_path: Path) -> None:
        self.root = root
        self.base_dir = base_dir
        self.index_path = index_path
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self.index_path.write_text("[]\n", encoding="utf-8")

    def record_run(self, task_id: str, run: Dict[str, Any]) -> Dict[str, Any]:
        """Append a single training/tuning run for the given task."""

        slug = sanitize_slug(task_id)
        task_dir = self.base_dir / slug
        task_dir.mkdir(parents=True, exist_ok=True)

        runs_file = task_dir / "runs.json"
        runs = self._load_json(runs_file, default=[])
        prepared = self._prepare_run_entry(run, task_dir, slug)
        runs.append(prepared)
        runs_file.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
        self._refresh_index()
        return prepared

    def _prepare_run_entry(
        self, run: Mapping[str, Any], task_dir: Path, slug: str
    ) -> Dict[str, Any]:
        payload = dict(run)
        payload.setdefault("timestamp", datetime.utcnow().isoformat(timespec="seconds") + "Z")
        log_dir = task_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        label = self._make_log_prefix(payload)
        for stream in ("stdout", "stderr"):
            content = payload.get(stream)
            if content:
                log_path = log_dir / f"{label}_{stream}.log"
                log_path.write_text(content, encoding="utf-8", errors="ignore")
                payload[f"{stream}_path"] = self._relative_url(log_path)
        return payload

    def _make_log_prefix(self, run: Mapping[str, Any]) -> str:
        timestamp = str(run.get("timestamp") or "")
        dataset = str(run.get("dataset") or "dataset")
        model = str(run.get("model_name") or "model")
        return sanitize_slug(f"{timestamp}_{dataset}_{model}") or "run"

    def _relative_url(self, path: Path) -> str:
        try:
            rel = path.resolve().relative_to(self.root)
        except ValueError:
            rel = path.resolve()
        return f"/{rel.as_posix()}"

    def _refresh_index(self) -> None:
        entries: List[Dict[str, Any]] = []
        for folder in sorted(self.base_dir.glob("*")):
            if not folder.is_dir():
                continue
            runs = self._load_json(folder / "runs.json", default=[])
            if not runs:
                continue
            entries.append(
                {
                    "task_id": folder.name,
                    "artifact_dir": self._relative_url(folder),
                    "runs": runs,
                    "last_updated": runs[-1].get("timestamp"),
                }
            )
        entries.sort(key=lambda item: item.get("last_updated") or "", reverse=True)
        self.index_path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @staticmethod
    def _load_json(path: Path, default):
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return default


__all__ = ["AdaptationResultStore"]
