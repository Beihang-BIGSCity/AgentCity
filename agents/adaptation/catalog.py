"""Persists the adaptation summary document for each finished task."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .utils import sanitize_slug


class AdaptationCatalog:
    """Writes ``documentation/<task_id>_adaptation_summary.md`` plus an index."""

    def __init__(self, *, index_path: Path, documentation_dir: Path, root: Path) -> None:
        self.index_path = index_path
        self.documentation_dir = documentation_dir
        self.root = root
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.documentation_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self.index_path.write_text("[]\n", encoding="utf-8")

    def record_summary(
        self,
        task_id: str,
        summary_text: str,
        *,
        task_spec: Mapping[str, Any] | None = None,
        verdict: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any] | None:
        """Persist the adaptation summary for one task and index it."""

        summary = (summary_text or "").strip()
        if not summary:
            return None
        slug = sanitize_slug(task_id)
        summary_file = self.documentation_dir / f"{slug}_adaptation_summary.md"
        summary_file.write_text(summary + ("" if summary.endswith("\n") else "\n"), encoding="utf-8")
        entry = {
            "task_id": task_id,
            "slug": slug,
            "dataset": (task_spec or {}).get("dataset"),
            "task": (task_spec or {}).get("task"),
            "model_name": (task_spec or {}).get("model_name"),
            "verdict": dict(verdict) if verdict else None,
            "summary_path": self._relative_url(summary_file),
            "summary_file": str(summary_file),
            "last_updated": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        self._upsert_entry(entry)
        return entry

    def _relative_url(self, path: Path) -> str:
        try:
            relative = path.resolve().relative_to(self.root)
        except ValueError:
            relative = path.resolve()
        return f"/{relative.as_posix()}"

    def _load_entries(self) -> List[Dict[str, Any]]:
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        return []

    def _upsert_entry(self, entry: Dict[str, Any]) -> None:
        entries = [
            existing for existing in self._load_entries() if existing.get("task_id") != entry["task_id"]
        ]
        entries.append(entry)
        entries.sort(key=lambda item: item.get("last_updated", ""), reverse=True)
        self.index_path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
        )


__all__ = ["AdaptationCatalog"]
