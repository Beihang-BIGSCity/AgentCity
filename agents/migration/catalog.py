from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping

from .utils import build_paper_signature, merge_article_metadata, sanitize_signature


class MigrationCatalog:
    """Stores high-level migration summaries for the dashboard."""

    def __init__(
        self, *, catalog_path: Path, documentation_dir: Path, root: Path
    ) -> None:
        self.catalog_path = catalog_path
        self.documentation_dir = documentation_dir
        self.root = root
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self.documentation_dir.mkdir(parents=True, exist_ok=True)
        if not self.catalog_path.exists():
            self.catalog_path.write_text("[]\n", encoding="utf-8")

    def record_summary(
        self, paper: Mapping[str, Any] | None, summary_text: str
    ) -> Dict[str, Any] | None:
        """Persist the latest migration summary for the provided paper."""

        summary = (summary_text or "").strip()
        if not summary:
            return None
        metadata = self._build_metadata(paper)
        summary_file = self._write_summary(metadata["slug"], summary)
        entry = {
            "paper_id": metadata["paper_id"],
            "title": metadata["title"],
            "conference": metadata["conference"],
            "model_name": metadata["model_name"],
            "repo_url": metadata["repo_url"],
            "pdf_path": metadata["pdf_path"],
            "summary_path": self._relative_url(summary_file),
            "summary_file": str(summary_file),
            "summary_excerpt": self._excerpt(summary),
            "last_updated": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        self._upsert_entry(entry)
        return entry

    def _build_metadata(self, paper: Mapping[str, Any] | None) -> Dict[str, Any]:
        base: MutableMapping[str, Any] = {
            "title": "Untitled",
            "conference": "N/A",
            "model_name": "",
            "repo_url": "",
            "pdf_path": "",
        }
        merge_article_metadata(base, paper or {})
        signature = build_paper_signature(base)
        base["paper_id"] = signature
        base["slug"] = sanitize_signature(signature)
        return dict(base)

    def _write_summary(self, slug: str, summary: str) -> Path:
        filename = f"{slug}_migration_summary.md"
        target = self.documentation_dir / filename
        if summary.endswith("\n"):
            payload = summary
        else:
            payload = f"{summary}\n"
        target.write_text(payload, encoding="utf-8")
        return target

    def _relative_url(self, path: Path) -> str:
        try:
            relative = path.resolve().relative_to(self.root)
        except ValueError:
            relative = path.resolve()
        return f"/{relative.as_posix()}"

    def _excerpt(self, summary: str, limit: int = 2000) -> str:
        if len(summary) <= limit:
            return summary
        return f"{summary[:limit]}..."

    def _load_entries(self) -> List[Dict[str, Any]]:
        try:
            data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        return []

    def _upsert_entry(self, entry: Dict[str, Any]) -> None:
        entries = [
            existing
            for existing in self._load_entries()
            if existing.get("paper_id") != entry["paper_id"]
        ]
        entries.append(entry)
        entries.sort(key=lambda item: item.get("last_updated", ""), reverse=True)
        self.catalog_path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
        )


__all__ = ["MigrationCatalog"]
