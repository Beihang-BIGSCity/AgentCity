from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping

from .utils import build_paper_signature, merge_article_metadata, sanitize_signature


class MigrationResultStore:
    """Persists migration outputs per paper and exposes aggregated summaries."""

    def __init__(self, *, root: Path, base_dir: Path, index_path: Path) -> None:
        self.root = root
        self.base_dir = base_dir
        self.index_path = index_path
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self.index_path.write_text("[]\n", encoding="utf-8")
        self._maybe_migrate_legacy_index()

    def record_run(self, paper: Mapping[str, Any] | None, run: Dict[str, Any]) -> Dict[str, Any]:
        """Append a single dataset run for the given paper."""

        metadata = self._build_metadata(paper)
        signature = metadata["paper_id"]
        slug = sanitize_signature(signature)
        paper_dir = self.base_dir / slug
        paper_dir.mkdir(parents=True, exist_ok=True)

        runs_file = paper_dir / "runs.json"
        metadata_file = paper_dir / "metadata.json"

        runs = self._load_json(runs_file, default=[])
        prepared_run = self._prepare_run_entry(run, paper_dir, signature)
        runs.append(prepared_run)
        runs_file.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")

        metadata["artifact_dir"] = self._relative_url(paper_dir)
        metadata_file.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        self._refresh_index()
        return prepared_run

    def _build_metadata(self, paper: Mapping[str, Any] | None) -> Dict[str, Any]:
        base: MutableMapping[str, Any] = {
            "title": "Untitled",
            "conference": "N/A",
            "repo_url": "",
            "pdf_path": "",
            "model_name": "",
        }
        merge_article_metadata(base, paper or {})
        signature = build_paper_signature(base)
        base["paper_id"] = signature
        return dict(base)

    def _prepare_run_entry(
        self, run: Mapping[str, Any], paper_dir: Path, signature: str
    ) -> Dict[str, Any]:
        payload = dict(run)
        payload.setdefault(
            "timestamp", datetime.utcnow().isoformat(timespec="seconds") + "Z"
        )
        payload.setdefault("paper_title", signature.split("__", 1)[0])

        log_dir = paper_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout = payload.get("stdout")
        stderr = payload.get("stderr")
        label = self._make_log_prefix(payload)
        if stdout:
            stdout_path = log_dir / f"{label}_stdout.log"
            stdout_path.write_text(stdout, encoding="utf-8", errors="ignore")
            payload["stdout_path"] = self._relative_url(stdout_path)
        if stderr:
            stderr_path = log_dir / f"{label}_stderr.log"
            stderr_path.write_text(stderr, encoding="utf-8", errors="ignore")
            payload["stderr_path"] = self._relative_url(stderr_path)
        return payload

    def _make_log_prefix(self, run: Mapping[str, Any]) -> str:
        timestamp = str(run.get("timestamp") or "")
        dataset = str(run.get("dataset") or "dataset")
        model = str(run.get("model_name") or "model")
        slug = f"{timestamp}_{dataset}_{model}".replace(" ", "_")
        return sanitize_signature(slug) or "run"

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
            metadata_file = folder / "metadata.json"
            runs_file = folder / "runs.json"
            metadata = self._load_json(metadata_file, default=None)
            if not metadata:
                continue
            runs = self._load_json(runs_file, default=[])
            metadata["runs"] = runs
            metadata.setdefault("artifact_dir", self._relative_url(folder))
            entries.append(metadata)
        entries.sort(
            key=lambda item: (item.get("runs") or [])[-1].get("timestamp")
            if item.get("runs")
            else "",
            reverse=True,
        )
        self.index_path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _maybe_migrate_legacy_index(self) -> None:
        payload = self._load_json(self.index_path, default=[])
        if not payload or not isinstance(payload, list):
            return
        if payload and "paper_id" in payload[0]:
            return
        grouped: Dict[str, Dict[str, Any]] = {}
        for entry in payload:
            title = entry.get("paper_title") or "Untitled"
            conference = entry.get("conference") or "N/A"
            paper = {"title": title, "conference": conference}
            signature = build_paper_signature(paper)
            bucket = grouped.setdefault(
                signature,
                {"paper": paper, "runs": []},
            )
            bucket["runs"].append(entry)
        for _, bundle in grouped.items():
            paper = bundle["paper"]
            runs = bundle["runs"]
            paper_dir = self.base_dir / sanitize_signature(build_paper_signature(paper))
            paper_dir.mkdir(parents=True, exist_ok=True)
            (paper_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "paper_id": build_paper_signature(paper),
                        "title": paper.get("title"),
                        "conference": paper.get("conference"),
                        "artifact_dir": self._relative_url(paper_dir),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (paper_dir / "runs.json").write_text(
                json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        self._refresh_index()

    @staticmethod
    def _load_json(path: Path, default):
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return default


__all__ = ["MigrationResultStore"]
