from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class StageStorage:
    """Persists stage-by-stage transcripts for the front-end to render."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.data: Dict[str, List[Dict[str, Any]]] = self._load_existing()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist()

    def append_stage(
        self,
        *,
        workflow: str,
        stage_key: str,
        title: str,
        prompt: str,
        summary: str,
        messages: List[Dict[str, Any]],
    ) -> None:
        entry = {
            "workflow": workflow,
            "key": stage_key,
            "title": title,
            "prompt": prompt,
            "summary": summary,
            "messages": messages,
        }
        self.data["stages"].append(entry)
        self._persist()

    def reset(self) -> None:
        """Clear existing stage records (used when starting a fresh run)."""

        self.data = {"stages": []}
        self._persist()

    def _load_existing(self) -> Dict[str, List[Dict[str, Any]]]:
        if not self.output_path.exists():
            return {"stages": []}
        try:
            payload = json.loads(self.output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"stages": []}
        if not isinstance(payload, dict) or "stages" not in payload:
            return {"stages": []}
        return payload

    def _persist(self) -> None:
        self.output_path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8"
        )


__all__ = ["StageStorage"]
