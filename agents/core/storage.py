from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class StageStorage:
    """Persists stage-by-stage transcripts for the front-end to render."""

    _MAX_MESSAGES = 200
    _MAX_TEXT_LEN = 2000
    _MAX_TOOL_INPUT_LEN = 1000

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.data: Dict[str, List[Dict[str, Any]]] = self._load_existing()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._sanitize_data()
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
        trimmed_messages = self._trim_messages(messages)
        entry = {
            "workflow": workflow,
            "key": stage_key,
            "title": title,
            "prompt": self._truncate_text(prompt, self._MAX_TEXT_LEN),
            "summary": self._truncate_text(summary, self._MAX_TEXT_LEN),
            "messages": trimmed_messages,
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

    def _sanitize_data(self) -> None:
        """Trim stored payload to keep the stage log small."""

        stages = self.data.get("stages")
        if not isinstance(stages, list):
            self.data["stages"] = []
            return
        trimmed: List[Dict[str, Any]] = []
        for entry in stages:
            if not isinstance(entry, dict):
                continue
            messages = entry.get("messages") or []
            entry["messages"] = self._trim_messages(messages)
            entry["summary"] = self._truncate_text(entry.get("summary", ""), self._MAX_TEXT_LEN)
            entry["prompt"] = self._truncate_text(entry.get("prompt", ""), self._MAX_TEXT_LEN)
            trimmed.append(entry)
        self.data["stages"] = trimmed[-self._MAX_MESSAGES :]

    def _persist(self) -> None:
        self.output_path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def _trim_messages(cls, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        trimmed: List[Dict[str, Any]] = []
        if not isinstance(messages, list):
            return trimmed
        for message in messages[-cls._MAX_MESSAGES :]:
            if not isinstance(message, dict):
                continue
            entry = dict(message)
            if entry.get("type") == "text":
                entry["content"] = cls._truncate_text(entry.get("content", ""), cls._MAX_TEXT_LEN)
            elif entry.get("type") == "tool_use":
                payload = entry.get("tool_input")
                try:
                    serialized = json.dumps(payload, ensure_ascii=False)
                except TypeError:
                    serialized = str(payload)
                entry["tool_input"] = cls._truncate_text(serialized, cls._MAX_TOOL_INPUT_LEN)
            trimmed.append(entry)
        return trimmed

    @staticmethod
    def _truncate_text(value: Any, limit: int) -> str:
        text = str(value or "")
        if len(text) > limit:
            return f"{text[:limit]}...[truncated]"
        return text


__all__ = ["StageStorage"]
