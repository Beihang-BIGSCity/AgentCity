"""Pi (pi-mono) execution backend driven through pi-py-sdk.

One ``PiStageSession`` wraps one ``pi --mode rpc`` subprocess. The pipeline
creates a session per stage so that per-stage stats and tool allowlists stay
isolated. Custom tools (catalog_article, test_migration, ...) and the ``task``
sub-agent tool are provided by the TypeScript extension in
``pi_backend/extensions/agentcity-tools.ts``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pi_py_sdk import PiAgent, PiConfig

# Claude Code tool names (used in AgentDefinition lists and prompts) mapped to
# the tool names exposed by pi / the agentcity extension.
TOOL_NAME_MAP: Dict[str, Any] = {
    "Read": "read",
    "Write": "write",
    "Edit": "edit",
    "Bash": "bash",
    "Grep": "grep",
    "Glob": ["find", "ls"],  # pi has no glob tool; find + ls cover discovery
    "WebSearch": "web_search",
    "WebFetch": "web_fetch",
    "Task": "task",
}

DEFAULT_MODEL_TIERS: Dict[str, str] = {
    "haiku": "deepseek/deepseek-flash",
    "sonnet": "deepseek/deepseek-flash",
    "opus": "deepseek/deepseek-flash",
}


def map_tool_names(tools: List[str]) -> List[str]:
    """Translate Claude-style tool names to pi tool names, preserving order."""
    mapped: List[str] = []
    for name in tools:
        target = TOOL_NAME_MAP.get(name, name)
        for item in target if isinstance(target, list) else [target]:
            if item not in mapped:
                mapped.append(item)
    return mapped


def resolve_model(tier_or_id: str | None, tiers: Dict[str, str]) -> str:
    """Resolve a model tier alias (haiku/sonnet/opus) to a pi model id."""
    if not tier_or_id:
        return tiers["sonnet"]
    return tiers.get(tier_or_id, tier_or_id)


@dataclass
class PiBackendConfig:
    """Static configuration for launching pi RPC subprocesses."""

    project_root: Path
    pi_bin: str = "pi_backend/bin/pi"
    extension_path: str = "pi_backend/extensions/agentcity-tools.ts"
    pi_agent_dir: str = "pi_backend/agent"
    model_tiers: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODEL_TIERS))
    prompt_timeout: float = 7200.0

    def resolve_paths(self) -> None:
        root = self.project_root
        if not Path(self.pi_bin).is_absolute():
            self.pi_bin = str(root / self.pi_bin)
        if not Path(self.extension_path).is_absolute():
            self.extension_path = str(root / self.extension_path)
        if not Path(self.pi_agent_dir).is_absolute():
            self.pi_agent_dir = str(root / self.pi_agent_dir)

    def base_env(self) -> Dict[str, str]:
        return {
            "PI_CODING_AGENT_DIR": self.pi_agent_dir,
            "AGENTCITY_ROOT": str(self.project_root),
            "AGENTCITY_PI_BIN": self.pi_bin,
            "AGENTCITY_PYTHON": os.environ.get("AGENTCITY_PYTHON", sys.executable),
        }


@dataclass
class StageRunResult:
    summary: str
    messages: List[dict]
    stats: Dict[str, float]


class PiStageSession:
    """Async context manager running one stage inside a pi RPC subprocess."""

    def __init__(
        self,
        config: PiBackendConfig,
        *,
        model: str | None = None,
        tools: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        extra_env: Optional[Dict[str, str]] = None,
        logger=None,
    ) -> None:
        self._config = config
        self._model = model or resolve_model(None, config.model_tiers)
        self._tools = tools
        self._system_prompt = system_prompt
        self._extra_env = extra_env or {}
        self._logger = logger
        self._agent: Optional[PiAgent] = None

    async def __aenter__(self) -> "PiStageSession":
        extra_args = [
            "--no-extensions",
            "-e",
            self._config.extension_path,
            "--no-skills",
            "--no-context-files",
            "--no-prompt-templates",
        ]
        if self._tools is not None:
            extra_args += ["--tools", ",".join(self._tools)]
        if self._system_prompt:
            extra_args += ["--system-prompt", self._system_prompt]

        config = PiConfig(
            bin=self._config.pi_bin,
            model=self._model,
            cwd=str(self._config.project_root),
            no_session=True,
            env={**self._config.base_env(), **self._extra_env},
            extra_args=extra_args,
        )
        self._agent = PiAgent(config=config)
        await self._agent.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._agent is not None:
            await self._agent.stop()
            self._agent = None

    async def abort(self) -> None:
        if self._agent is not None:
            await self._agent.abort()

    async def run(self, prompt: str) -> StageRunResult:
        """Send the stage prompt and collect messages until the run settles."""
        assert self._agent is not None, "session not started"
        start = time.time()
        stage_messages: List[dict] = []
        text_sections: List[str] = []

        async for event in self._agent.prompt_stream(
            prompt, timeout=self._config.prompt_timeout
        ):
            if event.type != "message_end":
                continue
            message = event.message
            if not isinstance(message, dict):
                continue
            records = self._serialize_message(message)
            stage_messages.extend(records)
            text_sections.extend(
                record["content"] for record in records if record["type"] == "text"
            )

        stats: Dict[str, float] = {}
        try:
            stats = self._extract_stats(await self._agent.get_session_stats())
        except Exception as exc:  # stats are best-effort
            if self._logger:
                self._logger.warning("Failed to collect pi session stats: %s", exc)
        stats["duration_seconds"] = round(time.time() - start, 2)

        return StageRunResult(
            summary="\n".join(text_sections).strip(),
            messages=stage_messages,
            stats=stats,
        )

    def _serialize_message(self, message: dict) -> List[dict]:
        """Map a pi AgentMessage to the stage-log record format."""
        role = message.get("role", "assistant")
        if role == "toolResult":
            text_parts = [
                block.get("text", "")
                for block in message.get("content") or []
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return [
                {
                    "role": "tool",
                    "type": "tool_result",
                    "tool_name": message.get("toolName"),
                    "content": "\n".join(part for part in text_parts if part)[:2000],
                    "is_error": bool(message.get("isError")),
                }
            ]
        if role != "assistant":
            return []
        records: List[dict] = []
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = (block.get("text") or "").strip()
                if text:
                    if self._logger:
                        self._logger.info("Assistant: %s", text)
                    records.append({"role": role, "type": "text", "content": text})
            elif block_type == "toolCall":
                payload = self._safe_payload(block.get("arguments"))
                if self._logger:
                    self._logger.info("Tool %s called with %s", block.get("name"), payload)
                records.append(
                    {
                        "role": role,
                        "type": "tool_use",
                        "tool_name": block.get("name"),
                        "tool_input": payload,
                    }
                )
        return records

    @staticmethod
    def _extract_stats(raw: Dict[str, Any]) -> Dict[str, float]:
        tokens = raw.get("tokens") or {}
        stats: Dict[str, float] = {
            "input_tokens": tokens.get("input", 0),
            "output_tokens": tokens.get("output", 0),
            "cache_read_input_tokens": tokens.get("cacheRead", 0),
            "cache_creation_input_tokens": tokens.get("cacheWrite", 0),
            "total_cost_usd": raw.get("cost", 0) or 0,
            "num_turns": raw.get("assistantMessages", 0),
        }
        return stats

    @staticmethod
    def _safe_payload(payload):
        try:
            json.dumps(payload)
            return payload
        except TypeError:
            return str(payload)


__all__ = [
    "DEFAULT_MODEL_TIERS",
    "PiBackendConfig",
    "PiStageSession",
    "StageRunResult",
    "TOOL_NAME_MAP",
    "map_tool_names",
    "resolve_model",
]
