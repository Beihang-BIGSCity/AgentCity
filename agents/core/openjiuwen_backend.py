"""openJiuwen execution backend for AgentCity pipelines.

Wraps openjiuwen 0.1.18 (TeamAgentSpec + native DeepAgent) so the pipeline
layer programs against ``AgentDefinition``/``MultiAgentStage`` only:

- profile loading (harness_profiles/deepagent.yaml): logical model roles and
  logical tool names are resolved here;
- team construction: lead prompt -> LeaderSpec, AgentDefinition dict ->
  predefined TeamMemberSpec entries, business tools -> stdio MCP server;
- event translation: team stream chunks -> the stage-message schema consumed
  by StageStorage / server.py ({"role", "type", "content"|"tool_name", ...}).

Runner lifecycle is managed by the orchestrator via ``ensure_runner_started``
/ ``shutdown_runner``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .types import AgentContext, MultiAgentStage  # noqa: F401  (typing aid)

PROFILE_ENV = "AGENTCITY_HARNESS_PROFILE"
DEFAULT_PROFILE_PATH = Path(__file__).parent / "harness_profiles" / "deepagent.yaml"
LOCAL_PROFILE_PATH = DEFAULT_PROFILE_PATH.with_suffix(".local.yaml")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class HarnessProfile:
    """Logical role/tool resolution for one harness backend."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "HarnessProfile":
        default_path = LOCAL_PROFILE_PATH if LOCAL_PROFILE_PATH.exists() else DEFAULT_PROFILE_PATH
        profile_path = Path(os.environ.get(PROFILE_ENV, "") or (path or default_path))
        with open(profile_path, "r", encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    # -- provider ------------------------------------------------------
    @property
    def provider_type(self) -> str:
        return self._data["provider"]["type"]

    @property
    def api_key(self) -> str:
        provider = self._data["provider"]
        env_name = provider.get("api_key_env")
        if env_name and os.environ.get(env_name):
            return os.environ[env_name]
        return provider.get("api_key", "")

    @property
    def api_base(self) -> str:
        provider = self._data["provider"]
        env_name = provider.get("api_base_env")
        if env_name and os.environ.get(env_name):
            return os.environ[env_name]
        return provider.get("api_base", "")

    # -- logical names ---------------------------------------------------
    def resolve_model(self, role: str) -> str:
        models = self._data.get("models", {})
        return models.get(role, role)

    def resolve_tool(self, logical_name: str) -> str:
        return self._data.get("tools", {}).get(logical_name, logical_name)

    def resolve_tools(self, logical_names: List[str]) -> List[str]:
        return [self.resolve_tool(name) for name in logical_names]


# ---------------------------------------------------------------------------
# Runner lifecycle (process-wide, reference counted)
# ---------------------------------------------------------------------------

_runner_refcount = 0


async def ensure_runner_started() -> None:
    global _runner_refcount
    if _runner_refcount == 0:
        from openjiuwen.core.runner.runner import Runner

        await Runner.start()
    _runner_refcount += 1


async def shutdown_runner() -> None:
    global _runner_refcount
    if _runner_refcount == 0:
        return
    _runner_refcount -= 1
    if _runner_refcount == 0:
        from openjiuwen.core.runner.runner import Runner

        try:
            await Runner.stop()
        except Exception:  # teardown errors (e.g. closed sqlite conns) are non-fatal
            logging.getLogger(__name__).warning("Runner.stop() raised during teardown", exc_info=True)


# ---------------------------------------------------------------------------
# Team construction
# ---------------------------------------------------------------------------


def _team_model_config(profile: HarnessProfile, role: str) -> "TeamModelConfig":
    from openjiuwen.core.foundation.llm.schema.config import (
        ModelClientConfig,
        ModelRequestConfig,
    )
    from openjiuwen.harness.schema.deep_agent_spec import TeamModelConfig

    return TeamModelConfig(
        model_client_config=ModelClientConfig(
            client_provider=profile.provider_type,
            api_key=profile.api_key,
            api_base=profile.api_base,
        ),
        model_request_config=ModelRequestConfig(model=profile.resolve_model(role)),
    )


def _business_mcp_config() -> "McpServerConfig":
    from openjiuwen.core.foundation.tool import McpServerConfig

    return McpServerConfig(
        server_name="agentcity",
        client_type="stdio",
        server_path="",
        params={
            "command": sys.executable,
            "args": ["-m", "agents.tools.server"],
            "cwd": str(PROJECT_ROOT),
        },
    )


def _model_pool(profile: HarnessProfile, roles: List[str]) -> list:
    from openjiuwen.agent_teams.models.pool import ModelPoolEntry

    seen: Dict[str, None] = {}
    for role in roles:
        seen.setdefault(profile.resolve_model(role))
    return [
        ModelPoolEntry(
            model_name=model_name,
            api_key=profile.api_key,
            api_base_url=profile.api_base,
            api_provider=profile.provider_type,
        )
        for model_name in seen
    ]


_MEMBER_TOOL_GUIDANCE = (
    "\n\n## Your designated tools\n"
    "You are expected to work with the following tools only: {tools}. "
    "Do not use other tools even if they appear available."
)


def build_team_spec(
    *,
    stage: MultiAgentStage,
    agent_definitions: Dict[str, "AgentDefinition"],
    lead_prompt: str,
    profile: HarnessProfile,
):
    """Build a TeamAgentSpec for one multi-agent stage."""
    from openjiuwen.agent_teams.schema.blueprint import (
        LeaderSpec,
        StorageSpec,
        TeamAgentSpec,
    )
    from openjiuwen.agent_teams.schema.team import TeamMemberSpec
    from openjiuwen.harness.schema.deep_agent_spec import DeepAgentSpec

    business_mcp = _business_mcp_config()

    member_specs: List[TeamMemberSpec] = []
    for name, agent_def in agent_definitions.items():
        resolved_tools = profile.resolve_tools(agent_def.tools)
        member_specs.append(
            TeamMemberSpec(
                member_name=name,
                display_name=name,
                desc=agent_def.description,
                prompt=agent_def.prompt
                + _MEMBER_TOOL_GUIDANCE.format(tools=", ".join(resolved_tools)),
                model_name=profile.resolve_model(agent_def.model),
            )
        )

    roles = [stage.lead_model] + [d.model for d in agent_definitions.values()]

    db_dir = PROJECT_ROOT / "data" / "runs" / "agent_teams"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / f"{stage.key}.db"
    # Fresh team state per stage run: stale rows (members/tasks) from a
    # previous run would otherwise be reattached.
    db_path.unlink(missing_ok=True)

    return TeamAgentSpec(
        agents={
            "leader": DeepAgentSpec(
                model=_team_model_config(profile, stage.lead_model),
                max_iterations=25,
            ),
            "teammate": DeepAgentSpec(
                model=_team_model_config(profile, "balanced"),
                mcps=[business_mcp],
                max_iterations=25,
            ),
        },
        team_name=f"agentcity_{stage.key}",
        leader=LeaderSpec(
            member_name="lead",
            display_name="Lead",
            desc=f"Lead coordinator of stage {stage.title}",
            prompt=lead_prompt,
        ),
        predefined_members=member_specs,
        # persistent lifecycle: the framework auto-closes the leader stream
        # (team.completed marker) once every task is terminal, every member
        # is settled and the leader is idle. Temporary teams rely on the
        # leader calling clean_team, which proved unreliable (LLM may skip
        # the instruction and then no event ever drives another leader turn).
        lifecycle="persistent",
        spawn_mode="inprocess",
        dispatch_mode="autonomous",
        model_pool=_model_pool(profile, roles),
        model_pool_strategy="by_model_name",
        # ``memory`` storage is broken in 0.1.18 (DDL lands on the write
        # engine while readers open a separate :memory: database), so use a
        # file-backed sqlite db per stage.
        storage=StorageSpec(type="sqlite", params={"connection_string": str(db_path)}),
        language="en",
    )


# ---------------------------------------------------------------------------
# Stream chunk -> stage message translation
#
# Verified against a live 0.1.18 run (Runner.run_agent_team_streaming).
# Chunks are OutputSchema-like objects: ``type`` / ``index`` / ``payload`` /
# ``source_member`` / ``role``. Observed types:
#   - "llm_output": token-level deltas, payload={"content": str, ...}
#   - "tool_call": payload={"name", "arguments"(json str), "tool_call_id"}
#   - "controller_output": lifecycle payload, e.g. TASK_COMPLETION
#   - "answer": final result, payload={"output": str}
#   - "llm_usage" / "context.usage" / team markers ("team.*"): skipped
# ---------------------------------------------------------------------------


def _safe_payload(payload: Any) -> Any:
    try:
        json.dumps(payload)
        return payload
    except TypeError:
        return str(payload)


_SKIP_TYPES = {"llm_usage", "context.usage", "runtime_ready", "round_aborted"}


class StreamTranslator:
    """Stateful translator aggregating token deltas into stage messages.

    Buffers ``llm_output`` text per ``source_member`` and flushes on member
    switch / tool call / final answer, producing the old SDK-era schema:
    ``{"role", "type": "text", "content"}`` and
    ``{"role", "type": "tool_use", "tool_name", "tool_input"}``.
    """

    def __init__(self, logger=None) -> None:
        self._logger = logger
        self._buffer: Dict[str, List[str]] = {}
        self._order: List[str] = []
        self._last_text: Dict[str, str] = {}

    def _log(self, msg: str, *args) -> None:
        if self._logger is not None:
            self._logger.info(msg, *args)

    @staticmethod
    def _field(chunk: Any, name: str) -> Any:
        if isinstance(chunk, dict):
            return chunk.get(name)
        return getattr(chunk, name, None)

    def _flush_member(self, member: str) -> List[dict]:
        parts = self._buffer.pop(member, [])
        if member in self._order:
            self._order.remove(member)
        text = "".join(parts).strip()
        if not text:
            return []
        self._log("%s: %s", member, text[:200])
        self._last_text[member] = text
        return [{"role": member, "type": "text", "content": text}]

    def flush_all(self) -> List[dict]:
        records: List[dict] = []
        for member in list(self._order):
            records.extend(self._flush_member(member))
        return records

    def feed(self, chunk: Any) -> List[dict]:
        records: List[dict] = []
        chunk_type = self._field(chunk, "type") or ""
        source = self._field(chunk, "source_member") or "assistant"
        source = str(source)
        payload = self._field(chunk, "payload")

        if not isinstance(chunk_type, str):
            chunk_type = str(chunk_type)
        if chunk_type.startswith("team.") or chunk_type in _SKIP_TYPES:
            return records
        # Team lifecycle markers ride the stream as type="message" chunks
        # whose payload carries event_type="team.*" (idle/completed/...).
        if isinstance(payload, dict):
            event_type = payload.get("event_type")
            if isinstance(event_type, str) and event_type.startswith("team."):
                return records

        if isinstance(payload, dict):
            if chunk_type == "llm_output":
                content = payload.get("content")
                if isinstance(content, str) and content:
                    if source not in self._buffer:
                        self._order.append(source)
                    self._buffer.setdefault(source, []).append(content)
                return records

            if chunk_type == "tool_call":
                # Flush pending text first so ordering reads naturally.
                records.extend(self.flush_all())
                name = payload.get("name") or "unknown"
                arguments = payload.get("arguments")
                self._log("tool_use %s: %s", source, name)
                records.append(
                    {
                        "role": source,
                        "type": "tool_use",
                        "tool_name": name,
                        "tool_input": _safe_payload(arguments),
                    }
                )
                return records

            if chunk_type in ("answer", "controller_output"):
                records.extend(self.flush_all())
                output = payload.get("output")
                if output is None:
                    # controller_output: data=[JsonDataFrame(type='json', data={'output': ...})]
                    for frame in payload.get("data") or []:
                        frame_data = frame.get("data") if isinstance(frame, dict) else self._field(frame, "data")
                        if isinstance(frame_data, dict) and frame_data.get("output"):
                            output = frame_data["output"]
                            break
                if isinstance(output, str) and output.strip():
                    text = output.strip()
                    # The same final text usually already arrived as buffered
                    # llm_output deltas; skip the duplicate.
                    if text == self._last_text.get(source):
                        return records
                    self._log("%s (final): %s", source, text[:200])
                    self._last_text[source] = text
                    records.append(
                        {"role": source, "type": "text", "content": text}
                    )
                return records

        return records

    # Backwards-compatible functional view.
    def translate(self, chunk: Any) -> List[dict]:
        return self.feed(chunk)

    @staticmethod
    def is_team_completed_marker(chunk: Any) -> bool:
        """True for the framework's team.completed marker (stream closes right after)."""
        payload = StreamTranslator._field(chunk, "payload")
        return (
            isinstance(payload, dict)
            and payload.get("event_type") == "team.completed"
        )


def translate_chunk(chunk: Any, logger=None) -> List[dict]:
    """Stateless single-chunk translation (no cross-chunk aggregation).

    Prefer StreamTranslator for real runs; this exists for unit probes.
    """
    return StreamTranslator(logger).feed(chunk)


# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------


async def run_team_stage(
    *,
    stage: MultiAgentStage,
    agent_definitions: Dict[str, "AgentDefinition"],
    lead_prompt: str,
    profile: HarnessProfile,
    logger,
    stage_timeout_s: int = 7200,
) -> Dict[str, Any]:
    """Run one multi-agent stage on an openJiuwen team.

    Returns {"summary": str, "messages": [stage-message dicts]}.

    Goes through ``Runner.run_agent_team_streaming`` (the supported entry
    point): the runtime manager activates the team (session creation, member
    spawn, session-scoped DB tables) and finalizes it afterwards, so we must
    NOT call ``spec.build()``/``team.stream()``/``stop_coordination`` by hand.
    """
    from openjiuwen.core.runner.runner import Runner

    spec = build_team_spec(
        stage=stage,
        agent_definitions=agent_definitions,
        lead_prompt=lead_prompt,
        profile=profile,
    )
    translator = StreamTranslator(logger)
    messages: List[dict] = []
    stream = Runner.run_agent_team_streaming(spec, {"query": lead_prompt})
    try:
        # Temporary teams only close the stream when the leader dissolves the
        # team via ``clean_team`` (the lead prompts instruct it to do so).
        # The team.completed marker (persistent-team auto-pause path) is a
        # valid terminal signal too. A leader that finishes without either
        # would hang the stage, so a per-stage timeout is the backstop;
        # ``aclose`` drives Runner's finally cleanup (runtime finalize,
        # session teardown) on every exit path.
        async with asyncio.timeout(stage_timeout_s):
            async for chunk in stream:
                messages.extend(translator.feed(chunk))
                if StreamTranslator.is_team_completed_marker(chunk):
                    break
    except TimeoutError:
        logger.warning(
            "team stage %s exceeded %ss; force-closing stream",
            stage.key, stage_timeout_s,
        )
    finally:
        await stream.aclose()
    messages.extend(translator.flush_all())

    text_sections = [r["content"] for r in messages if r["type"] == "text"]
    summary = "\n".join(text_sections).strip()
    return {"summary": summary, "messages": messages}


async def run_single_prompt(
    *,
    prompt: str,
    profile: HarnessProfile,
    logger,
    model_role: str = "balanced",
) -> Dict[str, Any]:
    """Run a single-agent stage as one DeepAgent (no team)."""
    from openjiuwen.harness import create_deep_agent

    agent = create_deep_agent(
        _team_model_config(profile, model_role).build(),
        system_prompt=prompt,
        mcps=[_business_mcp_config()],
        workspace=str(PROJECT_ROOT),
        language="en",
    )
    result = await agent.invoke({"query": prompt})
    output = ""
    if isinstance(result, dict):
        output = str(result.get("output") or "")
    elif result is not None:
        output = str(result)
    summary = output.strip()
    messages: List[dict] = (
        [{"role": "assistant", "type": "text", "content": summary}] if summary else []
    )
    return {"summary": summary, "messages": messages}


__all__ = [
    "HarnessProfile",
    "build_team_spec",
    "ensure_runner_started",
    "run_single_prompt",
    "run_team_stage",
    "StreamTranslator",
    "shutdown_runner",
    "translate_chunk",
]
