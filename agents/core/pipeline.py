"""Multi-stage workflow runner backed by the pi execution backend."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Union

from .pi_backend import (
    PiBackendConfig,
    PiStageSession,
    map_tool_names,
    resolve_model,
)
from .storage import StageStorage
from .types import AgentContext, MultiAgentStage, StageDefinition, WorkflowDefinition

StageCallback = Callable[
    [Union[StageDefinition, MultiAgentStage], str, AgentContext],
    Awaitable[None] | None,
]


class AgentPipeline:
    """Runs the multi-stage workflow and persists transcripts."""

    def __init__(
        self,
        *,
        backend: PiBackendConfig,
        workflows: List[WorkflowDefinition],
        storage: StageStorage,
        logger,
        context: AgentContext,
        stage_callback: Optional[StageCallback] = None,
    ) -> None:
        self.backend = backend
        self.workflows = workflows
        self.storage = storage
        self.logger = logger
        self.context = context
        self.stage_callback = stage_callback

    async def run(self) -> None:
        for workflow in self.workflows:
            self.logger.info("=== Workflow: %s ===", workflow.name)
            for stage in workflow.stages:
                if isinstance(stage, MultiAgentStage):
                    await self._run_multi_agent_stage(stage, workflow.name)
                else:
                    await self._run_stage(stage, workflow.name)
            self.logger.info("=== Completed workflow: %s ===", workflow.name)

    async def _run_stage(self, stage: StageDefinition, workflow_name: str) -> None:
        self.logger.info("Starting stage: %s (%s)", stage.title, workflow_name)
        if stage.runner:
            prompt = (
                stage.prompt_builder(self.context)
                if stage.prompt_builder
                else stage.description
            )
            summary = await self._run_custom_stage(stage)
            self._finalize_stage(
                workflow_name=workflow_name,
                stage=stage,
                prompt=prompt,
                summary=summary,
                stage_messages=[],
                stats={},
            )
            if summary:
                self.context.stage_notes[stage.key] = summary
            await self._handle_stage_callback(stage, summary)
            self.logger.info("Completed stage: %s", stage.title)
            return

        if not stage.prompt_builder:
            raise ValueError(f"Stage {stage.key} requires a prompt builder when no runner is set.")

        prompt = stage.prompt_builder(self.context)
        async with PiStageSession(
            self.backend,
            model=resolve_model(None, self.backend.model_tiers),
            logger=self.logger,
        ) as session:
            result = await session.run(prompt)

        self._finalize_stage(
            workflow_name=workflow_name,
            stage=stage,
            prompt=prompt,
            summary=result.summary,
            stage_messages=result.messages,
            stats=result.stats,
        )
        if result.summary:
            self.context.stage_notes[stage.key] = result.summary
        await self._handle_stage_callback(stage, result.summary)
        self.logger.info("Completed stage: %s", stage.title)

    async def _run_multi_agent_stage(
        self, stage: MultiAgentStage, workflow_name: str
    ) -> None:
        """Execute a lead-agent stage; subagents run via the ``task`` tool."""
        self.logger.info(
            "Starting multi-agent stage: %s (%s)", stage.title, workflow_name
        )

        lead_prompt = stage.lead_agent_prompt_builder(self.context)

        if callable(stage.agents):
            agent_definitions = stage.agents(self.context)
        else:
            agent_definitions = stage.agents

        subagents_file = self._write_subagent_defs(stage.key, agent_definitions)

        async with PiStageSession(
            self.backend,
            model=resolve_model(stage.lead_model, self.backend.model_tiers),
            tools=["task"],  # Lead agent only delegates
            system_prompt=lead_prompt,
            extra_env={"AGENTCITY_SUBAGENTS_FILE": str(subagents_file)},
            logger=self.logger,
        ) as session:
            result = await session.run(lead_prompt)

        self._finalize_stage(
            workflow_name=workflow_name,
            stage=stage,
            prompt=lead_prompt,
            summary=result.summary,
            stage_messages=result.messages,
            stats=result.stats,
        )
        if result.summary:
            self.context.stage_notes[stage.key] = result.summary
        await self._handle_multi_agent_stage_callback(stage, result.summary)
        self.logger.info("Completed multi-agent stage: %s", stage.title)

    def _write_subagent_defs(self, stage_key: str, agent_definitions) -> Path:
        """Serialize subagent definitions for the extension's ``task`` tool."""
        defs = {}
        for name, agent_def in agent_definitions.items():
            defs[name] = {
                "description": agent_def.description,
                "tools": map_tool_names(agent_def.tools),
                "prompt": agent_def.prompt,
                "model": resolve_model(agent_def.model, self.backend.model_tiers),
            }
        out_dir = Path(self.backend.project_root) / "data" / "runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f".subagents_{stage_key}.json"
        path.write_text(json.dumps(defs, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _finalize_stage(
        self,
        *,
        workflow_name: str,
        stage,
        prompt: str,
        summary: str,
        stage_messages: List[dict],
        stats: Dict[str, float],
    ) -> None:
        self._log_stage_stats(stage.title, stats)
        self.storage.append_stage(
            workflow=workflow_name,
            stage_key=stage.key,
            title=stage.title,
            prompt=prompt,
            summary=summary,
            messages=stage_messages,
            stats=stats,
        )

    async def _run_custom_stage(self, stage: StageDefinition) -> str:
        assert stage.runner is not None
        self.logger.info("Running custom stage: %s", stage.title)
        result = stage.runner(self.context, stage)
        if inspect.isawaitable(result):
            result = await result
        summary = (result or "").strip()
        if summary:
            self.logger.info("Custom stage summary: %s", summary)
        else:
            self.logger.info("Custom stage completed without summary output.")
        return summary

    async def _handle_stage_callback(self, stage: StageDefinition, summary: str) -> None:
        if not self.stage_callback:
            return
        result = self.stage_callback(stage, summary, self.context)
        if inspect.isawaitable(result):
            await result

    async def _handle_multi_agent_stage_callback(
        self, stage: MultiAgentStage, summary: str
    ) -> None:
        if not self.stage_callback:
            return
        result = self.stage_callback(stage, summary, self.context)
        if inspect.isawaitable(result):
            await result

    def _log_stage_stats(self, stage_title: str, stats: Dict[str, float]) -> None:
        parts = []
        if "duration_seconds" in stats:
            parts.append(f"runtime={stats['duration_seconds']:.1f}s")
        if stats.get("input_tokens"):
            parts.append(f"input={int(stats['input_tokens'])} tokens")
        if stats.get("output_tokens"):
            parts.append(f"output={int(stats['output_tokens'])} tokens")
        if stats.get("total_cost_usd"):
            parts.append(f"cost=${stats['total_cost_usd']:.4f}")
        if parts:
            self.logger.info("Stage [%s] stats: %s", stage_title, ", ".join(parts))


__all__ = ["AgentPipeline"]
