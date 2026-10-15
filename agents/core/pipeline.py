from __future__ import annotations

import inspect
from typing import Awaitable, Callable, List, Optional, Union

from .openjiuwen_backend import HarnessProfile, run_single_prompt, run_team_stage
from .storage import StageStorage
from .types import AgentContext, MultiAgentStage, StageDefinition, WorkflowDefinition

StageCallback = Callable[
    [Union[StageDefinition, MultiAgentStage], str, AgentContext],
    Awaitable[None] | None,
]


class AgentPipeline:
    """Runs the multi-stage workflow on the openJiuwen backend and persists transcripts."""

    def __init__(
        self,
        *,
        profile: HarnessProfile,
        workflows: List[WorkflowDefinition],
        storage: StageStorage,
        logger,
        context: AgentContext,
        stage_callback: Optional[StageCallback] = None,
    ) -> None:
        self.profile = profile
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
            stage_messages: List[dict] = []
            self.storage.append_stage(
                workflow=workflow_name,
                stage_key=stage.key,
                title=stage.title,
                prompt=prompt,
                summary=summary,
                messages=stage_messages,
            )
            if summary:
                self.context.stage_notes[stage.key] = summary
            await self._handle_stage_callback(stage, summary)
            self.logger.info("Completed stage: %s", stage.title)
            return

        if not stage.prompt_builder:
            raise ValueError(f"Stage {stage.key} requires a prompt builder when no runner is set.")

        prompt = stage.prompt_builder(self.context)
        result = await run_single_prompt(
            prompt=prompt,
            profile=self.profile,
            logger=self.logger,
            model_role=stage.model,
        )
        summary = result["summary"]
        if summary:
            self.context.stage_notes[stage.key] = summary
        self.storage.append_stage(
            workflow=workflow_name,
            stage_key=stage.key,
            title=stage.title,
            prompt=prompt,
            summary=summary,
            messages=result["messages"],
        )
        await self._handle_stage_callback(stage, summary)
        self.logger.info("Completed stage: %s", stage.title)

    async def _run_multi_agent_stage(
        self, stage: MultiAgentStage, workflow_name: str
    ) -> None:
        """Execute a multi-agent stage as an openJiuwen team (leader + members)."""
        self.logger.info(
            "Starting multi-agent stage: %s (%s)", stage.title, workflow_name
        )

        lead_prompt = stage.lead_agent_prompt_builder(self.context)

        if callable(stage.agents):
            agent_definitions = stage.agents(self.context)
        else:
            agent_definitions = stage.agents

        result = await run_team_stage(
            stage=stage,
            agent_definitions=agent_definitions,
            lead_prompt=lead_prompt,
            profile=self.profile,
            logger=self.logger,
        )
        summary = result["summary"]
        if summary:
            self.context.stage_notes[stage.key] = summary

        self.storage.append_stage(
            workflow=workflow_name,
            stage_key=stage.key,
            title=stage.title,
            prompt=lead_prompt,
            summary=summary,
            messages=result["messages"],
        )

        await self._handle_multi_agent_stage_callback(stage, summary)
        self.logger.info("Completed multi-agent stage: %s", stage.title)

    async def _handle_multi_agent_stage_callback(
        self, stage: MultiAgentStage, summary: str
    ) -> None:
        """Handle callback for multi-agent stages."""
        if not self.stage_callback:
            return
        result = self.stage_callback(stage, summary, self.context)
        if inspect.isawaitable(result):
            await result

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


__all__ = ["AgentPipeline"]
