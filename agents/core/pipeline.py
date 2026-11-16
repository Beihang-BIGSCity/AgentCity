from __future__ import annotations

import inspect
import os
import json
from pathlib import Path
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
        if os.environ.get("AGENTCITY_RESUME_PARTIAL") == "1" and stage.key == "task_planning":
            existing = Path(self.context.task_dir) / "plan.json"
            if existing.exists():
                json.loads(existing.read_text())
                self.context.stage_notes[stage.key] = "Resumed existing plan.json; original artifact preserved."
                self.logger.info("Resume: preserved existing plan.json")
                return
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

        if os.environ.get("AGENTCITY_SERIAL_ROLES") == "1":
            messages, summaries = [], []
            required = {"data-integrator": "data_integration.json", "model-composer": "composition.json",
                        "run-tester": "run_report.json", "hyper-searcher": "tuning.json",
                        "effect-evaluator": "verdict.json"}
            for name, definition in agent_definitions.items():
                self.logger.info("Serial role start: %s", name)
                artifact = required.get(name)
                if os.environ.get("AGENTCITY_RESUME_PARTIAL") == "1" and artifact:
                    existing = Path(self.context.task_dir) / artifact
                    if existing.exists():
                        json.loads(existing.read_text())
                        summaries.append(name + ": resumed existing " + artifact)
                        self.logger.info("Resume: preserved %s", artifact)
                        continue
                instruction = (definition.prompt + "\n\nExecution mode: you are invoked directly and sequentially. "
                    "Do not create teams, send team messages, claim tasks, wait for colleagues, or delegate. "
                    "Complete your own artifact with the file tools and then finish. "
                    "All predecessor roles have already returned. Use only validation evidence for decisions. "
                    "Do not inspect unrelated repository files or change infrastructure. "
                    "Never run additional training merely to verify a successful run. "
                    "For perf-debugger: if run_report and tuning both succeeded, append a short audit to tuning.json "
                    "without training. On failures, record diagnosis and stop; extra training would violate this experiment budget.")
                if name == "effect-evaluator" and os.environ.get("AGENTCITY_FREEZE_EXPERIENCE") == "1":
                    instruction += " Frozen-memory control: do not modify knowledge/ or skills/ by any means. record_experience may return skipped; this is expected."
                result_one = await run_single_prompt(prompt=instruction, profile=self.profile,
                                                    logger=self.logger, model_role=definition.model)
                artifact = required.get(name)
                if artifact:
                    path = Path(self.context.task_dir) / artifact
                    if not path.exists():
                        repair = instruction + f"\nYour required output {path} is missing. Write it from existing evidence now. Do not repeat training or invent results."
                        result_one = await run_single_prompt(prompt=repair, profile=self.profile,
                                                            logger=self.logger, model_role=definition.model)
                    json.loads(path.read_text())  # fail explicitly instead of deadlocking on a missing contract
                summaries.append(name + ": " + result_one["summary"])
                messages.extend(result_one["messages"])
                self.logger.info("Serial role complete: %s", name)
            result = {"summary": "\n".join(summaries), "messages": messages}
        else:
            result = await run_team_stage(
                stage=stage, agent_definitions=agent_definitions, lead_prompt=lead_prompt,
                profile=self.profile, logger=self.logger,
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
