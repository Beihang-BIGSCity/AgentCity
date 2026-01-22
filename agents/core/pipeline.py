from __future__ import annotations

import inspect
import json
from typing import Awaitable, Callable, List, Optional, Union

from claude_agent_sdk import (
    AgentDefinition as SDKAgentDefinition,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

from .storage import StageStorage
from .types import AgentContext, MultiAgentStage, StageDefinition, WorkflowDefinition

StageCallback = Callable[
    [Union[StageDefinition, MultiAgentStage], str, AgentContext],
    Awaitable[None] | None,
]


class AgentPipeline:
    """Runs the multi-stage Claude workflow and persists transcripts."""

    def __init__(
        self,
        *,
        client,
        workflows: List[WorkflowDefinition],
        storage: StageStorage,
        logger,
        context: AgentContext,
        stage_callback: Optional[StageCallback] = None,
    ) -> None:
        self.client = client
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
        await self.client.query(prompt)
        stage_messages = []
        text_sections: List[str] = []

        async for message in self.client.receive_response():
            serialized_messages = self._serialize_message(message)
            stage_messages.extend(serialized_messages)
            text_sections.extend(
                [chunk["content"] for chunk in serialized_messages if chunk["type"] == "text"]
            )

        summary = "\n".join(text_sections).strip()
        if summary:
            self.context.stage_notes[stage.key] = summary
        self.storage.append_stage(
            workflow=workflow_name,
            stage_key=stage.key,
            title=stage.title,
            prompt=prompt,
            summary=summary,
            messages=stage_messages,
        )
        await self._handle_stage_callback(stage, summary)
        self.logger.info("Completed stage: %s", stage.title)

    async def _run_multi_agent_stage(
        self, stage: MultiAgentStage, workflow_name: str
    ) -> None:
        """Execute a multi-agent stage with lead agent + subagents."""
        self.logger.info(
            "Starting multi-agent stage: %s (%s)", stage.title, workflow_name
        )

        # Build lead agent prompt with context
        lead_prompt = stage.lead_agent_prompt_builder(self.context)

        # Resolve agents: can be a dict or a callable that returns a dict
        if callable(stage.agents):
            agent_definitions = stage.agents(self.context)
        else:
            agent_definitions = stage.agents

        # Convert our AgentDefinition to SDK's AgentDefinition
        sdk_agents = {}
        for name, agent_def in agent_definitions.items():
            sdk_agents[name] = SDKAgentDefinition(
                description=agent_def.description,
                tools=agent_def.tools,
                prompt=agent_def.prompt,
                model=agent_def.model,
            )

        # Build options for this stage with lead agent configuration
        options = ClaudeAgentOptions(
            permission_mode="bypassPermissions",
            setting_sources=["project"],  # Load pdf skill, ralph-wiggum
            system_prompt=lead_prompt,
            allowed_tools=["Task"],  # Lead agent only delegates
            agents=sdk_agents,
            model=stage.lead_model,
        )

        stage_messages: List[dict] = []
        text_sections: List[str] = []

        # Create dedicated client for this multi-agent stage
        async with ClaudeSDKClient(options=options) as client:
            # Initial query to start the lead agent
            await client.query(prompt=lead_prompt)

            # Collect responses from lead agent and subagents
            async for message in client.receive_response():
                serialized_messages = self._serialize_message(message)
                stage_messages.extend(serialized_messages)
                text_sections.extend(
                    [
                        chunk["content"]
                        for chunk in serialized_messages
                        if chunk["type"] == "text"
                    ]
                )

        summary = "\n".join(text_sections).strip()
        if summary:
            self.context.stage_notes[stage.key] = summary

        self.storage.append_stage(
            workflow=workflow_name,
            stage_key=stage.key,
            title=stage.title,
            prompt=lead_prompt,
            summary=summary,
            messages=stage_messages,
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

    def _serialize_message(self, message) -> List[dict]:
        records = []
        role = "assistant"
        if isinstance(message, ResultMessage):
            role = "result"

        for block in getattr(message, "content", []) or []:
            if isinstance(block, TextBlock):
                text = block.text.strip()
                if text:
                    self.logger.info("%s: %s", role.capitalize(), text)
                    records.append({"role": role, "type": "text", "content": text})
            elif isinstance(block, ToolUseBlock):
                payload = self._safe_payload(block.input)
                self.logger.info("Tool %s called with %s", block.name, payload)
                records.append(
                    {
                        "role": role,
                        "type": "tool_use",
                        "tool_name": block.name,
                        "tool_input": payload,
                    }
                )
        return records

    @staticmethod
    def _safe_payload(payload):
        try:
            json.dumps(payload)
            return payload
        except TypeError:
            return str(payload)


__all__ = ["AgentPipeline"]
