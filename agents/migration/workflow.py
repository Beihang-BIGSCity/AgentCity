"""Migration workflow using multi-agent coordination.

This workflow uses a lead agent to coordinate specialized subagents for
repository cloning, model adaptation, configuration, and testing.
"""

from __future__ import annotations

from agents.core.types import MultiAgentStage, WorkflowDefinition
from agents.core.agent_registry import build_migration_agents

from .prompts.lead_agent import build_migration_lead_prompt


def get_migration_workflow() -> WorkflowDefinition:
    """Return the multi-agent migration workflow.

    The workflow consists of a single MultiAgentStage where a lead agent
    coordinates four specialized subagents:
    - repo-cloner: Clones and analyzes external repositories
    - model-adapter: Adapts PyTorch models to LibCity conventions
    - config-migrator: Creates and updates configuration files
    - migration-tester: Runs tests and diagnoses issues
    """

    return WorkflowDefinition(
        name="migration",
        description=(
            "Multi-agent workflow for porting external models to LibCity. "
            "A lead agent coordinates cloner, adapter, config, and tester subagents "
            "to migrate models from research repositories."
        ),
        stages=[
            MultiAgentStage(
                key="model_migration",
                title="Migration",
                description=(
                    "Coordinate repository cloning, model adaptation, configuration, "
                    "and testing using specialized subagents."
                ),
                lead_agent_prompt_builder=build_migration_lead_prompt,
                agents=build_migration_agents,  # Will be called with context
                lead_model="sonnet",
            )
        ],
    )


__all__ = ["get_migration_workflow"]
