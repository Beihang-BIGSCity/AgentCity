"""Tuning workflow using multi-agent coordination.

This workflow uses a lead agent to coordinate specialized subagents for
hyperparameter planning, execution, and result analysis.
"""

from __future__ import annotations

from agents.core.types import MultiAgentStage, WorkflowDefinition
from agents.core.agent_registry import build_tuning_agents

from .prompts.lead_agent import build_tuning_lead_prompt


def get_tuning_workflow() -> WorkflowDefinition:
    """Return the multi-agent tuning workflow.

    The workflow consists of a single MultiAgentStage where a lead agent
    coordinates three specialized subagents:
    - tuning-planner: Designs hyperparameter search spaces
    - tuning-executor: Runs tuning trials
    - result-analyzer: Analyzes results and generates reports
    """

    return WorkflowDefinition(
        name="tuning",
        description=(
            "Multi-agent workflow for hyperparameter optimization. "
            "A lead agent coordinates planner, executor, and analyzer subagents "
            "to tune migrated models."
        ),
        stages=[
            MultiAgentStage(
                key="hyperparameter_tuning",
                title="Hyperparameter Tuning",
                description=(
                    "Coordinate hyperparameter search space design, trial execution, "
                    "and result analysis using specialized subagents."
                ),
                lead_agent_prompt_builder=build_tuning_lead_prompt,
                agents=build_tuning_agents,  # Will be called with context
                lead_model="sonnet",
            )
        ],
    )


__all__ = ["get_tuning_workflow"]
