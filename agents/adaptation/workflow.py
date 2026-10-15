"""The single adaptation workflow: planning -> execution -> tuning.

The closed self-correction loop lives in the orchestrator: it re-runs this
workflow with the previous round's verdict as feedback until the target
threshold is met or ``max_rounds`` is exhausted.
"""

from __future__ import annotations

from agents.core.types import MultiAgentStage, StageDefinition, WorkflowDefinition

from .prompts.lead import build_execution_lead_prompt, build_tuning_lead_prompt
from .prompts.task_planner import build_task_planning_prompt
from .registry import build_execution_agents, build_tuning_agents


def get_adaptation_workflow() -> WorkflowDefinition:
    """Return the layered multi-agent adaptation workflow."""

    return WorkflowDefinition(
        name="adaptation",
        description=(
            "Model adaptation against a spatio-temporal model library: the decision layer plans, the execution "
            "layer integrates data / composes modules / runs the model, and the tuning layer searches "
            "hyperparameters, debugs performance and evaluates the effect."
        ),
        stages=[
            StageDefinition(
                key="task_planning",
                title="Task Planning",
                description=(
                    "Decompose the adaptation task, define the success criteria and derive the structural direction "
                    "from the scenario characteristics."
                ),
                prompt_builder=build_task_planning_prompt,
                model="strong",
            ),
            MultiAgentStage(
                key="execution",
                title="Execution Layer",
                description=(
                    "Data integration, atomic module composition and run verification, led by the decision layer "
                    "with intermediate artifact validation."
                ),
                lead_agent_prompt_builder=build_execution_lead_prompt,
                agents=build_execution_agents,
                lead_model="strong",
            ),
            MultiAgentStage(
                key="tuning",
                title="Tuning Layer",
                description=(
                    "Hyperparameter search, performance debugging and effect evaluation feeding the "
                    "experience back into the knowledge base and skills."
                ),
                lead_agent_prompt_builder=build_tuning_lead_prompt,
                agents=build_tuning_agents,
                lead_model="strong",
            ),
        ],
    )


__all__ = ["get_adaptation_workflow"]
