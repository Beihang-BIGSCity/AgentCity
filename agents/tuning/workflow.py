from __future__ import annotations

from agents.core.types import StageDefinition, WorkflowDefinition

from .prompts import build_tuning_prompt


def get_tuning_workflow() -> WorkflowDefinition:
    """Return the workflow definition dedicated to tuning migrated models."""

    stages = [
        StageDefinition(
            key="hyperparameter_tuning",
            title="Hyperparameter Tuning",
            description="Explore hyperparameters for migrated models using LibCity or the built-in grid search.",
            prompt_builder=build_tuning_prompt,
        )
    ]
    return WorkflowDefinition(
        name="tuning",
        description="Hyperparameter tuning workflow that follows migration.",
        stages=stages,
    )


__all__ = ["get_tuning_workflow"]
