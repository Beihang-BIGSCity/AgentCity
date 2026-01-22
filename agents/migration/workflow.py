from __future__ import annotations

from agents.core.types import StageDefinition, WorkflowDefinition

from .prompts import (
    build_migration_prompt,
    build_validation_prompt,
)


def get_migration_workflow() -> WorkflowDefinition:
    """Return the workflow definition that handles model migration."""

    stages = [
        StageDefinition(
            key="model_migration",
            title="Repository Migration",
            description="Clone repositories and port models into LibCity.",
            prompt_builder=build_migration_prompt,
        ),
        StageDefinition(
            key="verification",
            title="Verification",
            description="Run training/evaluation flows and capture logs.",
            prompt_builder=build_validation_prompt,
        )
    ]
    return WorkflowDefinition(
        name="migration",
        description="Clone external repos, port models, and verify metrics.",
        stages=stages,
    )


__all__ = ["get_migration_workflow"]
