from __future__ import annotations

from agents.core.types import StageDefinition, WorkflowDefinition

from .prompts import build_literature_prompt


def get_literature_workflow() -> WorkflowDefinition:
    """Return the workflow definition dedicated to literature search."""

    literature_stage = StageDefinition(
        key="literature_scan",
        title="Literature Sweep",
        description="Find and archive all relevant conference papers.",
        prompt_builder=build_literature_prompt,
    )
    return WorkflowDefinition(
        name="literature",
        description="Search, download, and catalog new papers.",
        stages=[literature_stage],
    )


__all__ = ["get_literature_workflow"]
