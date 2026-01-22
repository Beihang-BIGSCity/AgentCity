from __future__ import annotations

from dataclasses import dataclass
from typing import List

from agents.core.types import StageDefinition, WorkflowDefinition

from .analyze_agent import PaperAnalyzeAgent
from .search_agent import PaperSearchStageAgent


@dataclass
class LiteratureMASAgent:
    """Small descriptor for MAS agent assembly in the literature workflow."""

    name: str
    stage: StageDefinition


def build_literature_agents() -> List[LiteratureMASAgent]:
    """Return the MAS agent roster (paper search + paper analyze)."""

    search_agent = LiteratureMASAgent(
        name="paper_search_agent",
        stage=PaperSearchStageAgent().build_stage(),
    )
    analyze_agent = LiteratureMASAgent(
        name="paper_analyze_agent",
        stage=PaperAnalyzeAgent().build_stage(),
    )
    return [search_agent, analyze_agent]


def get_literature_workflow() -> WorkflowDefinition:
    """Return the MAS-based workflow chaining the search and analyze agents."""

    agents = build_literature_agents()
    return WorkflowDefinition(
        name="literature_mas",
        description="MAS workflow that invokes the Claude paper search agent then the Claude paper analyze agent.",
        stages=[agent.stage for agent in agents],
    )


__all__ = ["get_literature_workflow", "build_literature_agents"]
