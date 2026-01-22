"""Literature search workflow using multi-agent coordination.

This workflow uses a lead agent to coordinate specialized subagents for
paper searching, evaluation, and analysis.
"""

from __future__ import annotations

from agents.core.types import MultiAgentStage, WorkflowDefinition
from agents.core.agent_registry import build_literature_agents

from .prompts.lead_agent import build_literature_lead_prompt


def get_literature_workflow() -> WorkflowDefinition:
    """Return the multi-agent literature search and analysis workflow.

    The workflow consists of a single MultiAgentStage where a lead agent
    coordinates three specialized subagents:
    - paper-searcher: Searches academic databases
    - paper-evaluator: Scores paper relevance
    - paper-analyzer: Downloads and analyzes PDFs
    """

    return WorkflowDefinition(
        name="literature_mas",
        description=(
            "Multi-agent workflow for academic paper discovery. "
            "A lead agent coordinates searcher, evaluator, and analyzer subagents "
            "to find, score, and catalog relevant papers."
        ),
        stages=[
            MultiAgentStage(
                key="literature_search",
                title="Literature Search & Analysis",
                description=(
                    "Coordinate paper search, relevance evaluation, and PDF analysis "
                    "using specialized subagents."
                ),
                lead_agent_prompt_builder=build_literature_lead_prompt,
                agents=build_literature_agents,  # Will be called with context
                lead_model="sonnet",
            )
        ],
    )


__all__ = ["get_literature_workflow"]
