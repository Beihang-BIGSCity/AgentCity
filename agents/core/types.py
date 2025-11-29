from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


PromptBuilder = Callable[["AgentContext"], str]


@dataclass
class AgentContext:
    """Shared runtime information passed to every stage prompt."""

    repo_path: str
    benchmark_document: str
    article_dir: str
    article_catalog: str
    conferences: List[str]
    target_year: int
    stage_notes: Dict[str, str] = field(default_factory=dict)
    search_terms: List[str] = field(default_factory=list)
    selected_papers: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StageDefinition:
    key: str
    title: str
    description: str
    prompt_builder: PromptBuilder


@dataclass
class WorkflowDefinition:
    """Groups related stages (e.g., literature search vs. migration)."""

    name: str
    description: str
    stages: List[StageDefinition]


__all__ = ["AgentContext", "StageDefinition", "WorkflowDefinition", "PromptBuilder"]
