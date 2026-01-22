from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Union


PromptBuilder = Callable[["AgentContext"], str]
StageRunResult = str | None
StageRunner = Callable[
    ["AgentContext", "StageDefinition"],
    Awaitable[StageRunResult] | StageRunResult,
]
# AgentBuilder is a callable that takes context and returns agent definitions
AgentBuilder = Callable[["AgentContext"], Dict[str, "AgentDefinition"]]


@dataclass
class AgentContext:
    """Shared runtime information passed to every stage prompt."""

    repo_path: str
    benchmark_document: str
    article_dir: str
    article_catalog: str
    migration_catalog: str
    conferences: List[str]
    target_year: int
    stage_notes: Dict[str, str] = field(default_factory=dict)
    search_terms: List[str] = field(default_factory=list)
    selected_papers: List[Dict[str, Any]] = field(default_factory=list)
    paper_candidates: List[Dict[str, Any]] = field(default_factory=list)
    search_year_mode: str = "all"
    search_year_label: str = "全部"
    search_year_value: int | None = None
    search_year_start: int | None = None  # 年份范围起始
    search_year_end: int | None = None    # 年份范围结束
    search_conference_filters: List[str] = field(default_factory=list)


@dataclass
class StageDefinition:
    key: str
    title: str
    description: str
    prompt_builder: PromptBuilder | None = None
    runner: StageRunner | None = None


@dataclass
class AgentDefinition:
    """Defines a specialized subagent with constrained tools and prompts.

    Used by MultiAgentStage to configure subagents that the lead agent
    can delegate tasks to via the Task tool.
    """
    name: str
    description: str
    tools: List[str]
    prompt: str
    model: str = "sonnet"


@dataclass
class MultiAgentStage:
    """A stage that uses lead-agent + subagent coordination.

    The lead agent only has access to the Task tool and delegates
    work to specialized subagents defined in the agents dict.

    The agents field can be either:
    - A dict of AgentDefinition objects (static configuration)
    - A callable that takes AgentContext and returns the dict (dynamic configuration)
    """
    key: str
    title: str
    description: str
    lead_agent_prompt_builder: PromptBuilder
    agents: Union[Dict[str, AgentDefinition], AgentBuilder]
    lead_model: str = "sonnet"


@dataclass
class WorkflowDefinition:
    """Groups related stages (e.g., literature search vs. migration)."""

    name: str
    description: str
    stages: List[StageDefinition | MultiAgentStage]


__all__ = [
    "AgentBuilder",
    "AgentContext",
    "AgentDefinition",
    "MultiAgentStage",
    "StageDefinition",
    "StageRunner",
    "WorkflowDefinition",
    "PromptBuilder",
]
