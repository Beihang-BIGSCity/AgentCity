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
    task_id: str
    task_slug: str
    task_dir: str
    task_spec: Dict[str, Any] = field(default_factory=dict)
    round_index: int = 1
    max_rounds: int = 3
    modlib_dir: str = ""
    knowledge_dir: str = ""
    skills_dir: str = ""
    documentation_dir: str = ""
    feedback: str = ""
    dataset_profile: Dict[str, Any] = field(default_factory=dict)
    stage_notes: Dict[str, str] = field(default_factory=dict)


@dataclass
class StageDefinition:
    key: str
    title: str
    description: str
    prompt_builder: PromptBuilder | None = None
    runner: StageRunner | None = None
    model: str = "balanced"


@dataclass
class AgentDefinition:
    """Defines a specialized subagent with constrained tools and prompts.

    Used by MultiAgentStage to configure subagents that the lead agent
    can delegate tasks to via the team collaboration tools.
    """

    name: str
    description: str
    tools: List[str]
    prompt: str
    model: str = "balanced"


@dataclass
class MultiAgentStage:
    """A stage that uses lead-agent + subagent coordination.

    The lead agent delegates work to specialized subagents defined in the
    agents dict through the framework's in-process team tools.

    The agents field can be either:
    - A dict of AgentDefinition objects (static configuration)
    - A callable that takes AgentContext and returns the dict (dynamic configuration)
    """

    key: str
    title: str
    description: str
    lead_agent_prompt_builder: PromptBuilder
    agents: Union[Dict[str, AgentDefinition], AgentBuilder]
    lead_model: str = "balanced"


@dataclass
class WorkflowDefinition:
    """The stage sequence of one pipeline (here: the adaptation pipeline)."""

    name: str
    description: str
    stages: List[StageDefinition | MultiAgentStage]


__all__ = [
    "AgentBuilder",
    "AgentContext",
    "AgentDefinition",
    "MultiAgentStage",
    "PromptBuilder",
    "StageDefinition",
    "StageRunner",
    "WorkflowDefinition",
]
