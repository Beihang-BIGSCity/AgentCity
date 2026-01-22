from .workflow import get_literature_workflow

# Legacy imports for backward compatibility
from .analyze_agent import PaperAnalyzeAgent, build_analyze_agent_prompt
from .search_agent import PaperSearchStageAgent, build_search_agent_prompt

# New multi-agent imports
from .prompts.lead_agent import build_literature_lead_prompt
from agents.core.agent_registry import build_literature_agents

__all__ = [
    # Main workflow
    "get_literature_workflow",
    # Multi-agent components
    "build_literature_agents",
    "build_literature_lead_prompt",
    # Legacy (for backward compatibility)
    "PaperAnalyzeAgent",
    "build_analyze_agent_prompt",
    "PaperSearchStageAgent",
    "build_search_agent_prompt",
]
