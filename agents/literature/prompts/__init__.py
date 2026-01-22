"""Prompts for literature stage agents."""

from .lead_agent import build_literature_lead_prompt
from .searcher import SEARCHER_SYSTEM_PROMPT
from .analyzer import ANALYZER_SYSTEM_PROMPT
from .evaluator import EVALUATOR_SYSTEM_PROMPT

__all__ = [
    "build_literature_lead_prompt",
    "SEARCHER_SYSTEM_PROMPT",
    "ANALYZER_SYSTEM_PROMPT",
    "EVALUATOR_SYSTEM_PROMPT",
]
