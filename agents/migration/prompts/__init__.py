"""Prompts for migration stage agents."""

from .lead_agent import build_migration_lead_prompt
from .cloner import CLONER_SYSTEM_PROMPT
from .adapter import ADAPTER_SYSTEM_PROMPT
from .config import CONFIG_SYSTEM_PROMPT
from .tester import TESTER_SYSTEM_PROMPT

__all__ = [
    "build_migration_lead_prompt",
    "CLONER_SYSTEM_PROMPT",
    "ADAPTER_SYSTEM_PROMPT",
    "CONFIG_SYSTEM_PROMPT",
    "TESTER_SYSTEM_PROMPT",
]
