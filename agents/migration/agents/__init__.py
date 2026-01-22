"""Agent definitions for migration stage.

This module re-exports the agent builder from the central registry.
Individual agent definitions are created dynamically based on context.
"""

from agents.core.agent_registry import build_migration_agents

__all__ = ["build_migration_agents"]
