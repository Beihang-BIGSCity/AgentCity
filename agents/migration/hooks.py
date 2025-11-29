from __future__ import annotations

from typing import Callable

from agents.core.types import AgentContext, StageDefinition

from .catalog import MigrationCatalog
from .runtime import get_active_paper


def create_migration_stage_callback(
    catalog: MigrationCatalog,
) -> Callable[[StageDefinition, str, AgentContext], None]:
    """Return a stage callback that records migration summaries."""

    def _callback(stage: StageDefinition, summary: str, _context: AgentContext) -> None:
        if stage.key != "model_migration":
            return
        catalog.record_summary(get_active_paper(), summary)

    return _callback


__all__ = ["create_migration_stage_callback"]
