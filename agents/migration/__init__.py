from .catalog import MigrationCatalog
from .hooks import create_migration_stage_callback
from .workflow import get_migration_workflow

# New multi-agent imports
from .prompts.lead_agent import build_migration_lead_prompt
from agents.core.agent_registry import build_migration_agents

# Legacy imports (for backward compatibility)
from .legacy_prompts import build_migration_prompt, build_validation_prompt

__all__ = [
    # Main workflow
    "get_migration_workflow",
    # Multi-agent components
    "build_migration_agents",
    "build_migration_lead_prompt",
    # Utilities
    "create_migration_stage_callback",
    "MigrationCatalog",
    # Legacy (for backward compatibility)
    "build_migration_prompt",
    "build_validation_prompt",
]
