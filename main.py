"""CLI entry point for the AgentCity pipeline (openJiuwen backend).

Replaces the old claude_client.py: business tools now live in
agents/tools/ (FastMCP server) and the execution layer is
agents/core/openjiuwen_backend.py. This module only wires up the
orchestrator.

Run: ``python main.py``
"""

import asyncio

from agents import AppPaths, setup_logger
from agents.core.orchestrator import AgentOrchestrator
from agents.migration import MigrationCatalog, create_migration_stage_callback

paths = AppPaths()
paths.ensure()

migration_catalog = MigrationCatalog(
    catalog_path=paths.migration_catalog,
    documentation_dir=paths.documentation_dir,
    root=paths.root,
)
stage_callback = create_migration_stage_callback(migration_catalog)


async def main():
    logger = setup_logger(paths.log_file)
    orchestrator = AgentOrchestrator(
        paths=paths,
        logger=logger,
        stage_callback=stage_callback,
    )
    await orchestrator.run_full_pipeline()


if __name__ == "__main__":
    # openjiuwen teardown leaks non-daemon threads (aiosqlite connections),
    # which would hang the interpreter at exit; force-exit after cleanup.
    import os
    import sys
    import traceback

    exit_code = 0
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        traceback.print_exc()
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
