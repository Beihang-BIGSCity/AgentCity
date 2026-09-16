"""CLI entry point: run the full literature + migration pipeline."""

import asyncio

from agents import AppPaths, setup_logger
from agents.core.orchestrator import AgentOrchestrator
from agents.migration import MigrationCatalog, create_migration_stage_callback


async def main():
    paths = AppPaths()
    paths.ensure()
    migration_catalog = MigrationCatalog(
        catalog_path=paths.migration_catalog,
        documentation_dir=paths.documentation_dir,
        root=paths.root,
    )
    logger = setup_logger(paths.log_file)
    orchestrator = AgentOrchestrator(
        paths=paths,
        logger=logger,
        stage_callback=create_migration_stage_callback(migration_catalog),
    )
    await orchestrator.run_full_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
