from __future__ import annotations

import asyncio
from typing import Callable, Dict, Iterable, List, Optional

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

from agents.literature.workflow import get_literature_workflow
from agents.migration.runtime import clear_active_paper, set_active_paper
from agents.migration.workflow import get_migration_workflow

from .config import AppPaths, build_default_context, resolve_year_filter
from .pipeline import AgentPipeline
from .storage import StageStorage
from .types import WorkflowDefinition


WorkflowFactory = Callable[[], WorkflowDefinition]

class AgentOrchestrator:
    """Coordinates workflow execution for CLI runs and the API server."""

    def __init__(self, *, paths: AppPaths, logger, stage_callback=None) -> None:
        self.paths = paths
        self.logger = logger
        self._stage_callback = stage_callback
        self._lock = asyncio.Lock()
        self._workflow_registry: Dict[str, WorkflowFactory] = {
            "literature": get_literature_workflow,
            "migration": get_migration_workflow,
        }
        self._options = ClaudeAgentOptions(
            allowed_tools=[
                "Read",
                "Write",
                "Edit",
                "Bash",
                "Grep",
                "Glob",
                "WebSearch",
                "catalog_article",
                "test_migration",
                "tune_migration_model",
            ],
            permission_mode="bypassPermissions",
        )

    async def run_workflows(
        self,
        workflow_names: Iterable[str],
        *,
        search_terms: List[str] | None = None,
        selected_papers: List[dict] | None = None,
        year_filter: str | None = None,
        conference_filters: List[str] | None = None,
        preserve_stage_log: bool = False,
    ) -> None:
        workflows = self._materialize_workflows(workflow_names)
        if not workflows:
            raise ValueError("No workflows resolved for execution.")

        async with self._lock:
            context = build_default_context(self.paths)
            context.search_terms = list(search_terms or [])
            context.selected_papers = list(selected_papers or [])
            normalized_year, year_label, year_value = resolve_year_filter(year_filter)
            context.search_year_mode = normalized_year
            context.search_year_label = f"{year_label}{f'（{year_value}）' if year_value else ''}".strip()
            context.search_year_value = year_value
            conference_list = [
                entry.strip()
                for entry in (conference_filters or [])
                if isinstance(entry, str) and entry.strip()
            ]
            if conference_list:
                context.conferences = conference_list
                context.search_conference_filters = conference_list
            else:
                context.search_conference_filters = []

            storage = StageStorage(self.paths.stage_log)
            if not preserve_stage_log:
                storage.reset()

            async with ClaudeSDKClient(options=self._options) as client:
                pipeline = AgentPipeline(
                    client=client,
                    workflows=workflows,
                    storage=storage,
                    logger=self.logger,
                    context=context,
                    stage_callback=self._stage_callback,
                )
                await pipeline.run()

    async def run_full_pipeline(self) -> None:
        await self.run_workflows(["literature", "migration"])

    async def run_literature(
        self,
        search_terms: List[str],
        *,
        year_filter: str | None = None,
        conference_filters: List[str] | None = None,
    ) -> None:
        await self.run_workflows(
            ["literature"],
            search_terms=search_terms,
            year_filter=year_filter,
            conference_filters=conference_filters,
        )

    async def run_migration(self, selected_papers: List[dict]) -> None:
        await self.run_migration_sequence(selected_papers)

    async def run_migration_sequence(self, selected_papers: List[dict]) -> None:
        if not selected_papers:
            return
        for index, paper in enumerate(selected_papers):
            set_active_paper(paper)
            try:
                await self.run_workflows(
                    ["migration"],
                    selected_papers=[paper],
                    preserve_stage_log=index > 0,
                )
            finally:
                clear_active_paper()

    def _materialize_workflows(self, names: Iterable[str]) -> List[WorkflowDefinition]:
        workflows = []
        for name in names:
            factory = self._workflow_registry.get(name)
            if not factory:
                raise ValueError(f"Unknown workflow requested: {name}")
            workflows.append(factory())
        return workflows


__all__ = ["AgentOrchestrator"]
