from __future__ import annotations

import asyncio
import json
from typing import Callable, Dict, Iterable, List, Optional

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

from agents.literature.workflow import get_literature_workflow
from agents.migration.runtime import clear_active_paper, set_active_paper
from agents.migration.workflow import get_migration_workflow
from agents.tuning.workflow import get_tuning_workflow

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
            "tuning": get_tuning_workflow,
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
                "search_paper",
                "evaluate_paper_relevance",
                "catalog_article",
                "test_migration",
                "tune_migration_model",
            ],
            permission_mode="bypassPermissions",
            setting_sources=["project"],  # Load skills (pdf) and plugins (ralph-wiggum) from .claude directory
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
        stage_notes: Dict[str, str] | None = None,
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
            if stage_notes:
                context.stage_notes.update(stage_notes)

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

    async def run_tuning(self, selected_papers: List[dict]) -> None:
        await self.run_tuning_sequence(selected_papers)

    async def run_tuning_sequence(self, selected_papers: List[dict]) -> None:
        if not selected_papers:
            return
        stage_history = self._load_stage_notes_history()
        for index, paper in enumerate(selected_papers):
            preserve_log = bool(stage_history) or index > 0
            stage_notes = self._select_stage_notes(stage_history, index)
            set_active_paper(paper)
            try:
                await self.run_workflows(
                    ["tuning"],
                    selected_papers=[paper],
                    preserve_stage_log=preserve_log,
                    stage_notes=stage_notes,
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

    def _load_existing_stage_notes(self) -> Dict[str, str]:
        """Load the latest persisted stage summaries to seed follow-up workflows."""

        history = self._load_stage_notes_history()
        return {key: values[-1] for key, values in history.items() if values}

    def _load_stage_notes_history(self) -> Dict[str, List[str]]:
        """Return a chronological list of summaries per stage key."""

        try:
            payload = json.loads(self.paths.stage_log.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        stages = payload.get("stages") if isinstance(payload, dict) else []
        if not isinstance(stages, list):
            return {}
        history: Dict[str, List[str]] = {}
        for entry in stages:
            key = entry.get("key")
            summary = entry.get("summary")
            if not key or not isinstance(summary, str):
                continue
            cleaned = summary.strip()
            if not cleaned:
                continue
            history.setdefault(key, []).append(cleaned)
        return history

    @staticmethod
    def _select_stage_notes(history: Dict[str, List[str]], index: int) -> Dict[str, str]:
        """Pick the stage summaries aligned to the paper index when available."""

        notes: Dict[str, str] = {}
        for key, summaries in history.items():
            if not summaries:
                continue
            if 0 <= index < len(summaries):
                notes[key] = summaries[index]
            else:
                notes[key] = summaries[-1]
        return notes


__all__ = ["AgentOrchestrator"]
