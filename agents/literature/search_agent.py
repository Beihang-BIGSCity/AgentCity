from __future__ import annotations
import re
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

from agents.core.types import AgentContext, StageDefinition

from .pasa_adapter import PaSaSearchRunner

DEFAULT_KEYWORDS = [
    "traffic state prediction",
    "spatiotemporal forecasting",
]


@dataclass
class PaperSearchStageAgent:
    """MAS agent descriptor for PaSa-based crawling."""

    key: str = "paper_search_agent"
    title: str = "Paper Search Agent"
    description: str = "MAS agent: PaSa crawler producing candidate papers."

    def build_stage(self) -> StageDefinition:
        return StageDefinition(
            key=self.key,
            title=self.title,
            description=self.description,
            runner=run_paper_search_stage,
        )


class PaperSearchAgent:
    """Runs PaSa searches and returns deduplicated paper candidates."""

    def __init__(
        self,
        *,
        runner: PaSaSearchRunner | None = None,
        max_results: int = 25,
        max_queries: int = 6,
        max_conference_combos: int = 3,
        logger: logging.Logger | None = None,
    ) -> None:
        self.runner = runner or PaSaSearchRunner()
        self.max_results = max(5, max_results)
        self.max_queries = max(1, max_queries)
        self.max_conference_combos = max(1, max_conference_combos)
        self.logger = logger or logging.getLogger(__name__)

    def search(self, context: AgentContext) -> List[Dict[str, object]]:
        keywords = [term.strip() for term in context.search_terms if term.strip()]
        if keywords:
            keywords.extend(DEFAULT_KEYWORDS)
        else:
            keywords = list(DEFAULT_KEYWORDS)
        conferences: Sequence[str] = (
            context.search_conference_filters or context.conferences
        )
        year_filter = context.search_year_mode
        queries = self._build_queries(
            keywords,
            conferences,
            context.search_year_value or context.target_year,
        )
        aggregated: List[Dict[str, object]] = []
        seen: Set[str] = set()
        for query in queries:
            try:
                query = re.sub(r"\s*\[\s*|\s*\]\s*", " ", query).strip()
                results = self.runner.search(
                    query,
                    max_results=self.max_results,
                    year_filter=year_filter,
                    conference_filters=conferences,
                )
                print(f"PaSa search for query '{query}' returned {len(results)} results.")
            except Exception as exc:  # pragma: no cover - PaSa runtime errors
                self.logger.warning("Paper search failed for %s: %s", query, exc)
                continue
            for item in results:
                identifier = item.get("arxiv_id") or item.get("title")
                if not identifier or identifier in seen:
                    continue
                seen.add(identifier)
                aggregated.append(item)
        return aggregated

    def _build_queries(
        self,
        keywords: Sequence[str],
        conferences: Sequence[str],
        year_value: int | None,
    ) -> List[str]:
        queries: List[str] = []
        seen: Set[str] = set()

        def add_query(candidate: str) -> None:
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            if len(queries) < self.max_queries:
                queries.append(normalized)

        def with_year(term: str) -> str:
            return f"{term} {year_value}".strip() if year_value else term

        combo = with_year(f"{conferences[: self.max_conference_combos]} {keywords} ")
        add_query(combo)

        if not queries:
            add_query(with_year("traffic forecasting"))
        return queries


_SEARCH_AGENT: PaperSearchAgent | None = None


def _get_search_agent() -> PaperSearchAgent:
    global _SEARCH_AGENT
    if _SEARCH_AGENT is None:
        _SEARCH_AGENT = PaperSearchAgent()
    return _SEARCH_AGENT


async def run_paper_search_stage(context: AgentContext, stage: StageDefinition) -> str:
    """Stage runner that populates context.paper_candidates with PaSa results."""

    agent = _get_search_agent()
    results = await asyncio.to_thread(agent.search, context)
    context.paper_candidates = results
    if not results:
        return "Paper search agent did not locate any matching arXiv papers."
    #print(results)
    lines = [
        f"- {item.get('title', 'Untitled')} :: {item.get('abs_url') or item.get('pdf_url') or 'N/A'}"
        for item in results[: min(10, len(results))]
    ]
    summary = [
        f"Paper search agent located {len(results)} candidate papers for review.",
        "Top matches:",
        *lines,
    ]
    return "\n".join(summary)


__all__ = ["PaperSearchAgent", "PaperSearchStageAgent", "run_paper_search_stage"]
