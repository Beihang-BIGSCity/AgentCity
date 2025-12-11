from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, TYPE_CHECKING
import re

if TYPE_CHECKING:  # pragma: no cover - imported lazily at runtime
    from pasa.paper_agent import PaperAgent


DEFAULT_CRAWLER_MODEL = "bytedance-research/pasa-7b-crawler"
DEFAULT_SELECTOR_MODEL = "bytedance-research/pasa-7b-selector"


@dataclass
class PaSaSearchResult:
    title: str
    arxiv_id: str | None
    abstract: str
    score: float | None
    source: str
    depth: int
    abs_url: str
    pdf_url: str
    sections: Sequence[str]
    year: int | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "arxiv_id": self.arxiv_id,
            "abstract": self.abstract,
            "score": self.score,
            "source": self.source,
            "depth": self.depth,
            "abs_url": self.abs_url,
            "pdf_url": self.pdf_url,
            "sections": list(self.sections),
            "year": self.year,
        }


class PaSaSearchRunner:
    """Thin adapter around the PaSa paper search agent."""

    def __init__(
        self,
        *,
        base_dir: Path | str | None = None,
        crawler_model: str | None = None,
        selector_model: str | None = None,
        prompts_path: Path | str | None = None,
        expand_layers: int = 2,
        search_queries: int = 5,
        search_papers: int = 10,
        expand_papers: int = 20,
        threads_num: int = 20,
        agent_factory: Callable[[str, Dict[str, int], str], "PaperAgent"] | None = None,
    ) -> None:
        base = Path(base_dir) if base_dir else Path(__file__).resolve().parents[2] / "pasa"
        self.base_dir = base
        self.prompts_path = Path(prompts_path) if prompts_path else self.base_dir / "agent_prompt.json"
        if not self.prompts_path.exists():
            raise FileNotFoundError(
                f"PaSa prompt file not found at {self.prompts_path}. Ensure /pasa assets are available."
            )
        self.crawler_model = crawler_model or os.getenv("PASA_CRAWLER_MODEL", DEFAULT_CRAWLER_MODEL)
        self.selector_model = selector_model or os.getenv("PASA_SELECTOR_MODEL", DEFAULT_SELECTOR_MODEL)
        self.expand_layers = max(1, expand_layers)
        self.search_queries = max(1, search_queries)
        self.search_papers = max(1, search_papers)
        self.expand_papers = max(1, expand_papers)
        self.threads_num = max(1, threads_num)
        self._agent_factory = agent_factory
        self._crawler = None
        self._selector = None
        self._model_lock = Lock()

    def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        overrides: Dict[str, int] | None = None,
        end_date: str | None = None,
        year_filter: str | None = None,
        conference_filters: Sequence[str] | None = None,
    ) -> List[Dict[str, Any]]:
        cleaned_query = (query or "").strip()
        '''normalized_conferences = self._normalize_conferences(conference_filters)
        if normalized_conferences:
            conference_string = " ".join(normalized_conferences)
            cleaned_query = f"{conference_string} {cleaned_query}".strip()'''
        if not cleaned_query:
            raise ValueError("query is required")
        params = self._resolve_params(overrides or {})
        print(cleaned_query)
        agent = self._build_agent(cleaned_query, params, end_date=end_date)
        agent.search()
        results = self._extract_results(agent)
        target_year = self._resolve_target_year(year_filter)
        if target_year is not None:
            results = [item for item in results if item.year == target_year]
        if max_results is not None and max_results > 0:
            results = results[: max_results]
        return [result.to_dict() for result in results]

    def _resolve_params(self, overrides: Dict[str, int]) -> Dict[str, int]:
        def clamp(key: str, default: int) -> int:
            value = overrides.get(key, default)
            return default if value is None or value <= 0 else int(value)

        return {
            "expand_layers": clamp("expand_layers", self.expand_layers),
            "search_queries": clamp("search_queries", self.search_queries),
            "search_papers": clamp("search_papers", self.search_papers),
            "expand_papers": clamp("expand_papers", self.expand_papers),
            "threads_num": clamp("threads_num", self.threads_num),
        }

    def _build_agent(self, query: str, params: Dict[str, int], *, end_date: str | None = None):
        if self._agent_factory:
            return self._agent_factory(query, params, end_date or self._default_end_date())
        crawler, selector = self._ensure_models()
        from pasa.paper_agent import PaperAgent  # imported lazily

        return PaperAgent(
            user_query=query,
            crawler=crawler,
            selector=selector,
            end_date=end_date or self._default_end_date(),
            prompts_path=str(self.prompts_path),
            expand_layers=params["expand_layers"],
            search_queries=params["search_queries"],
            search_papers=params["search_papers"],
            expand_papers=params["expand_papers"],
            threads_num=params["threads_num"],
        )

    def _ensure_models(self):
        with self._model_lock:
            if self._crawler is None or self._selector is None:
                try:
                    from pasa.models import Agent as PaSaAgent  # type: ignore
                except ImportError as exc:  # pragma: no cover
                    raise RuntimeError(
                        "transformers is required to run the paper-search-agent. "
                        "Install dependencies from pasa/requirements.txt."
                    ) from exc
                self._crawler = PaSaAgent(self.crawler_model)
                self._selector = PaSaAgent(self.selector_model)
        return self._crawler, self._selector

    def _extract_results(self, agent) -> List[PaSaSearchResult]:
        root = getattr(agent, "root", None)
        if root is None:
            return []
        ranked: Dict[str, PaSaSearchResult] = {}
        for node in self._walk_nodes(root):
            identifier = getattr(node, "arxiv_id", None) or getattr(node, "title", None)
            title = getattr(node, "title", "") or ""
            if not identifier or not title:
                continue
            score = getattr(node, "select_score", None)
            candidate = self._node_to_result(node)
            if identifier not in ranked or self._is_better(score, ranked[identifier].score):
                ranked[identifier] = candidate
        return sorted(ranked.values(), key=lambda item: item.score or 0.0, reverse=True)

    @staticmethod
    def _is_better(score: float | None, current: float | None) -> bool:
        if current is None:
            return True
        if score is None:
            return False
        return score > current

    def _walk_nodes(self, root) -> Iterable[Any]:
        child_map = getattr(root, "child", {}) or {}
        stack: List[Any] = []
        for children in child_map.values():
            if isinstance(children, list):
                stack.extend(children)
        while stack:
            node = stack.pop()
            yield node
            nested = getattr(node, "child", {}) or {}
            for entries in nested.values():
                if isinstance(entries, list):
                    stack.extend(entries)

    def _node_to_result(self, node) -> PaSaSearchResult:
        arxiv_id = getattr(node, "arxiv_id", None) or None
        abstract = getattr(node, "abstract", "") or ""
        score = getattr(node, "select_score", None)
        depth = getattr(node, "depth", 0) or 0
        source = getattr(node, "source", "") or ""
        sections = []
        raw_sections = getattr(node, "sections", None)
        if isinstance(raw_sections, dict):
            sections = list(raw_sections.keys())
        abs_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""
        return PaSaSearchResult(
            title=getattr(node, "title", "") or "",
            arxiv_id=arxiv_id,
            abstract=abstract.strip(),
            score=score,
            source=source,
            depth=depth,
            abs_url=abs_url,
            pdf_url=pdf_url,
            sections=sections,
            year=self._infer_year(arxiv_id),
        )

    @staticmethod
    def _default_end_date() -> str:
        return datetime.utcnow().strftime("%Y%m%d")

    @staticmethod
    def _normalize_conferences(items: Sequence[str] | None) -> List[str]:
        normalized: List[str] = []
        if not items:
            return normalized
        for entry in items:
            value = str(entry).strip()
            if value:
                normalized.append(value)
        return normalized

    @staticmethod
    def _infer_year(arxiv_id: str | None) -> int | None:
        if not arxiv_id:
            return None
        match = re.match(r"(\d{2})(\d{2})\.", arxiv_id)
        if not match:
            return None
        yy = int(match.group(1))
        # arXiv switched to YYMM numbering in 2007; assume 20xx.
        return 2000 + yy

    @staticmethod
    def _resolve_target_year(year_filter: str | None) -> int | None:
        normalized = (year_filter or "").strip().lower()
        current_year = datetime.utcnow().year
        if normalized == "this_year":
            return current_year
        if normalized == "last_year":
            return current_year - 1
        return None


__all__ = ["PaSaSearchRunner", "PaSaSearchResult"]
