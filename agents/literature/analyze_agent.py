from __future__ import annotations

import json
from dataclasses import dataclass
from textwrap import dedent

from agents.core.types import AgentContext, StageDefinition


@dataclass
class PaperAnalyzeAgent:
    """Claude-driven MAS agent for reading/downloading papers."""

    key: str = "paper_analyze_agent"
    title: str = "Paper Analyze Agent"
    description: str = "MAS agent: Claude reviews PaSa candidates, downloads PDFs, and records datasets/repos."

    def build_stage(self) -> StageDefinition:
        return StageDefinition(
            key=self.key,
            title=self.title,
            description=self.description,
            prompt_builder=build_analyze_agent_prompt,
        )


def build_analyze_agent_prompt(context: AgentContext) -> str:
    conferences = ", ".join(context.conferences)
    keywords = (
        ", ".join(context.search_terms)
        if context.search_terms
        else "Not specified; cover canonical traffic forecasting topics and scan ICLR/ICML/NeurIPS venues."
    )
    year_filter = context.search_year_label or "All"
    conferences_payload = context.search_conference_filters or context.conferences
    conference_hint = json.dumps(conferences_payload, ensure_ascii=False)
    candidates = context.paper_candidates or []
    preview = (
        json.dumps(candidates[: min(len(candidates), 20)], ensure_ascii=False, indent=2)
        if candidates
        else "[]"
    )
    return dedent(
        f"""
        Stage: Paper Analyze (MAS Agent)
        Goal: Using the candidate list returned by the paper search agent, download, catalog, and analyze every traffic forecasting paper from {conferences}.

        User-provided keywords: {keywords}
        Year filter: {year_filter}
        Conference constraints: {", ".join(conferences_payload)} (payload hint: {conference_hint})

        Candidate papers discovered by the paper search agent ({len(candidates)} total):
        {preview}

        Requirements:
        1. Work through the candidates one by one. If you believe an essential paper is missing you may do limited manual web lookups, but DO NOT call `paper_search_agent` again—this stage is analysis only.
        2. Download PDFs for papers into {context.article_dir}. Name files `<conference>_<short-title>.pdf`, record the saved path, and confirm each download. If no repo exists, explain why the PDF was skipped.
        3. For each paper having a PDF, extract and document from the PDF:
           - Title, conference, track, and the main publication link (prefer arXiv or the official proceedings page).
           - Every dataset and evaluation metric mentioned.
           - The official GitHub repository URL. If none exists, explicitly note it.
        4. After documenting a paper, call `catalog_article` so {context.article_catalog} stays synchronized.
        5. Produce a markdown table listing all analyzed papers with columns ["title","conference","year","track","pdf_link","pdf_path","datasets","metrics","repo_url","keywords"].
        6. Summarize all saved PDF paths plus any missing information (e.g., repository absent, datasets unclear).

        Deliverables: the markdown table, PDF download confirmations, catalog updates, and any follow-up recommendations.
        """
    ).strip()


__all__ = ["PaperAnalyzeAgent", "build_analyze_agent_prompt"]
