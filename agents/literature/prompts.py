from __future__ import annotations

from textwrap import dedent

from agents.core.types import AgentContext


def build_literature_prompt(context: AgentContext) -> str:
    conferences = ", ".join(context.conferences)
    keywords = ", ".join(context.search_terms) if context.search_terms else "用户未指定，需覆盖常规交通预测主题，查找ICLR/ICML/NIPS的会议论文"
    return dedent(
        f"""
        Stage: Comprehensive Literature Sweep
        Goal: Enumerate EVERY accessible {context.target_year} paper from {conferences}
        that targets traffic state forecast (no dataset restrictions).

        User-provided keywords to prioritize: {keywords}

        Requirements:
        1. Use WebSearch repeatedly until no new candidates appear. Capture workshop, spotlight, and journal versions.
        2. For every paper collect:
           - Title, conference name, track, and publication link.
           - Datasets (no limitation—record all), metrics, and any official GitHub repository.
        3. Download each PDF into {context.article_dir}. Use Bash (curl/wget) or Write to save files named
           <conference>_<short-title>.pdf. Confirm path for every download. For the papers without GitHub repository, you don't need to download.
        4. After saving, call the `catalog_article` tool with metadata so {context.article_catalog} stays synchronized.
        5. Produce a markdown table that lists every paper, dataset, repo URL, and the saved PDF path.
        6. The catalog columns should be: ["title","conference","year","track","pdf_link","pdf_path","datasets","metrics","repo_url","keywords"]
        
        Deliverables: exhaustive table, confirmation of all saved PDFs, and catalog updates.
        """
    ).strip()


__all__ = ["build_literature_prompt"]
