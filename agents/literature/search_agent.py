from __future__ import annotations

import json
from dataclasses import dataclass
from textwrap import dedent
from typing import List

from agents.core.types import AgentContext, StageDefinition


DEFAULT_KEYWORDS = [
    "traffic state prediction",
    "spatiotemporal forecasting",
]


@dataclass
class PaperSearchStageAgent:
    """Claude-driven MAS agent for paper search."""

    key: str = "paper_search_agent"
    title: str = "Paper Search Agent"
    description: str = "MAS agent: Claude searches for relevant papers using search_paper tool."

    def build_stage(self) -> StageDefinition:
        return StageDefinition(
            key=self.key,
            title=self.title,
            description=self.description,
            prompt_builder=build_search_agent_prompt,
        )


def build_search_agent_prompt(context: AgentContext) -> str:
    """Build prompt for the Claude-driven paper search agent.

    This prompt incorporates the core logic from the original PaSa prompts:
    - generate_query: Generate search queries based on user query
    - get_selected/get_value: Evaluate paper relevance
    """

    # Build search keywords
    keywords = [term.strip() for term in context.search_terms if term.strip()]
    if keywords:
        keywords.extend(DEFAULT_KEYWORDS)
    else:
        keywords = list(DEFAULT_KEYWORDS)

    keywords_str = ", ".join(keywords)

    # Conference constraints
    conferences_payload = context.search_conference_filters or context.conferences
    conference_str = ", ".join(conferences_payload) if conferences_payload else "All major AI/ML conferences"

    # Year filter - support year range
    year_info = ""
    if context.search_year_start and context.search_year_end:
        if context.search_year_start == context.search_year_end:
            year_info = f"Focus ONLY on papers from year {context.search_year_start}. Include the year in every search query."
        else:
            year_info = f"Focus ONLY on papers from {context.search_year_start} to {context.search_year_end}. Include the year range in every search query (e.g., '2024 OR 2025')."
    elif context.search_year_value:
        year_info = f"Focus ONLY on papers from year {context.search_year_value}. Include the year in every search query."
    elif context.target_year:
        year_info = f"Focus on papers from year {context.target_year} or recent years."
    else:
        year_info = "Include papers from recent years (preferably 2020-2025)."

    return dedent(
        f"""
        You are an elite Academic Research Assistant specializing in Spatio-Temporal Data Mining (STDM) and Traffic Prediction.

        ## Your Task
        Search for relevant academic papers based on the user's research interests and build a candidate list for further analysis.

        ## User Query and Search Constraints
        - **Research Keywords**: {keywords_str}
        - **Target Conferences**: {conference_str}
        - **Year Constraint**: {year_info}

        ## Instructions

        ### Step 1: Generate Search Queries
        Based on the research keywords, generate multiple diverse and mutually exclusive search queries to maximize coverage:
        - Include both specific technical terms and broader topic terms
        - Consider different aspects: methods (deep learning, graph neural networks, transformers), applications (traffic forecasting, urban computing), and data types (spatial-temporal data, time series)
        - Searching for survey papers can help discover more related work

        Example queries you might generate:
        - "spatial-temporal graph neural network traffic prediction"
        - "deep learning urban traffic forecasting"
        - "transformer time series forecasting survey"

        ### Step 2: Execute Searches
        Use the `search_paper` tool to search for papers. Call it multiple times with different queries to get comprehensive results.

        Tool parameters:
        - `query`: Your search query string
        - `num`: Number of results to retrieve (recommend 10-15 per query)
        - `end_date`: Optional date filter in YYYYMMDD format

        ### Step 3: Evaluate Relevance
        For each paper found, evaluate whether it satisfies the user's research needs:

        **Evaluation Criteria** (from original PaSa prompts):
        - Does the paper fully satisfy the detailed requirements of the user query?
        - Is the paper relevant to spatial-temporal data mining or traffic prediction?
        - Does it present novel methods, datasets, or significant experimental results?

        Provide a relevance score (0-10) and brief reasoning for each paper.

        ### Step 4: Compile Results
        After searching and evaluating, compile a final list of candidate papers in JSON format:

        ```json
        [
            {{
                "title": "Paper Title",
                "arxiv_id": "2401.12345",
                "abstract": "Brief abstract...",
                "abs_url": "https://arxiv.org/abs/2401.12345",
                "pdf_url": "https://arxiv.org/pdf/2401.12345.pdf",
                "year": 2024,
                "relevance_score": 8,
                "relevance_reason": "Highly relevant because..."
            }}
        ]
        ```

        ## Output Requirements
        1. Execute at least 3-5 different search queries
        2. Aim to find 15-30 relevant candidate papers
        3. Rank papers by relevance score (highest first)
        4. Provide a summary of the search process and key findings
        5. The final JSON candidate list should be clearly marked for the next stage to process

        ## Important Notes
        - Focus on finding papers that could contribute to traffic forecasting benchmark studies
        - Prioritize papers with available code repositories
        - Include both foundational works and recent state-of-the-art methods
        - If a query returns no results, try alternative formulations

        Begin your paper search now. Start by generating your search queries, then execute them systematically.
        """
    ).strip()


# Keep backward compatibility exports
class PaperSearchAgent:
    """Deprecated: Use PaperSearchStageAgent instead. This class exists for backward compatibility."""

    def __init__(self, **kwargs):
        import warnings
        warnings.warn(
            "PaperSearchAgent is deprecated. Use PaperSearchStageAgent with Claude SDK instead.",
            DeprecationWarning,
            stacklevel=2
        )

    def search(self, context: AgentContext) -> List[dict]:
        raise NotImplementedError(
            "Direct search is no longer supported. Use the Claude-driven PaperSearchStageAgent instead."
        )


__all__ = ["PaperSearchStageAgent", "PaperSearchAgent", "build_search_agent_prompt"]
