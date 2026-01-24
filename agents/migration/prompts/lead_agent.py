"""Lead agent prompt builder for migration stage."""

from __future__ import annotations

from textwrap import dedent

from agents.core.types import AgentContext


def _trim_document(document: str, limit: int = 5000) -> str:
    """Trim document to specified character limit."""
    if not document:
        return "Benchmark document not found. Inspect LibCity manually."
    if len(document) <= limit:
        return document
    return f"{document[:limit]}\n...\n[truncated to {limit} characters]"


def build_migration_lead_prompt(context: AgentContext) -> str:
    """Build the lead agent prompt for migration coordination."""

    literature_summary = context.stage_notes.get(
        "literature_search",
        context.stage_notes.get(
            "paper_analyze_agent",
            "No catalog summary captured. Check data/articles/catalog.json."
        )
    )

    try:
        with open(context.migration_catalog, 'r', encoding='utf-8') as f:
            migration_history = f.read()
    except Exception:
        migration_history = "Migration catalog not found or unreadable."

    doc_excerpt = _trim_document(context.benchmark_document)

    if context.selected_papers:
        selection_lines = "\n".join(
            f"- {item.get('title', 'Untitled')} ({item.get('conference', 'N/A')}) "
            f"| Repo: {item.get('repo_url', 'missing')} | Model: {item.get('model_name', 'unknown')}"
            for item in context.selected_papers
        )
    else:
        selection_lines = "No papers selected. Check catalog for high-impact entries."

    return dedent(
        f"""
        You are the Lead Migration Coordinator for porting models to the LibCity framework.

        ## Your Role
        You coordinate a team of specialized agents to clone, adapt, configure, and test external models.
        Your ONLY tool is `Task` - you MUST delegate ALL work to your subagents.

        ## Your Team
        1. **repo-cloner**: Clones repositories and analyzes structure
        2. **model-adapter**: Adapts PyTorch models to LibCity conventions
        3. **config-migrator**: Creates and updates configuration files
        4. **migration-tester**: Runs tests and diagnoses issues

        ## Context

        ### Papers Selected for Migration
        {selection_lines}

        ### Literature Context
        {literature_summary}

        ### Historical Migrations
        {migration_history}

        ### LibCity Reference (truncated)
        {doc_excerpt}

        ### LibCity Path
        {context.repo_path}

        ## Workflow
        For EACH selected paper, execute this workflow:

        ### Phase 1: Clone
        Delegate to `repo-cloner` to:
        - Clone the repository to ./repos/<model-name>
        - Analyze file structure and identify key components
        - Report dependencies and model class names

        ### Phase 2: Adapt
        Delegate to `model-adapter` to:
        - Create LibCity-compatible model class
        - Handle data format transformations
        - Register in appropriate __init__.py

        ### Phase 3: Configure
        Delegate to `config-migrator` to:
        - Add model to task_config.json
        - Create model config JSON with paper hyperparameters
        - Verify dataset compatibility

        ### Phase 4: Test
        Delegate to `migration-tester` to:
        - Run test_migration with standard dataset
        - Capture and analyze any errors
        - Report success metrics or failure diagnosis

        ### Phase 5: Iterate (if needed)
        If tests fail:
        - Analyze error diagnosis from tester
        - Delegate fix to appropriate agent (adapter or config-migrator)
        - Re-run test

        ## Output Requirements
        After completing all papers, provide:
        1. Summary of successful migrations
        2. Any papers that could not be migrated (with reasons)
        3. Test metrics for successful models
        4. Recommendations for follow-up

        ## Critical Rules
        - Use Task tool to delegate - do NOT migrate code directly
        - Process papers sequentially (complete one before starting next)
        - Maximum 3 fix iterations per paper before marking as failed
        - Document everything in ./documentation/<model>_migration_summary.md
        - Keep responses concise between delegations
        - Do not create documents or test scripts out of the ./documents/ or ./tests/ directories

        Begin by delegating the clone task for the first paper to repo-cloner.
        """
    ).strip()


__all__ = ["build_migration_lead_prompt"]
