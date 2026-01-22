from __future__ import annotations

from textwrap import dedent
from agents.core.types import AgentContext


def _trim_document(document: str, limit: int = 5000) -> str:
    if not document:
        return "Benchmark document not found. Inspect LibCity manually."
    if len(document) <= limit:
        return document
    return f"{document[:limit]}\n...\n[truncated to {limit} characters]"


def build_migration_prompt(context: AgentContext) -> str:
    literature_summary = context.stage_notes.get(
        "paper_analyze_agent",
        "No catalog summary captured yet. Inspect data/articles/catalog.json for metadata before acting.",
    )
    try:
        with open(context.migration_catalog, 'r', encoding='utf-8') as f:
            migration_history_info = f.read()
    except Exception:
        migration_history_info = "Migration catalog not found or unreadable."
    doc_excerpt = _trim_document(context.benchmark_document)
    if context.selected_papers:
        selection_lines = "\n".join(
            f"- {item.get('title', 'Untitled')} ({item.get('conference', 'N/A')}) "
            f"| Repo: {item.get('repo_url', 'missing')} | Model: {item.get('model_name', 'unknown')}"
            for item in context.selected_papers
        )
    else:
        selection_lines = "No specific papers selected by the operator; prioritize highest impact entries from the catalog."
    return dedent(
        f"""
        You are a Senior AI Software Engineer and an expert in the benchmark framework. You excel at code refactoring and PyTorch model adaptation.

        Literature context:
        {literature_summary}

        Papers selected by the user for migration:
        {selection_lines}

        Historical Migration Records for Reference:
        {migration_history_info}

        Tasks:
        - Clone every listed repository (or as many as feasible) and port their models into LibCity
          at {context.repo_path}. Keep the upstream repo history intact and always clone into the project-level
          ./repos directory (e.g., ./repos/<model-name>) before wiring adapters/configs back into LibCity.
        - Determine whether this repository performs trajectory next-hop prediction, Estimated Time of Arrival, or traffic state prediction, and migrate it to the corresponding task folder. For the trajectory realted model, regardless of the model’s pre-training objective, after migration it should perform next-hop prediction at test time.
        - Align data ingestion: verify _init signatures, forward inputs, and dataset schemas. Create dataset subclasses
          or converters when shapes differ.
        - Modify config files to match LibCity's conventions, especially the config/task_config.json. Ensure hyperparameters, data paths, and training loops
        - Store automation helpers (e.g., prompt templates, config generators) in this agent project instead of scattering scripts.
        - Document all touched LibCity files and the ./repos/<model-name> directory. Produce only
          a concise summary markdown with no more than 2000 characters, and save it below ./documentation (one file per migration).
        - Test the migrated models with LibCity's existing training and evaluation scripts.

        Reference (truncated LibCity documentation):
        {doc_excerpt}

        Output: actionable checklist detailing edits, file paths, required summary file names under ./documentation.
        """
    ).strip()


def build_validation_prompt(context: AgentContext) -> str:
    migration_notes = context.stage_notes.get(
        "model_migration",
        "No migration log recorded. Summaries from the previous stage should be added here.",
    )
    return dedent(
        f"""
        You are a QA Engineer and Python Debugging Specialist. You are responsible for CI/CD and runtime error resolution.

        Summary of implemented migrations:
        {migration_notes}

        Instructions:
        - For every imported model, run the benchmark via tools (prefer `test_migration`) until training completes
          or a blocking error occurs. Capture stdout/stderr for each attempt.
        - When failures appear, patch the LibCity integration, document the fix, then rerun tests.
        - Record which datasets/configs were executed and whether the results converged.
        - Save structured logs to data/runs (e.g., data/runs/<model>_train.log) for future inspection.
        - Summarize everything in markdown with a comparison table and recommended follow-up experiments.

        Deliver the consolidated verification log and highlight outstanding issues.
        """
    ).strip()


'''def build_metrics_prompt(context: AgentContext) -> str:
    validation_notes = context.stage_notes.get(
        "verification",
        "Verification output missing—ensure tests ran before reporting metrics.",
    )
    return dedent(
        f"""
        Stage: Metrics & Analysis

        Verification recap:
        {validation_notes}

        Tasks:
        - Report the metrics for each migrationred model. Include dataset name and config file.
        - Compare observed metrics with values mentioned in the previously downloaded papers. Analyze deltas and reasons
          (data preprocessing, hyperparameters, missing features, etc.).
        - Recommend the search space and datasets/configs to feed into the dedicated tuning stage; do not launch tuning here.
        - If results diverge significantly, outline corrective steps or retry plans.
        - Summarize everything in markdown with a comparison table and recommended follow-up experiments.
        """
    ).strip()'''


__all__ = [
    "build_migration_prompt",
    "build_validation_prompt",
]
