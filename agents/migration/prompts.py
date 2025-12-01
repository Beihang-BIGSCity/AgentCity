from __future__ import annotations

from textwrap import dedent

from agents.core.types import AgentContext


def _trim_document(document: str, limit: int = 3500) -> str:
    if not document:
        return "Benchmark document not found. Inspect LibCity manually."
    if len(document) <= limit:
        return document
    return f"{document[:limit]}\n...\n[truncated to {limit} characters]"


def build_migration_prompt(context: AgentContext) -> str:
    literature_summary = context.stage_notes.get(
        "literature_scan",
        "No catalog summary captured yet. Inspect data/articles/catalog.json for metadata before acting.",
    )
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
        Stage: Repository Migration & Model Porting

        Literature context:
        {literature_summary}

        Papers selected by the user for migration:
        {selection_lines}

        Tasks:
        - Clone every listed repository (or as many as feasible) and port their traffic-speed models into LibCity
          at {context.repo_path}. Keep the upstream repo history intact and always clone into the project-level
          ./repos directory (e.g., ./repos/<model-name>) before wiring adapters/configs back into LibCity.
        - Align data ingestion: verify _init signatures, forward inputs, and dataset schemas. Create dataset subclasses
          or converters when shapes differ.
        - Store automation helpers (e.g., prompt templates, config generators) in this agent project instead of scattering scripts.
        - Document all touched LibCity files and reference directories such as {context.model_path}. Produce only
          a concise summary markdown and save it below ./documentation (one file per migration).
        - Update config files so every new model can be trained via LibCity CLI with minimal switches. Place new or
          updated automated tests exclusively under ./tests (use pytest-style modules).

        Reference (truncated LibCity documentation):
        {doc_excerpt}

        Output: actionable checklist detailing edits, file paths, required summary file names under ./documentation,
        and pytest targets under ./tests—no other artifacts.
        """
    ).strip()


def build_validation_prompt(context: AgentContext) -> str:
    migration_notes = context.stage_notes.get(
        "model_migration",
        "No migration log recorded. Summaries from the previous stage should be added here.",
    )
    return dedent(
        f"""
        Stage: Automated Verification

        Summary of implemented migrations:
        {migration_notes}

        Instructions:
        - For every imported model, run the benchmark via tools (prefer `test_migration`) until training completes
          or a blocking error occurs. Capture stdout/stderr for each attempt.
        - When failures appear, patch the LibCity integration, document the fix, then rerun tests.
        - Record which datasets/configs were executed and whether the results converged.
        - Save structured logs to data/runs (e.g., data/runs/<model>_train.log) for future inspection.

        Deliver the consolidated verification log and highlight outstanding issues.
        """
    ).strip()


def build_metrics_prompt(context: AgentContext) -> str:
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
        - Report masked_mae, masked_mape, masked_mase for each migrationred model. Include dataset name and config file.
        - Compare observed metrics with values mentioned in the previously downloaded papers. Analyze deltas and reasons
          (data preprocessing, hyperparameters, missing features, etc.).
        - Run `tune_migration_model` so it automatically tries LibCity's ./Bigscity-LibCity/run_hyper.py when the migrated
          model supports it (falling back to the built-in grid search otherwise); summarize the search space, best parameters,
          and where logs/results were saved.
        - If results diverge significantly, outline corrective steps or retry plans.
        - Summarize everything in markdown with a comparison table and recommended follow-up experiments.
        """
    ).strip()


__all__ = [
    "build_migration_prompt",
    "build_validation_prompt",
    "build_metrics_prompt",
]
