from __future__ import annotations

from textwrap import dedent

from agents.core.types import AgentContext


def _describe_selection(context: AgentContext) -> str:
    if not context.selected_papers:
        return "No papers explicitly selected. Align with the latest migration catalog or prompt the operator for a target."
    lines = []
    for item in context.selected_papers:
        title = item.get("title", "Untitled")
        conference = item.get("conference", "N/A")
        model_name = item.get("model_name", "") or "model unspecified"
        repo_url = item.get("repo_url", "missing")
        lines.append(f"- {title} ({conference}) | Model: {model_name} | Repo: {repo_url}")
    return "\n".join(lines)


def build_tuning_prompt(context: AgentContext) -> str:
    """Prompt for the dedicated tuning workflow that follows migration."""

    migration_notes = context.stage_notes.get(
        "model_migration",
        "Migration summaries not found. Ensure repository migration has been completed before tuning.",
    )
    verification_notes = context.stage_notes.get(
        "verification",
        "Verification notes missing. Re-run the verification stage or supply datasets/configs used during migration.",
    )
    metrics_notes = context.stage_notes.get(
        "metrics",
        "Metrics recap not captured yet. Use recent training logs or rerun evaluation before tuning.",
    )
    selection = _describe_selection(context)
    return dedent(
        f"""
        Stage: Hyperparameter Tuning

        Purpose: refine migrated LibCity models using run_hyper.py or a controlled grid search without re-running the migration steps.

        Papers selected by the user:
        {selection}

        Migration recap:
        {migration_notes}

        Verification recap:
        {verification_notes}

        Metrics recap:
        {metrics_notes}

        Instructions:
        - For each migrated model/dataset, call `tune_migration_model` (LibCity repo located at {context.repo_path}) with a clear search space.
        - Prefer LibCity's run_hyper.py when supported; otherwise fall back to the custom grid search path exposed by the tool.
        - Keep datasets/config files consistent with the migration and verification context; avoid re-running repository cloning or structural edits here.
        - Record best params/metrics, save any emitted hyper.result files and params files under ./data/runs, and note where artifacts were stored.
        - Produce a concise markdown summary describing the search space, trials executed, best scores, and recommended next steps.
        """
    ).strip()


__all__ = ["build_tuning_prompt"]
