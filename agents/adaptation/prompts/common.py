"""Shared prompt fragments for the adaptation roles."""

from __future__ import annotations

import json
import os

from agents.core.types import AgentContext


def describe_task(context: AgentContext) -> str:
    """Render the task specification handed in by the CLI."""

    return json.dumps(context.task_spec or {}, ensure_ascii=False, indent=2)


def gpu_flag(context: AgentContext) -> str:
    """Render the normalized task's device setting as a tool/CLI boolean."""

    train = (context.task_spec or {}).get("train") or {}
    return "true" if train.get("gpu", False) else "false"


def describe_compute_policy(context: AgentContext) -> str:
    """Keep every role aligned with the task's device and numeric budgets."""

    flag = gpu_flag(context)
    device = "GPU" if flag == "true" else "CPU"
    interpreter = os.environ.get("AGENTCITY_LIBCITY_PYTHON", "").strip()
    runtime = (
        f" The verified LibCity interpreter is `{interpreter}`; use it for torch/LibCity shell checks. "
        "The orchestration interpreter can intentionally lack torch; do not discover other environments."
        if interpreter else ""
    )
    return (
        f"Use {device} for this task: pass `gpu={flag}` to every training/tuning tool call "
        f"and `--gpu {flag}` to LibCity commands. "
        f"Iteration training uses {(context.task_spec.get('train') or {}).get('max_epoch', 35)} epochs per baseline/trial; "
        f"run {context.task_spec.get('tuning', {}).get('max_trials', 5)} tuning trials per round. "
        "Preserve the task's train.extra_args split in every run. Select models using validation only; "
        "the test partition is reserved for final evaluation after all loops. Final training uses final_train, "
        "not the iteration epoch budget. "
        "There is no default model parameter-count ceiling. The `lightweight` efficiency preference is not "
        "a numeric parameter limit. Do not import parameter caps from retrieved priors, skills, papers or "
        "previous-round feedback. Only an explicit parameter budget in the current task specification "
        "may impose such a cap. Report parameter counts and select capacity using validation results and "
        "actual device memory/runtime constraints; do not reject or shrink a model solely because it "
        "exceeds a historical parameter count." + runtime
    )


def describe_paths(context: AgentContext) -> str:
    """Render every path a role may need, so prompts stay self-contained."""

    return "\n".join(
        [
            f"- LibCity framework: {context.repo_path}",
            f"- Atomic module library: {context.modlib_dir}",
            f"- Expert knowledge base: {context.knowledge_dir}",
            f"- Agent Skills: {context.skills_dir}",
            f"- Task working directory: {context.task_dir}",
            f"- Summary documents: {context.documentation_dir}",
        ]
    )


def describe_round(context: AgentContext) -> str:
    """Render the current round within the closed self-correction loop."""

    return f"Round {context.round_index} of {context.max_rounds}. Fixed loops: {context.task_spec.get('fixed_rounds', True)}."


def describe_feedback(context: AgentContext) -> str:
    """Render the previous round result; empty on the first round."""

    feedback = (context.feedback or "").strip()
    if not feedback:
        return "No previous round: this is the first attempt for this task."
    return feedback


def describe_round_state(context: AgentContext) -> str:
    """Render the artifact paths produced earlier in the current round."""

    return "\n".join(
        [
            f"- dataset profile: {context.task_dir}/dataset_profile.json",
            f"- adaptation plan: {context.task_dir}/plan.json",
            f"- composition plan: {context.task_dir}/composition.json",
            f"- data integration notes: {context.task_dir}/data_integration.json",
            f"- run report: {context.task_dir}/run_report.json",
            f"- tuning result: {context.task_dir}/tuning.json",
            f"- verdict: {context.task_dir}/verdict.json",
        ]
    )


# Tokens substituted into member (teammate) prompt templates. Team members run
# inside their own isolated workspace, so a prompt that only says
# ``<task_dir>/plan.json`` or the relative ``Bigscity-LibCity/...`` leaves the
# model to guess the location (observed live: a member ran
# ``find / -type d -name METR_LA`` and burned minutes before failing). Every
# member prompt is therefore rendered with absolute paths at build time.
MEMBER_PATH_TOKENS = {
    "__REPO__": "repo_path",
    "__TASK_DIR__": "task_dir",
    "__MODLIB__": "modlib_dir",
    "__KNOWLEDGE__": "knowledge_dir",
    "__SKILLS__": "skills_dir",
    "__DOCS__": "documentation_dir",
}


def member_workspace_brief(context: AgentContext) -> str:
    """Absolute-path map plus filesystem rules appended to every member prompt."""

    return "\n".join(
        [
            "## Absolute Paths (use these verbatim)",
            f"- Repository / LibCity framework root: {context.repo_path}",
            f"- Task working directory (shared artifacts, one JSON file per stage): {context.task_dir}",
            f"- Atomic module library: {context.modlib_dir}",
            f"- Expert knowledge base: {context.knowledge_dir}",
            f"- Agent Skills: {context.skills_dir}",
            f"- Summary documents: {context.documentation_dir}",
            "",
            "## Device and Budget Policy",
            describe_compute_policy(context),
            "",
            "## Filesystem Rules",
            "- Your shell starts inside your own isolated workspace, NOT inside the repository, so relative project",
            "  paths do not resolve.",
            "- Use the absolute paths above verbatim, and pass `cwd` = the repository root for every shell command.",
            "- Never search the filesystem (`find /`, `ls ~`, repo-wide grep) and never go looking for the repository,",
            "  the dataset or the atomic files: they are already in place at the paths above.",
            "- Read and write artifacts through absolute paths. Do not spend a turn locating what is listed here.",
        ]
    )


def render_member_prompt(template: str, context: AgentContext) -> str:
    """Render one member prompt template against the task's absolute paths."""

    rendered = template
    for token, attribute in MEMBER_PATH_TOKENS.items():
        rendered = rendered.replace(token, str(getattr(context, attribute, "")))
    rendered = rendered.replace("__GPU_FLAG__", gpu_flag(context))
    return rendered.rstrip() + "\n\n" + member_workspace_brief(context) + "\n"


__all__ = [
    "MEMBER_PATH_TOKENS",
    "describe_compute_policy",
    "describe_feedback",
    "describe_paths",
    "describe_round",
    "describe_round_state",
    "describe_task",
    "gpu_flag",
    "member_workspace_brief",
    "render_member_prompt",
]
