"""Prompt builders for the seven adaptation roles.

Every role prompt is a builder over ``AgentContext`` rather than a constant:
team members run in isolated workspaces, so the member prompts must carry the
task's absolute paths instead of unsubstituted placeholders.
"""

from .common import member_workspace_brief, render_member_prompt
from .data_integrator import build_data_integrator_prompt
from .effect_evaluator import build_effect_evaluator_prompt
from .hyper_searcher import build_hyper_searcher_prompt
from .lead import build_execution_lead_prompt, build_tuning_lead_prompt
from .model_composer import build_model_composer_prompt
from .perf_debugger import build_perf_debugger_prompt
from .run_tester import build_run_tester_prompt
from .task_planner import build_task_planning_prompt

__all__ = [
    "build_data_integrator_prompt",
    "build_effect_evaluator_prompt",
    "build_execution_lead_prompt",
    "build_hyper_searcher_prompt",
    "build_model_composer_prompt",
    "build_perf_debugger_prompt",
    "build_run_tester_prompt",
    "build_task_planning_prompt",
    "build_tuning_lead_prompt",
    "member_workspace_brief",
    "render_member_prompt",
]
