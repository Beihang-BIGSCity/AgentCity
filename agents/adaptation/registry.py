"""Roster of the seven adaptation agents, grouped by layer.

Decision layer : ``task-planner`` (also the lead of both team stages)
Execution layer: ``data-integrator``, ``model-composer``, ``run-tester``
Tuning layer   : ``hyper-searcher``, ``perf-debugger``, ``effect-evaluator``
"""

from __future__ import annotations

from typing import Dict

from agents.core.types import AgentContext, AgentDefinition

from .prompts.data_integrator import build_data_integrator_prompt
from .prompts.effect_evaluator import build_effect_evaluator_prompt
from .prompts.hyper_searcher import build_hyper_searcher_prompt
from .prompts.model_composer import build_model_composer_prompt
from .prompts.perf_debugger import build_perf_debugger_prompt
from .prompts.run_tester import build_run_tester_prompt

EXECUTION_AGENT_NAMES = ("data-integrator", "model-composer", "run-tester")
TUNING_AGENT_NAMES = ("hyper-searcher", "perf-debugger", "effect-evaluator")
DECISION_AGENT_NAME = "task-planner"


def build_execution_agents(context: AgentContext) -> Dict[str, AgentDefinition]:
    """Build the execution-layer roster (数据集成 / 模型组合 / 运行测试)."""

    return {
        "data-integrator": AgentDefinition(
            name="data-integrator",
            description=(
                "数据集成智能体：调用 analyze_dataset 计算场景数据的统计特征(CVglobal/CVnode/SH/rho1/采样间隔/节点数)，"
                "核对 LibCity 原子文件，产出数据集成契约 data_integration.json。"
            ),
            tools=["analyze_dataset", "Read", "Write", "Glob", "Bash"],
            prompt=build_data_integrator_prompt(context),
            model="balanced",
        ),
        "model-composer": AgentDefinition(
            name="model-composer",
            description=(
                "模型组合智能体：依据数据特征画像、专家知识库与原子化模块库选择五类模块，生成可运行的 LibCity 模型代码"
                "并注册配置与 task_config。"
            ),
            tools=[
                "retrieve_knowledge",
                "list_modules",
                "get_module",
                "register_module",
                "Read",
                "Write",
                "Edit",
                "Glob",
                "Grep",
                "Bash",
            ],
            prompt=build_model_composer_prompt(context),
            model="strong",
        ),
        "run-tester": AgentDefinition(
            name="run-tester",
            description=(
                "运行测试智能体：调用 run_adaptation 在场景数据集上执行训练与测试，采集 LibCity 指标并落盘 run_report.json。"
            ),
            tools=["run_adaptation", "Read", "Write", "Bash"],
            prompt=build_run_tester_prompt(context),
            model="balanced",
        ),
    }


def build_tuning_agents(context: AgentContext) -> Dict[str, AgentDefinition]:
    """Build the tuning-layer roster (超参数搜索 / 性能调试 / 效果评估)."""

    return {
        "hyper-searcher": AgentDefinition(
            name="hyper-searcher",
            description=(
                "超参数搜索智能体：检索调参 Skills 设计小规模搜索空间，调用 tune_adaptation 执行 trial 并产出 tuning.json。"
            ),
            tools=["retrieve_knowledge", "tune_adaptation", "Read", "Write", "Bash"],
            prompt=build_hyper_searcher_prompt(context),
            model="balanced",
        ),
        "perf-debugger": AgentDefinition(
            name="perf-debugger",
            description=(
                "性能调试智能体：诊断 OOM/NaN/形状/数据契约类缺陷，修复生成模型或配置并重跑，把过程写入 tuning.json 的 debug_log。"
            ),
            tools=["run_adaptation", "Read", "Edit", "Write", "Grep", "Bash"],
            prompt=build_perf_debugger_prompt(context),
            model="balanced",
        ),
        "effect-evaluator": AgentDefinition(
            name="effect-evaluator",
            description=(
                "效果评估智能体：按目标指标阈值严格判定成败，写出 verdict.json，并通过 record_experience 把本轮经验回流到"
                "专家知识库与 Skills。"
            ),
            tools=["record_experience", "Read", "Write"],
            prompt=build_effect_evaluator_prompt(context),
            model="strong",
        ),
    }


__all__ = [
    "DECISION_AGENT_NAME",
    "EXECUTION_AGENT_NAMES",
    "TUNING_AGENT_NAMES",
    "build_execution_agents",
    "build_tuning_agents",
]
