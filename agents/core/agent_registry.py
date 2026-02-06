"""Central registry for building AgentDefinition instances.

This module provides factory functions that build AgentDefinition dictionaries
for each stage's subagents. These are used by MultiAgentStage to configure
the specialized agents that the lead agent can delegate tasks to.
"""

from __future__ import annotations

from typing import Dict

from .types import AgentContext, AgentDefinition


def build_literature_agents(context: AgentContext) -> Dict[str, AgentDefinition]:
    """Build literature stage agents.

    Returns agents for paper searching, analysis, and relevance evaluation.
    """
    from agents.literature.prompts.searcher import SEARCHER_SYSTEM_PROMPT
    from agents.literature.prompts.analyzer import ANALYZER_SYSTEM_PROMPT
    from agents.literature.prompts.evaluator import EVALUATOR_SYSTEM_PROMPT

    return {
        "paper-searcher": AgentDefinition(
            name="paper-searcher",
            description=(
                "Searches academic databases for relevant papers using configured "
                "keywords and filters. Uses WebSearch and search_paper tools to find "
                "papers from arXiv and other sources. Returns raw paper metadata "
                "(title, abstract, arxiv_id, URLs) without evaluating relevance."
            ),
            tools=["WebSearch", "search_paper"],
            prompt=SEARCHER_SYSTEM_PROMPT,
            model="haiku",
        ),
        "paper-analyzer": AgentDefinition(
            name="paper-analyzer",
            description=(
                "Reads PDF papers, extracts metadata, datasets, and repository information. "
                "Uses Read tool to read PDFs directly (Claude Code supports PDF natively). "
                f"Saves paper metadata to {context.article_dir}/catalog.json using Write tool."
            ),
            tools=["Read", "Write", "Glob"],
            prompt=ANALYZER_SYSTEM_PROMPT,
            model="opus",
        ),
        "paper-evaluator": AgentDefinition(
            name="paper-evaluator",
            description=(
                "Evaluates paper relevance against user query and domain requirements. "
                "Scores papers on a 0-10 scale with reasoning. Considers method novelty, "
                "dataset relevance, and reproducibility (code availability)."
            ),
            tools=["Read", "evaluate_paper_relevance"],
            prompt=EVALUATOR_SYSTEM_PROMPT,
            model="sonnet",
        ),
    }


def build_migration_agents(context: AgentContext) -> Dict[str, AgentDefinition]:
    """Build migration stage agents.

    Returns agents for repository cloning, model adaptation, config migration,
    testing, and dataset migration.
    """
    from agents.migration.prompts.cloner import CLONER_SYSTEM_PROMPT
    from agents.migration.prompts.adapter import ADAPTER_SYSTEM_PROMPT
    from agents.migration.prompts.config import CONFIG_SYSTEM_PROMPT
    from agents.migration.prompts.tester import TESTER_SYSTEM_PROMPT
    from agents.migration.prompts.dataset_downloader import DATASET_DOWNLOADER_SYSTEM_PROMPT
    from agents.migration.prompts.dataset_converter import DATASET_CONVERTER_SYSTEM_PROMPT

    return {
        "repo-cloner": AgentDefinition(
            name="repo-cloner",
            description=(
                "Clones external repositories and inspects their structure for migration. "
                "Identifies key files: model definitions, training scripts, configs, and "
                "dependencies. Clones to ./repos/<model-name> directory."
            ),
            tools=["Bash", "Glob", "Read"],
            prompt=CLONER_SYSTEM_PROMPT,
            model="sonnet",
        ),
        "model-adapter": AgentDefinition(
            name="model-adapter",
            description=(
                "Adapts PyTorch models to LibCity conventions. Handles inheritance from "
                "AbstractTrafficStateModel, forward signature alignment, and data loader "
                f"integration. Reference LibCity at {context.repo_path}."
            ),
            tools=["Read", "Write", "Edit", "Glob", "Grep"],
            prompt=ADAPTER_SYSTEM_PROMPT,
            model="opus",
        ),
        "config-migrator": AgentDefinition(
            name="config-migrator",
            description=(
                "Creates and updates LibCity configuration files for migrated models. "
                "Handles model config JSON, task_config.json registration, and "
                "hyperparameter defaults from original papers."
            ),
            tools=["Read", "Write", "Edit", "Glob"],
            prompt=CONFIG_SYSTEM_PROMPT,
            model="sonnet",
        ),
        "migration-tester": AgentDefinition(
            name="migration-tester",
            description=(
                "Runs LibCity benchmarks and captures test results for migrated models. "
                "Uses test_migration tool to execute training/evaluation, analyzes errors, "
                "and reports metrics when successful."
            ),
            tools=["test_migration", "Bash", "Read"],
            prompt=TESTER_SYSTEM_PROMPT,
            model="opus",
        ),
        "dataset-downloader": AgentDefinition(
            name="dataset-downloader",
            description=(
                "Downloads external datasets from various sources (direct links, GitHub, "
                "Google Drive, Kaggle, Zenodo). Extracts and analyzes dataset structure. "
                "Saves to ./datasets/<dataset-name>/ directory."
            ),
            tools=["Bash", "Glob", "Read", "WebFetch"],
            prompt=DATASET_DOWNLOADER_SYSTEM_PROMPT,
            model="sonnet",
        ),
        "dataset-converter": AgentDefinition(
            name="dataset-converter",
            description=(
                "Converts external datasets to LibCity atomic file format (.geo, .rel, .dyna, "
                ".usr, config.json). Creates conversion scripts in preprocess/ directory. "
                f"Saves converted data to {context.repo_path}/raw_data/<dataset-name>/."
            ),
            tools=["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
            prompt=DATASET_CONVERTER_SYSTEM_PROMPT,
            model="opus",
        ),
    }


def build_tuning_agents(context: AgentContext) -> Dict[str, AgentDefinition]:
    """Build tuning stage agents.

    Returns agents for tuning planning, execution, and result analysis.
    """
    from agents.tuning.prompts.planner import PLANNER_SYSTEM_PROMPT
    from agents.tuning.prompts.executor import EXECUTOR_SYSTEM_PROMPT
    from agents.tuning.prompts.analyzer import ANALYZER_SYSTEM_PROMPT

    return {
        "tuning-planner": AgentDefinition(
            name="tuning-planner",
            description=(
                "Analyzes model configs and previous runs to design hyperparameter "
                "search spaces. Considers learning rate, batch size, hidden dimensions, "
                "and references original paper hyperparameters."
            ),
            tools=["Read", "Glob", "Grep"],
            prompt=PLANNER_SYSTEM_PROMPT,
            model="opus",
        ),
        "tuning-executor": AgentDefinition(
            name="tuning-executor",
            description=(
                "Executes hyperparameter tuning using LibCity run_hyper.py or grid search. "
                "Calls tune_migration_model with provided search space and tracks trial "
                "progress and best results."
            ),
            tools=["tune_migration_model", "Bash", "Read"],
            prompt=EXECUTOR_SYSTEM_PROMPT,
            model="sonnet",
        ),
        "result-analyzer": AgentDefinition(
            name="result-analyzer",
            description=(
                "Analyzes tuning results and generates summary reports. Compares with "
                "paper-reported metrics, identifies best hyperparameters, and provides "
                "recommendations for next steps."
            ),
            tools=["Read", "Write", "Glob"],
            prompt=ANALYZER_SYSTEM_PROMPT,
            model="opus",
        ),
    }


__all__ = [
    "build_literature_agents",
    "build_migration_agents",
    "build_tuning_agents",
]
