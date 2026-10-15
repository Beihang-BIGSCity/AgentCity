"""AgentCity business tools exposed as a FastMCP (stdio) server.

Mounted on openjiuwen DeepAgent roles via ``McpServerConfig(client_type="stdio",
params={"command": python, "args": ["-m", "agents.tools.server"], "cwd": <repo root>})``.

Run directly: ``python -m agents.tools.server``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from agents.tools import impl

mcp = FastMCP("agentcity-tools")


@mcp.tool
async def analyze_dataset(
    dataset: Optional[str] = None,
    task: Optional[str] = None,
    windows: int = 10,
    max_timesteps: int = 20000,
) -> str:
    """Compute CVglobal/CVnode/spatial heterogeneity/lag-1 autocorrelation/sampling interval
    for a LibCity dataset and write dataset_profile.json into the task directory."""

    return await impl.analyze_dataset(
        {"dataset": dataset, "task": task, "windows": windows, "max_timesteps": max_timesteps}
    )


@mcp.tool
async def retrieve_knowledge(
    features: Optional[Dict[str, Any]] = None,
    top_k: int = 3,
    kinds: Optional[List[str]] = None,
    include_documents: bool = False,
) -> str:
    """Retrieve the closest "scenario feature -> structural/tuning strategy" knowledge entries
    and Agent Skills for a data feature profile (deterministic nearest neighbour)."""

    return await impl.retrieve_knowledge(
        {
            "features": features,
            "top_k": top_k,
            "kinds": kinds,
            "include_documents": include_documents,
        }
    )


@mcp.tool
async def record_experience(
    kind: str,
    id: Optional[str] = None,
    title: Optional[str] = None,
    family: Optional[str] = None,
    dimension: Optional[str] = None,
    feature_profile: Optional[Dict[str, Any]] = None,
    strategy: Optional[Dict[str, Any]] = None,
    tuning_direction: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
    evidence: Optional[List[Any]] = None,
    source: Optional[str] = None,
) -> str:
    """Feed a finished round back into the expert knowledge base (kind="knowledge")
    or the tuning skills (kind="skill"); failures are recorded as evidence too.

    kind="knowledge" stores a "scenario feature -> structural strategy" prior: pass `feature_profile` with the
    numeric retrieval features (cv_global, cv_node, spatial_heterogeneity, rho1, sampling_interval_min, node_count)
    and `strategy` as the five module families (encoder/spatial/temporal/fusion/head) mapped to the module ids that
    ran - composition.json's `modules` block already has that shape.

    kind="skill" stores a "data feature profile -> tuning direction" lesson and needs a numeric `feature_profile`
    plus at least one `tuning_direction`.

    The store normalises and validates what it receives (it keeps only the retrieval features and never replaces a
    hand-authored entry), so a rejected call reports the missing part and should be retried with the measured
    composition instead of being dropped."""

    entry = {
        key: value
        for key, value in {
            "id": id,
            "title": title,
            "family": family,
            "dimension": dimension,
            "feature_profile": feature_profile,
            "strategy": strategy,
            "tuning_direction": tuning_direction,
            "strategies": strategies,
            "evidence": evidence,
            "source": source,
        }.items()
        if value is not None
    }
    return await impl.record_experience({"kind": kind, "entry": entry})


@mcp.tool
async def list_modules(
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    include_code: bool = False,
) -> str:
    """List atomic modules from the module library (encoder/spatial/temporal/fusion/head)."""

    return await impl.list_modules(
        {"category": category, "tags": tags, "include_code": include_code}
    )


@mcp.tool
async def get_module(module_id: str, include_code: bool = True) -> str:
    """Return one atomic module entry with its source code and unified interface."""

    return await impl.get_module({"module_id": module_id, "include_code": include_code})


@mcp.tool
async def register_module(
    module_id: str,
    category: str,
    class_name: str,
    code: str,
    name: Optional[str] = None,
    desc: str = "",
    ctr_args: Optional[Dict[str, Any]] = None,
    io: str = "",
    tags: Optional[List[str]] = None,
    source: Optional[Dict[str, Any]] = None,
) -> str:
    """Deposit a newly extracted atomic module into the library (idempotent by module_id)."""

    return await impl.register_module(
        {
            "module_id": module_id,
            "category": category,
            "class_name": class_name,
            "code": code,
            "name": name,
            "desc": desc,
            "ctr_args": ctr_args,
            "io": io,
            "tags": tags,
            "source": source,
        }
    )


@mcp.tool
async def run_adaptation(
    model_name: str,
    dataset: Optional[str] = None,
    task: Optional[str] = None,
    max_epoch: Optional[int] = None,
    batch_size: Optional[int] = None,
    config_file: Optional[str] = None,
    gpu: bool = False,
    gpu_id: str = "0",
    seed: Optional[int] = None,
    exp_id: Optional[str] = None,
    saved_model: Optional[bool] = None,
    train: Optional[bool] = None,
    extra_cli_args: Optional[Dict[str, Any]] = None,
) -> str:
    """Run the composed model through LibCity run_model.py (CPU by default) and record the metrics."""

    return await impl.run_adaptation(
        {
            "model_name": model_name,
            "dataset": dataset,
            "task": task,
            "max_epoch": max_epoch,
            "batch_size": batch_size,
            "config_file": config_file,
            "gpu": gpu,
            "gpu_id": gpu_id,
            "seed": seed,
            "exp_id": exp_id,
            "saved_model": saved_model,
            "train": train,
            "extra_cli_args": extra_cli_args,
        }
    )


@mcp.tool
async def tune_adaptation(
    model_name: str,
    dataset: Optional[str] = None,
    task: Optional[str] = None,
    search_space: Optional[Dict[str, List[Any]]] = None,
    max_trials: Optional[int] = None,
    max_evals: Optional[int] = None,
    max_epoch: Optional[int] = None,
    batch_size: Optional[int] = None,
    config_file: Optional[str] = None,
    gpu: bool = False,
    gpu_id: str = "0",
    params_file: Optional[str] = None,
    hyper_algo: Optional[str] = None,
    extra_cli_args: Optional[Dict[str, Any]] = None,
) -> str:
    """Search hyperparameters for the composed model (LibCity run_hyper.py, falling back to a
    custom grid search) and persist the trials and the best configuration."""

    return await impl.tune_adaptation(
        {
            "model_name": model_name,
            "dataset": dataset,
            "task": task,
            "search_space": search_space,
            "max_trials": max_trials,
            "max_evals": max_evals,
            "max_epoch": max_epoch,
            "batch_size": batch_size,
            "config_file": config_file,
            "gpu": gpu,
            "gpu_id": gpu_id,
            "params_file": params_file,
            "hyper_algo": hyper_algo,
            # The tuning prompts tell the agent to forward the task's
            # train.extra_args so every trial keeps the CPU budget; the schema
            # must accept it or the call is rejected before it reaches impl.
            "extra_cli_args": extra_cli_args,
        }
    )


@mcp.tool
async def web_search(query: str, num: int = 10) -> str:
    """Search the web via the Serper Google Search API."""

    return await impl.web_search({"query": query, "num": num})


@mcp.tool
async def web_fetch(url: str, max_chars: int = 20000) -> str:
    """Fetch a URL and return its plain-text content."""

    return await impl.web_fetch({"url": url, "max_chars": max_chars})


@mcp.tool
async def read_pdf_text(pdf_path: str, max_pages: int = 40, max_chars: int = 60000) -> str:
    """Extract plain text from a PDF file."""

    return await impl.read_pdf_text(
        {"pdf_path": pdf_path, "max_pages": max_pages, "max_chars": max_chars}
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
