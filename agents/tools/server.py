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
async def catalog_article(
    title: str,
    conference: str = "",
    datasets: str = "",
    repo_url: str = "",
    repo_path: str = "",
    pdf_path: str = "",
    notes: str = "",
    model_name: str = "",
) -> str:
    """Record a paper's metadata (datasets, repo, pdf) into the article catalog."""
    return await impl.catalog_article({
        "title": title,
        "conference": conference,
        "datasets": datasets,
        "repo_url": repo_url,
        "repo_path": repo_path,
        "pdf_path": pdf_path,
        "notes": notes,
        "model_name": model_name,
    })


@mcp.tool
async def search_paper(query: str, num: int = 10, end_date: Optional[str] = None) -> str:
    """Search arXiv papers via Serper and return title/abstract/year/URLs metadata."""
    return await impl.search_paper({"query": query, "num": num, "end_date": end_date})


@mcp.tool
async def evaluate_paper_relevance(title: str, abstract: str, user_query: str) -> str:
    """Echo a paper evaluation request; the agent judges relevance itself."""
    return await impl.evaluate_paper_relevance({
        "title": title, "abstract": abstract, "user_query": user_query,
    })


@mcp.tool
async def test_migration(
    model_name: str,
    dataset: str = "METR_LA",
    task: str = "traffic_state_pred",
    gpu: str = "0",
    paper_title: Optional[str] = None,
    conference: Optional[str] = None,
) -> str:
    """Run LibCity training/evaluation for a migrated model and record the run."""
    return await impl.test_migration({
        "model_name": model_name,
        "dataset": dataset,
        "task": task,
        "gpu": gpu,
        "paper_title": paper_title,
        "conference": conference,
    })


@mcp.tool
async def tune_migration_model(
    model_name: str,
    dataset: str = "PEMSD8",
    task: str = "traffic_state_pred",
    gpu: str = "0",
    config_file: Optional[str] = None,
    hyper_algo: Optional[str] = None,
    search_space: Optional[Dict[str, List[Any]]] = None,
    params_file: Optional[str] = None,
    max_evals: Optional[int] = None,
    max_trials: Optional[int] = None,
    saved_model: Optional[bool] = None,
    train: Optional[bool] = None,
    extra_cli_args: Optional[Dict[str, Any]] = None,
    paper_title: Optional[str] = None,
    conference: Optional[str] = None,
) -> str:
    """Hyperparameter tuning for a migrated model (LibCity run_hyper or grid search)."""
    return await impl.tune_migration_model({
        "model_name": model_name,
        "dataset": dataset,
        "task": task,
        "gpu": gpu,
        "config_file": config_file,
        "hyper_algo": hyper_algo,
        "search_space": search_space,
        "params_file": params_file,
        "max_evals": max_evals,
        "max_trials": max_trials,
        "saved_model": saved_model,
        "train": train,
        "extra_cli_args": extra_cli_args,
        "paper_title": paper_title,
        "conference": conference,
    })


@mcp.tool
async def web_search(query: str, num: int = 10) -> str:
    """Search the web via the Serper Google Search API."""
    return await impl.web_search({"query": query, "num": num})


@mcp.tool
async def web_fetch(url: str, max_chars: int = 20000) -> str:
    """Fetch a URL and return plain-text content, truncated to max_chars."""
    return await impl.web_fetch({"url": url, "max_chars": max_chars})


@mcp.tool
async def read_pdf_text(pdf_path: str, max_pages: int = 40, max_chars: int = 60000) -> str:
    """Extract plain text from a PDF file (path relative to the project root or absolute)."""
    return await impl.read_pdf_text({
        "pdf_path": pdf_path, "max_pages": max_pages, "max_chars": max_chars,
    })


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
