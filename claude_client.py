import asyncio
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import warnings
import requests
from claude_agent_sdk import tool

from agents import AppPaths, setup_logger
from agents.core.orchestrator import AgentOrchestrator  # noqa: E501
from agents.migration import MigrationCatalog, create_migration_stage_callback
from agents.migration.runtime import get_active_paper
from agents.migration.storage import MigrationResultStore
from agents.migration.utils import extract_standard_metrics
from agents.tuning.tuning import (
    ensure_search_space,
    is_libcity_tuning_allowed,
    run_custom_grid_search,
    run_libcity_tuning,
    write_params_file,
)

paths = AppPaths()
paths.ensure()

ARTICLE_CATALOG_PATH = paths.article_catalog
MIGRATION_RESULTS_PATH = paths.run_dir / "migration_results.json"
migration_catalog = MigrationCatalog(
    catalog_path=paths.migration_catalog,
    documentation_dir=paths.documentation_dir,
    root=paths.root,
)
migration_store = MigrationResultStore(
    root=paths.root, base_dir=paths.run_dir / "migrations", index_path=MIGRATION_RESULTS_PATH
)
stage_callback = create_migration_stage_callback(migration_catalog)


def _as_bool(value, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _as_int(value, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_project_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = paths.root / candidate
    return candidate


def _append_article_record(record: dict) -> None:
    ARTICLE_CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing: List[dict] = json.loads(ARTICLE_CATALOG_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []

    # Avoid duplicates by title + conference
    signature = (record.get("title"), record.get("conference"))
    for item in existing:
        if (item.get("title"), item.get("conference")) == signature:
            item.update(record)
            ARTICLE_CATALOG_PATH.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return

    existing.append(record)
    ARTICLE_CATALOG_PATH.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
    )

def _truncate_text(value: str, limit: int = 12000) -> str:
    text = (value or "").strip()
    if len(text) > limit:
        return f"{text[:limit]}...[truncated]"
    return text


@tool(
    "catalog_article",
    "Persist metadata about a located research paper and its saved PDF",
    {
        "title": str,
        "conference": str,
        "datasets": str,
        "repo_url": str,
        "pdf_path": str,
        "notes": str,
        "model_name": str,
    },
)
async def catalog_article(args):
    record = {
        "title": args.get("title", "").strip(),
        "conference": args.get("conference", "").strip(),
        "datasets": args.get("datasets", ""),
        "repo_url": args.get("repo_url", ""),
        "repo_path": args.get("repo_path", ""),
        "pdf_path": args.get("pdf_path", ""),
        "notes": args.get("notes", ""),
        "model_name": args.get("model_name", ""),
    }
    if not record["title"]:
        return {
            "content": [
                {
                    "type": "text",
                    "text": "catalog_article failed: title is required.",
                }
            ]
        }

    _append_article_record(record)
    return {
        "content": [
            {
                "type": "text",
                "text": f"Catalog updated for {record['title']}",
            }
        ]
    }


def _fetch_arxiv_metadata(arxiv_id: str) -> Dict[str, Any] | None:
    """Fetch paper metadata from arXiv API."""
    try:
        api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        resp = requests.get(api_url, timeout=10)
        if resp.status_code != 200:
            return None
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None
        title_el = entry.find("atom:title", ns)
        abstract_el = entry.find("atom:summary", ns)
        published_el = entry.find("atom:published", ns)
        title = title_el.text.strip().replace("\n", " ") if title_el is not None else ""
        abstract = abstract_el.text.strip().replace("\n", " ") if abstract_el is not None else ""
        published = published_el.text.strip() if published_el is not None else ""
        year = None
        if published:
            year_match = re.match(r"(\d{4})", published)
            if year_match:
                year = int(year_match.group(1))
        return {
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "year": year,
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        }
    except Exception:
        return None


@tool(
    "search_paper",
    "Search for academic papers on arXiv using Google Search API. Returns paper metadata including title, abstract, arxiv_id, and URLs.",
    {
        "query": str,
        "num": int,
        "end_date": str,
    },
)
async def search_paper(args):
    """Search for papers and return detailed metadata."""
    url = "https://google.serper.dev/search"
    query = args.get("query")
    num = args.get("num", 10)
    end_date = args.get("end_date")

    if not query:
        return {
            "content": [{"type": "text", "text": "search_paper failed: query is required"}]
        }

    search_query = f"{query} site:arxiv.org"
    if end_date:
        try:
            parsed_date = datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d')
            search_query = f"{query} before:{parsed_date} site:arxiv.org"
        except Exception:
            pass

    payload = json.dumps({
        "q": search_query,
        "num": num,
        "page": 1,
    })

    headers = {
        'X-API-KEY': "1163e00449dce84869048401eccca059865553ff",
        'Content-Type': 'application/json'
    }

    papers = []
    seen_ids = set()

    for _ in range(3):
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            if response.status_code == 200:
                results = json.loads(response.text)
                for paper in results.get('organic', []):
                    link = paper.get("link", "")
                    match = re.search(r'arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)', link)
                    if match:
                        arxiv_id = match.group(1)
                        if arxiv_id in seen_ids:
                            continue
                        seen_ids.add(arxiv_id)
                        metadata = _fetch_arxiv_metadata(arxiv_id)
                        if metadata:
                            papers.append(metadata)
                        else:
                            papers.append({
                                "arxiv_id": arxiv_id,
                                "title": paper.get("title", ""),
                                "abstract": paper.get("snippet", ""),
                                "year": None,
                                "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
                                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                            })
                break
        except Exception as exc:
            warnings.warn(f"search_paper failed: {exc}")
            continue

    if not papers:
        return {
            "content": [{"type": "text", "text": f"No papers found for query: {query}"}]
        }

    result_text = json.dumps(papers, ensure_ascii=False, indent=2)
    return {
        "content": [{"type": "text", "text": f"Found {len(papers)} papers:\n{result_text}"}]
    }


@tool(
    "evaluate_paper_relevance",
    "Evaluate whether a paper is relevant to the user query based on title and abstract.",
    {
        "title": str,
        "abstract": str,
        "user_query": str,
    },
)
async def evaluate_paper_relevance(args):
    """Placeholder tool for paper relevance evaluation. Claude should use its own judgment."""
    title = args.get("title", "")
    abstract = args.get("abstract", "")
    user_query = args.get("user_query", "")

    return {
        "content": [{
            "type": "text",
            "text": f"Paper evaluation request received.\nTitle: {title}\nAbstract: {abstract[:500]}...\nQuery: {user_query}\n\nPlease use your judgment to evaluate this paper's relevance."
        }]
    }

@tool(
    "test_migration",
    "Run the LibCity benchmark with the provided model and dataset (supports GPU acceleration)",
    {"model_name": str, "dataset": str, "task": str, "paper_title": str, "gpu": str},
)
async def test_migration(args):
    model_name = args.get("model_name")
    dataset = args.get("dataset", "METR_LA")
    task = args.get("task", "traffic_state_pred")
    gpu = args.get("gpu", "0")  # Default to cuda:0
    if not model_name:
        return {
            "content": [
                {
                    "type": "text",
                    "text": "test_migration failed: provide model_name",
                }
            ]
        }

    repo_dir = Path("Bigscity-LibCity")
    cmd = [
        "python",
        "run_model.py",
        "--task",
        task,
        "--model",
        model_name,
        "--dataset",
        dataset,
        "--gpu_id",
        gpu,
    ]
    stdout = ""
    stderr = ""
    status = "success"
    try:
        result = subprocess.run(
            cmd, cwd=repo_dir, capture_output=True, text=True, check=True
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        output = stdout or "LibCity run completed without stdout output."
    except subprocess.CalledProcessError as exc:
        status = "error"
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        output = "\n".join(
            [
                "LibCity execution failed.",
                f"Command: {' '.join(cmd)}",
                f"Return code: {exc.returncode}",
                stdout,
                stderr,
            ]
        )
    metrics = extract_standard_metrics(stdout)
    active_paper = get_active_paper() or {}
    paper_info = {
        "title": active_paper.get("title") or args.get("paper_title") or model_name,
        "conference": active_paper.get("conference") or args.get("conference") or "N/A",
        "repo_url": active_paper.get("repo_url") or "",
        "pdf_path": active_paper.get("pdf_path") or "",
        "model_name": active_paper.get("model_name") or model_name,
    }
    migration_store.record_run(
        paper_info,
        {
            "model_name": model_name,
            "dataset": dataset,
            "paper_title": paper_info["title"],
            "conference": paper_info["conference"],
            "status": status,
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
    )
    return {
        "content": [
            {
                "type": "text",
                "text": _truncate_text(output),
            }
        ]
    }


@tool(
    "tune_migration_model",
    "Hyperparameter tuning for migrated LibCity models (supports GPU acceleration)",
    {
        "model_name": str,
        "dataset": str,
        "config_file": str,
        "task": str,
        "params_file": str,
        "search_space": dict,
        "hyper_algo": str,
        "max_evals": int,
        "max_trials": int,
        "saved_model": bool,
        "train": bool,
        "gpu": str,
        "extra_cli_args": dict,
    },
)
async def tune_migration_model(args):
    model_name = args.get("model_name")
    if not model_name:
        return {
            "content": [
                {
                    "type": "text",
                    "text": "tune_migration_model failed: provide model_name",
                }
            ]
        }

    dataset = args.get("dataset", "PEMSD8")
    task = args.get("task", "traffic_state_pred")
    gpu = args.get("gpu", "0")  # Default to cuda:0
    config_file = args.get("config_file")
    hyper_algo = args.get("hyper_algo")
    raw_search_space = args.get("search_space")
    if not isinstance(raw_search_space, dict):
        raw_search_space = None
    search_space = ensure_search_space(raw_search_space)
    params_file_arg = _resolve_project_path(args.get("params_file"))
    max_evals = _as_int(args.get("max_evals"))
    max_trials = _as_int(args.get("max_trials"))
    if max_evals is not None and max_evals <= 0:
        max_evals = None
    if max_trials is not None and max_trials <= 0:
        max_trials = None
    saved_model = _as_bool(args.get("saved_model"))
    train_flag = _as_bool(args.get("train"))
    extra_cli_args = args.get("extra_cli_args")
    cli_args: Dict[str, Any] = extra_cli_args if isinstance(extra_cli_args, dict) else {}
    repo_dir = Path("Bigscity-LibCity")

    libcity_env_enabled = is_libcity_tuning_allowed()
    result: Dict[str, Any] | None = None
    libcity_attempted = False
    libcity_failure = ""
    libcity_result: Dict[str, Any] | None = None

    if libcity_env_enabled:
        libcity_attempted = True
        try:
            params_file = (
                params_file_arg if params_file_arg and params_file_arg.exists() else write_params_file(paths.run_dir, search_space)
            )
            libcity_result = run_libcity_tuning(
                repo_dir=repo_dir,
                run_dir=paths.run_dir,
                model_name=model_name,
                dataset_name=dataset,
                params_file=params_file,
                task=task,
                config_file=config_file,
                hyper_algo=hyper_algo,
                max_evals=max_evals,
                saved_model=saved_model,
                train=train_flag,
                gpu_id=gpu,
                extra_cli_args=cli_args,
            )
            if libcity_result.get("status") == "success":
                result = libcity_result
            else:
                libcity_failure = f"LibCity tuner returned {libcity_result.get('status')}"
        except Exception as exc:
            libcity_failure = f"LibCity tuner failed: {exc}"

    if result is None:
        try:
            result = run_custom_grid_search(
                repo_dir=repo_dir,
                model_name=model_name,
                dataset_name=dataset,
                search_space=search_space,
                task=task,
                config_file=config_file,
                max_trials=max_trials,
                gpu_id=gpu,
            )
        except Exception as exc:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"tune_migration_model encountered an error: {exc}",
                    }
                ]
            }

    metrics = result.get("best_metrics") or {}
    paper = get_active_paper() or {}
    paper_info = {
        "title": paper.get("title") or args.get("paper_title") or model_name,
        "conference": paper.get("conference") or args.get("conference") or "N/A",
        "repo_url": paper.get("repo_url") or "",
        "pdf_path": paper.get("pdf_path") or "",
        "model_name": paper.get("model_name") or model_name,
    }
    tuning_payload: Dict[str, Any] = {
        "strategy": result.get("strategy"),
        "best_params": result.get("best_params", {}),
        "artifacts": result.get("artifacts") or {},
    }
    if result.get("trials"):
        tuning_payload["trials"] = result["trials"]

    migration_store.record_run(
        paper_info,
        {
            "model_name": model_name,
            "dataset": dataset,
            "paper_title": paper_info["title"],
            "conference": paper_info["conference"],
            "status": result.get("status", "error"),
            "stdout": (result.get("stdout") or "").strip(),
            "stderr": (result.get("stderr") or "").strip(),
            "metrics": metrics,
            "tuning": tuning_payload,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
    )

    summary_lines = [
        f"Tuning strategy: {result.get('strategy')}",
        f"Status: {result.get('status')}",
    ]
    if result.get("strategy") == "libcity":
        summary_lines.append("LibCity run_hyper.py executed successfully.")
    elif libcity_attempted:
        if libcity_failure:
            summary_lines.append(f"LibCity run_hyper.py skipped: {libcity_failure}. Used custom grid search instead.")
    elif not libcity_env_enabled:
        summary_lines.append("LibCity run_hyper.py disabled by environment; used custom grid search.")
    if result.get("best_params"):
        summary_lines.append(f"Best parameters: {json.dumps(result['best_params'], ensure_ascii=False)}")
    if metrics:
        metric_text = ", ".join(f"{key}: {value:.4f}" for key, value in metrics.items())
        summary_lines.append(f"Best metrics: {metric_text}")
    artifacts = result.get("artifacts") or {}
    if artifacts.get("hyper_result"):
        summary_lines.append(f"Hyper-result saved to: {artifacts['hyper_result']}")
    if result.get("trials"):
        summary_lines.append(f"Trials run: {len(result['trials'])}")
    return {
        "content": [
            {
                "type": "text",
                "text": _truncate_text("\n".join(summary_lines)),
            }
        ]
    }


async def main():
    logger = setup_logger(paths.log_file)
    orchestrator = AgentOrchestrator(
        paths=paths,
        logger=logger,
        stage_callback=stage_callback,
    )
    await orchestrator.run_full_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
