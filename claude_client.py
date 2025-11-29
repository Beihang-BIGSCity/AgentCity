import asyncio
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List
import warnings
import requests
from claude_agent_sdk import tool

from agents import AppPaths, setup_logger
from agents.core.orchestrator import AgentOrchestrator  # noqa: E501
from agents.migration import MigrationCatalog, create_migration_stage_callback
from agents.migration.runtime import get_active_paper
from agents.migration.storage import MigrationResultStore

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


def _extract_metrics(stdout: str) -> dict:
    metrics = {}
    if not stdout:
        return metrics
    for key in ("masked_mae", "masked_mape", "masked_mase"):
        pattern = rf"{key}\s*[:=]\s*([0-9.]+)"
        match = re.search(pattern, stdout, flags=re.IGNORECASE)
        if match:
            metrics[key.lower()] = float(match.group(1))
    return metrics


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


@tool(
    "google_search_arxiv_id",
    "search for papers",
    {"query": str, "num": int},
)
async def google_search_arxiv_id(args):
    url = "https://google.serper.dev/search"
    query = args.get("query")
    num = args.get("num", 10)
    search_query = f"{query} site:arxiv.org"
    if end_date:
        try:
            end_date = datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d')
            search_query = f"{query} before:{end_date} site:arxiv.org"
        except:
            search_query = f"{query} site:arxiv.org"
    
    payload = json.dumps({
        "q": search_query, 
        "num": num, 
        "page": 1, 
    })

    headers = {
        'X-API-KEY': "1163e00449dce84869048401eccca059865553ff",
        'Content-Type': 'application/json'
    }
    assert headers['X-API-KEY'] != 'your google keys', "add your google search key!!!"

    for _ in range(3):
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            print(response,payload)
            if response.status_code == 200:
                results = json.loads(response.text)
                arxiv_id_list = []
                for paper in results['organic']:
                    if re.search(r'arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)', paper["link"]):
                        arxiv_id = re.search(r'arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)', paper["link"]).group(1)
                        arxiv_id_list.append(arxiv_id)
                return list(set(arxiv_id_list))
        except:
            warnings.warn(f"google search failed, query: {query}")
            continue
    return []

@tool(
    "test_migration",
    "Run the LibCity benchmark with the provided model and dataset",
    {"model_name": str, "dataset": str, "paper_title": str},
)
async def test_migration(args):
    model_name = args.get("model_name")
    dataset = args.get("dataset", "PEMSD8")
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
        "conda",
        "run",
        "-n",
        "autogen",
        "python",
        "run_model.py",
        "--task",
        "traffic_state_pred",
        "--model",
        model_name,
        "--dataset",
        dataset,
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
    metrics = _extract_metrics(stdout)
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
                "text": output,
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
