from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents import AppPaths, setup_logger
from agents.core.jobs import JobManager
from agents.core.orchestrator import AgentOrchestrator
from agents.migration import MigrationCatalog, create_migration_stage_callback
from agents.migration.utils import build_paper_signature


class LiteratureRequest(BaseModel):
    keywords: List[str] = Field(default_factory=list)


class MigrationRequest(BaseModel):
    paper_ids: List[str] = Field(default_factory=list)


paths = AppPaths()
paths.ensure()
logger = setup_logger(paths.log_file)
migration_catalog = MigrationCatalog(
    catalog_path=paths.migration_catalog,
    documentation_dir=paths.documentation_dir,
    root=paths.root,
)
stage_callback = create_migration_stage_callback(migration_catalog)
orchestrator = AgentOrchestrator(
    paths=paths,
    logger=logger,
    stage_callback=stage_callback,
)
job_manager = JobManager()

STAGE_LOG_PATH = paths.stage_log
ARTICLE_CATALOG_PATH = paths.article_catalog
MIGRATION_RESULTS_PATH = paths.run_dir / "migration_results.json"
FRONTEND_DIR = paths.root / "frontend"

app = FastAPI(title="LibCity Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
app.mount("/documentation", StaticFiles(directory=paths.documentation_dir), name="documentation")
app.mount("/data", StaticFiles(directory=paths.data_dir), name="data")


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        return default


def _load_articles() -> List[Dict[str, object]]:
    data = _read_json(ARTICLE_CATALOG_PATH, [])
    return data if isinstance(data, list) else []


def _article_index() -> Dict[str, Dict[str, object]]:
    index: Dict[str, Dict[str, object]] = {}
    for entry in _load_articles():
        signature = build_paper_signature(entry)
        index[signature] = entry
    return index


@app.get("/")
async def root():
    if FRONTEND_DIR.exists():
        return RedirectResponse(url="/frontend/index.html")
    return {"status": "ok"}


@app.get("/api/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/stages")
async def get_stages():
    payload = _read_json(STAGE_LOG_PATH, {"stages": []})
    if not isinstance(payload, dict):
        return {"stages": []}
    payload.setdefault("stages", [])
    return payload


@app.get("/api/articles")
async def get_articles():
    return {"items": _load_articles()}


@app.get("/api/migration/catalog")
async def get_migration_catalog():
    data = _read_json(paths.migration_catalog, [])
    items = data if isinstance(data, list) else []
    return {"items": items}


@app.get("/api/migration/results")
async def get_migration_results():
    data = _read_json(MIGRATION_RESULTS_PATH, [])
    items = data if isinstance(data, list) else []
    return {"items": items}


@app.get("/api/jobs")
async def list_jobs():
    jobs = await job_manager.list_jobs()
    return {"items": jobs}


@app.post("/api/literature/run")
async def enqueue_literature_job(request: LiteratureRequest):
    keywords = [word.strip() for word in request.keywords if word.strip()]
    if not keywords:
        raise HTTPException(status_code=400, detail="keywords are required")

    async def runner():
        await orchestrator.run_literature(search_terms=keywords)

    job = await job_manager.enqueue(
        label="Literature Search",
        coro_factory=runner,
        metadata={"keywords": keywords},
    )
    return {"items": [job]}


@app.post("/api/migration/run")
async def enqueue_migration_job(request: MigrationRequest):
    paper_ids = [pid for pid in request.paper_ids if pid]
    if not paper_ids:
        raise HTTPException(status_code=400, detail="paper_ids are required")

    index = _article_index()
    missing = [pid for pid in paper_ids if pid not in index]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown paper IDs: {', '.join(missing)}",
        )
    selected = [index[pid] for pid in paper_ids]

    async def runner():
        await orchestrator.run_migration_sequence(selected)

    primary = selected[0]
    label_suffix = primary.get("title") or "Migration"
    if len(selected) > 1:
        label_suffix = f"{label_suffix} +{len(selected) - 1}"
    job = await job_manager.enqueue(
        label=f"Migration: {label_suffix}",
        coro_factory=runner,
        metadata={
            "paper_title": primary.get("title"),
            "model_name": primary.get("model_name"),
            "paper_ids": paper_ids,
        },
    )
    return {"items": [job]}
