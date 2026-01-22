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
    year_mode: str = Field(default="all")  # 默认不过滤年份
    conferences: List[str] = Field(default_factory=list)


class MigrationRequest(BaseModel):
    paper_ids: List[str] = Field(default_factory=list)


class TuningRequest(BaseModel):
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

def _truncate_text(value: str | None, limit: int) -> str:
    text = (value or "").strip()
    if len(text) > limit:
        return f"{text[:limit]}...[truncated]"
    return text

def _trim_stage_messages(messages, max_messages: int = 200, max_text: int = 2000, max_tool_input: int = 1000):
    trimmed = []
    if not isinstance(messages, list):
        return trimmed
    for message in messages[-max_messages:]:
        if not isinstance(message, dict):
            continue
        entry = dict(message)
        if entry.get("type") == "text":
            entry["content"] = _truncate_text(entry.get("content"), max_text)
        elif entry.get("type") == "tool_use":
            payload = entry.get("tool_input")
            try:
                serialized = json.dumps(payload, ensure_ascii=False)
            except TypeError:
                serialized = str(payload)
            entry["tool_input"] = _truncate_text(serialized, max_tool_input)
        trimmed.append(entry)
    return trimmed

def _sanitize_stage_payload(payload: Dict[str, object]) -> Dict[str, object]:
    if not isinstance(payload, dict):
        return {"stages": []}
    stages = payload.get("stages")
    if not isinstance(stages, list):
        return {"stages": []}
    sanitized = []
    for entry in stages:
        if not isinstance(entry, dict):
            continue
        item = dict(entry)
        item["messages"] = _trim_stage_messages(item.get("messages"))
        item["summary"] = _truncate_text(item.get("summary"), 2000)
        item["prompt"] = _truncate_text(item.get("prompt"), 2000)
        sanitized.append(item)
    return {"stages": sanitized[-200:]}


def _article_index() -> Dict[str, Dict[str, object]]:
    index: Dict[str, Dict[str, object]] = {}
    for entry in _load_articles():
        signature = build_paper_signature(entry)
        index[signature] = entry
    return index


def _resolve_selected_papers(paper_ids: List[str]) -> List[Dict[str, object]]:
    ids = [pid for pid in paper_ids if pid]
    if not ids:
        raise HTTPException(status_code=400, detail="paper_ids are required")

    index = _article_index()
    missing = [pid for pid in ids if pid not in index]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown paper IDs: {', '.join(missing)}",
        )
    return [index[pid] for pid in ids]


def _normalize_year_mode(value: str | None) -> str:
    """Normalize year mode, supporting predefined modes and direct year/range values."""
    if not value:
        return "all"

    raw = value.strip()
    raw_lower = raw.lower()
    mapping = {
        "this_year": "this_year",
        "今年": "this_year",
        "current": "this_year",
        "last_year": "last_year",
        "去年": "last_year",
        "previous": "last_year",
        "all": "all",
        "全部": "all",
    }

    if raw_lower in mapping:
        return mapping[raw_lower]

    if "-" in raw:
        parts = raw.split("-")
        if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
            return raw
    if raw.isdigit():
        return raw
    return "all"


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
    return _sanitize_stage_payload(payload)


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

    year_mode = _normalize_year_mode(request.year_mode)

    conferences = [word.strip() for word in request.conferences if word.strip()]

    async def runner():
        await orchestrator.run_literature(
            search_terms=keywords,
            year_filter=year_mode,
            conference_filters=conferences,
        )

    job = await job_manager.enqueue(
        label="Literature Search",
        coro_factory=runner,
        metadata={
            "keywords": keywords,
            "year_mode": year_mode,
            "conferences": conferences,
        },
    )
    return {"items": [job]}


@app.post("/api/tuning/run")
async def enqueue_tuning_job(request: TuningRequest):
    paper_ids = [pid for pid in request.paper_ids if pid]
    selected = _resolve_selected_papers(paper_ids)

    async def runner():
        await orchestrator.run_tuning_sequence(selected)

    primary = selected[0]
    label_suffix = primary.get("title") or "Tuning"
    if len(selected) > 1:
        label_suffix = f"{label_suffix} +{len(selected) - 1}"
    job = await job_manager.enqueue(
        label=f"Tuning: {label_suffix}",
        coro_factory=runner,
        metadata={
            "paper_title": primary.get("title"),
            "model_name": primary.get("model_name"),
            "paper_ids": paper_ids,
        },
    )
    return {"items": [job]}

@app.post("/api/migration/run")
async def enqueue_migration_job(request: MigrationRequest):
    paper_ids = [pid for pid in request.paper_ids if pid]
    selected = _resolve_selected_papers(paper_ids)

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
