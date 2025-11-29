from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional
from uuid import uuid4


JobCallable = Callable[[], Awaitable[None]]


class JobManager:
    """Simple in-memory tracker for long-running workflow executions."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def enqueue(
        self,
        label: str,
        coro_factory: JobCallable,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        job = await self._create_job(label, metadata or {})

        async def runner() -> None:
            await self._update_job(job["id"], status="running", started_at=_utc_now())
            try:
                await coro_factory()
            except Exception as exc:  # pragma: no cover - surfaced to job status
                await self._update_job(
                    job["id"],
                    status="failed",
                    finished_at=_utc_now(),
                    error=str(exc),
                )
                return
            await self._update_job(job["id"], status="succeeded", finished_at=_utc_now())

        asyncio.create_task(runner())
        return job

    async def _create_job(self, label: str, extra: Dict[str, Any]) -> Dict[str, Any]:
        async with self._lock:
            job_id = uuid4().hex
            job = {
                "id": job_id,
                "label": label,
                "status": "pending",
                "created_at": _utc_now(),
                "started_at": None,
                "finished_at": None,
                "error": None,
            }
            if extra:
                job.update(extra)
            self._jobs[job_id] = job
            return job

    async def _update_job(self, job_id: str, **updates: Any) -> None:
        async with self._lock:
            if job_id not in self._jobs:
                return
            self._jobs[job_id].update(updates)

    async def list_jobs(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return sorted(
                self._jobs.values(),
                key=lambda item: item.get("created_at") or "",
                reverse=True,
            )

    async def get_job(self, job_id: str) -> Dict[str, Any] | None:
        async with self._lock:
            return self._jobs.get(job_id)


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


__all__ = ["JobManager"]
