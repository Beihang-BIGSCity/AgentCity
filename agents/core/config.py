from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

from .types import AgentContext

DEFAULT_CONFERENCES: List[str] = ["ICLR", "ICML", "NeurIPS", "KDD", "ICDE", "WWW", "AAAI", "IJCAI"]
TARGET_YEAR = 2025
YEAR_FILTER_CHOICES = {
    "this_year": {"label": "今年", "offset": 0},
    "last_year": {"label": "去年", "offset": -1},
    "all": {"label": "全部", "offset": None},
}


@dataclass
class AppPaths:
    """Centralizes important repository paths for the agent."""

    root: Path = Path(__file__).resolve().parents[2]
    repo_path: Path = Path("./Bigscity-LibCity")
    data_dir: Path = root / "data"
    article_dir: Path = data_dir / "articles"
    article_catalog: Path = article_dir / "catalog.json"
    migration_dir: Path = data_dir / "migrations"
    migration_catalog: Path = migration_dir / "catalog.json"
    run_dir: Path = data_dir / "runs"
    stage_log: Path = run_dir / "latest_stage_log.json"
    log_file: Path = root / "claude_code.log"
    benchmark_doc: Path = root / "libcity_document.md"
    documentation_dir: Path = root / "documentation"

    def ensure(self) -> None:
        for folder in (
            self.data_dir,
            self.article_dir,
            self.migration_dir,
            self.run_dir,
            self.documentation_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)
        if not self.article_catalog.exists():
            self.article_catalog.write_text("[]\n", encoding="utf-8")
        if not self.migration_catalog.exists():
            self.migration_catalog.write_text("[]\n", encoding="utf-8")
        if not self.stage_log.exists():
            self.stage_log.write_text('{"stages": []}\n', encoding="utf-8")


def build_default_context(paths: AppPaths) -> AgentContext:
    try:
        benchmark_document = paths.benchmark_doc.read_text(encoding="utf-8")
    except FileNotFoundError:
        benchmark_document = ""

    mode, label, value = resolve_year_filter()
    return AgentContext(
        repo_path=str(paths.repo_path),
        benchmark_document=benchmark_document,
        article_dir=str(paths.article_dir),
        article_catalog=str(paths.article_catalog),
        migration_catalog=str(paths.migration_catalog),
        conferences=DEFAULT_CONFERENCES,
        target_year=TARGET_YEAR,
        search_terms=[],
        selected_papers=[],
        search_year_mode=mode,
        search_year_label=label,
        search_year_value=value,
        search_conference_filters=[],
    )


def resolve_year_filter(mode: str | None = None) -> tuple[str, str, int | None]:
    normalized = (mode or "all").strip().lower()
    if normalized not in YEAR_FILTER_CHOICES:
        normalized = "all"
    choice = YEAR_FILTER_CHOICES[normalized]
    label = choice["label"]
    offset = choice["offset"]
    if offset is None:
        value = None
    else:
        value = datetime.utcnow().year + offset
    return normalized, label, value


__all__ = [
    "AppPaths",
    "build_default_context",
    "DEFAULT_CONFERENCES",
    "TARGET_YEAR",
    "resolve_year_filter",
    "YEAR_FILTER_CHOICES",
]
