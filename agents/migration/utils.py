from __future__ import annotations

import re
from typing import Dict, Mapping, MutableMapping


def build_paper_signature(article: Mapping[str, object] | None) -> str:
    """Return a stable signature for a paper entry."""

    if not article:
        return "Unknown__N/A"
    title = str(article.get("title") or "Untitled").strip() or "Untitled"
    conference = str(article.get("conference") or "N/A").strip() or "N/A"
    return f"{title}__{conference}"


def sanitize_signature(signature: str) -> str:
    """Convert a paper signature into a filesystem-friendly slug."""

    normalized = re.sub(r"\s+", "_", signature.strip())
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", normalized)
    normalized = normalized.strip("._")
    return normalized.lower() or "paper"


def merge_article_metadata(
    base: MutableMapping[str, object], override: Mapping[str, object] | None
) -> MutableMapping[str, object]:
    """Merge article metadata, preferring non-empty override fields."""

    if not override:
        return base
    for key, value in override.items():
        if value not in (None, "", []):
            base[key] = value
    return base


def extract_standard_metrics(stdout: str) -> Dict[str, float]:
    """Extract common LibCity metrics (masked_mae/mape/mase) from CLI output."""

    metrics: Dict[str, float] = {}
    if not stdout:
        return metrics
    patterns = {
        "masked_mae": r"masked[_\s-]?mae\s*[:=]\s*([0-9.]+)",
        "masked_mape": r"masked[_\s-]?mape\s*[:=]\s*([0-9.]+)",
        "masked_mase": r"masked[_\s-]?mase\s*[:=]\s*([0-9.]+)",
        "mae": r"\bmae\s*[:=]\s*([0-9.]+)",
        "mape": r"\bmape\s*[:=]\s*([0-9.]+)",
        "mase": r"\bmase\s*[:=]\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout, flags=re.IGNORECASE)
        if match:
            metrics[key.lower()] = float(match.group(1))
    return metrics


__all__ = [
    "build_paper_signature",
    "sanitize_signature",
    "merge_article_metadata",
    "extract_standard_metrics",
]
