from __future__ import annotations

from typing import Any, Dict, Optional

_ACTIVE_PAPER: Optional[Dict[str, Any]] = None


def set_active_paper(paper: Dict[str, Any] | None) -> None:
    """Track which paper is currently being migrated."""

    global _ACTIVE_PAPER
    _ACTIVE_PAPER = dict(paper) if paper else None


def get_active_paper() -> Optional[Dict[str, Any]]:
    """Return the metadata for the paper currently being processed."""

    return _ACTIVE_PAPER


def clear_active_paper() -> None:
    """Clear the active paper pointer."""

    global _ACTIVE_PAPER
    _ACTIVE_PAPER = None


__all__ = ["set_active_paper", "get_active_paper", "clear_active_paper"]
