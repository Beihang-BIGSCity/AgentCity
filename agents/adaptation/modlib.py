"""Read/write access to the atomic module library (a data asset).

The library lives outside the LibCity source tree under ``data/modlib/``:

* ``index.json`` - five categories, one entry per atomic module
* ``code/<category>/<module_id>.py`` - a standalone ``torch.nn.Module``

Every module follows one of five unified interfaces so the composition agent
can assemble them without inspecting their internals:

* ``encoder``  ``forward(x: (B,T,N,C)) -> (B,T,N,d)``
* ``spatial``  ``forward(x: (B,T,N,d), adj: (N,N)|None) -> (B,T,N,d)``
* ``temporal`` ``forward(x: (B,T,N,d)) -> (B,T',N,d)``
* ``fusion``   ``forward(h_sp, h_tm) -> (B,T,N,d)``
* ``head``     ``forward(x: (B,T',N,d)) -> (B,H,N,output_dim)`` + ``loss(...)``
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

CATEGORIES: Sequence[str] = ("encoder", "spatial", "temporal", "fusion", "head")

CATEGORY_LABELS: Dict[str, str] = {
    "encoder": "数据编码器：Embedding / 数据归一化 / 缺失值补全",
    "spatial": "空间依赖模块",
    "temporal": "时间依赖模块",
    "fusion": "时空融合模块",
    "head": "输出头：预测层 + 损失函数",
}


def _index_path(modlib_dir: Path) -> Path:
    return modlib_dir / "index.json"


def load_index(modlib_dir: Path) -> Dict[str, Any]:
    """Load the module library index, tolerating a missing file."""

    path = _index_path(modlib_dir)
    if not path.exists():
        return {"version": 1, "categories": dict(CATEGORY_LABELS), "modules": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("categories", dict(CATEGORY_LABELS))
    payload.setdefault("modules", [])
    return payload


def save_index(modlib_dir: Path, index: Mapping[str, Any]) -> None:
    """Persist the module library index with a refreshed timestamp."""

    modlib_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(index)
    payload["version"] = payload.get("version", 1)
    payload["categories"] = payload.get("categories") or dict(CATEGORY_LABELS)
    payload["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    _index_path(modlib_dir).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def list_modules(
    modlib_dir: Path,
    *,
    category: str | None = None,
    tags: Iterable[str] | None = None,
    include_code: bool = False,
) -> List[Dict[str, Any]]:
    """Return module entries, optionally filtered by category and tags."""

    wanted_tags = {tag.strip().lower() for tag in (tags or []) if str(tag).strip()}
    modules = load_index(modlib_dir).get("modules") or []
    selected: List[Dict[str, Any]] = []
    for module in modules:
        if category and module.get("category") != category:
            continue
        if wanted_tags:
            module_tags = {str(tag).lower() for tag in (module.get("tags") or [])}
            if not wanted_tags.issubset(module_tags):
                continue
        entry = dict(module)
        if include_code:
            entry["code"] = module_code(modlib_dir, str(module.get("id")))
        selected.append(entry)
    return selected


def get_module(modlib_dir: Path, module_id: str, *, include_code: bool = True) -> Dict[str, Any]:
    """Return one module entry, raising ``KeyError`` when it is unknown."""

    for module in load_index(modlib_dir).get("modules") or []:
        if module.get("id") == module_id:
            entry = dict(module)
            if include_code:
                entry["code"] = module_code(modlib_dir, module_id)
            return entry
    raise KeyError(f"unknown module id: {module_id}")


def module_code(modlib_dir: Path, module_id: str) -> str:
    """Read the source file backing a module."""

    entry = get_module(modlib_dir, module_id, include_code=False)
    relative = str(entry.get("file") or "")
    path = modlib_dir / relative
    if not path.exists():
        raise FileNotFoundError(f"module source missing: {path}")
    return path.read_text(encoding="utf-8")


def register_module(
    modlib_dir: Path,
    *,
    module_id: str,
    category: str,
    class_name: str,
    code: str,
    name: str | None = None,
    desc: str = "",
    ctr_args: Mapping[str, Any] | None = None,
    io: str = "",
    tags: Sequence[str] | None = None,
    source: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Add a newly extracted atomic module to the library (idempotent by id)."""

    normalized_id = str(module_id).strip()
    if not normalized_id:
        raise ValueError("module_id is required")
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}; expected one of {list(CATEGORIES)}")
    if not str(class_name).strip():
        raise ValueError("class_name is required")
    if not str(code).strip():
        raise ValueError("code is required")

    index = load_index(modlib_dir)
    modules = list(index.get("modules") or [])
    relative = f"code/{category}/{normalized_id}.py"
    payload = {
        "id": normalized_id,
        "category": category,
        "name": name or normalized_id,
        "desc": desc,
        "file": relative,
        "class": class_name,
        "ctr_args": dict(ctr_args or {}),
        "io": io or _interface_hint(category),
        "tags": [str(tag) for tag in (tags or [])],
        "source": dict(source or {}),
        "status": "auto_added",
    }
    modules = [module for module in modules if module.get("id") != normalized_id]
    modules.append(payload)
    index["modules"] = modules
    save_index(modlib_dir, index)

    target = modlib_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(code if code.endswith("\n") else code + "\n", encoding="utf-8")
    return payload


def validate_index(modlib_dir: Path) -> List[str]:
    """Return a list of problems found in the on-disk module library."""

    problems: List[str] = []
    try:
        index = load_index(modlib_dir)
    except json.JSONDecodeError as exc:
        return [f"index.json is not valid JSON: {exc}"]

    seen: set[str] = set()
    for module in index.get("modules") or []:
        module_id = str(module.get("id") or "")
        if not module_id:
            problems.append("module entry without id")
            continue
        if module_id in seen:
            problems.append(f"duplicate module id: {module_id}")
        seen.add(module_id)
        category = str(module.get("category") or "")
        if category not in CATEGORIES:
            problems.append(f"{module_id}: unknown category {category!r}")
        relative = str(module.get("file") or "")
        path = modlib_dir / relative
        if not relative or not path.exists():
            problems.append(f"{module_id}: source file missing ({relative})")
            continue
        class_name = str(module.get("class") or "")
        if class_name and f"class {class_name}" not in path.read_text(encoding="utf-8"):
            problems.append(f"{module_id}: class {class_name} not found in {relative}")
    return problems


def _interface_hint(category: str) -> str:
    return {
        "encoder": "forward(x: (B,T,N,C)) -> (B,T,N,d)",
        "spatial": "forward(x: (B,T,N,d), adj: (N,N)|None) -> (B,T,N,d)",
        "temporal": "forward(x: (B,T,N,d)) -> (B,T',N,d)",
        "fusion": "forward(h_sp, h_tm) -> (B,T,N,d)",
        "head": "forward(x: (B,T',N,d)) -> (B,H,N,output_dim)",
    }.get(category, "")


__all__ = [
    "CATEGORIES",
    "CATEGORY_LABELS",
    "get_module",
    "list_modules",
    "load_index",
    "module_code",
    "register_module",
    "save_index",
    "validate_index",
]
