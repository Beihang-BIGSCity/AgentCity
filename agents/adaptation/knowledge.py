"""Expert knowledge base and Agent Skills with deterministic retrieval.

Both stores are data assets shared by the whole pipeline:

* ``knowledge/scenario_strategies.json`` - "scenario feature -> structural
  strategy" entries grouped into the three prior families used by the deck
  (environment / causality / constraint).
* ``skills/index.json`` + ``skills/<dimension>.md`` - tuning experience
  organised as "data feature profile -> tuning direction".

Retrieval is a deterministic nearest-neighbour lookup over the numeric
feature vector (no embeddings, no new dependencies): distances are computed
only over the features present in both the query and the entry, each scaled
by the spread of that feature across the corpus.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

FEATURE_KEYS: Sequence[str] = (
    "cv_global",
    "cv_node",
    "spatial_heterogeneity",
    "rho1",
    "sampling_interval_min",
    "node_count",
)

FAMILIES: Dict[str, str] = {
    "environment": "环境态势特征",
    "causality": "事件因果链路",
    "constraint": "业务场景约束",
}

SKILL_DIMENSIONS: Dict[str, str] = {
    "temporal": "时间表征",
    "spatial": "空间表征",
    "loss": "损失函数",
    "head": "预测模块",
}

# Module families every knowledge entry has to prescribe.
MODULE_FAMILIES: Sequence[str] = ("encoder", "spatial", "temporal", "fusion", "head")


def _knowledge_path(knowledge_dir: Path) -> Path:
    return knowledge_dir / "scenario_strategies.json"


def _skills_path(skills_dir: Path) -> Path:
    return skills_dir / "index.json"


def load_knowledge(knowledge_dir: Path) -> Dict[str, Any]:
    """Load the expert knowledge base (empty skeleton when missing)."""

    path = _knowledge_path(knowledge_dir)
    if not path.exists():
        return {"version": 1, "families": dict(FAMILIES), "entries": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("families", dict(FAMILIES))
    payload.setdefault("entries", [])
    return payload


def save_knowledge(knowledge_dir: Path, payload: Mapping[str, Any]) -> None:
    """Persist the knowledge base with a refreshed timestamp."""

    knowledge_dir.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["version"] = data.get("version", 1)
    data["families"] = data.get("families") or dict(FAMILIES)
    data["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    _knowledge_path(knowledge_dir).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_skills(skills_dir: Path) -> Dict[str, Any]:
    """Load the Agent Skills index (empty skeleton when missing)."""

    path = _skills_path(skills_dir)
    if not path.exists():
        return {"version": 1, "dimensions": dict(SKILL_DIMENSIONS), "skills": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("dimensions", dict(SKILL_DIMENSIONS))
    payload.setdefault("skills", [])
    return payload


def save_skills(skills_dir: Path, payload: Mapping[str, Any]) -> None:
    """Persist the skills index with a refreshed timestamp."""

    skills_dir.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["version"] = data.get("version", 1)
    data["dimensions"] = data.get("dimensions") or dict(SKILL_DIMENSIONS)
    data["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    _skills_path(skills_dir).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def skill_document(skills_dir: Path, skill: Mapping[str, Any]) -> str:
    """Read the markdown body backing one skill entry."""

    relative = str(skill.get("file") or "")
    if not relative:
        return ""
    path = skills_dir / relative
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def retrieve(
    *,
    knowledge_dir: Path,
    skills_dir: Path,
    features: Mapping[str, Any],
    top_k: int = 3,
    kinds: Iterable[str] = ("knowledge", "skill"),
    include_documents: bool = False,
) -> Dict[str, Any]:
    """Return the top-k knowledge entries / skills closest to ``features``."""

    wanted = {str(kind).strip().lower() for kind in kinds}
    corpus: List[Dict[str, Any]] = []
    if "knowledge" in wanted:
        for entry in load_knowledge(knowledge_dir).get("entries") or []:
            corpus.append({"kind": "knowledge", "entry": entry})
    if "skill" in wanted:
        for skill in load_skills(skills_dir).get("skills") or []:
            corpus.append({"kind": "skill", "entry": skill})

    query = _numeric_features(features)
    scales = _feature_scales([item["entry"].get("feature_profile") or {} for item in corpus])

    scored: List[Dict[str, Any]] = []
    for item in corpus:
        entry = item["entry"]
        distance = _distance(query, _numeric_features(entry.get("feature_profile") or {}), scales)
        if distance is None:
            continue
        scored.append({**item, "distance": distance})

    scored.sort(key=lambda item: (item["distance"], str(item["entry"].get("id") or "")))
    results: List[Dict[str, Any]] = []
    for item in scored[: max(1, int(top_k))]:
        entry = item["entry"]
        if item["kind"] == "knowledge":
            payload = {
                "kind": "knowledge",
                "id": entry.get("id"),
                "family": entry.get("family"),
                "title": entry.get("title"),
                "strategy": entry.get("strategy") or {},
                "tuning_direction": entry.get("tuning_direction") or [],
                "evidence": entry.get("evidence") or [],
                "source": entry.get("source"),
            }
        else:
            payload = {
                "kind": "skill",
                "id": entry.get("id"),
                "dimension": entry.get("dimension"),
                "title": entry.get("title"),
                "tuning_direction": entry.get("tuning_direction") or [],
                "strategies": entry.get("strategies") or [],
                "file": entry.get("file"),
                "source": entry.get("source"),
            }
            if include_documents:
                payload["document"] = skill_document(skills_dir, entry)
        payload["distance"] = round(float(item["distance"]), 6)
        payload["similarity"] = round(1.0 / (1.0 + float(item["distance"])), 6)
        results.append(payload)

    return {
        "query": dict(features),
        "shared_features": sorted(query.keys()),
        "candidates": len(corpus),
        "results": results,
    }


def record_experience(
    *,
    knowledge_dir: Path,
    skills_dir: Path,
    kind: str,
    entry: Mapping[str, Any],
) -> Dict[str, Any]:
    """Feed a finished round back into the knowledge base or the skills."""

    normalized = str(kind or "").strip().lower()
    if normalized == "knowledge":
        return _record_knowledge(knowledge_dir, dict(entry))
    if normalized == "skill":
        return _record_skill(skills_dir, dict(entry))
    raise ValueError("kind must be 'knowledge' or 'skill'")


def _record_knowledge(knowledge_dir: Path, entry: Dict[str, Any]) -> Dict[str, Any]:
    payload = load_knowledge(knowledge_dir)
    entries: List[Dict[str, Any]] = list(payload.get("entries") or [])
    normalised = normalise_knowledge_entry(
        entry,
        fallback_id=f"evolved-{len(entries) + 1:03d}",
        reserved_ids=_curated_ids(entries),
    )
    entries = [item for item in entries if item.get("id") != normalised["id"]]
    entries.append(normalised)
    payload["entries"] = entries
    save_knowledge(knowledge_dir, payload)
    return normalised


def _curated_ids(entries: Sequence[Mapping[str, Any]]) -> set:
    """Ids of hand-authored entries, which runtime feedback must never shadow."""

    return {
        str(item.get("id"))
        for item in entries
        if str(item.get("status") or "") != "evolved" and item.get("id")
    }


def _slug(text: Any, *, limit: int = 60) -> str:
    """Lowercase, hyphenated, filesystem-friendly form of ``text``."""

    slug = re.sub(r"[^0-9a-z]+", "-", str(text or "").lower()).strip("-")
    if slug.startswith("run-"):
        slug = slug[len("run-"):]
    return slug[:limit].strip("-")


def _experience_id(
    entry: Mapping[str, Any], *, curated_ids: set, fallback: str, prefix: str = "evolved"
) -> str:
    """Pick the id a recorded experience is stored under.

    A recorded round keeps its own stable id (derived from ``source``), so
    re-recording the same round updates its note, and it never reuses the id of
    a hand-authored entry. Observed live: a round recorded its lesson under the
    id of the curated prior it had just retrieved (``env-metr-la-...``), which
    deleted the authored entry instead of adding experience next to it.
    """

    wanted = str(entry.get("id") or "").strip()
    if wanted and wanted not in curated_ids:
        return wanted
    slug = _slug(entry.get("source")) or _slug(wanted)
    return f"{prefix}-{slug}" if slug else fallback


def _record_skill(skills_dir: Path, entry: Dict[str, Any]) -> Dict[str, Any]:
    payload = load_skills(skills_dir)
    skills: List[Dict[str, Any]] = list(payload.get("skills") or [])
    dimension = str(entry.get("dimension") or "temporal").strip().lower()
    if dimension not in SKILL_DIMENSIONS:
        raise ValueError(f"unknown skill dimension {dimension!r}")
    profile = retrieval_profile(entry.get("feature_profile") or entry)
    if not profile:
        raise ValueError(
            "tuning skills need at least one numeric retrieval feature "
            f"({list(FEATURE_KEYS)}) so the skill can be matched to a data profile"
        )
    if not entry.get("tuning_direction"):
        raise ValueError("tuning skills need at least one tuning_direction entry")
    entry["feature_profile"] = profile
    entry.setdefault("source", "runtime feedback")
    entry["id"] = _experience_id(
        {"id": entry.get("id"), "source": entry.get("source")},
        curated_ids=_curated_ids(skills),
        fallback=f"skill-{dimension}-evolved-{len(skills) + 1:03d}",
        # Skills and knowledge share one id namespace (the validator asserts it),
        # so a derived skill id must not look like a derived knowledge id.
        prefix="skill-evolved",
    )
    entry["dimension"] = dimension
    entry.setdefault("file", f"{dimension}.md")
    entry.setdefault("tuning_direction", [])
    entry.setdefault("strategies", [])
    entry.setdefault("status", "evolved")
    entry.setdefault("recorded_at", datetime.utcnow().isoformat(timespec="seconds") + "Z")
    skills = [item for item in skills if item.get("id") != entry["id"]]
    skills.append(entry)
    payload["skills"] = skills
    save_skills(skills_dir, payload)
    _append_skill_document(skills_dir, entry)
    return entry


def retrieval_profile(profile: Mapping[str, Any]) -> Dict[str, float]:
    """Keep only the numeric retrieval features (:data:`FEATURE_KEYS`).

    Agents hand over whatever profile they hold, usually the whole
    ``dataset_profile.json`` (dataset name, split sizes, graph metadata, ...).
    Retrieval only ever uses the six numeric features, and the store's contract
    is "feature profile == retrieval features", so everything else is dropped
    rather than written into the shared data asset.
    """

    if not isinstance(profile, Mapping):
        return {}
    sources: List[Mapping[str, Any]] = [profile]
    for nested in ("feature_profile", "dataset_profile", "profile"):
        candidate = profile.get(nested)
        if isinstance(candidate, Mapping):
            sources.append(candidate)
    result: Dict[str, float] = {}
    for key in FEATURE_KEYS:
        for source in sources:
            value = source.get(key)
            if value is None or isinstance(value, bool):
                continue
            try:
                result[key] = float(value)
            except (TypeError, ValueError):
                continue
            break
    return result


def strategy_module_ids(strategy: Any) -> Dict[str, str]:
    """Reduce a composition description to ``{family: module id}``.

    Accepts the store's own shape (``{"spatial": "cheb_gcn"}``) and the shape of
    a round's ``composition.json`` ``modules`` block
    (``{"spatial": {"id": "cheb_gcn", ...}}``), so the effect evaluator can
    forward the composition it just measured without reshaping it by hand.
    """

    if not isinstance(strategy, Mapping):
        return {}
    nested = strategy.get("modules")
    if isinstance(nested, Mapping):
        strategy = nested
    families: Dict[str, str] = {}
    for family in MODULE_FAMILIES:
        value = strategy.get(family)
        if isinstance(value, Mapping):
            value = value.get("id") or value.get("module") or value.get("name")
        if isinstance(value, str) and value.strip():
            families[family] = value.strip()
    return families


def normalise_knowledge_entry(
    entry: Mapping[str, Any],
    *,
    fallback_id: str,
    reserved_ids: Sequence[str] = (),
) -> Dict[str, Any]:
    """Coerce a recorded experience into the knowledge base's schema.

    Raises ``ValueError`` instead of storing an entry that would break the
    store's contract; the tool layer returns that message to the agent, which
    can then retry with the composition it just measured. Observed live: a
    round recorded its raw ``dataset_profile.json`` plus a prose architecture
    string, which left the shared data asset with entries that retrieval and
    the validator could no longer reason about.
    """

    profile = retrieval_profile(entry.get("feature_profile") or entry)
    if not profile:
        raise ValueError(
            "knowledge entries need numeric features among "
            f"{list(FEATURE_KEYS)}; got {sorted((entry.get('feature_profile') or entry))[:12]}"
        )
    strategy = strategy_module_ids(entry.get("strategy"))
    missing = [family for family in MODULE_FAMILIES if family not in strategy]
    if missing:
        raise ValueError(
            "knowledge entries need a module id for every module family; missing "
            f"{missing} (forward composition.json's modules block, "
            "{'spatial': {'id': 'cheb_gcn'}, ...})"
        )
    normalised = dict(entry)
    normalised["feature_profile"] = profile
    normalised["strategy"] = strategy
    family = str(normalised.get("family") or "").strip()
    normalised["family"] = family if family in FAMILIES else "environment"
    normalised["id"] = _experience_id(
        normalised, curated_ids=set(reserved_ids), fallback=fallback_id
    )
    normalised.setdefault("title", normalised["id"])
    normalised.setdefault("tuning_direction", [])
    normalised.setdefault("source", "runtime feedback")
    normalised.setdefault("status", "evolved")
    normalised.setdefault("recorded_at", datetime.utcnow().isoformat(timespec="seconds") + "Z")
    return normalised


def _append_skill_document(skills_dir: Path, entry: Mapping[str, Any]) -> None:
    """Append the new experience to the markdown skill document."""

    relative = str(entry.get("file") or f"{entry.get('dimension')}.md")
    path = skills_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "",
        f"## {entry.get('title') or entry.get('id')}",
        "",
        f"- 来源: {entry.get('source') or 'n/a'}",
        f"- 数据特征画像: {json.dumps(entry.get('feature_profile') or {}, ensure_ascii=False)}",
    ]
    directions = entry.get("tuning_direction") or []
    if directions:
        lines.append("- 调参方向:")
        lines.extend(f"  - {item}" for item in directions)
    strategies = entry.get("strategies") or []
    if strategies:
        lines.append("- 策略:")
        lines.extend(f"  - {item}" for item in strategies)
    evidence = entry.get("evidence") or []
    if evidence:
        lines.append(f"- 证据: {json.dumps(evidence, ensure_ascii=False)}")
    lines.append("")
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(existing + "\n".join(lines), encoding="utf-8")


def _numeric_features(payload: Mapping[str, Any]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key in FEATURE_KEYS:
        value = payload.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result[key] = number
    return result


def _feature_scales(profiles: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    scales: Dict[str, float] = {}
    for key in FEATURE_KEYS:
        values = [value[key] for value in (_numeric_features(profile) for profile in profiles) if key in value]
        if len(values) >= 2:
            spread = max(values) - min(values)
        else:
            spread = 0.0
        scales[key] = spread if spread > 1e-9 else 1.0
    return scales


def _distance(
    query: Mapping[str, float], profile: Mapping[str, float], scales: Mapping[str, float]
) -> float | None:
    shared = [key for key in query if key in profile]
    if not shared:
        return None
    total = 0.0
    for key in shared:
        total += abs(query[key] - profile[key]) / scales.get(key, 1.0)
    return total / len(shared)


__all__ = [
    "FAMILIES",
    "FEATURE_KEYS",
    "MODULE_FAMILIES",
    "SKILL_DIMENSIONS",
    "load_knowledge",
    "load_skills",
    "normalise_knowledge_entry",
    "record_experience",
    "retrieve",
    "retrieval_profile",
    "save_knowledge",
    "save_skills",
    "skill_document",
    "strategy_module_ids",
]
