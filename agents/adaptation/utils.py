"""Shared helpers for the model-adaptation pipeline.

Reused from the earlier migration/tuning stack (LibCity subprocess
interpreter, metric extraction, slug normalisation) plus deterministic
dataset profiling: the statistical signals (CVglobal, CVnode, SH, rho1,
sampling interval) that connect data characteristics to module choices.
"""

from __future__ import annotations

import json
import math
import os
import re
from csv import DictReader
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

LIBCITY_PYTHON_ENV = "AGENTCITY_LIBCITY_PYTHON"
LIBCITY_PYTHON_FILE = Path(__file__).resolve().parents[2] / "data" / "runs" / "libcity_python.txt"
NULL_VALUE = 0.0
DEFAULT_MAX_TIMESTEPS = 20000


def record_libcity_python(interpreter: str) -> Path | None:
    """Persist the LibCity interpreter so server-spawned tools can find it.

    The MCP tool server is started by the harness and may not inherit our
    environment, so the resolved interpreter is also written next to the run
    records as a fallback for ``libcity_python()``.
    """

    try:
        LIBCITY_PYTHON_FILE.parent.mkdir(parents=True, exist_ok=True)
        LIBCITY_PYTHON_FILE.write_text(str(interpreter) + "\n", encoding="utf-8")
    except OSError:
        return None
    return LIBCITY_PYTHON_FILE


def _recorded_libcity_python() -> str:
    try:
        return LIBCITY_PYTHON_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def libcity_python() -> str:
    """Interpreter used to launch LibCity subprocesses (run_model.py / run_hyper.py).

    The orchestrator runs under the env without torch; point
    AGENTCITY_LIBCITY_PYTHON at the env that can actually run LibCity.

    Resolution order: the environment variable, then the interpreter recorded
    by the CLI at ``data/runs/libcity_python.txt`` (the MCP tool server is
    spawned by the harness and may not inherit our environment — see
    ``agents/core/openjiuwen_backend.py::_business_mcp_config``), and finally
    ``python`` on PATH, which only works when that interpreter already carries
    LibCity's dependencies.
    """

    return os.environ.get(LIBCITY_PYTHON_ENV) or _recorded_libcity_python() or "python"


def sanitize_slug(value: str) -> str:
    """Convert an arbitrary label into a filesystem-friendly slug."""

    normalized = re.sub(r"\s+", "_", str(value).strip())
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", normalized)
    return normalized.strip("._").lower() or "task"


def extract_standard_metrics(stdout: str) -> Dict[str, float]:
    """Extract common LibCity metrics (masked_mae/mape/mase) from CLI output."""

    metrics: Dict[str, float] = {}
    if not stdout:
        return metrics
    patterns = {
        "masked_mae": r"masked[_\s-]?mae\s*[:=]\s*([0-9.]+)",
        "masked_rmse": r"masked[_\s-]?rmse\s*[:=]\s*([0-9.]+)",
        "masked_mape": r"masked[_\s-]?mape\s*[:=]\s*([0-9.]+)",
        "masked_mase": r"masked[_\s-]?mase\s*[:=]\s*([0-9.]+)",
        "mae": r"\bmae\s*[:=]\s*([0-9.]+)",
        "rmse": r"\brmse\s*[:=]\s*([0-9.]+)",
        "mape": r"\bmape\s*[:=]\s*([0-9.]+)",
        "mase": r"\bmase\s*[:=]\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout, flags=re.IGNORECASE)
        if match:
            metrics[key.lower()] = float(match.group(1))
    return metrics


def evaluate_cache_metrics(
    repo_dir: Path,
    *,
    model_name: str,
    dataset: str,
    since: float | None = None,
    exp_id: str | None = None,
) -> Dict[str, Any]:
    """Read the metrics LibCity writes to ``libcity/cache/<exp_id>/evaluate_cache``.

    ``run_model.py`` prints a *truncated* pandas frame, so scraping stdout is not
    reliable; the CSV next to it is the authoritative record. It holds one row per
    prediction horizon, so the headline number is the mean over horizons (the
    LibCity/ST-LEGO convention) and the per-horizon series is returned as well.
    """

    cache_root = Path(repo_dir) / "libcity" / "cache"
    if exp_id:
        search_roots = [cache_root / str(exp_id) / "evaluate_cache"]
    else:
        search_roots = sorted(cache_root.glob("*/evaluate_cache"))
    candidates: List[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in root.glob("*.csv"):
            if not path.name.endswith(f"{dataset}.csv"):
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if since is not None and mtime < since - 5:
                continue
            candidates.append(path)
    if not candidates:
        return {}

    wanted = str(model_name)
    named = [path for path in candidates if f"_{wanted}_{dataset}.csv" in path.name]
    pool = named or candidates
    newest = max(pool, key=lambda path: path.stat().st_mtime)
    rows = _read_evaluate_csv(newest)
    if not rows:
        return {}

    metrics: Dict[str, float] = {}
    for key in rows[0]:
        values = [row[key] for row in rows if _is_finite(row.get(key))]
        if values:
            metrics[key.strip().lower()] = sum(values) / len(values)
    masked_series = [row.get("masked_mae") for row in rows if _is_finite(row.get("masked_mae"))]
    return {
        "metrics": metrics,
        "horizons": len(rows),
        "source": str(newest),
        "masked_mae_by_horizon": masked_series,
        "model_used": newest.name.rsplit(f"_{dataset}.csv", 1)[0].rsplit("_", 1)[-1],
    }


def _read_evaluate_csv(path: Path) -> List[Dict[str, float]]:
    """Parse one LibCity evaluate_cache CSV into numeric rows."""

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = DictReader(handle)
            rows: List[Dict[str, float]] = []
            for raw in reader:
                row: Dict[str, float] = {}
                for key, value in raw.items():
                    if key is None:
                        continue
                    try:
                        row[key.strip().lower()] = float(value)
                    except (TypeError, ValueError):
                        continue
                if row:
                    rows.append(row)
            return rows
    except (OSError, ValueError):
        return []


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def collect_run_metrics(
    repo_dir: Path,
    *,
    model_name: str,
    dataset: str,
    stdout: str = "",
    since: float | None = None,
    exp_id: str | None = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Return a LibCity run's metrics and where they came from.

    The ``evaluate_cache`` CSV is authoritative; scraping stdout is only a
    fallback because ``run_model.py`` prints a truncated pandas frame.
    """

    from_stdout = extract_standard_metrics(stdout)
    cache = evaluate_cache_metrics(
        repo_dir, model_name=model_name, dataset=dataset, since=since, exp_id=exp_id
    )
    metrics = dict(from_stdout)
    if cache.get("metrics"):
        metrics.update(cache["metrics"])
        detail: Dict[str, Any] = {
            "source_kind": "evaluate_cache",
            "source": cache.get("source"),
            "horizons": cache.get("horizons"),
            "aggregation": "mean over prediction horizons",
            "masked_mae_by_horizon": cache.get("masked_mae_by_horizon"),
            "stdout_metrics": from_stdout,
        }
    elif from_stdout:
        detail = {"source_kind": "stdout", "stdout_metrics": from_stdout}
    else:
        detail = {"source_kind": "none"}
    return metrics, detail


def load_dataset_config(raw_data_dir: Path, dataset: str) -> Dict[str, Any]:
    """Load LibCity's dataset config (``raw_data/<dataset>/config.json``)."""

    config_path = raw_data_dir / dataset / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"dataset config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def dataset_data_col(config: Mapping[str, Any], task: str = "traffic_state_pred") -> str:
    """Pick the target column for a task from the dataset config info block."""

    info = config.get("info") or {}
    columns = [str(item) for item in (info.get("data_col") or []) if str(item).strip()]
    if not columns:
        raise ValueError("dataset config has no info.data_col entries")
    if task.endswith("speed_prediction") or task == "traffic_state_pred":
        for column in columns:
            if "speed" in column:
                return column
    return columns[0]


def read_dyna_series(
    *,
    dataset_dir: Path,
    dataset: str,
    data_col: str,
    max_timesteps: int = DEFAULT_MAX_TIMESTEPS,
    train_rate: float = 1.0,
) -> Tuple[List[str], "Any", Dict[str, Any]]:
    """Read the LibCity ``.dyna`` file into a ``(T, N)`` float32 matrix.

    Returns ``(timestamps, matrix, meta)``. Time steps are evenly subsampled
    when the series exceeds ``max_timesteps`` so profiling stays bounded; the
    sampling interval is therefore inferred from the *full* timestamp column
    before subsampling, otherwise the interval would be over-estimated by the
    stride.
    """

    import numpy as np
    import pandas as pd

    dyna_path = dataset_dir / f"{dataset}.dyna"
    if not dyna_path.exists():
        raise FileNotFoundError(f"dyna file not found: {dyna_path}")

    frame = pd.read_csv(
        dyna_path,
        usecols=["time", "entity_id", data_col],
        dtype={"entity_id": "int64", data_col: "float32"},
    )
    if frame.empty:
        raise ValueError(f"empty dyna file: {dyna_path}")

    time_codes, time_values = pd.factorize(frame["time"], sort=True)
    node_codes, node_values = pd.factorize(frame["entity_id"], sort=True)

    num_steps = int(time_codes.max()) + 1
    num_nodes = int(node_codes.max()) + 1
    matrix = np.full((num_steps, num_nodes), np.nan, dtype="float32")
    matrix[time_codes, node_codes] = frame[data_col].to_numpy(dtype="float32")

    if not 0 < train_rate <= 1:
        raise ValueError("profile train_rate must be in (0, 1]")
    timestamps = [str(item) for item in time_values]
    full_timestamps = timestamps
    profile_steps = int(num_steps * train_rate)
    if profile_steps < 2:
        raise ValueError("training profile needs at least two timestamps")
    matrix = matrix[:profile_steps]
    timestamps = timestamps[:profile_steps]
    if profile_steps > max_timesteps:
        stride = int(math.ceil(profile_steps / max_timesteps))
        keep = np.arange(0, profile_steps, stride)
        matrix = matrix[keep]
        timestamps = [timestamps[int(index)] for index in keep]
    meta = {
        "total_timesteps": int(num_steps),
        "sampled_timesteps": int(len(timestamps)),
        "sampling_interval_min": infer_sampling_interval(full_timestamps),
        "training_timesteps": profile_steps,
        "profile_train_rate": train_rate,
    }
    return timestamps, matrix, meta


def _population_cv(values: "Any") -> float:
    import numpy as np

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    mean = float(np.mean(finite))
    if abs(mean) < 1e-8:
        return 0.0
    return float(np.std(finite) / abs(mean))


def _node_cv(matrix: "Any") -> float:
    import numpy as np

    ratios: List[float] = []
    for column in range(matrix.shape[1]):
        series = matrix[:, column]
        finite = series[np.isfinite(series)]
        if finite.size < 2:
            continue
        mean = float(np.mean(finite))
        if abs(mean) < 1e-8:
            continue
        ratios.append(float(np.std(finite) / abs(mean)))
    return float(np.mean(ratios)) if ratios else 0.0


def _spatial_heterogeneity(matrix: "Any") -> float:
    import numpy as np

    node_means = np.array(
        [np.nanmean(matrix[:, column]) for column in range(matrix.shape[1])], dtype="float64"
    )
    node_means = node_means[np.isfinite(node_means)]
    if node_means.size < 2:
        return 0.0
    mean = float(np.mean(node_means))
    if abs(mean) < 1e-8:
        return 0.0
    return float(np.std(node_means) / abs(mean))


def _lag1_autocorrelation(matrix: "Any") -> float:
    import numpy as np

    coefficients: List[float] = []
    for column in range(matrix.shape[1]):
        series = matrix[:, column]
        finite = np.isfinite(series)
        if finite.sum() < 3:
            continue
        centered = series - np.nanmean(series)
        centered = np.nan_to_num(centered, nan=0.0)
        denominator = float(np.sum(centered ** 2))
        if denominator < 1e-12:
            continue
        coefficients.append(float(np.sum(centered[:-1] * centered[1:]) / denominator))
    return float(np.mean(coefficients)) if coefficients else 0.0


def _windowed_envelope(values: "Any", windows: int) -> Dict[str, float]:
    """Deterministic stability check: spread of the CV across equal windows."""

    import numpy as np

    if windows < 2 or values.shape[0] < windows * 2:
        return {"cv_window_mean": 0.0, "cv_window_std": 0.0}
    splits = np.array_split(values, windows)
    per_window = [_population_cv(part) for part in splits]
    return {
        "cv_window_mean": float(np.mean(per_window)),
        "cv_window_std": float(np.std(per_window)),
    }


def compute_dataset_profile(
    *,
    dataset: str,
    raw_data_dir: Path,
    task: str = "traffic_state_pred",
    windows: int = 10,
    max_timesteps: int = DEFAULT_MAX_TIMESTEPS,
    data_col: str | None = None,
    train_rate: float = 1.0,
) -> Dict[str, Any]:
    """Compute the task-level statistical signals for one LibCity dataset."""

    config = load_dataset_config(raw_data_dir, dataset)
    explicit_target = data_col is not None
    if data_col is None:
        data_col = dataset_data_col(config, task)
    info = config.get("info") or {}
    if data_col not in (info.get("data_col") or []):
        raise ValueError(f"profile column {data_col!r} is absent from dataset config")
    dataset_dir = raw_data_dir / dataset
    has_graph = any((dataset_dir / f"{dataset}.{suffix}").exists() for suffix in ("rel",))

    timestamps, matrix, series_meta = read_dyna_series(
        dataset_dir=dataset_dir,
        dataset=dataset,
        data_col=data_col,
        max_timesteps=max_timesteps,
        train_rate=train_rate,
    )
    sanitized = _sanitize_matrix(matrix)
    profile: Dict[str, Any] = {
        "dataset": dataset,
        "task": task,
        "data_col": data_col,
        "node_count": int(sanitized.shape[1]),
        "timesteps": int(series_meta["total_timesteps"]),
        "profiled_timesteps": int(series_meta["sampled_timesteps"]),
        "training_timesteps": int(series_meta["training_timesteps"]),
        "profile_train_rate": train_rate,
        "profile_partition": "training" if train_rate < 1 else "full",
        "sampling_interval_min": series_meta["sampling_interval_min"],
        "has_graph": bool(has_graph),
        "time_start": timestamps[0] if timestamps else None,
        "time_end": timestamps[-1] if timestamps else None,
        "null_ratio": _null_ratio(matrix),
        "output_dim": 1 if explicit_target else info.get("output_dim", 1),
        "time_intervals": info.get("time_intervals"),
        "cv_global": _population_cv(sanitized),
        "cv_node": _node_cv(sanitized),
        "spatial_heterogeneity": _spatial_heterogeneity(sanitized),
        "rho1": _lag1_autocorrelation(sanitized),
    }
    envelope = _windowed_envelope(sanitized, windows)
    profile["stability"] = {
        "windows": windows,
        "cv_window_mean": envelope["cv_window_mean"],
        "cv_window_std": envelope["cv_window_std"],
    }
    return profile


def _sanitize_matrix(matrix: "Any") -> "Any":
    """Replace non-finite entries and LibCity nulls with per-node means."""

    import numpy as np

    values = np.array(matrix, dtype="float64", copy=True)
    values[values == NULL_VALUE] = np.nan
    column_means = np.nanmean(values, axis=0)
    for column in range(values.shape[1]):
        fallback = column_means[column]
        if not math.isfinite(float(fallback)):
            fallback = 0.0
        column_values = values[:, column]
        column_values[~np.isfinite(column_values)] = fallback
    return values


def _null_ratio(matrix: "Any") -> float:
    import numpy as np

    total = matrix.size
    if not total:
        return 0.0
    nulls = int(np.count_nonzero(~np.isfinite(matrix)))
    nulls += int(np.count_nonzero(matrix == NULL_VALUE))
    return float(nulls / total)


def infer_sampling_interval(timestamps: Sequence[str], sample: int = 200) -> float | None:
    """Infer the sampling interval in minutes from a timestamp column.

    Uses the median of the adjacent deltas over a bounded sample so a single
    irregular or duplicated timestamp cannot skew the signal: the decision
    layer uses Δt to pick temporal structures.
    """

    if len(timestamps) < 2:
        return None
    from datetime import datetime

    def _parse(value: str):
        text = str(value).replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    deltas: List[float] = []
    for index in range(1, min(len(timestamps), sample + 1)):
        previous, current = _parse(timestamps[index - 1]), _parse(timestamps[index])
        if previous is None or current is None:
            continue
        delta = abs((current - previous).total_seconds())
        if delta > 0:
            deltas.append(delta)
    if not deltas:
        return None
    deltas.sort()
    return round(deltas[len(deltas) // 2] / 60.0, 4)


__all__ = [
    "DEFAULT_MAX_TIMESTEPS",
    "LIBCITY_PYTHON_FILE",
    "LIBCITY_PYTHON_ENV",
    "NULL_VALUE",
    "compute_dataset_profile",
    "dataset_data_col",
    "extract_standard_metrics",
    "evaluate_cache_metrics",
    "collect_run_metrics",
    "libcity_python",
    "load_dataset_config",
    "infer_sampling_interval",
    "read_dyna_series",
    "record_libcity_python",
    "sanitize_slug",
]
