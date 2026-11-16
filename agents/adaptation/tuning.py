from __future__ import annotations

import ast
import hashlib
import itertools
import math
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .utils import collect_run_metrics, extract_standard_metrics, libcity_python, sanitize_slug

LIBCITY_TUNING_ENV = "AGENTCITY_ALLOW_LIBCITY_TUNING"
_HORIZON_SUFFIX = re.compile(r"^(?P<name>[a-z0-9_]+)@(?P<horizon>\d+)$")
# Directory (relative to the LibCity checkout) holding per-trial config files.
TUNING_CONFIG_DIR = "tuning_configs"
# Exactly the flags ``run_model.py`` accepts on the command line: its own
# arguments plus ``libcity/utils/argument_list.py::general_arguments``.
# Everything else (hidden_dim, num_layers, dropout, num_diffusion_steps, ...) is
# a model-config key: argparse rejects it with "unrecognized arguments", so it
# must reach LibCity through ``--config_file`` instead. A trial that tried to
# pass such a key on the CLI failed before training and produced no metrics.
RUN_MODEL_CLI_FLAGS = frozenset({
    "task", "model", "dataset", "config_file", "saved_model", "train", "exp_id", "seed",
    "gpu", "gpu_id", "train_rate", "eval_rate", "batch_size", "learning_rate",
    "max_epoch", "dataset_class", "executor", "evaluator", "num_workers",
})
DEFAULT_TUNING_SPACE: Dict[str, List[Any]] = {
    "learning_rate": [0.01, 0.005, 0.001],
    "batch_size": [32, 64],
}


def is_libcity_tuning_allowed() -> bool:
    """Return True if run_hyper.py usage is allowed in this environment."""

    value = str(os.environ.get(LIBCITY_TUNING_ENV, "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def ensure_search_space(space: Mapping[str, Sequence[Any]] | None) -> Dict[str, List[Any]]:
    """Normalize the provided search space or fall back to defaults."""

    if not space:
        return {key: list(values) for key, values in DEFAULT_TUNING_SPACE.items()}
    normalized: Dict[str, List[Any]] = {}
    for key, value in space.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values = [item for item in value if item is not None]
        else:
            values = [value]
        if values:
            normalized[str(key)] = list(values)
    return normalized or {key: list(values) for key, values in DEFAULT_TUNING_SPACE.items()}


def write_params_file(base_dir: Path, search_space: Mapping[str, Sequence[Any]]) -> Path:
    """Write a LibCity hyperparameter search-space file."""

    target_dir = base_dir / "hyper_params"
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"search_space_{timestamp}.txt"
    path = target_dir / filename
    lines: List[str] = []
    for name, values in search_space.items():
        formatted_values = ", ".join(_format_param_literal(item) for item in values)
        lines.append(f"{name} choice [{formatted_values}]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_trial_config(
    repo_dir: Path,
    *,
    model_name: str,
    params: Mapping[str, Any],
    extra_args: Mapping[str, Any] | None = None,
    base_config: str | None = None,
) -> str | None:
    """Write one trial's model-config knobs and return a ``--config_file`` value.

    ``run_model.py`` only declares a fixed set of flags, so a trial knob that is
    a model-config key (``hidden_dim``, ``num_layers``, ``dropout``, ...) cannot
    be passed on the command line: argparse rejects the whole command with
    "unrecognized arguments" before training starts. Such knobs are written to a
    JSON config file that LibCity loads through ``--config_file``; knobs that are
    real CLI flags stay out of the file because the caller passes them as flags,
    where LibCity's command line keeps precedence.

    ``base_config`` is the trial's inherited config file (a ``--config_file``
    value, with or without the ``.json`` suffix); its content is merged first so
    an ablation/override config survives the trial.

    Returns ``None`` when the trial needs no generated config file at all.
    """

    knobs = {
        str(key): value
        for source in (extra_args or {}, params or {})
        for key, value in source.items()
        if str(key) not in RUN_MODEL_CLI_FLAGS
    }
    payload: Dict[str, Any] = {}
    if base_config:
        source_path = Path(str(base_config))
        if not source_path.is_absolute():
            source_path = Path(repo_dir) / f"{base_config}.json"
        if not source_path.exists():
            raise FileNotFoundError(f"trial base config not found: {source_path}")
        payload.update(json.loads(source_path.read_text(encoding="utf-8")))
    payload.update(knobs)
    if not payload:
        return None

    target_dir = Path(repo_dir) / TUNING_CONFIG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:8]
    label = "_".join(
        f"{sanitize_slug(str(key))}{sanitize_slug(str(value))}" for key, value in sorted(knobs.items())
    )[:60]
    stem = sanitize_slug(f"tune_{model_name}_{label}").strip("_") or "tune_trial"
    path = target_dir / f"{stem}_{digest}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return f"{TUNING_CONFIG_DIR}/{path.stem}"


def trial_command(
    *,
    repo_dir: Path,
    model_name: str,
    dataset_name: str,
    task: str,
    params: Mapping[str, Any],
    config_file: str | None = None,
    extra_cli_args: Mapping[str, Any] | None = None,
    gpu: bool | None = None,
    gpu_id: str | None = None,
) -> tuple[List[str], str | None]:
    """Build one grid trial's ``run_model.py`` command.

    Returns ``(command, config_file_used)``. Model-config knobs are routed into a
    generated ``--config_file`` (see :func:`write_trial_config`) instead of being
    forwarded as flags that ``run_model.py`` would reject.
    """

    trial_config = write_trial_config(
        repo_dir, model_name=model_name, params=params, extra_args=extra_cli_args, base_config=config_file
    )
    effective_config = trial_config or config_file
    cmd = [libcity_python(), "run_model.py", "--task", task, "--model", model_name, "--dataset", dataset_name]
    if gpu is not None:
        cmd.extend(["--gpu", _format_bool(gpu)])
    if gpu_id is not None:
        cmd.extend(["--gpu_id", str(gpu_id)])
    if effective_config:
        cmd.extend(["--config_file", effective_config])
    for key, value in (params or {}).items():
        if str(key) in RUN_MODEL_CLI_FLAGS:
            cmd.extend([f"--{key}", str(value)])
    for key, value in (extra_cli_args or {}).items():
        if value is None or key in (params or {}) or str(key) in {"gpu", "gpu_id"} or str(key) not in RUN_MODEL_CLI_FLAGS:
            continue
        cmd.extend([f"--{key}", str(value)])
    return cmd, effective_config


def parse_hyper_result_file(path: Path) -> Dict[str, Any]:
    """Parse the hyper.result file emitted by LibCity."""

    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8", errors="ignore")
    best_params = {}
    best_valid = {}
    best_test = {}
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if line.lower().startswith("best params"):
            payload = line.split(":", 1)[1].strip()
            best_params = _safe_literal_eval(payload) or {}
        elif line.lower().startswith("best_valid_score"):
            if index + 1 < len(lines):
                best_valid = _safe_literal_eval(lines[index + 1]) or {}
        elif line.lower().startswith("best_test_result"):
            if index + 1 < len(lines):
                best_test = _safe_literal_eval(lines[index + 1]) or {}
    return {
        "best_params": best_params,
        "best_valid_score": best_valid,
        "best_test_result": best_test,
    }


def run_libcity_tuning(
    *,
    repo_dir: Path,
    run_dir: Path,
    model_name: str,
    dataset_name: str,
    params_file: Path,
    task: str = "traffic_state_pred",
    config_file: str | None = None,
    hyper_algo: str | None = None,
    max_evals: int | None = None,
    saved_model: bool | None = None,
    train: bool | None = None,
    gpu_id: str | None = None,
    gpu: bool | None = None,
    extra_cli_args: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run LibCity's run_hyper.py and parse its output."""

    cmd = [
        libcity_python(),
        "run_hyper.py",
        "--task",
        task,
        "--model",
        model_name,
        "--dataset",
        dataset_name,
        "--params_file",
        str(params_file),
    ]
    if gpu is not None:
        cmd.extend(["--gpu", _format_bool(gpu)])
    if gpu_id is not None:
        cmd.extend(["--gpu_id", str(gpu_id)])
    if config_file:
        cmd.extend(["--config_file", config_file])
    if hyper_algo:
        cmd.extend(["--hyper_algo", hyper_algo])
    if max_evals:
        cmd.extend(["--max_evals", str(max_evals)])
    if saved_model is not None:
        cmd.extend(["--saved_model", _format_bool(saved_model)])
    if train is not None:
        cmd.extend(["--train", _format_bool(train)])
    for key, value in (extra_cli_args or {}).items():
        if value is None:
            continue
        cmd.extend([f"--{key}", str(value)])

    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    status = "success" if result.returncode == 0 else "error"
    hyper_file = repo_dir / "hyper.result"
    parsed = parse_hyper_result_file(hyper_file)
    best_metrics = _metrics_from_mapping(parsed.get("best_test_result") or {})

    artifacts_dir = run_dir / "hyper_results"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_label = (
        f"{sanitize_slug(model_name)}_{sanitize_slug(dataset_name)}_"
        f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.result"
    )
    saved_artifact = None
    if hyper_file.exists():
        saved_artifact = artifacts_dir / artifact_label
        shutil.copy2(hyper_file, saved_artifact)

    return {
        "strategy": "libcity",
        "status": status,
        "returncode": result.returncode,
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
        "best_params": parsed.get("best_params") or {},
        "best_metrics": best_metrics,
        "artifacts": {
            "params_file": str(params_file),
            "hyper_result": str(saved_artifact) if saved_artifact else "",
        },
    }


def run_custom_grid_search(
    *,
    repo_dir: Path,
    model_name: str,
    dataset_name: str,
    search_space: Mapping[str, Sequence[Any]],
    task: str = "traffic_state_pred",
    config_file: str | None = None,
    max_trials: int | None = None,
    gpu_id: str | None = None,
    gpu: bool | None = None,
    extra_cli_args: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Perform a simple grid search by repeatedly invoking run_model.py."""

    normalized = ensure_search_space(search_space)
    keys = list(normalized.keys())
    values = [list(normalized[key]) for key in keys]
    trials: List[Dict[str, Any]] = []
    best_run: Dict[str, Any] | None = None
    best_score: float | None = None

    combinations = itertools.product(*values)
    for index, combo in enumerate(combinations):
        if max_trials is not None and index >= max_trials:
            break
        params = dict(zip(keys, combo))
        cmd, trial_config = trial_command(
            repo_dir=repo_dir,
            model_name=model_name,
            dataset_name=dataset_name,
            task=task,
            params=params,
            config_file=config_file,
            extra_cli_args=extra_cli_args,
            gpu=gpu,
            gpu_id=gpu_id,
        )

        started_at = time.time()
        completed = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
        metrics, metrics_detail = collect_run_metrics(
            repo_dir,
            model_name=model_name,
            dataset=dataset_name,
            stdout=completed.stdout or "",
            since=started_at,
        )
        score = _score_metrics(metrics) if completed.returncode == 0 else None
        trial_entry = {
            "params": params,
            "config_file": trial_config,
            "metrics": metrics,
            "metrics_detail": metrics_detail,
            "status": "success" if completed.returncode == 0 else "error",
            "returncode": completed.returncode,
            "stdout": (completed.stdout or "")[-2000:],
            "stderr": (completed.stderr or "")[-2000:],
        }
        trials.append(trial_entry)
        if score is None:
            if best_run is None:
                best_run = trial_entry
            continue
        if best_score is None or score < best_score:
            best_score = score
            best_run = trial_entry

    best_run = best_run or {"params": {}, "metrics": {}, "status": "error", "stdout": "", "stderr": ""}
    return {
        "strategy": "custom_grid",
        "status": best_run.get("status", "error"),
        "returncode": best_run.get("returncode", 1),
        "best_params": best_run.get("params", {}),
        "best_metrics": best_run.get("metrics", {}),
        "stdout": best_run.get("stdout", ""),
        "stderr": best_run.get("stderr", ""),
        "trials": trials,
    }


def _score_metrics(metrics: Mapping[str, float]) -> float | None:
    """Return a comparable score given LibCity metric keys."""

    for key in ("masked_mae", "mae", "rmse"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _metrics_from_mapping(data: Mapping[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    horizons: Dict[str, List[float]] = {}
    for key, value in data.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        name = str(key).strip().lower()
        match = _HORIZON_SUFFIX.match(name)
        if match:
            horizons.setdefault(match.group("name"), []).append(number)
        else:
            metrics[name] = number
    for name, values in horizons.items():
        finite = [item for item in values if math.isfinite(item)]
        if finite and name not in metrics:
            metrics[name] = sum(finite) / len(finite)
    if not metrics and data:
        # Fallback to parsing from stringified dict.
        parsed = extract_standard_metrics(str(data))
        metrics.update(parsed)
    return metrics


def _format_param_literal(value: Any) -> str:
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _safe_literal_eval(payload: str) -> Dict[str, Any]:
    try:
        value = ast.literal_eval(payload)
        if isinstance(value, dict):
            return value
        return {}
    except (ValueError, SyntaxError):
        return {}


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


__all__ = [
    "DEFAULT_TUNING_SPACE",
    "ensure_search_space",
    "is_libcity_tuning_allowed",
    "parse_hyper_result_file",
    "run_custom_grid_search",
    "run_libcity_tuning",
    "write_params_file",
]
