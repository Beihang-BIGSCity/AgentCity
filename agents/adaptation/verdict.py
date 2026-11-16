"""Canonical pass/fail decisions from recorded metrics, independent of LLM JSON spelling."""
from __future__ import annotations

import math


def normalize_verdict(payload: dict, task_spec: dict) -> dict:
    raw = payload.get('verdict')
    if not isinstance(raw, dict) or not raw:
        return {}
    target = task_spec.get('target') or {}
    metric = target.get('metric', 'masked_mae')
    maximize = bool(target.get('maximize', False))
    threshold = target.get('threshold')
    value = None
    source = None
    for name, key in (('tuning', 'best_metrics'), ('run_report', 'metrics')):
        record = payload.get(name) or {}
        candidate = (record.get(key) or {}).get(metric)
        if record.get('status') != 'success' or isinstance(candidate, bool):
            continue
        try:
            number = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            value, source = number, f'{name}.{key}'
            break
    try:
        threshold = float(threshold)
        if not math.isfinite(threshold):
            threshold = None
    except (TypeError, ValueError):
        threshold = None
    passed = value is not None and threshold is not None and (
        value >= threshold if maximize else value <= threshold
    )
    result = dict(raw)
    if isinstance(raw.get('metric'), dict):
        result['metric_details'] = raw['metric']
    if raw.get('reason') and 'agent_reason' not in result:
        result['agent_reason'] = raw['reason']
    result.pop('pass', None)
    result.pop('verdict', None)
    reason = 'No finite recorded metric or target threshold is available.'
    if value is not None and threshold is not None:
        reason = f"Recorded {metric}={value} {'>=' if maximize else '<='} {threshold}: {passed}."
    result.update(passed=passed, metric=metric, value=value, threshold=threshold,
                  maximize=maximize, reason=reason, computed_from=source)
    return result
