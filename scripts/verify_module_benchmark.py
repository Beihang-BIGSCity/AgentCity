"""Audit the complete 72-run module experiment and its retained evidence."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("METR_LA", "PEMS_BAY", "PEMSD4", "PEMSD7")
VARIANTS = ("concat", "add", "weighted", "temporal_only", "spatial_only", "native_gru")


def audit(output):
    issues, checked = [], 0
    signatures = set()
    dataset_mask_counts = {}
    manifests = [json.loads(p.read_text()) for p in output.glob("manifest_gpu*.json")]
    if len(manifests) != 2:
        issues.append("expected two GPU manifests")
    for manifest in manifests:
        for relative, expected_hash in manifest["source_sha256"].items():
            snapshot = output / "source" / relative
            if not snapshot.exists() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected_hash:
                issues.append(f"source snapshot mismatch: {relative}")
            current = ROOT / relative
            if not current.exists() or hashlib.sha256(current.read_bytes()).hexdigest() != expected_hash:
                issues.append(f"training source changed during the experiment: {relative}")
    provenance = json.loads((output / "provenance.json").read_text())
    for relative, fingerprint in provenance["data_fingerprints"].items():
        digest = hashlib.sha256()
        with (ROOT / relative).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != fingerprint["sha256"]:
            issues.append(f"raw dataset changed during the experiment: {relative}")
    for ds, variant, seed in itertools.product(DATASETS, VARIANTS, (0, 1, 2)):
        name = f"{ds}_{variant}_s{seed}"
        path = output / "runs" / f"{name}.json"
        if not path.exists():
            issues.append(f"missing {name}")
            continue
        r = json.loads(path.read_text())
        checked += 1
        signatures.add(r["protocol_sha256"])
        if (r["dataset"], r["variant"], r["seed"], r["status"]) != (ds, variant, seed, "success"):
            issues.append(f"identity/status mismatch: {name}")
        history = r["history"]
        if r["epochs"] != 15 or [h["epoch"] for h in history] != list(range(1, 16)):
            issues.append(f"incomplete epochs: {name}")
        best = min(history, key=lambda h: h["val_masked_mae"])
        if best["epoch"] != r["best_epoch"] or best["val_masked_mae"] != r["best_val_masked_mae"]:
            issues.append(f"validation checkpoint mismatch: {name}")
        expected_lrs = [0.01] * 5 + [0.001] * 10
        if not np.allclose([h["lr"] for h in history], expected_lrs):
            issues.append(f"learning rate mismatch: {name}")
        test = r["test"]
        for key, scalar in (("mae_by_horizon", "masked_mae"),
                            ("rmse_by_horizon", "masked_rmse"),
                            ("mape_by_horizon", "masked_mape")):
            values = np.asarray(test[key])
            if len(values) != 12 or not np.isfinite(values).all() or not (values >= 0).all():
                issues.append(f"invalid {key}: {name}")
            elif not np.isclose(values.mean(), test[scalar], rtol=1e-12, atol=1e-12):
                issues.append(f"horizon mean mismatch {key}: {name}")
        counts = np.asarray(test["valid_counts_by_horizon"])
        profile = json.loads((output / f"{ds}_profile.json").read_text())
        maximum = profile["window_counts"][-1] * profile["nodes"]
        if len(counts) != 12 or not ((counts > 0) & (counts <= maximum)).all():
            issues.append(f"invalid masked counts: {name}")
        if ds in dataset_mask_counts and not np.array_equal(counts, dataset_mask_counts[ds]):
            issues.append(f"test target mask differs between variants/seeds: {name}")
        dataset_mask_counts[ds] = counts
        cuts = profile["raw_split_boundaries"]
        if profile["window_counts"] != [b - a - 23 for a, b in zip(cuts[:-1], cuts[1:])]:
            issues.append(f"window count/split mismatch: {name}")
        weighted_mae = np.average(test["mae_by_horizon"], weights=counts)
        if not np.isclose(weighted_mae, test["pooled_masked_mae"], rtol=1e-12):
            issues.append(f"pooled MAE mismatch: {name}")
        if not np.isfinite(r["wall_seconds"]) or r["wall_seconds"] <= 0:
            issues.append(f"invalid wall time: {name}")
        checkpoint = ROOT / r["checkpoint"]
        if not checkpoint.exists():
            issues.append(f"missing checkpoint: {name}")
        else:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            params = sum(v.numel() for k, v in state.items() if k != "adj")
            if params != r["parameters"] or not all(torch.isfinite(v).all() for v in state.values()):
                issues.append(f"invalid checkpoint parameters: {name}")
        if variant == "weighted" and not np.isclose(sum(r["fusion_weights"]), 1):
            issues.append(f"invalid fusion weights: {name}")
    if len(signatures) != 1:
        issues.append(f"expected a shared protocol fingerprint, got {len(signatures)}")
    actual = {p.stem for p in (output / "runs").glob("*.json")}
    expected = {f"{d}_{v}_s{s}" for d, v, s in itertools.product(DATASETS, VARIANTS, (0, 1, 2))}
    if actual - expected:
        issues.append(f"unexpected records: {sorted(actual - expected)}")
    summary = {"status": "passed" if not issues else "incomplete_or_failed",
               "checked_runs": checked, "expected_runs": 72,
               "checks": ["complete dataset/variant/seed matrix", "15 epochs per run",
                          "validation-only best epoch", "identical learning rate schedule",
                          "12 finite horizons and matching metric means", "identical target masks and split counts",
                          "finite checkpoints and matching parameter counts", "unchanged training source and raw data fingerprints",
                          "one protocol fingerprint"], "issues": issues}
    (output / "verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "result/module_benchmark_20261026")
    args = parser.parse_args()
    raise SystemExit(audit(args.output.resolve()))
