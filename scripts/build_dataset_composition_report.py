"""Report actual dataset-conditioned AgentCity compositions and their ablations.

The experiment plan defines the expected matrix. New training, real pipeline
artifacts and previously measured fixed-backbone controls remain distinct.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from pptx.enum.text import PP_ALIGN

from build_module_benchmark_report import (
    Deck, ROOT, TEAL, ORANGE, CHARCOAL, INK, MUTED, PAPER,
    avg, sd, fmt, csv_write, rgb, render_ppt,
)

DATASETS = ("PEMS_BAY", "PEMSD4", "PEMSD7")
TASKS = {"PEMS_BAY": "compose_pems_bay_20261027", "PEMSD4": "compose_pemsd4_20261027",
         "PEMSD7": "compose_pemsd7_20261027"}
FAMILIES = ("encoder", "spatial", "temporal", "fusion", "head")
FAMILY_LABEL = dict(zip(FAMILIES, ("输入编码", "空间模块", "时间模块", "融合模块", "预测头")))
CORE = ("full", "temporal_only", "spatial_only")
LABEL = {"full": "流程自产组合", "temporal_only": "仅时间分支", "spatial_only": "仅空间分支",
         "native_gru": "原生 GRU", "old_concat": "旧固定 STDiffGRU", "old_native_gru": "旧原生 GRU",
         "source_STGCN": "STGCN 相关图基线"}
SHORT = {"full": "Generated", "temporal_only": "Temporal only", "spatial_only": "Spatial only",
         "native_gru": "Native GRU", "old_concat": "Fixed STDiffGRU", "old_native_gru": "Reused GRU"}
COLORS = {"full": TEAL, "temporal_only": ORANGE, "spatial_only": "B96F39",
          "native_gru": "87929A", "old_concat": "547D86", "old_native_gru": "87929A"}
OLD = ROOT / "result/module_benchmark_20261026"


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_details(composition, family):
    selected = composition.get("modules", {}).get(family, {})
    catalog = read_json(ROOT / "data/modlib/index.json", {}).get("modules", [])
    match = next((m for m in catalog if m.get("id") == selected.get("id") and m.get("category") == family), {})
    source = selected.get("source", match.get("source", {}))
    if isinstance(source, dict):
        origin = source.get("libcity", "")
        source_class = source.get("class", Path(origin).stem if origin else "见源码")
    else:
        origin, source_class = str(source), str(source)
    return {**selected, "library_path": selected.get("library_path", match.get("file", "见实际生成源码")),
            "provenance": selected.get("provenance", origin or "见实际生成源码"), "source_class": source_class,
            "provenance_lookup": "composition module id cross-referenced against atomic-module index"}


def module_explanation(composition, family, profile):
    """Concise Chinese mechanism summary; verbatim selection rationale stays in notes."""
    module_id = composition.get("modules", {}).get(family, {}).get("id")
    descriptions = {
        "feature_embedding_encoder": "输入通道映射到hidden32；外部已标准化",
        "zscore_encoder": "按节点/通道归一化输入表征",
        "missing_fill_encoder": "沿历史时间填充缺失观测",
        "cheb_gcn": "固定任务图上的谱图卷积",
        "diffusion_gcn": "随机游走扩散建模节点依赖",
        "adaptive_gcn": "学习节点邻接，适应空间异质性",
        "spatial_attention": "注意力编码节点间依赖",
        "gru": "门控循环建模历史时间依赖",
        "lstm": "循环记忆建模稳定持续的时序",
        "gated_dilated_tcn": "门控膨胀卷积覆盖多尺度历史",
        "dilated_tcn": "膨胀卷积扩大历史感受野",
        "temporal_transformer": "自注意力编码历史时间关系",
        "add": "同宽分支相加，无额外融合参数",
        "concat": "拼接后投影，保留两分支表征",
        "weighted": "学习两路全局混合权重",
        "cross_attention": "交叉注意力融合两路表征",
        "linear_head": "直接线性回归未来12步",
        "mlp_head": "非线性预测头映射未来12步",
        "horizon_head": "按预测跨度分别输出回归结果",
    }
    return descriptions.get(module_id, "读取实际生成源码及原始选择理由")


def label(variant):
    return LABEL.get(variant, variant.removeprefix("source_") + " 完整模型")


def short(variant):
    return SHORT.get(variant, variant.removeprefix("source_"))


def primary_source_reference(pipeline, expected, dataset):
    """Use module lineage and the declared matrix, never test performance."""
    selected = pipeline["composition"].get("modules", {})
    lineage = (("spatial", "adaptive_gcn", "source_AGCRN"),
               ("temporal", "lstm", "source_LSTM"), ("temporal", "gru", "source_GRU"))
    allowed = {variant for d, variant, seed in expected if d == dataset}
    for family, module_id, variant in lineage:
        if selected.get(family, {}).get("id") == module_id and variant in allowed:
            return {"variant": variant, "selection_rule": f"actual {family} module {module_id} lineage; baseline predefined in experiment plan"}
    return {"variant": None, "selection_rule": "no directly traceable complete source model in the declared matrix"}


def verification_evidence(directory):
    audit = read_json(directory / "verification.json", {})
    recheck = read_json(directory / "checkpoint_recheck.json", {})
    design = read_json(directory / "pipeline_design_review.json", {})
    checked = recheck.get("checked_runs", [])
    maxima = {}
    for item in checked:
        for metric, difference in item.get("max_absolute_metric_difference", {}).items():
            maxima[metric] = max(maxima.get(metric, 0), difference)
    return {
        "record_audit": {"file": "verification.json", "status": audit.get("status", "not available"),
                         "passed": audit.get("passed", False), "expected_count": audit.get("expected_count"),
                         "found_count": audit.get("found_count"), "verified_record_count": len(audit.get("verified_runs", [])),
                         "scope": "record/checkpoint file, parameter, metric, input/graph and frozen-source audit; not a new inference of every checkpoint"},
        "checkpoint_recheck": {"file": "checkpoint_recheck.json", "status": recheck.get("status", "not available"),
                               "count": len(checked), "scope": recheck.get("scope"), "method": recheck.get("method"),
                               "training_rerun": recheck.get("training_rerun"), "maximum_metric_differences": maxima,
                               "native_cpu_operators": all("native CPU" in item.get("operator_backend", "") for item in checked) if checked else False},
        "pipeline_design_review": {"file": "pipeline_design_review.json", "review": design}}


def expected_matrix(plan):
    entries = plan.get("expected_runs")
    if isinstance(entries, list) and entries and isinstance(entries[0], dict):
        return {(r["dataset"], r["variant"], int(r["seed"])) for r in entries}, True
    variants = plan.get("expected_variants")
    if variants:
        if isinstance(variants, dict):
            return {(d, v, seed) for d, values in variants.items() for v in values for seed in (0, 1, 2)}, True
        return {(d, v, seed) for d in plan.get("datasets", DATASETS) for v in variants for seed in (0, 1, 2)}, True
    # A draft may describe the minimum core study while the baseline matrix is finalized.
    return {(d, v, seed) for d in DATASETS for v in CORE for seed in (0, 1, 2)}, False


def pipeline_auxiliary_values(value, path, source, issues):
    """Serialize non-finite unmasked MAPE only in derived pipeline summaries."""
    if isinstance(value, dict):
        return {key: pipeline_auxiliary_values(item, f"{path}.{key}", source, issues)
                for key, item in value.items()}
    if isinstance(value, list):
        return [pipeline_auxiliary_values(item, f"{path}[{index}]", source, issues)
                for index, item in enumerate(value)]
    if isinstance(value, float) and not math.isfinite(value):
        if path.rsplit(".", 1)[-1] != "MAPE":
            raise ValueError(f"Unexpected non-finite pipeline value outside unmasked MAPE: {source}:{path}")
        encoded = "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
        issues.append({"source_file": source, "path": path, "original_value": encoded,
                       "derived_value": encoded,
                       "explanation": "Unmasked auxiliary MAPE is non-finite with zero-flow targets; preserved as an explicit string in the derived summary only. Original artifact and masked metrics remain unchanged."})
        return encoded
    return value


def pipeline_artifacts(directory, dataset):
    task = ROOT / "data/adaptation" / TASKS[dataset]
    frozen = directory / "frozen" / dataset
    artifacts, paths, auxiliary_nonfinite = {}, {}, []
    for name in ("dataset_profile", "retrieval_before_planning", "plan", "data_integration", "composition",
                 "run_report", "tuning", "verdict"):
        file = frozen / f"{name}.json"
        if not file.exists():
            file = task / f"{name}.json"
        if file.exists():
            paths[name] = str(file.relative_to(ROOT))
            artifacts[name] = pipeline_auxiliary_values(read_json(file), name, paths[name], auxiliary_nonfinite)
    composition = artifacts.get("composition", {})
    if composition:
        assert composition.get("dataset") == dataset, f"composition dataset mismatch: {dataset}"
        modules = composition.get("modules", {})
        assert all(isinstance(modules.get(f), dict) and modules[f].get("id") for f in FAMILIES)
        assert composition.get("model_name") != "STDiffGRU", "old fixed model cannot represent actual composition"
    return {"task_id": TASKS[dataset], "artifacts": artifacts, "paths": paths,
            "auxiliary_nonfinite_values": auxiliary_nonfinite,
            "composition": composition, "composition_sha256": sha(ROOT / paths["composition"]) if composition else None}


def flatten_record(record, role, source):
    test = record["test"]
    return {"dataset": record["dataset"], "variant": record["variant"], "seed": record["seed"],
            "evidence_role": role, "model_name": record.get("model_name", "STDiffGRU" if record["variant"] == "old_concat" else "GRU"),
            "masked_mae": test["masked_mae"], "masked_rmse": test["masked_rmse"],
            "masked_mape_ratio": test["masked_mape"], "mae_h3": test["mae_by_horizon"][2],
            "mae_h6": test["mae_by_horizon"][5], "mae_h12": test["mae_by_horizon"][11],
            "best_val_masked_mae": record["best_val_masked_mae"], "best_epoch": record["best_epoch"],
            "parameters": record["parameters"], "wall_seconds": record["wall_seconds"],
            "allocated_parameters": record.get("allocated_parameters", record["parameters"]),
            "active_parameters": record.get("active_parameters"), "disabled_parameters": record.get("disabled_parameters"),
            "removed_parameters": record.get("removed_parameters"), "ablation_mode": record.get("ablation_mode", "not recorded"),
            "execution_backend_json": json.dumps(record.get("evidence", {}).get("execution_backend", {}), ensure_ascii=False),
            "training_loss_type": record.get("training_loss_type", "masked_MAE"),
            "optimizer_json": json.dumps(record.get("optimizer", {}), ensure_ascii=False),
            "source_json": source}


def old_comparability(directory, plan, dataset):
    """Legacy results require measured data agreement and explicit graph evidence."""
    current = read_json(directory / f"{dataset}_profile.json", {})
    previous = read_json(OLD / f"{dataset}_profile.json", {})
    fields = ("nodes", "timesteps", "raw_split_boundaries", "window_counts", "train_mean", "train_std")
    raw_equal = bool(current and previous) and all(current.get(f) == previous.get(f) for f in fields)
    contract = read_json(directory / "input_contracts" / f"{dataset}.json", current.get("input_contract", {}))
    target = "traffic_speed" if dataset == "PEMS_BAY" else "traffic_flow"
    # The old benchmark explicitly used the target plus time_in_day for every run.
    input_equal = contract.get("channels") == [target, "time_in_day"]
    data_equal = raw_equal and input_equal
    declared = plan.get("old_control_comparability", {}).get(dataset, {})
    old_graph = previous.get("task_graph", {})
    graph = read_json(directory / "graphs" / f"{dataset}.json", current.get("task_graph", {}))
    result = {}
    for variant in ("old_concat", "old_native_gru"):
        policy = declared.get(variant, {})
        data_ok = data_equal and policy.get("data_protocol_equal", True)
        if variant == "old_native_gru":
            graph_equal = None
            eligible = data_ok
            reason = "GRU不读取图；切分/scaler/输入通道实测一致" if eligible else "原始切分/scaler或输入通道不同/尚未核对"
        else:
            graph_equal = policy.get("graph_equal")
            if old_graph.get("adj_sha256") and graph.get("adj_sha256"):
                graph_equal = old_graph["adj_sha256"] == graph["adj_sha256"]
            eligible = data_ok and graph_equal is True
            reason = policy.get("reason") or ("数据与图协议一致" if eligible else
                     "新旧输入通道不同，不作模块增益归因" if not input_equal else
                     "新旧图协议不同，不作模块增益归因" if graph_equal is False else "图协议尚未核对，不作直接配对")
        result[variant] = {"eligible_for_paired_comparison": eligible, "data_protocol_equal": data_ok,
                           "raw_split_scaler_equal": raw_equal, "input_channels_equal": input_equal,
                           "graph_equal": graph_equal, "graph_independent": variant == "old_native_gru", "reason": reason}
    return result


def collect(directory, require_complete=False):
    plan = read_json(directory / "experiment_plan.json", {})
    expected, explicit_matrix = expected_matrix(plan)
    pipelines = {d: pipeline_artifacts(directory, d) for d in DATASETS}
    comparability = {d: old_comparability(directory, plan, d) for d in DATASETS}
    task_graphs = {d: read_json(directory / "graphs" / f"{d}.json", {}) for d in DATASETS}
    input_contracts = {d: read_json(directory / "input_contracts" / f"{d}.json", {}) for d in DATASETS}
    records, new_rows, horizons, errors = [], [], [], []
    seen = set()
    for path in sorted((directory / "runs").glob("*.json")):
        r = read_json(path)
        if r.get("status") != "success":
            errors.append({"path": str(path.relative_to(ROOT)), "status": r.get("status"), "error": r.get("error")})
            continue
        key = r["dataset"], r["variant"], int(r["seed"])
        assert key not in seen and key[0] in DATASETS and key[2] in (0, 1, 2), f"unexpected/duplicate run {key}"
        seen.add(key)
        assert r["epochs"] == len(r["history"]) == 15
        assert len(r["test"]["mae_by_horizon"]) == 12
        assert math.isclose(avg(r["test"]["mae_by_horizon"]), r["test"]["masked_mae"], abs_tol=1e-9)
        best = min(r["history"], key=lambda row: row["val_masked_mae"])
        assert best["epoch"] == r["best_epoch"]
        assert math.isclose(best["val_masked_mae"], r["best_val_masked_mae"], abs_tol=1e-9)
        assert (ROOT / r["checkpoint"]).is_file()
        assert all(math.isfinite(r["test"][m]) for m in ("masked_mae", "masked_rmse", "masked_mape"))
        assert all(math.isfinite(value) for metric in ("mae_by_horizon", "rmse_by_horizon", "mape_by_horizon")
                   for value in r["test"][metric]), f"non-finite masked horizon metric: {key}"
        run_graph = r.get("evidence", {}).get("task_graph", {})
        if task_graphs[key[0]].get("adj_sha256"):
            assert run_graph.get("adj_sha256") == task_graphs[key[0]]["adj_sha256"], f"run uses different task graph: {key}"
        if input_contracts[key[0]].get("sha256"):
            assert r.get("evidence", {}).get("input_contract", {}).get("sha256") == input_contracts[key[0]]["sha256"], f"run uses different input contract: {key}"
        if key[1] in CORE:
            pipeline = pipelines[key[0]]
            assert pipeline["composition"], f"cannot report run without actual composition: {key}"
            assert r["model_name"] == pipeline["composition"]["model_name"]
            assert r["evidence"]["composition_sha256"] == pipeline["composition_sha256"]
            assert {f: r["evidence"]["selected_modules"][f]["id"] for f in FAMILIES} == {
                f: pipeline["composition"]["modules"][f]["id"] for f in FAMILIES}
        records.append(r)
        new_rows.append(flatten_record(r, "new_training", str(path.relative_to(ROOT))))
        for h in range(12):
            horizons.append({"dataset": key[0], "variant": key[1], "seed": key[2], "horizon": h + 1,
                             "minutes_ahead": (h + 1) * 5, "masked_mae": r["test"]["mae_by_horizon"][h],
                             "masked_rmse": r["test"]["rmse_by_horizon"][h],
                             "masked_mape_ratio": r["test"]["mape_by_horizon"][h]})
    old_rows = []
    for path in sorted((OLD / "runs").glob("*.json")):
        r = read_json(path)
        if r.get("dataset") in DATASETS and r.get("variant") in ("concat", "native_gru") and r.get("status") == "success":
            original_variant = r["variant"]
            r["variant"] = "old_" + original_variant
            old_rows.append(flatten_record(r, "reused_20261026_supplement", str(path.relative_to(ROOT))))
    rows = new_rows + old_rows
    groups = {}
    variants = list(CORE) + sorted({r["variant"] for r in new_rows if r["variant"] not in CORE}) + ["old_concat", "old_native_gru"]
    group_rows = []
    for d in DATASETS:
        for v in variants:
            current = [r for r in rows if r["dataset"] == d and r["variant"] == v]
            if not current:
                continue
            g = {"dataset": d, "variant": v, "n_seeds": len(current), "evidence_role": current[0]["evidence_role"]}
            for field in ("masked_mae", "masked_rmse", "masked_mape_ratio", "mae_h3", "mae_h6", "mae_h12",
                          "best_val_masked_mae", "parameters", "allocated_parameters", "active_parameters",
                          "disabled_parameters", "removed_parameters", "wall_seconds"):
                values = [r[field] for r in current if r[field] is not None]
                g[field + "_mean"], g[field + "_std"] = avg(values), sd(values)
            groups[(d, v)] = g
            group_rows.append(g)
    lookup = {(r["dataset"], r["variant"], r["seed"]): r for r in rows}
    pairs, choices = [], {}
    for d in DATASETS:
        singles = [v for v in CORE[1:] if groups.get((d, v), {}).get("n_seeds") == 3]
        choices[d] = min(singles, key=lambda v: groups[(d, v)]["masked_mae_mean"]) if len(singles) == 2 else None
        for reference in [v for v in variants if v != "full"]:
            if reference.startswith("old_") and not comparability[d][reference]["eligible_for_paired_comparison"]:
                continue
            for seed in (0, 1, 2):
                candidate, baseline = lookup.get((d, "full", seed)), lookup.get((d, reference, seed))
                if candidate and baseline:
                    pairs.append({"dataset": d, "variant": "full", "reference_variant": reference,
                                  "reference_role": baseline["evidence_role"], "seed": seed,
                                  "candidate_mae": candidate["masked_mae"], "reference_mae": baseline["masked_mae"],
                                  "gain_absolute": baseline["masked_mae"] - candidate["masked_mae"],
                                  "gain_percent": 100 * (baseline["masked_mae"] - candidate["masked_mae"]) / baseline["masked_mae"],
                                  "is_posthoc_best_single": reference == choices[d]})
    missing = sorted(expected - seen)
    pipeline_required = ("dataset_profile", "retrieval_before_planning", "plan", "composition", "run_report", "tuning", "verdict")
    pipeline_missing = [f"{d}/{name}.json" for d in DATASETS for name in pipeline_required if name not in pipelines[d]["artifacts"]]
    if require_complete:
        if not explicit_matrix:
            raise ValueError("Formal report requires expected_variants or explicit expected_runs in experiment_plan.json")
        if not all(task_graphs[d].get("adj_sha256") for d in DATASETS):
            raise ValueError("Formal report requires actual task graph SHA for every dataset")
        if not all(input_contracts[d].get("sha256") for d in DATASETS):
            raise ValueError("Formal report requires actual input-contract SHA for every dataset")
        if missing or pipeline_missing or errors:
            raise ValueError(f"study incomplete: {len(missing)} training runs, pipeline {pipeline_missing}, errors={errors}")
        if seen - expected:
            raise ValueError(f"runs outside finalized matrix: {sorted(seen - expected)}")
        checked_evidence = verification_evidence(directory)
        if not checked_evidence["record_audit"]["passed"] or checked_evidence["record_audit"]["found_count"] != len(expected):
            raise ValueError("formal report requires the passing full-matrix verification.json")
        if checked_evidence["checkpoint_recheck"]["status"] != "passed" or checked_evidence["checkpoint_recheck"]["count"] != 9:
            raise ValueError("formal report requires the nine passing native-operator checkpoint rechecks")
    composition_rows = []
    for d, p in pipelines.items():
        c = p["composition"]
        for f in c.get("modules", {}):
            m = module_details(c, f)
            composition_rows.append({"dataset": d, "model_name": c["model_name"], "family": f,
                                     "module_id": m["id"], "class": m.get("class"), "library_path": m.get("library_path"),
                                     "provenance": m.get("provenance", m.get("source")),
                                     "rationale": c.get("rationale", {}).get(f, m.get("rationale")),
                                     "provenance_lookup": m["provenance_lookup"],
                                     "composition_path": p["paths"]["composition"], "composition_sha256": p["composition_sha256"]})
    primary_sources = {d: primary_source_reference(pipelines[d], expected, d) for d in DATASETS}
    source_comparisons = []
    for d in DATASETS:
        reference = primary_sources[d]["variant"]
        values = [p["gain_percent"] for p in pairs if p["dataset"] == d and p["reference_variant"] == reference]
        full_group, reference_group = groups.get((d, "full"), {}), groups.get((d, reference), {})
        ready = len(values) == 3
        source_comparisons.append({"dataset": d, "reference_variant": reference,
                                   "reference_selection_rule": primary_sources[d]["selection_rule"],
                                   "full_mae_mean": full_group.get("masked_mae_mean"),
                                   "source_mae_mean": reference_group.get("masked_mae_mean"), "paired_n": len(values),
                                   "full_allocated_parameters": full_group.get("allocated_parameters_mean"),
                                   "source_allocated_parameters": reference_group.get("allocated_parameters_mean"),
                                   "full_parameter_percent_of_source": 100 * full_group["allocated_parameters_mean"] / reference_group["allocated_parameters_mean"] if reference_group.get("allocated_parameters_mean") else None,
                                   "paired_gain_mean": avg(values) if ready else None,
                                   "paired_gain_std": sd(values) if ready else None,
                                   "outcome": "combination_better" if ready and avg(values) > 0 else "combination_worse" if ready and avg(values) < 0 else "equal" if ready else "pending"})
    summary = {"complete": explicit_matrix and not missing and not pipeline_missing and not errors,
               "new_completed_runs": len(new_rows), "expected_new_runs": len(expected), "explicit_expected_matrix": explicit_matrix,
               "reused_supplement_runs": len(old_rows), "missing_runs": [list(k) for k in missing],
               "pipeline_missing": pipeline_missing, "failed_records": errors,
               "plan": plan, "groups": group_rows, "test_posthoc_best_single": choices,
               "old_control_comparability": comparability,
               "task_graphs": task_graphs,
               "input_contracts": input_contracts,
               "primary_source_references": primary_sources, "source_model_comparisons": source_comparisons,
               "verification_evidence": verification_evidence(directory),
               "pipelines": pipelines, "total_new_worker_hours": sum(r["wall_seconds"] for r in new_rows) / 3600,
               "paired_formula": "100 * (reference MAE - full composition MAE) / reference MAE; same seed",
               "std_definition": "sample standard deviation across three seeds, ddof=1"}
    directory.mkdir(exist_ok=True, parents=True)
    csv_write(directory / "run_metrics.csv", new_rows)
    csv_write(directory / "reused_controls.csv", old_rows)
    csv_write(directory / "metrics_by_horizon.csv", horizons)
    csv_write(directory / "group_summary.csv", group_rows)
    csv_write(directory / "paired_comparisons.csv", pairs)
    csv_write(directory / "composition_evidence.csv", composition_rows)
    csv_write(directory / "source_model_comparisons.csv", source_comparisons)
    auxiliary = [dict(dataset=d, **item) for d, p in pipelines.items() for item in p["auxiliary_nonfinite_values"]]
    (directory / "pipeline_auxiliary_nonfinite.json").write_text(json.dumps(
        {"count": len(auxiliary), "scope": "derived pipeline summaries only; no change to source artifacts or new-run masked metrics", "values": auxiliary},
        ensure_ascii=False, indent=2, allow_nan=False))
    (directory / "legacy_protocol_comparison.json").write_text(json.dumps(comparability, ensure_ascii=False, indent=2))
    (directory / "results_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return summary, groups, rows, records, pairs, variants


def gain(pairs, d, reference):
    values = [p["gain_percent"] for p in pairs if p["dataset"] == d and p["reference_variant"] == reference]
    return avg(values), sd(values), len(values)


def metric_cell(groups, d, v, plan=None):
    g = groups.get((d, v))
    if not g:
        if plan and isinstance(plan.get("expected_variants"), dict) and v not in plan["expected_variants"].get(d, []):
            return "未纳入本数据集"
        return "待完成"
    return f"{fmt(g['masked_mae_mean'])} ± {fmt(g['masked_mae_std'])}" + (f" (n={g['n_seeds']})" if g["n_seeds"] != 3 else "")


def write_report(directory, summary, groups, pairs, variants):
    lines = ["# 每数据集真实 AgentCity 组合与消融实验", "",
             f"状态：{'完整实测' if summary['complete'] else 'DRAFT'}；新增训练 {summary['new_completed_runs']}/{summary['expected_new_runs']} 次；另复用旧实验 {summary['reused_supplement_runs']} 条对照记录，后者不算新增训练。", "",
             "## 研究问题与证据角色", "",
             "本轮先对各数据集执行真实 AgentCity 流程：仅训练时段画像 → 检索知识与技能 → 规划 → 五类原子模块选择 → 生成并注册独立模型/配置 → 真实 pilot → 冻结结构 → 多 seed 基准与结构自身的单分支消融。模块选择来自各自 composition.json，而非把旧 STDiffGRU 更换数据集后重新命名。",
             "不同数据集的实际结构可以相同；是否相同由实测流程产物决定，不人为制造差异。旧固定骨架实验只回答固定结构的跨数据集表现，单列为补充对照。", "",
             "## 训练时段画像与真实组合", "",
             "| 数据集 | 训练时间步 | 节点 | CVglobal | CVnode | SH | rho1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for d in DATASETS:
        p = summary["pipelines"][d]["artifacts"].get("dataset_profile", {})
        lines.append(f"| {d} | {p.get('training_timesteps', '—')} | {p.get('node_count', '—')} | {fmt(p.get('cv_global'))} | {fmt(p.get('cv_node'))} | {fmt(p.get('spatial_heterogeneity'))} | {fmt(p.get('rho1'))} |")
    for d in DATASETS:
        p = summary["pipelines"][d]
        c = p["composition"]
        lines += ["", f"### {d}：{c.get('model_name', '流程尚未完成')}", "",
                  f"画像、检索、规划、组合及pilot记录路径：`data/adaptation/{p['task_id']}/`。优先读取基准运行时保存的 frozen 快照。", ""]
        graph = summary["task_graphs"].get(d, {})
        if graph:
            lines += [f"本轮实际任务图：adj SHA `{graph.get('adj_sha256', '见graph记录')}`；配置与边统计详见 `graphs/{d}.json`。组合、消融和完整来源模型共享此图。", ""]
        contract = summary["input_contracts"].get(d, {})
        if contract:
            lines += [f"本轮实际输入通道：`{contract.get('channels')}`；feature_dim={contract.get('feature_dim')}，输入合同SHA `{contract.get('sha256')}`。", ""]
        if c:
            lines += [f"真实结构：`{c.get('architecture', '详见下方五类模块')}`", "",
                      "| 类别 | 实际模块 | 类 | 实现来源 |", "|---|---|---|---|"]
            for f in FAMILIES:
                m = module_details(c, f)
                lines.append(f"| {FAMILY_LABEL[f]} | `{m['id']}` | `{m.get('class', '见源码')}` | {m.get('provenance', m.get('library_path', '见composition.json'))} |")
            lines += ["", "实际组合器记录的选择理由（保留原文以便复核，不由报告生成器补造）：", ""]
            for f in FAMILIES:
                reason = c.get("rationale", {}).get(f, c["modules"][f].get("rationale"))
                if reason:
                    lines.append(f"- {FAMILY_LABEL[f]}：{reason}")
        retrieval = p["artifacts"].get("retrieval_before_planning", {})
        lines += ["", "规划前检索前列记录：", ""]
        for hit in retrieval.get("results", [])[:3]:
            lines.append(f"- `{hit.get('id')}`：{hit.get('title', '')}；similarity={hit.get('similarity', '未记录')}。")
        pilot = p["artifacts"].get("run_report", {})
        pilot_mae = pilot.get("metrics", {}).get("masked_mae")
        lines += ["", f"Pilot 状态：`{pilot.get('status', '未完成')}`；run_report 中 masked_mae={fmt(pilot_mae)}。Pilot只证明链路可运行，不能作为15epoch性能结论。", ""]
    lines += ["## 冻结结构后的统一评估协议", "",
              "目标：PEMS_BAY 为 traffic_speed；PEMSD4/7 仅为 traffic_flow。原始时间按60/20/20切分后各自构造12→12窗口；scaler只拟合训练原始时段。输入通道保留各自真实生成合同，可能仅有目标通道，也可能含time_in_day，具体见input_contracts/<dataset>.json。所有误差反归一化后计算，零目标按LibCity约定mask。主指标为12 horizon masked_MAE算术平均，masked_RMSE为逐horizon RMSE的均值。", "",
              "流程自产组合及两个消融采用hidden32、batch128、15epochs、Adam .01、第5轮后×.1、clip5、seed0/1/2；按验证集pooled masked_MAE最小的epoch选checkpoint，再评估测试集。消融属于零信息干预：只阻断当前自产结构的一路表征进入fusion，其余模块与形状保留。实际生成模型可保留未使用分支的分配参数，不能将其有效参数下降宣称为实际减参；逐run分别记录allocated/active/disabled/removed参数与ablation_mode。仅空间仍可能通过head使用历史时间。所有本数据集的组合、消融和完整来源模型共享该真实任务配置产生的邻接图；bidir、边权、阈值及adj哈希等记录在graphs/<dataset>.json。", "",
              "来源完整模型保留其原生结构、深度、训练损失和记录的优化设置，但与本数据集组合共享实际输入通道和任务图，不属于参数/损失/优化器严格匹配实验；具体设置逐run写在training_loss_type、optimizer和resolved_config中。不能把原生模型默认规模与hidden32的模块组合当作等参数比较。原生GRU若输入合同含time_in_day，可读取该未来已知辅助特征，不读取未来交通目标；输入为target-only时则不使用该辅助通道。旧实验复用单列且不计新增训练。", "",
              "## 新增实测的测试集 masked_MAE", "",
              "| 变体 | PEMS_BAY | PEMSD4 | PEMSD7 |", "|---|---:|---:|---:|"]
    for v in [v for v in variants if not v.startswith("old_")]:
        lines.append("| " + label(v) + " | " + " | ".join(metric_cell(groups, d, v, summary["plan"]) for d in DATASETS) + " |")
    lines += ["", "均值±样本标准差（ddof=1）。三个seed用于刻画波动，不宣称显著性。速度和流量的原始MAE不跨数据集求总平均。", "",
              "## 自产组合相对各自单分支的贡献", "",
              "同seed配对：`100 × (对照masked_MAE − full masked_MAE) / 对照masked_MAE`，先逐seed计算再平均。最佳单分支若展示，固定为三seed平均测试误差较小的一个分支，属于事后描述，不逐seed更换对照。", "",
              "| 数据集 | 单分支对照 | 配对增益均值 ± 样本标准差 | n |", "|---|---|---:|---:|"]
    for d in DATASETS:
        for ref in CORE[1:]:
            g, spread, n = gain(pairs, d, ref)
            lines.append(f"| {d} | {label(ref)} | {fmt(g, 2)}% ± {fmt(spread, 2)}% | {n} |")
    lines += ["", "## 结论须同时覆盖自身贡献和完整来源模型", "",
              "完整来源对照由实际模块来源和预先声明的实验矩阵确定，不按测试成绩挑选。自身组合优于单分支，不等于优于完整来源模型；容量、损失和优化设置不同限制了机制归因，但不能抹去实测负结果。", "",
              "| 数据集 | full均值 | 主要完整来源模型 | 来源模型均值 | full对来源的配对增益 | 结果 |",
              "|---|---:|---|---:|---:|---|"]
    for comparison in summary["source_model_comparisons"]:
        d, reference = comparison["dataset"], comparison["reference_variant"]
        outcome = {"combination_better": "组合优于完整来源", "combination_worse": "完整来源更好，保留负结果", "equal": "相同", "pending": "待完整3 seeds"}[comparison["outcome"]]
        delta = f"{comparison['paired_gain_mean']:+.2f}% ± {comparison['paired_gain_std']:.2f}%" if comparison["paired_gain_mean"] is not None else "待完整3 seeds"
        lines.append(f"| {d} | {fmt(comparison['full_mae_mean'])} | {label(reference) if reference else '未设置'} | {fmt(comparison['source_mae_mean'])} | {delta} | {outcome} |")
    lines += ["", "参数规模与实测精度的取舍（分配参数，不是仅有效参数）：", ""]
    for comparison in summary["source_model_comparisons"]:
        if comparison["source_allocated_parameters"] is not None:
            lines.append(f"- {comparison['dataset']}：组合 {comparison['full_allocated_parameters']:,.0f} 参数；{label(comparison['reference_variant'])} {comparison['source_allocated_parameters']:,.0f} 参数；组合参数量为来源模型的 {comparison['full_parameter_percent_of_source']:.2f}%。若来源模型误差更低，结论是小参数与当前预算下精度的取舍，不能借容量差异改写负结果。")
    lines += ["", "## 旧固定结构的补充对照（复用，不计新增训练）", "",
              "| 数据集 | 旧固定STDiffGRU | 旧原生GRU | 新组合相对旧固定结构的配对增益 |", "|---|---:|---:|---:|"]
    for d in DATASETS:
        g, spread, n = gain(pairs, d, "old_concat")
        note = f"{fmt(g, 2)}% ± {fmt(spread, 2)}% (n={n})" if n else summary["old_control_comparability"][d]["old_concat"]["reason"]
        lines.append(f"| {d} | {metric_cell(groups, d, 'old_concat')} | {metric_cell(groups, d, 'old_native_gru')} | {note} |")
    lines += ["", "## 历史METR_LA锚点：用户认可的早期独立流程", "",
              "原始独立流程生成STDiffGRU：feature embedding + DiffusionGCN + GRU + concat + linear head。历史15epoch实验使用10%/5%/85%的窗口划分，与本轮三个数据集的真实流程和原始时间60/20/20协议不同，不混为一张同协议结果表。", "",
              "| 历史变体 | masked_MAE | seed口径 |", "|---|---:|---|",
              "| 完整STDiffGRU组合 | 3.6313 | 3 seeds均值 |", "| 仅时间分支 | 3.6757 | 3 seeds均值 |",
              "| 仅空间分支 | 3.7302 | 3 seeds均值 |", "| 历史原生RNN | 4.6083 | 单seed弱基线 |", "",
              "历史约21.2%改善仅相对上述单seed RNN，不代表胜过完整来源模型，也不替代本轮逐数据集生成结构。来源：documentation/metr_la_speed_smoke_adaptation_summary.md中2026-10-15追加实验。", "",
              "## 审计、局限与复现", "",
              "完整产物：`composition_evidence.csv`、`run_metrics.csv`、`reused_controls.csv`、`metrics_by_horizon.csv`、`group_summary.csv`、`paired_comparisons.csv`、`results_summary.json`。",
              "原始流程证据位于各任务的dataset_profile/retrieval_before_planning/plan/composition/run_report/tuning/verdict；冻结结构、生成源码配置及原子模块快照在`frozen/<dataset>/`。逐run证据包含composition/source/config哈希及checkpoint哈希；`graphs/<dataset>.json`记录实际任务图参数与adj哈希。",
              "`audits/`验证消融参数边界、分支实际执行和零表征进入fusion；如存在verification或checkpoint_recheck，以该文件记录的实际检查范围为准，不把少量checkpoint复算说成全部训练均重新推理。",
              f"`verification.json`当前记录审计状态为`{summary['verification_evidence']['record_audit']['status']}`，覆盖{summary['verification_evidence']['record_audit']['found_count']}条训练记录；这是记录/权重文件及哈希等审计，不代表全部checkpoint重新推理。",
              f"`checkpoint_recheck.json`状态为`{summary['verification_evidence']['checkpoint_recheck']['status']}`，独立复算{summary['verification_evidence']['checkpoint_recheck']['count']}个seed0核心checkpoint（3数据集×full及两个消融），使用原生CPU算子和原始LibCity指标函数；D7也使用原生Conv2d，不使用lowering。这不是重新训练，逐horizon误差及容差原样保留在证据文件中。",
              "`pipeline_design_review.json`核对设计时间线：D4的TCN层数2→3、base dilation 1→2在首次pilot之前完成，最终dilations=[2,4,8]；不是看到测试结果后调结构。D7仅将C2合成冒烟说明更正为实际C1，构造与forward AST未变。原始artifact不改，各自冻结版本分别核对。",
              f"新增训练worker wall time合计 {summary['total_new_worker_hours']:.2f}小时，不包含pilot与数据加载。使用同型号GPU，但执行backend不同：Bay/D4使用各自冻结的原生卷积训练器，D7使用经过前向/梯度/整模型Adam对齐验证的等价FP32时间卷积lowering，不改变生成模型源码、参数、精度或预算。各自frozen trainer和backend哈希分别留存，不要求所有数据集trainer哈希相同；耗时仅为工程参考，不能作严格吞吐对比。",
              "Backend数值证据：[benchmark_validation_patch.json](benchmark_validation_patch.json)、[execution_versions.json](execution_versions.json)、[conv_backend_tests.log](conv_backend_tests.log)、[linear_conv_probe.log](linear_conv_probe.log)、[full_model_kernel_equivalence.log](full_model_kernel_equivalence.log)，以及逐run evidence.execution_backend和[D7冻结执行backend](frozen/PEMSD7/execution_backend.py)。",
              "审计链接：[verification.json](verification.json)、[checkpoint_recheck.json](checkpoint_recheck.json)、[pipeline_design_review.json](pipeline_design_review.json)、[provenance.json](provenance.json)。全量训练记录与哈希审计、九次原生CPU推理复算、设计时间线审查分开解释。",
              "pipeline_auxiliary_nonfinite.json逐项列出pilot原始未mask MAPE在零流量下产生的Infinity/NaN。派生summary将其显式字符串化并记录源路径，原文件不改；新run及所有masked主要指标不作静默清洗，发现非有限必须失败。",
              "结构选择来自训练画像和已有知识，不等于画像因果决定最优结构。15epochs不证明收敛；三个seed不支持严格显著性结论；零值mask可能忽略真实零流量；来源完整模型比较存在容量/损失/优化差异。", "",
              "```bash", "/home/wangwenrui/opt/miniconda3/envs/autogen/bin/python scripts/build_dataset_composition_report.py --require-complete --render", "```", ""]
    (directory / "真实流程组合与消融实验报告.md").write_text("\n".join(lines))


class CompositionDeck(Deck):
    def __init__(self, directory, summary):
        super().__init__(directory, summary)
        self.prs.core_properties.title = "每数据集真实AgentCity组合与模块消融"
        self.prs.core_properties.subject = "真实画像、检索规划、代码生成、pilot、冻结结构和多seed评估"

    def slide(self, title, subtitle="", dark=False):
        self.number += 1
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = rgb(CHARCOAL if dark else PAPER)
        self.text(s, "AGENTCITY  /  DATASET-CONDITIONED COMPOSITION", .6, .3, 12, .25, 10, "8ECBBF" if dark else MUTED, True)
        self.text(s, title, .6, .85, 12.1, .8, 36, "FFFFFF" if dark else INK, True)
        if subtitle:
            self.text(s, subtitle, .62, 1.76, 12.0, .6, 14, "B8CCC8" if dark else MUTED)
        status = "完整实测" if self.summary["complete"] else "DRAFT"
        self.text(s, f"2026.10.27   |   {status}   |   新训练 {self.summary['new_completed_runs']}/{self.summary['expected_new_runs']}   |   复用对照单列", .6, 7.02, 11.7, .22, 9.5, "B8CCC8" if dark else MUTED)
        self.text(s, f"{self.number:02d}", 12.1, 6.95, .6, .35, 12, "B8CCC8" if dark else MUTED, align=PP_ALIGN.RIGHT)
        return s


def structure_diagram(deck, slide, composition):
    modules = composition.get("modules", {})
    def name(family):
        module_id = modules.get(family, {}).get("id", "流程尚未完成")
        parts = module_id.split("_")
        if len(module_id) > 18 and len(parts) > 1:
            split = (len(parts) + 1) // 2
            return "_".join(parts[:split]) + "\n" + "_".join(parts[split:])
        return module_id
    boxes = [("12步历史输入", .7, 3.5, 1.8, .9), (name("encoder"), 2.9, 3.4, 2.2, 1.1),
             (name("spatial"), 5.5, 2.65, 2.2, .9), (name("temporal"), 5.5, 4.15, 2.2, .9),
             (name("fusion"), 8.1, 3.4, 1.7, 1.1), (name("head"), 10.25, 3.4, 2.3, 1.1)]
    for text, x, y, w, h in boxes:
        deck.rect(slide, x, y, w, h, "F4E3CB" if x == 8.1 else "D9EEE4", True)
        deck.text(slide, text, x + .1, y + .25, w - .2, h - .1, 13 if "_" in text else 16, INK, True, PP_ALIGN.CENTER)
    deck.connector(slide, (2.5, 3.95), (2.9, 3.95), True)
    deck.connector(slide, (5.1, 3.95), (5.3, 3.95))
    deck.connector(slide, (5.3, 3.1), (5.3, 4.6))
    for y in (3.1, 4.6):
        deck.connector(slide, (5.3, y), (5.5, y), True)
        deck.connector(slide, (7.7, y), (7.9, y))
    deck.connector(slide, (7.9, 3.1), (7.9, 4.6))
    deck.connector(slide, (7.9, 3.95), (8.1, 3.95), True)
    deck.connector(slide, (9.8, 3.95), (10.25, 3.95), True)


def build_slides(directory, summary, groups, rows, records, pairs, variants):
    deck = CompositionDeck(directory, summary)
    pipelines = summary["pipelines"]
    s = deck.slide("每数据集真实组合，再评估模块贡献", "让数据画像进入真实AgentCity流程；针对各自产结构做消融与完整模型对照", True)
    unique = {tuple(p["composition"].get("modules", {}).get(f, {}).get("id") for f in FAMILIES) for p in pipelines.values() if p["composition"]}
    for i, (title, number) in enumerate((("实际数据集", "3"), ("已生成结构", str(sum(bool(p["composition"]) for p in pipelines.values()))),
                                       ("不同模块组合", str(len(unique))), ("新增训练完成", str(summary["new_completed_runs"])))):
        deck.card(s, title, number, .65 + i * 3.1, 2.85, 2.7, ORANGE if i == 3 else "65D3B5", True)
    deck.text(s, "PEMS_BAY 速度  /  PEMSD4 流量  /  PEMSD7 流量", .7, 5.1, 11.8, .55, 23, "FFFFFF", True)
    deck.text(s, "每个数据集独立规划与生成；即使选择相同模块，也如实保留。", .7, 5.95, 11.8, .55, 17, "B8CCC8")

    s = deck.slide("三个证据层次，回答不同研究问题", "真正的流程组合、结构自身消融、旧固定骨架的补充表现分别记录")
    cards = [("真实流程", "训练画像 → 检索 → 规划\n生成独立模型和配置\n真实pilot后冻结结构"),
             ("新训练与消融", "每个自产结构及两种单分支\n3 seeds / 15 epochs\n来源完整模型另作对照"),
             ("旧结果复用", "旧固定STDiffGRU / 原生GRU\n明确标为补充对照\n不计本轮新增训练数")]
    for i, (title, body) in enumerate(cards):
        x = .7 + i * 4.15
        deck.rect(s, x, 2.65, 3.8, 3.2, "FFFFFF", True)
        deck.text(s, title, x + .25, 3.0, 3.3, .5, 24, TEAL, True)
        deck.text(s, body, x + .25, 3.9, 3.3, 1.5, 17, INK)
    deck.text(s, "本轮核心：结构来自真实流程产物，不能用旧模型换数据集后冒充。", .7, 6.3, 11.7, .45, 17, TEAL, True)

    s = deck.slide("真实流程证据链：每一步都有文件", "先用训练时段画像选结构；pilot只验证可运行性；最终测试不反推改结构")
    steps = [("训练画像", "dataset_profile"), ("知识检索", "retrieval_before_planning"), ("任务规划", "plan"),
             ("五类模块组合", "composition"), ("真实pilot", "run_report"), ("评价与冻结", "verdict")]
    for i, (title, key) in enumerate(steps):
        x, y = .7 + (i % 3) * 4.15, 2.5 + (i // 3) * 1.65
        count = sum(key in p["artifacts"] for p in pipelines.values())
        deck.rect(s, x, y, 3.8, 1.3, "D9EEE4" if count == 3 else "FFFFFF", True)
        deck.text(s, f"{i+1:02d}  {title}", x + .2, y + .2, 3.4, .4, 21, INK, True)
        deck.text(s, f"{count}/3 数据集已留存记录", x + .2, y + .82, 3.4, .3, 14, MUTED)
    deck.text(s, "证据路径：data/adaptation/compose_<dataset>_20261027/；最终训练保留frozen快照与源码哈希。", .7, 6.25, 11.8, .55, 14, MUTED)

    s = deck.slide("仅训练时段画像：数据差异先于模型选择", "CV刻画波动，SH刻画节点差异，rho1刻画时间持续性；这些量是选择依据而非最优性证明")
    table = [["数据集", "节点", "训练时间步", "CVglobal", "CVnode", "SH", "rho1"]]
    for d in DATASETS:
        p = pipelines[d]["artifacts"].get("dataset_profile", {})
        table.append([d, p.get("node_count", "—"), p.get("training_timesteps", "—"), fmt(p.get("cv_global")),
                      fmt(p.get("cv_node")), fmt(p.get("spatial_heterogeneity")), fmt(p.get("rho1"))])
    deck.table(s, table, .7, 2.65, [2.05, .85, 2.0, 1.65, 1.65, 1.65, 1.85], .7, 15)
    deck.rect(s, .7, 5.95, 11.7, .7, "D9EEE4", True)
    deck.text(s, "画像只读取前60%原始时间段；不使用最终验证/测试统计来选择模块。", .95, 6.12, 11.2, .4, 18, TEAL, True)

    s = deck.slide("检索进入规划：保留命中的真实知识条目", "近邻知识提供结构候选与经验；最终选择以各自composition.json为准")
    table = [["数据集", "规划前首个含结构的候选", "相似度", "候选结构中的空间 / 时间"]]
    for d in DATASETS:
        hits = pipelines[d]["artifacts"].get("retrieval_before_planning", {}).get("results", [])
        hit = next((candidate for candidate in hits if candidate.get("strategy")), hits[0] if hits else {})
        strategy = hit.get("strategy", {})
        table.append([d, hit.get("id", "流程尚未完成"), fmt(hit.get("similarity")), f"{strategy.get('spatial', '见记录')} / {strategy.get('temporal', '见记录')}"])
    deck.table(s, table, .7, 2.55, [1.9, 5.5, 1.2, 3.1], .85, 12)
    deck.text(s, "检索候选≠实际最终结构。完整检索内容、规划原文、选择理由分别保留，避免事后用结果编理由。", .7, 6.45, 11.7, .55, 16, TEAL, True)

    for d in DATASETS:
        p, c = pipelines[d], pipelines[d]["composition"]
        profile = p["artifacts"].get("dataset_profile", {})
        s = deck.slide(f"{d}：真实流程生成的结构", f"模型：{c.get('model_name', '生成中')}；五类模块名称直接读取composition.json")
        structure_diagram(deck, s, c)
        deck.text(s, f"训练画像：CV={fmt(profile.get('cv_global'))}，SH={fmt(profile.get('spatial_heterogeneity'))}，rho1={fmt(profile.get('rho1'))}", .7, 5.65, 11.7, .5, 20, TEAL, True)
        deck.text(s, "只消融这个实际结构的空间或时间分支；保留其余模块、fusion/head路径和实际深度。", .7, 6.35, 11.7, .5, 16, MUTED)
        s = deck.slide(f"{d}：模块选择与实现来源", "按真实组合器记录核对；原始理由与超参数完整保存在报告、演讲备注和composition_evidence.csv")
        table = [["模块类别", "实际选择", "中文机制概括", "原子模块来源"]]
        for family in FAMILIES:
            module = module_details(c, family)
            table.append([FAMILY_LABEL[family], module.get("id", "生成中"), module_explanation(c, family, profile), module["source_class"]])
        deck.table(s, table, .7, 2.5, [1.35, 3.3, 4.3, 2.75], .55, 12)
        pilot = p["artifacts"].get("run_report", {})
        pilot_text = f"真实pilot：{pilot.get('status', '尚未完成')}；masked_mae={fmt(pilot.get('metrics', {}).get('masked_mae'))}。仅作运行验证。"
        if d == "PEMSD4":
            pilot_text += "\nTCN感受野在首次pilot前确定：3层、dilations=[2,4,8]；非测试后调结构。"
        elif d == "PEMSD7":
            pilot_text += "\nC2→C1为输入说明更正；构造和forward代码未变。"
        deck.text(s, pilot_text, .7, 6.03 if d != "PEMS_BAY" else 6.25, 11.7, .72 if d != "PEMS_BAY" else .5,
                  13 if d != "PEMS_BAY" else 17, TEAL, True)

    s = deck.slide("评估协议：pilot与最终基准分开", "冻结结构后只按验证集选择训练epoch；不凭最终测试重选或改造组合")
    deck.table(s, [["阶段", "真实流程pilot", "冻结结构最终基准"], ["目的", "模型生成、注册、训练链路跑通", "组合与自身单分支的性能比较"],
                  ["结构", "训练画像 + 知识检索 + 规划选择", "读取实际生成模型/配置；保持结构"],
                  ["预算", "以run_report记录的pilot为准", "15epochs × 3 seeds；batch128"],
                  ["数据 / 选模", "以pilot数据合同记录为准", "原始时间60/20/20先切再造窗；验证选epoch"]], .7, 2.6, [1.65, 4.75, 5.3], .62, 14)
    deck.text(s, "统一12→12、train-only scaler；保留各任务真实输入通道和图。来源完整模型保留原生结构/损失/优化设置。", .7, 6.27, 11.7, .6, 14, MUTED)

    s = deck.slide("自产组合与单分支：测试集masked_MAE", "均值±样本标准差；所有行来自本轮新训练；数值越低越好")
    table = [["变体", *DATASETS]]
    for v in CORE:
        table.append([label(v)] + [metric_cell(groups, d, v, summary["plan"]) for d in DATASETS])
    if any((d, "native_gru") in groups for d in DATASETS):
        table.append([label("native_gru")] + [metric_cell(groups, d, "native_gru", summary["plan"]) for d in DATASETS])
    deck.table(s, table, .7, 2.65, [2.4, 3.1, 3.1, 3.1], .7, 15)
    deck.text(s, "单分支来自各自产结构，可能是不同的空间/时间模块；不能把所有时间分支统称为GRU。", .7, 6.3, 11.7, .55, 16, TEAL, True)

    s = deck.slide("模块贡献：同seed比较去掉分支的影响", "正数表示完整组合优于单分支；两个对照完整保留，不预设组合必胜")
    available = [d for d in DATASETS if all(groups.get((d, v), {}).get("n_seeds") == 3 for v in CORE)]
    series = [(label(v), [gain(pairs, d, v)[0] for d in available]) for v in CORE[1:]] if available else []
    deck.chart(s, available, series, .7, 2.5, 11.7, 3.85, colors=[ORANGE, "B96F39"], percent=True, legend=True)
    deck.text(s, "增益=100×（单分支MAE−完整组合MAE）/单分支MAE；先逐seed计算后平均。参数量并非严格匹配。", .7, 6.5, 11.7, .4, 12, MUTED)

    s = deck.slide("来源及相关完整模型：原生结构对照", "同数据切分、输入和评价口径；具体来源对应关系以原子库与composition证据为准")
    source_variants = [v for v in variants if v.startswith("source_")]
    table = [["完整模型 / 组合", *DATASETS]]
    for v in ["full"] + source_variants:
        table.append([label(v)] + [metric_cell(groups, d, v, summary["plan"]) for d in DATASETS])
    deck.table(s, table, .7, 2.55, [2.4, 3.1, 3.1, 3.1], min(.55, 3.45 / len(table)), 13)
    deck.text(s, "来源模型不是单模块消融：它们有完整原生网络、深度和训练目标。执行失败或未纳入者应在实验计划中说明。", .7, 6.35, 11.7, .6, 15, MUTED)

    s = deck.slide("补充：旧固定骨架的结果与协议核对", "旧固定STDiffGRU取预设concat配置；复用原始实测，明确不算新增训练")
    available = [d for d in DATASETS if groups.get((d, "full"), {}).get("n_seeds") == 3 and groups.get((d, "old_concat"), {}).get("n_seeds") == 3
                 and summary["old_control_comparability"][d]["old_concat"]["eligible_for_paired_comparison"]]
    if available:
        deck.chart(s, available, [("自产组合相对旧固定STDiffGRU", [gain(pairs, d, "old_concat")[0] for d in available])],
                   .7, 2.5, 8.0, 3.8, percent=True)
    else:
        table = [["数据集", "新任务组合", "旧固定骨架", "协议核对"]]
        for d in DATASETS:
            comparison = summary["old_control_comparability"][d]["old_concat"]
            status = ("输入不同" if not comparison["input_channels_equal"] else
                      "图不同" if comparison["graph_equal"] is False else "未证明一致")
            table.append([d, metric_cell(groups, d, "full"), metric_cell(groups, d, "old_concat"), status])
        deck.table(s, table, .7, 2.75, [1.5, 2.25, 2.25, 2.0], .65, 12)
        deck.text(s, "输入/图协议不同或尚未核对：\n旧数值只作补充背景，不计算‘模块组合带来的提升’。", .9, 5.65, 7.7, .7, 15, MUTED)
    deck.rect(s, 9.2, 2.75, 3.2, 2.95, "D9EEE4", True)
    deck.text(s, "证据角色\n\n旧实验：固定结构跨数据集\n新实验：按数据条件组合\n\n不把事后比较当作结构选择依据", 9.45, 3.02, 2.7, 2.5, 15, INK)
    legacy_note = ("同seed配对，正数为新组合更好；只有协议核对一致的数据集才计算增益。" if available else
                   "输入、图或原始切分不同，不能直接归因于模块组合。")
    deck.text(s, legacy_note, .7, 6.45, 11.7, .4, 13, MUTED)

    s = deck.slide("预测跨度：自产组合的短期与远期表现", "各数据集分别使用原始目标标度；曲线为3 seeds平均masked_MAE")
    for i, d in enumerate(DATASETS):
        x = .65 + i * 4.2
        deck.text(s, d, x + .1, 2.5, 3.8, .35, 18, INK, True)
        lines = []
        for v in CORE:
            data = [r for r in records if r["dataset"] == d and r["variant"] == v]
            if len(data) == 3:
                lines.append((short(v), [avg([r["test"]["mae_by_horizon"][h] for r in data]) for h in range(12)]))
        deck.chart(s, list(range(5, 65, 5)) if lines else [], lines, x, 3.0, 4.0, 2.85, "line",
                   colors=[COLORS[v] for v in CORE], legend=i == 0)
    deck.text(s, "横轴为预测分钟（5–60）；Generated / Temporal only / Spatial only。全部逐horizon数据见CSV。", .7, 6.4, 11.7, .45, 13, MUTED)

    s = deck.slide("稳定性与成本：固定预算下的工程表现", "样本标准差按已完成seed计算（目标3个）；参数分配/有效；异backend耗时仅作工程参考")
    table = [["数据集", "变体", "MAE样本标准差", "分配 / 有效参数", "平均秒数"]]
    for d in DATASETS:
        for v in CORE:
            g = groups.get((d, v), {})
            counts = f"{g['allocated_parameters_mean']:,.0f} / {g['active_parameters_mean']:,.0f}" if g and g.get("active_parameters_mean") is not None else "—"
            variant_label = label(v) + (f" (n={g['n_seeds']})" if g and g["n_seeds"] != 3 else "")
            table.append([d, variant_label, fmt(g.get("masked_mae_std")), counts,
                          fmt(g.get("wall_seconds_mean"), 1)])
    deck.table(s, table, .7, 2.47, [2.0, 2.35, 2.75, 2.3, 2.15], .39, 12)
    deck.text(s, "零信息消融可能保留未用分支参数，并非实际减参。D7为经数值验证的等价FP32卷积backend；Bay/D4为原生backend。", .7, 6.6, 11.7, .35, 11, MUTED)

    s = deck.slide("结论：分支有贡献，仍可能输给完整来源模型", "自身消融和完整来源对照同时报告；正数为full更好，负数为full更差", True)
    deck.text(s, "相对自身较优单分支", 5.75, 2.43, 3.0, .35, 13, "B8CCC8", True, PP_ALIGN.CENTER)
    deck.text(s, "相对完整来源模型", 9.1, 2.43, 3.0, .35, 13, "B8CCC8", True, PP_ALIGN.CENTER)
    source_comparisons = {row["dataset"]: row for row in summary["source_model_comparisons"]}
    for i, d in enumerate(DATASETS):
        c = pipelines[d]["composition"]
        ref = summary["test_posthoc_best_single"][d]
        g, spread, n = gain(pairs, d, ref) if ref else (None, None, 0)
        y = 2.95 + i * .98
        deck.rect(s, .7, y, 11.7, .85, "2B3D3E", True)
        deck.text(s, d, .95, y + .23, 1.8, .4, 20, "FFFFFF", True)
        deck.text(s, c.get("model_name", "流程进行中"), 2.8, y + .15, 2.9, .4, 17, "DCE8E3")
        deck.text(s, f"{g:+.2f}%" if g is not None else "待实测", 5.9, y + .18, 2.8, .5, 26,
                  "65D3B5" if g is not None and g >= 0 else "F1B37A", True, PP_ALIGN.CENTER)
        comparison = source_comparisons[d]
        if comparison["source_allocated_parameters"] is not None:
            deck.text(s, f"参数 {comparison['full_allocated_parameters']:,.0f} / 来源 {comparison['source_allocated_parameters']:,.0f}",
                      2.8, y + .59, 2.95, .2, 9.5, "B8CCC8")
        source_gain = comparison["paired_gain_mean"]
        deck.text(s, f"{source_gain:+.2f}%" if source_gain is not None else "待完整3seed", 9.1, y + .1, 3.0, .5,
                  26 if source_gain is not None else 17, "65D3B5" if source_gain is not None and source_gain >= 0 else "F1B37A", True, PP_ALIGN.CENTER)
        reference = comparison["reference_variant"]
        deck.text(s, short(reference) if reference else "未设完整来源", 9.1, y + .59, 3.0, .2, 10.5, "B8CCC8", align=PP_ALIGN.CENTER)
    deck.text(s, "较优单分支固定，属测试集事后描述。结构选择在最终基准前冻结。\n容量/损失/优化差异限制机制归因，但不能抹去完整来源模型更好的实测负结果。", .75, 6.05, 11.7, .75, 14, "B8CCC8")

    s = deck.slide("可追溯交付与研究局限", "真实流程产物、冻结源码、逐次训练和复用对照分别可核对")
    evidence = summary["verification_evidence"]
    cards = [("流程与结构", "data/adaptation/compose_*/\ncomposition_evidence.csv\nfrozen/<dataset>/"),
             ("新增实测", "runs/*.json / checkpoints/*.pt\ngroup_summary.csv\npaired_comparisons.csv"),
             ("记录与独立复核", f"verification.json：{evidence['record_audit']['found_count']}条记录\ncheckpoint_recheck.json：{evidence['checkpoint_recheck']['count']}项复算\npipeline_design_review.json")]
    for i, (title, body) in enumerate(cards):
        x = .7 + i * 4.15
        deck.rect(s, x, 2.65, 3.8, 2.5, "FFFFFF", True)
        deck.text(s, title, x + .2, 2.95, 3.4, .5, 23, TEAL, True)
        deck.text(s, body, x + .2, 3.75, 3.4, 1.0, 13, INK)
    deck.text(s, "局限：一次流程选择；固定预算；来源模型非等参数/等损失；零值mask；3 seeds。\n执行：Bay/D4原生卷积，D7等价FP32 lowering；耗时仅工程参考。", .7, 5.65, 11.7, .72, 14, INK, True)
    deck.text(s, "result/dataset_composition_20261027\n重建：python scripts/build_dataset_composition_report.py --require-complete --render", .7, 6.38, 11.7, .5, 12, MUTED)

    s = deck.slide("附录：原始METR_LA组合案例仍保留", "早期独立流程；历史10%/5%/85%窗口划分，不与本轮三个数据集的结果混表")
    deck.rect(s, .7, 2.5, 11.7, .65, "D9EEE4", True)
    deck.text(s, "Embedding →（DiffGCN ‖ GRU）→ Concat → Linear head", .95, 2.66, 11.2, .4, 20, TEAL, True, PP_ALIGN.CENTER)
    deck.table(s, [["历史15epoch变体", "masked_MAE", "统计口径"],
                  ["完整STDiffGRU组合", "3.6313", "3 seeds均值"], ["仅时间分支", "3.6757", "3 seeds均值"],
                  ["仅空间分支", "3.7302", "3 seeds均值"], ["历史原生RNN", "4.6083", "单seed弱基线"]],
               .7, 3.6, [5.1, 2.3, 4.3], .43, 14)
    deck.text(s, "历史21.2%改善只相对单seed RNN，不能称胜过完整来源模型。\n保留这个原始案例作为锚点；本轮新增证据仍是三个数据集各自的真实流程组合。", .7, 6.12, 11.7, .65, 14, MUTED)

    # Notes preserve the actual rationale instead of fabricating a Chinese causal explanation.
    for i, slide in enumerate(deck.prs.slides, 1):
        note = ["本页来自真实AgentCity各数据集流程产物及冻结结构后的实测。",
                "画像仅训练时段；pilot与最终15epoch/3seed基准分开。来源完整模型非参数/损失/优化器严格匹配。",
                "旧固定STDiffGRU及旧GRU明确标为复用对照，不计新增训练；旧固定结构只有原始切分/scaler、输入通道和图SHA可证一致时才作配对。",
                "配对增益=100*(对照MAE-fullMAE)/对照MAE。所有MAE为反归一化后12horizon masked_MAE均值；std为样本标准差。"]
        note += ["零信息消融可能保留分配参数：allocated、active、disabled、removed参数分别报告，不能把有效参数下降称为实际减参。",
                 "各数据集使用各自冻结trainer；Bay/D4原生卷积，D7采用经前向、梯度与整模型Adam对齐验证的等价FP32时间卷积lowering。源码/参数/精度/预算不改，耗时仅工程参考。",
                 "原始pipeline未mask MAPE的Infinity/NaN仅在派生summary显式字符串化并记录路径；新run和masked主要指标非有限仍必须失败。"]
        note += ["结论同时报告自身分支贡献和完整来源模型表现。完整来源对照按模块来源与预先声明矩阵确定，不按测试成绩选取；组合弱于完整来源模型的负结果保留。",
                 "verification.json是全矩阵记录/权重文件审计；checkpoint_recheck.json是9个seed0核心checkpoint用原生CPU算子独立重新推理，其中D7使用原生Conv2d而非lowering。两者范围不同。",
                 "pipeline_design_review.json确认D4的TCN感受野调整在首次pilot前完成；D7仅更正输入说明，构造与forward AST未变。"]
        if 6 <= i <= 11:
            d = DATASETS[(i - 6) // 2]
            p = pipelines[d]
            note += [f"实际证据：{json.dumps(p['paths'], ensure_ascii=False)}",
                     f"组合器原始理由：{json.dumps(p['composition'].get('rationale', {}), ensure_ascii=False)}"]
        if i == 21:
            note += ["历史来源：documentation/metr_la_speed_smoke_adaptation_summary.md的2026-10-15追加实验。",
                     "历史完整/仅时序/仅空间为3seed均值3.6313/3.6757/3.7302；RNN单seed4.6083。约21.2%相对弱RNN，不是完整来源模型比较，也不是本轮新画像选结构。"]
        if i == 19:
            note.append("完整来源模型对照的实测统计：" + json.dumps(summary["source_model_comparisons"], ensure_ascii=False))
        slide.notes_slide.notes_text_frame.text = "\n".join(note)
    name = "真实流程组合与消融_多数据集实验" + ("" if summary["complete"] else "_DRAFT") + ".pptx"
    path = directory / name
    deck.prs.save(path)
    content, bounds = [], []
    for index, slide in enumerate(deck.prs.slides, 1):
        content.append(f"## 第{index}页")
        for shape in slide.shapes:
            if shape.has_text_frame:
                content.append(shape.text)
            if shape.has_table:
                content += [" | ".join(cell.text for cell in row.cells) for row in shape.table.rows]
            if shape.left < 0 or shape.top < 0 or shape.left + shape.width > deck.prs.slide_width + 10 or shape.top + shape.height > deck.prs.slide_height + 10:
                bounds.append({"slide": index, "shape": shape.name})
    text = "\n\n".join(content)
    (directory / "presentation_text.txt").write_text(text)
    leftovers = [x for x in ("DRAFT", "待完成", "流程尚未完成", "生成中", "待实测") if summary["complete"] and x in text]
    audit = {"slides": len(deck.prs.slides), "out_of_bounds": bounds, "leftover_placeholders": leftovers,
             "new_completed_runs": summary["new_completed_runs"], "reused_supplement_runs": summary["reused_supplement_runs"],
             "complete": summary["complete"], "editable_charts": sum(shape.has_chart for slide in deck.prs.slides for shape in slide.shapes)}
    (directory / "presentation_content_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    if bounds or leftovers:
        raise ValueError(f"presentation QA failed: bounds={bounds}; placeholders={leftovers}")
    return path, len(deck.prs.slides)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "result/dataset_composition_20261027")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    summary, groups, rows, records, pairs, variants = collect(directory, args.require_complete)
    write_report(directory, summary, groups, pairs, variants)
    ppt, count = build_slides(directory, summary, groups, rows, records, pairs, variants)
    rendered = render_ppt(ppt) if args.render else None
    print(json.dumps({"complete": summary["complete"], "new_training": summary["new_completed_runs"],
                      "reused_controls": summary["reused_supplement_runs"], "slides": count,
                      "rendered_pages": rendered, "pptx": str(ppt)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
