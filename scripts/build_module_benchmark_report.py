"""Rebuild the module benchmark's auditable tables, Chinese report and slides.

Usage: python scripts/build_module_benchmark_report.py --require-complete --render
The presentation uses editable PowerPoint text, tables, shapes and charts.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION, XL_TICK_LABEL_POSITION
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("METR_LA", "PEMS_BAY", "PEMSD4", "PEMSD7")
VARIANTS = ("concat", "add", "weighted", "temporal_only", "spatial_only", "native_gru")
LABEL = {"concat": "拼接融合", "add": "相加融合", "weighted": "加权融合",
         "temporal_only": "仅 GRU", "spatial_only": "仅 DiffGCN", "native_gru": "原生 GRU"}
SHORT = {"concat": "Concat", "add": "Add", "weighted": "Weighted",
         "temporal_only": "GRU only", "spatial_only": "DiffGCN only", "native_gru": "Native GRU"}
COLORS = {"concat": "099E89", "add": "55B7A6", "weighted": "246C64",
          "temporal_only": "E59243", "spatial_only": "B96F39", "native_gru": "87929A"}
CHARCOAL, INK, TEAL, ORANGE = "1F2B2D", "263B3D", "099E89", "E59243"
PAPER, MUTED, RULE = "F4F6F3", "647575", "DAE2DD"
FONT = "Noto Sans CJK SC"


def avg(values):
    return statistics.mean(values) if values else None


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else None


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def csv_write(path, rows, fields=None):
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(directory, require_complete=False):
    records = [json.loads(path.read_text()) for path in sorted((directory / "runs").glob("*.json"))]
    seen, hashes = set(), set()
    for record in records:
        key = record["dataset"], record["variant"], record["seed"]
        if key in seen or key[0] not in DATASETS or key[1] not in VARIANTS or key[2] not in (0, 1, 2):
            raise ValueError(f"unexpected/duplicate experiment: {key}")
        seen.add(key)
        hashes.add(record["protocol_sha256"])
        assert record["status"] == "success"
        assert len(record["history"]) == record["epochs"] == 15
        assert len(record["test"]["mae_by_horizon"]) == 12
        assert math.isclose(avg(record["test"]["mae_by_horizon"]), record["test"]["masked_mae"], abs_tol=1e-10)
        best = min(record["history"], key=lambda h: h["val_masked_mae"])
        assert record["best_epoch"] == best["epoch"]
        assert math.isclose(best["val_masked_mae"], record["best_val_masked_mae"], abs_tol=1e-10)
        assert all(math.isfinite(record["test"][key]) for key in ("masked_mae", "masked_rmse", "masked_mape"))
        assert (ROOT / record["checkpoint"]).is_file()
    if len(hashes) > 1:
        raise ValueError("result files contain different experiment protocols")
    missing = [f"{d}_{v}_s{s}" for d in DATASETS for v in VARIANTS for s in (0, 1, 2)
               if (d, v, s) not in seen]
    if require_complete and missing:
        raise ValueError(f"expected 72 completed runs; found {len(records)}; missing {missing}")
    groups = {}
    flat, horizons = [], []
    for record in records:
        test = record["test"]
        flat.append({"dataset": record["dataset"], "variant": record["variant"], "seed": record["seed"],
                     "masked_mae": test["masked_mae"], "masked_rmse": test["masked_rmse"],
                     "masked_mape_ratio": test["masked_mape"], "mae_h3": test["mae_by_horizon"][2],
                     "mae_h6": test["mae_by_horizon"][5], "mae_h12": test["mae_by_horizon"][11],
                     "best_val_masked_mae": record["best_val_masked_mae"],
                     "best_epoch": record["best_epoch"], "parameters": record["parameters"],
                     "wall_seconds": record["wall_seconds"], "gpu": record["gpu"],
                     "device_index": record["device_index"], "source_json": f"runs/{record['name']}.json"})
        for h in range(12):
            horizons.append({"dataset": record["dataset"], "variant": record["variant"],
                             "seed": record["seed"], "horizon": h + 1, "minutes_ahead": 5 * (h + 1),
                             "masked_mae": test["mae_by_horizon"][h],
                             "masked_rmse": test["rmse_by_horizon"][h],
                             "masked_mape_ratio": test["mape_by_horizon"][h],
                             "valid_count": test["valid_counts_by_horizon"][h]})
    group_rows = []
    for d in DATASETS:
        for v in VARIANTS:
            current = [r for r in flat if r["dataset"] == d and r["variant"] == v]
            if not current:
                continue
            result = {"dataset": d, "variant": v, "n_seeds": len(current)}
            for metric in ("masked_mae", "masked_rmse", "masked_mape_ratio", "mae_h3", "mae_h6", "mae_h12",
                           "best_val_masked_mae", "wall_seconds", "parameters"):
                values = [r[metric] for r in current]
                result[metric + "_mean"], result[metric + "_std"] = avg(values), sd(values)
            groups[(d, v)] = result
            group_rows.append(result)
    paired, decisions = [], {}
    lookup = {(r["dataset"], r["variant"], r["seed"]): r for r in flat}
    for d in DATASETS:
        # Only select a baseline when all three seeds of both single branches exist.
        singles = [v for v in ("temporal_only", "spatial_only") if groups.get((d, v), {}).get("n_seeds") == 3]
        combinations = [v for v in VARIANTS[:3] if groups.get((d, v), {}).get("n_seeds") == 3]
        single = min(singles, key=lambda v: groups[(d, v)]["best_val_masked_mae_mean"]) if len(singles) == 2 else None
        combination = min(combinations, key=lambda v: groups[(d, v)]["best_val_masked_mae_mean"]) if len(combinations) == 3 else None
        observed_single = min(singles, key=lambda v: groups[(d, v)]["masked_mae_mean"]) if len(singles) == 2 else None
        decisions[d] = {"validation_selected_single": single, "validation_selected_combination": combination,
                        "test_observed_best_single": observed_single,
                        "test_single_comparison_note": "fixed variant selected by mean test MAE over seeds; post-hoc descriptive comparison"}
        for v in VARIANTS:
            references = [("concat", "predefined concat reference"),
                          ("temporal_only", "predefined GRU-only branch ablation"),
                          ("spatial_only", "predefined DiffGCN-only branch ablation")]
            for ref, rule in references:
                for seed in (0, 1, 2):
                    a, b = lookup.get((d, v, seed)), lookup.get((d, ref, seed))
                    if a and b:
                        paired.append({"dataset": d, "variant": v, "reference_variant": ref,
                                       "reference_rule": rule, "seed": seed,
                                       "is_test_observed_best_single": ref == observed_single,
                                       "is_validation_selected_single": ref == single,
                                       "candidate_test_mae": a["masked_mae"], "reference_test_mae": b["masked_mae"],
                                       "improvement_absolute": b["masked_mae"] - a["masked_mae"],
                                       "improvement_percent": 100 * (b["masked_mae"] - a["masked_mae"]) / b["masked_mae"]})
    profiles = {d: json.loads((directory / f"{d}_profile.json").read_text()) for d in DATASETS
                if (directory / f"{d}_profile.json").exists()}
    manifests = [json.loads(p.read_text()) for p in sorted(directory.glob("manifest_gpu*.json"))]
    checkpoint_recheck_file = directory / "checkpoint_recheck.json"
    checkpoint_recheck = json.loads(checkpoint_recheck_file.read_text()) if checkpoint_recheck_file.exists() else {}
    evidence = {"matrix_audit": "verification.json", "provenance": "provenance.json", "source_snapshot": "source/",
                "checkpoint_recheck": "checkpoint_recheck.json",
                "checkpoint_recheck_count": len(checkpoint_recheck.get("checked_runs", [])),
                "checkpoint_recheck_status": checkpoint_recheck.get("status", "not available")}
    result = {"complete": not missing, "completed_runs": len(records), "expected_runs": 72,
              "missing_runs": missing, "protocol_sha256": sorted(hashes), "groups": group_rows,
              "validation_choices": decisions, "profiles": profiles,
              "total_training_wall_seconds": sum(r["wall_seconds"] for r in flat),
              "paired_gain_formula": "100 * (reference_test_MAE - candidate_test_MAE) / reference_test_MAE; fixed reference variant across seeds",
              "std_definition": "sample standard deviation across seeds (ddof=1); n=1 is undefined",
              "manifests": manifests, "verification_evidence": evidence}
    csv_write(directory / "run_metrics.csv", flat)
    csv_write(directory / "metrics_by_horizon.csv", horizons)
    csv_write(directory / "group_summary.csv", group_rows)
    csv_write(directory / "paired_comparisons.csv", paired)
    (directory / "results_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result, groups, flat, paired


def paired_gain(paired, dataset, variant, reference):
    values = [r["improvement_percent"] for r in paired if r["dataset"] == dataset
              and r["variant"] == variant and r["reference_variant"] == reference]
    return avg(values), sd(values), len(values)


def write_report(directory, summary, groups, paired):
    state = "完整实测" if summary["complete"] else "DRAFT：实验仍在运行"
    lines = [f"# 跨数据集模块组合与单模块实验（{state}）", "",
             f"已完成 {summary['completed_runs']} / 72 次训练；4 个数据集 × 6 个变体 × 3 seeds。",
             "", "## 实验协议", "",
             "METR_LA、PEMS_BAY 预测速度；PEMSD4、PEMSD7 只预测 traffic_flow，避免混合单位。所有数据保留原始标度，不假定速度的具体物理单位。",
             "原始时间步按 60% / 20% / 20% 连续切分，再分别构造 12→12 的滑动窗口；各分区的原始时间戳互不重叠。标准化参数仅由训练时段的原始目标通道拟合（包含零值）。输入是目标通道和 time_in_day。",
             "每次训练 15 epochs，batch=128，hidden=32，Adam lr=0.01，在第 5 个 epoch 后乘 0.1；梯度范数裁剪 5。seed=0/1/2，float32，TF32 关闭。用验证集全元素 pooled masked_MAE 最小的 epoch 选 checkpoint，然后一次性评估测试集。",
             "主指标是反归一化后 12 个 horizon 的 masked_MAE 算术平均，零目标被 mask；masked_RMSE 是逐 horizon RMSE 的算术平均，MAPE 在 CSV 中以比例记录。均值±标准差为三个 seed 的均值和样本标准差（ddof=1），不作显著性推断。",
             "", "## 模块定义", "",
             "三种组合都采用共享 encoder → 并行 DiffusionGCN / GRU → fusion → 线性 head，仅 fusion 不同。Concat 为拼接后线性映射；Add 为相加；Weighted 用两个 softmax 标量对分支加权。",
             "单模块消融保留 encoder / concat fusion / head，将被禁用分支置零。仅空间表示没有 GRU，但 head 仍读取所有历史时间步；不能说完全不使用历史信息。不同变体参数量不同，不属于参数严格匹配实验。原生 GRU 是 LibCity 的跨节点展平、递归预测模型，属于外部结构基线；它从 y 读取未来已知的 time_in_day 辅助特征，不使用未来交通目标。主要单模块对照始终是两个严格分支消融，原生 GRU 单独作为结构基线解释。",
             "", "## 测试集主指标", "",
             "| 变体 | METR_LA（速度） | PEMS_BAY（速度） | PEMSD4（流量） | PEMSD7（流量） |", "|---|---:|---:|---:|---:|"]
    for v in VARIANTS:
        cells = []
        for d in DATASETS:
            g = groups.get((d, v))
            cells.append(f"{fmt(g['masked_mae_mean'])} ± {fmt(g['masked_mae_std'])} (n={g['n_seeds']})" if g else "待完成")
        lines.append("| " + LABEL[v] + " | " + " | ".join(cells) + " |")
    lines += ["", "## 组合相对单分支的配对增益", "",
              "每个数据集的“最佳单模块”固定为两个严格单分支消融中三 seed 平均测试 MAE 更低的变体，不能每个 seed 更换对照。这是测试集事后描述性比较，不用于验证选型。三种融合中的组合选择仍依据三 seed 平均验证 MAE。随后在相同 seed 上比较测试 MAE；正数表示组合更好，负数表示更差。公式为 `100 × (对照测试MAE − 组合测试MAE) / 对照测试MAE`，再对三 seed 取均值和样本标准差。CSV 同时保留每种组合对每种单分支的完整比较，以及验证集所选单分支标记。", ""]
    for d in DATASETS:
        choices = summary["validation_choices"][d]
        v, ref = choices["validation_selected_combination"], choices["test_observed_best_single"]
        if v and ref:
            gain, std, n = paired_gain(paired, d, v, ref)
            lines.append(f"- **{d}**：验证集选择 {LABEL[v]}；测试均值较优的固定单分支对照为 {LABEL[ref]}；测试配对增益 {gain:+.2f}% ± {std:.2f}%（n={n}；事后描述性对照）。")
        else:
            lines.append(f"- **{d}**：仍需完整三 seed 结果，暂不作比较结论。")
    lines += ["", "完整的组合 × 单分支比较如下（同 seed 配对，正数为组合更好）：", "",
              "| 数据集 | 组合 | 单分支对照 | 平均增益 ± 样本标准差 | seed数 |",
              "|---|---|---|---:|---:|"]
    for d in DATASETS:
        for v in VARIANTS[:3]:
            for ref in ("temporal_only", "spatial_only"):
                gain, spread, n = paired_gain(paired, d, v, ref)
                if n:
                    lines.append(f"| {d} | {LABEL[v]} | {LABEL[ref]} | {gain:+.2f}% ± {fmt(spread, 2)}% | {n} |")
    lines += ["", "## 成本与局限", "",
              f"已记录训练/验证/测试总耗时 {summary['total_training_wall_seconds']/3600:.2f} GPU-worker 小时（运行的 wall time 加总，不是整机自然时间）。数据加载不包含在单次耗时内。各变体使用同型号 GPU；耗时用于工程参考。",
              "15 epochs 是固定预算，不代表完全收敛；3 seeds 仅刻画波动。未调每个变体的专属超参数，不宣称最佳可达精度。零值 mask 的含义沿用 LibCity 约定，真实零流量也可能被忽略。速度和流量的绝对 MAE 不跨数据集求总平均。",
              "历史 METR_LA 实验采用 10%/5%/85% 的窗口切分，与本次先切原始时间的 60%/20%/20% 协议不同，未混入本报告。",
              "", "## 结果证据与复现", "",
              "- `runs/*.json`：每次实测的训练曲线、验证最优 epoch、测试逐 horizon 指标。",
              "- `checkpoints/*.pt`：验证集选择的模型权重。",
              "- `manifest_gpu*.json`：实验协议、软件版本、代码哈希。",
              "- `verification.json`：全部 72 次训练记录的矩阵完整性、epoch、参数量、指标均值和哈希全量审计；这不等于对 72 个 checkpoint 全部重新推理。",
              f"- `checkpoint_recheck.json`：独立重新载入 {summary['verification_evidence']['checkpoint_recheck_count']} 个 checkpoint 复算（4 数据集 × concat/native_gru × seed0），使用原始 LibCity 指标函数核对，并检查未来交通目标扰动不影响预测；记录状态为 `{summary['verification_evidence']['checkpoint_recheck_status']}`。该复算范围与全矩阵记录审计明确区分。",
              "- `provenance.json` + `source/`：原始数据指纹、源码快照及溯源记录。",
              "- `*_profile.json`：节点数、时间范围、样本量、scaler 参数和零值比例。",
              "- `run_metrics.csv` / `group_summary.csv` / `metrics_by_horizon.csv` / `paired_comparisons.csv`：可核对的明细与统计。",
              "- `results_summary.json`：完整性审计与验证集选择记录。", "",
              "重建全部交付物：", "", "```bash",
              "/home/wangwenrui/opt/miniconda3/envs/autogen/bin/python scripts/build_module_benchmark_report.py --require-complete --render",
              "```", ""]
    if summary["missing_runs"]:
        lines += ["尚未完成的训练：", "", ", ".join(summary["missing_runs"]), ""]
    (directory / "实验结果报告.md").write_text("\n".join(lines))


def rgb(hex_color):
    return RGBColor.from_string(hex_color)


class Deck:
    def __init__(self, directory, summary):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(13.333), Inches(7.5)
        self.prs.core_properties.title = "多数据集模块组合与单模块性能实验"
        self.prs.core_properties.subject = "72 次实测；组合、消融与原生 GRU 基线"
        self.directory, self.summary = directory, summary
        self.number = 0

    def text(self, slide, text, x, y, w, h, size=17, color=INK, bold=False, align=None):
        box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame
        tf.clear()
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = 0
        tf.margin_top = tf.margin_bottom = 0
        for index, line in enumerate(str(text).split("\n")):
            p = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
            p.text = line
            p.font.name, p.font.size = FONT, Pt(size)
            p.font.bold, p.font.color.rgb = bold, rgb(color)
            p.space_after = Pt(7)
            if align is not None:
                p.alignment = align
            for run in p.runs:
                props = run._r.get_or_add_rPr()
                east = OxmlElement("a:ea")
                east.set("typeface", FONT)
                props.append(east)
        return box

    def rect(self, slide, x, y, w, h, fill, radius=False):
        shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius else MSO_AUTO_SHAPE_TYPE.RECTANGLE,
                                      Inches(x), Inches(y), Inches(w), Inches(h))
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(fill)
        shape.line.fill.background()
        if radius:
            shape.adjustments[0] = 0.08
        return shape

    def slide(self, title, subtitle="", dark=False):
        self.number += 1
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = rgb(CHARCOAL if dark else PAPER)
        self.text(slide, "AGENTCITY  /  MODULE BENCHMARK", .6, .3, 10, .25, 10, "8ECBBF" if dark else MUTED, True)
        self.text(slide, title, .6, .85, 12.1, .8, 36, "FFFFFF" if dark else INK, True)
        if subtitle:
            self.text(slide, subtitle, .62, 1.76, 12.0, .6, 14, "B8CCC8" if dark else MUTED)
        status = "完整实测" if self.summary["complete"] else f"DRAFT · {self.summary['completed_runs']}/72 已完成"
        self.text(slide, f"2026.10.26   |   {status}   |   主指标越低越好", .6, 7.02, 11.5, .22, 9.5, "B8CCC8" if dark else MUTED)
        self.text(slide, f"{self.number:02d}", 12.1, 6.95, .6, .35, 12, "B8CCC8" if dark else MUTED, align=PP_ALIGN.RIGHT)
        return slide

    def table(self, slide, rows, x, y, widths, row_height=.5, fontsize=13, highlight=None):
        shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(sum(widths)), Inches(row_height * len(rows)))
        table = shape.table
        for i, width in enumerate(widths):
            table.columns[i].width = Inches(width)
        for ri, row in enumerate(rows):
            for ci, value in enumerate(row):
                cell = table.cell(ri, ci)
                cell.text = str(value)
                cell.margin_left = Inches(.12)
                cell.margin_right = Inches(.08)
                cell.margin_top = Inches(.07)
                cell.margin_bottom = Inches(.03)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb(CHARCOAL if ri == 0 else ("E0F0E9" if highlight and highlight(ri, ci) else ("FFFFFF" if ri % 2 else "EAF0EB")))
                for p in cell.text_frame.paragraphs:
                    p.font.name, p.font.size = FONT, Pt(fontsize)
                    p.font.color.rgb = rgb("FFFFFF" if ri == 0 else INK)
                    p.font.bold = ri == 0 or bool(highlight and highlight(ri, ci))
                    p.space_after = Pt(0)
        return table

    def chart(self, slide, categories, series, x, y, w, h, kind="bar", colors=None, percent=False, legend=False):
        if not categories or not series:
            self.rect(slide, x, y, w, h, "E7EDE8", True)
            self.text(slide, "训练进行中\n完成后自动填入真实结果", x + .3, y + h / 2 - .5, w - .6, 1.2, 24, MUTED, True, PP_ALIGN.CENTER)
            return None
        data = CategoryChartData()
        data.categories = categories
        for name, values in series:
            data.add_series(name, values)
        chart_type = XL_CHART_TYPE.LINE_MARKERS if kind == "line" else XL_CHART_TYPE.COLUMN_CLUSTERED
        chart = slide.shapes.add_chart(chart_type, Inches(x), Inches(y), Inches(w), Inches(h), data).chart
        chart.has_title = False
        chart.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
        chart.has_legend = legend
        if legend:
            chart.legend.position = XL_LEGEND_POSITION.BOTTOM
            chart.legend.include_in_layout = False
            chart.legend.font.size = Pt(10)
            chart.legend.font.name = FONT
        for axis in (chart.category_axis, chart.value_axis):
            axis.tick_labels.font.name, axis.tick_labels.font.size = FONT, Pt(10)
            axis.tick_labels.font.color.rgb = rgb(MUTED)
            axis.format.line.fill.background()
        chart.value_axis.has_major_gridlines = True
        if kind == "bar" and not percent:
            chart.value_axis.minimum_scale = 0
        chart.value_axis.major_gridlines.format.line.color.rgb = rgb(RULE)
        chart.value_axis.tick_labels.number_format = '0.0"%"' if percent else "0.0"
        plot = chart.plots[0]
        if kind == "bar":
            plot.gap_width = 65
            plot.has_data_labels = True
            plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
            plot.data_labels.font.name, plot.data_labels.font.size = FONT, Pt(10)
            plot.data_labels.number_format = '0.0"%"' if percent else "0.00"
        palette = colors or [TEAL, ORANGE, "246C64", "87929A", "B96F39", "55B7A6"]
        for i, s in enumerate(chart.series):
            c = palette[i % len(palette)]
            s.format.line.color.rgb = rgb(c)
            s.format.line.width = Pt(2)
            if kind == "bar":
                s.format.fill.solid()
                s.format.fill.fore_color.rgb = rgb(c)
        return chart

    def card(self, slide, label, number, x, y, w, color=TEAL, dark=False):
        self.rect(slide, x, y, w, 1.55, "2B3D3E" if dark else "FFFFFF", True)
        self.text(slide, number, x + .23, y + .12, w - .46, .85, 44, color, True)
        self.text(slide, label, x + .23, y + 1.05, w - .46, .32, 14, "DCE8E3" if dark else MUTED)

    def connector(self, slide, start, end, arrow=False):
        line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(start[0]), Inches(start[1]),
                                          Inches(end[0]), Inches(end[1]))
        line.line.color.rgb, line.line.width = rgb(TEAL), Pt(2)
        if arrow:
            tail = OxmlElement("a:tailEnd")
            tail.set("type", "triangle")
            tail.set("w", "sm")
            tail.set("len", "sm")
            line.line._get_or_add_ln().append(tail)
        return line


def build_ppt(directory, summary, groups, flat, paired):
    deck = Deck(directory, summary)
    s = deck.slide("多数据集模块组合与单模块性能实验", "统一训练预算 · 同时评估组合增益、单分支能力与融合方式", True)
    for i, (label, number) in enumerate((("数据集覆盖", "4"), ("模块配置", "6"), ("随机种子", "3"), ("已完成训练", f"{summary['completed_runs']}/72"))):
        deck.card(s, label, number, .65 + i * 3.1, 2.8, 2.7, ORANGE if i == 3 else "65D3B5", True)
    deck.text(s, "速度：METR_LA / PEMS_BAY     流量：PEMSD4 / PEMSD7", .7, 5.05, 11.8, .5, 21, "FFFFFF", True)
    deck.text(s, "所有结论来自本轮实测；历史冒烟结果不混入统一比较。", .7, 5.82, 11.8, .6, 16, "B8CCC8")

    s = deck.slide("实验覆盖：从单数据集扩展到四个", "每个数据集 18 次训练；同时涵盖速度与流量目标")
    rows = [["数据集", "预测目标", "节点", "时间步", "train / val / test 窗口"]]
    for d in DATASETS:
        p = summary["profiles"].get(d, {})
        rows.append([d, "速度" if d in DATASETS[:2] else "流量", p.get("nodes", "读取中"), p.get("timesteps", "—"),
                     " / ".join(map(str, p.get("window_counts", []))) or "读取中"])
    deck.table(s, rows, .65, 2.5, [2.1, 1.4, 1.1, 1.7, 5.7], .65, 15)
    deck.text(s, "4 数据集 ×（3 种融合组合 + 2 种单分支 + 1 个原生基线）× 3 seeds = 72", .7, 6.17, 11.9, .5, 20, TEAL, True)

    s = deck.slide("组合与单模块：共享骨架，控制替换位置", "消融衡量分支信息贡献；三种融合组合衡量融合策略的影响")
    boxes = [("输入\n12步 × 节点 × 2特征", .7, 3.35, 2.1, 1.0), ("Encoder", 3.12, 3.35, 1.65, 1.0),
             ("DiffusionGCN", 5.1, 2.65, 2.2, .85), ("GRU", 5.1, 4.03, 2.2, .85),
             ("Fusion\nConcat / Add / Weighted", 7.68, 3.25, 2.65, 1.2), ("Head\n未来12步", 10.73, 3.35, 1.8, 1.0)]
    for text, x, y, w, h in boxes:
        deck.rect(s, x, y, w, h, "D9EEE4" if "Fusion" not in text else "F4E3CB", True)
        deck.text(s, text, x + .1, y + .16, w - .2, h - .1, 15, INK, True, PP_ALIGN.CENTER)
    deck.connector(s, (2.8, 3.85), (3.12, 3.85), True)
    deck.connector(s, (4.77, 3.85), (4.93, 3.85))
    deck.connector(s, (4.93, 3.075), (4.93, 4.455))
    for center in (3.075, 4.455):
        deck.connector(s, (4.93, center), (5.1, center), True)
        deck.connector(s, (7.3, center), (7.48, center))
    deck.connector(s, (7.48, 3.075), (7.48, 4.455))
    deck.connector(s, (7.48, 3.85), (7.68, 3.85), True)
    deck.connector(s, (10.33, 3.85), (10.73, 3.85), True)
    deck.text(s, "仅 GRU：空间分支置零     |     仅 DiffGCN：GRU 分支置零", .7, 5.37, 11.9, .5, 19, TEAL, True)
    deck.text(s, "原生 GRU：LibCity 跨节点展平 + 递归输出，作为额外结构基线。\n各变体参数量不同；“仅空间”仍由 Head 读取全部历史步。", .7, 6.04, 11.9, .68, 14, MUTED)

    s = deck.slide("统一协议：先切时间，再造窗口", "训练 / 验证 / 测试原始时间戳互不重叠；测试集不参与 checkpoint 选择")
    x = .7
    for width, label, color in ((7.1, "训练 60%", TEAL), (2.3, "验证 20%", "55B7A6"), (2.3, "测试 20%", ORANGE)):
        deck.rect(s, x, 2.65, width, .85, color, True)
        deck.text(s, label, x + .15, 2.87, width - .3, .35, 20, "FFFFFF", True, PP_ALIGN.CENTER)
        x += width + .15
    deck.table(s, [["固定项", "设置", "控制目的"], ["窗口 / 特征", "12 → 12；目标 + time_in_day", "相同可用信息"],
                  ["训练预算", "15 epochs；batch 128；hidden 32", "相同优化预算"],
                  ["优化器", "Adam 0.01；第5轮后 × 0.1", "统一学习率计划"],
                  ["选模与指标", "验证 pooled masked_MAE；测试 mean-horizon masked_MAE", "测试只用于报告"]], .7, 3.95, [2.0, 6.0, 3.7], .49, 13)
    deck.text(s, "标准化只拟合训练原始时段；零值 mask 沿用 LibCity。seed = 0 / 1 / 2。", .7, 6.58, 11.7, .35, 12, MUTED)

    s = deck.slide("总表：组合与单模块的 masked_MAE", "均值 ± 样本标准差；每格 n=3 后才能用于完整比较；不同目标不汇总原始 MAE")
    rows = [["变体", "METR_LA", "PEMS_BAY", "PEMSD4", "PEMSD7"]]
    best = {}
    for d in DATASETS:
        vals = [(v, groups[(d, v)]["masked_mae_mean"]) for v in VARIANTS if (d, v) in groups]
        best[d] = min(vals, key=lambda pair: pair[1])[0] if vals else None
    for v in VARIANTS:
        row = [LABEL[v]]
        for d in DATASETS:
            g = groups.get((d, v))
            row.append(f"{fmt(g['masked_mae_mean'])} ± {fmt(g['masked_mae_std'])}" + (f" [n={g['n_seeds']}]" if g["n_seeds"] < 3 else "") if g else "待完成")
        rows.append(row)
    deck.table(s, rows, .65, 2.57, [2.2, 2.45, 2.45, 2.45, 2.45], .48, 13,
               lambda ri, ci: ci > 0 and best[DATASETS[ci - 1]] == VARIANTS[ri - 1])
    deck.text(s, "浅绿为当前测试均值最低项，仅用于描述；选模依据始终是验证集。", .7, 6.35, 11.8, .5, 15, TEAL)

    for d in DATASETS:
        s = deck.slide(f"{d}：六种配置的实测表现", "目标：" + ("速度（原始标度）" if d in DATASETS[:2] else "traffic_flow（原始标度）"))
        available = [v for v in VARIANTS if (d, v) in groups]
        deck.text(s, "masked_MAE ↓（原始标度）", .85, 2.32, 7.5, .35, 12, MUTED)
        chart = deck.chart(s, [SHORT[v] for v in available], [("测试 masked_MAE", [groups[(d, v)]["masked_mae_mean"] for v in available])], .6, 2.77, 8.25, 3.48)
        if chart:
            for i, v in enumerate(available):
                point = chart.series[0].points[i]
                point.format.fill.solid()
                point.format.fill.fore_color.rgb = rgb(COLORS[v])
        p = summary["profiles"].get(d, {})
        deck.card(s, "节点数", str(p.get("nodes", "—")), 9.35, 2.55, 3.0)
        choices = summary["validation_choices"][d]
        v, ref = choices["validation_selected_combination"], choices["test_observed_best_single"]
        if v and ref:
            gain, spread, n = paired_gain(paired, d, v, ref)
            txt = f"验证选择：{LABEL[v]}\n测试较优单分支：{LABEL[ref]}\n配对增益 {gain:+.2f}%\nseed标准差 {spread:.2f}%"
        else:
            count = sum(r["dataset"] == d for r in flat)
            txt = f"已完成 {count}/18 次\n等待完整三 seed 结果\n暂不作排名结论"
        deck.text(s, txt, 9.4, 4.5, 3.0, 1.65, 15, INK)
        deck.text(s, "柱高为跨 seed 均值；单分支对照按三 seed 测试均值固定，为事后描述性比较，不用于选型。", .7, 6.5, 11.8, .35, 11, MUTED)

    s = deck.slide("单模块贡献：组合是否带来可重复收益？", "对照为三 seed 测试均值较优的固定单分支；测试事后描述性比较；正数表示组合更好")
    available = [d for d in DATASETS if summary["validation_choices"][d]["test_observed_best_single"] and all((d, v) in groups and groups[(d, v)]["n_seeds"] == 3 for v in VARIANTS[:3])]
    series = [(SHORT[v], [paired_gain(paired, d, v, summary["validation_choices"][d]["test_observed_best_single"])[0] for d in available]) for v in VARIANTS[:3]] if available else []
    deck.chart(s, list(available), series, .7, 2.4, 11.8, 3.85, colors=[COLORS[v] for v in VARIANTS[:3]], percent=True, legend=True)
    deck.text(s, "增益 = 100 ×（对照 MAE − 组合 MAE）/ 对照 MAE；先逐 seed 计算，再取均值。\n不假定组合必然优于单模块；负结果同样保留。", .7, 6.35, 11.8, .55, 13, MUTED)

    s = deck.slide("融合策略：空间与时间信息如何结合？", "以预设 Concat 为参考；避免跨数据集直接平均速度与流量的绝对误差")
    complete_ds = [d for d in DATASETS if all(groups.get((d, v), {}).get("n_seeds") == 3 for v in VARIANTS[:3])]
    fusion_series = [(SHORT[v], [paired_gain(paired, d, v, "concat")[0] for d in complete_ds]) for v in ("add", "weighted")] if complete_ds else []
    deck.chart(s, complete_ds, fusion_series, .7, 2.45, 8.5, 3.8, colors=[COLORS["add"], COLORS["weighted"]], percent=True, legend=True)
    deck.rect(s, 9.7, 2.65, 2.8, 3.2, "E0F0E9", True)
    deck.text(s, "Concat\n线性投影，参数更多\n\nAdd\n无额外融合参数\n\nWeighted\n2个全局混合标量", 9.95, 2.9, 2.3, 2.8, 15, INK)
    deck.text(s, "这里报告的是实测差异；不将测试集最低者当作未经验证的新推荐配置。", .7, 6.4, 11.8, .45, 13, MUTED)

    s = deck.slide("预测跨度：短期与一小时预测是否一致？", "每格为不同数据集；曲线为三 seed 平均 masked_MAE，原始目标标度")
    for i, d in enumerate(DATASETS):
        x, y = .65 + (i % 2) * 6.25, 2.32 + (i // 2) * 2.13
        deck.text(s, d, x + .1, y, 5.75, .3, 15, INK, True)
        variants = [v for v in ("concat", "temporal_only", "spatial_only") if groups.get((d, v), {}).get("n_seeds") == 3]
        lines = []
        for v in variants:
            records = [r for r in json_records(directory) if r["dataset"] == d and r["variant"] == v]
            lines.append((SHORT[v], [avg([r["test"]["mae_by_horizon"][h] for r in records]) for h in range(12)]))
        deck.chart(s, list(range(5, 65, 5)) if lines else [], lines, x, y + .35, 6.0, 1.65, "line", colors=[COLORS[v] for v in variants], legend=i == 0)
    deck.text(s, "横轴：预测分钟（5–60）；Concat / GRU only / DiffGCN only。逐 horizon 的全部六变体结果保存在 CSV。", .7, 6.78, 11.8, .23, 10, MUTED)

    s = deck.slide("稳定性与成本：把精度收益放回预算中看", "标准差为三 seed 的样本标准差；耗时包含训练、验证和最终测试，不含数据加载")
    rows = [["数据集", "验证所选组合", "MAE样本标准差", "参数量", "平均单次耗时"]]
    for d in DATASETS:
        v = summary["validation_choices"][d]["validation_selected_combination"]
        g = groups.get((d, v)) if v else None
        rows.append([d, LABEL.get(v, "待完成"), fmt(g["masked_mae_std"]) if g else "—",
                     f"{g['parameters_mean']:,.0f}" if g else "—", f"{g['wall_seconds_mean']:.1f}s" if g else "—"])
    deck.table(s, rows, .7, 2.5, [2.0, 2.5, 2.7, 2.0, 2.5], .62, 14)
    deck.rect(s, .7, 6.0, 11.7, .65, "E0F0E9", True)
    deck.text(s, "相同 hidden 与 epoch ≠ 相同参数量、FLOPs或收敛程度；本轮衡量固定预算下的工程表现。", .93, 6.14, 11.25, .35, 14, TEAL, True)

    s = deck.slide("结论：用多数据集证据界定组合的收益", "组合按验证集选择；单分支对照按三 seed 测试均值固定，属于事后描述性比较", True)
    rows = []
    for d in DATASETS:
        c = summary["validation_choices"][d]
        v, ref = c["validation_selected_combination"], c["test_observed_best_single"]
        if v and ref:
            gain, spread, n = paired_gain(paired, d, v, ref)
            rows.append((d, f"{LABEL[v]} vs {LABEL[ref]}", f"{gain:+.2f}%", "65D3B5" if gain >= 0 else "F1B37A"))
        else:
            rows.append((d, "等待完整实验", "—", "B8CCC8"))
    for i, (d, description, gain, color) in enumerate(rows):
        y = 2.5 + i * .8
        deck.rect(s, .7, y, 11.8, .66, "2B3D3E", True)
        deck.text(s, d, .94, y + .13, 2.0, .4, 18, "FFFFFF", True)
        deck.text(s, description, 3.1, y + .16, 6.5, .35, 16, "DCE8E3")
        deck.text(s, gain, 10.1, y + .1, 2.1, .45, 24, color, True, PP_ALIGN.RIGHT)
    deck.text(s, "3 seeds 不支持严格显著性宣称；15 epochs 结果仅对应当前预算与统一超参数。", .75, 6.23, 11.7, .5, 15, "B8CCC8")

    s = deck.slide("可追溯交付：从每张图回到每次训练", "PPT 图表与文本均可编辑；CSV / JSON / checkpoint 提供复核入口")
    cards = [("逐次实测", "runs/*.json / checkpoints/*.pt\n每轮训练、验证和测试指标\n验证集最优模型权重"),
             ("统计复核", "group_summary.csv 等统计表\nverification.json：72次记录审计\ncheckpoint_recheck.json：8项复算"),
             ("协议与溯源", "manifest_gpu*.json / *_profile.json\nprovenance.json / source/\n实验协议、数据指纹与源码快照")]
    for i, (title, body) in enumerate(cards):
        x = .7 + i * 4.15
        deck.rect(s, x, 2.55, 3.8, 2.3, "FFFFFF", True)
        deck.text(s, title, x + .2, 2.8, 3.4, .45, 22, TEAL, True)
        deck.text(s, body, x + .2, 3.48, 3.4, 1.0, 12.5, INK)
    deck.text(s, "局限：固定预算；非参数匹配；零值 mask；统一超参数未逐模型调优。", .7, 5.4, 11.8, .45, 17, INK, True)
    deck.text(s, "结果目录：result/module_benchmark_20261026\n重建：python scripts/build_module_benchmark_report.py --require-complete --render", .7, 6.15, 11.8, .55, 12, MUTED)
    name = "模块组合与单模块_多数据集实验.pptx" if summary["complete"] else "模块组合与单模块_多数据集实验_DRAFT.pptx"
    path = directory / name
    text_dump = []
    bounds = []
    for index, slide in enumerate(deck.prs.slides, 1):
        slide.notes_slide.notes_text_frame.text = (
            f"本页由实测结果自动生成。完成数 {summary['completed_runs']}/72。\n"
            "数据源：result/module_benchmark_20261026/runs/*.json；"
            "统计：group_summary.csv、paired_comparisons.csv、metrics_by_horizon.csv。\n"
            "MAE为12个预测horizon的masked_MAE算术平均；误差均在反归一化后计算。"
            "样本标准差ddof=1。\n"
            "组合按三seed平均验证MAE选择。最佳单分支为三seed平均测试MAE较低的固定变体，"
            "只用于事后描述比较。配对增益=100*(对照测试MAE-组合测试MAE)/对照测试MAE，正数为组合更好。\n"
            "消融保持encoder/fusion/head，仅禁用分支；仅空间仍通过head使用历史步。"
            "参数非严格匹配，15epochs不代表完全收敛，3seeds不支持严格显著性宣称。\n"
            "原生GRU从y读取未来已知的time_in_day辅助特征，不使用未来交通目标；"
            "主要单模块对照为两个严格分支消融，原生GRU属于额外结构基线。\n"
            "验证证据：verification.json是72次记录的矩阵/epoch/参数/指标均值/哈希全量审计；"
            f"checkpoint_recheck.json独立重算了{summary['verification_evidence']['checkpoint_recheck_count']}个checkpoint"
            "（4数据集×concat/native_gru×seed0），使用原始LibCity指标函数，并检查未来目标扰动不影响预测；"
            "不能将8个checkpoint复算宣称为72个全部复算。provenance.json与source/提供数据指纹和源码快照。")
        text_dump += [f"## Slide {index}"]
        for shape in slide.shapes:
            if shape.has_text_frame:
                text_dump.append(shape.text)
            if shape.has_table:
                text_dump += [" | ".join(cell.text for cell in row.cells) for row in shape.table.rows]
        for shape in slide.shapes:
            if shape.left < 0 or shape.top < 0 or shape.left + shape.width > deck.prs.slide_width + 10 or shape.top + shape.height > deck.prs.slide_height + 10:
                bounds.append({"slide": index, "shape": shape.name})
    deck.prs.save(path)
    (directory / "presentation_text.txt").write_text("\n\n".join(text_dump))
    text_content = "\n".join(text_dump)
    banned = ("lorem ipsum", "xxxx", "this slide layout")
    leftovers = [term for term in banned if term in text_content.lower()]
    if summary["complete"]:
        leftovers += [term for term in ("DRAFT", "待完成", "读取中", "等待完整实验", "训练进行中") if term in text_content]
    audit = {"slides": len(deck.prs.slides), "chart_count": sum(shape.has_chart for slide in deck.prs.slides for shape in slide.shapes),
             "table_count": sum(shape.has_table for slide in deck.prs.slides for shape in slide.shapes),
             "out_of_bounds": bounds, "leftover_placeholders": leftovers,
             "completed_runs": summary["completed_runs"], "complete": summary["complete"],
             "content_extraction": "python-pptx text/table extraction; MarkItDown optional pptx dependencies unavailable"}
    (directory / "presentation_content_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    if bounds:
        raise ValueError(f"slide objects extend beyond canvas: {bounds}")
    if leftovers:
        raise ValueError(f"presentation contains placeholder content: {leftovers}")
    return path, len(deck.prs.slides)


def json_records(directory):
    return [json.loads(p.read_text()) for p in (directory / "runs").glob("*.json")]


def render_ppt(path):
    portable = Path("/tmp/lo_apt/root/usr/lib/libreoffice/program/soffice")
    binary = shutil.which("libreoffice") or shutil.which("soffice") or (str(portable) if portable.exists() else None)
    if not binary:
        raise RuntimeError("LibreOffice is required for actual PPTX-to-PDF visual validation")
    env = dict(os.environ)
    if binary == str(portable):
        env["LD_LIBRARY_PATH"] = ":".join([str(portable.parent), "/tmp/lo_apt/root/usr/lib/x86_64-linux-gnu", "/tmp/lo_apt/root/lib/x86_64-linux-gnu", env.get("LD_LIBRARY_PATH", "")])
    with tempfile.TemporaryDirectory(prefix="agentcity-ppt-lo-") as profile:
        completed = subprocess.run([binary, f"-env:UserInstallation={Path(profile).as_uri()}", "--headless", "--convert-to", "pdf", "--outdir", str(path.parent), str(path)],
                                   env=env, capture_output=True, text=True, timeout=120)
    if completed.returncode or not path.with_suffix(".pdf").exists():
        raise RuntimeError(f"LibreOffice conversion failed: {completed.stdout}\n{completed.stderr}")
    import fitz
    from PIL import Image, ImageOps, ImageDraw
    preview = path.parent / "preview"
    preview.mkdir(exist_ok=True)
    doc = fitz.open(path.with_suffix(".pdf"))
    thumbnails = []
    for index, page in enumerate(doc, 1):
        image_path = preview / f"slide-{index:02d}.png"
        page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(image_path)
        im = Image.open(image_path).convert("RGB")
        im.thumbnail((533, 300))
        tile = Image.new("RGB", (553, 335), "#E5EBE7")
        tile.paste(im, ((553 - im.width) // 2, 5))
        ImageDraw.Draw(tile).text((12, 311), f"Slide {index:02d}", fill="#263B3D")
        thumbnails.append(tile)
    columns = 3
    sheet = Image.new("RGB", (553 * columns, 335 * math.ceil(len(thumbnails) / columns)), "#E5EBE7")
    for index, im in enumerate(thumbnails):
        sheet.paste(im, ((index % columns) * 553, (index // columns) * 335))
    sheet.save(path.parent / "slides_contact_sheet.png")
    return len(doc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "result/module_benchmark_20261026")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args.directory = args.directory.resolve()
    summary, groups, flat, paired = summarize(args.directory, args.require_complete)
    write_report(args.directory, summary, groups, paired)
    ppt, count = build_ppt(args.directory, summary, groups, flat, paired)
    rendered = render_ppt(ppt) if args.render else None
    print(json.dumps({"completed_runs": summary["completed_runs"], "complete": summary["complete"], "slides": count,
                      "pptx": str(ppt), "rendered_pdf_pages": rendered}, ensure_ascii=False))


if __name__ == "__main__":
    main()
