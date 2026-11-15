"""Add source-backed historical STSGCN metrics to the complete-model slide."""

import hashlib
import json
from pathlib import Path
import re
import statistics

from pptx import Presentation
from pptx.util import Inches, Pt

from append_tech3_results import BLUE, GRAY, color, format_paragraph, table, text
from build_module_benchmark_report import render_ppt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result/historical_stsgcn_supplement_20261028"
SOURCE = ROOT / "result/atomic_module_library_20261028/技术点三_架构与实验补充_含原子模块库.pptx"
LOG = ROOT / "Bigscity-LibCity/libcity/log/172955-STSGCN-METR_LA-Mar-21-2026_05-00-27.log"
ROW = re.compile(
    r"^\s*(\d+)\s+([\d.]+)\s+inf\s+([\d.]+)\s+\.\.\.\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", re.M
)


def history():
    source = LOG.read_text()
    rows = ROW.findall(source)
    assert [int(row[0]) for row in rows] == list(range(1, 13))
    batch = (ROOT / "batch_logs/STSGCN_test.log").read_text()
    assert ROW.findall(batch.replace("[OUT] ", "")) == rows
    for fragment in ["'seed': 0", "'max_epoch': 35", "'train_rate': 0.7", "'eval_rate': 0.1",
                     "Total parameter numbers: 1150668", "Loaded model at 18"]:
        assert fragment in source, fragment
    return {
        "model": "STSGCN", "dataset": "METR_LA", "run_id": 172955,
        "date": "2026-03-21", "epochs": 35, "seed": 0,
        "train_eval_test_percent": [70, 10, 20], "parameters": 1150668,
        "source_log": str(LOG.relative_to(ROOT)),
        "source_log_sha256": hashlib.sha256(LOG.read_bytes()).hexdigest(),
        "cross_checked_batch_log": "batch_logs/STSGCN_test.log",
        "ordinary_mae_horizon_mean": statistics.mean(float(r[1]) for r in rows),
        "masked_rmse_horizon_mean": statistics.mean(float(r[3]) for r in rows),
        "masked_mae": None,
        "masked_mae_note": "Not recoverable from retained truncated DataFrame printout; referenced CSV is absent.",
        "aggregation": "Arithmetic mean over the 12 printed per-horizon values (six decimal places).",
        "comparable_to_current_paired_results": False,
        "per_horizon": [{"horizon": int(r[0]), "ordinary_mae": float(r[1]),
                         "ordinary_mse": float(r[2]), "masked_rmse": float(r[3])}
                        for r in rows],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    record = history()
    presentation = Presentation(SOURCE)
    assert len(presentation.slides) == 15
    slide = presentation.slides[11]
    original_table = next(sh.table for sh in slide.shapes if sh.has_table)
    original_table.rows[0].height = Inches(.48)
    for row in list(original_table.rows)[1:]:
        row.height = Inches(.58)
    caption = next(sh for sh in slide.shapes if sh.has_text_frame and sh.text.startswith("组合结构见前页"))
    format_paragraph(caption.text_frame.paragraphs[0],
                     "组合结构见前页；主表为统一协议实验，误差下降按同 seed 配对计算；完整模型基线采用原生配置。", 14)
    text(slide, "历史补充：STSGCN（2026-03-21；协议不同，仅作参考）",
         .70, 4.80, 11.9, .23, 12, True, BLUE)
    rows = [
        ["数据集", "历史完整模型", "普通 MAE", "masked RMSE", "训练设置", "数据切分 / 指标说明"],
        ["METR_LA", "STSGCN", f"{record['ordinary_mae_horizon_mean']:.3f}",
         f"{record['masked_rmse_horizon_mean']:.3f}", "35 epochs，seed 0",
         "70/10/20；MAE 与主表 masked MAE 口径不同"],
    ]
    historical = table(slide, rows, [1.10, 1.75, 1.05, 1.40, 2.15, 4.11],
                       x=.67, y=5.08, row_height=.43, size=11)
    historical.rows[0].height = Inches(.34)
    historical.rows[1].height = Inches(.43)
    for cell in historical.rows[0].cells:
        cell.fill.fore_color.rgb = color(GRAY)
    conclusion = next(sh for sh in slide.shapes if sh.has_text_frame and sh.text.startswith("结论："))
    format_paragraph(conclusion.text_frame.paragraphs[0],
                     conclusion.text.replace("结论：", "主表结论：", 1), 16, True, BLUE)
    foot = next(sh for sh in slide.shapes if sh.has_text_frame and sh.text.startswith("* METR_LA"))
    foot.height = Inches(.73)
    foot.text_frame.clear()
    notes = [
        "* METR_LA 主表对照为原生 GRU（hidden=32，单层）；尚无完整 DCRNN/STFGNN 同协议对照。",
        "主表采用固定轮数、非等参数/等损失/等优化设置；三 seed 描述波动，不作显著性或充分调优后的最优性结论。",
        "历史 STSGCN 的 masked MAE 未保留；普通 MAE、masked RMSE 均为日志中 12 个预测步的算术均值。",
    ]
    for i, line in enumerate(notes):
        paragraph = foot.text_frame.paragraphs[0] if i == 0 else foot.text_frame.add_paragraph()
        format_paragraph(paragraph, line, 10.5, fill=GRAY)
        paragraph.line_spacing = 1.0
        paragraph.space_after = Pt(1)
    slide.notes_slide.notes_text_frame.text += (
        "\n历史补充来源与精度边界：" + json.dumps(record, ensure_ascii=False) +
        "\n日志表头可见 MAE、MAPE、MSE、…、masked_RMSE、R2、EVAR；"
        "masked_MAE 被省略，不能从普通 MAE 推断。原始 CSV 与 checkpoint 未在当前缓存找到。"
        "本历史记录只供展示，不与主表进行改善率、统计检验或优劣结论计算。"
    )
    target = OUT / "技术点三_架构与实验补充_含原子模块库与历史指标.pptx"
    presentation.save(target)
    (OUT / "historical_stsgcn_metrics.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_ppt(target)
    import fitz
    doc = fitz.open(target.with_suffix(".pdf"))
    doc[11].get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False).save(OUT / "完整模型对照_含历史STSGCN.png")
    single = fitz.open()
    single.insert_pdf(doc, from_page=11, to_page=11)
    single.save(OUT / "完整模型对照_含历史STSGCN.pdf")
    print(json.dumps({"pptx": str(target), "page": 12,
                      "ordinary_MAE": record['ordinary_mae_horizon_mean'],
                      "masked_RMSE": record['masked_rmse_horizon_mean']}, ensure_ascii=False))


if __name__ == "__main__":
    main()
