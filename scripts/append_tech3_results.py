"""Append one generated architecture image and two results pages to 技术点三.

All pre-existing package parts (including the nine original slides and their
relationships) are retained byte-for-byte, apart from the package slide list,
its relationship list, content-type declarations and application slide count.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result/dataset_composition_20261027/template_extension"
BACKUP = OUT / "技术点三_原始9页.pptx"
BLUE, PALE, INK, GRAY = "005FAF", "EDF4FB", "172B3A", "536779"
FONT = "Noto Sans CJK SC"


def color(value):
    return RGBColor.from_string(value)


def format_paragraph(paragraph, text, size=15, bold=False, fill=INK, align=PP_ALIGN.LEFT):
    paragraph.text = text
    paragraph.font.name, paragraph.font.size = FONT, Pt(size)
    paragraph.font.bold, paragraph.font.color.rgb = bold, color(fill)
    paragraph.alignment = align
    paragraph.space_after = Pt(3)
    for run in paragraph.runs:
        properties = run._r.get_or_add_rPr()
        ea = OxmlElement("a:ea")
        ea.set("typeface", FONT)
        properties.append(ea)


def text(slide, value, x, y, w, h, size=15, bold=False, fill=INK):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.word_wrap = True
    for index, line in enumerate(value.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        format_paragraph(paragraph, line, size, bold, fill)
    return box


def new_slide(presentation, layout, title):
    slide = presentation.slides.add_slide(layout)
    title_box = slide.shapes.title
    for shape in list(slide.shapes):
        if shape.shape_id != title_box.shape_id:
            shape._element.getparent().remove(shape._element)
    title_box.text = title
    # Retain the original placeholder/theme style and geometry.
    return slide


def table(slide, rows, widths, x=.67, y=1.85, row_height=.84, size=15, highlights=()):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y),
                                   Inches(sum(widths)), Inches(.55 + row_height * (len(rows) - 1)))
    native = shape.table
    for i, width in enumerate(widths):
        native.columns[i].width = Inches(width)
    for r, row in enumerate(rows):
        native.rows[r].height = Inches(.55 if r == 0 else row_height)
        for c, value in enumerate(row):
            cell = native.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(.10)
            cell.margin_top = cell.margin_bottom = Inches(.05)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = color(BLUE if r == 0 else (PALE if r % 2 else "FFFFFF"))
            cell.text_frame.clear()
            for i, line in enumerate(str(value).split("\n")):
                paragraph = cell.text_frame.paragraphs[0] if i == 0 else cell.text_frame.add_paragraph()
                shade = "FFFFFF" if r == 0 else (BLUE if (r, c) in highlights else INK)
                if r > 0 and str(value).startswith("−"):
                    shade = "B55522"
                format_paragraph(paragraph, line, size if r else size - 1,
                                 r == 0 or c == 0 or (r, c) in highlights, shade,
                                 PP_ALIGN.LEFT if c < 2 else PP_ALIGN.CENTER)
    return native


def rows_for(directory, dataset, variant):
    records = [json.loads(p.read_text()) for p in (directory / "runs").glob(f"{dataset}_*.json")]
    result = sorted([r for r in records if r["variant"] == variant], key=lambda r: r["seed"])
    assert [r["seed"] for r in result] == [0, 1, 2], (dataset, variant)
    return result


def metric(records):
    values = [r["test"]["masked_mae"] for r in records]
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def preserve_original_parts(original, extended, target):
    allowed = {"ppt/presentation.xml", "ppt/_rels/presentation.xml.rels", "[Content_Types].xml", "docProps/app.xml"}
    with ZipFile(original) as old, ZipFile(extended) as new:
        ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
        def rels(data):
            return {x.get("Id"): (x.get("Type"), x.get("Target"), x.get("TargetMode"))
                    for x in etree.fromstring(data).findall("r:Relationship", ns)}
        old_rel, new_rel = rels(old.read("ppt/_rels/presentation.xml.rels")), rels(new.read("ppt/_rels/presentation.xml.rels"))
        assert all(new_rel.get(k) == v for k, v in old_rel.items()), "Original relationship IDs changed"
        ct = etree.fromstring(old.read("[Content_Types].xml"))
        known = {(etree.QName(x).localname, x.get("PartName") or x.get("Extension")) for x in ct}
        for entry in etree.fromstring(new.read("[Content_Types].xml")):
            key = etree.QName(entry).localname, entry.get("PartName") or entry.get("Extension")
            if key not in known:
                ct.append(deepcopy(entry)); known.add(key)
        content_types = etree.tostring(ct, xml_declaration=True, encoding="UTF-8", standalone=True)
        app = etree.fromstring(old.read("docProps/app.xml"))
        for element in app.iter():
            if etree.QName(element).localname == "Slides":
                element.text = "12"
        app_xml = etree.tostring(app, xml_declaration=True, encoding="UTF-8", standalone=True)
        with ZipFile(target, "w", ZIP_DEFLATED) as result:
            for name in old.namelist():
                if name == "[Content_Types].xml":
                    data = content_types
                elif name == "docProps/app.xml":
                    data = app_xml
                elif name in allowed:
                    data = new.read(name)
                else:
                    data = old.read(name)
                result.writestr(name, data)
            for name in new.namelist():
                if name not in old.namelist():
                    result.writestr(name, new.read(name))
    with ZipFile(original) as old, ZipFile(target) as updated:
        changed = [name for name in old.namelist() if name not in allowed and old.read(name) != updated.read(name)]
    assert not changed, changed
    return {"original_package_parts_preserved": True, "original_slide_count": 9,
            "final_slide_count": 12, "appended_pages": [10, 11, 12], "unexpected_changed_original_parts": changed}


def build(image_path, target):
    presentation = Presentation(BACKUP)
    assert len(presentation.slides) == 9
    layout = presentation.slides[8].slide_layout
    old_results = ROOT / "result/module_benchmark_20261026"
    new_results = ROOT / "result/dataset_composition_20261027"
    cases = [("METR_LA", old_results, "concat", "DiffGCN ∥ GRU → Concat", "native_gru", "原生 GRU*"),
             ("PEMS_BAY", new_results, "full", "ChebGCN ∥ LSTM → Add", "source_LSTM", "完整 LSTM"),
             ("PEMSD4", new_results, "full", "AdaptiveGCN ∥ GLU-TCN → Concat", "source_AGCRN", "完整 AGCRN"),
             ("PEMSD7", new_results, "full", "AdaptiveGCN ∥ GLU-TCN → Concat", "source_AGCRN", "完整 AGCRN")]
    s = new_slide(presentation, layout, "当前架构：AgentCity 模型适配闭环")
    width, height = Image.open(image_path).size
    w, h = 12.0, 12.0 * height / width
    if h > 5.95:
        h, w = 5.95, 5.95 * width / height
    s.shapes.add_picture(str(image_path), Inches((13.333333 - w) / 2), Inches(1.10), width=Inches(w), height=Inches(h))
    text(s, "当前实现：七类智能体分层协作；知识、Skills 与结构化执行记录共同支持适配闭环。", .70, 7.15, 11.9, .22, 10.5, fill=GRAY)
    s.notes_slide.notes_text_frame.text = "GPT Image 生成当前逻辑架构。任务规划统筹执行与调参层；模型组合按画像和知识检索五类原子模块，运行结果反馈，经验回流知识库和Skills。"

    s = new_slide(presentation, layout, "实验结果一：模块组合与单模块消融")
    text(s, "15 epochs × 3 seeds；12→12；原始时间 60/20/20 切分；指标为平均 masked_MAE（均值±标准差，↓）。", .70, 1.43, 11.9, .35, 14)
    data = [["数据集", "实际组合：空间 ∥ 时间 → 融合", "完整组合", "仅时间分支", "仅空间分支"]]
    exact = []
    for dataset, directory, full_name, architecture, reference, label in cases:
        full = rows_for(directory, dataset, full_name)
        time_only, space_only = rows_for(directory, dataset, "temporal_only"), rows_for(directory, dataset, "spatial_only")
        data.append([dataset, architecture, metric(full), metric(time_only), metric(space_only)])
        exact.append({"dataset": dataset, "full": full_name, "architecture": architecture,
                      "full_metric": metric(full), "time_metric": metric(time_only), "space_metric": metric(space_only)})
    table(s, data, [1.25, 4.25, 2.02, 2.02, 2.02], size=14.5, highlights={(i, 2) for i in range(1, 5)})
    text(s, "结果：完整组合在四组实测中均优于自身单分支；PEMSD4/7 的空间分支贡献更明显。", .70, 5.98, 11.9, .44, 17, True, BLUE)
    text(s, "METR_LA：原流程结构的统一协议复训（10月26日）；其余三组：本轮逐数据集运行流程组合。\n消融关闭分支信息，保持编码器、融合和预测头；本轮新组合保留未使用参数，不等于实际减参。", .70, 6.55, 11.9, .57, 11.5, fill=GRAY)
    s.notes_slide.notes_text_frame.text = ("GLU-TCN 为门控膨胀时序卷积。METR_LA 使用原先数据定制的 STDiffGRU 在统一协议下复训的三seed结果，输入含time_in_day；新三组均按实际C1输入和各自任务图合同评估。速度、流量误差不跨数据集求平均。D4/D7独立运行流程后选择同一组合家族，不制造不同结构。完整源记录：result/module_benchmark_20261026/runs 与 result/dataset_composition_20261027/runs。")

    s = new_slide(presentation, layout, "实验结果二：完整模型对照与容量取舍")
    text(s, "同 seed 配对计算相对误差下降；正数为组合更好，负数为组合更差。完整来源模型使用原生配置。", .70, 1.43, 11.9, .35, 14)
    data = [["数据集", "组合 MAE", "完整来源 / 原生基线", "基线 MAE", "误差下降", "参数：组合 / 基线"]]
    comparisons = []
    highlights = set()
    for row, (dataset, directory, full_name, architecture, reference, label) in enumerate(cases, 1):
        full, baseline = rows_for(directory, dataset, full_name), rows_for(directory, dataset, reference)
        gains = [100 * (b["test"]["masked_mae"] - a["test"]["masked_mae"]) / b["test"]["masked_mae"] for a, b in zip(full, baseline)]
        gain = statistics.mean(gains)
        prefix = "+" if gain >= 0 else "−"
        allocated = full[0]["parameters"], baseline[0]["parameters"]
        data.append([dataset, metric(full), label, metric(baseline), f"{prefix}{abs(gain):.2f}%",
                     f"{allocated[0]/1000:.1f}k / {allocated[1]/1000:.1f}k"])
        highlights.add((row, 1 if gain >= 0 else 3))
        comparisons.append({"dataset": dataset, "reference": reference, "reference_label": label,
                            "mean_paired_improvement_percent": gain, "full_parameters": allocated[0],
                            "reference_parameters": allocated[1]})
    table(s, data, [1.25, 2.02, 2.18, 2.02, 1.55, 2.54], size=14.5, highlights=highlights)
    text(s, "结论：组合优于自身单分支，但 PEMSD4/7 仍弱于完整 AGCRN；更小参数规模并不保证更高精度。", .70, 5.98, 11.9, .45, 16.5, True, BLUE)
    text(s, "* METR_LA 对照为原生 GRU（hidden=32，单层），尚无完整 DCRNN/STFGNN 同协议对照。\n固定轮数、非等参数/等损失/等优化设置；三 seed 只描述波动，不作显著性或充分调优后的最优性结论。", .70, 6.55, 11.9, .57, 11.5, fill=GRAY)
    s.notes_slide.notes_text_frame.text = ("主要完整来源：PEMS_BAY 的 LSTM 源自 LibCity RNN.py，原生 hidden=64；PEMSD4/7 的 AGCRN 是 AdaptiveGCN 的完整来源模型。STGCN相关图基线保存在完整实验报告，不在短表内。新组合与完整来源模型原生超参数/损失/容量不同。D7使用已验证数值等价的FP32时间卷积执行后端；原生CPU九checkpoint复核通过，耗时不作跨后端严格算法效率比较。51条正式记录全量审计通过。")
    with tempfile.TemporaryDirectory(prefix="tech3-append-") as temporary:
        intermediate = Path(temporary) / "extended.pptx"
        presentation.save(intermediate)
        audit = preserve_original_parts(BACKUP, intermediate, target)
    audit.update({"architecture_image": str(image_path.relative_to(ROOT)), "experimental_rows": exact,
                  "complete_source_comparisons": comparisons, "image_generation": "built-in GPT Image",
                  "original_sha256": hashlib.sha256(BACKUP.read_bytes()).hexdigest(),
                  "extended_sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    (OUT / "append_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    assert len(Presentation(target).slides) == 12
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "技术点三_架构与实验补充.pptx")
    parser.add_argument("--update-original", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not BACKUP.exists():
        shutil.copy2(ROOT / "技术点三.pptx", BACKUP)
    audit = build(args.image.resolve(), args.output.resolve())
    if args.update_original:
        shutil.copy2(args.output, ROOT / "技术点三.pptx")
    print(json.dumps({"slides": audit["final_slide_count"], "original_parts_preserved": True,
                      "output": str(args.output.resolve()), "original_updated": args.update_original}, ensure_ascii=False))


if __name__ == "__main__":
    main()
