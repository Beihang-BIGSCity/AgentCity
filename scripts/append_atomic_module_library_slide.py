"""Append an editable inventory of the actual atomic module library."""

import json
from collections import Counter
from pathlib import Path

from pptx import Presentation
from pptx.util import Pt

from append_tech3_results import format_paragraph
from build_knowledge_skill_diagrams import Diagram, blank, BLUE, BORDER, PALE
from build_module_benchmark_report import render_ppt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result/atomic_module_library_20261028"
SOURCE = ROOT / "result/knowledge_skill_diagrams_20261028/技术点三_架构与实验补充_知识库与Skill图示.pptx"
LABELS = {
    "zscore_encoder": "Z-Score 标准化",
    "missing_fill_encoder": "缺失值前向填充",
    "feature_embedding_encoder": "Linear + ReLU 嵌入",
    "cheb_gcn": "切比雪夫图卷积",
    "diffusion_gcn": "双向扩散图卷积",
    "adaptive_gcn": "自适应图卷积",
    "spatial_attention": "空间多头注意力",
    "gru": "GRU",
    "lstm": "LSTM",
    "dilated_tcn": "tanh 门控膨胀 TCN",
    "gated_dilated_tcn": "GLU 膨胀 TCN",
    "temporal_transformer": "时间 Transformer",
    "add": "相加融合",
    "concat": "拼接 + 线性投影",
    "weighted": "Softmax 加权融合",
    "cross_attention": "交叉注意力融合",
    "linear_head": "线性预测头",
    "mlp_head": "MLP 预测头",
    "horizon_head": "分预测步输出头",
}
GROUPS = [
    ("encoder", "数据编码器", "归一化 / 补全 / 升维"),
    ("spatial", "空间依赖", "邻接传播 / 节点交互"),
    ("temporal", "时间依赖", "循环 / 卷积 / 注意力"),
    ("fusion", "时空融合", "汇合空间与时间表征"),
    ("head", "输出头", "多步预测 + 损失函数"),
]


def cell_text(cell, value, size, bold=False, color="172B3A"):
    cell.text_frame.clear()
    for index, line in enumerate(value.split("\n")):
        paragraph = cell.text_frame.paragraphs[0] if index == 0 else cell.text_frame.add_paragraph()
        format_paragraph(paragraph, line, size, bold, color)
        paragraph.line_spacing = 1.05
        paragraph.space_after = Pt(2 if index < len(value.split("\n")) - 1 else 0)


def complete_experiment_labels(presentation):
    """Keep measurements intact while spelling out each evaluated pipeline."""
    table = next(sh.table for sh in presentation.slides[10].shapes if sh.has_table)
    cell_text(table.cell(0, 1), "编码 → [空间 ∥ 时间] → 融合 → 输出头", 12, True, "FFFFFF")
    branch_labels = {
        "METR_LA": ("DiffGCN", "GRU", "Concat"),
        "PEMS_BAY": ("ChebGCN", "LSTM", "Add"),
        "PEMSD4": ("AdaptiveGCN", "GLU-TCN", "Concat"),
        "PEMSD7": ("AdaptiveGCN", "GLU-TCN", "Concat"),
    }
    for row in range(1, 5):
        dataset = table.cell(row, 0).text
        if dataset != "METR_LA":
            record = ROOT / "result/dataset_composition_20261027/frozen" / dataset / "composition.json"
            actual = json.loads(record.read_text())
            modules = actual["modules"]
            assert modules["encoder"]["id"] == "feature_embedding_encoder"
            assert modules["head"]["id"] == "linear_head"
            assert actual["loss_type"] == "masked_mae"
            assert modules["spatial"]["id"] == ("cheb_gcn" if dataset == "PEMS_BAY" else "adaptive_gcn")
            assert modules["temporal"]["id"] == ("lstm" if dataset == "PEMS_BAY" else "gated_dilated_tcn")
            assert modules["fusion"]["id"] == ("add" if dataset == "PEMS_BAY" else "concat")
        spatial, temporal, fusion = branch_labels[dataset]
        cell_text(table.cell(row, 1),
                  f"Linear+ReLU → [{spatial} ∥ {temporal}]\n→ {fusion} → Linear Head", 12.5)
    foot = next(sh for sh in presentation.slides[10].shapes if sh.has_text_frame and sh.text.startswith("METR_LA："))
    foot.text_frame.clear()
    lines = [
        "编码器为 Linear+ReLU；输出头为直接多步 Linear Head；四组训练损失均为 masked MAE。",
        "∥ 为并行分支；METR_LA 为原流程结构复训，其余为本轮逐数据集运行流程组合。",
        "消融关闭分支信息，保留编码器、融合、输出头；未使用参数保留，不等于实际减参。",
    ]
    for i, line in enumerate(lines):
        paragraph = foot.text_frame.paragraphs[0] if i == 0 else foot.text_frame.add_paragraph()
        format_paragraph(paragraph, line, 10.5, fill="536779")
        paragraph.line_spacing = 1.0
        paragraph.space_after = Pt(1)
    presentation.slides[10].notes_slide.notes_text_frame.text += (
        "\n补全组合说明：四组均使用 feature_embedding_encoder 和 linear_head，训练损失 masked_mae。"
        "METR_LA 对照 STDiffGRU.py 与 module_benchmark_20261026 记录；其余三组对照各自 frozen/composition.json。"
        "输出头执行 direct multi-step 预测；本实现没有额外独立注册的 decoder 类别。"
    )
    slide = presentation.slides[11]
    baseline_table = next(sh.table for sh in slide.shapes if sh.has_table)
    cell_text(baseline_table.cell(0, 2), "完整模型基线", 14.5, True, "FFFFFF")
    caption = next(sh for sh in slide.shapes if sh.has_text_frame and sh.text.startswith("同 seed"))
    format_paragraph(caption.text_frame.paragraphs[0],
                     "组合结构见前页；同 seed 配对计算误差下降（正值为组合更好）。完整模型基线采用原生配置。", 14)


def make_slide(presentation, modules):
    slide = blank(presentation)
    d = Diagram(slide)
    counts = Counter(m["category"] for m in modules)
    assert set(LABELS) == {m["id"] for m in modules}
    assert len(modules) == 19 and len(counts) == 5
    d.header("原子化模型库：五类模块，独立组合", "当前已实现 5 类、19 个原子模块；按任务需求选择模块，组合为新的预测模型。")
    for i, (category, title, role) in enumerate(GROUPS):
        x = 48 + i * 176
        d.rect(category + "-card", x, 132, 160, 224, PALE, BORDER)
        d.rect(category + "-header", x, 132, 160, 36, BLUE, BLUE)
        d.text(category + "-title", f"{title} · {counts[category]}", x + 10, 141,
               140, 24, 15, True, "FFFFFF", centered=True)
        selected = [m for m in modules if m["category"] == category]
        for j, module in enumerate(selected):
            d.text(module["id"], LABELS[module["id"]], x + 10, 186 + j * 26,
                   140, 21, 12.5)
        d.text(category + "-role", role, x + 8, 331, 144, 18, 10.5)

    d.text("composition-label", "组合方式示意：每类选择一个模块", 48, 376, 864, 23, 14, True)
    boxes = [
        ("encoder-example", "数据编码", 48, 421, 144, 38),
        ("spatial-example", "空间模块", 256, 402, 144, 30),
        ("temporal-example", "时间模块", 256, 448, 144, 30),
        ("fusion-example", "融合模块", 464, 421, 128, 38),
        ("head-example", "输出头", 640, 421, 128, 38),
        ("prediction", "预测结果", 816, 421, 96, 38),
    ]
    for key, label, x, y, w, h in boxes:
        d.rect(key, x, y, w, h, "FFFFFF", BLUE)
        d.text(key + "-label", label, x + 6, y + (h - 18) / 2,
               w - 12, 20, 13, centered=True)
    # The space and time modules are parallel branches, not a serial chain.
    d.arrow("split-in", 195, 440, 224, 440, head=False)
    d.arrow("split-top", 224, 440, 224, 417, head=False)
    d.arrow("split-bottom", 224, 440, 224, 463, head=False)
    d.arrow("spatial-in", 224, 417, 253, 417)
    d.arrow("temporal-in", 224, 463, 253, 463)
    d.arrow("spatial-out", 403, 417, 432, 417, head=False)
    d.arrow("temporal-out", 403, 463, 432, 463, head=False)
    d.arrow("merge-top", 432, 417, 432, 440, head=False)
    d.arrow("merge-bottom", 432, 463, 432, 440, head=False)
    d.arrow("fusion-in", 432, 440, 461, 440)
    d.arrow("head-in", 595, 440, 637, 440)
    d.arrow("prediction-in", 771, 440, 813, 440)
    d.text("independence", "独立实现、统一接口、跨来源组合；完整模型用于实验对照。", 48, 496, 864, 23, 14, True)
    d.text("loss-note", "损失函数归入输出头：masked MAE / MSE / MAPE。", 48, 512, 864, 16, 10.5)
    slide.notes_slide.notes_text_frame.text = (
        "当前实现清单：data/modlib/index.json；模块源码：data/modlib/code/<category>/；"
        "接口定义与管理：agents/adaptation/modlib.py；"
        "组合规则：agents/adaptation/prompts/model_composer.py。"
        "五类模块分别有 3、4、5、4、3 个，共 19 个。模块源码仅导入 PyTorch 和标准库；"
        "各模块可以参考、抽象或改写已有模型设计，来源记录不表示依赖完整模型。"
        "本页组合图展示当前实验采用的并行空间/时间分支，未声称所有可能组合只有该结构。"
        "dilated_tcn 使用 tanh/sigmoid 双分支门控；gated_dilated_tcn 使用 GLU。"
        "三个输出头都提供 masked_mae、masked_mse、masked_mape，可通过 loss_type 选择。"
        "本页不把 GAT 或普通 RNN 计入已实现条目。完整模块标识："
        + json.dumps({category: [m["id"] for m in modules if m["category"] == category]
                      for category, _, _ in GROUPS}, ensure_ascii=False)
    )
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index_path = ROOT / "data/modlib/index.json"
    modules = json.loads(index_path.read_text())["modules"]
    assert all((index_path.parent / m["file"]).is_file() for m in modules)
    presentation = Presentation(SOURCE)
    count = len(presentation.slides)
    assert count == 14
    complete_experiment_labels(presentation)
    diagram = make_slide(presentation, modules)
    target = OUT / "技术点三_架构与实验补充_含原子模块库.pptx"
    presentation.save(target)
    single = Presentation(target)
    for slide_id in list(single.slides._sldIdLst)[:count]:
        single.part.drop_rel(slide_id.rId)
        single.slides._sldIdLst.remove(slide_id)
    single.save(OUT / "原子化模型库_五类19模块.pptx")
    (OUT / "原子化模型库_五类19模块.svg").write_text(
        "\n".join(diagram.svg + ["</svg>"]), encoding="utf-8"
    )
    (OUT / "geometry.json").write_text(
        json.dumps(diagram.measurements, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_ppt(target)
    import fitz
    full = fitz.open(target.with_suffix(".pdf"))
    full[count].get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False).save(
        OUT / "原子化模型库_五类19模块.png"
    )
    pdf = fitz.open()
    pdf.insert_pdf(full, from_page=count, to_page=count)
    pdf.save(OUT / "原子化模型库_五类19模块.pdf")
    print(json.dumps({"pptx": str(target), "pages": len(presentation.slides),
                      "added_page": count + 1, "module_counts": dict(Counter(m["category"] for m in modules))},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
