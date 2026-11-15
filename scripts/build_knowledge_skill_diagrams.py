"""Build two independent, editable PPT diagrams from the adaptation contracts."""

import json
from pathlib import Path
from xml.sax.saxutils import escape

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Pt

from append_tech3_results import format_paragraph
from build_module_benchmark_report import render_ppt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result/knowledge_skill_diagrams_20261028"
FONT = "Noto Sans CJK SC"
BLUE, INK, PALE, BORDER = "005FAF", "172B3A", "EDF4FB", "B3C6D7"
WIDTH, HEIGHT = 960, 540
XS, BOX_W, BOX_H = (48, 272, 496, 720), 192, 110


class Diagram:
    """One geometry description feeds native PPT shapes and editable SVG."""

    def __init__(self, slide):
        self.slide = slide
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="960pt" '
            f'height="540pt" viewBox="0 0 {WIDTH} {HEIGHT}">',
            '<rect width="960" height="540" fill="white"/>',
        ]
        self.measurements = []

    def rect(self, key, x, y, w, h, fill="FFFFFF", line=BORDER):
        shape = self.slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.RECTANGLE, Pt(x), Pt(y), Pt(w), Pt(h)
        )
        shape.name = key
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(fill)
        shape.line.color.rgb = RGBColor.from_string(line)
        shape.line.width = Pt(1)
        self.svg.append(
            f'<rect id="{key}" x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="#{fill}" stroke="#{line}" stroke-width="1"/>'
        )

    def text(self, key, value, x, y, w, h, size=13, bold=False,
             color=INK, centered=False):
        box = self.slide.shapes.add_textbox(Pt(x), Pt(y), Pt(w), Pt(h))
        box.name = key
        frame = box.text_frame
        frame.margin_left = frame.margin_right = 0
        frame.margin_top = frame.margin_bottom = 0
        frame.word_wrap = False
        align = PP_ALIGN.CENTER if centered else PP_ALIGN.LEFT
        for index, line in enumerate(value.split("\n")):
            paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
            format_paragraph(paragraph, line, size, bold, color, align)
            paragraph.space_after = Pt(4)
            paragraph.line_spacing = 1.0
            baseline = y + size * .90 + index * (size + 4)
            anchor = "middle" if centered else "start"
            tx = x + w / 2 if centered else x
            self.svg.append(
                f'<text id="{key}-{index}" x="{tx}" y="{baseline}" '
                f'font-family="{FONT}" font-size="{size}" '
                f'font-weight="{"bold" if bold else "normal"}" '
                f'text-anchor="{anchor}" fill="#{color}">{escape(line)}</text>'
            )
            self.measurements.append(dict(text=line, x=x, y=y, w=w, h=h, size=size))

    def arrow(self, key, x1, y1, x2, y2, dashed=False, head=True):
        shape = self.slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT, Pt(x1), Pt(y1), Pt(x2), Pt(y2)
        )
        shape.name = key
        shape.line.color.rgb = RGBColor.from_string(BLUE)
        shape.line.width = Pt(1.7)
        line = shape._element.spPr.find(
            "{http://schemas.openxmlformats.org/drawingml/2006/main}ln"
        )
        if dashed:
            dash = OxmlElement("a:prstDash")
            dash.set("val", "dash")
            line.append(dash)
        if head:
            end = OxmlElement("a:tailEnd")
            end.set("type", "triangle")
            end.set("w", "sm")
            end.set("len", "sm")
            line.append(end)
        dash_svg = ' stroke-dasharray="6 4"' if dashed else ""
        self.svg.append(
            f'<line id="{key}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#{BLUE}" stroke-width="1.7"{dash_svg}/>'
        )
        if not head:
            return
        dx, dy = x2 - x1, y2 - y1
        length = (dx * dx + dy * dy) ** .5
        ux, uy = dx / length, dy / length
        bx, by = x2 - 7 * ux, y2 - 7 * uy
        self.svg.append(
            f'<polygon points="{x2},{y2} {bx-3*uy},{by+3*ux} '
            f'{bx+3*uy},{by-3*ux}" fill="#{BLUE}"/>'
        )

    def card(self, key, x, y, number, title, body, accent=False):
        self.rect(key, x, y, BOX_W, BOX_H,
                  BLUE if accent else PALE, BLUE if accent else BORDER)
        color = "FFFFFF" if accent else INK
        self.text(key + "-step", number, x + 14, y + 10, BOX_W - 28, 14,
                  size=10.5, color=color)
        self.text(key + "-title", title, x + 14, y + 30, BOX_W - 28, 23,
                  size=16, bold=True, color=color)
        self.text(key + "-body", body, x + 14, y + 64, BOX_W - 28, 40,
                  size=12, color=color)

    def header(self, title, subtitle):
        self.rect("title-marker", 0, 24, 10, 38, BLUE, BLUE)
        self.text("title", title, 48, 25, 864, 38, 26, True, BLUE)
        self.text("subtitle", subtitle, 48, 81, 864, 22, 13)

    def save(self, stem):
        path = OUT / (stem + ".svg")
        path.write_text("\n".join(self.svg + ["</svg>"]), encoding="utf-8")
        return self.measurements


def blank(presentation):
    layout = min(presentation.slide_layouts, key=lambda layout: len(layout.placeholders))
    slide = presentation.slides.add_slide(layout)
    for shape in list(slide.shapes):
        shape._element.getparent().remove(shape._element)
    slide._element.set("showMasterSp", "0")
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor(255, 255, 255)
    return slide


def knowledge(slide):
    d = Diagram(slide)
    d.header("专家知识库：建立与更新", "把领域先验组织为“场景特征 → 结构策略”，再用真实适配案例持续补充。")
    d.text("build-lane", "初始建立", 48, 120, 300, 20, 14, True)
    cards = [
        ("sources", "01", "收集知识来源", "论文与领域知识\n模型配置与数据统计"),
        ("extract", "02", "提炼场景先验", "环境态势 · 事件链路\n业务约束与适用条件"),
        ("structure", "03", "组织知识条目", "场景画像 → 结构策略\n覆盖五类原子模块"),
        ("store", "04", "专家知识库", "结构策略 + 来源依据\n人工先验与运行经验"),
    ]
    for x, (key, num, title, body) in zip(XS, cards):
        d.card(key, x, 150, num, title, body, accent=key == "store")
    for index in range(3):
        d.arrow(f"build-{index}", XS[index] + BOX_W + 3, 205, XS[index + 1] - 3, 205)
    d.arrow("retrieval-from-store", 768, 263, 768, 281, head=False)
    d.arrow("retrieval-route", 768, 281, 144, 281, head=False)
    d.arrow("retrieval-to-task", 144, 281, 144, 327)
    d.text("retrieval-label", "按数据画像检索", 360, 263, 200, 15, 10.5)
    d.text("update-lane", "持续使用与更新", 272, 300, 400, 22, 14, True)
    updates = [
        ("retrieve", "01", "新场景调用", "统计数据特征画像\n检索相近场景策略"),
        ("execute", "02", "适配与验证", "运行模型组合方案\n记录指标与成功/失败"),
        ("case", "03", "提炼适配案例", "数据画像 + 实际结构\n实测指标 + 经验结论"),
        ("write", "04", "校验与写回", "检查画像与模块完整性\n新增或更新经验条目"),
    ]
    for x, (key, num, title, body) in zip(XS, updates):
        d.card(key, x, 330, num, title, body)
    for index in range(3):
        d.arrow(f"update-{index}", XS[index] + BOX_W + 3, 385, XS[index + 1] - 3, 385)
    d.arrow("feedback-to-knowledge", 816, 327, 816, 263, dashed=True)
    d.text("write-label", "经验回流", 833, 283, 79, 20, 11.5)
    d.text("takeaway", "更新原则：成功与失败均留证；运行经验独立保存，保留人工先验。", 48, 470, 864, 26, 15, True)
    d.text("legend", "实线：建立 / 执行流程    虚线：经验回流", 48, 510, 864, 17, 10.5)
    slide.notes_slide.notes_text_frame.text = (
        "来源：knowledge/scenario_strategies.json；agents/adaptation/knowledge.py 的 "
        "retrieve、normalise_knowledge_entry、_record_knowledge、_experience_id；"
        "agents/adaptation/prompts/model_composer.py、effect_evaluator.py。"
        "初始来源来自现有手工条目，并不表示实现了自动论文采集。检索为六项数值特征的归一化最近邻。"
        "五类模块为编码器、空间、时间、融合、输出头。入库检查是结构完整性检查，"
        "不是独立实验证明。经验以来源/条目标识区分，同标识更新，不覆盖人工先验。"
    )
    return d.save("专家知识库_建立与更新")


def skill(slide):
    d = Diagram(slide)
    d.header("Agent Skill：基于调参反馈的持续更新", "将每轮搜索、调试和评估结果，沉淀为“数据画像 → 调参方向”的可复用经验。")
    d.text("execute-lane", "调用与执行", 48, 120, 500, 20, 14, True)
    top = [
        ("existing", "01", "现有 Skill", "时间 · 空间 · 损失\n预测模块的调参经验"),
        ("match", "02", "按画像匹配", "检索适用的调参方向\n设定搜索范围与预算"),
        ("trials", "03", "搜索与调试", "超参数搜索 + 性能调试\n记录配置、指标与问题"),
        ("evaluate", "04", "效果评估", "对照任务目标判断达标\n保留成功与失败证据"),
    ]
    for x, (key, num, title, body) in zip(XS, top):
        d.card(key, x, 150, num, title, body, accent=key == "existing")
    for index in range(3):
        d.arrow(f"execute-{index}", XS[index] + BOX_W + 3, 205, XS[index + 1] - 3, 205)
    d.arrow("evaluation-to-evidence", 816, 263, 816, 327)
    d.text("evidence-label", "本轮反馈", 833, 283, 79, 20, 11.5)
    d.text("evolve-lane", "经验提炼与更新", 272, 296, 400, 20, 14, True)
    bottom = [
        ("updated", "08", "更新后的 Skill", "更新索引与维度文档\n供后续任务检索复用"),
        ("validate", "07", "校验与写回", "检查数据画像与调参方向\n独立保存运行经验"),
        ("lesson", "06", "提炼可复用经验", "适用条件 + 参数方向\n有效做法与失败教训"),
        ("evidence", "05", "汇总运行证据", "数据画像 + 调参轨迹\n最优配置 + 最终效果"),
    ]
    for x, (key, num, title, body) in zip(XS, bottom):
        d.card(key, x, 330, num, title, body, accent=key == "updated")
    for index in range(3):
        d.arrow(f"evolve-{index}", XS[index + 1] - 3, 385, XS[index] + BOX_W + 3, 385)
    d.arrow("feedback-to-skill", 144, 327, 144, 263, dashed=True)
    d.text("reuse-label", "后续复用", 48, 283, 81, 20, 11.5)
    d.text("takeaway", "更新对象：有适用条件、调参方向和实测证据的经验条目。", 48, 470, 864, 26, 15, True)
    d.text("legend", "实线：执行 / 更新流程    虚线：更新后的经验供后续任务复用", 48, 510, 864, 17, 10.5)
    slide.notes_slide.notes_text_frame.text = (
        "来源：skills/index.json、skills/{temporal,spatial,loss,head}.md；"
        "agents/adaptation/prompts/hyper_searcher.py、perf_debugger.py、effect_evaluator.py；"
        "agents/adaptation/knowledge.py 的 _record_skill、_append_skill_document。"
        "图中 Skill 指应用侧调参经验条目。超参数搜索智能体与效果评估智能体均可写入经验；"
        "图为一轮调参的概括，不能解释为只有最终评估后才允许写入。"
        "结构校验要求至少一个数值画像特征和至少一个调参方向。"
        "更新索引并追加维度 Markdown；不代表已有自动知识冲突消解或效果提升保证。"
    )
    return d.save("Skill_持续更新")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = ROOT / "技术点三_架构与实验补充.pptx"
    presentation = Presentation(original)
    count = len(presentation.slides)
    assert count == 12
    measurements = {}
    measurements["knowledge"] = knowledge(blank(presentation))
    measurements["skill"] = skill(blank(presentation))
    extended = OUT / "技术点三_架构与实验补充_知识库与Skill图示.pptx"
    presentation.save(extended)
    # Reopen the saved deck before removing the original pages for the companion.
    standalone = Presentation(extended)
    for slide_id in list(standalone.slides._sldIdLst)[:count]:
        standalone.part.drop_rel(slide_id.rId)
        standalone.slides._sldIdLst.remove(slide_id)
    companion = OUT / "专家知识库与Skill_独立流程图.pptx"
    standalone.save(companion)
    (OUT / "diagram_geometry.json").write_text(
        json.dumps(measurements, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_ppt(extended)
    # Import after lxml/pptx to keep the installed shared-library load order.
    import fitz
    full_pdf = fitz.open(extended.with_suffix(".pdf"))
    doc = fitz.open()
    doc.insert_pdf(full_pdf, from_page=count, to_page=count + 1)
    doc.save(companion.with_suffix(".pdf"))
    for page, stem in zip(doc, ("专家知识库_建立与更新", "Skill_持续更新")):
        page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False).save(OUT / (stem + ".png"))
        single = fitz.open()
        single.insert_pdf(doc, from_page=page.number, to_page=page.number)
        single.save(OUT / (stem + ".pdf"))
    print(json.dumps({"output": str(OUT), "original_slides": count,
                      "extended_slides": len(presentation.slides),
                      "editable_companion_slides": len(standalone.slides)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
