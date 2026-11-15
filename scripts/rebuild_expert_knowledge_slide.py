"""Replace the sparse knowledge diagram with a source-backed content overview."""

from collections import Counter
import json
from pathlib import Path

from pptx import Presentation

from build_knowledge_skill_diagrams import Diagram, BLUE, BORDER, PALE
from build_module_benchmark_report import render_ppt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result/expert_knowledge_detail_20261028"
SOURCE = ROOT / "result/historical_stsgcn_supplement_20261028/技术点三_架构与实验补充_含原子模块库与历史指标.pptx"


def section(d, key, title, x, width):
    d.rect(key + "-header", x, 154, width, 28, BLUE, BLUE)
    d.text(key + "-heading", title, x + 8, 159, width - 16, 23,
           14, True, "FFFFFF", centered=True)


def draw(slide, entries):
    for shape in list(slide.shapes):
        shape._element.getparent().remove(shape._element)
    d = Diagram(slide)
    counts = Counter(e["family"] for e in entries)
    evolved = [e for e in entries if e.get("status") == "evolved"]
    assert counts == {"environment": 11, "causality": 6, "constraint": 7}
    assert len(entries) == 24 and len(evolved) == 4
    example = next(e for e in entries if e["id"] == "kb-pems-bay-acbaycompose-chebgcn-lstm-add-linear-r1")
    assert example["strategy"] == {
        "encoder": "feature_embedding_encoder", "spatial": "cheb_gcn", "temporal": "lstm",
        "fusion": "add", "head": "linear_head",
    }
    profile = example["feature_profile"]

    d.header("专家知识库：内容构成、知识条目与更新",
             "24 条知识 = 20 条初始先验 + 4 条运行经验；为“场景画像 → 模型结构策略”提供依据。")
    d.rect("initial-build", 36, 111, 888, 28, PALE, BORDER)
    d.text("initial-sources", "初始建立：论文方法与模块特性 + LibCity 配置 + 场景数据统计 → 人工整理为 20 条先验",
           48, 117, 864, 20, 11.5)
    d.arrow("initial-to-entry", 475, 142, 475, 151)

    section(d, "contents", "知识内容 · 三类 24 条", 36, 284)
    blocks = [
        ("environment", "环境态势 · 11 条（含 4 条回流）", 188, 68,
         "METR-LA、PEMS-BAY、PEMSD4/7\n波动 / 持续性 / 异质性 / 稀疏采样"),
        ("causality", "事件因果链路 · 6 条", 262, 68,
         "事故传播 / 天气退化 / 节假日漂移\n多物理量耦合 / 拥堵回流 / 场馆激增"),
        ("constraint", "业务场景约束 · 7 条", 336, 78,
         "CPU部署 / 无邻接 / 大网络显存预算\n长预测窗 / 短历史窗 / 冷启动新节点\n高缺失率与传感器掉线"),
    ]
    for key, title, y, height, body in blocks:
        d.rect(key + "-card", 36, y, 284, height, PALE, BORDER)
        d.text(key + "-title", title, 46, y + 5, 264, 20, 12, True)
        d.text(key + "-examples", body, 46, y + 26, 264, height - 27, 10.5)

    section(d, "schema", "每条知识保存什么", 340, 270)
    d.rect("entry-schema", 340, 188, 270, 226, "FFFFFF", BORDER)
    fields = [
        ("profile", "① 场景画像 · 六项检索特征", 198,
         "CV整体 / CV节点 / 空间异质性\n时间自相关 / 采样间隔 / 节点数", 221),
        ("strategy", "② 结构策略 · 五类模块选择", 263,
         "编码器 / 空间 / 时间 / 融合 / 输出头", 286),
        ("evidence", "③ 调参提示与支撑证据", 311,
         "指标 / 训练预算 / 稳定性 / 产物路径", 334),
        ("source", "④ 来源与记录状态", 359,
         "论文 / 配置 / 任务ID / 时间 / 状态", 382),
    ]
    for key, label, label_y, body, body_y in fields:
        d.text(key + "-field", label, 350, label_y, 250, 20, 12, True)
        d.text(key + "-value", body, 350, body_y, 250, 36 if "\n" in body else 20, 10.5)

    section(d, "example", "真实条目 · PEMS-BAY", 630, 294)
    d.rect("real-example", 630, 188, 294, 226, "FFFFFF", BORDER)
    d.text("example-pattern", "低空间异质性 + 强时间持续性", 642, 199, 270, 20, 12, True)
    d.text("example-nodes", f"节点数 {int(profile['node_count'])} / 采样间隔 {int(profile['sampling_interval_min'])} min",
           642, 223, 270, 19, 11)
    d.text("example-stats", f"空间异质性 {profile['spatial_heterogeneity']:.4f} / 时间自相关 {profile['rho1']:.4f}",
           642, 241, 270, 19, 11)
    for key, label, y in [
        ("encoder", "特征嵌入（Linear + ReLU）", 267),
        ("branches", "ChebGCN ∥ LSTM", 306),
        ("output", "Add → Linear Head", 345),
    ]:
        d.rect("example-" + key, 646, y, 262, 26, PALE, BLUE)
        d.text("example-" + key + "-label", label, 654, y + 4, 246, 20, 11.5, centered=True)
    d.arrow("example-flow-1", 777, 296, 777, 303)
    d.arrow("example-flow-2", 777, 335, 777, 342)
    d.text("example-run", "运行证据：1 epoch / seed 0 / 指标与日志", 642, 383, 270, 18, 10.5)
    d.text("example-normalization", "数据已标准化，使用嵌入避免重复归一化", 642, 399, 270, 15, 10)

    d.text("lifecycle", "调用与更新", 36, 425, 220, 22, 13, True)
    d.text("writeback-label", "经验入库", 622, 419, 126, 15, 10.5)
    flow = [
        ("new-task", "新场景画像", "6 项统计特征"),
        ("retrieve", "数值近邻检索", "匹配场景策略"),
        ("compose", "模型组合", "选择五类模块"),
        ("run", "运行验证", "记录成功 / 失败"),
        ("lesson", "经验提炼", "画像 + 结构 + 证据"),
        ("write", "校验写回", "保留人工先验"),
    ]
    for i, (key, title, body) in enumerate(flow):
        x = 36 + i * 150
        d.rect(key, x, 454, 138, 40, PALE, BORDER)
        d.text(key + "-title", title, x + 4, 459, 130, 18, 11.5, True, centered=True)
        d.text(key + "-body", body, x + 4, 479, 130, 15, 10, centered=True)
        if i < len(flow) - 1:
            d.arrow("flow-" + key, x + 140, 474, x + 147, 474)
    d.arrow("return-up", 855, 451, 855, 435, dashed=True, head=False)
    d.arrow("return-across", 855, 435, 475, 435, dashed=True, head=False)
    d.arrow("return-to-entry", 475, 435, 475, 417, dashed=True)
    d.text("implementation-boundary", "当前按 6 项数值画像检索；事件类条目为场景先验。4 条运行经验主要来自短轮次试运行。",
           36, 518, 888, 17, 10.5)
    slide.notes_slide.notes_text_frame.text = (
        "本页基于 knowledge/scenario_strategies.json 当前 24 条真实记录重绘。"
        "类别计数为 environment=11（7条初始+4条运行经验）、causality=6、constraint=7。"
        "数据检索字段：agents/adaptation/knowledge.py 的 FEATURE_KEYS。"
        "距离在双方共有的数值特征上计算，以语料特征极差归一化后平均，再返回最近邻；"
        "不包含事件识别或因果图推理。事件场景条目是人工整理的先验建议。"
        "条目包含 id、family、title、feature_profile、strategy、tuning_direction、evidence、source；"
        "evolved 条目另外记录 status 和 recorded_at，部分 tuning_direction 为空。"
        "右侧例子来自 " + example["id"] + "，来源 " + example["source"] + "。"
        "例子1epoch预算只是短轮试运行证据；不能当作PPT中15epochs三seed正式实验的结果。"
        "初始先验引用论文、模型配置和数据统计，不表示实现了自动文献采集。"
        "运行经验通过 record_experience 写回；字段完整性校验不等同于实验证明；保留人工先验。"
        "所有场景例子应受当前任务的数据契约与硬件预算约束，不能把旧CPU场景条目当作全局硬件事实。"
        "完整条目标题：" + json.dumps([{"family": e["family"], "title": e["title"]} for e in entries], ensure_ascii=False)
    )
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    entries = json.loads((ROOT / "knowledge/scenario_strategies.json").read_text())["entries"]
    presentation = Presentation(SOURCE)
    assert len(presentation.slides) == 15
    diagram = draw(presentation.slides[12], entries)
    target = OUT / "技术点三_架构与实验补充_专家知识库详解.pptx"
    presentation.save(target)
    companion = Presentation(target)
    for i, slide_id in reversed(list(enumerate(list(companion.slides._sldIdLst)))):
        if i != 12:
            companion.part.drop_rel(slide_id.rId)
            companion.slides._sldIdLst.remove(slide_id)
    companion.save(OUT / "专家知识库_内容与更新.pptx")
    (OUT / "专家知识库_内容与更新.svg").write_text("\n".join(diagram.svg + ["</svg>"]), encoding="utf-8")
    (OUT / "geometry.json").write_text(json.dumps(diagram.measurements, ensure_ascii=False, indent=2), encoding="utf-8")
    render_ppt(target)
    import fitz
    pdf = fitz.open(target.with_suffix(".pdf"))
    pdf[12].get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False).save(OUT / "专家知识库_内容与更新.png")
    single = fitz.open()
    single.insert_pdf(pdf, from_page=12, to_page=12)
    single.save(OUT / "专家知识库_内容与更新.pdf")
    print(json.dumps({"pptx": str(target), "replaced_page": 13, "slides": 15}, ensure_ascii=False))


if __name__ == "__main__":
    main()
