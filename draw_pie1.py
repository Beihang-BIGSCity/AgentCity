import json
import matplotlib.pyplot as plt

# ==========================================
# 0. 核心工具函数：处理阈值和排序
# ==========================================
def process_data_with_threshold(data_dict, threshold_ratio=0.05):
    """
    1. 按数值从大到小排序
    2. 将占比小于 threshold_ratio 的项合并为 "Others"
    3. 返回 labels 和 sizes 列表
    """
    # 1. 先按数量从大到小排序
    sorted_items = sorted(data_dict.items(), key=lambda item: item[1], reverse=True)
    
    total = sum(data_dict.values())
    if total == 0:
        return [], []

    final_labels = []
    final_sizes = []
    others_count = 0

    for key, value in sorted_items:
        ratio = value / total
        # 判断占比是否大于等于阈值
        if ratio >= threshold_ratio:
            final_labels.append(key)
            final_sizes.append(value)
        else:
            others_count += value
    
    # 如果有合并项，将 Others 加到最后
    if others_count > 0:
        final_labels.append("Others")
        final_sizes.append(others_count)
        
    return final_labels, final_sizes

# ==========================================
# 1. 数据读取与预处理
# ==========================================

def normalize_conference(conf_str):
    if not conf_str:
        return "Unknown"
    conf_upper = conf_str.upper()
    
    mappings = {
        "NEURIPS": "NeurIPS", "KDD": "KDD", "ICLR": "ICLR", "AAAI": "AAAI",
        "IJCAI": "IJCAI", "SIGIR": "SIGIR", "CIKM": "CIKM", "WWW": "WWW",
        "THE WEB": "WWW", "ICDE": "ICDE", "VLDB": "VLDB", "SIGMOD": "SIGMOD",
        "SIGSPATIAL": "SIGSPATIAL", "GIS": "SIGSPATIAL", "IEEE T-ITS": "IEEE T-ITS",
        "INTELLIGENT TRANSPORTATION SYSTEMS": "IEEE T-ITS", "IEEE TKDE": "IEEE TKDE",
        "KNOWLEDGE AND DATA ENGINEERING": "IEEE TKDE", "ACM TIST": "ACM TIST",
        "ACM TKDD": "ACM TKDD", "ICML": "ICML", "ACL": "ACL", "CVPR": "CVPR",
        "ECCV": "ECCV", "ICCV": "ICCV", "ARXIV": "arXiv", "IEEE RA-L": "IEEE RA-L",
        "EDBT": "EDBT", "AGILE": "AGILE", "SCIENTIFIC REPORTS": "Scientific Reports",
        "KNOWLEDGE-BASED SYSTEMS": "Knowledge-Based Systems", "IEEE MDM": "IEEE MDM",
        "ICORES": "ICORES"
    }
    
    for key, value in mappings.items():
        if key in conf_upper:
            return value
    return conf_str

file_path = 'migration_flow.json'
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"错误: 找不到文件 {file_path}")
    exit()

unique_papers = {}
for entry in data:
    title = entry.get('title', '').strip()
    if title:
        unique_papers[title] = entry
papers = list(unique_papers.values())

conference_counts = {}
year_counts = {}

for paper in papers:
    conf_raw = paper.get('conference') or paper.get('venue') or "Unknown"
    conf_norm = normalize_conference(conf_raw)
    conference_counts[conf_norm] = conference_counts.get(conf_norm, 0) + 1
    
    year = paper.get('year')
    year_str = str(year) if year else "Unknown"
    year_counts[year_str] = year_counts.get(year_str, 0) + 1

task_counts = {
    "Traffic State Prediction": 36,
    "Trajectory Location Prediction": 18,
    "ETA Prediction": 11,
    "Map Matching": 9
}

# ==========================================
# 2. 应用数据处理逻辑 (关键修改点)
# ==========================================

# --- 修改点：会议图使用 0.02 (2%) 的阈值 ---
conf_labels, conf_sizes = process_data_with_threshold(conference_counts, threshold_ratio=0.02)

# 其他图保持 0.05 (5%) 或根据需要调整
year_labels, year_sizes = process_data_with_threshold(year_counts, threshold_ratio=0.05)
task_labels, task_sizes = process_data_with_threshold(task_counts, threshold_ratio=0.05)

# ==========================================
# 3. 统一配色方案
# ==========================================

PALETTE = [
    '#7BA3CC', '#F2C47A', '#8CC5A2', '#E8A0A0', '#B0A7D4',
    '#C4B49A', '#E8B8D4', '#B5B5B5', '#D9CF9E', '#95CDE0',
    '#8DD4D0', '#F0B87A', '#9AA3CB', '#8FD4BE', '#BDD88A',
    '#D9A0CC', '#6BB5A8', '#E8C86A', '#7DC0E0', '#A98DC4',
]

def get_colors(n):
    return [PALETTE[i % len(PALETTE)] for i in range(n)]

def make_autopct(sizes, show_count=False):
    """生成autopct函数，show_count=True显示实际数量，False显示百分比"""
    def autopct(pct):
        if show_count:
            total = sum(sizes)
            val = int(round(pct * total / 100.0))
            return f'{val}'
        else:
            return f'{pct:.1f}%'
    return autopct

# 控制饼图显示模式：True=显示数量，False=显示百分比
SHOW_COUNT = False

# ==========================================
# 4. 绘制横向三合一图
# ==========================================

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 20,
})

datasets = [
    (conf_sizes, conf_labels, '(a) Venue Distribution'),
    (year_sizes, year_labels, '(b) Year Distribution'),
    (task_sizes, task_labels, '(c) Task Distribution'),
]

fig, axes = plt.subplots(1, 3, figsize=(30, 9))

for ax, (sizes, labels, title) in zip(axes, datasets):
    colors = get_colors(len(labels))

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct=make_autopct(sizes, SHOW_COUNT),
        startangle=90,
        counterclock=False,
        colors=colors,
        pctdistance=0.75,
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},
        textprops={'fontsize': 18},
    )

    for at in autotexts:
        at.set_fontsize(18)
        at.set_color('#333333')
        at.set_fontweight('bold')
        pct_val = float(at.get_text().strip('%'))
        if pct_val < 5.0:
            at.set_visible(False)

    # 缩短过长的图例标签
    short_labels = [l.replace('Trajectory Location', 'Traj. Loc.') for l in labels]

    ax.legend(
        wedges, short_labels,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        fontsize=18,
        frameon=False,
        handlelength=1.4,
        handleheight=1.4,
    )

    ax.set_title(title, fontsize=24, fontweight='bold', pad=20)

plt.tight_layout(rect=[0, 0, 1, 1], w_pad=6)
plt.savefig('pie_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("已保存: pie_combined.png")
