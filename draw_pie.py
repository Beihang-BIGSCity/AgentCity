import json
import matplotlib.pyplot as plt
import numpy as np

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
    "Traffic State Prediction": 31,
    "Traj Location Prediction": 16,
    "Estimated Time of Arrival": 14,
    "Map Matching": 7
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
# 3. 绘图函数
# ==========================================

def plot_pie_chart(sizes, labels, title, filename, color_map_name='Set3'):
    plt.figure(figsize=(10, 8))
    
    if color_map_name == 'Pastel1':
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))
    else:
        # 如果类别很多（因为阈值降低了），Set3 只有12种颜色可能不够循环
        # 这里改用 tab20c 或 tab20，颜色更多
        if len(labels) > 12:
            colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
        else:
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    patches, texts, autotexts = plt.pie(
        sizes, 
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,        # 12点方向开始
        counterclock=False,   # 顺时针
        colors=colors,
        pctdistance=0.85,
        textprops={'fontsize': 11} # 字体稍微调小一点，防止重叠
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    
    for text in texts: 
        text.set_color('black')
        text.set_weight('medium')
    for autotext in autotexts: 
        autotext.set_color('black')
        autotext.set_weight('bold')
        # 如果切片太小，隐藏百分比文字以防重叠
        # if float(autotext.get_text().strip('%')) < 2.0:
        #     autotext.set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"已保存: {filename}")

# ==========================================
# 4. 生成图片
# ==========================================

# 1. 会议 (阈值 2%)
plot_pie_chart(conf_sizes, conf_labels, 
               '', 
               'pie_conference.png', 
               color_map_name='Set3') # 内部会自动切换到 tab20 如果类别太多

# 2. 年份
plot_pie_chart(year_sizes, year_labels, 
               '', 
               'pie_year.png', 
               color_map_name='Set3')

# 3. 任务
plot_pie_chart(task_sizes, task_labels, 
               '', 
               'pie_task.png', 
               color_map_name='Pastel1')