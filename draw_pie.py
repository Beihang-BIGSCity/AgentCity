import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from wordcloud import WordCloud  # [新增] 导入词云库

# ==========================================
# 0. 核心工具函数：处理阈值和排序
# ==========================================
def process_data_with_threshold(data_dict, threshold_ratio=0.05):
    """
    1. 按数值从大到小排序
    2. 将占比小于 threshold_ratio 的项合并为 "Others"
    3. 返回 labels 和 sizes 列表
    """
    sorted_items = sorted(data_dict.items(), key=lambda item: item[1], reverse=True)
    
    total = sum(data_dict.values())
    if total == 0:
        return [], []

    final_labels = []
    final_sizes = []
    others_count = 0

    for key, value in sorted_items:
        ratio = value / total
        if ratio >= threshold_ratio:
            final_labels.append(key)
            final_sizes.append(value)
        else:
            others_count += value
    
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
    # ... (保持原有的映射逻辑不变) ...
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
all_keywords = [] # [新增] 用于存储所有关键词

for paper in papers:
    # 统计会议
    conf_raw = paper.get('conference') or paper.get('venue') or "Unknown"
    conf_norm = normalize_conference(conf_raw)
    conference_counts[conf_norm] = conference_counts.get(conf_norm, 0) + 1
    
    # 统计年份
    year = paper.get('year')
    year_str = str(year) if year else "Unknown"
    year_counts[year_str] = year_counts.get(year_str, 0) + 1

    # [新增] 提取并统计关键词
    # 假设 json 中的关键词字段叫 'keywords'，可能是列表也可能是逗号分隔的字符串
    kws = paper.get('keywords', [])
    if isinstance(kws, str):
        # 如果是 "AI, Traffic, DL" 这种字符串格式
        kws = [k.strip() for k in kws.split(',')]
    
    # 简单的清洗：转Title Case以统一格式 (例如 "deep learning" -> "Deep Learning")
    if isinstance(kws, list):
        cleaned_kws = [k.strip().title() for k in kws if k and k.strip()]
        all_keywords.extend(cleaned_kws)

task_counts = {
    "Traffic State Prediction": 31,
    "Traj Location Prediction": 16,
    "Estimated Time of Arrival": 14,
    "Map Matching": 7
}

# ==========================================
# 2. 应用数据处理逻辑
# ==========================================

conf_labels, conf_sizes = process_data_with_threshold(conference_counts, threshold_ratio=0.02)
year_labels, year_sizes = process_data_with_threshold(year_counts, threshold_ratio=0.05)
task_labels, task_sizes = process_data_with_threshold(task_counts, threshold_ratio=0.05)

# [新增] 统计关键词频率
keyword_counts = Counter(all_keywords)
# 如果需要过滤掉太少见的词，可以在这里操作，例如只保留出现次数 > 1 的
# keyword_counts = {k: v for k, v in keyword_counts.items() if v > 1}

# ==========================================
# 3. 绘图函数
# ==========================================

def plot_pie_chart(sizes, labels, title, filename, color_map_name='Set3'):
    plt.figure(figsize=(10, 8))
    
    if color_map_name == 'Pastel1':
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))
    else:
        if len(labels) > 12:
            colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
        else:
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    patches, texts, autotexts = plt.pie(
        sizes, 
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False,
        colors=colors,
        pctdistance=0.85,
        textprops={'fontsize': 11}
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    
    for text in texts: 
        text.set_color('black')
        text.set_weight('medium')
    for autotext in autotexts: 
        autotext.set_color('black')
        autotext.set_weight('bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"已保存饼图: {filename}")

# [新增] 词云绘图函数
def plot_word_cloud(frequency_dict, filename):
    if not frequency_dict:
        print("警告: 没有找到关键词，跳过生成词云。")
        return

    print(f"正在生成词云，共 {len(frequency_dict)} 个唯一关键词...")
    
    # 配置词云对象 

    wc = WordCloud(
        width=1600, 
        height=800, 
        background_color='white', # 背景颜色
        max_words=200,            # 最大显示的词数
        colormap='viridis',       # 颜色风格，可选 'magma', 'inferno', 'plasma' 等
        margin=5
    )
    
    # 根据频率生成
    wc.generate_from_frequencies(frequency_dict)
    
    # 绘图
    plt.figure(figsize=(20, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off') # 不显示坐标轴
    plt.tight_layout(pad=0)
    
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"已保存词云: {filename}")

# ==========================================
# 4. 生成图片
# ==========================================

# 1. 会议
plot_pie_chart(conf_sizes, conf_labels, '', 'pie_conference.png', color_map_name='Set3')

# 2. 年份
plot_pie_chart(year_sizes, year_labels, '', 'pie_year.png', color_map_name='Set3')

# 3. 任务
plot_pie_chart(task_sizes, task_labels, '', 'pie_task.png', color_map_name='Pastel1')

# 4. [新增] 关键词词云
plot_word_cloud(keyword_counts, 'wordcloud_keywords.png')