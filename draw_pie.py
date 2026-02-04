import json
import matplotlib.pyplot as plt

# 1. 定义会议名称标准化函数 (保持不变)
def normalize_conference(conf_str):
    if not conf_str:
        return "Unknown"
    conf_upper = conf_str.upper()
    
    mappings = {
        "NEURIPS": "NeurIPS",
        "KDD": "KDD",
        "ICLR": "ICLR",
        "AAAI": "AAAI",
        "IJCAI": "IJCAI",
        "SIGIR": "SIGIR",
        "CIKM": "CIKM",
        "WWW": "WWW",
        "THE WEB": "WWW",
        "ICDE": "ICDE",
        "VLDB": "VLDB",
        "SIGMOD": "SIGMOD",
        "SIGSPATIAL": "SIGSPATIAL",
        "GIS": "SIGSPATIAL",
        "IEEE T-ITS": "IEEE T-ITS",
        "INTELLIGENT TRANSPORTATION SYSTEMS": "IEEE T-ITS",
        "IEEE TKDE": "IEEE TKDE",
        "KNOWLEDGE AND DATA ENGINEERING": "IEEE TKDE",
        "ACM TIST": "ACM TIST",
        "ACM TKDD": "ACM TKDD",
        "ICML": "ICML",
        "ACL": "ACL",
        "CVPR": "CVPR",
        "ECCV": "ECCV",
        "ICCV": "ICCV",
        "ARXIV": "arXiv",
        "IEEE RA-L": "IEEE RA-L",
        "EDBT": "EDBT",
        "AGILE": "AGILE",
        "SCIENTIFIC REPORTS": "Scientific Reports",
        "KNOWLEDGE-BASED SYSTEMS": "Knowledge-Based Systems",
        "IEEE MDM": "IEEE MDM",
        "ICORES": "ICORES"
    }
    
    for key, value in mappings.items():
        if key in conf_upper:
            return value
            
    return conf_str

# 2. 读取数据
file_path = 'migration_flow.json'
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"错误: 找不到文件 {file_path}")
    exit()

# 数据去重
unique_papers = {}
for entry in data:
    title = entry.get('title', '').strip()
    if title:
        unique_papers[title] = entry

papers = list(unique_papers.values())

# 3. 统计数据
conference_counts = {}
year_counts = {}  # 新增：年份统计字典

for paper in papers:
    # 统计会议
    conf_raw = paper.get('conference') or paper.get('venue') or "Unknown"
    conf_norm = normalize_conference(conf_raw)
    conference_counts[conf_norm] = conference_counts.get(conf_norm, 0) + 1
    
    # 统计年份 (新增)
    # 获取年份，如果没有则标记为 'Unknown'，并将数字转为字符串以便作为标签
    year = paper.get('year')
    if year:
        year_str = str(year)
    else:
        year_str = "Unknown"
    year_counts[year_str] = year_counts.get(year_str, 0) + 1

# 4. 准备绘图数据并排序
# 会议排序
sorted_conf = dict(sorted(conference_counts.items(), key=lambda item: item[1], reverse=True))
conf_labels = list(sorted_conf.keys())
conf_sizes = list(sorted_conf.values())

# 年份排序 (按年份本身排序通常更直观，而不是按数量)
# 尝试将年份转为整数排序，处理不了的（如Unknown）放最后
def sort_year_key(key):
    try:
        return int(key)
    except ValueError:
        return 9999  # 将Unknown放到最后

sorted_year_keys = sorted(year_counts.keys(), key=sort_year_key)
# 或者如果你想按数量排序（哪一年论文最多），使用下面这行：
# sorted_year_keys = sorted(year_counts.keys(), key=lambda k: year_counts[k], reverse=True)

year_labels = sorted_year_keys
year_sizes = [year_counts[k] for k in sorted_year_keys]

# 5. 绘制图形 (使用 Subplots)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))  # 1行2列

# --- 图1：会议分布 ---
patches1, texts1, autotexts1 = ax1.pie(
    conf_sizes, 
    labels=conf_labels,
    autopct='%1.1f%%',
    startangle=140,
    pctdistance=0.85,
    textprops={'fontsize': 9}
)
ax1.set_title(f'Paper Distribution by Conference', fontsize=14)
# 优化字体颜色
for text in texts1: text.set_color('black')
for autotext in autotexts1: autotext.set_color('white')

# --- 图2：年份分布 (新增) ---
# 建议年份图使用不同的颜色映射，或者使用 Pastel 颜色
colors_year = plt.cm.Set3(range(len(year_labels)))

patches2, texts2, autotexts2 = ax2.pie(
    year_sizes, 
    labels=year_labels,
    autopct='%1.1f%%',
    startangle=90,     # 年份通常从12点方向开始看比较舒服
    colors=colors_year,
    pctdistance=0.75,  # 稍微往里一点
    textprops={'fontsize': 11}
)
ax2.set_title(f'Paper Distribution by Year', fontsize=14)

# 优化年份图字体
for text in texts2: 
    text.set_weight('bold')
for autotext in autotexts2: 
    autotext.set_color('black') # 年份块通常较大，黑色字体在浅色背景下更清晰
    autotext.set_weight('bold')

# 整体设置
plt.suptitle(f'Literature Statistics (Total Papers: {len(papers)})', fontsize=20)
plt.tight_layout()
plt.savefig('pie.png')