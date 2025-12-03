## 已移植模型

1. PatchSTG
2. LSTGAN
3. MLCAFormer
4. RSTIB
5. GriddedTNP
6. EAC
7. SRSNet
8. ST-SSDL

**在LibCity仓库的`/model/traffic_speed_prediction`目录下**

## Quick Start

### 安装依赖
`pip install -r requirements.txt`

### 设置环境变量
`export ANTHROPIC_BASE_URL=""`

`export ANTHROPIC_API_KEY=""`

### 运行

`uvicorn server:app --host 0.0.0.0 --port 8000`

### 验证迁移效果

需自行下载LibCity数据集并放入`/Bigscity-LibCity/raw_data`，然后运行`python run_model.py --task traffic_state_pred --model model_name --dataset dataset_name`
