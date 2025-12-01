## 已移植模型

1. PatchSTG
2. LSTGAN
3. MLCAFormer
4. RSTIB
5. GriddedTNP
6. EAC
7. SRSNet

**在LibCity仓库的`/model/traffic_speed_prediction`目录下**

## 环境变量设置

`export ANTHROPIC_BASE_URL=""`
`export ANTHROPIC_API_KEY=""`

## 运行

`uvicorn server:app --host 0.0.0.0 --port 8000`
