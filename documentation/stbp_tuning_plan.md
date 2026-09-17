# STBP Hyperparameter Tuning Plan

## Model Overview
- Model: STBP (Spatio-Temporal Backpropagation)
- Paper: Continual Spatio-Temporal Forecasting (ICLR)
- Task: traffic_state_pred
- Config: Bigscity-LibCity/libcity/config/model/traffic_state_pred/STBP.json

## Key Config Discrepancy
- Paper uses learning_rate=0.01 with AdamW optimizer
- LibCity config has learning_rate=0.001 with adam optimizer
- This 10x difference is the highest-priority tuning target

## Proposed Search Space (36 trials)
- learning_rate: [0.001, 0.005, 0.01]
- hidden_channel: [128, 192, 256]
- weight_decay: [0.0, 0.0001]
- dropout: [0.0, 0.1]

## Fixed Parameters
- input_window: 12, output_window: 12
- in_channel: 12, out_channel: 12
- head: 1, batch_size: 64
- learner: adamw, max_epoch: 100, patience: 30

## Execution Command
cd /home/wangwenrui/shk/AgentCity/Bigscity-LibCity
python run_hyper.py --task traffic_state_pred --model STBP --dataset METR_LA --params_file stbp_hyper.txt --hyper_algo grid
