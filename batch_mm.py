#!/usr/bin/env python3
"""
批量运行脚本：根据 catalog.json 对每篇论文执行迁移、调参和性能测试，
最终生成 Excel 对比表格。支持并发处理多篇论文。

Usage:
    python batch_runner.py --catalog data/articles/catalog.json --output results.xlsx
    python batch_runner.py --catalog data/articles/catalog.json --skip-migration --skip-tuning
    python batch_runner.py --catalog data/articles/catalog.json --models STGCN,DCRNN
    python batch_runner.py --catalog data/articles/catalog.json --concurrency 3
"""
from multiprocessing import Pool
import time
import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import shutil
import pandas as pd
import concurrent.futures
import itertools
from pathlib import Path
from multiprocessing import Pool, Manager
import pynvml

list = ['EAC','GriddedTNP','LSTGAN','MLCAFormer','PatchSTG','SRSNet']
# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent
LIBCITY_DIR = ROOT_DIR / "Bigscity-LibCity"
DEFAULT_CATALOG = ROOT_DIR / "test_flow.json"
DEFAULT_OUTPUT = ROOT_DIR / "benchmark_results.csv"
LOGS_DIR = ROOT_DIR / "batch_logs"  # 日志目录



async def _run_subprocess(
        cmd: List[str],
        cwd: Path,
        log_file: Optional[Path] = None,
        prefix: str = "",
    ) -> Tuple[int, str, str]:
        """异步运行子进程，实时输出并写入日志文件"""
        stdout_lines: List[str] = []
        stderr_lines: List[str] = []

        # 准备日志文件
        if log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"{'='*60}\n\n")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            async def read_stream(stream, lines: List[str], stream_name: str):
                """实时读取并输出流"""
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="replace").rstrip()
                    lines.append(decoded)
                    # 实时打印（带前缀区分不同模型）
                    if prefix:
                        print(f"[{prefix}] {decoded}")
                    else:
                        print(decoded)
                    # 实时写入日志
                    if log_file:
                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write(f"[{stream_name}] {decoded}\n")

            try:
                # 同时读取 stdout 和 stderr
                await asyncio.wait_for(
                    asyncio.gather(
                        read_stream(process.stdout, stdout_lines, "OUT"),
                        read_stream(process.stderr, stderr_lines, "ERR"),
                    ),
                    timeout=86400  # 24小时
                )
                await process.wait()

                # 记录返回码
                if log_file:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(f"\n[Return Code: {process.returncode}]\n")

                return (
                    process.returncode or 0,
                    "\n".join(stdout_lines),
                    "\n".join(stderr_lines),
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return -1, "\n".join(stdout_lines), "Timeout"
        except Exception as e:
            if log_file:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n[EXCEPTION: {str(e)}]\n")
            print(f"[{prefix}] EXCEPTION: {str(e)}")
            return -1, "", str(e)

def _parse_metrics(task, model_name, dataset, exp_id):
        log_dir = LIBCITY_DIR / "libcity" / "cache" / str(exp_id) / "evaluate_cache"
        #import pdb; pdb.set_trace()
        for file in log_dir.iterdir():
            if file.suffix == '.csv' or file.suffix == '.json' or file.suffix == '.jsonl':
                print(file)
                new_name = f'{model_name}{file.suffix}'   # 随意改名字
                target_dir = Path(ROOT_DIR) / 'result' / task / dataset
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, target_dir / new_name)

async def run_test(model_name,dataset,gpu_id,max_epochs=35,task='traffic_state_pred'):
        """运行单个测试"""
        print(f"[INFO] Testing {model_name} on {dataset}...")

        # 标准化数据集名称
        target_path = Path(ROOT_DIR) / 'result' / task / dataset / f'{model_name}.csv'
        target_path1 = Path(ROOT_DIR) / 'result' / task / dataset / f'{model_name}.json'
        target_path2 = Path(ROOT_DIR) / 'result' / task / dataset / f'{model_name}.jsonl'
        if target_path.exists() or target_path1.exists() or target_path2.exists():
            print(f"[INFO] Result for {model_name} on {dataset} already exists. Skipping...")
            return
        # 日志文件
        log_file = LOGS_DIR / f"{model_name}_test.log"

        start_time = datetime.now()
        import random
        exp_id = random.randint(100000, 300000)
        cmd = [
            sys.executable,
            "run_model.py",
            "--task", task,
            "--model", model_name,
            "--dataset", dataset,
            "--train", "true",
            "--max_epoch", str(max_epochs),
            "--gpu_id", f"{gpu_id}",
            "--exp_id", str(exp_id),
        ]

        returncode, stdout, stderr = await _run_subprocess(
            cmd, LIBCITY_DIR,
            log_file=log_file,
            prefix=f"{model_name}:test:{dataset}",
        )
        while returncode < 0:
            time.sleep(1600)
            returncode, stdout, stderr = await _run_subprocess(
            cmd, LIBCITY_DIR,
            log_file=log_file,
            prefix=f"{model_name}:test:{dataset}",
        )

        runtime = (datetime.now() - start_time).total_seconds()
        _parse_metrics(task, model_name, dataset, exp_id)

def worker(model, dataset, gpu_id, max_epochs=35):
    """线程入口：捕获异常，防止一个任务崩掉整个池"""
    try:
        run_test(model, dataset, gpu_id, max_epochs=max_epochs)
    except Exception as e:
        print(f"[Error] {model}-{dataset} on GPU{gpu_id}: {e}")

def get_gpu_free_memory():
    """获取所有GPU的空闲显存(GB)，返回 [(gpu_id, free_memory), ...]"""
    try:
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        gpu_info = []
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gb = info.free / 1024**3
            gpu_info.append((i, free_gb))
        # 按空闲显存从大到小排序
        return sorted(gpu_info, key=lambda x: x[1], reverse=True)
    except Exception as e:
        print(f"警告: 无法获取GPU显存信息: {e}")
        return [(0, 0), (1, 0), (2, 0), (3,0)]  # 默认3卡
    
def run_test_process(args):
    model, dataset, task, gpu_queue = args
    
    # 从队列阻塞式获取GPU（自动等待直到有可用）
    gpu_id = gpu_queue.get()
    
    try:
        # 关键：必须在import torch之前设置！
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # 延迟导入torch，确保环境变量生效
        import torch
        torch.cuda.set_device(0)  # 因为设置了VISIBLE_DEVICES，这里永远是0
        
        print(f"[{model}-{dataset}] 运行在 GPU {gpu_id} (剩余显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB)")
        
        # 执行你的异步任务
        result = asyncio.run(run_test(model, dataset, 0, task=task, max_epochs=35))
        return result
        
    except Exception as e:
        print(f"[{model}-{dataset}] GPU {gpu_id} 出错: {str(e)[:100]}")
        raise
    finally:
        # 任务完成，归还GPU到队列，让下一个任务自动获取
        gpu_queue.put(gpu_id)
        print(f"[{model}-{dataset}] 释放 GPU {gpu_id}")

def main():

    #model_list = ['EAC','GriddedTNP','LSTGAN','MLCAFormer','PatchSTG','SRSNet',''ASeer']
    #model_list = ['GriddedTNP','STHSepNet','BigST','STWave','UniST','LSTTN','LightST','RSTIB','DSTAGNN','STID']
    #model_list = ["PatchTST","DCST","STLLM","TGraphormer","CKGGNN","EasyST","LEAF","MetaDG","TRACK","HiMSNet","DST2former"]
    '''model_list = ["AutoSTF", "STSSDL", "STDMAE", "LSTTN","UniST","STID","ConvTimeNet","FlashST","Fredformer","GNNRF","PatchSTG","PatchTST","RevIN"
                  ,"Trafformer",'DMSTGCN','GWNET','STAEformer','STMGAT','TGCLSTM','STID','Trafformer','D2STGNN'
                  ,'STWave','TimeMixer','STResNet','STDN','ResLSTM','DSTAGNN','LSTTN','STTSNet','STID','Trafformer',"ConvTimeNet","HTVGNN","Pathformer","DCST","CKGGNN","HiMSNet","LEAF"]'''
    #model_list = ['STID','Trafformer',"ConvTimeNet","HTVGNN","Pathformer","DCST","CKGGNN","HiMSNet","LEAF"]
    model_list = ['DeepMM','DiffMM','RLOMM','L2MM','FMM','HMMM','IVMM','STMatching']
    task = 'map_matching'
    dataset_list = ['Neftekamsk','Valky','Santander','Spaichingen']
    manager = Manager()
    gpu_queue = manager.Queue()
    
    # 查询当前GPU状态，按空闲显存排序后初始化队列
    # 这样显存占用小的GPU会先被分配
    gpu_status = get_gpu_free_memory()
    print(f"GPU状态 (ID: 空闲显存): {gpu_status}")
    
    for gpu_id, free_mem in gpu_status:
        gpu_queue.put(gpu_id)
        print(f"GPU {gpu_id} 已加入队列 (空闲 {free_mem:.1f}GB)")
    
    # 构建任务列表，传入共享队列
    tasks = [(m, d, task, gpu_queue) 
             for m, d in itertools.product(model_list, dataset_list)]
    
    # 进程数 = GPU数量，确保每个GPU同一时间只跑一个任务
    num_gpus = 1
    with Pool(processes=num_gpus) as pool:
        results = pool.map(run_test_process, tasks)


if __name__ == "__main__":
    main()
