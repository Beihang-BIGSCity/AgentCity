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
        target_path = Path(ROOT_DIR) / 'result' / task / dataset / f'model_name.csv'
        target_path1 = Path(ROOT_DIR) / 'result' / task / dataset / f'model_name.json'
        target_path2 = Path(ROOT_DIR) / 'result' / task / dataset / f'model_name.jsonl'
        if target_path.exists() and target_path1.exists() and target_path2.exists():
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

def run_test_process(args):
    model, dataset, gpu_id = args
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return asyncio.run(run_test(model, dataset, gpu_id, max_epochs=35))

def main():

    #model_list = ['EAC','GriddedTNP','LSTGAN','MLCAFormer','PatchSTG','SRSNet',''ASeer']
    model_list = ['GriddedTNP','MLCAFormer','PatchSTG', 'HSTWAVE','STHSepNet','BigST','DSTMamba','STWave','UniST','LSTTN','LightST','RSTIB','DSTAGNN']
    task = 'traffic_state_pred'
    dataset_list = ['METR_LA','PEMSD7','PEMS_BAY']
    tasks = [(m, d, i % 3+1) for i, (m, d) in enumerate(itertools.product(model_list, dataset_list))]
    
    with Pool(processes=4) as pool:  # 4个进程对应4张卡
        results = pool.map(run_test_process, tasks)


if __name__ == "__main__":
    main()
