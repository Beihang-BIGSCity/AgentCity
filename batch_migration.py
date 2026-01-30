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

import pandas as pd

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent
LIBCITY_DIR = ROOT_DIR / "Bigscity-LibCity"
DEFAULT_CATALOG = ROOT_DIR / "test_flow.json"
DEFAULT_OUTPUT = ROOT_DIR / "benchmark_results.csv"
LOGS_DIR = ROOT_DIR / "batch_logs"  # 日志目录


@dataclass
class PaperEntry:
    """论文条目"""
    title: str
    model_name: str
    datasets: List[str]
    metrics: Dict[str, Any]
    github: Optional[str] = None
    venue: str = ""
    year: int = 0
    pdf_path: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "PaperEntry":
        return cls(
            title=data.get("title", "Unknown"),
            model_name=data.get("model_name", ""),
            datasets=data.get("datasets", []),
            metrics=data.get("metrics", {}),
            github=data.get("github") or data.get("repo_url") or data.get('repository_url'),
            venue=data.get("venue") or data.get("conference", ""),
            year=data.get("year", 0),
            pdf_path=data.get("pdf_path"),
        )


@dataclass
class TestResult:
    """测试结果"""
    model_name: str
    dataset: str
    success: bool
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    error_message: str = ""
    runtime_seconds: float = 0.0
    exp_id: int = 0

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "dataset": self.dataset,
            "success": self.success,
            "MAE": self.mae,
            "RMSE": self.rmse,
            "MAPE": self.mape,
            "error": self.error_message,
            "runtime_s": self.runtime_seconds,
            "exp_id": self.exp_id,
        }


@dataclass
class BatchRunnerConfig:
    """批量运行配置"""
    catalog_path: Path
    output_path: Path
    skip_migration: bool = False
    skip_tuning: bool = False
    skip_test: bool = False
    selected_models: Optional[List[str]] = None
    max_epochs: int = 10
    timeout_seconds: int = 3600  # 1小时超时
    task: str = "traffic_state_pred"
    concurrency: int = 1  # 并发数


class BatchRunner:
    """批量运行器（支持并发）"""

    def __init__(self, config: BatchRunnerConfig):
        self.config = config
        self.papers: List[PaperEntry] = []
        self.test_results: List[TestResult] = []
        self.migration_status: Dict[str, str] = {}
        self.tuning_status: Dict[str, str] = {}
        self._lock = asyncio.Lock()  # 保护共享状态
        self._ensure_logs_dir()

    def _ensure_logs_dir(self) -> None:
        """确保日志目录存在"""
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def load_catalog(self) -> None:
        """加载 catalog.json"""
        if not self.config.catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found: {self.config.catalog_path}")

        with open(self.config.catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            if not item.get("model_name"):
                print(f"[WARN] Skipping paper without model_name: {item.get('title', 'Unknown')}")
                continue
            paper = PaperEntry.from_dict(item)

            # 过滤模型
            if self.config.selected_models:
                if paper.model_name not in self.config.selected_models:
                    continue

            self.papers.append(paper)

        print(f"[INFO] Loaded {len(self.papers)} papers from catalog")

    async def _run_subprocess(
        self,
        cmd: List[str],
        cwd: Path,
        timeout: int,
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
                    timeout=timeout,
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
                if log_file:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(f"\n[TIMEOUT after {timeout} seconds]\n")
                print(f"[{prefix}] TIMEOUT after {timeout} seconds")
                return -1, "\n".join(stdout_lines), "Timeout"
        except Exception as e:
            if log_file:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n[EXCEPTION: {str(e)}]\n")
            print(f"[{prefix}] EXCEPTION: {str(e)}")
            return -1, "", str(e)

    async def run_migration(self, paper: PaperEntry) -> bool:
        """运行迁移工作流"""
        if self.config.skip_migration:
            print(f"[SKIP] Migration skipped for {paper.model_name}")
            async with self._lock:
                self.migration_status[paper.model_name] = "skipped"
            return True

        print(f"[INFO] Running migration for {paper.model_name}...")

        # 检查模型是否已存在于 LibCity
        model_file = LIBCITY_DIR / "libcity" / "model" / "traffic_speed_prediction" / f"{paper.model_name}.py"
        if model_file.exists():
            print(f"[INFO] Model {paper.model_name} already exists in LibCity")
            async with self._lock:
                self.migration_status[paper.model_name] = "exists"
            return True

        # 构建 paper dict，与 catalog.json 格式一致
        paper_dict = {
            "title": paper.title,
            "model_name": paper.model_name,
            "repo_url": paper.github or "",
            "datasets": paper.datasets,
            "conference": paper.venue,
            "year": paper.year,
            "pdf_path": paper.pdf_path or "",
        }

        # 日志文件
        log_file = LOGS_DIR / f"{paper.model_name}_migration.log"

        # 调用 AgentOrchestrator 运行迁移
        script = f"""
import asyncio
import json
from agents import AppPaths, setup_logger
from agents.core.orchestrator import AgentOrchestrator

async def run():
    paths = AppPaths()
    paths.ensure()
    logger = setup_logger(paths.log_file)

    orchestrator = AgentOrchestrator(paths=paths, logger=logger)

    paper = json.loads({repr(json.dumps(paper_dict))})
    await orchestrator.run_migration([paper])

asyncio.run(run())
"""
        cmd = [sys.executable, "-c", script]

        returncode, stdout, stderr = await self._run_subprocess(
            cmd, ROOT_DIR, self.config.timeout_seconds,
            log_file=log_file,
            prefix=f"{paper.model_name}:migration",
        )

        async with self._lock:
            if returncode == 0:
                self.migration_status[paper.model_name] = "success"
                print(f"[OK] Migration completed for {paper.model_name} (log: {log_file})")
                return True
            elif returncode == -1 and stderr == "Timeout":
                self.migration_status[paper.model_name] = "timeout"
                print(f"[ERROR] Migration timeout for {paper.model_name} (log: {log_file})")
                return False
            else:
                self.migration_status[paper.model_name] = f"failed: {stderr}"
                print(f"[ERROR] Migration failed for {paper.model_name} (log: {log_file})")
                return False

    async def run_tuning(self, paper: PaperEntry) -> bool:
        """运行调参工作流"""
        if self.config.skip_tuning:
            print(f"[SKIP] Tuning skipped for {paper.model_name}")
            async with self._lock:
                self.tuning_status[paper.model_name] = "skipped"
            return True

        print(f"[INFO] Running tuning for {paper.model_name}...")

        # 构建 paper dict，与 catalog.json 格式一致
        paper_dict = {
            "title": paper.title,
            "model_name": paper.model_name,
            "repo_url": paper.github or "",
            "datasets": paper.datasets,
            "conference": paper.venue,
            "year": paper.year,
            "pdf_path": paper.pdf_path or "",
        }

        # 日志文件
        log_file = LOGS_DIR / f"{paper.model_name}_tuning.log"

        script = f"""
import asyncio
import json
from agents import AppPaths, setup_logger
from agents.core.orchestrator import AgentOrchestrator

async def run():
    paths = AppPaths()
    paths.ensure()
    logger = setup_logger(paths.log_file)

    orchestrator = AgentOrchestrator(paths=paths, logger=logger)

    paper = json.loads({repr(json.dumps(paper_dict))})
    await orchestrator.run_tuning([paper])

asyncio.run(run())
"""
        cmd = [sys.executable, "-c", script]

        returncode, stdout, stderr = await self._run_subprocess(
            cmd, ROOT_DIR, self.config.timeout_seconds,
            log_file=log_file,
            prefix=f"{paper.model_name}:tuning",
        )

        async with self._lock:
            if returncode == 0:
                self.tuning_status[paper.model_name] = "success"
                print(f"[OK] Tuning completed for {paper.model_name} (log: {log_file})")
                return True
            elif returncode == -1 and stderr == "Timeout":
                self.tuning_status[paper.model_name] = "timeout"
                print(f"[ERROR] Tuning timeout for {paper.model_name} (log: {log_file})")
                return False
            else:
                self.tuning_status[paper.model_name] = f"failed: {stderr}"
                print(f"[ERROR] Tuning failed for {paper.model_name} (log: {log_file})")
                return False

    async def run_test(self, paper: PaperEntry, dataset: str) -> TestResult:
        """运行单个测试"""
        print(f"[INFO] Testing {paper.model_name} on {dataset}...")

        # 标准化数据集名称
        dataset_normalized = self._normalize_dataset_name(dataset)

        # 日志文件
        log_file = LOGS_DIR / f"{paper.model_name}_test.log"

        start_time = datetime.now()
        import random
        exp_id = random.randint(50000, 100000)
        cmd = [
            sys.executable,
            "run_model.py",
            "--task", self.config.task,
            "--model", paper.model_name,
            "--dataset", dataset_normalized,
            "--train", "true",
            "--max_epoch", str(self.config.max_epochs),
            "--gpu_id", "0",
            "--exp_id", str(exp_id),
        ]

        returncode, stdout, stderr = await self._run_subprocess(
            cmd, LIBCITY_DIR, self.config.timeout_seconds,
            log_file=log_file,
            prefix=f"{paper.model_name}:test:{dataset}",
        )

        runtime = (datetime.now() - start_time).total_seconds()

        if returncode == 0:
            # 解析输出获取指标
            mae, rmse, mape = self._parse_metrics(exp_id)
            return TestResult(
                model_name=paper.model_name,
                dataset=dataset,
                success=True,
                mae=mae,
                rmse=rmse,
                mape=mape,
                runtime_seconds=runtime,
                exp_id=exp_id
            )
        elif returncode == -1 and stderr == "Timeout":
            return TestResult(
                model_name=paper.model_name,
                dataset=dataset,
                success=False,
                error_message="Timeout",
                runtime_seconds=self.config.timeout_seconds,
                exp_id=exp_id
            )
        else:
            return TestResult(
                model_name=paper.model_name,
                dataset=dataset,
                success=False,
                error_message=stderr,
                runtime_seconds=runtime,
                exp_id=exp_id
            )

    def _normalize_dataset_name(self, dataset: str) -> str:
        """标准化数据集名称以匹配 LibCity 格式"""
        # 常见映射
        mappings = {
            "METR-LA": "METR_LA",
            "METR_LA": "METR_LA",
            "PeMS-Bay": "PEMS_BAY",
            "PEMS-BAY": "PEMS_BAY",
            "PEMS_BAY": "PEMS_BAY",
            "PeMSD4": "PEMSD4",
            "PeMSD7": "PEMSD7",
            "PeMSD7(M)": "PEMSD7",
            "PeMSD7(L)": "PEMSD7",
            "PeMSD8": "PEMSD8",
            "PEMSD4": "PEMSD4",
            "PEMSD7": "PEMSD7",
            "PEMSD8": "PEMSD8",
            "TaxiBJ": "TaxiBJ",
            "NYC-Taxi": "NYCTaxi",
            "NYC-Bike": "NYCBike",
            "BJER4": "BJER4",
        }
        return mappings.get(dataset, dataset.replace("-", "_"))

    def _parse_metrics(self, exp_id) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        从混合了日志和表格的复杂输出中提取第12行的指标。
        策略：行遍历 + 严格特征匹配
        """
        mae, rmse, mape = None, None, None
        log_dir = LIBCITY_DIR / "libcity" / "cache" / str(exp_id) / "evaluate_cache"
        #import pdb; pdb.set_trace()
        for files in log_dir.iterdir():
            if files.suffix == '.csv':
                output = pd.read_csv(log_dir / files)
                break
        if 'output' not in locals():
            return mae, rmse, mape
        mae = output['masked_MAE'].iloc[11]
        rmse = output['masked_RMSE'].iloc[11]
        mape = output['masked_MAPE'].iloc[11]
        return mae, rmse, mape
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
        
            if not line.startswith('12'):
                continue

            parts = line.split()
            if parts[0] == '12' and len(parts) >= 9:
                try:
                    candidate_mae = float(parts[5])   # masked_MAE
                    candidate_mape = float(parts[6])  # masked_MAPE
                    candidate_rmse = float(parts[8])  # masked_RMSE
                
                    mae, mape, rmse = candidate_mae, candidate_mape, candidate_rmse
                    return mae, rmse, mape 
                
                except (ValueError, IndexError):
                    continue
        if mae is None:
            patterns = [
                (r"MAE[:\s]+([0-9.]+)", r"RMSE[:\s]+([0-9.]+)", r"MAPE[:\s]+([0-9.]+)")
            ]
            for mae_pat, rmse_pat, mape_pat in patterns:
                m_mae = re.search(mae_pat, output, re.IGNORECASE)
                m_rmse = re.search(rmse_pat, output, re.IGNORECASE)
                m_mape = re.search(mape_pat, output, re.IGNORECASE)
                if m_mae and m_rmse and m_mape:
                    return float(m_mae.group(1)), float(m_rmse.group(1)), float(m_mape.group(1))

        return mae, rmse, mape

    async def run_all_tests(self, paper: PaperEntry) -> List[TestResult]:
        """运行论文的所有数据集测试"""
        if self.config.skip_test:
            print(f"[SKIP] Tests skipped for {paper.model_name}")
            return []

        results = []
        for dataset in paper.datasets:
            result = await self.run_test(paper, dataset)
            results.append(result)
            async with self._lock:
                self.test_results.append(result)

            if result.success:
                print(f"  [OK] {dataset}: MAE={result.mae}, RMSE={result.rmse}, MAPE={result.mape}")
            else:
                print(f"  [FAIL] {dataset}: {result.error_message}")

        return results

    async def process_paper(self, paper: PaperEntry, index: int, total: int) -> None:
        """处理单篇论文的完整流程"""
        print(f"\n{'='*60}")
        print(f"[{index}/{total}] Processing: {paper.model_name} ({paper.title[:50]}...)")
        print(f"{'='*60}")

        # 1. 运行迁移
        migration_ok = await self.run_migration(paper)

        # 2. 运行调参
        '''if migration_ok:
            await self.run_tuning(paper)'''

        # 3. 运行测试
        if migration_ok:
            await self.run_all_tests(paper)

    def generate_excel_report(self) -> None:
        """生成 Excel 对比报告"""
        print(f"\n[INFO] Generating Excel report: {self.config.output_path}")

        # 准备数据
        comparison_data = []

        for paper in self.papers:
            for dataset in paper.datasets:
                # 获取原论文指标
                paper_metrics = self._get_paper_metrics(paper, dataset)

                # 获取测试结果
                test_result = self._find_test_result(paper.model_name, dataset)

                row = {
                    "Model": paper.model_name,
                    "Dataset": dataset,
                    "Venue": paper.venue,
                    "Year": paper.year,
                    # 原论文指标
                    "Paper_MAE": paper_metrics.get("MAE"),
                    "Paper_RMSE": paper_metrics.get("RMSE"),
                    "Paper_MAPE": paper_metrics.get("MAPE"),
                    # 测试指标
                    "Test_MAE": test_result.mae if test_result else None,
                    "Test_RMSE": test_result.rmse if test_result else None,
                    "Test_MAPE": test_result.mape if test_result else None,
                    "exp_id": test_result.exp_id if test_result else None,
                    # 状态
                    "Test_Status": "Success" if test_result and test_result.success else "Failed" if test_result else "Not Run",
                    "Migration_Status": self.migration_status.get(paper.model_name, "N/A"),
                    "Tuning_Status": self.tuning_status.get(paper.model_name, "N/A"),
                    "Runtime_s": test_result.runtime_seconds if test_result else None,
                }

                # 计算差异
                if row["Paper_MAE"] and row["Test_MAE"]:
                    row["MAE_Diff"] = row["Test_MAE"] - row["Paper_MAE"]
                    row["MAE_Diff%"] = (row["MAE_Diff"] / row["Paper_MAE"]) * 100
                if row["Paper_RMSE"] and row["Test_RMSE"]:
                    row["RMSE_Diff"] = row["Test_RMSE"] - row["Paper_RMSE"]
                    row["RMSE_Diff%"] = (row["RMSE_Diff"] / row["Paper_RMSE"]) * 100

                comparison_data.append(row)

        # 创建 DataFrame
        df = pd.DataFrame(comparison_data)

        # 创建 Excel writer
        #with pd.ExcelWriter(self.config.output_path, engine="openpyxl") as writer:
            # 主对比表
        df.to_csv(self.config.output_path, index=False)

        '''# 摘要表
            summary_data = []
            for paper in self.papers:
                summary_data.append({
                    "Model": paper.model_name,
                    "Title": paper.title,
                    "Venue": paper.venue,
                    "Year": paper.year,
                    "Datasets": ", ".join(paper.datasets),
                    "GitHub": paper.github or "N/A",
                    "Migration_Status": self.migration_status.get(paper.model_name, "N/A"),
                    "Tuning_Status": self.tuning_status.get(paper.model_name, "N/A"),
                })

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            # 测试结果明细
            if self.test_results:
                results_df = pd.DataFrame([r.to_dict() for r in self.test_results])
                results_df.to_excel(writer, sheet_name="Test_Results", index=False)'''

        print(f"[OK] Excel report saved to: {self.config.output_path}")

    def _get_paper_metrics(self, paper: PaperEntry, dataset: str) -> Dict[str, Any]:
        """获取论文中报告的指标"""
        metrics = paper.metrics

        # 尝试直接匹配
        if dataset in metrics:
            return self._normalize_metrics(metrics[dataset])

        # 尝试模糊匹配
        dataset_lower = dataset.lower().replace("-", "_").replace(" ", "_")
        for key, value in metrics.items():
            key_lower = key.lower().replace("-", "_").replace(" ", "_")
            if dataset_lower in key_lower or key_lower in dataset_lower:
                return self._normalize_metrics(value)

        return {}

    def _normalize_metrics(self, metrics: Any) -> Dict[str, Any]:
        """标准化指标格式"""
        if not isinstance(metrics, dict):
            return {}

        result = {}
        for key, value in metrics.items():
            key_upper = key.upper()
            if "MAE" in key_upper:
                result["MAE"] = self._parse_metric_value(value)
            elif "RMSE" in key_upper:
                result["RMSE"] = self._parse_metric_value(value)
            elif "MAPE" in key_upper:
                result["MAPE"] = self._parse_metric_value(value)

        return result

    def _parse_metric_value(self, value: Any) -> Optional[float]:
        """解析指标值"""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # 移除百分号和其他字符
            cleaned = re.sub(r"[%\s]", "", value)
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    def _find_test_result(self, model_name: str, dataset: str) -> Optional[TestResult]:
        """查找测试结果"""
        for result in self.test_results:
            if result.model_name == model_name and result.dataset == dataset:
                return result
        return None

    async def run_async(self) -> None:
        """异步运行完整流程（支持并发）"""
        print("=" * 60)
        print("AgentCity Batch Runner (Concurrent Mode)")
        print(f"Concurrency: {self.config.concurrency}")
        print("=" * 60)

        # 加载 catalog
        self.load_catalog()

        if not self.papers:
            print("[WARN] No papers to process")
            return

        total = len(self.papers)
        semaphore = asyncio.Semaphore(self.config.concurrency)

        async def process_with_semaphore(paper: PaperEntry, index: int) -> None:
            async with semaphore:
                await self.process_paper(paper, index, total)

        # 并发处理所有论文
        tasks = [
            process_with_semaphore(paper, i)
            for i, paper in enumerate(self.papers, 1)
        ]
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            print(f"[ERROR] Error during async processing: {e}")

        # 生成报告
        self.generate_excel_report()

        print("\n" + "=" * 60)
        print("Batch Run Complete!")
        print("=" * 60)

    def run(self) -> None:
        """运行完整流程"""
        asyncio.run(self.run_async())


def main():
    parser = argparse.ArgumentParser(
        description="批量运行迁移、调参和测试，生成 Excel 对比报告"
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help=f"Catalog JSON 文件路径 (default: {DEFAULT_CATALOG})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"输出 Excel 文件路径 (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--skip-migration",
        action="store_true",
        help="跳过迁移步骤",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="跳过调参步骤",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="跳过测试步骤（仅生成报告框架）",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="指定要处理的模型（逗号分隔），例如: STGCN,DCRNN",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=20,
        help="测试时的最大训练 epoch 数 (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20000,
        help="每个任务的超时时间（秒）(default: 3600)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="traffic_state_pred",
        help="LibCity 任务类型 (default: traffic_state_pred)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="并发处理的论文数量 (default: 1)",
    )

    args = parser.parse_args()

    # 解析模型列表
    selected_models = None
    if args.models:
        selected_models = [m.strip() for m in args.models.split(",") if m.strip()]

    config = BatchRunnerConfig(
        catalog_path=args.catalog,
        output_path=args.output,
        skip_migration=args.skip_migration,
        skip_tuning=args.skip_tuning,
        skip_test=args.skip_test,
        selected_models=selected_models,
        max_epochs=args.max_epochs,
        timeout_seconds=args.timeout,
        task=args.task,
        concurrency=args.concurrency,
    )

    runner = BatchRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
