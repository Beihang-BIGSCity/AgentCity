#!/usr/bin/env python3
"""
Test script to measure token usage and runtime for each pipeline stage.
Runs: Literature Retrieval, Model & Data Integration, Standardized Evaluation.
Outputs stats from data/runs/latest_stage_log.json after each stage.
"""

import asyncio
import json
import time
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
LIBCITY_DIR = ROOT_DIR / "Bigscity-LibCity"
STAGE_LOG = ROOT_DIR / "data" / "runs" / "latest_stage_log.json"

sys.path.insert(0, str(ROOT_DIR))

from agents import AppPaths, setup_logger
from agents.core.orchestrator import AgentOrchestrator
from agents.migration import MigrationCatalog, create_migration_stage_callback


def read_stage_log():
    """Read stage log and return list of stage entries."""
    try:
        data = json.loads(STAGE_LOG.read_text(encoding="utf-8"))
        return data.get("stages", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def print_stage_stats(stages, label=""):
    """Print stats for each stage entry."""
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    for i, stage in enumerate(stages):
        stats = stage.get("stats") or {}
        print(f"\n  Stage [{stage.get('key', '?')}] - {stage.get('title', '?')}")
        print(f"    Workflow:       {stage.get('workflow', '?')}")
        print(f"    Duration (s):   {stats.get('duration_seconds', 'N/A')}")
        print(f"    API duration:   {stats.get('duration_api_ms', 'N/A')} ms")
        print(f"    Input tokens:   {stats.get('input_tokens', 'N/A')}")
        print(f"    Output tokens:  {stats.get('output_tokens', 'N/A')}")
        total_tokens = None
        if stats.get('input_tokens') is not None and stats.get('output_tokens') is not None:
            total_tokens = int(stats['input_tokens']) + int(stats['output_tokens'])
        print(f"    Total tokens:   {total_tokens or 'N/A'}")
        print(f"    Cache read:     {stats.get('cache_read_input_tokens', 'N/A')}")
        print(f"    Cache create:   {stats.get('cache_creation_input_tokens', 'N/A')}")
        print(f"    Cost (USD):     {stats.get('total_cost_usd', 'N/A')}")
        print(f"    Turns:          {stats.get('num_turns', 'N/A')}")


async def test_literature():
    """Stage 1: Literature Retrieval"""
    print("\n" + "="*60)
    print("  STAGE 1: Literature Retrieval")
    print("="*60)

    paths = AppPaths()
    paths.ensure()
    logger = setup_logger(paths.log_file)

    orchestrator = AgentOrchestrator(paths=paths, logger=logger)

    # Clear stage log
    STAGE_LOG.write_text(json.dumps({"stages": []}, indent=2), encoding="utf-8")

    start = time.time()
    await orchestrator.run_literature(
        search_terms=["spatio-temporal graph transformer traffic forecasting"],
        year_filter="2025",
    )
    wall_time = time.time() - start

    stages = read_stage_log()
    print_stage_stats(stages, "Literature Retrieval Results")
    print(f"\n    Wall-clock time: {wall_time:.1f}s ({wall_time/60:.1f}m)")
    return stages


async def test_migration():
    """Stage 2: Model & Data Integration"""
    print("\n" + "="*60)
    print("  STAGE 2: Model & Data Integration (Migration)")
    print("="*60)

    paths = AppPaths()
    paths.ensure()
    logger = setup_logger(paths.log_file)

    migration_catalog = MigrationCatalog(
        catalog_path=paths.migration_catalog,
        documentation_dir=paths.documentation_dir,
        root=paths.root,
    )
    stage_callback = create_migration_stage_callback(migration_catalog)
    orchestrator = AgentOrchestrator(
        paths=paths, logger=logger, stage_callback=stage_callback
    )

    # Clear stage log
    STAGE_LOG.write_text(json.dumps({"stages": []}, indent=2), encoding="utf-8")

    # Use a paper from iclr26.json - pick one that is already migrated so the
    # migration agent can still exercise the full workflow but will detect it exists
    # Use STBP which is in the catalog
    paper = {
        "title": "Continual Spatio-Temporal Forecasting",
        "model_name": "STBP",
        "repo_url": "https://github.com/panzijie825/STBP",
        "datasets": ["PEMSD7"],
        "conference": "ICLR",
        "year": 2026,
    }

    start = time.time()
    await orchestrator.run_migration([paper])
    wall_time = time.time() - start

    stages = read_stage_log()
    print_stage_stats(stages, "Migration Results")
    print(f"\n    Wall-clock time: {wall_time:.1f}s ({wall_time/60:.1f}m)")
    return stages


async def test_evaluation():
    """Stage 3: Standardized Evaluation (tuning + testing)"""
    print("\n" + "="*60)
    print("  STAGE 3: Standardized Evaluation (Tuning)")
    print("="*60)

    paths = AppPaths()
    paths.ensure()
    logger = setup_logger(paths.log_file)

    migration_catalog = MigrationCatalog(
        catalog_path=paths.migration_catalog,
        documentation_dir=paths.documentation_dir,
        root=paths.root,
    )
    stage_callback = create_migration_stage_callback(migration_catalog)
    orchestrator = AgentOrchestrator(
        paths=paths, logger=logger, stage_callback=stage_callback
    )

    # Clear stage log
    STAGE_LOG.write_text(json.dumps({"stages": []}, indent=2), encoding="utf-8")

    paper = {
        "title": "Continual Spatio-Temporal Forecasting",
        "model_name": "STBP",
        "repo_url": "https://github.com/panzijie825/STBP",
        "datasets": ["PEMSD7"],
        "conference": "ICLR",
        "year": 2026,
    }

    start = time.time()
    await orchestrator.run_tuning([paper])
    wall_time = time.time() - start

    stages = read_stage_log()
    print_stage_stats(stages, "Evaluation Results")
    print(f"\n    Wall-clock time: {wall_time:.1f}s ({wall_time/60:.1f}m)")
    return stages


async def main():
    all_results = {}

    print("\n" + "#"*60)
    print("#  AgentCity Stage Token & Runtime Test")
    print("#"*60)

    # Stage 1: Literature
    try:
        lit_stages = await test_literature()
        all_results["literature"] = lit_stages
    except Exception as e:
        print(f"\n  [ERROR] Literature test failed: {e}")
        all_results["literature"] = []

    # Stage 2: Migration
    try:
        mig_stages = await test_migration()
        all_results["migration"] = mig_stages
    except Exception as e:
        print(f"\n  [ERROR] Migration test failed: {e}")
        all_results["migration"] = []

    # Stage 3: Evaluation
    try:
        eval_stages = await test_evaluation()
        all_results["evaluation"] = eval_stages
    except Exception as e:
        print(f"\n  [ERROR] Evaluation test failed: {e}")
        all_results["evaluation"] = []

    # Summary table
    print("\n\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    total_tokens = 0
    total_time = 0
    for name, stages in all_results.items():
        for stage in stages:
            stats = stage.get("stats") or {}
            inp = stats.get("input_tokens", 0) or 0
            out = stats.get("output_tokens", 0) or 0
            dur = stats.get("duration_seconds", 0) or 0
            cost = stats.get("total_cost_usd", 0) or 0
            tok = int(inp) + int(out)
            total_tokens += tok
            total_time += dur
            print(f"  {name:20s} | {tok:>10,} tokens | {dur:>8.1f}s | ${cost:.4f}")

    print(f"  {'TOTAL':20s} | {total_tokens:>10,} tokens | {total_time:>8.1f}s")

    # Save results
    results_path = ROOT_DIR / "stage_test_results.json"
    results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
