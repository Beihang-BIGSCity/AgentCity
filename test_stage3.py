#!/usr/bin/env python3
"""
Test Stage 3 (Standardized Evaluation / Tuning) only.
Collects token usage and runtime stats.
"""

import asyncio
import json
import time
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
STAGE_LOG = ROOT_DIR / "data" / "runs" / "latest_stage_log.json"

sys.path.insert(0, str(ROOT_DIR))

from agents import AppPaths, setup_logger
from agents.core.orchestrator import AgentOrchestrator
from agents.migration import MigrationCatalog, create_migration_stage_callback


def read_stage_log():
    try:
        data = json.loads(STAGE_LOG.read_text(encoding="utf-8"))
        return data.get("stages", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def print_stage_stats(stages):
    for stage in stages:
        stats = stage.get("stats") or {}
        print(f"\n  Stage [{stage.get('key', '?')}] - {stage.get('title', '?')}")
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


async def main():
    print("=" * 60)
    print("  STAGE 3: Standardized Evaluation (Tuning) - Full Run")
    print("=" * 60)

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
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print_stage_stats(stages)
    print(f"\n    Wall-clock time: {wall_time:.1f}s ({wall_time/60:.1f}m)")

    # Save
    results_path = ROOT_DIR / "stage3_test_results.json"
    results_path.write_text(json.dumps({
        "stages": stages,
        "wall_time_seconds": wall_time,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
