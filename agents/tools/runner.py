from __future__ import annotations

import asyncio
import json
import sys
import traceback

from agents.tools import impl

TOOLS = {
    "catalog_article": impl.catalog_article,
    "search_paper": impl.search_paper,
    "evaluate_paper_relevance": impl.evaluate_paper_relevance,
    "test_migration": impl.test_migration,
    "tune_migration_model": impl.tune_migration_model,
    "web_search": impl.web_search,
    "web_fetch": impl.web_fetch,
}


async def _run(tool_name: str) -> dict:
    func = TOOLS.get(tool_name)
    if func is None:
        return {"error": f"unknown tool: {tool_name}"}
    try:
        raw = sys.stdin.read()
        args = json.loads(raw) if raw.strip() else {}
        if not isinstance(args, dict):
            raise ValueError("tool arguments must be a JSON object")
        text = await func(args)
        return {"text": text}
    except Exception:
        return {"error": traceback.format_exc()}


def main() -> None:
    tool_name = sys.argv[1] if len(sys.argv) > 1 else ""
    result = asyncio.run(_run(tool_name))
    sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
