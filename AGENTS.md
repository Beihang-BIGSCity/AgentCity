# Repository Guidelines

## Project Structure & Module Organization
- `claude_client.py` orchestrates Claude Agent SDK conversations, the `test_migration` tool, and logging to `claude_code.log`; keep orchestration here and place new helper modules under a small `agents/` or `tools/` package.
- Local clones (`Bigscity-LibCity/`, `DualCast-main/`), datasets, PDFs, and `libcity_document.md` must sit beside this repo yet stay untracked; document any additional required paths and group reusable prompts or adapters under clearly named subdirectories (`configs/traffic/`, `datasets/adapter/`).

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` followed by `python -m pip install -U pip claude-agent-sdk anthropic` prepares an isolated runtime; add LibCity or torch wheels before invoking migrations.
- `python claude_client.py` runs the async workflow from the repo root so relative paths resolve; pass `LOG_LEVEL=DEBUG` when you need verbose tracing of tool calls or subprocess failures.
- Use `python -m pip install -r requirements.txt` once you codify dependencies so the team installs identical versions.

## Coding Style & Naming Conventions
- Target Python 3.10+, four-space indentation, ≤100-character lines, and `snake_case` functions like `test_migration`; reserve `CapWords` for classes and keep constants (paths, dataset names) grouped near the top of the file.
- Annotate coroutine signatures, document external folder expectations, and run `ruff check . && ruff format .` (or `black .`) before committing to keep imports sorted and style uniform.

## Testing Guidelines
- Add a `tests/` package driven by `pytest`; mock `ClaudeSDKClient` and mark coroutine tests with `@pytest.mark.asyncio` to validate prompt construction and tool payloads without contacting remote APIs.
- Stub `subprocess.run` to verify LibCity command strings, cover relative-path helpers, and run `pytest -q` (with lightweight fixtures such as sample benchmark documents) before every pull request.

## Commit & Pull Request Guidelines
- With no history yet, follow Conventional Commit prefixes (`feat:`, `fix:`, `docs:`) written in the imperative present and squash WIP commits before publishing branches.
- Pull requests must state scope, touched paths (`claude_client.py`, configs, dataset adapters), external assets required, verification commands (`python claude_client.py`, `pytest -q`), and include sanitized metrics or log excerpts when relevant; reference related papers or LibCity issues to ground the change.

## Security & Configuration Tips
- Keep API keys in environment variables or ignored `.env` files, scrub secrets from `claude_code.log`, and confirm that large datasets, cloned benchmarks, and proprietary PDFs remain untracked before pushing.
