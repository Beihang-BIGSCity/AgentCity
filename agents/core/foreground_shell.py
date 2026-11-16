"""Keep shell completion observable to adaptation agents."""
from __future__ import annotations

import json

from openjiuwen.harness.rails.base import DeepAgentRail


class ForegroundShellRail(DeepAgentRail):
    """Background bash returns a PID but no completion event in this runtime."""

    async def before_tool_call(self, ctx) -> None:
        inputs = ctx.inputs
        if getattr(inputs, 'tool_name', '') != 'bash':
            return
        raw = inputs.tool_args
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        except (TypeError, ValueError):
            return
        if args.get('run_in_background'):
            args['run_in_background'] = False
            inputs.tool_args = json.dumps(args) if isinstance(raw, str) else args
