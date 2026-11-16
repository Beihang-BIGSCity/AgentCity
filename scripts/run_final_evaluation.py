"""Retrain a frozen composition with LibCity's default budget, then evaluate test data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.core.config import load_task_spec, LIBCITY_EXECUTOR_DEFAULTS
from agents.adaptation.tuning import trial_command


def build_command(spec, model, config_file, repo, gpu):
    extra = dict(spec['train']['extra_args'])
    extra.update(spec['final_train'])
    extra['patience'] = LIBCITY_EXECUTOR_DEFAULTS['patience']
    extra['batch_size'] = spec['train']['batch_size']
    command, _ = trial_command(
        repo_dir=repo, model_name=model, dataset_name=spec['dataset'],
        task=spec['task'], params={}, config_file=config_file,
        extra_cli_args=extra, gpu=gpu,
    )
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--model', required=True, help='Frozen model selected on validation results')
    parser.add_argument('--config-file', required=True, help='Frozen winning parameter JSON, absolute path')
    parser.add_argument('--gpu', choices=['on', 'off'], default='off')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    spec = load_task_spec(Path(args.task))
    repo = ROOT / 'Bigscity-LibCity'
    if not args.dry_run:
        context_path = ROOT / 'data/adaptation' / spec['task_id'] / 'context.json'
        context = json.loads(context_path.read_text())
        if len(context.get('history', [])) < spec['max_rounds']:
            parser.error('Complete all adaptation loops before evaluating the test partition')
    command = build_command(spec, args.model, str(Path(args.config_file).resolve()), repo, args.gpu == 'on')
    print(json.dumps(command, ensure_ascii=False))
    if args.dry_run:
        return 0
    env = os.environ.copy()
    env.update(AGENTCITY_VALIDATION_ONLY='0', AGENTCITY_STRICT_SPLIT='0')
    return subprocess.run(command, cwd=repo, env=env, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
