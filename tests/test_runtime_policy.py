"""Protocol regression tests; no model API calls or training workloads."""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.core.config import AppPaths, load_task_spec, LIBCITY_DATA_DEFAULTS, LIBCITY_EXECUTOR_DEFAULTS
from agents.core.orchestrator import AgentOrchestrator
from agents.adaptation.tuning import run_custom_grid_search
from agents.tools import impl
from scripts.run_final_evaluation import build_command

ROOT = Path(__file__).resolve().parents[1]


def test_all_presets_align_iteration_and_final_budgets():
    for path in (ROOT / 'tasks').glob('*.json'):
        spec = load_task_spec(path)
        assert spec['train']['max_epoch'] == 35
        assert spec['tuning']['max_trials'] == 5
        assert spec['max_rounds'] == 6 and spec['fixed_rounds']
        assert spec['final_train']['max_epoch'] == LIBCITY_EXECUTOR_DEFAULTS['max_epoch'] == 100
        for key in ('train_rate', 'eval_rate'):
            assert spec['train']['extra_args'][key] == LIBCITY_DATA_DEFAULTS[key]


def test_invalid_split_is_rejected(tmp_path):
    path = tmp_path / 'task.json'
    path.write_text(json.dumps({'task_id': 'x', 'dataset': 'D', 'task': 'traffic_state_pred',
                               'train': {'extra_args': {'train_rate': .8, 'eval_rate': .3}}}))
    with pytest.raises(ValueError, match='partitions'):
        load_task_spec(path)


def test_final_command_uses_100_epochs_over_stale_model_config(tmp_path):
    spec = load_task_spec(ROOT / 'tasks/metr_la_speed_smoke.json')
    frozen = tmp_path / 'frozen.json'
    frozen.write_text(json.dumps({'max_epoch': 1, 'use_early_stop': False, 'hidden_dim': 48}))
    cmd = build_command(spec, 'FrozenModel', str(frozen), tmp_path, False)
    assert cmd[cmd.index('--max_epoch') + 1] == '100'
    assert cmd[cmd.index('--train_rate') + 1] == '0.7'
    assert cmd[cmd.index('--eval_rate') + 1] == '0.1'
    config = json.loads((tmp_path / (cmd[cmd.index('--config_file') + 1] + '.json')).read_text())
    assert config['use_early_stop'] is True
    assert config['hidden_dim'] == 48


def test_grid_runs_five_trials_and_never_selects_failed_trial(tmp_path):
    results = [SimpleNamespace(returncode=1 if i == 0 else 0, stdout='', stderr='') for i in range(5)]
    metrics = [({'masked_mae': i}, {}) for i in range(5)]
    with patch('agents.adaptation.tuning.subprocess.run', side_effect=results) as run, \
         patch('agents.adaptation.tuning.collect_run_metrics', side_effect=metrics):
        result = run_custom_grid_search(repo_dir=tmp_path, model_name='M', dataset_name='D',
                                        search_space={'learning_rate': list(range(7))}, max_trials=5)
    assert run.call_count == 5
    assert len(result['trials']) == 5
    assert result['best_metrics']['masked_mae'] == 1
    assert result['status'] == 'success'


def test_tool_rejects_split_search_and_too_small_grid():
    spec = load_task_spec(ROOT / 'tasks/metr_la_speed_smoke.json')
    with patch.object(impl, '_active_task', return_value=spec):
        result = asyncio.run(impl.tune_adaptation({'model_name': 'M', 'search_space': {'train_rate': [.6, .7]}}))
        assert 'fixed by the task' in result
        result = asyncio.run(impl.tune_adaptation({'model_name': 'M', 'search_space': {'learning_rate': [.01]}}))
        assert 'at least 5' in result


def test_baseline_ignores_stale_agent_epoch_and_split_arguments():
    spec = load_task_spec(ROOT / 'tasks/metr_la_speed_smoke.json')
    with patch.object(impl, '_active_task', return_value=spec), \
         patch.object(impl, '_run_subprocess', new_callable=AsyncMock, return_value=(0, '', '')) as run, \
         patch.object(impl, 'collect_run_metrics', return_value=({'masked_mae': 1}, {})), \
         patch.object(impl, '_write_artifact'), patch.object(impl, '_append_context'), \
         patch.object(impl.result_store, 'record_run'):
        asyncio.run(impl.run_adaptation({'model_name': 'M', 'max_epoch': 1,
                                         'extra_cli_args': {'max_epoch': 2, 'train_rate': .1}}))
    cmd = run.call_args.args[0]
    assert cmd[cmd.index('--max_epoch') + 1] == '35'
    assert cmd[cmd.index('--train_rate') + 1] == '0.7'


@pytest.mark.parametrize('fixed, expected', [(True, 6), (False, 1)])
def test_loop_count_and_environment_restoration(tmp_path, fixed, expected):
    defaults = AppPaths()
    paths = AppPaths(**{name: tmp_path / value.relative_to(defaults.root) for name, value in vars(defaults).items()})
    paths.ensure()
    agent = AgentOrchestrator(paths=paths, logger=logging.getLogger('test'))
    async def check_policy():
        assert os.environ['AGENTCITY_VALIDATION_ONLY'] == '1'
        assert os.environ['AGENTCITY_STRICT_SPLIT'] == '0'
    with patch.dict(os.environ, {'AGENTCITY_VALIDATION_ONLY': '0', 'AGENTCITY_STRICT_SPLIT': '1', 'AGENTCITY_RESUME': '0'}), \
         patch('agents.core.orchestrator.ensure_runner_started', new_callable=AsyncMock), \
         patch('agents.core.orchestrator.shutdown_runner', new_callable=AsyncMock), \
         patch('agents.core.orchestrator.set_active_task'), patch('agents.core.orchestrator.clear_active_task'), \
         patch('agents.core.orchestrator.AgentPipeline.run', side_effect=check_policy) as run, \
         patch.object(agent, '_collect_round', return_value={'verdict': {'passed': True}}), \
         patch.object(agent, '_write_summary', return_value=None):
        result = asyncio.run(agent.run_adaptation({'task_id': 'policy_test', 'max_rounds': 6, 'fixed_rounds': fixed}))
        assert run.await_count == len(result['rounds']) == expected
        assert os.environ['AGENTCITY_VALIDATION_ONLY'] == '0'
        assert os.environ['AGENTCITY_STRICT_SPLIT'] == '1'


def test_tuned_cli_parameter_is_not_overwritten_by_base_budget(tmp_path):
    from agents.adaptation.tuning import trial_command
    cmd, _ = trial_command(repo_dir=tmp_path, model_name='M', dataset_name='D',
                           task='traffic_state_pred', params={'batch_size': 64},
                           extra_cli_args={'batch_size': 32, 'max_epoch': 35})
    assert cmd.count('--batch_size') == 1
    assert cmd[cmd.index('--batch_size') + 1] == '64'
