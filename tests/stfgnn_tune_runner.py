#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""STFGNN hyperparameter grid sweep for LibCity (crash-safe, incremental CSV).

Why this exists (instead of `run_hyper.py` directly):
  LibCity's run_hyper.py only writes `hyper.result` ONCE at the very end of the
  whole sweep, so a mid-sweep crash loses every trial.  This runner calls the
  EXACT same training path -- libcity.pipeline.objective_function(), the function
  HyperTuning.fn() invokes -- but loops over the grid itself and appends one row
  per finished trial to a CSV using an advisory file lock (fsync'd), giving a
  best-so-far at all times and allowing two GPU workers to share one CSV.

Env workaround: an empty `torch_sparse` module is pre-registered BEFORE importing
libcity (the broken torch_sparse install in this env otherwise aborts all dataset
imports).  See documentation/STFGNN_test_report.md section 4.  Safe for STFGNN.
"""
import argparse
import csv
import fcntl
import gc
import itertools
import json
import os
import shutil
import sys
import time
import traceback
import types
from datetime import datetime
from pathlib import Path

# ---- env workaround: MUST happen before importing libcity ----
sys.modules['torch_sparse'] = types.ModuleType('torch_sparse')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBCITY = str(PROJECT_ROOT / 'Bigscity-LibCity')
os.chdir(LIBCITY)
sys.path.insert(0, LIBCITY)

import torch  # noqa: E402
from libcity.pipeline import objective_function  # noqa: E402
from libcity.utils import get_logger  # noqa: E402

DOC = str(PROJECT_ROOT / 'documentation')
DEFAULT_SPACE = os.path.join(DOC, 'STFGNN_search_space.json')

FIELDS = ['trial_id', 'learning_rate', 'batch_size', 'first_layer_embedding_size',
          'status', 'best_valid_score', 'test_MAE', 'test_RMSE', 'test_MAPE',
          'test_masked_MAE', 'test_masked_MAPE', 'wall_sec', 'max_epoch', 'gpu_id',
          'error']

PARAM_ORDER = ['learning_rate', 'batch_size', 'first_layer_embedding_size']


class _Cfg(dict):
    """Minimal config stand-in for get_logger()."""
    def get(self, k, d=None):
        return dict.get(self, k, d)


def pick_metric(tr, key):
    """Mean of a metric over horizons.

    objective_function returns test_result as a pandas DataFrame whose columns are
    the metric names and whose index is the horizon (1..12).  Some LibCity versions
    return a dict {'<key>@i': val}; both are handled.  Returns None if unavailable.
    """
    try:
        if hasattr(tr, 'columns'):          # pandas DataFrame (columns = metrics)
            return float(tr[key].mean()) if key in tr.columns else None
        if isinstance(tr, dict):            # {'<key>@i': val}
            vals = [v for k, v in tr.items()
                    if k.startswith(key + '@') and isinstance(v, (int, float))]
            return sum(vals) / len(vals) if vals else None
    except Exception:
        return None
    return None


def build_combos(space):
    lists = [space[k] for k in PARAM_ORDER]
    combos = [dict(zip(PARAM_ORDER, vals)) for vals in itertools.product(*lists)]
    # faster batch_size=64 trials first (warm dataset cache, quick early signal)
    combos.sort(key=lambda c: (-int(c['batch_size']), float(c['learning_rate']),
                               int(c['first_layer_embedding_size'])))
    return combos


def key_of(c):
    return (('learning_rate', float(c['learning_rate'])),
            ('batch_size', int(c['batch_size'])),
            ('first_layer_embedding_size', int(c['first_layer_embedding_size'])))


def locked_append(path, row):
    """Append one CSV row under an exclusive advisory lock (multi-writer safe)."""
    with open(path, 'a', newline='') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            if os.path.getsize(path) == 0:
                csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n').writeheader()
            csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n').writerow(row)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            try:
                for row in csv.DictReader(f):
                    if row.get('status') == 'OK':
                        done.add(key_of(row))
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--space', default=DEFAULT_SPACE)
    ap.add_argument('--csv', default=os.path.join(DOC, 'STFGNN_tune_results.csv'))
    ap.add_argument('--max-epoch', type=int, default=None)
    ap.add_argument('--gpu-id', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--only', default=None, help='comma list of trial_ids to run')
    ap.add_argument('--shard', type=int, default=0, help='this shard index (0-based)')
    ap.add_argument('--nshards', type=int, default=1, help='total shards')
    args = ap.parse_args()

    with open(args.space) as f:
        meta = json.load(f)
    space = meta['search_space']
    max_epoch = args.max_epoch if args.max_epoch is not None else int(meta.get('per_trial_max_epoch', 5))
    combos = build_combos(space)
    only = set(int(x) for x in args.only.split(',')) if args.only else None

    # enable LibCity console (stdout) logging
    get_logger(_Cfg(exp_id='stfgnn_runner_g%d' % args.gpu_id, model='STFGNN',
                    dataset='PEMSD7', log_level='INFO'))

    done = load_done(args.csv)
    print('[runner] csv=%s' % args.csv, flush=True)
    print('[runner] space=%s' % space, flush=True)
    print('[runner] combos=%d max_epoch=%d gpu_id=%d seed=%d shard=%d/%d resume_done=%d only=%s'
          % (len(combos), max_epoch, args.gpu_id, args.seed, args.shard, args.nshards,
             len(done), args.only), flush=True)
    print('[runner] start %s' % datetime.now().isoformat(), flush=True)

    t_start = time.time()
    for idx, c in enumerate(combos, start=1):
        if only is not None and idx not in only:
            continue
        if args.nshards > 1 and (idx % args.nshards) != args.shard:
            continue
        if key_of(c) in done:
            print('[runner] skip trial %d (already OK): %s' % (idx, c), flush=True)
            continue
        row = dict(c)
        row.update(trial_id=idx, status='', max_epoch=max_epoch, gpu_id=args.gpu_id, error='')
        print('\n[runner] === trial %d/%d : %s ===' % (idx, len(combos), c), flush=True)
        t0 = time.time()
        res = None
        try:
            res = objective_function(
                task='traffic_state_pred', model_name='STFGNN', dataset_name='PEMSD7',
                config_file=None, saved_model=True, train=True,
                other_args={'max_epoch': max_epoch, 'gpu': True, 'gpu_id': args.gpu_id,
                            'seed': args.seed, 'exp_id': 'stfgnn_tune_t%d' % idx},
                hyper_config_dict=c,
            )
            # record the selection metric FIRST, so it survives any metric-parse hiccup
            row['status'] = 'OK'
            row['best_valid_score'] = res.get('best_valid_score')
            tr = res.get('test_result')
            row['test_MAE'] = pick_metric(tr, 'MAE')
            row['test_RMSE'] = pick_metric(tr, 'RMSE')
            row['test_MAPE'] = pick_metric(tr, 'MAPE')
            row['test_masked_MAE'] = pick_metric(tr, 'masked_MAE')
            row['test_masked_MAPE'] = pick_metric(tr, 'masked_MAPE')
        except Exception as e:  # noqa: BLE001 -- record, never abort the sweep
            row['status'] = 'FAILED'
            row['error'] = repr(e)[:500]
            print('[runner] TRIAL %d FAILED: %s' % (idx, repr(e)), flush=True)
            traceback.print_exc()
        row['wall_sec'] = round(time.time() - t0, 1)
        locked_append(args.csv, row)
        print('[runner] trial %d done status=%s wall=%.1fs best_valid_score=%s test_MAE=%s'
              % (idx, row['status'], row['wall_sec'], row.get('best_valid_score'),
                 row.get('test_MAE')), flush=True)
        del res
        # free disk: drop this trial's checkpoints / tensorboard / eval cache
        shutil.rmtree(os.path.join(LIBCITY, 'libcity/cache/stfgnn_tune_t%d' % idx),
                      ignore_errors=True)
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    print('\n[runner] ALL DONE in %.1fs (%s)' % (time.time() - t_start, datetime.now().isoformat()),
          flush=True)


if __name__ == '__main__':
    main()
