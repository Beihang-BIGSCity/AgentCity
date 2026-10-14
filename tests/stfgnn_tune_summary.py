#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize STFGNN_tune_results.csv: rank trials, pick the best, emit markdown."""
import argparse
import csv
import sys
from pathlib import Path

FIELDS_NUM = ['best_valid_score', 'test_MAE', 'test_RMSE', 'test_MAPE',
              'test_masked_MAE', 'test_masked_MAPE', 'wall_sec']
PARAMS = ['learning_rate', 'batch_size', 'first_layer_embedding_size']


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=str(Path(__file__).resolve().parents[1] / 'documentation' / 'STFGNN_tune_results.csv'))
    args = ap.parse_args()
    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    ok = [r for r in rows if r.get('status') == 'OK' and fnum(r.get('best_valid_score')) is not None]
    bad = [r for r in rows if r.get('status') != 'OK']
    for r in ok:
        for k in FIELDS_NUM:
            r[k + '_v'] = fnum(r.get(k))
    ok.sort(key=lambda r: r['best_valid_score_v'])
    print('total rows=%d  OK=%d  FAILED=%d' % (len(rows), len(ok), len(bad)))
    print('total wall (sum of trial wall_sec) = %.1f min' % (sum(fnum(r.get('wall_sec')) or 0 for r in rows) / 60.0))
    print()
    hdr = ['trial', 'lr', 'bs', 'emb', 'best_valid(Huber)', 'MAE', 'RMSE', 'MAPE',
           'masked_MAE', 'masked_MAPE', 'wall_min', 'gpu']
    print(' | '.join(hdr))
    for r in sorted(rows, key=lambda x: int(x['trial_id'])):
        print(' | '.join(str(r.get(k, '')) for k in
                         ['trial_id', 'learning_rate', 'batch_size', 'first_layer_embedding_size',
                          'best_valid_score', 'test_MAE', 'test_RMSE', 'test_MAPE',
                          'test_masked_MAE', 'test_masked_MAPE', 'wall_sec', 'gpu_id']))
    print()
    if ok:
        b = ok[0]
        print('BEST (min best_valid_score = de-norm val Huber loss):')
        print({k: b[k] for k in PARAMS})
        print('  best_valid_score=%s test_MAE=%s test_RMSE=%s test_masked_MAPE=%s'
              % (b['best_valid_score'], b['test_MAE'], b['test_RMSE'], b['test_masked_MAPE']))
    else:
        print('NO successful trials yet.')
    if bad:
        print('\nFAILED trials:')
        for r in bad:
            print('  trial %s %s : %s' % (r['trial_id'],
                  {k: r[k] for k in PARAMS}, r.get('error', '')))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
