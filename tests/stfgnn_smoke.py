# -*- coding: utf-8 -*-
"""Lightweight CPU smoke check for the migrated STFGNN LibCity model.

Runs: module import -> model construction -> forward / predict / loss / backward,
plus a get_model registration check. No dataset is required.
"""
import sys
from pathlib import Path

LIBCITY_DIR = str(Path(__file__).resolve().parents[1] / 'Bigscity-LibCity')
sys.path.insert(0, LIBCITY_DIR)

import numpy as np
import torch

from libcity.model.traffic_speed_prediction.STFGNN import STFGNN
from libcity.utils.utils import get_model


class DummyScaler(object):
    def inverse_transform(self, x):
        return x


def make_data_feature(num_nodes=20, adj_mx=None):
    if adj_mx is None:
        adj_mx = (np.random.rand(num_nodes, num_nodes) > 0.7).astype(np.float32)
        np.fill_diagonal(adj_mx, 1.0)
    return {
        'num_nodes': num_nodes,
        'feature_dim': 1,
        'output_dim': 1,
        'scaler': DummyScaler(),
        'adj_mx': adj_mx,
    }


def base_config():
    return {
        'device': torch.device('cpu'),
        'input_window': 12,
        'output_window': 12,
        'first_layer_embedding_size': 64,
        'filters': [[64, 64, 64], [64, 64, 64], [64, 64, 64]],
        'act_type': 'GLU',
        'module_type': 'individual',
        'use_mask': True,
        'temporal_emb': True,
        'spatial_emb': True,
        'strides': 4,
        'rho': 1.0,
        'out_layer_dim': 128,
    }


def run():
    N = 20
    B = 4
    data_feature = make_data_feature(N)
    config = base_config()

    model = STFGNN(config, data_feature)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print('[ok] model constructed; parameters = %d' % n_params)
    print('[ok] fusion adj shape =', tuple(model.adj_base.shape))

    x = torch.randn(B, 12, N, 1)
    y = torch.randn(B, 12, N, 1)
    batch = {'X': x, 'y': y}

    out = model(batch)
    print('[ok] forward out shape =', tuple(out.shape))
    assert out.shape == (B, 12, N, 1), out.shape

    pred = model.predict(batch)
    print('[ok] predict out shape =', tuple(pred.shape))
    assert pred.shape == (B, 12, N, 1), pred.shape

    loss_val = model.calculate_loss(batch)
    print('[ok] calculate_loss =', float(loss_val))
    loss_val.backward()
    print('[ok] backward ok')

    # multi-channel input must be cropped to channel 0
    x2 = torch.randn(B, 12, N, 3)
    out2 = model({'X': x2})
    print('[ok] multi-channel input forward out shape =', tuple(out2.shape))
    assert out2.shape == (B, 12, N, 1), out2.shape

    # missing adjacency -> identity fallback (no hard fail)
    df_none = make_data_feature(N)
    df_none['adj_mx'] = None
    m2 = STFGNN(base_config(), df_none)
    print('[ok] adj_mx=None fallback adj shape =', tuple(m2.adj_base.shape))

    # get_model registration resolution
    cfg = base_config()
    cfg['task'] = 'traffic_state_pred'
    cfg['model'] = 'STFGNN'
    m3 = get_model(cfg, data_feature)
    print('[ok] get_model("STFGNN") resolved ->', type(m3).__name__)

    print('SMOKE CHECK PASSED')


if __name__ == '__main__':
    run()
