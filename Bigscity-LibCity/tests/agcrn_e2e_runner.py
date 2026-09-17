"""Throwaway harness: stubs the ABI-broken torch_sparse so libcity.data.dataset
can be imported (SparseTensor is only used inside DeepMapMatchingDataset methods).
No repo code is modified."""
import sys, types
sys.path.insert(0, '.')

_stub = types.ModuleType('torch_sparse')
class SparseTensor:  # never instantiated for traffic_state_pred
    def __init__(self, *a, **k):
        raise RuntimeError('torch_sparse is stubbed in this test harness')
_stub.SparseTensor = SparseTensor
_stub.__version__ = '0.0.0-stub'
sys.modules['torch_sparse'] = _stub

from libcity.pipeline import run_model
run_model(task='traffic_state_pred', model_name='AGCRN', dataset_name='PEMSD4',
          config_file='./tests/agcrn_e2e', saved_model=True, train=True,
          other_args={'seed': 10})
