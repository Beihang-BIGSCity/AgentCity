"""Harness for AGCRN hyperparameter tuning.

Stubs the ABI-broken `torch_sparse` (only used by DeepMapMatchingDataset, which is
eagerly imported by `libcity/data/dataset/__init__.py`) and then executes the stock
`run_hyper.py` as __main__ with the CLI args passed through unchanged.

No repository code is modified.
"""
import os
import sys
import types
import runpy

sys.path.insert(0, os.getcwd())

# --- torch_sparse stub (must be installed before libcity.data.dataset is imported) ---
_stub = types.ModuleType('torch_sparse')


class SparseTensor:  # never instantiated for traffic_state_pred
    def __init__(self, *a, **k):
        raise RuntimeError('torch_sparse is stubbed in this tuning harness')


_stub.SparseTensor = SparseTensor
_stub.__version__ = '0.0.0-stub'
sys.modules['torch_sparse'] = _stub
# -----------------------------------------------------------------------------------

sys.argv = ['run_hyper.py'] + sys.argv[1:]
runpy.run_path('run_hyper.py', run_name='__main__')
