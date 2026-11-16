"""Load only the selected model/executor and its optional dependencies.

Unrelated CUDA/vision/graph packages must not block CPU traffic prediction.
Public exports are preserved; a selected component still raises its real import error.
"""
from importlib import import_module

__all__ = ['HSTWAVE',
 'STHSepNet',
 'ASeer',
 'FreTS',
 'LSTTN',
 'AGCRN',
 'ASTGCN',
 'MSTGCN',
 'ACFM',
 'STResNet',
 'STResNetCommon',
 'ACFMCommon',
 'ASTGCNCommon',
 'MSTGCNCommon',
 'ToGCN',
 'CONVGCN',
 'STDN',
 'STSGCN',
 'STNN',
 'ResLSTM',
 'DGCN',
 'MultiSTGCnet',
 'CRANN',
 'CONVGCNCommon',
 'DSAN',
 'MultiSTGCnetCommon',
 'STGODE',
 'STNorm',
 'ESG',
 'SSTBAN',
 'STTSNet',
 'FOGS',
 'RGSL',
 'DSTAGNN',
 'STPGCN',
 'MultiSPANS',
 'SimST',
 'STSSL',
 'STWave',
 'PDFormer',
 'STGNCDE',
 'ASTGNN',
 'ASTGNNCommon',
 'MetaDG',
 'HiMSNet',
 'DST2former']

_COMPONENTS = {'AGCRN': ('libcity.model.traffic_flow_prediction.AGCRN', 'AGCRN'),
 'ASTGCN': ('libcity.model.traffic_flow_prediction.ASTGCN', 'ASTGCN'),
 'MSTGCN': ('libcity.model.traffic_flow_prediction.MSTGCN', 'MSTGCN'),
 'ACFM': ('libcity.model.traffic_flow_prediction.ACFM', 'ACFM'),
 'STResNet': ('libcity.model.traffic_flow_prediction.STResNet', 'STResNet'),
 'STResNetCommon': ('libcity.model.traffic_flow_prediction.STResNetCommon', 'STResNetCommon'),
 'ACFMCommon': ('libcity.model.traffic_flow_prediction.ACFMCommon', 'ACFMCommon'),
 'ASTGCNCommon': ('libcity.model.traffic_flow_prediction.ASTGCNCommon', 'ASTGCNCommon'),
 'MSTGCNCommon': ('libcity.model.traffic_flow_prediction.MSTGCNCommon', 'MSTGCNCommon'),
 'ToGCN': ('libcity.model.traffic_flow_prediction.ToGCN', 'ToGCN'),
 'CONVGCN': ('libcity.model.traffic_flow_prediction.CONVGCN', 'CONVGCN'),
 'STDN': ('libcity.model.traffic_flow_prediction.STDN', 'STDN'),
 'STSGCN': ('libcity.model.traffic_flow_prediction.STSGCN', 'STSGCN'),
 'STNN': ('libcity.model.traffic_flow_prediction.STNN', 'STNN'),
 'ResLSTM': ('libcity.model.traffic_flow_prediction.ResLSTM', 'ResLSTM'),
 'DGCN': ('libcity.model.traffic_flow_prediction.DGCN', 'DGCN'),
 'MultiSTGCnet': ('libcity.model.traffic_flow_prediction.MultiSTGCnet', 'MultiSTGCnet'),
 'CRANN': ('libcity.model.traffic_flow_prediction.CRANN', 'CRANN'),
 'CONVGCNCommon': ('libcity.model.traffic_flow_prediction.CONVGCNCommon', 'CONVGCNCommon'),
 'DSAN': ('libcity.model.traffic_flow_prediction.DSAN', 'DSAN'),
 'MultiSTGCnetCommon': ('libcity.model.traffic_flow_prediction.MultiSTGCnetCommon',
                        'MultiSTGCnetCommon'),
 'STGODE': ('libcity.model.traffic_flow_prediction.STGODE', 'STGODE'),
 'STNorm': ('libcity.model.traffic_flow_prediction.STNorm', 'STNorm'),
 'ESG': ('libcity.model.traffic_flow_prediction.ESG', 'ESG'),
 'SSTBAN': ('libcity.model.traffic_flow_prediction.SSTBAN', 'SSTBAN'),
 'STTSNet': ('libcity.model.traffic_flow_prediction.STTSNet', 'STTSNet'),
 'FOGS': ('libcity.model.traffic_flow_prediction.FOGS', 'FOGS'),
 'RGSL': ('libcity.model.traffic_flow_prediction.RGSL', 'RGSL'),
 'DSTAGNN': ('libcity.model.traffic_flow_prediction.DSTAGNN', 'DSTAGNN'),
 'STPGCN': ('libcity.model.traffic_flow_prediction.STPGCN', 'STPGCN'),
 'MultiSPANS': ('libcity.model.traffic_flow_prediction.MultiSPANS', 'MultiSPANS'),
 'SimST': ('libcity.model.traffic_flow_prediction.SimST', 'SimST'),
 'STSSL': ('libcity.model.traffic_flow_prediction.STSSL', 'STSSL'),
 'STWave': ('libcity.model.traffic_flow_prediction.STWave', 'STWave'),
 'PDFormer': ('libcity.model.traffic_flow_prediction.PDFormer', 'PDFormer'),
 'STGNCDE': ('libcity.model.traffic_flow_prediction.STGNCDE', 'STGNCDE'),
 'ASTGNN': ('libcity.model.traffic_flow_prediction.ASTGNN', 'ASTGNN'),
 'ASTGNNCommon': ('libcity.model.traffic_flow_prediction.ASTGNNCommon', 'ASTGNNCommon'),
 'FreTS': ('libcity.model.traffic_flow_prediction.FreTS', 'FreTS'),
 'LSTTN': ('libcity.model.traffic_flow_prediction.LSTTN', 'LSTTN'),
 'MetaDG': ('libcity.model.traffic_flow_prediction.MetaDG', 'MetaDG'),
 'DST2former': ('libcity.model.traffic_flow_prediction.DST2former', 'DST2former'),
 'HiMSNet': ('libcity.model.traffic_flow_prediction.HiMSNet', 'HiMSNet'),
 'ASeer': ('libcity.model.traffic_flow_prediction.ASeer', 'ASeer'),
 'STHSepNet': ('libcity.model.traffic_flow_prediction.STHSepNet', 'STHSepNet'),
 'HSTWAVE': ('libcity.model.traffic_flow_prediction.HSTWAVE', 'HSTWAVE')}

def __getattr__(name):
    if name not in _COMPONENTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _COMPONENTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_COMPONENTS))
