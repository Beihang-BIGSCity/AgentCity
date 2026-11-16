"""Load only the selected model/executor and its optional dependencies.

Unrelated CUDA/vision/graph packages must not block CPU traffic prediction.
Public exports are preserved; a selected component still raises its real import error.
"""
from importlib import import_module

__all__ = ['TrajLocPredExecutor',
 'TrafficStateExecutor',
 'DCRNNExecutor',
 'MTGNNExecutor',
 'HyperTuning',
 'GeoSANExecutor',
 'MapMatchingExecutor',
 'DeepMapMatchingExecutor',
 'GEMLExecutor',
 'AbstractTraditionExecutor',
 'ChebConvExecutor',
 'LINEExecutor',
 'ETAExecutor',
 'GensimExecutor',
 'SSTBANExecutor',
 'STTSNetExecutor',
 'FOGSExecutor',
 'TESTAMExecutor',
 'TimeMixerExecutor',
 'STSSLExecutor',
 'MegaCRNExecutor',
 'TrafformerExecutor',
 'PDFormerExecutor',
 'ASTGNNExecutor',
 'Scheduler',
 'CosineLRScheduler',
 'STARTAbstractExecutor',
 'BertBaseExecutor',
 'ContrastiveExecutor',
 'ContrastiveMLMExecutor',
 'ContrastiveSplitMLMExecutor',
 'TrajEmbeddingExecutor',
 'KgContextExecutor']

_COMPONENTS = {'DCRNNExecutor': ('libcity.executor.dcrnn_executor', 'DCRNNExecutor'),
 'FOGSExecutor': ('libcity.executor.fogs_executor', 'FOGSExecutor'),
 'GEMLExecutor': ('libcity.executor.geml_executor', 'GEMLExecutor'),
 'GeoSANExecutor': ('libcity.executor.geosan_executor', 'GeoSANExecutor'),
 'HyperTuning': ('libcity.executor.hyper_tuning', 'HyperTuning'),
 'LINEExecutor': ('libcity.executor.line_executor', 'LINEExecutor'),
 'MapMatchingExecutor': ('libcity.executor.map_matching_executor', 'MapMatchingExecutor'),
 'DeepMapMatchingExecutor': ('libcity.executor.deep_map_matching_executor',
                             'DeepMapMatchingExecutor'),
 'MTGNNExecutor': ('libcity.executor.mtgnn_executor', 'MTGNNExecutor'),
 'STTSNetExecutor': ('libcity.executor.sttsnet_executor', 'STTSNetExecutor'),
 'TrafficStateExecutor': ('libcity.executor.traffic_state_executor', 'TrafficStateExecutor'),
 'TrajLocPredExecutor': ('libcity.executor.traj_loc_pred_executor', 'TrajLocPredExecutor'),
 'AbstractTraditionExecutor': ('libcity.executor.abstract_tradition_executor',
                               'AbstractTraditionExecutor'),
 'ChebConvExecutor': ('libcity.executor.chebconv_executor', 'ChebConvExecutor'),
 'ETAExecutor': ('libcity.executor.eta_executor', 'ETAExecutor'),
 'GensimExecutor': ('libcity.executor.gensim_executor', 'GensimExecutor'),
 'SSTBANExecutor': ('libcity.executor.sstban_executor', 'SSTBANExecutor'),
 'TESTAMExecutor': ('libcity.executor.testam_executor', 'TESTAMExecutor'),
 'TimeMixerExecutor': ('libcity.executor.timemixer_executor', 'TimeMixerExecutor'),
 'STSSLExecutor': ('libcity.executor.STSSL_executor', 'STSSLExecutor'),
 'MegaCRNExecutor': ('libcity.executor.megacrn_executor', 'MegaCRNExecutor'),
 'TrafformerExecutor': ('libcity.executor.trafformer_executor', 'TrafformerExecutor'),
 'PDFormerExecutor': ('libcity.executor.pdformer_executor', 'PDFormerExecutor'),
 'ASTGNNExecutor': ('libcity.executor.astgnn_executor', 'ASTGNNExecutor'),
 'Scheduler': ('libcity.executor.scheduler', 'Scheduler'),
 'CosineLRScheduler': ('libcity.executor.scheduler', 'CosineLRScheduler'),
 'STARTAbstractExecutor': ('libcity.executor.start_abstract_executor', 'STARTAbstractExecutor'),
 'BertBaseExecutor': ('libcity.executor.bert_executor', 'BertBaseExecutor'),
 'ContrastiveExecutor': ('libcity.executor.contrastive_executor', 'ContrastiveExecutor'),
 'ContrastiveMLMExecutor': ('libcity.executor.contra_mlm_executor', 'ContrastiveMLMExecutor'),
 'ContrastiveSplitMLMExecutor': ('libcity.executor.contrastive_split_mlm_executor',
                                 'ContrastiveSplitMLMExecutor'),
 'TrajEmbeddingExecutor': ('libcity.executor.traj_embedding_executor', 'TrajEmbeddingExecutor'),
 'KgContextExecutor': ('libcity.executor.kg_context_executor', 'KgContextExecutor')}

def __getattr__(name):
    if name not in _COMPONENTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _COMPONENTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_COMPONENTS))
