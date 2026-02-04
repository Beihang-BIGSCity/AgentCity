from libcity.model.traffic_speed_prediction.DCRNN import DCRNN
from libcity.model.traffic_speed_prediction.STGCN import STGCN
from libcity.model.traffic_speed_prediction.GWNET import GWNET
from libcity.model.traffic_speed_prediction.MTGNN import MTGNN
from libcity.model.traffic_speed_prediction.TGCLSTM import TGCLSTM
from libcity.model.traffic_speed_prediction.TGCN import TGCN
from libcity.model.traffic_speed_prediction.RNN import RNN
from libcity.model.traffic_speed_prediction.Seq2Seq import Seq2Seq
from libcity.model.traffic_speed_prediction.AutoEncoder import AutoEncoder
from libcity.model.traffic_speed_prediction.TemplateTSP import TemplateTSP
from libcity.model.traffic_speed_prediction.ATDM import ATDM
from libcity.model.traffic_speed_prediction.GMAN import GMAN
from libcity.model.traffic_speed_prediction.STAGGCN import STAGGCN
from libcity.model.traffic_speed_prediction.GTS import GTS
from libcity.model.traffic_speed_prediction.HGCN import HGCN
#from libcity.model.traffic_speed_prediction.STMGAT import STMGAT
from libcity.model.traffic_speed_prediction.DKFN import DKFN
from libcity.model.traffic_speed_prediction.STTN import STTN
from libcity.model.traffic_speed_prediction.D2STGNN import D2STGNN
from libcity.model.traffic_speed_prediction.FNN import FNN
from libcity.model.traffic_speed_prediction.STID import STID
from libcity.model.traffic_speed_prediction.DMSTGCN import DMSTGCN
from libcity.model.traffic_speed_prediction.HIEST import HIEST
from libcity.model.traffic_speed_prediction.STAEformer import STAEformer
from libcity.model.traffic_speed_prediction.TESTAM import TESTAM
from libcity.model.traffic_speed_prediction.MegaCRN import MegaCRN
from libcity.model.traffic_speed_prediction.Trafformer import Trafformer
from libcity.model.traffic_speed_prediction.LSTGAN import LSTGAN
from libcity.model.traffic_speed_prediction.MLCAFormer import MLCAFormer
from libcity.model.traffic_speed_prediction.RSTIB import RSTIB
from libcity.model.traffic_speed_prediction.GriddedTNP import GriddedTNP
from libcity.model.traffic_speed_prediction.EAC import EAC
from libcity.model.traffic_speed_prediction.SRSNet import SRSNet
from libcity.model.traffic_speed_prediction.STDMAE import STDMAE
from libcity.model.traffic_speed_prediction.AutoSTF import AutoSTF
from libcity.model.traffic_speed_prediction.STSSDL import STSSDL
from libcity.model.traffic_speed_prediction.LightST import LightST
from libcity.model.traffic_speed_prediction.BigST import BigST
from libcity.model.traffic_speed_prediction.STWave import STWave
from libcity.model.traffic_speed_prediction.DCST import DCST
from libcity.model.traffic_speed_prediction.CKGGNN import CKGGNN
from libcity.model.traffic_speed_prediction.EasyST import EasyST
from libcity.model.traffic_speed_prediction.FlashST import FlashST
from libcity.model.eta.MTSTAN import MTSTAN
try:
    from libcity.model.traffic_speed_prediction.DSTMamba import DSTMamba
except ImportError:
    DSTMamba = None  # mamba_ssm not installed
try:
    from libcity.model.traffic_speed_prediction.UrbanDiT import UrbanDiT
except ImportError:
    UrbanDiT = None  # einops not installed
try:
    from libcity.model.traffic_speed_prediction.UniST import UniST
except ImportError:
    UniST = None  # timm not installed
try:
    from libcity.model.traffic_speed_prediction.STLLM import STLLM
except ImportError:
    STLLM = None  # transformers not installed
from libcity.model.traffic_speed_prediction.TGraphormer import TGraphormer
from libcity.model.traffic_speed_prediction.LEAF import LEAF
from libcity.model.traffic_speed_prediction.TRACK import TRACK
from libcity.model.traffic_speed_prediction.TimeMixerPP import TimeMixerPP
from libcity.model.traffic_speed_prediction.TimeMixer import TimeMixer
from libcity.model.traffic_speed_prediction.PatchTST import PatchTST
from libcity.model.traffic_speed_prediction.ConvTimeNet import ConvTimeNet
from libcity.model.traffic_speed_prediction.Pathformer import Pathformer
try:
    from libcity.model.traffic_speed_prediction.Fredformer import Fredformer
except ImportError:
    Fredformer = None  # einops not installed
try:
    from libcity.model.traffic_speed_prediction.PatchSTG import PatchSTG
except ImportError:
    PatchSTG = None  # timm not installed
from libcity.model.traffic_speed_prediction.GNNRF import GNNRF
from libcity.model.traffic_speed_prediction.FCNNBus import FCNNBus
from libcity.model.traffic_speed_prediction.Garner import Garner
from libcity.model.traffic_speed_prediction.MVGRL import MVGRL

__all__ = [
    "DCRNN",
    "STGCN",
    "GWNET",
    "TGCLSTM",
    "TGCN",
    "TemplateTSP",
    "RNN",
    "Seq2Seq",
    "AutoEncoder",
    "MTGNN",
    "ATDM",
    "GMAN",
    "GTS",
    "HGCN",
    "STAGGCN",
    "STMGAT",
    "DKFN",
    "STTN",
    "D2STGNN",
    "FNN",
    "STID",
    "DMSTGCN",
    "HIEST",
    "STAEformer",
    "TESTAM",
    "MegaCRN",
    "Trafformer",
    "LSTGAN",
    "MLCAFormer",
    "RSTIB",
    "GriddedTNP",
    "EAC",
    "SRSNet",
    "STDMAE",
    "AutoSTF",
    "STSSDL",
    "LightST",
    "BigST",
    "STWave",
    "MTSTAN",
    "DSTMamba",
    "UrbanDiT",
    "UniST",
    "STLLM",
    "TGraphormer",
    "DCST",
    "CKGGNN",
    "EasyST",
    "FlashST",
    "LEAF",
    "TRACK",
    "TimeMixerPP",
    "TimeMixer",
    "PatchTST",
    "ConvTimeNet",
    "Fredformer",
    "PatchSTG",
    "Pathformer",
    "GNNRF",
    "FCNNBus",
    "Garner",
    "MVGRL",
]
