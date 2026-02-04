"""
Trajectory Embedding Models

This module contains models for trajectory representation learning,
including pre-training models like START (BERT-based trajectory embedding)
and JCLRNT (Joint Contrastive Learning for Road Network and Trajectory).
"""

from libcity.model.trajectory_embedding.START import (
    START,
    BERT,
    BERTLM,
    BERTContrastive,
    BERTContrastiveLM,
    BERTDownstream,
    LinearETA,
    LinearClassify,
    LinearSim,
    LinearNextLoc
)
from libcity.model.trajectory_embedding.JCLRNT import JCLRNT

__all__ = [
    "START",
    "BERT",
    "BERTLM",
    "BERTContrastive",
    "BERTContrastiveLM",
    "BERTDownstream",
    "LinearETA",
    "LinearClassify",
    "LinearSim",
    "LinearNextLoc",
    "JCLRNT"
]
