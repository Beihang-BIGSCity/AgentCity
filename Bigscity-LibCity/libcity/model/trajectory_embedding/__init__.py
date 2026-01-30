"""
Trajectory Embedding Models

This module contains models for trajectory representation learning,
including pre-training models like START (BERT-based trajectory embedding).
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
    "LinearNextLoc"
]
