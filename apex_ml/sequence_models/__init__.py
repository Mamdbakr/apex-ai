"""
Deep sequence models for pose data.

This module is torch-dependent. If torch is not installed the imports
will raise on first use; the rest of apex_ml (rule-based feedback, state
machine, recommender) continues to work torch-free.
"""

from .encoders import PoseLSTM, PoseTCN, PoseTransformer, build_model
from .training import (
    PoseSequenceDataset, pad_collate,
    TrainConfig, train_model, InferenceEngine,
)

__all__ = [
    "PoseLSTM", "PoseTCN", "PoseTransformer", "build_model",
    "PoseSequenceDataset", "pad_collate",
    "TrainConfig", "train_model", "InferenceEngine",
]
