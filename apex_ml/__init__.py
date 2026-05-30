"""
apex_ml — Additive ML/AI layer for the apex_ai project.

This package is designed to be *additive only*. It does not import from,
patch, or modify any existing project module. To integrate, host code
imports from `apex_ml` explicitly; nothing here runs as a side effect of
being on the Python path.

Public submodules:
    - temporal_pose      : sequence buffer, phase + state machine, rep counter
    - sequence_models    : LSTM / TCN / Transformer encoders for pose sequences
    - motion_analysis    : velocity, acceleration, smoothness, symmetry, ROM
    - form_correction    : real-time biomechanical feedback engine
    - recommendation     : adaptive workout & coaching recommender
    - api                : optional FastAPI router (mount only if desired)
    - utils              : landmark indices, geometry, filtering

Import only what you need; nothing here forces a heavy dependency at
import time. Torch is lazy-imported inside the sequence_models module.
"""

__version__ = "1.0.0"
__all__ = [
    "temporal_pose",
    "sequence_models",
    "motion_analysis",
    "form_correction",
    "recommendation",
    "api",
    "utils",
]
