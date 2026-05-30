"""
Project-specific bridges between apex_ml and host applications.

Currently provides:
    apex_ai_bridge — wires the temporal pose pipeline into the apex_ai
                     CV pipeline (YOLOv8-pose COCO-17 keypoints).
"""
from .apex_ai_bridge import TemporalSessions, get_temporal_sessions, EXERCISE_ID_MAP

__all__ = ["TemporalSessions", "get_temporal_sessions", "EXERCISE_ID_MAP"]
