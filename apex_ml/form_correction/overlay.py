"""
Visual overlay payload builder.

The frontend already renders MediaPipe landmarks (we are NOT modifying
that). This module ONLY constructs a JSON payload describing what to
draw ON TOP of the existing skeleton — corrections, indicators, the
movement path. The frontend can ignore the payload entirely, render it
in a new layer, or progressively adopt fields.

This decouples ML feedback from rendering: no frontend changes are
required to start using apex_ml, and we never reach inside any existing
component.

Payload schema (versioned for forward compatibility)
----------------------------------------------------
    {
      "version": 1,
      "exercise": "squat",
      "phase": "concentric",
      "rep_count": 7,
      "feedback": [
          {"rule": "...", "message": "...", "severity": 2, "body_part": "knees"}
      ],
      "indicators": {
          "spine_aligned":  true,
          "knees_tracking": false,
          "stable":         true
      },
      "path": [[x0,y0], [x1,y1], ...],     # COM trajectory (last N frames)
      "quality": {                           # only present on rep completion
          "form": 85, "depth": 78, "stability": 90, "tempo": 82, "overall": 84
      }
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..temporal_pose.sequence_buffer import SequenceBuffer
from ..temporal_pose.phase_detector import Phase, RepRecord
from .engine import FormCorrectionEngine, RepQuality
from .rules import Feedback


def build_overlay(exercise: str,
                  phase: Phase,
                  rep_count: int,
                  feedback: List[Feedback],
                  buf: SequenceBuffer,
                  path_length: int = 30,
                  quality: Optional[RepQuality] = None) -> dict:
    """Construct the JSON-serializable overlay payload."""
    # Movement path: COM x/y over the last N frames
    path = [
        [float(f.com[0]), float(f.com[1])]
        for f in list(buf.frames)[-path_length:]
    ]

    # Active rules signal which body parts have warnings right now
    bad_parts = {fb.body_part for fb in feedback if fb.body_part}
    indicators = {
        "spine_aligned":  "spine"   not in bad_parts and "torso" not in bad_parts,
        "knees_tracking": "knees"   not in bad_parts and "front_knee" not in bad_parts,
        "elbows_aligned": "elbows"  not in bad_parts,
        "hips_stable":    "hips"    not in bad_parts,
    }

    payload = {
        "version": 1,
        "exercise": exercise,
        "phase": phase.value if hasattr(phase, "value") else str(phase),
        "rep_count": int(rep_count),
        "feedback": [fb.to_dict() for fb in feedback],
        "indicators": indicators,
        "path": path,
    }
    if quality is not None:
        payload["quality"] = quality.to_dict()
    return payload
