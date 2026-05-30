"""
Temporal pose analysis — sequence buffers, phase detection, state machine.

This subpackage upgrades frame-based pose classification to sequence-based
motion understanding. It does *not* replace MediaPipe pose detection;
it consumes the landmarks your existing CV pipeline already produces.

Typical wiring (additive — no existing code is changed):

    from apex_ml.temporal_pose import SequenceBuffer, ExerciseStateMachine

    buf = SequenceBuffer(window_size=60)
    fsm = ExerciseStateMachine("squat")

    # Inside your existing per-frame loop:
    buf.push(landmarks_33x3, t=frame_timestamp_seconds)
    phase = fsm.update(buf)
    reps  = fsm.rep_count
"""

from .sequence_buffer import SequenceBuffer, FrameSample
from .phase_detector import Phase, PhaseConfig, PhaseDetector, RepRecord
from .state_machine import ExerciseStateMachine, EXERCISE_DEFAULTS

__all__ = [
    "SequenceBuffer", "FrameSample",
    "Phase", "PhaseConfig", "PhaseDetector", "RepRecord",
    "ExerciseStateMachine", "EXERCISE_DEFAULTS",
]
