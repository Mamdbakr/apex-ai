"""
Phase detection for repetitive exercise motion.

We classify each frame as one of:
    START, CONCENTRIC, ECCENTRIC, LOCKOUT, RESET

For a squat the canonical pattern is:
    START (standing) -> ECCENTRIC (descending) -> LOCKOUT (bottom hold)
    -> CONCENTRIC (rising) -> RESET (top hold) -> ECCENTRIC ...

The detector watches the *primary joint angle* (squat = knee, push-up =
elbow, curl = elbow, etc.) and its smoothed derivative. We use a small
hysteresis band around zero velocity to debounce noise — this is far
more robust than naive thresholding against an absolute angle.

For curls/pulls the angle DECREASES during the concentric phase, so each
exercise carries a `concentric_direction` of +1 (angle grows = concentric,
e.g. push-up) or -1 (angle shrinks = concentric, e.g. bicep curl). The
detector keeps logic identical; only the sign convention changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

from ..temporal_pose.sequence_buffer import SequenceBuffer


class Phase(str, Enum):
    START      = "start"
    CONCENTRIC = "concentric"
    ECCENTRIC  = "eccentric"
    LOCKOUT    = "lockout"
    RESET      = "reset"


@dataclass
class PhaseConfig:
    """Tuneable parameters for the phase detector.

    Attributes
    ----------
    primary_joint : str
        Joint angle that defines the rep (e.g. 'left_knee' for squat).
    concentric_direction : int
        +1 if angle grows during concentric, -1 if it shrinks.
    velocity_eps : float
        Hysteresis band (deg/sec). Below this we consider motion paused.
    hold_frames : int
        Frames a near-zero velocity must persist to count as LOCKOUT/RESET.
    rom_min_deg : float
        Minimum angular range a rep must span to be counted.
    smooth_window : int
        Frames used to smooth the angle derivative.
    """
    primary_joint: str
    concentric_direction: int = -1   # most exercises flex inward = decreasing angle
    velocity_eps: float = 15.0
    hold_frames: int = 3
    rom_min_deg: float = 40.0
    smooth_window: int = 5


@dataclass
class RepRecord:
    """Summary of one completed repetition."""
    index: int
    start_t: float
    end_t: float
    rom_deg: float
    peak_angle: float
    trough_angle: float
    concentric_t: float
    eccentric_t: float
    valid: bool


class PhaseDetector:
    """Stateful classifier from buffer state to current Phase."""

    def __init__(self, cfg: PhaseConfig):
        self.cfg = cfg
        self.current: Phase = Phase.START
        self._hold_count = 0
        self._extremum_angle: Optional[float] = None
        self._last_phase_change_t: Optional[float] = None

    def reset(self) -> None:
        self.current = Phase.START
        self._hold_count = 0
        self._extremum_angle = None
        self._last_phase_change_t = None

    # ------------------------------------------------------------------ step
    def step(self, buf: SequenceBuffer) -> Phase:
        """Update phase given the buffer state; return new Phase.

        Algorithm:
            1. Smooth the joint angle derivative.
            2. Compare its sign (adjusted for concentric_direction) to a
               hysteresis band.
            3. Phase transitions only fire after `hold_frames` of agreement
               to debounce jitter at top/bottom positions.
        """
        if not buf.is_ready(min_frames=self.cfg.smooth_window + 1):
            return self.current

        s = buf.angle_series(self.cfg.primary_joint)
        t = buf.timestamps()
        # Smoothed derivative (deg/sec) via centered differences
        dt = max(float(t[-1] - t[-self.cfg.smooth_window]),
                 1e-3)
        d_angle = float(s[-1] - s[-self.cfg.smooth_window]) / dt
        # Normalize sign so positive = motion in concentric direction
        d_signed = d_angle * self.cfg.concentric_direction

        eps = self.cfg.velocity_eps
        now = float(t[-1])

        if d_signed > eps:
            self._set_phase(Phase.CONCENTRIC, now)
            self._hold_count = 0
        elif d_signed < -eps:
            self._set_phase(Phase.ECCENTRIC, now)
            self._hold_count = 0
        else:
            # Near-zero motion: candidate LOCKOUT (after eccentric) or
            # RESET (after concentric) once it persists.
            self._hold_count += 1
            if self._hold_count >= self.cfg.hold_frames:
                if self.current == Phase.ECCENTRIC:
                    self._set_phase(Phase.LOCKOUT, now)
                elif self.current == Phase.CONCENTRIC:
                    self._set_phase(Phase.RESET, now)

        return self.current

    def _set_phase(self, new_phase: Phase, t: float) -> None:
        if new_phase != self.current:
            self.current = new_phase
            self._last_phase_change_t = t
