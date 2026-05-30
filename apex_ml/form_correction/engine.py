"""
Real-time form correction engine.

Combines:
    - Exercise-specific biomechanical rules (form_correction.rules)
    - Generic motion quality features (motion_analysis.features)
    - Optional deep-model signals (sequence_models.InferenceEngine)

into a single streaming API:

    engine = FormCorrectionEngine(exercise="squat")
    feedback_list = engine.step(buf)           # called each frame
    quality       = engine.rep_quality(rep)    # called on rep completion

The engine deduplicates messages — emitting the same correction every
frame would flood the UI — and applies a cooldown so corrections feel
like coaching, not nagging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..motion_analysis import (
    jerk_smoothness, range_of_motion, symmetry_score,
    stability_score, tempo_consistency, momentum_index,
)
from ..temporal_pose.sequence_buffer import SequenceBuffer
from ..temporal_pose.phase_detector import RepRecord
from .rules import EXERCISE_RULES, Feedback


@dataclass
class RepQuality:
    """Composite per-rep score breakdown."""
    form: float        # how clean the rule-based checks came out, 0..100
    depth: float       # how close to target ROM
    stability: float   # COM jitter, 0..100
    tempo: float       # eccentric/concentric balance, 0..100
    overall: float     # weighted aggregate, 0..100

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class FormCorrectionEngine:
    """Stateful streaming feedback for one exercise.

    Parameters
    ----------
    exercise : str
        Key into EXERCISE_RULES.
    cooldown_seconds : float
        Minimum time between two emissions of the same rule.
    target_rom_deg : float, optional
        Override the ROM goal used for depth scoring.
    primary_joint : str
        Joint angle used for depth & tempo calculations.
    """

    def __init__(self, exercise: str,
                 cooldown_seconds: float = 1.5,
                 target_rom_deg: Optional[float] = None,
                 primary_joint: str = "left_knee"):
        if exercise not in EXERCISE_RULES:
            raise KeyError(f"No rules for exercise: {exercise!r}")
        self.exercise = exercise
        self.rules = EXERCISE_RULES[exercise]
        self.cooldown = cooldown_seconds
        self.target_rom = target_rom_deg or 80.0
        self.primary_joint = primary_joint

        self._last_emit: Dict[str, float] = {}
        self._all_emitted: List[Feedback] = []

    # ---------------------------------------------------------------- step
    def step(self, buf: SequenceBuffer) -> List[Feedback]:
        """Run all applicable rules. Return only NEW (not in cooldown) feedback."""
        if not buf.is_ready():
            return []
        now = buf.last().t
        new: List[Feedback] = []
        for rule in self.rules:
            fb = rule(buf)
            if fb is None:
                continue
            last = self._last_emit.get(fb.rule, -1e9)
            if now - last >= self.cooldown:
                self._last_emit[fb.rule] = now
                self._all_emitted.append(fb)
                new.append(fb)
        return new

    # -------------------------------------------------------- rep quality
    def rep_quality(self, rep: RepRecord, buf: SequenceBuffer) -> RepQuality:
        """Composite score for a completed rep, in [0, 100].

        Component scores
        ----------------
        - form: 100 minus a penalty for each rule emitted during the rep.
                Severity-weighted (1pt -> 8, 2pt -> 18, 3pt -> 35).
        - depth: linear ramp on rom / target_rom, clipped to [0, 100].
        - stability: stability_score(buf) * 100.
        - tempo: penalizes very fast and very momentum-driven motion.
        """
        # ---- form ----
        rule_penalty = {1: 8, 2: 18, 3: 35}
        penalty = sum(rule_penalty.get(f.severity, 10)
                      for f in self._all_emitted
                      if rep.start_t <= self._last_emit.get(f.rule, 0) <= rep.end_t)
        form = float(max(0.0, 100.0 - penalty))

        # ---- depth ----
        depth = float(np.clip(rep.rom_deg / max(self.target_rom, 1e-3), 0, 1) * 100)

        # ---- stability ----
        stability = float(stability_score(buf) * 100)

        # ---- tempo ----
        mi = momentum_index(buf, self.primary_joint)
        # Ideal momentum_index is ~1.5–2.5; >4 = whippy, <1.2 = mechanical
        if mi <= 2.5:
            tempo_raw = 1.0
        elif mi >= 6:
            tempo_raw = 0.0
        else:
            tempo_raw = 1.0 - (mi - 2.5) / 3.5
        tempo = float(tempo_raw * 100)

        overall = float(0.4 * form + 0.25 * depth + 0.20 * stability + 0.15 * tempo)
        return RepQuality(form=form, depth=depth, stability=stability,
                          tempo=tempo, overall=overall)

    # ------------------------------------------------------------ session
    def session_summary(self, buf: SequenceBuffer,
                         rep_durations: List[float]) -> dict:
        """Cumulative metrics for the workout session so far."""
        return {
            "tempo_consistency": tempo_consistency(rep_durations),
            "symmetry":          symmetry_score(buf),
            "rom":               range_of_motion(buf, self.primary_joint),
            "stability":         stability_score(buf),
            "smoothness":        jerk_smoothness(buf, landmark_idx=0),
        }

    def reset(self) -> None:
        self._last_emit.clear()
        self._all_emitted.clear()
