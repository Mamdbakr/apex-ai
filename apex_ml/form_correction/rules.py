"""
Exercise-specific biomechanical rules.

Each rule is a small dataclass that:
    1. Inspects the current SequenceBuffer state, and
    2. Emits a human-readable correction string + a severity.

Rules are intentionally simple and inspectable — the deep model (when
trained) is layered on top to catch patterns rules miss. Splitting them
this way means: even without trained weights, the form-correction engine
produces useful output immediately. The model improves recall over time.

Severity scale: 1 (gentle nudge) ... 3 (urgent — stop / reset).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from ..temporal_pose.sequence_buffer import SequenceBuffer
from ..utils.geometry import vector_angle_to_vertical
from ..utils.landmarks import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_WRIST, RIGHT_WRIST, LEFT_ELBOW, RIGHT_ELBOW,
    NOSE,
)


@dataclass
class Feedback:
    """One piece of feedback emitted by a rule."""
    rule: str               # rule id, for deduplication
    message: str            # user-facing string
    severity: int = 1       # 1..3
    body_part: str = ""     # optional, for UI overlay highlighting

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "message": self.message,
            "severity": self.severity,
            "body_part": self.body_part,
        }


RuleFn = Callable[[SequenceBuffer], Optional[Feedback]]


# ----------------------------------------------------------- helpers
def _last_landmarks(buf: SequenceBuffer):
    return buf.last().smoothed if buf.last() else None


# ============================================================ squat
def squat_back_alignment(buf: SequenceBuffer) -> Optional[Feedback]:
    """Spine should not pitch far past vertical at the bottom of a squat."""
    lm = _last_landmarks(buf)
    if lm is None:
        return None
    shoulder_mid = 0.5 * (lm[LEFT_SHOULDER] + lm[RIGHT_SHOULDER])
    hip_mid = 0.5 * (lm[LEFT_HIP] + lm[RIGHT_HIP])
    angle = vector_angle_to_vertical(shoulder_mid, hip_mid)
    if angle > 55:
        return Feedback(
            rule="squat_back_pitch", body_part="spine", severity=2,
            message="Keep your chest up — your back is leaning too far forward.",
        )
    return None


def squat_knee_tracking(buf: SequenceBuffer) -> Optional[Feedback]:
    """Detect knees caving inward past the ankles (valgus)."""
    lm = _last_landmarks(buf)
    if lm is None:
        return None
    # Compare horizontal distance between knees vs ankles
    knee_sep = abs(lm[LEFT_KNEE][0] - lm[RIGHT_KNEE][0])
    ankle_sep = abs(lm[LEFT_ANKLE][0] - lm[RIGHT_ANKLE][0])
    if ankle_sep > 1e-6 and knee_sep < 0.75 * ankle_sep:
        return Feedback(
            rule="squat_knee_valgus", body_part="knees", severity=3,
            message="Knees are collapsing inward — push them out over your toes.",
        )
    return None


def squat_depth(buf: SequenceBuffer) -> Optional[Feedback]:
    """At the bottom of the rep, knee angle should reach parallel (~90°)."""
    if not buf.is_ready(min_frames=10):
        return None
    knee = buf.angle_series("left_knee")
    # Bottom of a squat = minimum knee angle in the recent window
    if float(knee.min()) > 100:
        return Feedback(
            rule="squat_shallow", body_part="hips", severity=2,
            message="Increase squat depth — aim for thighs parallel to the floor.",
        )
    return None


# =========================================================== push-up
def pushup_hip_sag(buf: SequenceBuffer) -> Optional[Feedback]:
    """Hips should stay in line with shoulders & ankles (plank line)."""
    lm = _last_landmarks(buf)
    if lm is None:
        return None
    shoulder = 0.5 * (lm[LEFT_SHOULDER] + lm[RIGHT_SHOULDER])
    hip = 0.5 * (lm[LEFT_HIP] + lm[RIGHT_HIP])
    ankle = 0.5 * (lm[LEFT_ANKLE] + lm[RIGHT_ANKLE])
    # Vertical offset of hip from the shoulder-ankle line, normalized
    line = ankle - shoulder
    line_norm = float(np.linalg.norm(line))
    if line_norm < 1e-3:
        return None
    # Perpendicular distance via 2D cross product (z ignored for image)
    v = hip - shoulder
    cross_z = line[0] * v[1] - line[1] * v[0]
    offset = abs(cross_z) / line_norm
    if offset > 0.08:
        # Determine direction (sag vs pike) by sign of y component
        if hip[1] > 0.5 * (shoulder[1] + ankle[1]):
            return Feedback(
                rule="pushup_hip_sag", body_part="hips", severity=2,
                message="Lift your hips slightly — keep a straight line shoulder to ankle.",
            )
        return Feedback(
            rule="pushup_hip_pike", body_part="hips", severity=2,
            message="Lower your hips — you're piking upward.",
        )
    return None


def pushup_elbow_flare(buf: SequenceBuffer) -> Optional[Feedback]:
    """Elbows should track at ~45° to the torso, not flared to 90°."""
    lm = _last_landmarks(buf)
    if lm is None:
        return None
    # Heuristic: distance from elbow to torso midline > 0.6 * shoulder width
    sh_mid_x = 0.5 * (lm[LEFT_SHOULDER][0] + lm[RIGHT_SHOULDER][0])
    sh_width = abs(lm[LEFT_SHOULDER][0] - lm[RIGHT_SHOULDER][0])
    if sh_width < 1e-3:
        return None
    elbow_offset = max(
        abs(lm[LEFT_ELBOW][0] - sh_mid_x),
        abs(lm[RIGHT_ELBOW][0] - sh_mid_x),
    )
    if elbow_offset > 0.85 * sh_width:
        return Feedback(
            rule="pushup_elbow_flare", body_part="elbows", severity=2,
            message="Tuck your elbows closer to your body.",
        )
    return None


# ============================================================ plank
def plank_alignment(buf: SequenceBuffer) -> Optional[Feedback]:
    """Same alignment rule as the pushup, but emitted as a hold cue."""
    fb = pushup_hip_sag(buf)
    if fb is not None:
        fb.rule = "plank_alignment"
        fb.message = fb.message.replace("push-up", "plank")
        return fb
    return None


# ============================================================ lunge
def lunge_front_knee(buf: SequenceBuffer) -> Optional[Feedback]:
    """Front knee shouldn't travel far past the ankle."""
    lm = _last_landmarks(buf)
    if lm is None:
        return None
    # Whichever knee is more forward (smaller y in image) is the front
    if lm[LEFT_KNEE][1] < lm[RIGHT_KNEE][1]:
        knee, ankle = lm[LEFT_KNEE], lm[LEFT_ANKLE]
    else:
        knee, ankle = lm[RIGHT_KNEE], lm[RIGHT_ANKLE]
    if knee[0] - ankle[0] > 0.05:
        return Feedback(
            rule="lunge_knee_past_toes", body_part="front_knee", severity=2,
            message="Front knee is going past your toes — shift weight back.",
        )
    return None


def lunge_torso_lean(buf: SequenceBuffer) -> Optional[Feedback]:
    """Torso should stay upright during the lunge."""
    lm = _last_landmarks(buf)
    if lm is None:
        return None
    sh = 0.5 * (lm[LEFT_SHOULDER] + lm[RIGHT_SHOULDER])
    hp = 0.5 * (lm[LEFT_HIP] + lm[RIGHT_HIP])
    if vector_angle_to_vertical(sh, hp) > 25:
        return Feedback(
            rule="lunge_torso_lean", body_part="torso", severity=1,
            message="Keep your torso upright.",
        )
    return None


# ======================================================== bicep curl
def curl_elbow_drift(buf: SequenceBuffer) -> Optional[Feedback]:
    """Elbow should stay pinned to the side, not drift forward."""
    lm = _last_landmarks(buf)
    if lm is None:
        return None
    # Elbow x should stay near shoulder x
    drift = max(
        abs(lm[LEFT_ELBOW][0] - lm[LEFT_SHOULDER][0]),
        abs(lm[RIGHT_ELBOW][0] - lm[RIGHT_SHOULDER][0]),
    )
    if drift > 0.08:
        return Feedback(
            rule="curl_elbow_drift", body_part="elbows", severity=2,
            message="Keep your elbows pinned to your sides — don't swing them forward.",
        )
    return None


def curl_body_swing(buf: SequenceBuffer) -> Optional[Feedback]:
    """Torso should remain stable; swinging = momentum cheat."""
    if not buf.is_ready(min_frames=10):
        return None
    coms = np.stack([f.com for f in buf.frames])
    if float(coms[:, 0].std()) > 0.03:
        return Feedback(
            rule="curl_body_swing", body_part="torso", severity=2,
            message="Stop using body momentum — let the biceps do the work.",
        )
    return None


# ====================================================== rule registry
EXERCISE_RULES: Dict[str, List[RuleFn]] = {
    "squat":      [squat_back_alignment, squat_knee_tracking, squat_depth],
    "pushup":     [pushup_hip_sag, pushup_elbow_flare],
    "plank":      [plank_alignment],
    "lunge":      [lunge_front_knee, lunge_torso_lean],
    "bicep_curl": [curl_elbow_drift, curl_body_swing],
}
