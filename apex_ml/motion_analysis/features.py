"""
Motion quality features derived from a SequenceBuffer.

These are model-independent signals — the deep models in sequence_models/
consume the same buffer and learn higher-level structure, but for fast
heuristic feedback we compute interpretable metrics here.

Metrics
-------
- velocity_profile : peak / mean absolute velocity of the working joint
- jerk_smoothness  : normalized integral of jerk; low = smooth, high = jerky
- range_of_motion  : peak-to-peak angle of a chosen joint over the window
- symmetry_score   : left/right joint-angle correlation, in [0, 1]
- tempo_consistency: coefficient of variation between rep durations
- momentum_index   : ratio of peak-velocity to average-velocity at the
                     start of the eccentric phase (high = cheating)
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from ..utils.landmarks import SYMMETRIC_PAIRS
from ..temporal_pose.sequence_buffer import SequenceBuffer


def velocity_profile(buf: SequenceBuffer, landmark_idx: int) -> dict:
    """Speed statistics for a specific landmark."""
    if not buf.is_ready(min_frames=3):
        return {"peak": 0.0, "mean": 0.0, "std": 0.0}
    v = buf.velocity()                  # (T, 33, 3)
    speed = np.linalg.norm(v[:, landmark_idx, :], axis=-1)
    return {
        "peak": float(np.max(speed)),
        "mean": float(np.mean(speed)),
        "std":  float(np.std(speed)),
    }


def jerk_smoothness(buf: SequenceBuffer, landmark_idx: int) -> float:
    """Log-dimensionless jerk (smoother is closer to 0; jerkier is more negative).

    Lower magnitude = smoother. We return a positive "roughness" score
    suitable for thresholding: 0 means perfectly smooth.
    """
    if not buf.is_ready(min_frames=4):
        return 0.0
    a = buf.acceleration()              # (T, 33, 3)
    t = buf.timestamps()
    jerk = np.gradient(a[:, landmark_idx, :], t, axis=0, edge_order=1)
    j_mag = np.linalg.norm(jerk, axis=-1)
    duration = max(float(t[-1] - t[0]), 1e-3)
    # Mean squared jerk integrated over time (no fancy normalization needed
    # for a relative score; the form-correction engine compares against a
    # threshold rather than an absolute scale).
    return float(np.mean(j_mag ** 2) * duration)


def range_of_motion(buf: SequenceBuffer, joint: str) -> float:
    """Peak-to-peak angle range over the buffer, in degrees."""
    if not buf.is_ready(min_frames=2):
        return 0.0
    s = buf.angle_series(joint)
    return float(np.max(s) - np.min(s))


def symmetry_score(buf: SequenceBuffer,
                   pairs: Iterable[tuple] = SYMMETRIC_PAIRS) -> float:
    """Pearson correlation between left/right joint angle time series.

    Returns a score in [0, 1] where 1.0 = perfectly mirrored motion.
    A negative correlation is clamped to 0 (it would indicate the limbs
    are doing opposite motions, which for two-sided exercises is bad).
    """
    if not buf.is_ready(min_frames=5):
        return 1.0
    corrs: List[float] = []
    for left, right in pairs:
        try:
            a = buf.angle_series(left)
            b = buf.angle_series(right)
        except KeyError:
            continue
        if np.std(a) < 1e-6 or np.std(b) < 1e-6:
            corrs.append(1.0)            # both flat = symmetric
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        corrs.append(max(0.0, c))
    return float(np.mean(corrs)) if corrs else 1.0


def tempo_consistency(rep_durations: List[float]) -> float:
    """Return 1 - CV of rep durations, clamped to [0, 1].

    Higher = more consistent tempo. Requires at least 2 reps.
    """
    if len(rep_durations) < 2:
        return 1.0
    arr = np.asarray(rep_durations, dtype=np.float64)
    mean = float(np.mean(arr))
    if mean < 1e-6:
        return 1.0
    cv = float(np.std(arr) / mean)
    return float(max(0.0, 1.0 - cv))


def momentum_index(buf: SequenceBuffer, joint: str) -> float:
    """Detect momentum-driven 'cheating' reps.

    Heuristic: if the rate-of-change of the joint angle spikes far above
    its mean magnitude, the user is likely whipping the weight rather
    than controlling it. Returns peak/mean ratio; 1.0 = perfectly even,
    >>1 = bursty / momentum-driven.
    """
    if not buf.is_ready(min_frames=4):
        return 1.0
    s = buf.angle_series(joint)
    t = buf.timestamps()
    rate = np.abs(np.gradient(s, t, edge_order=1))
    mean = float(np.mean(rate))
    if mean < 1e-6:
        return 1.0
    return float(np.max(rate) / mean)


def stability_score(buf: SequenceBuffer) -> float:
    """COM jitter relative to torso scale; returns score in [0, 1].

    1.0 = rock solid. 0.0 = highly unstable. We compare the COM's
    standard deviation to a reference fraction of torso scale.
    """
    if not buf.is_ready(min_frames=5):
        return 1.0
    coms = np.stack([f.com for f in buf.frames])         # (T, 3)
    jitter = float(np.linalg.norm(coms.std(axis=0)))
    # 5% of normalized torso ~ unstable; 1% ~ excellent
    return float(np.clip(1.0 - jitter / 0.05, 0.0, 1.0))
