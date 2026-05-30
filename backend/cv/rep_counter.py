"""
backend/cv/rep_counter.py
──────────────────────────
Public rep-counter facade — preserves the legacy API used by:
  • backend/routes/vision.py    (`get_rep_counter`, `RepCounter`, `RepState`)
  • tests/test_smoke.py         (`RepCounter().update("t1", "squat", angles_10)`)

Internally it now delegates to `AIGymCounter` (an Ultralytics-style angle
state machine — see `backend/cv/ai_gym.py`). When called with the new
keypoint-based path (from pipeline.py), AIGymCounter does the real work.
When called with the legacy `angles_10` path (from old code & tests), this
file translates those angles back into the right pair of profile angles and
runs the same state machine.

That dual-path approach is why no caller has to change.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from backend.cv.ai_gym import AIGymCounter, angle_from_kpts, get_aigym_counter
from backend.cv.exercise_profiles import ExerciseProfile, get_profile


# ─── LEGACY angle layout (kept identical to v13 for back-compat) ─────────────
# The 10 angles produced by `compute_joint_angles(pts_xy_vis)` are:
#   0 left elbow      | 1 right elbow
#   2 left knee       | 3 right knee
#   4 left hip        | 5 right hip
#   6 L torso (sh-hip)| 7 R torso (sh-hip)
#   8 L neck          | 9 R neck
#
# For each profile, we know which two of these legacy slots correspond to
# its tracked left/right angle, so legacy callers (and the smoke tests) keep
# working unchanged.
_LEGACY_ANGLE_INDEX: dict[str, tuple[int, int]] = {
    "squat":               (2, 3),
    "leg_extension":       (2, 3),
    "deadlift":            (4, 5),
    "romanian_deadlift":   (4, 5),
    "leg_raises":          (4, 5),
    "push_up":             (0, 1),
    "bench_press":         (0, 1),
    "shoulder_press":      (0, 1),
    "tricep_dips":         (0, 1),
    "pull_up":             (0, 1),
    "lat_pulldown":        (0, 1),
    "t_bar_row":           (0, 1),
    "barbell_biceps_curl": (0, 1),
    "lateral_raise":       (6, 7),
    "plank":               (4, 5),
}


# ─── LEGACY shape exposed to callers ─────────────────────────────────────────

@dataclass
class RepState:
    """Legacy shape — populated lazily from the underlying GymState."""
    exercise: str = "unknown"
    reps: int = 0
    phase: str = "up"
    last_angle: float = 180.0
    depth_history: List[float] = field(default_factory=list)
    symmetry_history: List[float] = field(default_factory=list)
    smoothness_history: List[float] = field(default_factory=list)
    hold_start: Optional[float] = None
    hold_seconds: float = 0.0
    form_score_sum: float = 0.0
    form_score_n: int = 0
    last_form_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)

    @property
    def avg_form_score(self) -> float:
        if self.form_score_n == 0:
            return 0.0
        return round((self.form_score_sum / self.form_score_n) / 100.0, 3)


# ─── Public counter ──────────────────────────────────────────────────────────

class RepCounter:
    """Stateful counter — one instance per process; sessions are keyed by id.

    The actual state machine lives in AIGymCounter. This class is a thin
    adapter that:
      • lets new code call `update_keypoints(session_id, profile, kp51)` directly
      • lets legacy code call `update(session_id, exercise_id, angles_10)`
        (used by the smoke tests and by anything that still hands us a
        precomputed 10-angle vector instead of full keypoints)
    """

    def __init__(self, ai_gym: Optional[AIGymCounter] = None):
        self._gym: AIGymCounter = ai_gym or get_aigym_counter()

    # ── session bookkeeping ────────────────────────────────────────────────

    def get(self, session_id: str) -> RepState:
        gst = self._gym.get(session_id)
        st  = RepState()
        st.exercise        = gst.exercise or "unknown"
        st.reps            = gst.reps
        st.phase           = "down" if gst.stage == "down" else "up"
        st.last_angle      = gst.last_angle
        st.depth_history   = list(gst.rep_depth_history)
        st.symmetry_history= list(gst.rep_lr_diff_history)
        st.hold_seconds    = gst.hold_seconds
        st.form_score_sum  = gst.form_score_sum
        st.form_score_n    = gst.form_score_n
        st.last_form_score = gst.form_score_ema
        st.created_at      = gst.created_at
        st.last_update     = gst.last_update
        return st

    def reset(self, session_id: str) -> None:
        self._gym.reset(session_id)

    # ── new path: full keypoints ──────────────────────────────────────────

    def update_keypoints(
        self,
        session_id: str,
        exercise_id: str,
        kp51: np.ndarray,
    ) -> dict:
        """Preferred path used by the new pipeline."""
        profile = get_profile(exercise_id)
        if profile is None:
            return self._unknown_snapshot(session_id, exercise_id)
        state = self._gym.update(session_id, profile, kp51)
        return self._wrap_legacy(state)

    # ── legacy path: precomputed 10-angle vector ──────────────────────────
    #
    # Some callers (and the unit tests) feed us a 10-angle vector instead of
    # full keypoints. We synthesise a minimal kp51 with just the 6 keypoints
    # the active profile needs — placed so the angle ABC equals the requested
    # value — then run the AI Gym counter as usual. Result: legacy and modern
    # paths share one state machine.

    def update(
        self,
        session_id: str,
        exercise_id: str,
        angles_10: np.ndarray,
    ) -> dict:
        if exercise_id == "unknown" or exercise_id is None:
            return self._unknown_snapshot(session_id, exercise_id)
        profile = get_profile(exercise_id)
        if profile is None:
            return self._unknown_snapshot(session_id, exercise_id)

        # Translate legacy angle slot → angle value
        slot = _LEGACY_ANGLE_INDEX.get(profile.name)
        if slot is None or len(angles_10) < 10:
            return self._unknown_snapshot(session_id, exercise_id)
        try:
            left_angle  = float(angles_10[slot[0]])
            right_angle = float(angles_10[slot[1]])
        except (TypeError, ValueError):
            return self._unknown_snapshot(session_id, exercise_id)

        kp51 = self._synthesise_kp51(profile, left_angle, right_angle)
        state = self._gym.update(session_id, profile, kp51)
        return self._wrap_legacy(state)

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _synthesise_kp51(
        profile: ExerciseProfile, left_angle: float, right_angle: float
    ) -> np.ndarray:
        """Build the smallest kp51 that makes angle_from_kpts(profile.kpts_*)
        return (left_angle, right_angle). Used only for legacy callers that
        bypass the YOLO frontend.

        We place each triplet (a, b, c) so that:
          • b sits at the origin (0.5, 0.5)
          • a sits at (0.5 - 0.1, 0.5)        ← unit vector along -X
          • c is rotated by (180 - angle)°    ← gives the requested angle
        Confidence is set to 1.0 on every keypoint.
        """
        kp = np.zeros(51, dtype=np.float32)

        def place(joints, angle_deg):
            a_idx, b_idx, c_idx = joints
            # Anchor b at (0.5, 0.5). a is a fixed reference unit vector.
            kp[b_idx * 3:b_idx * 3 + 3] = (0.50, 0.50, 1.0)
            kp[a_idx * 3:a_idx * 3 + 3] = (0.40, 0.50, 1.0)
            # Place c so the angle ABC = angle_deg.
            theta = np.radians(angle_deg)
            cx = 0.5 + 0.10 * np.cos(np.pi - theta)
            cy = 0.5 - 0.10 * np.sin(np.pi - theta)
            kp[c_idx * 3:c_idx * 3 + 3] = (float(cx), float(cy), 1.0)

        place(profile.kpts_left,  left_angle)
        place(profile.kpts_right, right_angle)

        # Fill any unset keypoints with a high-confidence "centred body" so
        # form-check rules don't accidentally fire on uninitialised zeros.
        for i in range(17):
            if kp[i * 3 + 2] < 0.1:
                kp[i * 3 + 0] = 0.5
                kp[i * 3 + 1] = 0.5
                kp[i * 3 + 2] = 1.0
        return kp

    def _wrap_legacy(self, gym_state: dict) -> dict:
        """Map the AIGymCounter dict back into the legacy snapshot shape that
        existing callers (vision.py, tests) expect."""
        gs = gym_state
        return {
            "state": {
                "exercise_id":   gs.get("exercise", "unknown"),
                "exercise_name": gs.get("exercise_name", ""),
                "reps":          gs.get("reps", 0),
                "phase":         "down" if gs.get("stage") == "down" else "up",
                "stage":         gs.get("stage", "idle"),
                "hold_seconds":  gs.get("hold_seconds", 0.0),
                "mode":          gs.get("mode", "rep"),
                "primary_angle": gs.get("primary_angle"),
                "left_angle":    gs.get("left_angle"),
                "right_angle":   gs.get("right_angle"),
                "visible":       gs.get("visible", True),
            },
            "form_score": gs.get("form_score", 0.0),
            "form_cues":  gs.get("form_cues", []),
        }

    def _unknown_snapshot(self, session_id: str, exercise_id: Optional[str]) -> dict:
        gst = self._gym.get(session_id)
        return {
            "state": {
                "exercise_id":   exercise_id or "unknown",
                "exercise_name": "",
                "reps":          gst.reps,
                "phase":         "down" if gst.stage == "down" else "up",
                "stage":         gst.stage,
                "hold_seconds":  gst.hold_seconds,
                "mode":          "none",
                "primary_angle": None,
                "left_angle":    None,
                "right_angle":   None,
                "visible":       False,
            },
            "form_score": 0.0,
            "form_cues":  [],
        }


# ─── MODULE-LEVEL SINGLETON ──────────────────────────────────────────────────

_counter = RepCounter()

def get_rep_counter() -> RepCounter:
    return _counter
