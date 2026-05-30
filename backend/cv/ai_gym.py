"""
backend/cv/ai_gym.py
──────────────────────
Production-grade AI-Gym rep counter, modelled on Ultralytics' `AIGym`
solution but extended in three ways that matter for real users:

  1. **Bilateral angle averaging.** Ultralytics tracks ONE limb (the user
     picks left or right). We track BOTH and average — much more stable
     when the camera is at a slight angle, and lets us also detect L/R
     asymmetry as a form cue.

  2. **Visibility gating.** Frames where any of the 6 tracked keypoints
     are below the profile's `min_visibility` are skipped from the rep
     counter (still rendered, still in the keypoint stream — just not
     counted). This eliminates the spurious "30 reps" you get when the
     user steps out of frame and the joints flicker.

  3. **Real-time form scoring.** Each profile carries a `FormCheck` rule
     set; we evaluate them every frame, build a 0–100 score with EMA
     smoothing, and surface up to 3 actionable cues.

Public API (used by pipeline.py):
    counter = AIGymCounter()
    state   = counter.update(session_id, profile, kp51)
    counter.reset(session_id)
    counter.get(session_id)         # state inspection without mutating

The shape of `state` is:
    {
      "exercise":      "squat",
      "reps":          7,
      "stage":         "up" | "down" | "idle",
      "primary_angle": 167.4,                # current bilateral mean
      "left_angle":    166.1,
      "right_angle":   168.7,
      "hold_seconds":  0.0,                  # > 0 only for hold-mode profiles
      "mode":          "rep" | "hold" | "skip",
      "form_score":    87.2,                 # 0–100, EMA-smoothed
      "form_cues":     ["Keep your back straight", ...],
      "visible":       True,                 # tracked-kpts visibility OK
    }

Frame annotator:
    AIGymCounter.annotate(frame_bgr, kp51, state, draw_skeleton=True) -> frame
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from backend.cv.exercise_profiles import (
    EXERCISE_PROFILES, ExerciseProfile, FormCheck, get_profile,
)


# ─── Geometry helpers ────────────────────────────────────────────────────────

def _kp_xy(kp51: np.ndarray, idx: int) -> Tuple[float, float, float]:
    """Return (x, y, conf) for keypoint `idx` from a flat 51-vector."""
    base = idx * 3
    return float(kp51[base]), float(kp51[base + 1]), float(kp51[base + 2])


def _angle_deg(a: Tuple[float, float],
               b: Tuple[float, float],
               c: Tuple[float, float]) -> float:
    """Angle ABC in degrees. Robust against zero-length segments and NaNs."""
    ax, ay = a; bx, by = b; cx, cy = c
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    n1 = (v1x * v1x + v1y * v1y) ** 0.5
    n2 = (v2x * v2x + v2y * v2y) ** 0.5
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos = (v1x * v2x + v1y * v2y) / (n1 * n2)
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def angle_from_kpts(kp51: np.ndarray, joints: Tuple[int, int, int]
                   ) -> Tuple[float, float]:
    """Returns (angle_degrees, mean_visibility) for the 3 chosen keypoints."""
    a_idx, b_idx, c_idx = joints
    ax, ay, av = _kp_xy(kp51, a_idx)
    bx, by, bv = _kp_xy(kp51, b_idx)
    cx, cy, cv = _kp_xy(kp51, c_idx)
    angle = _angle_deg((ax, ay), (bx, by), (cx, cy))
    vis   = (av + bv + cv) / 3.0
    return angle, vis


# ─── Per-session state ───────────────────────────────────────────────────────

@dataclass
class GymState:
    """One running session — usually one user with their webcam."""
    exercise:      str   = ""
    reps:          int   = 0
    stage:         str   = "idle"          # "idle" | "up" | "down"
    last_angle:    float = 180.0
    angle_window:  Deque[float] = field(default_factory=lambda: deque(maxlen=3))

    # hold mode
    hold_start:    Optional[float] = None
    hold_seconds:  float = 0.0

    # form quality tracking
    form_score_ema: float = 100.0
    form_score_sum: float = 0.0   # for averaging across the whole session
    form_score_n:   int   = 0
    last_cues:      List[str] = field(default_factory=list)

    # rep stats
    rep_depth_history: List[float] = field(default_factory=list)   # bottom angle each rep
    rep_lr_diff_history: List[float] = field(default_factory=list) # |L-R| each rep

    # housekeeping
    created_at:     float = field(default_factory=time.time)
    last_update:    float = field(default_factory=time.time)
    frames_seen:    int   = 0
    frames_visible: int   = 0
    # Track the smoothing window size separately so it can be retuned per
    # profile when the exercise switches.
    _smoothing:     int   = 3

    @property
    def avg_form_score_0_1(self) -> float:
        """Average form score across every counted frame, in [0, 1]."""
        if self.form_score_n == 0:
            return 0.0
        return round((self.form_score_sum / self.form_score_n) / 100.0, 3)


# ─── Counter ─────────────────────────────────────────────────────────────────

class AIGymCounter:
    """One counter instance can serve any number of independent sessions."""

    def __init__(self, ema_alpha: float = 0.4):
        self._sessions: Dict[str, GymState] = {}
        # EMA weight on the current frame's score; higher = more reactive.
        self._ema_alpha = float(ema_alpha)

    # ── session management ─────────────────────────────────────────────────

    def get(self, session_id: str) -> GymState:
        if session_id not in self._sessions:
            self._sessions[session_id] = GymState()
        return self._sessions[session_id]

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def reset_all(self) -> None:
        self._sessions.clear()

    # ── form scoring ───────────────────────────────────────────────────────

    @staticmethod
    def _eval_check(kp51: np.ndarray, fc: FormCheck) -> bool:
        """Return True if the rule fires (form is bad)."""
        a, _ = angle_from_kpts(kp51, fc.joints)
        if   fc.op == "<":  return a < fc.threshold
        elif fc.op == "<=": return a <= fc.threshold
        elif fc.op == ">":  return a > fc.threshold
        elif fc.op == ">=": return a >= fc.threshold
        return False

    def _score_form(
        self, kp51: np.ndarray, profile: ExerciseProfile,
        st: GymState, left: float, right: float,
    ) -> Tuple[float, List[str]]:
        score = 100.0
        cues: List[str] = []

        # 1. Profile-defined geometric rules
        seen_cues: set[str] = set()
        for fc in profile.form_checks:
            if self._eval_check(kp51, fc):
                if fc.cue not in seen_cues:
                    cues.append(fc.cue)
                    seen_cues.add(fc.cue)
                    score -= fc.weight

        # 2. Bilateral asymmetry — penalise > 18° L/R difference
        diff = abs(left - right)
        if diff > 30.0:
            cues.append("Left and right sides are very out of sync — slow down and stay centred.")
            score -= 18
        elif diff > 18.0:
            cues.append("Slight L/R asymmetry — keep both sides moving at the same pace.")
            score -= 8

        # 3. Depth consistency across reps (only meaningful after 3+ reps)
        if len(st.rep_depth_history) >= 3:
            depth_std = float(np.std(st.rep_depth_history[-6:]))
            if depth_std > 22.0:
                cues.append("Rep depth is inconsistent — hit the same range every time.")
                score -= 10

        # 4. EMA smoothing over time so the live score doesn't flicker
        score = max(0.0, min(100.0, score))
        st.form_score_ema = (
            self._ema_alpha * score + (1.0 - self._ema_alpha) * st.form_score_ema
        )

        # Keep at most 3 cues, prefer fresh ones
        return round(st.form_score_ema, 1), cues[:3]

    # ── core update ────────────────────────────────────────────────────────

    def update(
        self,
        session_id: str,
        profile: ExerciseProfile,
        kp51: np.ndarray,
    ) -> dict:
        """Feed one frame's keypoints. Returns the refreshed state dict."""
        st  = self.get(session_id)
        now = time.time()
        st.frames_seen += 1
        st.last_update  = now

        # Reset session counters when the user switches exercise
        if profile.name != st.exercise:
            st.exercise = profile.name
            st.reps = 0
            st.stage = "idle"
            # Resize the angle window to match the profile's preference.
            st._smoothing = max(1, int(profile.smoothing or 3))
            st.angle_window = deque(maxlen=st._smoothing)
            st.rep_depth_history.clear()
            st.rep_lr_diff_history.clear()
            st.hold_start = None
            st.hold_seconds = 0.0
            st.form_score_ema = 100.0
            st.form_score_sum = 0.0
            st.form_score_n = 0

        # Compute bilateral angles + visibility gate
        left,  vis_l = angle_from_kpts(kp51, profile.kpts_left)
        right, vis_r = angle_from_kpts(kp51, profile.kpts_right)
        primary = (left + right) / 2.0
        visible = (vis_l >= profile.min_visibility) and \
                  (vis_r >= profile.min_visibility)

        if not visible:
            return self._snapshot(st, profile, primary, left, right,
                                  mode="skip", visible=False, cues=[
                "Move so your full body is in frame — keypoints are unstable.",
            ])

        st.frames_visible += 1

        # Smooth the angle so a single noisy frame doesn't trigger a phantom rep
        st.angle_window.append(primary)
        smooth_primary = float(np.mean(st.angle_window))

        # ── HOLD MODE ──
        if profile.mode == "hold":
            if st.hold_start is None:
                st.hold_start = now
            st.hold_seconds = now - st.hold_start
            score, cues = self._score_form(kp51, profile, st, left, right)
            st.form_score_sum += score
            st.form_score_n   += 1
            st.last_cues = cues
            return self._snapshot(st, profile, smooth_primary, left, right,
                                  mode="hold", visible=True, cues=cues,
                                  form_score=score)

        # ── REP MODE ──
        # Convention: up_angle > down_angle (extended limb has the larger angle).
        # We accept a stage transition if EITHER the smoothed angle OR the raw
        # current frame crosses the threshold. The smoothed test gives us
        # phantom-rep protection on live video; the raw test keeps the v13
        # 1-frame state machine working for sparse callers (smoke tests, etc.)
        # and means a single deep rep at the bottom registers immediately
        # rather than waiting for the rolling average to catch up.
        is_up_smooth   = smooth_primary >= profile.up_angle
        is_down_smooth = smooth_primary <= profile.down_angle
        is_up_raw      = primary >= profile.up_angle
        is_down_raw    = primary <= profile.down_angle
        is_up   = is_up_smooth   or is_up_raw
        is_down = is_down_smooth or is_down_raw

        if st.stage == "idle":
            if is_up:   st.stage = "up"
            elif is_down: st.stage = "down"          # bottom-start exercises
        elif st.stage == "up" and is_down:
            st.stage = "down"
            st.rep_depth_history.append(smooth_primary)
            st.rep_lr_diff_history.append(abs(left - right))
        elif st.stage == "down" and is_up:
            st.stage = "up"
            st.reps += 1

        score, cues = self._score_form(kp51, profile, st, left, right)
        st.form_score_sum += score
        st.form_score_n   += 1
        st.last_angle = smooth_primary
        st.last_cues  = cues

        return self._snapshot(st, profile, smooth_primary, left, right,
                              mode="rep", visible=True, cues=cues,
                              form_score=score)

    # ── snapshot serialisation ─────────────────────────────────────────────

    @staticmethod
    def _snapshot(
        st: GymState, profile: ExerciseProfile,
        primary: float, left: float, right: float,
        *, mode: str, visible: bool,
        cues: List[str], form_score: Optional[float] = None,
    ) -> dict:
        return {
            "exercise":      profile.name,
            "exercise_name": profile.display_name,
            "reps":          int(st.reps),
            "stage":         st.stage,
            "primary_angle": round(primary, 1),
            "left_angle":    round(left, 1),
            "right_angle":   round(right, 1),
            "hold_seconds":  round(st.hold_seconds, 1),
            "mode":          mode,
            "visible":       bool(visible),
            "form_score":    round(form_score if form_score is not None
                                  else st.form_score_ema, 1),
            "form_cues":     cues,
            "frames_seen":   st.frames_seen,
        }

    # ── frame annotator (server-side overlay, like the AI Gym demo video) ──

    @staticmethod
    def annotate(
        frame_bgr: np.ndarray,
        kp51: Optional[np.ndarray],
        state: dict,
        draw_skeleton: bool = True,
    ) -> np.ndarray:
        """Draw the AI-Gym style overlay onto a frame in-place and return it.

        Used by:
          • scripts/run_webcam.py
          • the optional frame-stream WS path (server returns annotated JPEG)
        Frontend skeleton drawing uses `landmarks` instead — this function is
        only for cases where the server wants to ship a ready-to-display
        rendered frame.
        """
        import cv2

        h, w = frame_bgr.shape[:2]

        if draw_skeleton and kp51 is not None and len(kp51) == 51:
            from backend.cv.yolo_pose import SKELETON_EDGES
            pts = kp51.reshape(17, 3)
            for a, b in SKELETON_EDGES:
                xa, ya, va = pts[a]; xb, yb, vb = pts[b]
                if va < 0.2 or vb < 0.2: continue
                cv2.line(frame_bgr,
                         (int(xa * w), int(ya * h)),
                         (int(xb * w), int(yb * h)),
                         (0, 200, 255), 3, cv2.LINE_AA)
            for x, y, v in pts:
                if v < 0.2: continue
                cv2.circle(frame_bgr, (int(x * w), int(y * h)),
                           5, (0, 255, 120), -1, cv2.LINE_AA)

        # Top banner
        cv2.rectangle(frame_bgr, (0, 0), (w, 80), (0, 0, 0), -1)
        title = f"{state.get('exercise_name', state.get('exercise', '?'))}"
        cv2.putText(frame_bgr, title, (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        if state.get("mode") == "hold":
            sub = f"Hold: {state.get('hold_seconds', 0):.1f}s   Form: {state.get('form_score', 0)}"
        else:
            sub = (f"Reps: {state.get('reps', 0)}   "
                   f"Stage: {state.get('stage', 'idle')}   "
                   f"Form: {state.get('form_score', 0)}   "
                   f"Angle: {state.get('primary_angle', 0)}°")
        cv2.putText(frame_bgr, sub, (16, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2, cv2.LINE_AA)

        # Bottom cues
        cues = state.get("form_cues") or []
        if cues:
            cv2.rectangle(frame_bgr, (0, h - 30 * len(cues) - 10), (w, h),
                          (0, 0, 0), -1)
            for i, c in enumerate(cues):
                cv2.putText(frame_bgr, "• " + c,
                            (16, h - 30 * (len(cues) - i - 1) - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 200, 255),
                            1, cv2.LINE_AA)

        return frame_bgr


# ─── Module-level singleton (one counter for the whole process) ──────────────

_singleton: Optional[AIGymCounter] = None


def get_aigym_counter() -> AIGymCounter:
    global _singleton
    if _singleton is None:
        _singleton = AIGymCounter()
    return _singleton
