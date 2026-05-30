"""
apex_ml ↔ apex_ai per-session adapter.

This is the single place the existing apex_ai CV pipeline talks to the
apex_ml temporal layer. It's purpose-built for this project:

- Accepts COCO-17 keypoints (the project's pose contract) and converts
  to MediaPipe-33 internally.
- Holds a per-session SequenceBuffer + ExerciseStateMachine +
  FormCorrectionEngine, keyed by (session_id, exercise_id).
- When the exercise changes mid-session (the project supports hint
  retuning), it transparently rebuilds the state machine for the new
  exercise so reps are counted under the right rules.
- Returns a serializable `temporal` dict the pipeline appends to its
  existing FrameResult — additive only; no field is changed or removed.

Mapping from project exercise IDs to apex_ml exercise keys:
    "squat"   → "squat"        "push_up"  → "pushup"
    "pushup"  → "pushup"       "lunge"    → "lunge"
    "plank"   → "plank"        "bicep_curl"  → "bicep_curl"
    "curl"    → "bicep_curl"   (anything else → no temporal output)

The pipeline calls `TemporalSessions.update()` from a sync context (it's
already wrapped in `asyncio.to_thread`). Nothing here uses async I/O so
that's a clean fit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional, Tuple

import numpy as np

from apex_ml.form_correction import FormCorrectionEngine, build_overlay
from apex_ml.temporal_pose import (
    EXERCISE_DEFAULTS, ExerciseStateMachine, SequenceBuffer,
)
from apex_ml.utils.coco_adapter import kp51_to_landmarks33


# Map project exercise IDs (from exercise_profiles.py) to apex_ml keys.
# Unrecognized exercises return None — the temporal block is omitted.
EXERCISE_ID_MAP: Dict[str, str] = {
    "squat":       "squat",
    "push_up":     "pushup",
    "pushup":      "pushup",
    "plank":       "plank",
    "lunge":       "lunge",
    "lunges":      "lunge",
    "bicep_curl":  "bicep_curl",
    "curl":        "bicep_curl",
    "biceps_curl": "bicep_curl",
}


@dataclass
class _Session:
    exercise_key: str
    buf: SequenceBuffer
    fsm: ExerciseStateMachine
    engine: FormCorrectionEngine
    last_quality: Optional[dict] = None
    rep_durations: list = None
    created_at: float = 0.0

    def __post_init__(self):
        if self.rep_durations is None:
            self.rep_durations = []


class TemporalSessions:
    """Thread-safe per-session temporal-pose store.

    All public methods are safe to call from worker threads (FastAPI's
    `asyncio.to_thread` pool). The internal lock is brief (dict mutation
    only); the heavy work runs outside the lock.
    """

    def __init__(self, window_size: int = 60, fps_hint: float = 30.0):
        self._sessions: Dict[str, _Session] = {}
        self._lock = Lock()
        self.window_size = window_size
        self.fps_hint = fps_hint

    # -------------------------------------------------------------- API
    def update(
        self,
        session_id: str,
        exercise_id: Optional[str],
        kp51: np.ndarray,
        t: Optional[float] = None,
    ) -> Optional[dict]:
        """Push one frame; return a serializable temporal block (or None).

        Returns None when:
          - exercise_id is missing or unmapped (e.g. "unknown", "none")
          - kp51 is missing/empty (no person detected)

        Returning None means "skip the temporal field in this frame" —
        the host pipeline must handle that gracefully (FrameResult does).
        """
        if exercise_id is None or exercise_id == "":
            return None
        key = EXERCISE_ID_MAP.get(exercise_id)
        if key is None:
            return None
        if kp51 is None or (hasattr(kp51, "size") and kp51.size == 0):
            return None

        sess = self._get_or_create(session_id, key)

        # If the exercise hint changed mid-session, swap the state machine
        # but keep the buffer so the new analysis warms up immediately.
        if sess.exercise_key != key:
            sess = self._swap_exercise(session_id, sess, key)

        ts = float(t if t is not None else time.time())
        try:
            lm33 = kp51_to_landmarks33(np.asarray(kp51, dtype=np.float64))
        except ValueError:
            return None
        sess.buf.push(lm33, t=ts)
        phase = sess.fsm.update(sess.buf)
        feedback = sess.engine.step(sess.buf)

        # build_overlay is JSON-serializable by construction
        overlay = build_overlay(
            exercise=key,
            phase=phase,
            rep_count=sess.fsm.rep_count,
            feedback=feedback,
            buf=sess.buf,
            path_length=20,
            quality=None,  # quality is attached to specific rep events below
        )
        # Attach the most recent rep quality (set by on_rep_completed) so
        # the frontend can render the badge on the rep just completed.
        if sess.last_quality is not None:
            overlay["last_rep_quality"] = sess.last_quality
            sess.last_quality = None       # one-shot; clear after sending

        # Add the project-specific extras
        overlay["partial_rep_count"] = sess.fsm.partial_count
        return overlay

    def reset(self, session_id: str) -> None:
        """Drop temporal state for a session (called on /vision/reset)."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def end(self, session_id: str) -> Optional[dict]:
        """Return a session summary and drop state."""
        with self._lock:
            sess = self._sessions.pop(session_id, None)
        if sess is None:
            return None
        summary = sess.engine.session_summary(sess.buf, sess.rep_durations)
        summary.update({
            "rep_count": sess.fsm.rep_count,
            "partial_rep_count": sess.fsm.partial_count,
            "duration_seconds": time.time() - sess.created_at,
            "exercise": sess.exercise_key,
        })
        return summary

    # ---------------------------------------------------------- internals
    def _get_or_create(self, session_id: str, exercise_key: str) -> _Session:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                sess = self._build(exercise_key)
                self._sessions[session_id] = sess
            return sess

    def _swap_exercise(self, session_id: str, old: _Session,
                        new_key: str) -> _Session:
        """Replace the state machine + form engine when the exercise changes."""
        new_sess = self._build(new_key, reuse_buf=old.buf)
        with self._lock:
            self._sessions[session_id] = new_sess
        return new_sess

    def _build(self, key: str,
                reuse_buf: Optional[SequenceBuffer] = None) -> _Session:
        buf = reuse_buf or SequenceBuffer(
            window_size=self.window_size, sample_rate_hz=self.fps_hint,
        )
        fsm = ExerciseStateMachine(key)
        primary_joint = EXERCISE_DEFAULTS[key].primary_joint
        engine = FormCorrectionEngine(key, primary_joint=primary_joint,
                                       cooldown_seconds=1.2)
        sess = _Session(
            exercise_key=key, buf=buf, fsm=fsm, engine=engine,
            created_at=time.time(),
        )

        # Wire rep-completion to compute quality once per rep
        def _on_rep(rep, _s=sess):
            q = _s.engine.rep_quality(rep, _s.buf)
            _s.last_quality = q.to_dict()
            _s.rep_durations.append(rep.end_t - rep.start_t)
        fsm.on_rep_completed = _on_rep

        return sess


# Module-level singleton, mirroring the pattern the project uses for
# get_cv_pipeline() / get_rep_counter() / get_pose_extractor().
_INSTANCE: Optional[TemporalSessions] = None
_INSTANCE_LOCK = Lock()


def get_temporal_sessions() -> TemporalSessions:
    """Singleton accessor. Lazy-initialized on first call."""
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = TemporalSessions(window_size=60, fps_hint=30.0)
    return _INSTANCE
