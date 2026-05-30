"""
backend/cv/pipeline.py
────────────────────────
High-level CV pipeline — the only entry point the rest of the app talks to.

Architecture (matches the YOLOv8 AI Gym reference):

      frame bytes
            │
            ▼
       _decode (cv2.imdecode)
            │
            ▼
   PoseExtractor.extract  ────► (kp51, landmarks)         [YOLOv8-pose primary]
            │
   ┌────────┴────────┐
   ▼                 ▼
 ExerciseClassifier   AIGymCounter (via RepCounter.update_keypoints)
 (auto-detect, EMA    angles → stage machine → reps + form score
  smoothed; only used
  if no exercise_hint)
   │                 │
   └────────┬────────┘
            ▼
        FrameResult

Legacy `FrameResult` shape is preserved. We add ONE extra field, `landmarks`,
because the frontend already reads `data.landmarks` (see Vision.jsx). All
existing routes and tests keep working.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import List, Optional, Union

import numpy as np
from loguru import logger

from backend.cv.exercise_classifier import (
    ExerciseClassifier, build_feature_vector, FRIENDLY_NAMES, get_classifier,
)
from backend.cv.exercise_profiles import get_profile
from backend.cv.pose_extractor import PoseExtractor, get_pose_extractor
from backend.cv.rep_counter import RepCounter, get_rep_counter


@dataclass
class FrameResult:
    """The stable contract the API + WS protocol publishes."""
    detected:       bool
    exercise_id:    str
    exercise_name:  str
    confidence:     float
    top_3:          list
    reps:           int
    phase:          str
    hold_seconds:   float
    form_score:     float                # 0–100
    form_cues:      list
    keypoints:      list                 # flat list of 51 floats (x, y, conf)*17
    landmarks:      list = field(default_factory=list)   # frontend skeleton
    fps:            float = 0.0
    # ── new fields the frontend can opportunistically use; old clients ignore
    stage:          str   = "idle"
    primary_angle:  Optional[float] = None
    left_angle:     Optional[float] = None
    right_angle:    Optional[float] = None
    visible:        bool  = True
    backend:        str   = "yolov8-pose"
    person_detected: bool = False        # alias for legacy frontend keys
    pose_detected:   bool = False        # alias for legacy frontend keys
    rep_count:       int  = 0            # alias for legacy frontend keys
    feedback_cues:   list = field(default_factory=list)  # alias
    # apex_ml temporal layer output (additive — None when not applicable;
    # old frontend clients ignore unknown fields). See
    # apex_ml.integrations.apex_ai_bridge for the contract.
    temporal:        Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Pipeline ────────────────────────────────────────────────────────────────

class CVPipeline:
    """Orchestrates pose extraction + exercise resolution + rep counting."""

    def __init__(
        self,
        pose:        PoseExtractor,
        classifier:  Optional[ExerciseClassifier],
        rep_counter: RepCounter,
    ):
        self.pose        = pose
        self.classifier  = classifier
        self.rep_counter = rep_counter
        self._last_ts:   float = 0.0
        self._fps_window: List[float] = []
        # EMA on classifier probabilities — stabilises the "current exercise"
        # readout when the user holds an ambiguous mid-rep pose.
        self._prob_ema: Optional[np.ndarray] = None
        self._ema_alpha = 0.30

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _decode(frame: Union[np.ndarray, bytes]) -> Optional[np.ndarray]:
        """Accept either a pre-decoded BGR ndarray or raw JPEG bytes."""
        if isinstance(frame, np.ndarray):
            return frame
        try:
            import cv2
            arr = np.frombuffer(frame, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.warning(f"Could not decode frame: {e}")
            return None

    def _fps(self) -> float:
        now = time.time()
        if self._last_ts:
            self._fps_window.append(1.0 / max(now - self._last_ts, 1e-3))
            if len(self._fps_window) > 10:
                self._fps_window.pop(0)
        self._last_ts = now
        if not self._fps_window:
            return 0.0
        return round(float(np.mean(self._fps_window)), 1)

    def _smooth_probs(self, probs: np.ndarray) -> np.ndarray:
        if self._prob_ema is None or self._prob_ema.shape != probs.shape:
            self._prob_ema = probs.copy()
        else:
            self._prob_ema = (
                self._ema_alpha * probs + (1.0 - self._ema_alpha) * self._prob_ema
            )
        return self._prob_ema

    def _classify(self, kp51: np.ndarray) -> tuple[str, float, list]:
        """Returns (exercise_id, confidence, top_3). 'unknown' if classifier
        isn't loaded — pipeline still runs, just without auto-detection."""
        if self.classifier is None or not self.classifier.is_ready:
            return "unknown", 0.0, []
        import torch, torch.nn.functional as F

        feat = build_feature_vector(kp51, scaler=self.classifier.scaler)
        x    = torch.from_numpy(feat).unsqueeze(0).to(self.classifier.device)
        with torch.no_grad():
            logits = self.classifier.net(x)
            probs  = F.softmax(logits, dim=1)[0].cpu().numpy()
        probs = self._smooth_probs(probs)

        top_idx  = int(np.argmax(probs))
        cls_id   = self.classifier.classes[top_idx]
        conf     = float(probs[top_idx])

        top3_idx = np.argsort(probs)[::-1][:3]
        top_3 = [
            {
                "exercise_id":   self.classifier.classes[i],
                "exercise_name": FRIENDLY_NAMES.get(self.classifier.classes[i],
                                                    self.classifier.classes[i]),
                "confidence":    round(float(probs[i]), 4),
            }
            for i in top3_idx
        ]
        return cls_id, conf, top_3

    # ── public ─────────────────────────────────────────────────────────────

    def analyze_frame(
        self,
        frame: Union[np.ndarray, bytes],
        session_id: str = "default",
        exercise_hint: Optional[str] = None,
    ) -> FrameResult:
        """Run the full pipeline on one frame.

        Args:
            frame         : BGR ndarray or JPEG bytes.
            session_id    : key for the per-user rep counter.
            exercise_hint : if the user has chosen an exercise from the UI
                            (squat / push-up / etc.), pass it here. We will
                            skip auto-classification and run the AIGym counter
                            for that profile directly — much more accurate.
        """
        fps = self._fps()
        bgr = self._decode(frame)
        if bgr is None:
            return self._no_person(fps)

        kp51, landmarks = self.pose.extract(bgr)
        if kp51 is None:
            return self._no_person(fps)

        # Resolve the exercise — hint wins, classifier is the fallback.
        exercise_id: str
        confidence:  float
        top_3:       list
        hint_profile = get_profile(exercise_hint) if exercise_hint else None
        if hint_profile is not None:
            exercise_id = hint_profile.name
            confidence  = 1.0
            top_3       = []
        else:
            exercise_id, confidence, top_3 = self._classify(kp51)

        # Run the AI-Gym state machine on the chosen profile
        rep_state = self.rep_counter.update_keypoints(
            session_id, exercise_id, kp51,
        )
        s = rep_state["state"]

        cues = rep_state.get("form_cues") or []

        # ── apex_ml temporal layer (additive; failures are silent) ────────
        # We append a `temporal` dict to FrameResult so the frontend can
        # opportunistically render phase / rep quality / form-correction
        # overlays. Existing clients ignore unknown fields, so this is
        # fully backward-compatible. The block is None when apex_ml has
        # nothing useful to say for this exercise.
        temporal_block = None
        try:
            from apex_ml.integrations.apex_ai_bridge import get_temporal_sessions
            temporal_block = get_temporal_sessions().update(
                session_id=session_id, exercise_id=exercise_id, kp51=kp51,
            )
        except Exception as _e:
            logger.debug(f"apex_ml temporal layer skipped: {_e}")

        return FrameResult(
            detected=True,
            person_detected=True,
            pose_detected=True,
            exercise_id=exercise_id,
            exercise_name=FRIENDLY_NAMES.get(exercise_id, exercise_id),
            confidence=round(confidence, 4),
            top_3=top_3,
            reps=int(s.get("reps", 0)),
            rep_count=int(s.get("reps", 0)),
            phase=s.get("phase", "up"),
            stage=s.get("stage", "idle"),
            hold_seconds=float(s.get("hold_seconds", 0.0) or 0.0),
            form_score=float(rep_state.get("form_score", 0.0)),
            form_cues=cues,
            feedback_cues=cues,
            keypoints=kp51.tolist(),
            landmarks=landmarks or [],
            fps=fps,
            primary_angle=s.get("primary_angle"),
            left_angle=s.get("left_angle"),
            right_angle=s.get("right_angle"),
            visible=bool(s.get("visible", True)),
            backend=self.pose.backend,
            temporal=temporal_block,
        )

    # ── helpers ────────────────────────────────────────────────────────────

    def _no_person(self, fps: float) -> FrameResult:
        return FrameResult(
            detected=False,
            person_detected=False,
            pose_detected=False,
            exercise_id="none", exercise_name="No person detected",
            confidence=0.0, top_3=[], reps=0, rep_count=0,
            phase="up", stage="idle", hold_seconds=0.0,
            form_score=0.0,
            form_cues=["Step into the frame so your full body is visible."],
            feedback_cues=["Step into the frame so your full body is visible."],
            keypoints=[], landmarks=[],
            fps=fps, visible=False,
            backend=self.pose.backend,
        )


# ─── Factory ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_cv_pipeline() -> CVPipeline:
    return CVPipeline(
        pose=get_pose_extractor(),
        classifier=get_classifier(),
        rep_counter=get_rep_counter(),
    )
