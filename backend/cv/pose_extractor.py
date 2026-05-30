"""
backend/cv/pose_extractor.py
──────────────────────────────
Pose backbone with two interchangeable implementations:

  1. **YOLOv8-pose** (Ultralytics) — primary, the same model the "AI Gym"
     reference architecture uses. Faster on GPU, more accurate on hard poses,
     and ships with a tiny default checkpoint that auto-downloads.

  2. **MediaPipe** — fallback, used when ultralytics isn't installed or fails
     to load. Same I/O contract; downstream code (pipeline.py, rep counter,
     classifier) doesn't need to know which one ran.

Public API (UNCHANGED — every existing caller still works):
    extractor = get_pose_extractor()
    kp51, landmarks = extractor.extract(frame_bgr)

  • kp51       : np.ndarray (51,) float32 — 17 (x, y, conf) keypoints,
                 x and y are normalised to [0, 1].
  • landmarks  : list[ {"x", "y", "visibility", ...} ] — used by the frontend
                 skeleton renderer.

The chosen backend is logged at startup and surfaced via `extractor.backend`
so /health can report it.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger


# MediaPipe Pose-Landmarker → COCO-17 mapping (used by the fallback path)
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
COCO_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

DEFAULT_MP_MODEL = str(
    Path(__file__).parent.parent.parent / "ai_models" / "pose_landmarker_full.task"
)


# ─── YOLO backend (preferred) ────────────────────────────────────────────────

class _YOLOBackend:
    backend = "yolov8-pose"

    def __init__(self, model_path: str, device: str, conf_threshold: float):
        from backend.cv.yolo_pose import YOLOPoseExtractor
        self._impl = YOLOPoseExtractor(
            model_path=model_path,
            device=device,
            conf_threshold=conf_threshold,
        )
        self.device = self._impl.device

    def extract(self, frame_bgr: np.ndarray):
        return self._impl.extract(frame_bgr)

    def close(self):
        self._impl.close()


# ─── MediaPipe backend (fallback) ────────────────────────────────────────────

class _MediaPipeBackend:
    backend = "mediapipe"
    device  = "cpu"   # mediapipe-tasks runs on CPU by default

    def __init__(self, model_path: str = DEFAULT_MP_MODEL):
        from mediapipe.tasks.python import vision, BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarkerOptions, RunningMode,
        )
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"MediaPipe pose model not found at {model_path}. "
                "Download it from https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_full/float16/latest/"
                "pose_landmarker_full.task"
            )
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lm = vision.PoseLandmarker.create_from_options(options)
        logger.info("MediaPipe PoseLandmarker initialised (Tasks API)")

    def extract(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[List[dict]]]:
        import cv2
        from mediapipe import Image, ImageFormat

        if frame_bgr is None or frame_bgr.size == 0:
            return None, None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = self._lm.detect(mp_image)
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None, None

        lm_list = result.pose_landmarks[0]
        kp51 = np.zeros(51, dtype=np.float32)
        for coco_i, mp_i in enumerate(MP_TO_COCO):
            if mp_i < len(lm_list):
                lm = lm_list[mp_i]
                kp51[coco_i * 3 + 0] = lm.x
                kp51[coco_i * 3 + 1] = lm.y
                kp51[coco_i * 3 + 2] = (
                    getattr(lm, "visibility", 1.0) or 1.0
                )

        landmarks = [
            {
                "x": float(kp51[i * 3]),
                "y": float(kp51[i * 3 + 1]),
                "visibility": float(kp51[i * 3 + 2]),
                "name": COCO_NAMES[i],
            }
            for i in range(17)
        ]
        return kp51, landmarks

    def close(self):
        try:
            self._lm.close()
        except Exception:
            pass


# ─── PUBLIC FACADE ───────────────────────────────────────────────────────────

class PoseExtractor:
    """Stable public API. Internally delegates to YOLO or MediaPipe.

    `backend` is a read-only string ("yolov8-pose" or "mediapipe") so the
    /health route can tell which engine is actually running.
    """

    def __init__(
        self,
        prefer: str = "yolo",                      # "yolo" | "mediapipe"
        yolo_model_path: str = "yolov8n-pose.pt",
        yolo_device: str = "auto",
        yolo_conf_threshold: float = 0.30,
        mediapipe_model_path: str = DEFAULT_MP_MODEL,
    ):
        self._impl = None
        if prefer.lower() in ("yolo", "yolov8", "ultralytics"):
            try:
                self._impl = _YOLOBackend(
                    model_path=yolo_model_path,
                    device=yolo_device,
                    conf_threshold=yolo_conf_threshold,
                )
                logger.info(f"PoseExtractor backend=yolov8-pose · device={self._impl.device}")
            except Exception as e:
                logger.warning(f"YOLO backend unavailable, falling back to MediaPipe: {e}")

        if self._impl is None:
            try:
                self._impl = _MediaPipeBackend(model_path=mediapipe_model_path)
                logger.info("PoseExtractor backend=mediapipe (fallback)")
            except FileNotFoundError:
                logger.error(
                    "BOTH pose backends failed. YOLO unavailable and MediaPipe model "
                    "not found. Fix: set CV_DEVICE=cpu in .env and restart."
                )
                raise RuntimeError(
                    "No pose backend available. Set CV_DEVICE=cpu in .env and restart the server."
                )
    @property
    def backend(self) -> str:
        return getattr(self._impl, "backend", "unknown")

    @property
    def device(self) -> str:
        return getattr(self._impl, "device", "cpu")

    def extract(self, frame_bgr: np.ndarray):
        return self._impl.extract(frame_bgr)

    def close(self):
        if self._impl is not None:
            self._impl.close()


# ─── Factory ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_pose_extractor() -> PoseExtractor:
    """Singleton — read settings from backend.core.config so the runtime
    can be tuned without touching code."""
    try:
        from backend.core.config import settings
        s = settings()
        return PoseExtractor(
            prefer=getattr(s, "POSE_BACKEND", "yolo"),
            yolo_model_path=getattr(s, "YOLO_MODEL_PATH", "yolov8n-pose.pt"),
            yolo_device=getattr(s, "CV_DEVICE", "auto"),
            yolo_conf_threshold=getattr(s, "YOLO_CONF_THRESHOLD", 0.30),
        )
    except Exception:
        # Allow the module to import in environments without the full settings
        # (e.g. unit tests of pure CV code) — pick a sensible default.
        return PoseExtractor()
