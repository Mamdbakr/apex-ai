"""
backend/cv/yolo_pose.py
─────────────────────────
Ultralytics YOLOv8-pose backbone — the exact stack the "AI Gym" demo uses.

Output contract (matches MediaPipe shim so the rest of the pipeline doesn't care):
    extract(frame_bgr) -> (kp51, landmarks_list)
    where:
      kp51        : np.ndarray of 51 floats — 17 (x, y, conf) keypoints,
                    x and y normalised to [0, 1] in the input frame.
      landmarks   : list[ {"x", "y", "visibility"} ] — frontend uses this
                    directly to draw the skeleton overlay.

The model file `yolov8n-pose.pt` (≈ 6.5 MB) downloads on first use and is
cached by Ultralytics under ~/.config/Ultralytics/ — no manual setup.

If `ultralytics` isn't installed, this module raises ImportError lazily; the
PoseExtractor in pose_extractor.py catches that and falls back to MediaPipe.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger


# COCO-17 layout — same indices used by compute_joint_angles.
COCO_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# Skeleton edges used for the annotated frame
SKELETON_EDGES: List[Tuple[int, int]] = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]


class YOLOPoseExtractor:
    """Ultralytics YOLOv8-pose wrapper. One instance per process is enough."""

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        device: str = "auto",
        conf_threshold: float = 0.30,
        imgsz: int = 640,
    ):
        # Lazy import — keeps `import backend.cv.yolo_pose` cheap and lets
        # MediaPipe-only deployments still boot.
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics>=8.2.0"
            ) from e

        # Resolve device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device
        self.conf_threshold = float(conf_threshold)
        self.imgsz = int(imgsz)

        # Resolve model path. Ultralytics' YOLO() accepts either a bare model
        # name (auto-downloads to its cache) OR an absolute path. We try the
        # absolute path first if it exists, otherwise fall back to the name.
        path = Path(model_path)
        load_arg = str(path) if path.exists() else model_path

        logger.info(f"Loading YOLOv8-pose · model='{load_arg}' device={device}")
        self.model = YOLO(load_arg)
        # Move to device once (rather than every predict call).
        try:
            self.model.to(device)
        except Exception:
            pass  # CPU-only torch installs don't expose .to() on the wrapper
        # Warm up the GPU kernels with a dummy frame so the first real frame
        # isn't penalised with the JIT compile latency.
        try:
            warm = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            _ = self.model.predict(
                warm, verbose=False, conf=0.01,
                imgsz=self.imgsz, device=device,
            )
            logger.info("YOLOv8-pose warm-up complete")
        except Exception as e:
            logger.debug(f"YOLOv8-pose warm-up skipped: {e}")

    # ── inference ───────────────────────────────────────────────────────────

    def extract(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[List[dict]]]:
        """Run pose estimation on a single BGR frame.

        Returns:
            (None, None)    if no person was detected.
            (kp51, lms)     where kp51 is a (51,) float32 array in normalised
                            coords and lms is a 17-element list of
                            {"x","y","visibility"} dicts.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None, None

        h, w = frame_bgr.shape[:2]
        try:
            results = self.model.predict(
                frame_bgr,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                verbose=False,
                device=self.device,
            )
        except Exception as e:
            logger.warning(f"YOLO predict failed: {e}")
            return None, None

        if not results:
            return None, None

        r = results[0]
        # `keypoints` is a Keypoints object; .xy is (N, 17, 2) tensor in pixel
        # coords, .conf is (N, 17). N=number of detected people.
        kpts = getattr(r, "keypoints", None)
        if kpts is None or kpts.xy is None or kpts.xy.shape[0] == 0:
            return None, None

        # Pick the highest-confidence person — most fitness scenes have one.
        try:
            xy   = kpts.xy.cpu().numpy()         # (N, 17, 2) pixels
            conf = kpts.conf.cpu().numpy() if kpts.conf is not None \
                   else np.ones(xy.shape[:2], dtype=np.float32)
        except Exception:
            xy, conf = np.array(kpts.xy), np.ones((1, 17), dtype=np.float32)

        if xy.shape[0] == 0 or xy.shape[1] < 17:
            return None, None

        # Score each detection by mean visible-keypoint confidence and pick the best
        person_scores = conf.mean(axis=1)
        i_best = int(np.argmax(person_scores))
        if person_scores[i_best] < 0.10:
            return None, None

        kp_xy   = xy[i_best]                     # (17, 2) px
        kp_conf = conf[i_best]                   # (17,)

        # Normalise to [0,1] so the downstream code (which assumes MediaPipe
        # convention) keeps working unchanged.
        kp51 = np.zeros(51, dtype=np.float32)
        for i in range(17):
            x_n = float(kp_xy[i, 0]) / max(w, 1)
            y_n = float(kp_xy[i, 1]) / max(h, 1)
            kp51[i * 3 + 0] = x_n
            kp51[i * 3 + 1] = y_n
            kp51[i * 3 + 2] = float(kp_conf[i])

        landmarks = [
            {"x": float(kp51[i * 3]),
             "y": float(kp51[i * 3 + 1]),
             "visibility": float(kp51[i * 3 + 2]),
             "name": COCO_NAMES[i]}
            for i in range(17)
        ]
        return kp51, landmarks

    def close(self) -> None:
        # Ultralytics does not require explicit cleanup; placeholder for parity
        # with PoseExtractor.
        try:
            del self.model
        except Exception:
            pass


# ─── Factory ─────────────────────────────────────────────────────────────────

_yolo_singleton: Optional[YOLOPoseExtractor] = None


def get_yolo_extractor(
    model_path: str = "yolov8n-pose.pt",
    device: str = "auto",
    conf_threshold: float = 0.30,
) -> Optional[YOLOPoseExtractor]:
    """Lazy singleton. Returns None if Ultralytics is unavailable."""
    global _yolo_singleton
    if _yolo_singleton is not None:
        return _yolo_singleton
    try:
        _yolo_singleton = YOLOPoseExtractor(
            model_path=model_path,
            device=device,
            conf_threshold=conf_threshold,
        )
        return _yolo_singleton
    except Exception as e:
        logger.warning(f"YOLOv8-pose unavailable, will fall back: {e}")
        return None
