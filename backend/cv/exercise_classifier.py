"""
backend/cv/exercise_classifier.py
───────────────────────────────────
ExerciseNet — 4-layer MLP on 61 features (51 keypoints + 10 joint angles).

Compatible with the checkpoint shipped at ai_models/dl_models/exercise_classifier.pth.
GPU auto-detected (RTX 3050 works out of the box); falls back to CPU.

Public API:
    classifier = get_classifier()
    result = classifier.predict(kp51)    # {"exercise_id", "exercise_name", "confidence", "top_3"}
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from backend.core.config import settings


# ─── FRIENDLY NAMES / CUES ────────────────────────────────────────────────────

FRIENDLY_NAMES = {
    "barbell_biceps_curl": "Biceps Curl",
    "bench_press":         "Bench Press",
    "deadlift":            "Deadlift",
    "lat_pulldown":        "Lat Pulldown",
    "lateral_raise":       "Lateral Raise",
    "leg_extension":       "Leg Extension",
    "leg_raises":          "Leg Raises",
    "plank":               "Plank",
    "pull_up":             "Pull-Up",
    "push_up":             "Push-Up",
    "romanian_deadlift":   "Romanian Deadlift",
    "shoulder_press":      "Shoulder Press",
    "squat":               "Squat",
    "t_bar_row":           "T-Bar Row",
    "tricep_dips":         "Tricep Dips",
}


# ─── MODEL ────────────────────────────────────────────────────────────────────

def _build_mlp(in_dim: int, num_classes: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.35),
        nn.Linear(512, 512),    nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.30),
        nn.Linear(512, 256),    nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.25),
        nn.Linear(256, 128),    nn.GELU(),           nn.Dropout(0.15),
        nn.Linear(128, num_classes),
    )


# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────

def compute_joint_angles(pts_xy_vis: np.ndarray) -> np.ndarray:
    """10 joint angles from 17 (x, y, vis) keypoints — matches training."""
    def angle(a, b, c):
        try:
            v1 = pts_xy_vis[a, :2] - pts_xy_vis[b, :2]
            v2 = pts_xy_vis[c, :2] - pts_xy_vis[b, :2]
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
            cos = float(np.dot(v1, v2) / denom)
            return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        except Exception:
            return 90.0

    return np.array([
        angle(5, 7, 9),     # left elbow
        angle(6, 8, 10),    # right elbow
        angle(11, 13, 15),  # left knee
        angle(12, 14, 16),  # right knee
        angle(5, 11, 13),   # left hip
        angle(6, 12, 14),   # right hip
        angle(7, 5, 11),    # left shoulder–hip torso
        angle(8, 6, 12),    # right shoulder–hip torso
        angle(0, 5, 11),    # left neck–shoulder–hip
        angle(0, 6, 12),    # right neck–shoulder–hip
    ], dtype=np.float32)


def build_feature_vector(kp51: np.ndarray, scaler=None) -> np.ndarray:
    pts = kp51.reshape(17, 3)
    angles = compute_joint_angles(pts)
    feat = np.hstack([kp51, angles]).astype(np.float32)
    if scaler is not None:
        feat = scaler.transform(feat.reshape(1, -1)).flatten().astype(np.float32)
    return feat


# ─── CLASSIFIER ───────────────────────────────────────────────────────────────

class ExerciseClassifier:
    def __init__(
        self,
        model_path: str,
        config_path: str,
        scaler_path: Optional[str] = None,
        device: str = "auto",
    ):
        import torch, joblib

        # Device selection (RTX 3050 → cuda; cpu fallback)
        if device == "auto" or device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        logger.info(f"ExerciseNet device: {self.device}")

        # Config (class list + input dim)
        cfg_path = Path(config_path)
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            self.classes: list[str] = cfg["classes"]
            self.in_dim: int = cfg["in_dim"]
        else:
            logger.warning(f"CV config missing ({cfg_path}); using defaults")
            self.classes = list(FRIENDLY_NAMES.keys())
            self.in_dim = 61

        # Model
        net = _build_mlp(self.in_dim, len(self.classes))
        state = torch.load(Path(model_path), map_location=self.device)

        # Support both `state_dict` formats (bare MLP or nested under 'net.')
        try:
            net.load_state_dict(state)
        except RuntimeError:
            from collections import OrderedDict
            cleaned = OrderedDict((k.replace("net.", "", 1), v) for k, v in state.items())
            net.load_state_dict(cleaned)

        net.eval().to(self.device)
        self.net = net

        # Scaler (optional — if ExerciseNet was trained with a StandardScaler)
        self.scaler = None
        sp = Path(scaler_path) if scaler_path else None
        if sp and sp.exists():
            self.scaler = joblib.load(sp)
            logger.info(f"Loaded keypoint scaler: {sp.name}")

        logger.info(f"ExerciseNet ready · {len(self.classes)} classes · in_dim={self.in_dim}")

    @property
    def is_ready(self) -> bool:
        return self.net is not None

    def predict(self, kp51: np.ndarray) -> dict:
        """Single-frame inference. Returns exercise + top-3."""
        import torch, torch.nn.functional as F
        feat = build_feature_vector(kp51, scaler=self.scaler)
        x = torch.from_numpy(feat).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.net(x)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        cls_id = self.classes[top_idx]

        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [
            {
                "exercise_id": self.classes[i],
                "exercise_name": FRIENDLY_NAMES.get(self.classes[i], self.classes[i]),
                "confidence": round(float(probs[i]), 4),
            }
            for i in top3_idx
        ]
        return {
            "exercise_id": cls_id,
            "exercise_name": FRIENDLY_NAMES.get(cls_id, cls_id),
            "confidence": round(top_conf, 4),
            "top_3": top3,
        }


# ─── FACTORY ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_classifier() -> Optional[ExerciseClassifier]:
    s = settings()
    try:
        return ExerciseClassifier(
            model_path=s.CV_MODEL_PATH,
            config_path=s.CV_CONFIG_PATH,
            scaler_path=s.CV_SCALER_PATH,
            device=s.CV_DEVICE,
        )
    except Exception as e:
        logger.error(f"Could not load ExerciseNet: {e}")
        return None
