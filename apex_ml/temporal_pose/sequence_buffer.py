"""
Sequence buffer for temporal pose analysis.

We hold a fixed-length sliding window of recent landmark frames plus
their timestamps, then derive velocity and acceleration on demand. The
buffer is the single source of truth used by the state machine, motion
quality analyzer, and the deep sequence models.

Design notes
------------
- We store landmarks AFTER One-Euro smoothing. Models trained on jitter
  perform worse than models trained on smoothed inputs.
- The buffer is *bounded* (collections.deque(maxlen=...)) so memory is
  O(window_size) regardless of session length.
- Velocity uses centered differences when possible (lower error than
  forward differences) and falls back to forward at the edges.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

from ..utils.filters import OneEuroFilter
from ..utils.landmarks import (
    JOINT_ANGLES, NUM_LANDMARKS,
    LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER,
)
from ..utils.geometry import all_joint_angles, normalize_landmarks, center_of_mass
from ..utils.landmarks import COM_LANDMARKS


@dataclass
class FrameSample:
    """One pose frame with derived features."""
    t: float                          # timestamp in seconds
    raw: np.ndarray                   # (33, 3) raw landmarks
    smoothed: np.ndarray              # (33, 3) one-euro smoothed
    normalized: np.ndarray            # (33, 3) hip-centered, torso-scaled
    angles: dict                      # {joint_name: degrees}
    com: np.ndarray                   # (3,) center of mass


class SequenceBuffer:
    """Sliding window of pose frames with derived motion features.

    Parameters
    ----------
    window_size : int
        Number of frames retained. 30–60 covers ~1–2 seconds at 30 FPS,
        which is enough to span a single rep of most exercises.
    smooth : bool
        Whether to apply One-Euro smoothing on push().
    sample_rate_hz : float
        Hint to the smoother; doesn't need to be exact.
    """

    def __init__(self, window_size: int = 60, smooth: bool = True,
                 sample_rate_hz: float = 30.0):
        if window_size < 4:
            raise ValueError("window_size must be >= 4 for velocity/accel")
        self.window_size = window_size
        self.smooth = smooth
        self.frames: Deque[FrameSample] = deque(maxlen=window_size)
        self._filter = OneEuroFilter(freq=sample_rate_hz) if smooth else None

    # ------------------------------------------------------------------ push
    def push(self, landmarks: np.ndarray, t: float) -> FrameSample:
        """Append a frame and return the resulting FrameSample.

        `landmarks` must be shape (33, 3) in MediaPipe normalized space.
        """
        landmarks = np.asarray(landmarks, dtype=np.float64)
        if landmarks.shape != (NUM_LANDMARKS, 3):
            raise ValueError(
                f"Expected landmarks shape ({NUM_LANDMARKS}, 3); "
                f"got {landmarks.shape}"
            )
        smoothed = self._filter(landmarks) if self._filter is not None else landmarks
        normalized = normalize_landmarks(
            smoothed, LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER
        )
        angles = all_joint_angles(smoothed, JOINT_ANGLES)
        com = center_of_mass(smoothed, COM_LANDMARKS)
        sample = FrameSample(
            t=float(t), raw=landmarks, smoothed=smoothed,
            normalized=normalized, angles=angles, com=com,
        )
        self.frames.append(sample)
        return sample

    def reset(self) -> None:
        self.frames.clear()
        if self._filter is not None:
            self._filter.reset()

    # -------------------------------------------------------------- queries
    def __len__(self) -> int:
        return len(self.frames)

    def is_ready(self, min_frames: int = 4) -> bool:
        return len(self.frames) >= min_frames

    def last(self) -> Optional[FrameSample]:
        return self.frames[-1] if self.frames else None

    def angle_series(self, joint: str) -> np.ndarray:
        """Return the time series of a given joint angle across the buffer."""
        return np.array([f.angles[joint] for f in self.frames])

    def timestamps(self) -> np.ndarray:
        return np.array([f.t for f in self.frames])

    # --------------------------------------------------------- motion feats
    def landmark_tensor(self, normalized: bool = True) -> np.ndarray:
        """Stack landmarks across the buffer into shape (T, 33, 3)."""
        attr = "normalized" if normalized else "smoothed"
        return np.stack([getattr(f, attr) for f in self.frames])

    def velocity(self, normalized: bool = True) -> np.ndarray:
        """Per-landmark velocity, shape (T, 33, 3).

        Uses centered differences internally; endpoints use forward/back.
        Units: normalized-space per second.
        """
        x = self.landmark_tensor(normalized=normalized)
        t = self.timestamps()
        if len(t) < 2:
            return np.zeros_like(x)
        # np.gradient supports an axis-aware variable-step derivative
        return np.gradient(x, t, axis=0, edge_order=1)

    def acceleration(self, normalized: bool = True) -> np.ndarray:
        """Per-landmark acceleration, shape (T, 33, 3)."""
        v = self.velocity(normalized=normalized)
        t = self.timestamps()
        if len(t) < 3:
            return np.zeros_like(v)
        return np.gradient(v, t, axis=0, edge_order=1)

    def feature_tensor(self) -> np.ndarray:
        """Build the canonical model input.

        Concatenates per-frame:
            - normalized landmarks      (33 * 3 = 99)
            - joint angles              (10)
            - per-landmark speed scalar (33)

        Shape: (T, 142). This is the format consumed by the LSTM / TCN /
        Transformer encoders in sequence_models/.
        """
        if not self.frames:
            return np.zeros((0, 142), dtype=np.float32)
        lm = self.landmark_tensor(normalized=True).reshape(len(self), -1)  # (T, 99)
        ang = np.array([[f.angles[k] for k in JOINT_ANGLES] for f in self.frames])  # (T, 10)
        v = self.velocity(normalized=True)                                # (T, 33, 3)
        speed = np.linalg.norm(v, axis=-1)                                # (T, 33)
        return np.concatenate([lm, ang, speed], axis=1).astype(np.float32)
