"""
Geometric utilities for pose analysis.

All functions accept numpy arrays of shape (N, 3) representing landmarks
in MediaPipe's normalized [0, 1] coordinate space (x, y, z). Functions
that take a single landmark accept shape (3,).

Conventions:
    - Angles are returned in degrees, in [0, 180].
    - Distances are in normalized image units (callers can rescale).
    - We treat z as relative depth from MediaPipe; useful but noisier.
"""

from __future__ import annotations

import numpy as np


def angle_between(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> float:
    """Return the angle at `vertex` formed by rays to `a` and `c`, in degrees.

    Uses the numerically stable arctan2-based formulation rather than
    acos(dot/|u||v|) to avoid blowups when vectors are nearly parallel.
    """
    v1 = a - vertex
    v2 = c - vertex
    # Cross product magnitude (3D) handles both 2D-in-3D and full 3D inputs
    cross = np.linalg.norm(np.cross(v1, v2))
    dot = float(np.dot(v1, v2))
    if cross == 0.0 and dot == 0.0:
        return 0.0
    return float(np.degrees(np.arctan2(cross, dot)))


def joint_angle(landmarks: np.ndarray, a_idx: int, v_idx: int, c_idx: int) -> float:
    """Convenience wrapper: compute a joint angle from index triplet."""
    return angle_between(landmarks[a_idx], landmarks[v_idx], landmarks[c_idx])


def all_joint_angles(landmarks: np.ndarray, joint_map: dict) -> dict:
    """Compute every joint angle in `joint_map` -> {name: degrees}."""
    return {
        name: joint_angle(landmarks, a, v, c)
        for name, (a, v, c) in joint_map.items()
    }


def center_of_mass(landmarks: np.ndarray, indices: list) -> np.ndarray:
    """Approximate COM as the centroid of the supplied landmark indices.

    Real biomechanical COM uses segment masses; we approximate with the
    torso centroid which is sufficient for relative stability tracking.
    """
    return np.mean(landmarks[indices], axis=0)


def torso_length(landmarks: np.ndarray,
                 shoulder_l: int, shoulder_r: int,
                 hip_l: int, hip_r: int) -> float:
    """Distance from shoulder midpoint to hip midpoint.

    Used to normalize other distances so they are scale-invariant to the
    person's distance from the camera.
    """
    shoulder_mid = 0.5 * (landmarks[shoulder_l] + landmarks[shoulder_r])
    hip_mid = 0.5 * (landmarks[hip_l] + landmarks[hip_r])
    return float(np.linalg.norm(shoulder_mid - hip_mid))


def normalize_landmarks(landmarks: np.ndarray,
                        hip_l: int, hip_r: int,
                        shoulder_l: int, shoulder_r: int) -> np.ndarray:
    """Translate to hip-midpoint origin, scale by torso length.

    Makes downstream models invariant to position-in-frame and distance.
    """
    hip_mid = 0.5 * (landmarks[hip_l] + landmarks[hip_r])
    scale = torso_length(landmarks, shoulder_l, shoulder_r, hip_l, hip_r)
    if scale < 1e-6:
        scale = 1.0
    return (landmarks - hip_mid) / scale


def vector_angle_to_vertical(p_top: np.ndarray, p_bottom: np.ndarray) -> float:
    """Angle (deg) between the vector p_top->p_bottom and the vertical y-axis.

    In image coordinates y grows downward, so we compare against (0, 1, 0).
    Used for spine alignment and stance vertical checks.
    """
    v = p_top - p_bottom
    n = np.linalg.norm(v)
    if n < 1e-6:
        return 0.0
    v = v / n
    vertical = np.array([0.0, -1.0, 0.0])  # "up" in image space
    cosang = np.clip(np.dot(v, vertical), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def safe_div(num: float, denom: float, default: float = 0.0) -> float:
    return num / denom if abs(denom) > 1e-9 else default
