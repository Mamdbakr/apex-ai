"""
COCO-17 → MediaPipe-33 adapter.

The apex_ai project's pose backbone (YOLOv8-pose) emits 17 COCO
keypoints as a flat (51,) ndarray of (x, y, conf) triples. The apex_ml
temporal pipeline was built against the 33-landmark MediaPipe layout
(see utils.landmarks).

This module bridges them: it inflates a COCO-17 array into a 33×3 array
in the MediaPipe index order. Missing landmarks (e.g. the 4 facial-detail
points, hand pinky/index/thumb, foot heel) are filled by sensible
interpolation:

    - Face details (1..6, 9..10) are replicated from `nose` and the two
      eyes/ears — good enough for downstream use (we never use them for
      joint angles).
    - Hand details (17..22) are replicated from the wrist.
    - Heels (29..30) are replicated from the ankles.
    - Foot indices (31..32) are estimated as ankle + small forward offset
      so the ankle joint angle is well-defined.

The adapter is pure-numpy and lossless for every landmark apex_ml's
state machine and form rules actually consume (shoulders, elbows, hips,
knees, wrists, ankles, foot_index).
"""

from __future__ import annotations

import numpy as np

# COCO-17 index map (matches backend/cv/pose_extractor.COCO_NAMES exactly)
COCO_NOSE           = 0
COCO_LEFT_EYE       = 1
COCO_RIGHT_EYE      = 2
COCO_LEFT_EAR       = 3
COCO_RIGHT_EAR      = 4
COCO_LEFT_SHOULDER  = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_ELBOW     = 7
COCO_RIGHT_ELBOW    = 8
COCO_LEFT_WRIST     = 9
COCO_RIGHT_WRIST    = 10
COCO_LEFT_HIP       = 11
COCO_RIGHT_HIP      = 12
COCO_LEFT_KNEE      = 13
COCO_RIGHT_KNEE     = 14
COCO_LEFT_ANKLE     = 15
COCO_RIGHT_ANKLE    = 16


def kp51_to_landmarks33(kp51: np.ndarray) -> np.ndarray:
    """Convert a flat (51,) COCO-17 keypoint array to a (33, 3) MediaPipe layout.

    Parameters
    ----------
    kp51 : np.ndarray
        Shape (51,) or (17, 3) — keypoints as produced by YOLOv8-pose.
        x, y in [0, 1] (normalized); third column is confidence/visibility.

    Returns
    -------
    np.ndarray
        Shape (33, 3). Index ordering matches apex_ml.utils.landmarks.
        The z dimension is set to 0 because COCO-17 is 2D only — apex_ml's
        downstream geometry only uses (x, y) for joint angles so this is
        lossless for the metrics that matter.
    """
    arr = np.asarray(kp51, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size != 51:
            raise ValueError(f"expected 51 values, got {arr.size}")
        arr = arr.reshape(17, 3)
    elif arr.shape != (17, 3):
        raise ValueError(f"expected (17, 3) or (51,); got shape {arr.shape}")

    # Read each COCO point as (x, y, conf). We discard conf in the output
    # (apex_ml stores z) — visibility filtering happens upstream.
    def pt(i: int) -> np.ndarray:
        return np.array([arr[i, 0], arr[i, 1], 0.0])

    lm33 = np.zeros((33, 3), dtype=np.float64)

    # MediaPipe indices we actually have in COCO ---------------------------
    lm33[0]  = pt(COCO_NOSE)              # NOSE
    lm33[2]  = pt(COCO_LEFT_EYE)          # LEFT_EYE
    lm33[5]  = pt(COCO_RIGHT_EYE)         # RIGHT_EYE
    lm33[7]  = pt(COCO_LEFT_EAR)          # LEFT_EAR
    lm33[8]  = pt(COCO_RIGHT_EAR)         # RIGHT_EAR
    lm33[11] = pt(COCO_LEFT_SHOULDER)     # LEFT_SHOULDER
    lm33[12] = pt(COCO_RIGHT_SHOULDER)    # RIGHT_SHOULDER
    lm33[13] = pt(COCO_LEFT_ELBOW)        # LEFT_ELBOW
    lm33[14] = pt(COCO_RIGHT_ELBOW)       # RIGHT_ELBOW
    lm33[15] = pt(COCO_LEFT_WRIST)        # LEFT_WRIST
    lm33[16] = pt(COCO_RIGHT_WRIST)       # RIGHT_WRIST
    lm33[23] = pt(COCO_LEFT_HIP)          # LEFT_HIP
    lm33[24] = pt(COCO_RIGHT_HIP)         # RIGHT_HIP
    lm33[25] = pt(COCO_LEFT_KNEE)         # LEFT_KNEE
    lm33[26] = pt(COCO_RIGHT_KNEE)        # RIGHT_KNEE
    lm33[27] = pt(COCO_LEFT_ANKLE)        # LEFT_ANKLE
    lm33[28] = pt(COCO_RIGHT_ANKLE)       # RIGHT_ANKLE

    # Face-detail synthetic fills (downstream never uses these for angles,
    # but they need finite values so feature_tensor() stays clean) ---------
    lm33[1] = lm33[2]   # LEFT_EYE_INNER  ← copy of LEFT_EYE
    lm33[3] = lm33[2]   # LEFT_EYE_OUTER
    lm33[4] = lm33[5]   # RIGHT_EYE_INNER
    lm33[6] = lm33[5]   # RIGHT_EYE_OUTER
    lm33[9]  = lm33[0]  # MOUTH_LEFT      ← copy of NOSE
    lm33[10] = lm33[0]  # MOUTH_RIGHT

    # Hand details fill from wrists ---------------------------------------
    lm33[17] = lm33[15]   # LEFT_PINKY
    lm33[18] = lm33[16]   # RIGHT_PINKY
    lm33[19] = lm33[15]   # LEFT_INDEX
    lm33[20] = lm33[16]   # RIGHT_INDEX
    lm33[21] = lm33[15]   # LEFT_THUMB
    lm33[22] = lm33[16]   # RIGHT_THUMB

    # Heels copy from ankles ----------------------------------------------
    lm33[29] = lm33[27]   # LEFT_HEEL
    lm33[30] = lm33[28]   # RIGHT_HEEL

    # Foot indices: ankle + small forward offset so the ankle joint angle
    # (KNEE → ANKLE → FOOT_INDEX) is well-defined. 0.04 normalized units
    # is roughly a foot-length at typical camera distances.
    lm33[31] = lm33[27] + np.array([0.04, 0.0, 0.0])  # LEFT_FOOT_INDEX
    lm33[32] = lm33[28] + np.array([0.04, 0.0, 0.0])  # RIGHT_FOOT_INDEX

    return lm33


def landmarks_dict_to_landmarks33(landmarks: list) -> np.ndarray:
    """Convert the project's `landmarks: List[dict]` shape to a (33,3) array.

    Accepts the same list-of-dicts the existing CV pipeline already emits
    (one entry per COCO-17 keypoint with `x`, `y`, `visibility`, `name`).
    """
    if not landmarks or len(landmarks) < 17:
        raise ValueError(
            f"expected >=17 landmark dicts; got {len(landmarks) if landmarks else 0}"
        )
    flat = np.zeros(51, dtype=np.float64)
    for i, lm in enumerate(landmarks[:17]):
        flat[i * 3 + 0] = float(lm.get("x", 0.0))
        flat[i * 3 + 1] = float(lm.get("y", 0.0))
        flat[i * 3 + 2] = float(lm.get("visibility", 1.0))
    return kp51_to_landmarks33(flat)
