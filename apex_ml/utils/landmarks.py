"""
MediaPipe Pose landmark indices.

MediaPipe Pose returns 33 landmarks per frame. We expose the canonical
index map and a few derived groupings used by the rest of the package.

Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
"""

# 33 MediaPipe Pose landmarks
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

NUM_LANDMARKS = 33

# Joint angle definitions: (a, vertex, c). Angle is measured at `vertex`,
# between vectors (a - vertex) and (c - vertex).
JOINT_ANGLES = {
    "left_elbow":     (LEFT_SHOULDER, LEFT_ELBOW,    LEFT_WRIST),
    "right_elbow":    (RIGHT_SHOULDER, RIGHT_ELBOW,  RIGHT_WRIST),
    "left_shoulder":  (LEFT_ELBOW,    LEFT_SHOULDER, LEFT_HIP),
    "right_shoulder": (RIGHT_ELBOW,   RIGHT_SHOULDER, RIGHT_HIP),
    "left_hip":       (LEFT_SHOULDER, LEFT_HIP,      LEFT_KNEE),
    "right_hip":      (RIGHT_SHOULDER, RIGHT_HIP,    RIGHT_KNEE),
    "left_knee":      (LEFT_HIP,      LEFT_KNEE,     LEFT_ANKLE),
    "right_knee":     (RIGHT_HIP,     RIGHT_KNEE,    RIGHT_ANKLE),
    "left_ankle":     (LEFT_KNEE,     LEFT_ANKLE,    LEFT_FOOT_INDEX),
    "right_ankle":    (RIGHT_KNEE,    RIGHT_ANKLE,   RIGHT_FOOT_INDEX),
}

# Symmetric pairs used for asymmetry / balance scoring
SYMMETRIC_PAIRS = [
    ("left_elbow", "right_elbow"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
]

# Landmarks used for center-of-mass approximation (torso centroid)
COM_LANDMARKS = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
