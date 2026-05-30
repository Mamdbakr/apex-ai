"""
backend/cv/exercise_profiles.py
─────────────────────────────────
Per-exercise CV configuration — the single source of truth that feeds the
Ultralytics-style AI Gym rep counter.

Each profile says, for one exercise:
  • kpts_left  / kpts_right  : the 3-keypoint triplet to compute the joint angle
                                from on each side (COCO-17 indices).
  • up_angle / down_angle    : angles (degrees) defining the two stages.
                                Convention: "up" is the EXTENDED position
                                (arms straight, knees straight, hips locked),
                                "down" is the FLEXED position. The counter
                                increments one rep on every up→down→up cycle.
  • mode                     : "rep"  → counted by stage cycle
                               "hold" → counted by elapsed seconds in stage
                               "none" → no counter (warmup, walking, …)
  • min_visibility           : per-keypoint confidence/visibility threshold —
                                if any of the 6 tracked keypoints is below
                                this, the frame is skipped from the counter.
  • form_checks              : optional list of (keypoints, angle, op, threshold,
                                cue) rules used by the form scorer.

COCO-17 keypoint indices (the format YOLOv8-pose and our MediaPipe shim emit):
   0 nose      5 L-shoulder   6 R-shoulder    7 L-elbow    8 R-elbow
   1 L-eye     9 L-wrist     10 R-wrist      11 L-hip     12 R-hip
   2 R-eye    13 L-knee      14 R-knee       15 L-ankle   16 R-ankle
   3 L-ear
   4 R-ear

Adding a new exercise = adding one entry to EXERCISE_PROFILES below. No code
changes elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Form-check rule — evaluated each frame in the AI Gym counter.
#   joints   : 3 keypoint indices to compute an angle from
#   op       : "<", ">", "<=", ">="
#   threshold: angle in degrees
#   cue      : human-readable feedback shown to the user when the rule fires
#   weight   : how many points the form score loses if this rule fires
#              (default 10, range 0–40)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FormCheck:
    joints:    Tuple[int, int, int]
    op:        str
    threshold: float
    cue:       str
    weight:    float = 10.0


@dataclass(frozen=True)
class ExerciseProfile:
    name:           str
    display_name:   str
    mode:           str                          # "rep" | "hold" | "none"
    kpts_left:      Tuple[int, int, int]
    kpts_right:     Tuple[int, int, int]
    up_angle:       float
    down_angle:     float
    min_visibility: float = 0.20
    smoothing:      int   = 3                    # frames in the angle EMA
    form_checks:    Tuple[FormCheck, ...] = ()
    # Optional secondary angle (e.g. torso) — used only by form scorer
    secondary_left:  Optional[Tuple[int, int, int]] = None
    secondary_right: Optional[Tuple[int, int, int]] = None


# ──────────────────────────────────────────────────────────────────────────────
# Common form-check rule fragments
# ──────────────────────────────────────────────────────────────────────────────

# Torso lean: shoulder–hip–knee. <140° means rounding/folding too far.
_TORSO_ROUND_L = FormCheck((5, 11, 13), "<", 140.0,
                           "Keep your back straight — chest up, brace your core.", 12)
_TORSO_ROUND_R = FormCheck((6, 12, 14), "<", 140.0,
                           "Keep your back straight — chest up, brace your core.", 12)

# Knees caving (knee should stay over ankle on the descent)
# Approximated by the hip–knee–ankle angle going below ~80° at the bottom.
_DEEP_KNEE_L   = FormCheck((11, 13, 15), "<", 60.0,
                           "Knees collapsing too deep — push them outward.", 8)
_DEEP_KNEE_R   = FormCheck((12, 14, 16), "<", 60.0,
                           "Knees collapsing too deep — push them outward.", 8)

# Hips sagging on push-up / plank — measured by shoulder–hip–knee
_HIP_SAG_L     = FormCheck((5, 11, 13), "<", 160.0,
                           "Hips sagging — squeeze glutes, keep one straight line.", 10)
_HIP_SAG_R     = FormCheck((6, 12, 14), "<", 160.0,
                           "Hips sagging — squeeze glutes, keep one straight line.", 10)


# ──────────────────────────────────────────────────────────────────────────────
# REGISTRY
# ──────────────────────────────────────────────────────────────────────────────

EXERCISE_PROFILES: dict[str, ExerciseProfile] = {

    # ── Lower body ──────────────────────────────────────────────────────────
    "squat": ExerciseProfile(
        name="squat", display_name="Squat", mode="rep",
        kpts_left =(11, 13, 15),    # L hip → L knee → L ankle
        kpts_right=(12, 14, 16),    # R hip → R knee → R ankle
        up_angle=165.0, down_angle=95.0,
        secondary_left =(5, 11, 13),   # torso for posture cue
        secondary_right=(6, 12, 14),
        form_checks=(_TORSO_ROUND_L, _TORSO_ROUND_R),
    ),

    "deadlift": ExerciseProfile(
        name="deadlift", display_name="Deadlift", mode="rep",
        kpts_left =(5, 11, 13),     # shoulder-hip-knee (hip hinge angle)
        kpts_right=(6, 12, 14),
        up_angle=165.0, down_angle=120.0,
    ),

    "romanian_deadlift": ExerciseProfile(
        name="romanian_deadlift", display_name="Romanian Deadlift", mode="rep",
        kpts_left =(5, 11, 13),
        kpts_right=(6, 12, 14),
        up_angle=165.0, down_angle=130.0,
    ),

    "leg_extension": ExerciseProfile(
        # Inverted thresholds: at the TOP of the rep the knee is EXTENDED (~170°)
        # and at the BOTTOM (start) it's FLEXED (~90°). The counter detects
        # cycles regardless of direction, so we still use up_angle > down_angle.
        name="leg_extension", display_name="Leg Extension", mode="rep",
        kpts_left =(11, 13, 15),
        kpts_right=(12, 14, 16),
        up_angle=160.0, down_angle=95.0,
    ),

    "leg_raises": ExerciseProfile(
        # Hip flexion: standing/lying → legs raised.
        name="leg_raises", display_name="Leg Raises", mode="rep",
        kpts_left =(5, 11, 13),
        kpts_right=(6, 12, 14),
        up_angle=170.0, down_angle=110.0,
    ),

    # ── Push (chest / shoulders / triceps) ──────────────────────────────────
    "push_up": ExerciseProfile(
        name="push_up", display_name="Push-Up", mode="rep",
        kpts_left =(5, 7, 9),       # L shoulder → L elbow → L wrist
        kpts_right=(6, 8, 10),
        up_angle=160.0, down_angle=90.0,
        form_checks=(_HIP_SAG_L, _HIP_SAG_R),
    ),

    "bench_press": ExerciseProfile(
        name="bench_press", display_name="Bench Press", mode="rep",
        kpts_left =(5, 7, 9),
        kpts_right=(6, 8, 10),
        up_angle=160.0, down_angle=85.0,
    ),

    "shoulder_press": ExerciseProfile(
        name="shoulder_press", display_name="Shoulder Press", mode="rep",
        kpts_left =(5, 7, 9),
        kpts_right=(6, 8, 10),
        up_angle=160.0, down_angle=80.0,
    ),

    "tricep_dips": ExerciseProfile(
        name="tricep_dips", display_name="Tricep Dips", mode="rep",
        kpts_left =(5, 7, 9),
        kpts_right=(6, 8, 10),
        up_angle=160.0, down_angle=85.0,
    ),

    "lateral_raise": ExerciseProfile(
        # Arm-to-torso angle: hip → shoulder → elbow.
        # Bottom (resting) ≈ 20°,  top (parallel to floor) ≈ 90°.
        name="lateral_raise", display_name="Lateral Raise", mode="rep",
        kpts_left =(11, 5, 7),
        kpts_right=(12, 6, 8),
        up_angle=85.0, down_angle=25.0,
    ),

    # ── Pull (back / biceps) ────────────────────────────────────────────────
    "pull_up": ExerciseProfile(
        name="pull_up", display_name="Pull-Up", mode="rep",
        kpts_left =(5, 7, 9),
        kpts_right=(6, 8, 10),
        up_angle=160.0, down_angle=70.0,
    ),

    "lat_pulldown": ExerciseProfile(
        name="lat_pulldown", display_name="Lat Pulldown", mode="rep",
        kpts_left =(5, 7, 9),
        kpts_right=(6, 8, 10),
        up_angle=160.0, down_angle=80.0,
    ),

    "t_bar_row": ExerciseProfile(
        name="t_bar_row", display_name="T-Bar Row", mode="rep",
        kpts_left =(5, 7, 9),
        kpts_right=(6, 8, 10),
        up_angle=160.0, down_angle=80.0,
    ),

    "barbell_biceps_curl": ExerciseProfile(
        name="barbell_biceps_curl", display_name="Biceps Curl", mode="rep",
        kpts_left =(5, 7, 9),
        kpts_right=(6, 8, 10),
        up_angle=150.0, down_angle=55.0,
    ),

    # ── Hold-style (timer, no rep counting) ─────────────────────────────────
    "plank": ExerciseProfile(
        name="plank", display_name="Plank", mode="hold",
        kpts_left =(5, 11, 13),
        kpts_right=(6, 12, 14),
        up_angle=180.0, down_angle=160.0,           # hip ~ straight line
        form_checks=(_HIP_SAG_L, _HIP_SAG_R),
    ),
}


# Aliases so the auto-classifier's class IDs always resolve to a profile.
ALIASES = {
    "pushup":    "push_up",
    "pullup":    "pull_up",
    "biceps_curl":      "barbell_biceps_curl",
    "bicep_curl":       "barbell_biceps_curl",
    "deadlifts":         "deadlift",
    "shoulder_pres":     "shoulder_press",
}


def get_profile(exercise_id: Optional[str]) -> Optional[ExerciseProfile]:
    """Look up a profile, honouring aliases. Returns None for unknown IDs."""
    if not exercise_id:
        return None
    key = exercise_id.lower().strip()
    key = ALIASES.get(key, key)
    return EXERCISE_PROFILES.get(key)


def all_exercise_ids() -> List[str]:
    return sorted(EXERCISE_PROFILES.keys())
