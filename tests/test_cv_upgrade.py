"""
tests/test_cv_upgrade.py
─────────────────────────
Integration tests for the YOLOv8 AI-Gym CV upgrade.

These tests use a stubbed pose backbone (no real YOLO weights needed) so they
run instantly in CI and don't depend on network access. The stub emits
deterministic keypoints; the assertions verify that the AI Gym counter, form
scorer, and FrameResult contract all behave correctly.

Run:
    pytest tests/test_cv_upgrade.py -v
"""
from __future__ import annotations

import time
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── stub pose backbone ──────────────────────────────────────────────────────

class _StubPose:
    """Pose backbone that emits known keypoints for one of two scenarios."""
    backend = "yolov8-pose"
    device  = "cpu"

    def __init__(self, scenario: str = "extended"):
        self.scenario = scenario

    def extract(self, frame_bgr):
        kp = np.zeros(51, dtype=np.float32)
        # All keypoints visible at neutral position
        for i in range(17):
            kp[i * 3] = 0.5
            kp[i * 3 + 1] = 0.5
            kp[i * 3 + 2] = 0.95

        if self.scenario == "extended":
            # standing / arms straight
            kp[11 * 3 : 11 * 3 + 2] = [0.4, 0.4]
            kp[13 * 3 : 13 * 3 + 2] = [0.4, 0.6]
            kp[15 * 3 : 15 * 3 + 2] = [0.4, 0.8]
            kp[12 * 3 : 12 * 3 + 2] = [0.6, 0.4]
            kp[14 * 3 : 14 * 3 + 2] = [0.6, 0.6]
            kp[16 * 3 : 16 * 3 + 2] = [0.6, 0.8]
        elif self.scenario == "squatted":
            # bottom of a squat — knee bent
            kp[11 * 3 : 11 * 3 + 2] = [0.4, 0.5]
            kp[13 * 3 : 13 * 3 + 2] = [0.4, 0.7]
            kp[15 * 3 : 15 * 3 + 2] = [0.55, 0.7]
            kp[12 * 3 : 12 * 3 + 2] = [0.6, 0.5]
            kp[14 * 3 : 14 * 3 + 2] = [0.6, 0.7]
            kp[16 * 3 : 16 * 3 + 2] = [0.45, 0.7]
        elif self.scenario == "plank":
            # straight line, hip ≈ 180°
            kp[5 * 3 : 5 * 3 + 2] = [0.3, 0.5]
            kp[11 * 3 : 11 * 3 + 2] = [0.5, 0.5]
            kp[13 * 3 : 13 * 3 + 2] = [0.7, 0.5]
            kp[6 * 3 : 6 * 3 + 2] = [0.3, 0.55]
            kp[12 * 3 : 12 * 3 + 2] = [0.5, 0.55]
            kp[14 * 3 : 14 * 3 + 2] = [0.7, 0.55]

        from backend.cv.yolo_pose import COCO_NAMES
        landmarks = [
            {"x": float(kp[i * 3]), "y": float(kp[i * 3 + 1]),
             "visibility": float(kp[i * 3 + 2]), "name": COCO_NAMES[i]}
            for i in range(17)
        ]
        return kp, landmarks

    def close(self):
        pass


def _make_pipeline(initial_scenario: str = "extended"):
    """Build a pipeline using the stub pose backend (no YOLO weights needed)."""
    from backend.cv.ai_gym import AIGymCounter
    from backend.cv.pipeline import CVPipeline
    from backend.cv.rep_counter import RepCounter

    return CVPipeline(
        pose=_StubPose(initial_scenario),
        classifier=None,
        rep_counter=RepCounter(ai_gym=AIGymCounter()),
    )


def _step(pipe, scenario: str, exercise_hint: str = "squat",
          session_id: str = "test"):
    """Push one frame through the pipeline at the chosen scenario."""
    pipe.pose = _StubPose(scenario)
    return pipe.analyze_frame(
        np.zeros((240, 320, 3), dtype=np.uint8),
        session_id=session_id,
        exercise_hint=exercise_hint,
    )


# ─── tests ───────────────────────────────────────────────────────────────────

def test_squat_counts_three_reps():
    pipe = _make_pipeline()
    # 3 squats: extended → squatted → extended  (one rep each)
    _step(pipe, "extended"); _step(pipe, "extended"); _step(pipe, "extended")
    for _ in range(3):
        _step(pipe, "squatted")
        _step(pipe, "squatted")
        r = _step(pipe, "extended")
    assert r.reps == 3
    assert r.exercise_id == "squat"
    assert r.stage == "up"


def test_frame_result_has_all_frontend_keys():
    pipe = _make_pipeline()
    _step(pipe, "extended")
    r = _step(pipe, "squatted")
    d = r.to_dict()
    # every key the existing frontend reads
    for k in ("detected", "person_detected", "pose_detected",
              "exercise_id", "exercise_name", "confidence", "top_3",
              "reps", "rep_count", "phase", "stage", "hold_seconds",
              "form_score", "form_cues", "feedback_cues",
              "keypoints", "landmarks", "fps", "backend",
              "primary_angle", "left_angle", "right_angle", "visible"):
        assert k in d, f"missing key: {k}"


def test_no_person_returns_helpful_payload():
    from backend.cv.ai_gym import AIGymCounter
    from backend.cv.pipeline import CVPipeline
    from backend.cv.rep_counter import RepCounter

    class _Empty:
        backend, device = "yolov8-pose", "cpu"
        def extract(self, f): return None, None
        def close(self): pass

    pipe = CVPipeline(pose=_Empty(), classifier=None,
                      rep_counter=RepCounter(ai_gym=AIGymCounter()))
    r = pipe.analyze_frame(np.zeros((240, 320, 3), dtype=np.uint8))
    assert not r.detected
    assert not r.person_detected
    assert r.exercise_id == "none"
    assert r.form_cues  # at least one helpful cue


def test_plank_hold_mode_accumulates_seconds():
    pipe = _make_pipeline()
    r1 = _step(pipe, "plank", exercise_hint="plank", session_id="hold")
    time.sleep(0.4)
    r2 = _step(pipe, "plank", exercise_hint="plank", session_id="hold")
    assert r2.hold_seconds > r1.hold_seconds
    assert r2.hold_seconds >= 0.3


def test_reset_zeros_reps():
    pipe = _make_pipeline()
    # Build up a couple of reps
    for _ in range(2):
        _step(pipe, "squatted")
        _step(pipe, "extended")
    assert pipe.rep_counter.get("test").reps == 2
    pipe.rep_counter.reset("test")
    r = _step(pipe, "extended")
    assert r.reps == 0


def test_legacy_angles_path_still_counts_squats():
    """The exact legacy test from test_smoke — must keep passing forever."""
    from backend.cv.rep_counter import RepCounter
    rc = RepCounter()
    angles = lambda knee: np.array(
        [180, 180, knee, knee, 170, 170, 170, 170, 170, 170],
        dtype=np.float32,
    )
    rc.update("legacy", "squat", angles(180))
    rc.update("legacy", "squat", angles(80))
    rc.update("legacy", "squat", angles(80))
    rc.update("legacy", "squat", angles(170))
    state = rc.update("legacy", "squat", angles(180))
    assert state["state"]["reps"] == 1


def test_exercise_profile_registry_complete():
    """Every exercise the classifier emits must have a profile, or the new
    pipeline can't resolve it."""
    import json
    from backend.core.config import settings
    from backend.cv.exercise_profiles import get_profile

    cfg_path = Path(settings().CV_CONFIG_PATH)
    if not cfg_path.exists():
        pytest.skip("classifier config not present in this checkout")
    cfg = json.loads(cfg_path.read_text())
    missing = [c for c in cfg["classes"] if get_profile(c) is None]
    assert not missing, f"no profile for: {missing}"


def test_form_score_drops_when_rules_fire():
    """When a profile-defined form rule trips, the score should decrease and
    a cue should appear."""
    from backend.cv.ai_gym import AIGymCounter, angle_from_kpts
    from backend.cv.exercise_profiles import EXERCISE_PROFILES

    counter = AIGymCounter()
    profile = EXERCISE_PROFILES["push_up"]

    # Build a "good" push-up: shoulder, hip, knee form one straight line.
    # Position them along a slightly downward slope so the angle ABC is
    # well-defined and close to 180° (no rule fires).
    good = np.zeros(51, dtype=np.float32)
    for i in range(17):
        good[i * 3] = 0.5; good[i * 3 + 1] = 0.5; good[i * 3 + 2] = 0.95
    good[5 * 3 : 5 * 3 + 2]  = [0.20, 0.45]   # L shoulder (high-left)
    good[7 * 3 : 7 * 3 + 2]  = [0.25, 0.65]   # L elbow
    good[9 * 3 : 9 * 3 + 2]  = [0.30, 0.85]   # L wrist
    good[11 * 3 : 11 * 3 + 2] = [0.50, 0.50]  # L hip (middle)
    good[13 * 3 : 13 * 3 + 2] = [0.80, 0.55]  # L knee (low-right)
    good[6 * 3 : 6 * 3 + 2]  = [0.20, 0.50]
    good[8 * 3 : 8 * 3 + 2]  = [0.25, 0.70]
    good[10 * 3 : 10 * 3 + 2] = [0.30, 0.90]
    good[12 * 3 : 12 * 3 + 2] = [0.50, 0.55]
    good[14 * 3 : 14 * 3 + 2] = [0.80, 0.60]

    # Sanity: the "good" angle should be very close to 180° (straight body)
    good_angle, _ = angle_from_kpts(good, profile.form_checks[0].joints)
    assert good_angle > 170, f"test setup wrong: good angle={good_angle}"

    # "Bad" pose: drop the hip way down — shoulder–hip–knee becomes a tight V
    bad = good.copy()
    bad[11 * 3 : 11 * 3 + 2] = [0.50, 0.85]   # hip dropped a lot
    bad[12 * 3 : 12 * 3 + 2] = [0.50, 0.85]
    bad_angle, _ = angle_from_kpts(bad, profile.form_checks[0].joints)
    assert bad_angle < 160, f"test setup wrong: bad angle={bad_angle}"

    s_good = counter.update("form_test_good", profile, good)
    s_bad  = counter.update("form_test_bad",  profile, bad)
    assert s_bad["form_score"] < s_good["form_score"]
    assert s_bad["form_cues"], "expected at least one cue when rule fires"
