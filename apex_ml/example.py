"""
Self-contained runnable example.

Demonstrates:
    1. Streaming pose frames through SequenceBuffer + state machine
    2. Receiving phase transitions and rep quality
    3. Generating an adaptive workout from a synthetic profile
    4. Getting AI coaching suggestions

No external dependencies beyond numpy. Run with:

    python -m apex_ml.example
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from apex_ml.temporal_pose import SequenceBuffer, ExerciseStateMachine, Phase
from apex_ml.form_correction import FormCorrectionEngine, build_overlay
from apex_ml.recommendation import (
    AICoach, UserGoals, UserProfile, WorkoutGenerator, WorkoutSession,
    WorkoutSet, estimate_recovery,
)
from apex_ml.utils.landmarks import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
)


def synthetic_squat_frame(knee_angle_deg: float) -> np.ndarray:
    """Build a realistic 33×3 pose with given knee bend via 2D IK."""
    lm = np.zeros((33, 3))
    L = 0.22
    cx, ankle_y = 0.5, 0.92
    K = math.radians(knee_angle_deg)
    hip_ankle = 2 * L * math.sin(K / 2)
    knee_perp = L * math.cos(K / 2)
    hip_y = ankle_y - hip_ankle
    knee_y = (hip_y + ankle_y) / 2
    lean = (180 - knee_angle_deg) / 100 * 0.04

    lm[LEFT_HIP]    = [cx - 0.07, hip_y, 0.0]
    lm[RIGHT_HIP]   = [cx + 0.07, hip_y, 0.0]
    lm[LEFT_KNEE]   = [cx - 0.07 - knee_perp, knee_y, 0.0]
    lm[RIGHT_KNEE]  = [cx + 0.07 - knee_perp, knee_y, 0.0]
    lm[LEFT_ANKLE]  = [cx - 0.08, ankle_y, 0.0]
    lm[RIGHT_ANKLE] = [cx + 0.08, ankle_y, 0.0]
    lm[31] = lm[LEFT_ANKLE]  + [-0.05, 0.0, 0.0]
    lm[32] = lm[RIGHT_ANKLE] + [-0.05, 0.0, 0.0]
    lm[LEFT_SHOULDER]  = [cx - 0.10 - lean, hip_y - 0.30, 0.0]
    lm[RIGHT_SHOULDER] = [cx + 0.10 - lean, hip_y - 0.30, 0.0]
    lm[LEFT_ELBOW]  = lm[LEFT_SHOULDER]  + [-0.03, 0.15, 0.0]
    lm[RIGHT_ELBOW] = lm[RIGHT_SHOULDER] + [+0.03, 0.15, 0.0]
    lm[LEFT_WRIST]  = lm[LEFT_ELBOW]     + [-0.02, 0.15, 0.0]
    lm[RIGHT_WRIST] = lm[RIGHT_ELBOW]    + [+0.02, 0.15, 0.0]
    return lm


def demo_live_pipeline() -> None:
    print("\n=== 1. Live pose pipeline (synthetic squat) ===")
    fps = 30.0
    buf = SequenceBuffer(window_size=int(fps * 3))
    fsm = ExerciseStateMachine("squat")
    engine = FormCorrectionEngine("squat", primary_joint="left_knee",
                                   cooldown_seconds=0.5)
    fsm.on_rep_completed = lambda r: print(
        f"  rep #{r.index} complete: ROM={r.rom_deg:.1f}°, "
        f"ecc={r.eccentric_t:.2f}s, con={r.concentric_t:.2f}s"
    )

    t, dt = 0.0, 1.0 / fps
    for rep in range(3):
        # eccentric: stand → bottom
        for i in range(int(fps)):
            angle = 180 - (180 - 80) * (i / fps)
            buf.push(synthetic_squat_frame(angle), t=t)
            fsm.update(buf); engine.step(buf); t += dt
        # concentric: bottom → stand
        for i in range(int(fps)):
            angle = 80 + (180 - 80) * (i / fps)
            buf.push(synthetic_squat_frame(angle), t=t)
            fsm.update(buf); engine.step(buf); t += dt

    overlay = build_overlay("squat", fsm.detector.current,
                             fsm.rep_count, [], buf)
    print(f"  total reps: {fsm.rep_count}, "
          f"phase: {overlay['phase']}, "
          f"path samples: {len(overlay['path'])}")


def demo_recommendation() -> None:
    print("\n=== 2. Adaptive workout generation ===")
    profile = UserProfile(
        user_id="demo_user",
        goals=UserGoals(primary="hypertrophy", weekly_sessions=4,
                        available_minutes=45,
                        equipment=["bodyweight", "dumbbell"]),
    )
    now = datetime.now(timezone.utc)
    # 6 weeks of progressively heavier squats + steady pushups
    for w in range(6, 0, -1):
        weight = 30 + (6 - w) * 2.5      # gradual progression
        profile.add_session(WorkoutSession(
            timestamp=now - timedelta(days=w * 4),
            duration_minutes=42,
            perceived_difficulty=6.5,
            sets=[
                WorkoutSet("squat", reps=10, weight_kg=weight, quality_score=78),
                WorkoutSet("pushup", reps=14, quality_score=82),
                WorkoutSet("row", reps=10, weight_kg=15, quality_score=70),
            ],
        ))

    rec = estimate_recovery(profile)
    print(f"  recovery: readiness={rec.readiness:.2f}, "
          f"ratio={rec.ratio:.2f}, days_since_last={rec.days_since_last}")
    print(f"           → {rec.recommendation}")

    workout = WorkoutGenerator().generate(profile)
    print(f"  generated workout ({workout.estimated_duration_min}min, "
          f"goal={workout.goal}):")
    for b in workout.blocks:
        load = f" @ {b.target_load_kg}kg" if b.target_load_kg else ""
        print(f"    • {b.exercise:14s} {b.sets}×{b.reps}{load}  — {b.rationale}")


def demo_coaching() -> None:
    print("\n=== 3. AI coaching layer ===")
    # Build a profile with weeks of low form scores on squat — coach
    # should suggest a regression.
    profile = UserProfile(user_id="demo_user_2",
                           goals=UserGoals(weekly_sessions=3))
    now = datetime.now(timezone.utc)
    for w in range(8, 0, -1):
        profile.add_session(WorkoutSession(
            timestamp=now - timedelta(days=w * 3),
            sets=[WorkoutSet("squat", reps=8, weight_kg=50, quality_score=48)],
        ))
    for s in AICoach().suggest(profile):
        print(f"  [{s.kind:13s} p{s.priority}] {s.message}")


def main() -> None:
    print("apex_ml — end-to-end demo")
    demo_live_pipeline()
    demo_recommendation()
    demo_coaching()
    print("\n✓ all demos completed")


if __name__ == "__main__":
    main()
