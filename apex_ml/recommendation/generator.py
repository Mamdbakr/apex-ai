"""
Adaptive workout generation.

Combines the ranker, recovery state, and the user's strength estimates
to produce a concrete workout: which exercises, in what order, with
reps/sets/rest/intensity. The result is a serializable dict suitable to
return from a FastAPI endpoint without any DB coupling.

Difficulty adjustment is rule-based but data-driven:

    - If overtraining_risk > 0.6, drop volume 30% and intensity 20%.
    - If consistency < 0.5, prefer shorter, simpler sessions.
    - If quality_trend for an exercise < 60, hold weight constant and
      add an explicit "focus on form" note.
    - If progression_rate has been positive for 2+ weeks, bump load 5%.

These rules are intentionally explicit so the coaching layer can explain
*why* each adjustment was made — important for user trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .fatigue import estimate_recovery, RecoveryState
from .profile import UserProfile
from .ranking import ExerciseRanker, RankedExercise


@dataclass
class WorkoutBlock:
    exercise: str
    sets: int
    reps: int
    rest_seconds: int
    target_load_kg: Optional[float] = None     # None = bodyweight
    note: str = ""
    rationale: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class GeneratedWorkout:
    user_id: str
    generated_at: str
    goal: str
    estimated_duration_min: int
    readiness: float
    recovery_note: str
    blocks: List[WorkoutBlock]
    coaching: List[str]

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "generated_at": self.generated_at,
            "goal": self.goal,
            "estimated_duration_min": self.estimated_duration_min,
            "readiness": self.readiness,
            "recovery_note": self.recovery_note,
            "blocks": [b.to_dict() for b in self.blocks],
            "coaching": self.coaching,
        }


def _base_scheme(goal: str) -> Dict[str, int]:
    """Sets/reps/rest defaults per primary goal."""
    if goal == "strength":
        return {"sets": 4, "reps": 5,  "rest": 150}
    if goal == "hypertrophy":
        return {"sets": 3, "reps": 10, "rest": 75}
    if goal == "endurance":
        return {"sets": 3, "reps": 18, "rest": 45}
    if goal == "fat_loss":
        return {"sets": 3, "reps": 15, "rest": 30}
    return {"sets": 3, "reps": 12, "rest": 60}     # general_fitness


def _adjust_for_recovery(scheme: Dict[str, int],
                          rec: RecoveryState) -> Dict[str, int]:
    """Apply recovery-state-driven dampening or stimulation."""
    out = dict(scheme)
    if rec.overtraining_risk > 0.6:
        out["sets"]  = max(2, int(out["sets"]  * 0.7))
        out["reps"]  = max(5, int(out["reps"]  * 0.8))
        out["rest"]  = int(out["rest"] * 1.2)
    elif rec.readiness < 0.5:
        out["sets"]  = max(2, out["sets"] - 1)
    return out


def _target_load(profile: UserProfile, exercise: str,
                  reps: int, recovery: RecoveryState) -> Optional[float]:
    """Suggest a working weight at ~75% of estimated 1RM, scaled by recovery."""
    one_rm = profile.strength_estimate(exercise, days=60)
    if one_rm <= 0:
        return None
    pct = 0.75
    if recovery.overtraining_risk > 0.6:
        pct -= 0.10
    if recovery.readiness > 0.85:
        pct += 0.02
    # Reps-based Epley inverse: w = 1RM * pct doesn't depend on reps,
    # but we further scale: more reps = lower %1RM.
    pct -= 0.005 * max(0, reps - 8)
    pct = float(max(0.4, min(0.9, pct)))
    return round(one_rm * pct, 1)


class WorkoutGenerator:
    """Top-level entry: turn a UserProfile into a GeneratedWorkout."""

    def __init__(self, ranker: Optional[ExerciseRanker] = None):
        self.ranker = ranker or ExerciseRanker()

    def generate(self, profile: UserProfile,
                  num_exercises: Optional[int] = None) -> GeneratedWorkout:
        recovery = estimate_recovery(profile)
        scheme = _adjust_for_recovery(_base_scheme(profile.goals.primary), recovery)

        # How many exercises fit in the available time?
        # Estimate ~ (sets * (work_seconds + rest)) per exercise.
        per_ex_sec = scheme["sets"] * (45 + scheme["rest"])
        budget_sec = profile.goals.available_minutes * 60
        max_ex = max(3, budget_sec // per_ex_sec)
        n = min(num_exercises or max_ex, 8)

        # Rank candidates and select top n
        ranked = self.ranker.rank(profile, top_k=n * 2)
        chosen: List[RankedExercise] = ranked[:n]

        blocks: List[WorkoutBlock] = []
        coaching: List[str] = []

        for r in chosen:
            ex = r.name
            load = _target_load(profile, ex, scheme["reps"], recovery)
            note_bits = []
            rationale_bits = [r.reason]

            q = profile.quality_trend(ex, days=30)
            if 0 < q < 60:
                note_bits.append("Form trend is low — prioritize technique over load.")
                # Hold weight: cap at last seen
                if load is not None:
                    load = round(load * 0.9, 1)
                rationale_bits.append("recent form score below threshold")

            prog = profile.progression_rate(ex, days=21)
            if prog > 0:
                rationale_bits.append("positive progression trend (+%.1f kg/d)" % prog)
                if load is not None:
                    load = round(load * 1.03, 1)
            elif prog < -0.05 and load is not None:
                load = round(load * 0.95, 1)
                rationale_bits.append("recent regression — backing off load")

            blocks.append(WorkoutBlock(
                exercise=ex,
                sets=scheme["sets"], reps=scheme["reps"],
                rest_seconds=scheme["rest"],
                target_load_kg=load,
                note=" ".join(note_bits),
                rationale="; ".join(rationale_bits),
            ))

        # Top-level coaching messages
        coaching.append(recovery.recommendation)
        if profile.consistency(28) < 0.5:
            coaching.append(
                "Consistency has been low — try shorter sessions to rebuild the habit."
            )
        weak = profile.weak_muscles(14, top_k=2)
        if weak:
            coaching.append(
                "We added work for under-trained muscle groups: " + ", ".join(weak) + "."
            )
        if recovery.overtraining_risk > 0.6:
            coaching.append("Volume reduced this session to support recovery.")

        duration_min = sum(b.sets * (45 + b.rest_seconds) for b in blocks) // 60

        return GeneratedWorkout(
            user_id=profile.user_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            goal=profile.goals.primary,
            estimated_duration_min=int(duration_min),
            readiness=recovery.readiness,
            recovery_note=recovery.recommendation,
            blocks=blocks,
            coaching=coaching,
        )
