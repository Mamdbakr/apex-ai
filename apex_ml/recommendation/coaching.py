"""
Coaching layer — long-form suggestions synthesized from sub-models.

This sits ABOVE the workout generator. The generator says "do these
exercises today"; the coaching layer says "here is the broader pattern
in your training" — deload suggestions, exercise substitutions, recovery
suggestions, and progression guidance.

Output is a list of structured suggestions; the host app can render them
in whatever UI surface already exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .fatigue import estimate_recovery
from .profile import EXERCISE_MUSCLES, UserProfile
from .ranking import DEFAULT_CATALOG


@dataclass
class CoachingSuggestion:
    kind: str            # 'deload' | 'substitution' | 'recovery' | 'progression' | 'consistency'
    message: str
    priority: int = 2    # 1 = info, 2 = important, 3 = urgent

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class AICoach:
    """Generate prioritized coaching suggestions for a user."""

    def suggest(self, profile: UserProfile) -> List[CoachingSuggestion]:
        out: List[CoachingSuggestion] = []
        rec = estimate_recovery(profile)

        # ----- deload --------------------------------------------------
        if rec.overtraining_risk > 0.6:
            out.append(CoachingSuggestion(
                kind="deload", priority=3,
                message=(
                    "Your acute training load is high relative to your "
                    f"baseline (ratio {rec.ratio:.2f}). Plan a deload week: "
                    "drop volume ~40% and intensity ~10%."
                ),
            ))
        elif rec.overtraining_risk > 0.3:
            out.append(CoachingSuggestion(
                kind="deload", priority=2,
                message="Training load is creeping high — consider a lighter session next.",
            ))

        # ----- recovery ------------------------------------------------
        if (rec.days_since_last or 0) >= 4 and rec.ratio < 0.6:
            out.append(CoachingSuggestion(
                kind="recovery", priority=2,
                message="It's been several days since your last session — start back "
                        "moderately to avoid soreness.",
            ))

        # ----- consistency --------------------------------------------
        cons = profile.consistency(28)
        if cons < 0.5:
            out.append(CoachingSuggestion(
                kind="consistency", priority=2,
                message=f"You've completed only {int(cons*100)}% of expected sessions in "
                        "the last 4 weeks. Shorter, more frequent workouts can help.",
            ))

        # ----- substitution suggestions -------------------------------
        # If quality is consistently low on an exercise, suggest a regression.
        for ex in DEFAULT_CATALOG:
            q = profile.quality_trend(ex, days=30)
            if 0 < q < 55:
                regress = _regression_for(ex)
                if regress:
                    out.append(CoachingSuggestion(
                        kind="substitution", priority=2,
                        message=(
                            f"Form scores on {ex} have been low (avg {q:.0f}/100). "
                            f"Try substituting {regress} for a few sessions to rebuild the pattern."
                        ),
                    ))

        # ----- progression guidance ------------------------------------
        for ex in DEFAULT_CATALOG:
            slope = profile.progression_rate(ex, days=60)
            if slope > 0.15:
                out.append(CoachingSuggestion(
                    kind="progression", priority=1,
                    message=(
                        f"Strong upward trend on {ex} "
                        f"(+{slope:.2f} kg/day). Continue current programming."
                    ),
                ))
            elif slope < -0.10 and profile.strength_estimate(ex, 60) > 0:
                out.append(CoachingSuggestion(
                    kind="progression", priority=2,
                    message=(
                        f"Strength regression detected on {ex} "
                        f"({slope:.2f} kg/day). Check recovery, sleep, and nutrition."
                    ),
                ))

        # Sort by priority descending so urgent items come first
        out.sort(key=lambda s: -s.priority)
        return out


# ----- Helpers ----------------------------------------------------------
_REGRESSIONS = {
    "pullup":      "assisted_pullup or row",
    "pushup":      "knee_pushup",
    "squat":       "bodyweight_squat",
    "deadlift":    "romanian_deadlift",
    "bench_press": "dumbbell_floor_press",
}


def _regression_for(exercise: str) -> Optional[str]:
    return _REGRESSIONS.get(exercise)
