"""
User profile model for adaptive recommendations.

Holds workout history and derives aggregates used by the recommender:

    - per-exercise 1RM-style strength estimate (Epley formula)
    - muscle-group volume over a configurable lookback window
    - consistency (% of scheduled days completed)
    - preference scores (frequency-weighted)
    - per-exercise quality trend (rolling mean of quality scores)

Everything is pure-Python + numpy. No DB layer — the host application
owns persistence and serializes profiles as dicts via to_dict / from_dict.
This keeps apex_ml DB-agnostic so it works with whatever ORM is already
in the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, List, Optional

import numpy as np


# Coarse muscle-group taxonomy. Maps exercise -> muscle group set.
# Used for fatigue / weakness analysis. Extend freely without breaking
# anything — unknown exercises just don't contribute to grouped stats.
EXERCISE_MUSCLES: Dict[str, List[str]] = {
    "squat":         ["quads", "glutes", "core"],
    "deadlift":      ["hamstrings", "back", "glutes", "core"],
    "pushup":        ["chest", "triceps", "shoulders", "core"],
    "bench_press":   ["chest", "triceps", "shoulders"],
    "overhead_press":["shoulders", "triceps", "core"],
    "row":           ["back", "biceps"],
    "pullup":        ["back", "biceps"],
    "bicep_curl":    ["biceps"],
    "tricep_ext":    ["triceps"],
    "lunge":         ["quads", "glutes", "hamstrings"],
    "plank":         ["core"],
}


@dataclass
class WorkoutSet:
    """One set of one exercise from a workout."""
    exercise: str
    reps: int
    weight_kg: float = 0.0           # 0 for bodyweight
    rpe: Optional[float] = None      # rate of perceived exertion 1-10
    quality_score: Optional[float] = None  # 0..100 from FormCorrectionEngine
    completed: bool = True

    def estimated_1rm(self) -> float:
        """Epley formula: 1RM ≈ w * (1 + reps/30). Bodyweight => 0."""
        if self.weight_kg <= 0 or self.reps <= 0:
            return 0.0
        return float(self.weight_kg * (1.0 + self.reps / 30.0))


@dataclass
class WorkoutSession:
    """A single dated workout."""
    timestamp: datetime
    sets: List[WorkoutSet] = field(default_factory=list)
    duration_minutes: Optional[float] = None
    perceived_difficulty: Optional[float] = None  # 1-10
    notes: str = ""


@dataclass
class UserGoals:
    """Training intent. Drives workout generation."""
    primary: str = "general_fitness"   # strength|hypertrophy|endurance|fat_loss|general_fitness
    weekly_sessions: int = 3
    available_minutes: int = 45
    equipment: List[str] = field(default_factory=lambda: ["bodyweight"])


class UserProfile:
    """Aggregates a single user's history into recommender features."""

    def __init__(self, user_id: str, goals: Optional[UserGoals] = None):
        self.user_id = user_id
        self.goals = goals or UserGoals()
        self.sessions: List[WorkoutSession] = []

    # -------------------------------------------------------- mutation
    def add_session(self, session: WorkoutSession) -> None:
        self.sessions.append(session)
        self.sessions.sort(key=lambda s: s.timestamp)

    # ------------------------------------------------------ aggregates
    def sessions_in_window(self, days: int) -> List[WorkoutSession]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return [s for s in self.sessions if s.timestamp >= cutoff]

    def strength_estimate(self, exercise: str, days: int = 60) -> float:
        """Best estimated 1RM for the exercise within the last `days`."""
        best = 0.0
        for s in self.sessions_in_window(days):
            for st in s.sets:
                if st.exercise == exercise and st.completed:
                    best = max(best, st.estimated_1rm())
        return best

    def volume_by_muscle(self, days: int = 7) -> Dict[str, float]:
        """Sum of reps * max(1, weight) per muscle group in the last `days`."""
        out: Dict[str, float] = defaultdict(float)
        for s in self.sessions_in_window(days):
            for st in s.sets:
                muscles = EXERCISE_MUSCLES.get(st.exercise, [])
                load = st.reps * max(1.0, st.weight_kg)
                share = load / max(len(muscles), 1)
                for m in muscles:
                    out[m] += share
        return dict(out)

    def consistency(self, days: int = 28) -> float:
        """Fraction of expected weekly sessions actually completed."""
        recent = self.sessions_in_window(days)
        expected = max(1, int(self.goals.weekly_sessions * (days / 7)))
        return float(min(1.0, len(recent) / expected))

    def preference_scores(self, days: int = 60) -> Dict[str, float]:
        """Normalized frequency of each exercise. Sums to 1."""
        counts: Dict[str, float] = defaultdict(float)
        for s in self.sessions_in_window(days):
            for st in s.sets:
                counts[st.exercise] += 1
        total = sum(counts.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in counts.items()}

    def quality_trend(self, exercise: str, days: int = 30) -> float:
        """Average quality_score for `exercise` over the window (0..100)."""
        scores = [
            st.quality_score
            for s in self.sessions_in_window(days)
            for st in s.sets
            if st.exercise == exercise and st.quality_score is not None
        ]
        return float(np.mean(scores)) if scores else 0.0

    def progression_rate(self, exercise: str, days: int = 60) -> float:
        """Linear-regression slope of estimated 1RM vs days. kg/day."""
        pts = []
        for s in self.sessions_in_window(days):
            best = 0.0
            for st in s.sets:
                if st.exercise == exercise and st.completed:
                    best = max(best, st.estimated_1rm())
            if best > 0:
                pts.append((s.timestamp.timestamp(), best))
        if len(pts) < 2:
            return 0.0
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        # Convert seconds to days for an interpretable slope
        xs_days = (xs - xs[0]) / 86400.0
        if xs_days[-1] - xs_days[0] < 1e-3:
            return 0.0
        slope = float(np.polyfit(xs_days, ys, 1)[0])
        return slope

    def weak_muscles(self, days: int = 14, top_k: int = 3) -> List[str]:
        """Muscle groups with the lowest recent volume.

        We compare each known muscle's volume to the mean and return the
        bottom-k. Useful for filling gaps in a programmed workout.
        """
        vol = self.volume_by_muscle(days)
        all_muscles = {m for ms in EXERCISE_MUSCLES.values() for m in ms}
        # Treat unworked muscles as zero volume
        v_full = {m: vol.get(m, 0.0) for m in all_muscles}
        ranked = sorted(v_full.items(), key=lambda kv: kv[1])
        return [m for m, _ in ranked[:top_k]]

    # ------------------------------------------------------ I/O helpers
    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "goals": asdict(self.goals),
            "sessions": [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "duration_minutes": s.duration_minutes,
                    "perceived_difficulty": s.perceived_difficulty,
                    "notes": s.notes,
                    "sets": [asdict(st) for st in s.sets],
                }
                for s in self.sessions
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserProfile":
        goals = UserGoals(**d.get("goals", {}))
        p = cls(user_id=d["user_id"], goals=goals)
        for raw in d.get("sessions", []):
            sess = WorkoutSession(
                timestamp=datetime.fromisoformat(raw["timestamp"]),
                duration_minutes=raw.get("duration_minutes"),
                perceived_difficulty=raw.get("perceived_difficulty"),
                notes=raw.get("notes", ""),
                sets=[WorkoutSet(**st) for st in raw.get("sets", [])],
            )
            p.sessions.append(sess)
        p.sessions.sort(key=lambda s: s.timestamp)
        return p
