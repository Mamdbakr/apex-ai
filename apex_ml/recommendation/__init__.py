"""Adaptive recommendation, fatigue tracking, coaching."""
from .profile import (
    UserProfile, UserGoals, WorkoutSession, WorkoutSet, EXERCISE_MUSCLES,
)
from .fatigue import estimate_recovery, RecoveryState
from .ranking import ExerciseRanker, ExerciseMeta, DEFAULT_CATALOG, RankedExercise
from .generator import WorkoutGenerator, GeneratedWorkout, WorkoutBlock
from .performance import PerformancePredictor, PerformancePrediction
from .coaching import AICoach, CoachingSuggestion

__all__ = [
    "UserProfile", "UserGoals", "WorkoutSession", "WorkoutSet", "EXERCISE_MUSCLES",
    "estimate_recovery", "RecoveryState",
    "ExerciseRanker", "ExerciseMeta", "DEFAULT_CATALOG", "RankedExercise",
    "WorkoutGenerator", "GeneratedWorkout", "WorkoutBlock",
    "PerformancePredictor", "PerformancePrediction",
    "AICoach", "CoachingSuggestion",
]
