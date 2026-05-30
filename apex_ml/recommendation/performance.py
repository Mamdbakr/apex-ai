"""
Performance prediction.

Given a planned workout block and the user's profile, predict the
probability the user will complete it cleanly (all reps, RPE <= 8.5).
We use a simple logistic regression so it's trainable on tiny datasets;
when historical data is sparse it falls back to a prior derived from
RPE and current recovery state.

This is intentionally lightweight — XGBoost / LightGBM can be swapped
in by replacing the `_predict_proba` method without changing the public
surface. We do not require those libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .fatigue import estimate_recovery
from .profile import UserProfile


@dataclass
class PerformancePrediction:
    completion_probability: float    # 0..1
    expected_rpe: float              # 1..10
    confidence: float                # 0..1


def _features(profile: UserProfile, exercise: str,
              reps: int, load_kg: Optional[float]) -> np.ndarray:
    rec = estimate_recovery(profile)
    one_rm = profile.strength_estimate(exercise, days=60)
    intensity = (load_kg / one_rm) if (load_kg and one_rm > 0) else 0.0
    prog = profile.progression_rate(exercise, days=30)
    quality = profile.quality_trend(exercise, days=30) / 100.0
    return np.array([
        intensity,
        reps / 20.0,
        rec.readiness,
        rec.overtraining_risk,
        prog,
        quality,
        profile.consistency(28),
    ], dtype=np.float64)


class PerformancePredictor:
    """Logistic-regression-style scorer.

    The default weights are sensibly hand-set; calling .fit() with real
    history (features, labels) will overwrite them via a simple gradient
    descent in pure numpy.
    """

    def __init__(self) -> None:
        # Index: [intensity, reps_norm, readiness, ot_risk, prog, quality, consistency]
        self.w = np.array([-3.0, -1.5, +2.0, -2.5, +0.5, +1.5, +1.0])
        self.b = +0.5

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 200, lr: float = 0.05) -> None:
        """Logistic regression with L2. Tiny optimizer; CPU-only."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        w = self.w.copy()
        b = self.b
        n = len(X)
        for _ in range(epochs):
            z = X @ w + b
            p = 1.0 / (1.0 + np.exp(-z))
            grad_w = (X.T @ (p - y)) / n + 0.01 * w
            grad_b = float((p - y).mean())
            w -= lr * grad_w
            b -= lr * grad_b
        self.w, self.b = w, b

    def predict(self, profile: UserProfile, exercise: str,
                reps: int, load_kg: Optional[float] = None
                 ) -> PerformancePrediction:
        x = _features(profile, exercise, reps, load_kg)
        z = float(x @ self.w + self.b)
        p = 1.0 / (1.0 + np.exp(-z))
        # Map logit-distance to expected RPE: stronger overshoot => higher RPE
        expected_rpe = float(np.clip(10.0 - 6.0 * p, 1.0, 10.0))
        # Confidence: how far from 0.5
        conf = float(min(1.0, abs(p - 0.5) * 2))
        return PerformancePrediction(
            completion_probability=float(p),
            expected_rpe=expected_rpe,
            confidence=conf,
        )
