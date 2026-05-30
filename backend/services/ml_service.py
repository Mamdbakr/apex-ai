"""
backend/services/ml_service.py
────────────────────────────────
APEX AI v12 — Prediction service for the trained scikit-learn models, now
with explainability built in.

Public API
    svc = get_ml_service()
    svc.predict_calories(...)           → {value, source, explanation}
    svc.predict_weight_change(...)      → {value, source, explanation}
    svc.classify_fitness(...)           → {level, probabilities, explanation}
    svc.recommend_exercises(level_id, top_k=5, profile=None)

Every prediction call also returns an `explanation` dict from
backend/services/explanation_service.py — feature attributions + a single
human-readable headline. The route layer can either pass it straight through
to the frontend or pluck the headline for chat replies.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from backend.core.config import settings
from backend.services.explanation_service import get_explanation_service


FEATURE_COLUMNS = [
    "age", "weight_kg", "height_cm", "activity_level", "gender",
    "bmi", "bmr", "age_squared", "weight_height",
]
FALLBACK_LABELS = {0: "Beginner", 1: "Intermediate", 2: "Advanced"}


# ─── FEATURE BUILDER (must match train_ml.engineer_features) ─────────────────

def build_features(
    age: int, weight_kg: float, height_cm: float,
    activity_level: int, gender: int,
    calories_tdee: Optional[float] = None,
) -> pd.DataFrame:
    h_m = height_cm / 100
    bmi = weight_kg / (h_m ** 2)
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + (5 if gender == 1 else -161)
    row = {
        "age": age, "weight_kg": weight_kg, "height_cm": height_cm,
        "activity_level": activity_level, "gender": gender,
        "bmi": bmi, "bmr": bmr,
        "age_squared": age ** 2,
        "weight_height": weight_kg / height_cm,
    }
    if calories_tdee is not None:
        row["calories_tdee"] = calories_tdee
    else:
        mult = {1: 1.2, 2: 1.375, 3: 1.55, 4: 1.725, 5: 1.9}
        row["calories_tdee"] = bmr * mult.get(int(activity_level), 1.375)
    return pd.DataFrame([row])


# ─── SERVICE ──────────────────────────────────────────────────────────────────

class MLService:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.calorie     = self._load("calorie_regression.pkl")
        self.weight      = self._load("weight_regression.pkl")
        self.fitness_clf = self._load("fitness_classifier.pkl")
        self.labels      = self._load("fitness_classifier_labels.pkl") or FALLBACK_LABELS
        self.recommender = self._load("recommender.pkl")
        self.exp = get_explanation_service()
        logger.info(
            f"MLService · calorie={bool(self.calorie)} weight={bool(self.weight)} "
            f"fitness_clf={bool(self.fitness_clf)} recommender={bool(self.recommender)}"
        )

    def _load(self, name: str):
        p = self.model_dir / name
        if not p.exists():
            logger.warning(f"  · missing {name} — using heuristic fallback")
            return None
        try:
            return joblib.load(p)
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            return None

    # ── calorie target ───────────────────────────────────────────────────────

    def predict_calories(self, age: int, weight_kg: float, height_cm: float,
                         activity_level: int, gender: int) -> dict:
        X_full = build_features(age, weight_kg, height_cm, activity_level, gender)
        X = X_full[FEATURE_COLUMNS]                              # exact 9 cols
        feature_row = X.iloc[0].to_dict()

        if self.calorie is None:
            v = float(X_full["calories_tdee"].iloc[0])
            return {
                "calories": round(v),
                "source":   "heuristic_tdee",
                "explanation": self.exp.explain_calories(None, feature_row, v),
            }
        try:
            pred = float(self.calorie.predict(X)[0])
            return {
                "calories": round(pred),
                "source":   "ml_ridge",
                "explanation": self.exp.explain_calories(self.calorie, feature_row, pred),
            }
        except Exception as e:
            logger.warning(f"Calorie prediction fell back: {e}")
            v = float(X_full["calories_tdee"].iloc[0])
            return {
                "calories": round(v),
                "source":   "heuristic_fallback",
                "explanation": self.exp.explain_calories(None, feature_row, v),
            }

    # ── 30-day weight change ─────────────────────────────────────────────────

    def predict_weight_change(self, age: int, weight_kg: float, height_cm: float,
                              activity_level: int, gender: int,
                              calories_tdee: Optional[float] = None) -> dict:
        X_full = build_features(age, weight_kg, height_cm, activity_level, gender, calories_tdee)
        # weight model expects all 10 columns including calories_tdee
        X = X_full[FEATURE_COLUMNS + ["calories_tdee"]]
        tdee = float(X_full["calories_tdee"].iloc[0])
        feature_row = X.iloc[0].to_dict()

        if self.weight is None:
            fallback = round((tdee - 2000) / 7700 * 30, 2)
            return {
                "weight_change_kg_30d": fallback,
                "source": "heuristic",
                "explanation": self.exp.explain_weight_change(None, feature_row, fallback),
            }
        try:
            pred = float(self.weight.predict(X)[0])
            return {
                "weight_change_kg_30d": round(pred, 2),
                "source": "ml_gbr",
                "explanation": self.exp.explain_weight_change(self.weight, feature_row, pred),
            }
        except Exception as e:
            logger.warning(f"Weight prediction fell back: {e}")
            fallback = round((tdee - 2000) / 7700 * 30, 2)
            return {
                "weight_change_kg_30d": fallback,
                "source": "heuristic_fallback",
                "explanation": self.exp.explain_weight_change(None, feature_row, fallback),
            }

    # ── fitness level classifier ─────────────────────────────────────────────

    FITNESS_FEATURES = ["age", "weight_kg", "height_cm", "activity_level", "gender"]

    def classify_fitness(self, age: int, weight_kg: float, height_cm: float,
                         activity_level: int, gender: int) -> dict:
        X_full = build_features(age, weight_kg, height_cm, activity_level, gender)
        # The fitness classifier was trained on the 5 raw inputs only
        X = X_full[self.FITNESS_FEATURES]
        feature_row = X.iloc[0].to_dict()

        if self.fitness_clf is None:
            score = activity_level * 20 + (10 if age < 35 else 0)
            lvl = 0 if score < 30 else (1 if score < 60 else 2)
            probs_arr = [0.0, 0.0, 0.0]; probs_arr[lvl] = 1.0
            confidence = 1.0
        else:
            try:
                lvl = int(self.fitness_clf.predict(X)[0])
                last = self.fitness_clf[-1] if hasattr(self.fitness_clf, "steps") else self.fitness_clf
                if hasattr(last, "predict_proba"):
                    raw = self.fitness_clf.predict_proba(X)[0]
                    probs_arr = [float(v) for v in raw]
                    confidence = float(max(raw))
                else:
                    probs_arr = [0.0, 0.0, 0.0]; probs_arr[lvl] = 1.0
                    confidence = 1.0
            except Exception as e:
                logger.warning(f"Fitness classification fell back: {e}")
                score = activity_level * 20 + (10 if age < 35 else 0)
                lvl = 0 if score < 30 else (1 if score < 60 else 2)
                probs_arr = [0.0, 0.0, 0.0]; probs_arr[lvl] = 1.0
                confidence = 1.0

        # If labels is a dict {0:"Beginner",...} or list ["Beginner",...]
        def _lbl(i: int) -> str:
            if isinstance(self.labels, dict):
                return self.labels.get(i, FALLBACK_LABELS.get(i, "Unknown"))
            try:
                return self.labels[i]
            except Exception:
                return FALLBACK_LABELS.get(i, "Unknown")

        level_name = _lbl(lvl)
        probs_named = {_lbl(i): round(p, 3) for i, p in enumerate(probs_arr)}

        return {
            "level_id":      lvl,
            "level_name":    level_name,
            "probabilities": probs_named,
            "confidence":    round(confidence, 3),
            "explanation":   self.exp.explain_fitness(self.fitness_clf, feature_row, level_name, confidence),
        }

    # ── recommendations ──────────────────────────────────────────────────────

    DEFAULTS = {
        0: ["Push-ups", "Squats", "Plank", "Lunges", "Burpees"],
        1: ["Bench Press", "Deadlifts", "Pull-ups", "Barbell Row", "Overhead Press"],
        2: ["Deadlifts", "Back Squat", "Weighted Pull-ups", "Bench Press", "OHP"],
    }

    def recommend_exercises(
        self,
        level_id: int,
        top_k: int = 5,
        profile: Optional[dict] = None,
    ) -> list[dict]:
        """Personalise to the user when a profile is supplied: filter by goal
        + dietary habits + pull from the trained recommender's catalog when
        available, otherwise from DEFAULTS."""
        catalog: list[dict]
        if isinstance(self.recommender, dict):
            raw = self.recommender.get(level_id, self.recommender.get(1, []))
            catalog = [r if isinstance(r, dict) else {"exercise": str(r)} for r in raw]
        else:
            catalog = [
                {"exercise": e, "muscle_group": "—", "difficulty": level_id + 1}
                for e in self.DEFAULTS.get(level_id, self.DEFAULTS[1])
            ]

        # Light personalisation: tag rationale per item using the user's goal.
        goal = (profile or {}).get("goal", "").lower() if profile else ""

        out: list[dict] = []
        for item in catalog[: top_k * 2]:    # take a buffer in case we filter
            ex = item.get("exercise", "")
            rationale = self._exercise_rationale(ex, level_id, goal)
            out.append({**item, "rationale": rationale})
            if len(out) >= top_k:
                break
        return out[:top_k]

    @staticmethod
    def _exercise_rationale(exercise: str, level_id: int, goal: str) -> str:
        """One-line plain-English reason this exercise is suggested."""
        e = exercise.lower()
        is_compound = any(k in e for k in ("squat", "deadlift", "bench", "row", "press", "pull-up", "pull up"))
        if goal in ("lose", "cut", "fat_loss", "fat loss"):
            if is_compound:
                return "high calorie burn from compound movement — efficient for fat loss"
            return "great for accumulating volume on a calorie deficit"
        if goal in ("gain", "bulk", "build", "muscle_gain"):
            if is_compound:
                return "primary mass-builder; recruits the most muscle per set"
            return "good accessory to round out a hypertrophy block"
        if level_id == 0:
            return "fundamental movement to master before adding load"
        return "matches your current fitness level for steady progression"


@lru_cache(maxsize=1)
def get_ml_service() -> MLService:
    return MLService(settings().ML_MODEL_DIR)
