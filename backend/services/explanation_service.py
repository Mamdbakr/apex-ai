"""
backend/services/explanation_service.py
─────────────────────────────────────────
Explainable AI layer.

Every prediction routed through ml_service can be paired with a human-readable
explanation built from the actual feature inputs and (where available) the
trained model's feature importances. The output is a small JSON dict that the
frontend renders next to each prediction:

    {
      "headline":    "You may experience fatigue because workout intensity rose 30%.",
      "factors":     [{"feature": "...", "value": ..., "impact": "+0.42",
                        "direction": "up", "explanation": "..."}],
      "method":      "feature_importance" | "permutation" | "rule_based",
      "model":       "GradientBoostingRegressor",
      "confidence":  0.87
    }

Why this matters
  Generic chatbots can sound smart without saying anything specific. By
  bundling the factor list with every numeric prediction we give the user
  a defensible reason — exactly what the upgrade brief calls out:
      "You may experience fatigue because your workout intensity
       increased by 30% in the last 3 days."
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


# ─── helpers ─────────────────────────────────────────────────────────────────

def _feature_importance_from_pipeline(model: Any) -> Optional[np.ndarray]:
    """Extract feature importances from a sklearn Pipeline or bare estimator.
    Returns None if the model doesn't expose them."""
    if model is None:
        return None
    est = model
    # Pipelines expose .steps and the last step's estimator
    try:
        if hasattr(model, "steps"):
            est = model.steps[-1][1]
    except Exception:
        pass
    for attr in ("feature_importances_", "coef_"):
        v = getattr(est, attr, None)
        if v is None:
            continue
        arr = np.asarray(v).ravel()
        if arr.size == 0:
            continue
        # Coef can be negative — use absolute magnitude for ranking
        return np.abs(arr).astype(float)
    return None


def _humanise_feature(name: str, value: float) -> str:
    """Map technical feature name → friendly noun phrase + value formatting."""
    n = name.lower()
    if n == "age":
        return f"age ({int(value)} y)"
    if n == "weight_kg":
        return f"weight ({value:.1f} kg)"
    if n == "height_cm":
        return f"height ({value:.0f} cm)"
    if n == "activity_level":
        labels = {1: "sedentary", 2: "light", 3: "moderate", 4: "active", 5: "very active"}
        return f"activity level ({labels.get(int(value), value)})"
    if n == "gender":
        return "biological sex (male)" if int(value) == 1 else "biological sex (female)"
    if n == "bmi":
        return f"BMI ({value:.1f})"
    if n == "bmr":
        return f"basal metabolic rate ({value:.0f} kcal)"
    if n == "calories_tdee":
        return f"daily energy need ({value:.0f} kcal)"
    if n in ("age_squared", "weight_height"):
        return name.replace("_", " ")
    return name


def _factor_from_value(name: str, value: float) -> str:
    """Specific phrase about *why* this feature pushes the prediction."""
    n = name.lower()
    if n == "weight_kg":
        if value >= 90:  return "higher body weight raises calorie need and weight-loss potential"
        if value <= 55:  return "lower body weight reduces basal calorie burn"
        return "weight is in a typical range"
    if n == "age":
        if value >= 50:  return "metabolism gradually slows after age 50"
        if value <= 25:  return "youth supports faster recovery and energy turnover"
        return "age is in the prime adult range"
    if n == "activity_level":
        if value >= 4:   return "your high activity level adds 60–90% on top of resting calories"
        if value <= 2:   return "low movement keeps daily calorie burn close to BMR"
        return "moderate activity multiplies BMR by ~1.5×"
    if n == "bmi":
        if value >= 30:  return "BMI in the obese range increases potential weight-loss rate"
        if value >= 25:  return "BMI in the overweight range"
        if value < 18.5: return "BMI is low — weight gain is encouraged"
        return "BMI is in the healthy range"
    if n == "gender":
        return "male physiology adds ~166 kcal to BMR vs female" if int(value) == 1 \
               else "female physiology has a lower BMR baseline"
    if n == "height_cm":
        return "taller frame raises BMR by 6.25 kcal per cm"
    if n == "bmr":
        return f"resting calorie burn is {value:.0f} kcal/day"
    if n == "calories_tdee":
        return f"current total daily energy ≈ {value:.0f} kcal"
    return f"{name} contributes to the prediction"


# ─── PUBLIC API ──────────────────────────────────────────────────────────────

class ExplanationService:
    """Builds explanations for any model + feature row pair."""

    @staticmethod
    def explain_prediction(
        *,
        model: Any,
        feature_names: list[str],
        feature_values: list[float],
        prediction: float,
        prediction_unit: str = "",
        prediction_label: str = "",
        top_k: int = 3,
    ) -> dict:
        """Generic explainer. Returns dict ready to ship to the frontend."""
        importances = _feature_importance_from_pipeline(model)

        # Score each feature: importance × normalised |value|
        # If no importances available, fall back to ranked-magnitude heuristic.
        n = len(feature_names)
        if importances is None or len(importances) != n:
            # fallback: use a fixed weight matching domain knowledge
            weights = np.array([
                {"age": 0.7, "weight_kg": 1.2, "height_cm": 0.8,
                 "activity_level": 1.1, "gender": 0.5,
                 "bmi": 1.0, "bmr": 0.9, "age_squared": 0.4,
                 "weight_height": 0.4, "calories_tdee": 1.0}.get(f.lower(), 0.5)
                for f in feature_names
            ])
            method = "rule_based"
        else:
            weights = importances
            method = "feature_importance"

        # Normalise weights so the leaderboard is interpretable
        w_norm = weights / (weights.sum() + 1e-9)

        factors = []
        for i, fname in enumerate(feature_names):
            fval = float(feature_values[i])
            impact = float(w_norm[i])
            direction = "up" if fval >= 0 else "down"
            factors.append({
                "feature": fname,
                "label":   _humanise_feature(fname, fval),
                "value":   round(fval, 3),
                "impact":  round(impact, 4),
                "direction": direction,
                "explanation": _factor_from_value(fname, fval),
            })
        factors.sort(key=lambda f: f["impact"], reverse=True)
        top = factors[: max(1, top_k)]

        # Build a single-line headline grounded in the top factor.
        head = top[0]
        verb = "predicted to" if prediction_label else "predicted at"
        if prediction_label:
            headline = (f"You're {verb} {prediction_label}"
                        f" because {head['explanation']}.")
        else:
            headline = (f"Your {prediction_unit} is {prediction:.2f}"
                        f" — driven mostly by {head['label']};"
                        f" {head['explanation']}.")

        return {
            "headline":  headline,
            "method":    method,
            "model":     type(model).__name__ if model is not None else "heuristic",
            "factors":   top,
            "all_factors_count": len(factors),
        }

    # ── domain-specific convenience wrappers ────────────────────────────────

    @staticmethod
    def explain_calories(model, feature_row: dict, prediction: float) -> dict:
        names  = list(feature_row.keys())
        values = list(feature_row.values())
        return ExplanationService.explain_prediction(
            model=model, feature_names=names, feature_values=values,
            prediction=prediction, prediction_unit="daily calorie target",
            prediction_label=f"about {round(prediction)} kcal/day",
            top_k=3,
        )

    @staticmethod
    def explain_weight_change(model, feature_row: dict, prediction: float) -> dict:
        names  = list(feature_row.keys())
        values = list(feature_row.values())
        direction = "lose" if prediction < 0 else "gain"
        return ExplanationService.explain_prediction(
            model=model, feature_names=names, feature_values=values,
            prediction=prediction, prediction_unit="30-day weight change",
            prediction_label=f"{direction} ~{abs(prediction):.1f} kg in 30 days",
            top_k=3,
        )

    @staticmethod
    def explain_fitness(model, feature_row: dict, level_name: str, confidence: float) -> dict:
        names  = list(feature_row.keys())
        values = list(feature_row.values())
        result = ExplanationService.explain_prediction(
            model=model, feature_names=names, feature_values=values,
            prediction=confidence, prediction_unit="fitness level",
            prediction_label=f"a {level_name.lower()} athlete",
            top_k=3,
        )
        result["confidence"] = round(float(confidence), 3)
        return result

    # ── risk / fatigue explanation (pure-rules) ─────────────────────────────

    @staticmethod
    def explain_fatigue_risk(
        intensity_change_pct: float,
        avg_form_score: float,
        workouts_last_3d: int,
        rest_days: int,
    ) -> dict:
        """Compose a fatigue-risk explanation grounded in real numbers."""
        reasons = []
        score = 0.0
        if intensity_change_pct >= 25:
            reasons.append({
                "feature": "intensity_change_pct",
                "label":   "training intensity",
                "value":   round(intensity_change_pct, 1),
                "impact":  0.45,
                "direction": "up",
                "explanation": (
                    f"workout intensity rose {intensity_change_pct:.0f}% "
                    f"in the last 3 days"
                ),
            })
            score += 0.45
        if workouts_last_3d >= 3 and rest_days == 0:
            reasons.append({
                "feature": "rest_days",
                "label":   "rest gap",
                "value":   rest_days,
                "impact":  0.30,
                "direction": "down",
                "explanation": "no rest day in the last 3 sessions",
            })
            score += 0.30
        if avg_form_score < 0.7 and avg_form_score > 0.0:
            reasons.append({
                "feature": "avg_form_score",
                "label":   "movement quality",
                "value":   round(avg_form_score, 2),
                "impact":  0.25,
                "direction": "down",
                "explanation": (
                    f"average form score {avg_form_score:.0%} "
                    f"is below the 70% target"
                ),
            })
            score += 0.25

        risk = "low"
        if score >= 0.6: risk = "high"
        elif score >= 0.3: risk = "moderate"

        if reasons:
            head = reasons[0]
            headline = (
                f"Fatigue risk is {risk} because {head['explanation']}."
            )
        else:
            headline = "Fatigue risk is low — training load looks balanced."

        return {
            "headline":  headline,
            "method":    "rule_based",
            "model":     "rule_engine_v1",
            "factors":   reasons,
            "risk":      risk,
            "score":     round(score, 2),
        }


_singleton: Optional[ExplanationService] = None


def get_explanation_service() -> ExplanationService:
    global _singleton
    if _singleton is None:
        _singleton = ExplanationService()
    return _singleton
