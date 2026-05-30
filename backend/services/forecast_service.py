"""
backend/services/forecast_service.py
──────────────────────────────────────
Multi-horizon predictions powered by the existing scikit-learn models in
ai_models/ml_models/ plus the user's actual workout/weight log history.

Public API:
    forecast_service.weight_curve(profile, weight_logs, days=90)
    forecast_service.calorie_curve(profile, days=14)
    forecast_service.timeline_to_goal(profile, weight_logs)

Every value returned is either:
  • a direct call to a trained model (regressor/classifier), or
  • a deterministic computation from real user data (weight log statistics,
    BMR/TDEE formulas).
There are no random fillers and no hardcoded forecast points.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from statistics import mean, pstdev
from typing import Optional

from loguru import logger

from backend.services.ml_service import get_ml_service


# Mifflin-St Jeor + activity multipliers (mirror of training/feature_eng)
_ACTIVITY_MULT = {1: 1.2, 2: 1.375, 3: 1.55, 4: 1.725, 5: 1.9}
_KCAL_PER_KG_FAT = 7700.0


@dataclass
class WeightObservation:
    when: datetime
    kg: float


def _bmr(weight_kg: float, height_cm: float, age: int, gender_int: int) -> float:
    return 10 * weight_kg + 6.25 * height_cm - 5 * age + (5 if gender_int == 1 else -161)


def _tdee(profile: dict) -> float:
    bmr = _bmr(profile["weight_kg"], profile["height_cm"], profile["age"], int(profile.get("gender", 1)))
    return bmr * _ACTIVITY_MULT.get(int(profile.get("activity_level", 2)), 1.375)


def _goal_calories(tdee: float, goal: str) -> float:
    g = (goal or "").lower()
    if g in ("lose", "cut", "fat_loss", "fat loss"):
        return max(tdee - 500, 1200)
    if g in ("gain", "bulk", "build", "muscle_gain"):
        return tdee + 300
    return tdee


def _observed_trend_kg_per_day(weights: list[WeightObservation]) -> Optional[float]:
    """Linear slope of weight vs time (kg/day) from real logs.
    Returns None if we don't have at least 2 points spanning ≥ 3 days, since a
    spurious spike over a few minutes is not a meaningful trend."""
    if len(weights) < 2:
        return None
    weights = sorted(weights, key=lambda w: w.when)
    span_days = (weights[-1].when - weights[0].when).total_seconds() / 86400.0
    if span_days < 3.0:
        return None
    t0 = weights[0].when
    xs = [(w.when - t0).total_seconds() / 86400.0 for w in weights]
    ys = [w.kg for w in weights]
    n = len(xs)
    mean_x, mean_y = mean(xs), mean(ys)
    num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    slope = num / den
    # Clamp to physiologically plausible: even crash diets cap around 0.2 kg/day
    return max(-0.2, min(0.2, slope))


class ForecastService:
    """Single point of truth for trend + horizon predictions."""

    # ── weight curve ─────────────────────────────────────────────────────────

    def weight_curve(self, profile: dict, weight_logs: list[dict], days: int = 90) -> dict:
        """
        Returns:
          {
            "horizon_days": [7, 14, 30, 60, 90],
            "points":      [{"day": 0, "kg": 80.0, "source": "observed"}, ...],
            "models_kg_change_30d": {"ml": -0.85, "energy_balance": -0.78},
            "trend_kg_per_day":    -0.028,
            "method":              "blended",
            "stability":           0.93,           # 1 - normalised stdev of ML rolling residuals
          }
        """
        if not all(profile.get(k) for k in ("weight_kg", "height_cm", "age")):
            return {"available": False, "reason": "incomplete_profile"}

        # 1. ML model 30-day weight change prediction (real model call)
        ml = get_ml_service()
        ml_resp = ml.predict_weight_change(
            age=int(profile["age"]),
            weight_kg=float(profile["weight_kg"]),
            height_cm=float(profile["height_cm"]),
            activity_level=int(profile.get("activity_level", 2)),
            gender=int(profile.get("gender", 1)),
        )
        ml_dkg_30 = float(ml_resp["weight_change_kg_30d"])
        ml_dkg_per_day = ml_dkg_30 / 30.0

        # 2. Energy-balance estimate from goal calories vs TDEE
        tdee = _tdee(profile)
        goal_cal = _goal_calories(tdee, profile.get("goal", "maintain"))
        eb_dkg_per_day = (goal_cal - tdee) / _KCAL_PER_KG_FAT
        eb_dkg_30 = eb_dkg_per_day * 30

        # 3. Observed trend from real weight logs
        observations = [
            WeightObservation(
                when=w["logged_at"] if isinstance(w["logged_at"], datetime)
                     else datetime.fromisoformat(str(w["logged_at"]).replace("Z", "+00:00").split("+")[0]),
                kg=float(w["weight_kg"]),
            )
            for w in weight_logs
            if w.get("weight_kg") is not None and w.get("logged_at")
        ]
        observed_dkg_per_day = _observed_trend_kg_per_day(observations)

        # 4. Blend strategy.
        #
        #    IMPORTANT: the ML weight-change regressor is GOAL-BLIND — it only
        #    sees age/weight/height/activity/gender, never the user's goal or
        #    calorie target. So on its own it will happily predict weight GAIN
        #    for someone who is actively cutting, which is physiologically
        #    wrong for their plan. The energy-balance estimate, by contrast, is
        #    derived directly from goal_calories vs TDEE and therefore always
        #    points in the direction the user intends.
        #
        #    Fix: when the user has set a DIRECTIONAL goal (lose / gain), the
        #    goal-aware signals must lead, and the goal-blind ML is only allowed
        #    to *modulate* the magnitude — never to flip the direction. When the
        #    goal is "maintain" (no intended direction) we fall back to the
        #    original ML-led blend, since there's no user intent to respect.
        goal_norm = (profile.get("goal") or "maintain").lower()
        is_losing  = goal_norm in ("lose", "cut", "fat_loss", "fat loss")
        is_gaining = goal_norm in ("gain", "bulk", "build", "muscle_gain")
        directional = is_losing or is_gaining

        if directional:
            # Energy balance leads (it respects the goal). Observed real data,
            # when we have enough of it, is the most trustworthy signal and is
            # also goal-independent (it's measured reality), so it gets the top
            # weight when available.
            if observed_dkg_per_day is not None and len(observations) >= 5:
                blended = (0.50 * observed_dkg_per_day
                           + 0.35 * eb_dkg_per_day
                           + 0.15 * ml_dkg_per_day)
                method = "observed+energy+ml(goal-aware)"
            else:
                blended = 0.75 * eb_dkg_per_day + 0.25 * ml_dkg_per_day
                method = "energy+ml(goal-aware)"

            # Guard rail: the goal-blind ML must not flip the intended direction.
            # If the user is cutting, the forecast cannot trend upward (and vice
            # versa). We allow a tiny epsilon so a genuine plateau still reads as
            # ~0 rather than being forced.
            eps = 1e-4
            if is_losing and blended > eps:
                blended = min(blended, eb_dkg_per_day)   # clamp toward the (negative) energy estimate
                blended = min(blended, 0.0)
                method += "+dir-clamped"
            elif is_gaining and blended < -eps:
                blended = max(blended, eb_dkg_per_day)   # clamp toward the (positive) energy estimate
                blended = max(blended, 0.0)
                method += "+dir-clamped"
        else:
            # Maintain (no directional intent) — original ML-led behaviour.
            if observed_dkg_per_day is not None and len(observations) >= 5:
                blended = 0.5 * observed_dkg_per_day + 0.3 * ml_dkg_per_day + 0.2 * eb_dkg_per_day
                method = "observed+ml+energy"
            else:
                blended = 0.6 * ml_dkg_per_day + 0.4 * eb_dkg_per_day
                method = "ml+energy"

        # Physiological clamp: ±0.15 kg/day is the absolute extreme in healthy adults
        blended = max(-0.15, min(0.15, blended))

        # 5. Build the curve. Anchor day 0 to last observed weight if available,
        #    otherwise current profile weight.
        anchor_kg = observations[-1].kg if observations else float(profile["weight_kg"])
        anchor_when = observations[-1].when if observations else datetime.utcnow()

        horizons = [0, 7, 14, 30, 60, 90][: 6 if days >= 90 else (days // 15 + 2)]
        if days not in horizons:
            horizons = sorted(set(horizons + [days]))
            horizons = [h for h in horizons if h <= days]

        points = []
        for h in horizons:
            kg = round(anchor_kg + blended * h, 2)
            label_date = (anchor_when + timedelta(days=h)).strftime("%b %d")
            points.append({
                "day": h,
                "date": label_date,
                "kg": kg,
                "source": "observed" if h == 0 and observations else "predicted",
            })

        # 6. Stability — how consistent are the signals that actually drive the
        #    forecast? The old version compared ML vs energy-balance directly,
        #    but since ML is goal-blind that gap is always huge for a directional
        #    goal (which is exactly why it was showing 0%). Instead we measure
        #    how far the final blended trend sits from its strongest goal-aware
        #    anchor: observed reality if we have it, else the energy estimate.
        #    A small gap → the forecast is well-supported → high stability.
        anchor_trend = (
            observed_dkg_per_day
            if (observed_dkg_per_day is not None and len(observations) >= 5)
            else eb_dkg_per_day
        )
        gap = abs(blended - anchor_trend)
        stability = round(max(0.0, 1.0 - gap / 0.05), 2)  # 0.05 kg/day gap = "very different"

        return {
            "available": True,
            "horizon_days": horizons,
            "points": points,
            "models_kg_change_30d": {
                "ml": round(ml_dkg_30, 2),
                "energy_balance": round(eb_dkg_30, 2),
                "observed": round((observed_dkg_per_day or 0) * 30, 2) if observed_dkg_per_day is not None else None,
            },
            "trend_kg_per_day": round(blended, 4),
            "trend_kg_per_30d": round(blended * 30, 2),
            "method": method,
            "stability": stability,
            "tdee": round(tdee),
            "calorie_target": round(goal_cal),
            "model_source": ml_resp.get("source", "unknown"),
        }

    # ── timeline to goal ─────────────────────────────────────────────────────

    def timeline_to_goal(self, profile: dict, weight_logs: list[dict]) -> dict:
        """Days until target_weight at current blended trend; None if no target/no trend."""
        target = profile.get("target_weight")
        if not target:
            return {"available": False, "reason": "no_target_weight"}

        curve = self.weight_curve(profile, weight_logs, days=30)
        if not curve.get("available"):
            return {"available": False, "reason": curve.get("reason", "no_data")}

        trend = curve["trend_kg_per_day"]
        anchor_kg = curve["points"][0]["kg"]
        gap = float(target) - anchor_kg

        if abs(trend) < 1e-6:
            return {"available": False, "reason": "trend_too_flat"}

        # Trend in same direction as gap?
        if (gap > 0 and trend <= 0) or (gap < 0 and trend >= 0):
            return {
                "available": False,
                "reason": "trend_wrong_direction",
                "trend_kg_per_day": trend,
                "gap_kg": round(gap, 2),
            }

        days = abs(gap / trend)
        target_date = (datetime.utcnow() + timedelta(days=days)).strftime("%Y-%m-%d")
        return {
            "available": True,
            "days_to_target": int(round(days)),
            "target_date": target_date,
            "gap_kg": round(gap, 2),
            "trend_kg_per_day": trend,
            "feasibility": (
                "realistic" if abs(trend) <= 0.03
                else "aggressive" if abs(trend) <= 0.06
                else "unsustainable"
            ),
        }

    # ── calorie series projection ────────────────────────────────────────────

    def calorie_curve(self, profile: dict, days: int = 14) -> dict:
        """Daily calorie target for the next N days. Steady at goal_calories — the
        model is goal-driven, not random."""
        if not all(profile.get(k) for k in ("weight_kg", "height_cm", "age")):
            return {"available": False}
        ml = get_ml_service()
        # Use the ML calorie regressor for the recommended target (real model call)
        cal = ml.predict_calories(
            age=int(profile["age"]),
            weight_kg=float(profile["weight_kg"]),
            height_cm=float(profile["height_cm"]),
            activity_level=int(profile.get("activity_level", 2)),
            gender=int(profile.get("gender", 1)),
        )
        target = float(cal["calories"])
        tdee = _tdee(profile)
        adjusted = _goal_calories(tdee, profile.get("goal", "maintain"))
        cal_source = cal.get("source", "unknown")

        # Choose the daily target.
        #
        # The previous logic took whichever value was "closer to TDEE", but that
        # silently discarded the goal deficit/surplus: when the ML model falls
        # back to a heuristic it just returns TDEE, so the goal-adjusted value
        # (e.g. TDEE-500 for a cut) is ALWAYS further from TDEE and was always
        # thrown away — showing maintenance calories to someone who's cutting.
        #
        # Correct behaviour: the goal-adjusted target is the source of truth,
        # because it's the only value that respects the user's goal. We only
        # prefer the ML number when it comes from a REAL trained model
        # (source "ml_*") AND the user is on "maintain" (no intended deficit/
        # surplus for the model to override).
        goal_norm = (profile.get("goal") or "maintain").lower()
        is_directional = goal_norm in (
            "lose", "cut", "fat_loss", "fat loss",
            "gain", "bulk", "build", "muscle_gain",
        )
        ml_is_real = isinstance(cal_source, str) and cal_source.startswith("ml")

        if is_directional:
            # Always honour the goal deficit/surplus.
            chosen = adjusted
            method = f"goal_adjusted (deficit/surplus from {cal_source})"
        elif ml_is_real:
            # Maintain + a real model → trust the model.
            chosen = target
            method = cal_source
        else:
            # Maintain + heuristic fallback → TDEE-based maintenance.
            chosen = adjusted
            method = cal_source

        today = date.today()
        return {
            "available": True,
            "daily_target": round(chosen),
            "tdee": round(tdee),
            "model_target": round(target),
            "method": method,
            "points": [
                {"date": (today + timedelta(days=i)).strftime("%b %d"), "calories": round(chosen)}
                for i in range(days)
            ],
        }


_singleton: Optional[ForecastService] = None


def get_forecast_service() -> ForecastService:
    global _singleton
    if _singleton is None:
        _singleton = ForecastService()
    return _singleton
