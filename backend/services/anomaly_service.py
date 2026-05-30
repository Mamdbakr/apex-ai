"""
backend/services/anomaly_service.py
─────────────────────────────────────
Detects anomalies in a user's real history. Pure statistics — no random
fillers.

Detected anomalies:
  • weight_jump        — single weight log differs from rolling median by > 3σ
  • weight_stagnation  — < 0.1 kg movement over the last 21 days while user has
                         a non-maintain goal
  • form_decline       — 7-day mean form score has dropped > 15 percentile points
                         vs the user's all-time mean
  • workout_drop       — last 7 days has < 30% of the user's 28-day workout rate
  • volume_spike       — last week's training volume > 2× preceding 4-week mean

All thresholds are documented and conservative.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import mean, median, pstdev
from typing import Iterable, Optional


@dataclass
class Anomaly:
    code: str
    severity: str          # "info" | "warn" | "alert"
    title: str
    detail: str
    metric: dict           # numeric evidence behind the alert

    def to_dict(self) -> dict:
        return {
            "code": self.code, "severity": self.severity,
            "title": self.title, "detail": self.detail,
            "metric": self.metric,
        }


def _to_dt(v) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00").split("+")[0])
    except Exception:
        return None


class AnomalyService:

    # ── weight: outlier jump ─────────────────────────────────────────────────

    @staticmethod
    def _weight_jump(weights: list[dict]) -> Optional[Anomaly]:
        if len(weights) < 5:
            return None
        kgs = [float(w["weight_kg"]) for w in weights[-15:]]
        if not kgs:
            return None
        m = median(kgs[:-1])
        sd = pstdev(kgs[:-1]) or 0.5
        latest = kgs[-1]
        delta = latest - m
        if abs(delta) > 3 * sd and abs(delta) > 1.0:
            direction = "spike" if delta > 0 else "drop"
            return Anomaly(
                code="weight_jump",
                severity="warn",
                title=f"Unusual weight {direction} detected",
                detail=(
                    f"Your latest weight ({latest:.1f} kg) is {abs(delta):.1f} kg "
                    f"away from the median of your recent logs. This could be normal "
                    f"variation (hydration, time of day) or a sign to recheck the scale."
                ),
                metric={"latest": round(latest, 1), "median_recent": round(m, 1),
                        "std": round(sd, 2), "delta": round(delta, 2)},
            )
        return None

    # ── weight: stagnation despite cut/build goal ────────────────────────────

    @staticmethod
    def _weight_stagnation(weights: list[dict], goal: str) -> Optional[Anomaly]:
        if (goal or "").lower() not in ("lose", "cut", "gain", "bulk", "build"):
            return None
        if len(weights) < 4:
            return None
        cutoff = datetime.utcnow() - timedelta(days=21)
        recent = [w for w in weights
                  if (_to_dt(w.get("logged_at")) or datetime.min) >= cutoff]
        if len(recent) < 4:
            return None
        kgs = [float(w["weight_kg"]) for w in recent]
        movement = abs(kgs[-1] - kgs[0])
        if movement < 0.1:
            return Anomaly(
                code="weight_stagnation",
                severity="info",
                title="3-week plateau detected",
                detail=(
                    f"Your weight has barely moved in 21 days while your goal is "
                    f"'{goal}'. Consider revisiting calorie targets or training intensity."
                ),
                metric={"movement_kg": round(movement, 2), "days": 21},
            )
        return None

    # ── form: decline in moving mean ─────────────────────────────────────────

    @staticmethod
    def _form_decline(workouts: list[dict]) -> Optional[Anomaly]:
        if len(workouts) < 6:
            return None
        scores = [(w.get("form_score") or 0) for w in workouts if w.get("form_score") is not None]
        scores = [s for s in scores if s > 0]
        if len(scores) < 6:
            return None
        cutoff = datetime.utcnow() - timedelta(days=7)
        recent = [(w.get("form_score") or 0) for w in workouts
                  if (_to_dt(w.get("logged_at")) or datetime.min) >= cutoff
                  and w.get("form_score")]
        if len(recent) < 3:
            return None
        all_mean = mean(scores)
        recent_mean = mean(recent)
        delta_pct_pts = (all_mean - recent_mean) * 100
        if delta_pct_pts > 15:
            return Anomaly(
                code="form_decline",
                severity="warn",
                title="Form quality is dropping",
                detail=(
                    f"Your 7-day average form score ({recent_mean:.0%}) is "
                    f"{delta_pct_pts:.0f} points below your all-time average "
                    f"({all_mean:.0%}). Drop weight 10% and focus on tempo."
                ),
                metric={"recent_mean": round(recent_mean, 2),
                        "all_mean": round(all_mean, 2),
                        "delta_pct": round(delta_pct_pts, 1)},
            )
        return None

    # ── activity: weekly drop ────────────────────────────────────────────────

    @staticmethod
    def _workout_drop(workouts: list[dict]) -> Optional[Anomaly]:
        if not workouts:
            return None
        now = datetime.utcnow()
        last_7 = sum(1 for w in workouts
                     if (_to_dt(w.get("logged_at")) or datetime.min) >= now - timedelta(days=7))
        last_28 = sum(1 for w in workouts
                      if (_to_dt(w.get("logged_at")) or datetime.min) >= now - timedelta(days=28))
        rate_28 = last_28 / 4.0  # weekly average
        if rate_28 >= 1.0 and last_7 < rate_28 * 0.3:
            return Anomaly(
                code="workout_drop",
                severity="alert",
                title="You're behind your usual training pace",
                detail=(
                    f"You did {last_7} workouts in the last 7 days vs your "
                    f"4-week average of {rate_28:.1f}/week. Even a 20-minute session today "
                    f"keeps the streak intact."
                ),
                metric={"last_7": last_7, "weekly_avg_28d": round(rate_28, 1)},
            )
        return None

    # ── activity: volume spike ───────────────────────────────────────────────

    @staticmethod
    def _volume_spike(workouts: list[dict]) -> Optional[Anomaly]:
        if len(workouts) < 8:
            return None
        now = datetime.utcnow()
        last_week = [w for w in workouts
                     if (_to_dt(w.get("logged_at")) or datetime.min) >= now - timedelta(days=7)]
        prev_4w = [w for w in workouts
                   if now - timedelta(days=35) <= (_to_dt(w.get("logged_at")) or datetime.min)
                   < now - timedelta(days=7)]
        if not prev_4w:
            return None

        def vol(w):
            return (w.get("sets") or 0) * (w.get("reps") or 0) * (w.get("weight_kg") or 0)

        vol_lw = sum(vol(w) for w in last_week)
        vol_prev_avg = sum(vol(w) for w in prev_4w) / 4.0
        if vol_prev_avg > 0 and vol_lw > 2 * vol_prev_avg:
            return Anomaly(
                code="volume_spike",
                severity="warn",
                title="Sudden training volume spike",
                detail=(
                    f"This week's volume ({vol_lw:.0f} kg·reps) is more than 2× "
                    f"your 4-week average ({vol_prev_avg:.0f}). Watch for fatigue and recovery."
                ),
                metric={"volume_last_7d": round(vol_lw),
                        "weekly_avg_prev_4w": round(vol_prev_avg)},
            )
        return None

    # ── public ───────────────────────────────────────────────────────────────

    def detect(self, profile: dict, workouts: list[dict],
               weights: list[dict]) -> list[dict]:
        weights_sorted = sorted(weights, key=lambda w: _to_dt(w.get("logged_at")) or datetime.min)
        workouts_sorted = sorted(workouts, key=lambda w: _to_dt(w.get("logged_at")) or datetime.min)
        results: list[Anomaly] = []
        for fn, args in [
            (self._weight_jump, (weights_sorted,)),
            (self._weight_stagnation, (weights_sorted, profile.get("goal", ""))),
            (self._form_decline, (workouts_sorted,)),
            (self._workout_drop, (workouts_sorted,)),
            (self._volume_spike, (workouts_sorted,)),
        ]:
            try:
                a = fn(*args)
                if a:
                    results.append(a)
            except Exception:
                continue
        return [a.to_dict() for a in results]


_singleton: Optional[AnomalyService] = None


def get_anomaly_service() -> AnomalyService:
    global _singleton
    if _singleton is None:
        _singleton = AnomalyService()
    return _singleton
