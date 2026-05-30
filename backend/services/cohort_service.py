"""
backend/services/cohort_service.py
────────────────────────────────────
Peer-comparison metrics. Given a user, finds similar users in the database
(matched on goal + age band + BMI band) and reports percentile rankings on
real metrics (workouts/week, form score, weight progress).

Privacy: only aggregates are returned — no peer user IDs, names, or emails.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from statistics import mean
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.db import UserProfile, WorkoutLog, WeightLog


def _bmi(weight_kg: float, height_cm: float) -> float:
    h = height_cm / 100
    return weight_kg / (h * h) if h > 0 else 0


def _age_band(age: int) -> tuple[int, int]:
    if age < 25: return (18, 25)
    if age < 35: return (25, 35)
    if age < 45: return (35, 45)
    if age < 55: return (45, 55)
    return (55, 100)


def _bmi_band(bmi: float) -> tuple[float, float]:
    if bmi < 18.5: return (10, 18.5)
    if bmi < 25:   return (18.5, 25)
    if bmi < 30:   return (25, 30)
    return (30, 60)


def _percentile_of(value: float, sample: list[float]) -> Optional[float]:
    """Return the percentile rank of `value` within `sample` (0-100)."""
    if not sample:
        return None
    below = sum(1 for s in sample if s < value)
    equal = sum(1 for s in sample if s == value)
    rank = (below + 0.5 * equal) / len(sample)
    return round(rank * 100, 1)


class CohortService:

    async def compare(self, db: AsyncSession, profile_dict: dict, user_id: int) -> dict:
        """
        Compares the user against peers on three real metrics:
          • workouts_per_week (last 30 days)
          • avg_form_score
          • weight_movement_30d
        """
        if not profile_dict.get("goal"):
            return {"available": False, "reason": "no_goal_set"}

        bmi = _bmi(float(profile_dict["weight_kg"]), float(profile_dict["height_cm"]))
        age_lo, age_hi = _age_band(int(profile_dict["age"]))
        bmi_lo, bmi_hi = _bmi_band(bmi)

        # 1. Find peers (excluding the current user)
        peer_q = (
            select(UserProfile)
            .where(UserProfile.user_id != user_id)
            .where(UserProfile.goal == profile_dict["goal"])
            .where(UserProfile.age >= age_lo, UserProfile.age < age_hi)
        )
        peers = (await db.execute(peer_q)).scalars().all()

        # Filter further by BMI band
        peer_ids = []
        for p in peers:
            try:
                pbmi = _bmi(float(p.weight_kg), float(p.height_cm))
                if bmi_lo <= pbmi < bmi_hi:
                    peer_ids.append(p.user_id)
            except Exception:
                continue

        if len(peer_ids) < 1:
            return {
                "available": False,
                "reason": "not_enough_peers",
                "cohort_size": len(peer_ids),
                "filters": {"goal": profile_dict["goal"],
                            "age_band": [age_lo, age_hi],
                            "bmi_band": [round(bmi_lo, 1), round(bmi_hi, 1)]},
            }

        cutoff_30d = datetime.utcnow() - timedelta(days=30)

        # 2. Workouts/week (last 30d) for user + peers
        my_count = (await db.execute(
            select(func.count(WorkoutLog.id))
            .where(WorkoutLog.user_id == user_id, WorkoutLog.logged_at >= cutoff_30d)
        )).scalar_one()
        my_wpw = my_count / 4.3

        peer_wpws = []
        for pid in peer_ids:
            c = (await db.execute(
                select(func.count(WorkoutLog.id))
                .where(WorkoutLog.user_id == pid, WorkoutLog.logged_at >= cutoff_30d)
            )).scalar_one()
            peer_wpws.append(c / 4.3)

        # 3. Avg form score
        my_form = (await db.execute(
            select(func.avg(WorkoutLog.form_score))
            .where(WorkoutLog.user_id == user_id, WorkoutLog.form_score > 0)
        )).scalar_one() or 0.0
        peer_forms = []
        for pid in peer_ids:
            v = (await db.execute(
                select(func.avg(WorkoutLog.form_score))
                .where(WorkoutLog.user_id == pid, WorkoutLog.form_score > 0)
            )).scalar_one()
            if v is not None and v > 0:
                peer_forms.append(float(v))

        # 4. Weight movement over last 30d
        async def _weight_movement(uid: int) -> Optional[float]:
            rows = (await db.execute(
                select(WeightLog).where(WeightLog.user_id == uid,
                                        WeightLog.logged_at >= cutoff_30d)
                .order_by(WeightLog.logged_at.asc())
            )).scalars().all()
            if len(rows) < 2:
                return None
            return float(rows[-1].weight_kg - rows[0].weight_kg)

        my_move = await _weight_movement(user_id)
        peer_moves = []
        for pid in peer_ids:
            m = await _weight_movement(pid)
            if m is not None:
                peer_moves.append(m)

        # 5. Build comparison
        comparison = {
            "available": True,
            "cohort_size": len(peer_ids),
            "filters": {
                "goal": profile_dict["goal"],
                "age_band": [age_lo, age_hi],
                "bmi_band": [round(bmi_lo, 1), round(bmi_hi, 1)],
            },
            "you_vs_cohort": {
                "workouts_per_week": {
                    "you": round(my_wpw, 1),
                    "cohort_avg": round(mean(peer_wpws), 1) if peer_wpws else None,
                    "your_percentile": _percentile_of(my_wpw, peer_wpws),
                },
                "avg_form_score": {
                    "you": round(float(my_form), 3) if my_form else None,
                    "cohort_avg": round(mean(peer_forms), 3) if peer_forms else None,
                    "your_percentile": _percentile_of(float(my_form), peer_forms) if my_form else None,
                },
                "weight_change_30d_kg": {
                    "you": round(my_move, 2) if my_move is not None else None,
                    "cohort_avg": round(mean(peer_moves), 2) if peer_moves else None,
                    "your_percentile": _percentile_of(my_move, peer_moves) if my_move is not None else None,
                },
            },
        }
        return comparison


_singleton: Optional[CohortService] = None


def get_cohort_service() -> CohortService:
    global _singleton
    if _singleton is None:
        _singleton = CohortService()
    return _singleton
