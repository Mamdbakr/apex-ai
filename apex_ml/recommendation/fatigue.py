"""
Fatigue and recovery modeling.

We use a simplified acute-vs-chronic workload model adapted from the
sports-science literature: the ratio of last-7-days load to last-28-days
load is a known predictor of overtraining injury risk. Values:

    < 0.8 :  detrained — ramping up safely is fine
    0.8–1.3: optimal load zone
    1.3–1.5: high load — caution
    > 1.5  : overload — recommend deload

We also produce a 0..1 "readiness" score combining workload ratio with
days-since-last-session and self-reported difficulty trend.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .profile import UserProfile, WorkoutSession


def session_load(s: WorkoutSession) -> float:
    """Heuristic load = sum(reps * max(1, weight)) * difficulty multiplier."""
    base = 0.0
    for st in s.sets:
        if st.completed:
            base += st.reps * max(1.0, st.weight_kg)
    if s.perceived_difficulty is not None:
        base *= (s.perceived_difficulty / 5.0)
    return base


@dataclass
class RecoveryState:
    acute_load: float          # last 7 days
    chronic_load: float        # last 28 days (avg per 7d)
    ratio: float               # acute / chronic
    readiness: float           # 0..1
    overtraining_risk: float   # 0..1
    days_since_last: Optional[int]
    recommendation: str        # short text


def estimate_recovery(profile: UserProfile,
                       now: Optional[datetime] = None) -> RecoveryState:
    """Compute the user's current recovery state."""
    now = now or datetime.now(timezone.utc)

    last7 = profile.sessions_in_window(7)
    last28 = profile.sessions_in_window(28)

    acute = float(sum(session_load(s) for s in last7))
    # Express chronic as a *weekly-equivalent* by dividing by 4
    chronic = float(sum(session_load(s) for s in last28)) / 4.0

    if chronic < 1e-6:
        ratio = 1.0 if acute < 1e-6 else 2.0
    else:
        ratio = acute / chronic

    days_since = None
    if profile.sessions:
        last_t = profile.sessions[-1].timestamp
        days_since = max(0, (now - last_t).days)

    # Readiness: ratio in optimal band + at least one rest day = high
    if ratio < 0.8:
        band_score = 0.9
    elif ratio <= 1.3:
        band_score = 1.0
    elif ratio <= 1.5:
        band_score = 0.6
    else:
        band_score = 0.3
    rest_score = 1.0 if (days_since or 0) >= 1 else 0.5
    readiness = float(np.clip(0.7 * band_score + 0.3 * rest_score, 0.0, 1.0))

    overtraining_risk = float(np.clip((ratio - 1.3) / 0.7, 0.0, 1.0))

    if ratio > 1.5:
        rec = "Schedule a deload — your recent training load is high."
    elif ratio < 0.7 and (days_since or 0) > 5:
        rec = "You've been resting — start back with a moderate-intensity session."
    elif (days_since or 0) == 0:
        rec = "Trained today already — recovery work or light mobility is best."
    else:
        rec = "Recovery looks good — proceed with planned training."

    return RecoveryState(
        acute_load=acute, chronic_load=chronic, ratio=ratio,
        readiness=readiness, overtraining_risk=overtraining_risk,
        days_since_last=days_since, recommendation=rec,
    )
