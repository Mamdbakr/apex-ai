"""
Exercise ranking model.

We rank candidate exercises for a user using two signals fused linearly:

    1. Content score — how well an exercise's tags (muscle group, equipment,
       difficulty, goal-fit) match the user's current state. Cheap and
       fully cold-start friendly.

    2. Collaborative score — cosine similarity between the user's
       preference vector and historical preference vectors of "similar"
       users. Falls back to zero when no peer data is available.

We deliberately avoid heavy dependencies. FAISS / LightGBM are optional;
the default implementation uses numpy. If FAISS is installed, the index
class accelerates the similarity step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .profile import EXERCISE_MUSCLES, UserProfile


# ============================================================ catalog
@dataclass
class ExerciseMeta:
    """Static metadata used by content scoring."""
    name: str
    muscles: List[str]
    equipment: List[str] = field(default_factory=lambda: ["bodyweight"])
    difficulty: int = 2                  # 1..5
    goal_fit: Dict[str, float] = field(default_factory=dict)
    # goal_fit example: {"strength": 1.0, "hypertrophy": 0.8, "endurance": 0.4}


DEFAULT_CATALOG: Dict[str, ExerciseMeta] = {
    "squat":          ExerciseMeta("squat",          ["quads","glutes","core"],            ["bodyweight","barbell","dumbbell"], 3, {"strength":1.0,"hypertrophy":0.9,"general_fitness":0.9}),
    "deadlift":       ExerciseMeta("deadlift",       ["hamstrings","back","glutes","core"],["barbell","dumbbell"],              4, {"strength":1.0,"hypertrophy":0.8}),
    "pushup":         ExerciseMeta("pushup",         ["chest","triceps","shoulders","core"],["bodyweight"],                     2, {"endurance":0.9,"general_fitness":1.0}),
    "bench_press":    ExerciseMeta("bench_press",    ["chest","triceps","shoulders"],      ["barbell","dumbbell"],              3, {"strength":1.0,"hypertrophy":1.0}),
    "overhead_press": ExerciseMeta("overhead_press", ["shoulders","triceps","core"],       ["barbell","dumbbell"],              3, {"strength":0.9,"hypertrophy":0.8}),
    "row":            ExerciseMeta("row",            ["back","biceps"],                    ["barbell","dumbbell","cable"],      2, {"strength":0.9,"hypertrophy":0.9}),
    "pullup":         ExerciseMeta("pullup",         ["back","biceps"],                    ["pullup_bar"],                      4, {"strength":1.0,"hypertrophy":0.8}),
    "bicep_curl":     ExerciseMeta("bicep_curl",     ["biceps"],                           ["dumbbell","barbell","cable"],      1, {"hypertrophy":0.9}),
    "tricep_ext":     ExerciseMeta("tricep_ext",     ["triceps"],                          ["dumbbell","cable"],                1, {"hypertrophy":0.8}),
    "lunge":          ExerciseMeta("lunge",          ["quads","glutes","hamstrings"],      ["bodyweight","dumbbell"],           2, {"strength":0.8,"hypertrophy":0.8,"general_fitness":0.9}),
    "plank":          ExerciseMeta("plank",          ["core"],                             ["bodyweight"],                      1, {"endurance":1.0,"general_fitness":0.9}),
}


# ============================================================ scoring
def _content_score(ex: ExerciseMeta, profile: UserProfile,
                   weak_muscles: List[str]) -> float:
    """Higher when the exercise hits weak muscles, matches goal & equipment."""
    score = 0.0

    # Muscle weakness boost (most important for adaptive variety)
    muscle_match = sum(1 for m in ex.muscles if m in weak_muscles)
    score += 0.5 * (muscle_match / max(len(ex.muscles), 1))

    # Goal fit
    score += 0.3 * ex.goal_fit.get(profile.goals.primary, 0.4)

    # Equipment availability — exercise's equipment field lists alternatives
    # where any ONE is sufficient (e.g. squat can be bodyweight OR barbell).
    # We treat 'bodyweight' as a special-case alternative meaning "no kit
    # required". An exercise like 'pullup' that *requires* a pullup_bar in
    # addition to bodyweight should list ['pullup_bar'] only.
    equip_ok = any(e in profile.goals.equipment for e in ex.equipment)
    if not equip_ok:
        return 0.0
    score += 0.1

    # Mild penalty if the user has been doing this exercise constantly
    prefs = profile.preference_scores(days=14)
    overuse = prefs.get(ex.name, 0.0)
    score -= 0.1 * overuse

    return float(max(0.0, score))


def _collab_score(target_vec: np.ndarray,
                   peer_vecs: np.ndarray,
                   peer_choices: List[Dict[str, float]],
                   exercise: str) -> float:
    """Cosine-weighted preference for `exercise` across similar peers."""
    if peer_vecs.size == 0:
        return 0.0
    # Normalize
    t_n = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    p_n = peer_vecs / (np.linalg.norm(peer_vecs, axis=1, keepdims=True) + 1e-9)
    sims = p_n @ t_n                                 # cosine similarities
    sims = np.clip(sims, 0.0, 1.0)
    if sims.sum() <= 1e-9:
        return 0.0
    weighted = float(sum(s * peer_choices[i].get(exercise, 0.0)
                         for i, s in enumerate(sims)))
    return weighted / float(sims.sum())


@dataclass
class RankedExercise:
    name: str
    score: float
    content_score: float
    collab_score: float
    reason: str


class ExerciseRanker:
    """Ranks the catalog for a given user.

    Parameters
    ----------
    catalog : dict of ExerciseMeta
        Defaults to DEFAULT_CATALOG. Extend freely.
    alpha : float
        Weight of the content score (1-alpha for collab).
    """

    def __init__(self, catalog: Optional[Dict[str, ExerciseMeta]] = None,
                 alpha: float = 0.7):
        self.catalog = dict(catalog or DEFAULT_CATALOG)
        self.alpha = alpha
        self._peer_vecs: np.ndarray = np.zeros((0, len(self.catalog)), dtype=np.float32)
        self._peer_prefs: List[Dict[str, float]] = []
        self._exercise_index = {name: i for i, name in enumerate(self.catalog)}

    # ------------------------- peer pool management (optional) -------------
    def update_peers(self, peer_profiles: List[UserProfile]) -> None:
        """Refresh the peer preference matrix for collaborative similarity."""
        vecs = []
        prefs = []
        for p in peer_profiles:
            pref = p.preference_scores(days=60)
            v = np.zeros(len(self.catalog), dtype=np.float32)
            for name, idx in self._exercise_index.items():
                v[idx] = pref.get(name, 0.0)
            vecs.append(v)
            prefs.append(pref)
        self._peer_vecs = np.stack(vecs) if vecs else np.zeros((0, len(self.catalog)), dtype=np.float32)
        self._peer_prefs = prefs

    # ------------------------- main entry point ----------------------------
    def rank(self, profile: UserProfile, top_k: int = 8) -> List[RankedExercise]:
        weak = profile.weak_muscles(days=14, top_k=4)
        prefs = profile.preference_scores(days=60)
        target_vec = np.zeros(len(self.catalog), dtype=np.float32)
        for name, idx in self._exercise_index.items():
            target_vec[idx] = prefs.get(name, 0.0)

        ranked: List[RankedExercise] = []
        for name, meta in self.catalog.items():
            c = _content_score(meta, profile, weak)
            # content_score == 0 means equipment-blocked; exclude entirely
            if c <= 0.0:
                continue
            v = _collab_score(target_vec, self._peer_vecs, self._peer_prefs, name)
            s = self.alpha * c + (1 - self.alpha) * v
            reason_bits = []
            if any(m in weak for m in meta.muscles):
                reason_bits.append("targets a recently under-trained muscle group")
            if meta.goal_fit.get(profile.goals.primary, 0) > 0.7:
                reason_bits.append(f"strong match for {profile.goals.primary}")
            if prefs.get(name, 0) > 0.2:
                reason_bits.append("you do this frequently")
            reason = "; ".join(reason_bits) or "good general-purpose choice"
            ranked.append(RankedExercise(
                name=name, score=s,
                content_score=c, collab_score=v,
                reason=reason,
            ))
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked[:top_k]
