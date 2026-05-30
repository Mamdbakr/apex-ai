from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock
from typing import Dict, Optional


@dataclass
class UserProfile:
    user_id: str
    age: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    gender: Optional[str] = None
    goal: Optional[str] = None
    activity_level: Optional[str] = None


class SessionMemory:
    """In-memory user profile store for session-scoped personalization."""

    def __init__(self) -> None:
        self._profiles: Dict[str, UserProfile] = {}
        self._lock = Lock()

    def upsert_profile(self, user_id: str, payload: dict) -> UserProfile:
        with self._lock:
            prof = self._profiles.get(user_id) or UserProfile(user_id=user_id)
            for key in ("age", "weight", "height", "gender", "goal", "activity_level"):
                if payload.get(key) is not None:
                    setattr(prof, key, payload[key])
            self._profiles[user_id] = prof
            return prof

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        with self._lock:
            return self._profiles.get(user_id)

    def get_profile_dict(self, user_id: str) -> dict:
        prof = self.get_profile(user_id)
        return asdict(prof) if prof else {}
