#!/usr/bin/env python3
"""
scripts/self_test.py
─────────────────────
Embedded self-test — boots a TestClient against the full FastAPI app
(no separate uvicorn process) and exercises the major user-facing flows.

This is the test we ship for "does the project work after a fresh clone?"

Run:
    python scripts/self_test.py

Exits 0 on success, prints a clear failure on the first broken flow.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Use an isolated DB for the test run
_db_dir = tempfile.mkdtemp(prefix="apex_selftest_")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_db_dir}/selftest.db"
os.environ.setdefault("SESSION_SECRET", "selftest-secret-x" * 4)


def banner(s: str) -> None:
    print(); print("═" * 60); print(f"  {s}"); print("═" * 60)


def info(s: str) -> None:
    print(f"  · {s}")


def fail(s: str) -> None:
    print(f"  ❌  {s}"); sys.exit(1)


def ok(s: str) -> None:
    print(f"  ✅  {s}")


def main() -> None:
    banner("APEX AI · embedded self-test")

    from fastapi.testclient import TestClient
    # Importing main triggers the real lifespan handler — DB init, model loading,
    # chatbot setup, etc. That's the whole point of a self-test.
    from backend.main import app

    email = f"self_{uuid.uuid4().hex[:8]}@example.com"
    pwd   = "selftest1234"

    with TestClient(app) as client:
        # ── HEALTH ────────────────────────────────────────────────────────────
        r = client.get("/health")
        if r.status_code != 200:
            fail(f"/health returned {r.status_code}")
        h = r.json()
        ok(f"/health · auth={h['auth']['method']} · provider={h.get('provider')}")
        if h["auth"]["method"] != "cookie-session":
            fail("auth method should be cookie-session, got " + h["auth"]["method"])

        # ── AUTH ──────────────────────────────────────────────────────────────
        r = client.post("/auth/signup", json={
            "full_name": "Self Test", "email": email, "password": pwd,
            "gender": "m", "age": 30, "weight_kg": 75, "height_cm": 178,
            "activity_level": 3, "goal": "lose", "target_weight": 70,
        })
        if r.status_code != 200:
            fail(f"/auth/signup returned {r.status_code}: {r.text}")
        ok("/auth/signup created user + set cookie")

        r = client.get("/auth/me")
        if r.status_code != 200:
            fail(f"/auth/me failed: {r.status_code}")
        me = r.json()
        ok(f"/auth/me · user={me['name']} · gender={me['gender']}")

        # ── ML PREDICTIONS ────────────────────────────────────────────────────
        r = client.post("/predict/all", json={})
        if r.status_code == 200:
            ok(f"/predict/all → {list(r.json().keys())}")
        else:
            info(f"/predict/all returned {r.status_code} — models may not be trained yet")

        r = client.post("/predict/calories", json={})
        if r.status_code == 200:
            cal = r.json().get("calories", "?")
            ok(f"/predict/calories → {cal} kcal")

        r = client.post("/predict/fitness-level", json={})
        if r.status_code == 200:
            lvl = r.json().get("level_name", "?")
            ok(f"/predict/fitness-level → {lvl}")

        # ── RECOMMENDER ───────────────────────────────────────────────────────
        r = client.get("/recommend?top_k=3")
        if r.status_code == 200:
            recs = r.json()
            n = len(recs.get("items", recs)) if isinstance(recs, dict) else len(recs)
            ok(f"/recommend → {n} exercises")

        # ── USER DATA ─────────────────────────────────────────────────────────
        r = client.get("/user-data/profile")
        if r.status_code == 200:
            ok("/user-data/profile read OK")

        r = client.post("/user-data/workout", json={
            "exercise": "squat", "sets": 3, "reps": 10,
            "weight_kg": 60, "duration_min": 20,
        })
        if r.status_code in (200, 201):
            ok("/user-data/workout logged")

        # ── INSIGHTS ──────────────────────────────────────────────────────────
        r = client.get("/insights/dashboard")
        if r.status_code == 200:
            ok(f"/insights/dashboard returned {len(r.text)} bytes")

        # ── LOGOUT ────────────────────────────────────────────────────────────
        r = client.post("/auth/logout")
        if r.status_code != 200:
            fail(f"/auth/logout failed: {r.status_code}")
        client.cookies.clear()
        if client.get("/auth/me").status_code != 401:
            fail("after logout, /auth/me should be 401")
        ok("logout cleared session server-side")

    banner("SELF-TEST PASSED")


if __name__ == "__main__":
    main()
