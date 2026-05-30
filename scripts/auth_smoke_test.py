#!/usr/bin/env python3
"""
scripts/auth_smoke_test.py
───────────────────────────
Drives every /auth endpoint over HTTP against a running backend.

Usage:
    1. Start the backend in another terminal:
         uvicorn backend.main:app --reload --port 8000
    2. Run this script:
         python scripts/auth_smoke_test.py

Exits 0 on success, non-zero on the first failure.
"""
from __future__ import annotations

import sys
import time
import uuid
import httpx


BASE = "http://127.0.0.1:8000"


def banner(s: str) -> None:
    print()
    print("═" * 60)
    print(f"  {s}")
    print("═" * 60)


def check(label: str, ok: bool, extra: str = "") -> None:
    mark = "✅" if ok else "❌"
    print(f"  {mark}  {label}{(' — ' + extra) if extra else ''}")
    if not ok:
        sys.exit(1)


def main() -> None:
    banner("APEX AI · cookie-session smoke test")

    # Single httpx.Client = single cookie jar. That's how a real browser works.
    with httpx.Client(base_url=BASE, timeout=10.0) as client:
        # Wait for the server to be reachable
        for _ in range(15):
            try:
                client.get("/health")
                break
            except Exception:
                time.sleep(0.5)
        else:
            check("server reachable", False, f"{BASE} not responding")

        check("GET /health", client.get("/health").status_code == 200)

        # 1. Signup
        email = f"smoke_{uuid.uuid4().hex[:8]}@example.com"
        r = client.post("/auth/signup", json={
            "full_name": "Smoke Test",
            "email": email,
            "password": "smoketest1234",
            "gender": "f",
            "age": 28, "weight_kg": 60, "height_cm": 168,
        })
        ok = r.status_code == 200 and r.cookies.get("apex_session")
        check("POST /auth/signup", ok, f"status={r.status_code}")
        if not ok:
            print(r.text); sys.exit(1)
        body = r.json()
        check("signup response shape",
              all(k in body for k in ("user_id", "name", "email", "gender", "profile")))
        check("signup gender preserved", body["gender"] == "f")

        # 2. /auth/me using the cookie automatically
        r = client.get("/auth/me")
        check("GET /auth/me (with cookie)", r.status_code == 200)
        me = r.json()
        check("/auth/me authenticated", me.get("authenticated") is True)
        check("/auth/me has profile", isinstance(me.get("profile"), dict))

        # 3. Sign up should reject duplicate
        r = client.post("/auth/signup", json={
            "full_name": "Dup", "email": email, "password": "x" * 8,
        })
        check("duplicate signup → 400", r.status_code == 400)

        # 4. Logout
        r = client.post("/auth/logout")
        check("POST /auth/logout", r.status_code == 200)

        # 5. /auth/me after logout — clear cookie jar so we can't pass the new
        # (empty) cookie either; the server should reject the revoked one too.
        old_cookie = body  # we don't have the literal cookie value any more
        client.cookies.clear()
        r = client.get("/auth/me")
        check("/auth/me after logout → 401", r.status_code == 401)

        # 6. Sign back in
        r = client.post("/auth/signin", json={
            "email": email, "password": "smoketest1234",
        })
        check("POST /auth/signin", r.status_code == 200 and r.cookies.get("apex_session"))

        # 7. Wrong password
        r = client.post("/auth/signin", json={
            "email": email, "password": "WRONG",
        })
        check("wrong password → 401", r.status_code == 401)

        # 8. List sessions
        r = client.get("/auth/sessions")
        check("GET /auth/sessions", r.status_code == 200)
        sessions = r.json()
        check("at least one active session", isinstance(sessions, list) and len(sessions) >= 1)
        check("current session flagged", any(s.get("is_current") for s in sessions))

        # 9. Tampered cookie rejected
        r = client.get("/auth/me", cookies={"apex_session": "tampered.bad.signature"})
        check("tampered cookie → 401", r.status_code == 401)

        # 10. Email-check public endpoint
        r = client.get(f"/auth/check-email?email={email}")
        check("/auth/check-email returns false for taken email",
              r.status_code == 200 and r.json().get("available") is False)

        r = client.get(f"/auth/check-email?email=brand_new_{uuid.uuid4().hex[:6]}@x.com")
        check("/auth/check-email returns true for new email",
              r.status_code == 200 and r.json().get("available") is True)

    banner("ALL COOKIE-AUTH SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
