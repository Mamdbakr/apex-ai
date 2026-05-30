"""
backend/routes/auth.py
────────────────────────
APEX AI v14 — Cookie-based authentication routes.

Public endpoints
  POST   /auth/signup        — create account, sets session cookie
  POST   /auth/signin        — log in,         sets session cookie
  POST   /auth/logout        — clear session   (this device)
  POST   /auth/logout-all    — kill every session for the user
  GET    /auth/me            — return current user + profile (cookie-auth)
  GET    /auth/sessions      — list this user's active sessions
  DELETE /auth/sessions/{id} — revoke a specific session
  GET    /auth/check-email   — public — is this email already registered?

The frontend never needs to read or send a token. It just calls the API
with `credentials: 'include'` and the browser handles the cookie.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.db import (
    Session as SessionRow, User, UserProfile, WeightLog, get_db,
)
from backend.middleware.auth_guard import (
    clear_session_cookie, get_current_user, hash_password, issue_session,
    password_needs_rehash, revoke_all_sessions, revoke_session,
    set_session_cookie, verify_password,
)


router = APIRouter(prefix="/auth", tags=["Authentication"])


# ─── SCHEMAS ─────────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    full_name:      str        = Field(..., min_length=1, max_length=80)
    email:          EmailStr
    password:       str        = Field(..., min_length=6, max_length=128)
    gender:         str        = "m"
    age:            int        = Field(default=25, ge=10, le=100)
    weight_kg:      float      = Field(default=70.0, ge=20, le=300)
    height_cm:      float      = Field(default=175.0, ge=100, le=250)
    activity_level: int        = Field(default=2, ge=1, le=5)
    goal:           str        = "lose"
    target_weight:  float      = Field(default=65.0, ge=20, le=300)
    dietary_pref:   str        = "No Restrictions"
    timeframe:      str        = "3-6 months"


class SigninRequest(BaseModel):
    email:    EmailStr
    password: str = Field(..., min_length=1, max_length=128)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _client_meta(request: Request) -> tuple[str, str]:
    ua = (request.headers.get("user-agent") or "")[:500]
    ip = request.client.host if request.client else ""
    return ua, ip


def _profile_dict(profile: Optional[UserProfile], gender: str) -> dict:
    if not profile:
        return {}
    return {
        "name":          profile.name,
        "age":           profile.age,
        "weight_kg":     profile.weight_kg,
        "height_cm":     profile.height_cm,
        "activity_level": profile.activity_level,
        "goal":          profile.goal,
        "target_weight": profile.target_weight,
        "gender":        gender,
        "dietary_pref":  profile.dietary_pref,
        "timeframe":     profile.timeframe,
    }


def _user_payload(user: User, profile: Optional[UserProfile]) -> dict:
    """Shape the response body so the frontend has everything in one round-trip."""
    return {
        "user_id":  user.id,
        "name":     user.full_name,
        "email":    user.email,
        "gender":   user.gender,        # 'm' or 'f' — drives the theme switch
        "role":     user.role,
        "profile":  _profile_dict(profile, user.gender),
    }


# ─── SIGNUP ──────────────────────────────────────────────────────────────────

@router.post("/signup")
async def signup(
    req: SignupRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    email = req.email.lower().strip()

    exists = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if exists:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        full_name=req.full_name.strip(),
        email=email,
        hashed_password=hash_password(req.password),
        gender=req.gender if req.gender in ("m", "f") else "m",
        role="user",
        is_active=True,
        created_at=datetime.utcnow(),
        last_login_at=datetime.utcnow(),
    )
    db.add(user)
    await db.flush()  # populates user.id

    profile = UserProfile(
        user_id=user.id,
        name=req.full_name.strip(),
        age=req.age,
        weight_kg=req.weight_kg,
        height_cm=req.height_cm,
        activity_level=req.activity_level,
        gender=0 if user.gender == "f" else 1,
        goal=req.goal,
        target_weight=req.target_weight,
        dietary_pref=req.dietary_pref,
        timeframe=req.timeframe,
    )
    db.add(profile)
    db.add(WeightLog(user_id=user.id, weight_kg=req.weight_kg, logged_at=datetime.utcnow()))

    ua, ip = _client_meta(request)
    sid = await issue_session(db, user.id, ua, ip)
    await db.commit()

    set_session_cookie(response, sid)

    first_name = req.full_name.split()[0] if req.full_name.strip() else "Athlete"
    return {
        "success": True,
        "message": f"Welcome to APEX AI, {first_name}!",
        **_user_payload(user, profile),
    }


# ─── SIGNIN ──────────────────────────────────────────────────────────────────

@router.post("/signin")
async def signin(
    req: SigninRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    email = req.email.lower().strip()
    user = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if not user or not user.is_active or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    # Transparent migration: re-hash any legacy SHA256 password to bcrypt while
    # we still have the plaintext.
    if password_needs_rehash(user.hashed_password):
        user.hashed_password = hash_password(req.password)

    user.last_login_at = datetime.utcnow()

    profile = (await db.execute(
        select(UserProfile).where(UserProfile.user_id == user.id)
    )).scalar_one_or_none()

    ua, ip = _client_meta(request)
    sid = await issue_session(db, user.id, ua, ip)
    await db.commit()

    set_session_cookie(response, sid)

    first_name = user.full_name.split()[0] if user.full_name else "Athlete"
    return {
        "success": True,
        "message": f"Welcome back, {first_name}!",
        **_user_payload(user, profile),
    }


# ─── LOGOUT ──────────────────────────────────────────────────────────────────

@router.post("/logout")
async def logout(
    response: Response,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Revoke the session that issued the current cookie."""
    sid = user.get("session_id")
    if sid:
        await revoke_session(db, sid, user["user_id"])
        await db.commit()
    clear_session_cookie(response)
    return {"success": True, "message": "Signed out"}


@router.post("/logout-all")
async def logout_all(
    response: Response,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Revoke every session for the current user (logout-all-devices)."""
    n = await revoke_all_sessions(db, user["user_id"])
    await db.commit()
    clear_session_cookie(response)
    return {"success": True, "sessions_revoked": n}


# ─── ME ──────────────────────────────────────────────────────────────────────

@router.get("/me")
async def me(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the current user + profile. The frontend calls this on app boot
    to know if the cookie is still valid."""
    profile = (await db.execute(
        select(UserProfile).where(UserProfile.user_id == user["user_id"])
    )).scalar_one_or_none()
    db_user = (await db.execute(select(User).where(User.id == user["user_id"]))).scalar_one_or_none()
    if not db_user:
        raise HTTPException(status_code=401, detail="User no longer exists")

    return {
        "authenticated": True,
        "user_id":  db_user.id,
        "name":     db_user.full_name,
        "email":    db_user.email,
        "gender":   db_user.gender,
        "role":     db_user.role,
        "session_id": user.get("session_id"),
        "profile":  _profile_dict(profile, db_user.gender),
    }


# ─── SESSIONS ────────────────────────────────────────────────────────────────

@router.get("/sessions")
async def list_sessions(
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    rows = (await db.execute(
        select(SessionRow).where(
            SessionRow.user_id == user["user_id"],
            SessionRow.revoked_at.is_(None),
        ).order_by(SessionRow.last_seen_at.desc())
    )).scalars().all()

    current_sid = user.get("session_id")
    return [
        {
            "session_id":   s.id,
            "user_agent":   s.user_agent,
            "ip_address":   s.ip_address,
            "created_at":   s.created_at.isoformat() if s.created_at else None,
            "last_seen_at": s.last_seen_at.isoformat() if s.last_seen_at else None,
            "expires_at":   s.expires_at.isoformat() if s.expires_at else None,
            "is_current":   s.id == current_sid,
        }
        for s in rows
    ]


@router.delete("/sessions/{session_id}")
async def revoke_one(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    ok = await revoke_session(db, session_id, user["user_id"])
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    await db.commit()
    return {"success": True, "session_id": session_id}


# ─── EMAIL CHECK (public) ────────────────────────────────────────────────────

@router.get("/check-email")
async def check_email(email: str, db: AsyncSession = Depends(get_db)):
    found = (await db.execute(
        select(User).where(User.email == email.lower().strip())
    )).scalar_one_or_none()
    return {"available": found is None}
