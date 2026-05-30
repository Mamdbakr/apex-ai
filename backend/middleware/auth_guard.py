"""
backend/middleware/auth_guard.py
─────────────────────────────────
APEX AI v14 — Cookie-based session authentication.

This module REPLACES the old JWT-based system entirely.

Why cookies + DB sessions instead of JWT?
  • No tokens to rotate, no refresh-token race conditions
  • Server-side revocation is instant (just delete the row)
  • No "token expired in the middle of a tab" bugs
  • Works automatically with `credentials: 'include'` on the frontend
  • No client-side storage of secrets — cookies are HttpOnly

Public surface (everything the routes import):
  hash_password(plain) -> str
  verify_password(plain, stored) -> bool
  password_needs_rehash(stored) -> bool
  issue_session(db, user_id, ua, ip) -> session_id (str)
  revoke_session(db, session_id, user_id) -> bool
  revoke_all_sessions(db, user_id) -> int
  get_current_user (FastAPI dep)         — required auth
  get_optional_user (FastAPI dep)        — auth-or-anonymous
  set_session_cookie(response, sid)      — sets the signed cookie
  clear_session_cookie(response)         — wipes the cookie
"""
from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, Request, Response, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.db import Session as SessionRow, User, get_db


# ─── CONFIG ──────────────────────────────────────────────────────────────────

COOKIE_NAME    = "apex_session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30        # 30 days
SESSION_TTL    = timedelta(days=30)


def _secret() -> str:
    """The signing key for cookie integrity. Falls back to a stable dev key."""
    return os.environ.get(
        "SESSION_SECRET",
        "apex-ai-dev-session-secret-please-change-in-production",
    )


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(_secret(), salt="apex-session-v1")


def _is_secure_cookie() -> bool:
    """Whether to set the Secure flag (HTTPS only)."""
    return os.environ.get("COOKIE_SECURE", "false").lower() in ("1", "true", "yes")


def _cookie_samesite() -> str:
    return os.environ.get("COOKIE_SAMESITE", "lax")


# ─── PASSWORDS (bcrypt + legacy SHA256 migration) ────────────────────────────

def hash_password(plain: str) -> str:
    """Hash a password using bcrypt with cost=12. Stored as 'bcrypt:<hash>'."""
    salt = bcrypt.gensalt(rounds=12)
    return "bcrypt:" + bcrypt.hashpw(plain.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain: str, stored: str) -> bool:
    """Verify against bcrypt or legacy salt:sha256 hashes."""
    if not stored:
        return False
    try:
        if stored.startswith("bcrypt:"):
            return bcrypt.checkpw(plain.encode("utf-8"), stored[7:].encode("utf-8"))
        # legacy "salt:sha256(salt+plain)" format from earlier versions
        if ":" in stored:
            salt, hashed = stored.split(":", 1)
            return hashlib.sha256((salt + plain).encode()).hexdigest() == hashed
    except Exception:
        return False
    return False


def password_needs_rehash(stored: str) -> bool:
    """Old SHA256 hashes get transparently re-hashed to bcrypt on next signin."""
    return not (stored or "").startswith("bcrypt:")


# ─── COOKIE SIGNING ──────────────────────────────────────────────────────────

def sign_session_id(session_id: str) -> str:
    """Sign + timestamp the session id. The cookie value is opaque to clients."""
    return _serializer().dumps({"sid": session_id})


def unsign_session_id(value: str) -> Optional[str]:
    """Verify a signed cookie value. Returns the session_id or None."""
    if not value:
        return None
    try:
        payload = _serializer().loads(value, max_age=COOKIE_MAX_AGE)
        sid = payload.get("sid") if isinstance(payload, dict) else None
        return sid if isinstance(sid, str) and sid else None
    except (BadSignature, SignatureExpired, Exception):
        return None


def set_session_cookie(response: Response, session_id: str) -> None:
    """Attach the signed session cookie to a response."""
    response.set_cookie(
        key=COOKIE_NAME,
        value=sign_session_id(session_id),
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        secure=_is_secure_cookie(),
        samesite=_cookie_samesite(),
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    """Remove the session cookie (used on logout)."""
    response.delete_cookie(key=COOKIE_NAME, path="/")


# ─── SESSION HELPERS ─────────────────────────────────────────────────────────

async def issue_session(
    db: AsyncSession,
    user_id: int,
    user_agent: str = "",
    ip_address: str = "",
) -> str:
    """Create a new server-side session row and return its id."""
    sid = uuid.uuid4().hex
    now = datetime.utcnow()
    db.add(SessionRow(
        id=sid,
        user_id=user_id,
        user_agent=user_agent[:500],
        ip_address=ip_address[:64],
        created_at=now,
        last_seen_at=now,
        expires_at=now + SESSION_TTL,
    ))
    return sid


async def revoke_session(db: AsyncSession, session_id: str, user_id: int) -> bool:
    """Revoke a single session (logout this device)."""
    sess = (await db.execute(
        select(SessionRow).where(
            SessionRow.id == session_id,
            SessionRow.user_id == user_id,
        )
    )).scalar_one_or_none()
    if not sess:
        return False
    sess.revoked_at = datetime.utcnow()
    return True


async def revoke_all_sessions(db: AsyncSession, user_id: int) -> int:
    """Revoke every session a user has (logout-all-devices)."""
    rows = (await db.execute(
        select(SessionRow).where(
            SessionRow.user_id == user_id,
            SessionRow.revoked_at.is_(None),
        )
    )).scalars().all()
    now = datetime.utcnow()
    for s in rows:
        s.revoked_at = now
    return len(rows)


# ─── FASTAPI DEPENDENCIES ────────────────────────────────────────────────────

async def _resolve_user_from_cookie(
    request: Request,
    db: AsyncSession,
) -> Optional[dict]:
    """
    Read the cookie, verify the signature, look up the session row, return
    the user dict — or None if anything fails.

    This function NEVER raises; the dependencies above do that based on whether
    auth is required or optional.
    """
    raw = request.cookies.get(COOKIE_NAME)
    sid = unsign_session_id(raw) if raw else None
    if not sid:
        return None

    try:
        sess = (await db.execute(
            select(SessionRow).where(SessionRow.id == sid)
        )).scalar_one_or_none()
    except Exception:
        return None

    if not sess:
        return None
    if sess.revoked_at is not None:
        return None
    if sess.expires_at and sess.expires_at < datetime.utcnow():
        return None

    try:
        user = (await db.execute(
            select(User).where(User.id == sess.user_id)
        )).scalar_one_or_none()
    except Exception:
        return None

    if not user or not user.is_active:
        return None

    return {
        "user_id":    user.id,
        "email":      user.email,
        "name":       user.full_name,
        "role":       user.role or "user",
        "gender":     user.gender or "m",
        "session_id": sid,
    }


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Required auth — raises 401 if not signed in."""
    user = await _resolve_user_from_cookie(request, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated — please sign in.",
        )
    return user


async def get_optional_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Optional[dict]:
    """Auth-or-anonymous — returns None if not signed in, never raises."""
    return await _resolve_user_from_cookie(request, db)


def require_role(*roles: str):
    """Role gate. Use as a FastAPI dependency: `Depends(require_role('admin'))`."""
    async def _dep(user: dict = Depends(get_current_user)) -> dict:
        if user.get("role") not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role(s): {', '.join(roles)}",
            )
        return user
    return _dep


# ─── WEBSOCKET HELPERS ───────────────────────────────────────────────────────

async def get_user_for_websocket(
    websocket_cookies: dict,
    query_session: Optional[str],
    db: AsyncSession,
) -> Optional[dict]:
    """
    WS connections receive cookies on same-origin handshakes. For cross-origin
    clients we also accept ?session=<signed value> as a query param.
    """
    raw = websocket_cookies.get(COOKIE_NAME) or query_session
    sid = unsign_session_id(raw) if raw else None
    if not sid:
        return None

    sess = (await db.execute(
        select(SessionRow).where(SessionRow.id == sid)
    )).scalar_one_or_none()
    if not sess or sess.revoked_at is not None:
        return None
    if sess.expires_at and sess.expires_at < datetime.utcnow():
        return None

    user = (await db.execute(
        select(User).where(User.id == sess.user_id)
    )).scalar_one_or_none()
    if not user or not user.is_active:
        return None

    return {
        "user_id":    user.id,
        "email":      user.email,
        "name":       user.full_name,
        "role":       user.role or "user",
        "session_id": sid,
    }
