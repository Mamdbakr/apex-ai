"""
backend/main.py
─────────────────
APEX AI v14 — Production FastAPI entry point.

Run:
    uvicorn backend.main:app --reload --port 8000

Top-level routes (full list at /docs):
    /auth/*          signup, signin, logout, sessions, me        (cookie-based)
    /chat            LLM + RAG chatbot (auth required)
    /chat/stream     SSE streaming
    /vision/analyze  single image CV (auth required)
    /vision/stream   real-time WebSocket CV (auth required)
    /vision/history  per-user CV analyses
    /predict/*       calorie / weight-change / fitness-level + explanations
    /recommend       personalised exercise slate + rationale
    /insights/*      AI dashboard payload (auth required)
    /user-data/*     profile, workouts, weights, nutrition (auth required)
    /data/*          ingest pipeline (auth required, admin for batch ETL)
    /health          deep health probe
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from backend.core.config import settings
from backend.core.logging import setup_logging
from backend.database.db import init_db


ROOT = Path(__file__).resolve().parent.parent
VERSION = "14.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    s = settings()
    is_dev = "change" in s.SESSION_SECRET.lower() or "dev" in s.SESSION_SECRET.lower()

    logger.info("═" * 64)
    logger.info(f"  🚀  APEX AI v{VERSION} starting · env={'DEV' if is_dev else 'PROD'}")
    logger.info(f"  🔐  Auth: cookie-based sessions (no JWT)")
    if is_dev:
        logger.warning("  ⚠️   SESSION_SECRET is the development default — set a real one in .env for production!")
    if s.has_groq:
        logger.info(f"  🦙  Groq · model={s.GROQ_MODEL}")
    elif s.has_openai:
        logger.info(f"  🤖  OpenAI · model={s.OPENAI_MODEL}")
    elif s.has_anthropic:
        logger.info(f"  🤖  Anthropic · model={s.ANTHROPIC_MODEL}")
    else:
        logger.warning("  ⚠️   No LLM API key found — chatbot will use heuristic fallbacks")
    logger.info("═" * 64)

    # 1. Database
    try:
        await init_db()
        logger.info("  ✅  Database ready")
    except Exception as e:
        logger.error(f"  ❌  Database init failed: {e}")

    # 2. v9 fallback vector store (best effort)
    try:
        from backend.chatbot.vector_store import get_vector_store
        vs = get_vector_store()
        logger.info(f"  ✅  v9 vector store · {vs.count()} chunks indexed")
    except Exception as e:
        logger.debug(f"v9 vector store not initialised: {e}")

    # 3. Chatbot service
    try:
        from backend.services.chatbot_service import get_chatbot_service
        svc = get_chatbot_service()
        _ = svc.engine
    except Exception as e:
        logger.warning(f"  ⚠️   Chatbot service init failed: {e}")

    # 4. CV classifier
    try:
        from backend.cv.exercise_classifier import get_classifier
        clf = get_classifier()
        if clf and clf.is_ready:
            logger.info(f"  ✅  ExerciseNet loaded · {len(clf.classes)} classes · {clf.device}")
        else:
            logger.warning("  ⚠️   ExerciseNet not loaded — run: python -m training.train_cv  (or open the notebook)")
    except Exception as e:
        logger.error(f"  ❌  CV classifier error: {e}")

    # 4b. Pose backbone (YOLOv8-pose primary, MediaPipe fallback)
    try:
        from backend.cv.pose_extractor import get_pose_extractor
        pose = get_pose_extractor()
        logger.info(f"  ✅  Pose backbone · {pose.backend} · device={pose.device}")
    except Exception as e:
        logger.error(f"  ❌  Pose backbone error: {e}")

    # 5. ML service
    try:
        from backend.services.ml_service import get_ml_service
        svc = get_ml_service()
        loaded = sum(1 for m in (svc.calorie, svc.weight, svc.fitness_clf) if m is not None)
        logger.info(f"  ✅  ML service ready · {loaded}/3 models loaded · explainability=on")
    except Exception as e:
        logger.error(f"  ❌  ML service error: {e}")

    logger.info(f"  🌐  Listening on http://{s.HOST}:{s.PORT}  ·  docs at /docs")
    logger.info("═" * 64)
    yield

    logger.info("👋  APEX AI shutting down")
    try:
        from backend.chatbot.llm_provider import get_llm
        llm = get_llm()
        if hasattr(llm, "aclose"):
            await llm.aclose()
    except Exception:
        pass


app = FastAPI(
    title="APEX AI",
    description=(
        "AI-powered fitness platform — cookie-session auth + RAG chatbot + "
        "real-time CV + ML predictions + explainability"
    ),
    version=VERSION,
    lifespan=lifespan,
)

_s = settings()

# IMPORTANT: With cookie-based auth, allow_credentials=True is required, and
# you cannot use the wildcard '*' for origins. We always pass an explicit list.
_origins = _s.cors_origins_list if _s.cors_origins_list != ["*"] else [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:3000", "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["set-cookie"],
)


@app.exception_handler(Exception)
async def global_error_handler(request, exc):
    logger.exception(f"Unhandled: {request.method} {request.url.path} — {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error":  "internal_error",
            "detail": "The server had a problem. Please try again.",
        },
    )


# ─── ROUTERS ──────────────────────────────────────────────────────────────────

from backend.routes import auth, chat, data, insights, predict, recommend, user_data, vision

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(vision.router)
app.include_router(predict.router)
app.include_router(recommend.router)
app.include_router(data.router)
app.include_router(insights.router)
app.include_router(user_data.router)

# ── apex_ml additive ML layer (workout generation, coaching, temporal pose) ──
# Mounted under /ml/* so it cannot collide with any existing route. Failures
# in apex_ml never break app startup — they only disable the /ml endpoints.
try:
    from backend.routes import ml_coaching
    app.include_router(ml_coaching.router)
except Exception as _e:
    logger.warning(f"apex_ml routes not mounted: {_e}")


# ─── HEALTH + ROOT ────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    return {
        "name":    "APEX AI",
        "version": VERSION,
        "auth":    "cookie-session",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health", tags=["System"])
async def health():
    """Deep health check — used by Docker/K8s readiness probes."""
    s = settings()
    out = {
        "status":  "ok",
        "version": VERSION,
        "provider": s.LLM_PROVIDER,
        "auth": {
            "method":         "cookie-session",
            "session_max_days": 30,
            "password_hash":  "bcrypt",
            "cookie_secure":  s.COOKIE_SECURE,
            "cookie_samesite": s.COOKIE_SAMESITE,
        },
    }
    try:
        from backend.services.chatbot_service import get_chatbot_service
        out["chatbot"] = get_chatbot_service().stats()
    except Exception as e:
        out["chatbot"] = {"engine": "error", "error": str(e)}

    try:
        from backend.cv.exercise_classifier import get_classifier
        clf = get_classifier()
        out["cv_model"]  = bool(clf and clf.is_ready)
        out["cv_device"] = clf.device if clf else "none"
    except Exception:
        out["cv_model"] = False

    try:
        from backend.cv.pose_extractor import get_pose_extractor
        pose = get_pose_extractor()
        out["pose_backend"] = pose.backend
        out["pose_device"]  = pose.device
    except Exception:
        out["pose_backend"] = "unavailable"

    try:
        from backend.services.ml_service import get_ml_service
        svc = get_ml_service()
        out["ml_models"] = {
            "calorie":      bool(svc.calorie),
            "weight":       bool(svc.weight),
            "fitness_clf":  bool(svc.fitness_clf),
            "explanations": True,
        }
    except Exception:
        out["ml_models"] = {}

    return out


if __name__ == "__main__":
    import uvicorn
    s = settings()
    uvicorn.run("backend.main:app", host=s.HOST, port=s.PORT, reload=True)
