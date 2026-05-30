"""
backend/database/db.py
────────────────────────
APEX AI v14 — Production schema.

Changes vs v13:
  • Removed `RefreshToken` table — JWT auth was retired.
  • Added `expires_at` to `Session` so cookie-based sessions can expire on the
    server side without needing a JWT exp claim.
  • Everything else is unchanged so existing data carries over.

Every table is keyed on user_id and indexed for the queries we actually run.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import os

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text,
    Index,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker


DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./apex_ai.db")
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


# ─── USERS & AUTH ────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    full_name       = Column(String,  default="User")
    email           = Column(String,  unique=True, index=True, nullable=False)
    hashed_password = Column(String,  nullable=False)
    gender          = Column(String,  default="m")
    role            = Column(String,  default="user")          # "user" | "admin"
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=datetime.utcnow)
    last_login_at   = Column(DateTime, nullable=True)


class Session(Base):
    """One active login session — a phone, a browser, etc.

    The (signed) primary key of this row is what lives in the user's cookie.
    """
    __tablename__ = "sessions"
    id           = Column(String,  primary_key=True)
    user_id      = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    user_agent   = Column(String,  default="")
    ip_address   = Column(String,  default="")
    created_at   = Column(DateTime, default=datetime.utcnow)
    last_seen_at = Column(DateTime, default=datetime.utcnow)
    expires_at   = Column(DateTime, default=lambda: datetime.utcnow() + timedelta(days=30))
    revoked_at   = Column(DateTime, nullable=True)


# ─── PROFILE & LOGS ──────────────────────────────────────────────────────────

class UserProfile(Base):
    __tablename__ = "user_profiles"
    id             = Column(Integer, primary_key=True)
    user_id        = Column(Integer, ForeignKey("users.id"), index=True, nullable=False, unique=True)
    name           = Column(String,  default="User")
    age            = Column(Integer, default=25)
    weight_kg      = Column(Float,   default=70.0)
    height_cm      = Column(Float,   default=175.0)
    activity_level = Column(Integer, default=2)
    gender         = Column(Integer, default=1)
    goal           = Column(String,  default="lose")
    target_weight  = Column(Float,   default=65.0)
    dietary_pref   = Column(String,  default="No Restrictions")
    timeframe      = Column(String,  default="3-6 months")
    created_at     = Column(DateTime, default=datetime.utcnow)
    updated_at     = Column(DateTime, default=datetime.utcnow)


class WeightLog(Base):
    __tablename__ = "weight_logs"
    id        = Column(Integer, primary_key=True)
    user_id   = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    weight_kg = Column(Float,  nullable=False)
    body_fat  = Column(Float,  nullable=True)
    logged_at = Column(DateTime, default=datetime.utcnow, index=True)


class WorkoutLog(Base):
    __tablename__ = "workout_logs"
    id              = Column(Integer, primary_key=True)
    user_id         = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    exercise        = Column(String)
    muscle_group    = Column(String,  default="")
    sets            = Column(Integer, default=3)
    reps            = Column(Integer, default=10)
    weight_kg       = Column(Float,   default=0.0)
    duration_min    = Column(Integer, default=30)
    reps_counted    = Column(Integer, default=0)
    form_score      = Column(Float,   default=1.0)
    rpe             = Column(Float,   nullable=True)
    calories_burned = Column(Float,   nullable=True)
    notes           = Column(Text,    default="")
    logged_at       = Column(DateTime, default=datetime.utcnow, index=True)


class NutritionLog(Base):
    __tablename__ = "nutrition_logs"
    id        = Column(Integer, primary_key=True)
    user_id   = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    date      = Column(String,  nullable=False)
    calories  = Column(Float,   default=0.0)
    protein_g = Column(Float,   default=0.0)
    carbs_g   = Column(Float,   default=0.0)
    fat_g     = Column(Float,   default=0.0)
    water_ml  = Column(Float,   default=0.0)
    logged_at = Column(DateTime, default=datetime.utcnow, index=True)


# ─── AI / ML LOGS ────────────────────────────────────────────────────────────

class PredictionLog(Base):
    """Every ML prediction made for the user, with full inputs/outputs and
    a feature-importance JSON for explainability."""
    __tablename__ = "prediction_logs"
    id              = Column(Integer, primary_key=True)
    user_id         = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    prediction_type = Column(String, index=True)
    input_data      = Column(JSON)
    output_data     = Column(JSON)
    explanation     = Column(JSON)
    model_version   = Column(String,  default="v14")
    confidence      = Column(Float,   nullable=True)
    predicted_at    = Column(DateTime, default=datetime.utcnow, index=True)


class RecommendationLog(Base):
    __tablename__ = "recommendation_logs"
    id           = Column(Integer, primary_key=True)
    user_id      = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    rec_type     = Column(String, index=True)
    items        = Column(JSON)
    rationale    = Column(JSON)
    fitness_level_id = Column(Integer, nullable=True)
    served_at    = Column(DateTime, default=datetime.utcnow, index=True)


class CVAnalysis(Base):
    """Persisted CV pose-analysis results for replay + dashboard intelligence."""
    __tablename__ = "cv_analyses"
    id            = Column(Integer, primary_key=True)
    user_id       = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    session_id    = Column(String,  index=True)
    exercise      = Column(String,  default="unknown")
    confidence    = Column(Float,   default=0.0)
    reps          = Column(Integer, default=0)
    form_score    = Column(Float,   default=0.0)
    feedback      = Column(Text,    default="")
    suggestions   = Column(JSON,    default=list)
    duration_s    = Column(Float,   default=0.0)
    frame_count   = Column(Integer, default=0)
    keypoint_summary = Column(JSON, default=dict)
    analysed_at   = Column(DateTime, default=datetime.utcnow, index=True)


class UserFeatureVector(Base):
    __tablename__ = "user_feature_vectors"
    id                = Column(Integer, primary_key=True)
    user_id           = Column(Integer, ForeignKey("users.id"), unique=True, index=True, nullable=False)
    bmi               = Column(Float,   default=0.0)
    tdee              = Column(Float,   default=0.0)
    fitness_level     = Column(Integer, default=1)
    workouts_30d      = Column(Integer, default=0)
    avg_duration      = Column(Float,   default=0.0)
    avg_form_score    = Column(Float,   default=0.0)
    consistency_score = Column(Float,   default=0.0)
    weight_trend_7d   = Column(Float,   default=0.0)
    weight_trend_30d  = Column(Float,   default=0.0)
    tdee_trend        = Column(Float,   default=0.0)
    cluster_id        = Column(Integer, nullable=True)
    cluster_label     = Column(String,  nullable=True)
    feature_json      = Column(JSON,    default=dict)
    updated_at        = Column(DateTime, default=datetime.utcnow)


class ChatHistory(Base):
    __tablename__ = "chat_history"
    id        = Column(Integer, primary_key=True)
    user_id   = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    role      = Column(String)
    content   = Column(Text)
    model     = Column(String,  default="")
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


# Composite indexes for the hot query paths
Index("ix_workout_user_date", WorkoutLog.user_id, WorkoutLog.logged_at.desc())
Index("ix_weight_user_date",  WeightLog.user_id,  WeightLog.logged_at.desc())
Index("ix_chat_user_ts",      ChatHistory.user_id, ChatHistory.timestamp.desc())
Index("ix_cv_user_date",      CVAnalysis.user_id,  CVAnalysis.analysed_at.desc())
Index("ix_rec_user_date",     RecommendationLog.user_id, RecommendationLog.served_at.desc())
Index("ix_session_user",      Session.user_id, Session.revoked_at)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
