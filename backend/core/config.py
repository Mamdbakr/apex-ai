"""
backend/core/config.py
────────────────────────
Central configuration. Reads .env, validates types, exposes a cached settings()
singleton. All env access should go through this module.

v14 changes:
  • Removed JWT_* settings (JWT auth was retired)
  • Added SESSION_SECRET / COOKIE_SECURE / COOKIE_SAMESITE
  • Added Groq as a first-class LLM provider
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    # Groq is the recommended provider (fastest + free Llama-3 access).
    LLM_PROVIDER: Literal["groq", "openai", "anthropic", "ollama"] = "groq"

    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.1-70b-versatile"

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20241022"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"

    # ── Embeddings / Vector DB ───────────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: Literal["cpu", "cuda"] = "cpu"
    VECTOR_DB_PATH: str = str(ROOT / "ai_models" / "vector_store")
    VECTOR_COLLECTION: str = "apex_fitness_kb"

    # ── Auth (cookie-based sessions) ─────────────────────────────────────────
    # Used to sign the session cookie. MUST be set to a random value in prod.
    SESSION_SECRET: str = "apex-ai-dev-session-secret-please-change-in-production"
    # Only set the Secure flag when serving over HTTPS in production.
    COOKIE_SECURE: bool = False
    # 'lax' is the safest default for local dev. Use 'none' when frontend and
    # backend live on different domains AND COOKIE_SECURE is true.
    COOKIE_SAMESITE: Literal["lax", "strict", "none"] = "lax"

    BCRYPT_ROUNDS: int = 12

    # ── DB ───────────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./apex_ai.db"

    # ── CV ───────────────────────────────────────────────────────────────────
    CV_DEVICE: Literal["cuda", "cpu", "auto"] = "auto"
    CV_MODEL_PATH: str = str(ROOT / "ai_models" / "dl_models" / "exercise_classifier.pth")
    CV_SCALER_PATH: str = str(ROOT / "ai_models" / "dl_models" / "cv_keypoint_scaler.pkl")
    CV_CONFIG_PATH: str = str(ROOT / "ai_models" / "dl_models" / "exercise_classifier_config.json")

    # YOLOv8-pose backbone (the AI-Gym engine). Set POSE_BACKEND=mediapipe to
    # disable YOLO and use the legacy MediaPipe path; everything else stays
    # the same.
    POSE_BACKEND: Literal["yolo", "mediapipe"] = "yolo"
    YOLO_MODEL_PATH: str = "yolov8n-pose.pt"     # auto-downloads on first run
    YOLO_CONF_THRESHOLD: float = 0.15
    YOLO_IMG_SIZE: int =480

    # ── ML ───────────────────────────────────────────────────────────────────
    ML_MODEL_DIR: str = str(ROOT / "ai_models" / "ml_models")

    # ── Server ───────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    # CORS: comma-separated list. With cookie auth, you CANNOT use '*' AND
    # credentials at the same time — pin to your frontend origin in prod.
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origins_list(self) -> list[str]:
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def has_groq(self) -> bool:
        return bool(self.GROQ_API_KEY and self.GROQ_API_KEY.startswith("gsk_"))

    @property
    def has_openai(self) -> bool:
        return bool(self.OPENAI_API_KEY and self.OPENAI_API_KEY.startswith("sk-"))

    @property
    def has_anthropic(self) -> bool:
        return bool(self.ANTHROPIC_API_KEY and self.ANTHROPIC_API_KEY.startswith("sk-"))


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()
