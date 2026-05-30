"""
backend/core/logging.py
─────────────────────────
Structured logging via loguru. Call setup_logging() once at app startup.
"""
import sys
import logging
from loguru import logger
from backend.core.config import settings


class _InterceptHandler(logging.Handler):
    """Redirect stdlib logging → loguru."""
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging() -> None:
    s = settings()
    logger.remove()
    logger.add(
        sys.stdout,
        level=s.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True,
    )
    logger.add(
        "logs/apex_{time:YYYY-MM-DD}.log",
        level=s.LOG_LEVEL,
        rotation="00:00",
        retention="14 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )
    # Redirect stdlib loggers (uvicorn, fastapi, etc) into loguru.
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "httpx"):
        logging.getLogger(name).handlers = [_InterceptHandler()]
