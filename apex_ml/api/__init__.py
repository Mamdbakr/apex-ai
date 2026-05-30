"""
FastAPI integration for apex_ml.

Importing this submodule is optional — if FastAPI is not installed,
the import silently sets `router = None` and the rest of apex_ml is
unaffected.

Mount with:

    from apex_ml.api import router
    if router is not None:
        app.include_router(router, prefix="/ml")
"""
from .router import router

__all__ = ["router"]
