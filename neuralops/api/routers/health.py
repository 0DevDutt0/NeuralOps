"""Health and readiness check endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from neuralops.api.dependencies import get_db
from neuralops.core.config import settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    environment: str
    timestamp: datetime
    llm_provider: str


class ReadinessResponse(BaseModel):
    """Readiness probe response."""

    ready: bool
    database: str
    timestamp: datetime


@router.get("/", response_model=HealthResponse, summary="Liveness check")
async def health() -> HealthResponse:
    """Return application health status.

    Returns:
        HealthResponse with status, version, and configuration info.
    """
    return HealthResponse(
        status="ok",
        version="0.1.0",
        environment=settings.neuralops_environment,
        timestamp=datetime.utcnow(),
        llm_provider=settings.llm_provider,
    )


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness check")
async def readiness(db: AsyncSession = Depends(get_db)) -> ReadinessResponse:
    """Check whether the application is ready to serve traffic.

    Validates database connectivity.

    Args:
        db: Injected database session.

    Returns:
        ReadinessResponse indicating readiness and database status.
    """
    try:
        await db.execute(text("SELECT 1"))
        db_status = "ok"
        ready = True
    except Exception as exc:
        db_status = f"error: {exc}"
        ready = False

    return ReadinessResponse(
        ready=ready,
        database=db_status,
        timestamp=datetime.utcnow(),
    )
