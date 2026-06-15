"""FastAPI application factory.

Creates the FastAPI app, registers middleware, includes all routers,
and manages the application lifespan (startup/shutdown hooks).
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from neuralops.api.middleware import (
    RequestIDMiddleware,
    RequestLoggingMiddleware,
    configure_cors,
)
from neuralops.api.routers import drift, experiments, health, models, prompts
from neuralops.core.config import settings
from neuralops.core.database import engine
from neuralops.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.

    Handles startup (logging, scheduler) and shutdown (cleanup) events.
    """
    # Startup
    configure_logging(
        level="DEBUG" if settings.is_development else "INFO",
        json_logs=not settings.is_development,
    )
    logger.info(
        "NeuralOps starting",
        environment=settings.neuralops_environment,
        port=settings.neuralops_port,
    )

    # Start APScheduler for drift detection
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import]

        from neuralops.core.database import AsyncSessionLocal
        from neuralops.services.drift_service import run_drift_check

        scheduler = AsyncIOScheduler()

        async def scheduled_drift_check():
            async with AsyncSessionLocal() as db:
                await run_drift_check(db)

        scheduler.add_job(
            scheduled_drift_check,
            "interval",
            minutes=settings.drift_check_interval_minutes,
            id="drift_check",
            replace_existing=True,
        )
        scheduler.start()
        app.state.scheduler = scheduler
        logger.info(
            "Drift scheduler started",
            interval_minutes=settings.drift_check_interval_minutes,
        )
    except Exception as exc:
        logger.warning("Scheduler failed to start", error=str(exc))
        app.state.scheduler = None

    yield

    # Shutdown
    if hasattr(app.state, "scheduler") and app.state.scheduler:
        app.state.scheduler.shutdown(wait=False)
    await engine.dispose()
    logger.info("NeuralOps shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="NeuralOps API",
        description="Production-grade LLMOps platform — prompts versioned, tested, monitored.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Middleware (applied in reverse order — last added = outermost)
    configure_cors(app)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # Routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(prompts.router, prefix="/api/v1/prompts", tags=["prompts"])
    app.include_router(experiments.router, prefix="/api/v1/experiments", tags=["experiments"])
    app.include_router(models.router, prefix="/api/v1/models", tags=["model-registry"])
    app.include_router(drift.router, prefix="/api/v1/drift", tags=["drift"])

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception", error=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    return app


app: FastAPI = create_app()
