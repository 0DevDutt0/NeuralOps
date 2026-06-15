"""Model registry endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from neuralops.api.dependencies import get_db
from neuralops.core.exceptions import NeuralOpsError, to_http_exception
from neuralops.schemas.model_registry import (
    RegisteredModelCreate,
    RegisteredModelResponse,
    RegisteredModelUpdate,
)
from neuralops.services import model_service

router = APIRouter(tags=["model-registry"])


@router.post("/", response_model=RegisteredModelResponse, status_code=201, summary="Register model")
async def register_model(
    data: RegisteredModelCreate,
    db: AsyncSession = Depends(get_db),
) -> RegisteredModelResponse:
    """Register a new LLM model in the NeuralOps registry.

    Args:
        data: Model name, provider, cost data, and capabilities.
        db: Database session.

    Returns:
        The newly registered model.
    """
    try:
        return await model_service.register_model(db, data)
    except NeuralOpsError as exc:
        raise to_http_exception(exc)


@router.get("/", response_model=list[RegisteredModelResponse], summary="List models")
async def list_models(
    active_only: bool = Query(False, description="Only return active models"),
    provider: str | None = Query(None, description="Filter by provider"),
    db: AsyncSession = Depends(get_db),
) -> list[RegisteredModelResponse]:
    """List registered models with optional filters.

    Args:
        active_only: If True, exclude inactive models.
        provider: Filter by provider name.
        db: Database session.

    Returns:
        List of registered models ordered by routing priority.
    """
    return await model_service.list_models(db, active_only=active_only, provider=provider)


@router.get("/{model_id}", response_model=RegisteredModelResponse, summary="Get model")
async def get_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
) -> RegisteredModelResponse:
    """Fetch a registered model by ID.

    Args:
        model_id: UUID of the model.
        db: Database session.

    Returns:
        The requested model.
    """
    try:
        return await model_service.get_model(db, model_id)
    except NeuralOpsError as exc:
        raise to_http_exception(exc)


@router.patch("/{model_id}", response_model=RegisteredModelResponse, summary="Update model")
async def update_model(
    model_id: str,
    data: RegisteredModelUpdate,
    db: AsyncSession = Depends(get_db),
) -> RegisteredModelResponse:
    """Update a registered model's metadata or status.

    Args:
        model_id: UUID of the model.
        data: Fields to update (all optional).
        db: Database session.

    Returns:
        Updated model record.
    """
    try:
        return await model_service.update_model(db, model_id, data)
    except NeuralOpsError as exc:
        raise to_http_exception(exc)


@router.delete("/{model_id}", status_code=204, summary="Delete model")
async def delete_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Remove a model from the registry.

    Args:
        model_id: UUID of the model to remove.
        db: Database session.
    """
    try:
        await model_service.delete_model(db, model_id)
    except NeuralOpsError as exc:
        raise to_http_exception(exc)


@router.get("/routing/best", response_model=RegisteredModelResponse | None, summary="Get best model")
async def get_best_model(
    provider: str | None = Query(None, description="Restrict to this provider"),
    db: AsyncSession = Depends(get_db),
) -> RegisteredModelResponse | None:
    """Return the highest-priority active model for routing.

    Args:
        provider: Optional provider filter.
        db: Database session.

    Returns:
        Best available model, or null if none are active.
    """
    return await model_service.get_best_model(db, provider=provider)
