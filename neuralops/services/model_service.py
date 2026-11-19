"""Model registry service.

CRUD operations for registered LLM models and routing rule management.
"""

import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from neuralops.core.exceptions import ModelAlreadyRegisteredError, ModelNotFoundError
from neuralops.core.logging import get_logger
from neuralops.models.model_registry import RegisteredModel
from neuralops.schemas.model_registry import (
    RegisteredModelCreate,
    RegisteredModelResponse,
    RegisteredModelUpdate,
)

logger = get_logger(__name__)


async def register_model(
    db: AsyncSession, data: RegisteredModelCreate
) -> RegisteredModelResponse:
    """Register a new LLM model in the registry.

    Args:
        db: Async database session.
        data: RegisteredModelCreate request body.

    Returns:
        RegisteredModelResponse for the newly registered model.

    Raises:
        ModelAlreadyRegisteredError: If model+provider combination already exists.
    """
    existing = await db.execute(
        select(RegisteredModel).where(
            RegisteredModel.name == data.name,
            RegisteredModel.provider == data.provider,
        )
    )
    if existing.scalar_one_or_none() is not None:
        raise ModelAlreadyRegisteredError(
            f"Model '{data.name}' from '{data.provider}' is already registered"
        )

    model = RegisteredModel(
        id=str(uuid.uuid4()),
        name=data.name,
        provider=data.provider,
        display_name=data.display_name,
        description=data.description,
        context_window=data.context_window,
        cost_per_1k_input_tokens=data.cost_per_1k_input_tokens,
        cost_per_1k_output_tokens=data.cost_per_1k_output_tokens,
        routing_priority=data.routing_priority,
        capabilities=data.capabilities,
    )
    db.add(model)
    await db.commit()
    await db.refresh(model)
    logger.info("Model registered", model_id=model.id, name=model.name, provider=model.provider)
    return RegisteredModelResponse.model_validate(model)


async def get_model(db: AsyncSession, model_id: str) -> RegisteredModelResponse:
    """Fetch a registered model by ID.

    Args:
        db: Async database session.
        model_id: UUID of the model.

    Returns:
        RegisteredModelResponse.

    Raises:
        ModelNotFoundError: If the model doesn't exist.
    """
    result = await db.execute(select(RegisteredModel).where(RegisteredModel.id == model_id))
    model = result.scalar_one_or_none()
    if model is None:
        raise ModelNotFoundError(f"Model '{model_id}' not found")
    return RegisteredModelResponse.model_validate(model)


async def list_models(
    db: AsyncSession, active_only: bool = False, provider: str | None = None
) -> list[RegisteredModelResponse]:
    """List registered models with optional filters.

    Args:
        db: Async database session.
        active_only: If True, only return active models.
        provider: Filter by provider name.

    Returns:
        List of RegisteredModelResponse, ordered by routing_priority.
    """
    query = select(RegisteredModel).order_by(RegisteredModel.routing_priority)
    if active_only:
        query = query.where(RegisteredModel.is_active == True)  # noqa: E712
    if provider:
        query = query.where(RegisteredModel.provider == provider)
    result = await db.execute(query)
    models = result.scalars().all()
    return [RegisteredModelResponse.model_validate(m) for m in models]


async def update_model(
    db: AsyncSession, model_id: str, data: RegisteredModelUpdate
) -> RegisteredModelResponse:
    """Update a registered model's metadata.

    Args:
        db: Async database session.
        model_id: UUID of the model to update.
        data: RegisteredModelUpdate with fields to change.

    Returns:
        Updated RegisteredModelResponse.

    Raises:
        ModelNotFoundError: If the model doesn't exist.
    """
    result = await db.execute(select(RegisteredModel).where(RegisteredModel.id == model_id))
    model = result.scalar_one_or_none()
    if model is None:
        raise ModelNotFoundError(f"Model '{model_id}' not found")

    update_data = data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(model, key, value)
    model.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(model)
    logger.info("Model updated", model_id=model_id)
    return RegisteredModelResponse.model_validate(model)


async def delete_model(db: AsyncSession, model_id: str) -> None:
    """Remove a model from the registry.

    Args:
        db: Async database session.
        model_id: UUID of the model to delete.

    Raises:
        ModelNotFoundError: If the model doesn't exist.
    """
    result = await db.execute(select(RegisteredModel).where(RegisteredModel.id == model_id))
    model = result.scalar_one_or_none()
    if model is None:
        raise ModelNotFoundError(f"Model '{model_id}' not found")
    await db.delete(model)
    await db.commit()
    logger.info("Model deleted", model_id=model_id)


async def get_best_model(db: AsyncSession, provider: str | None = None) -> RegisteredModelResponse | None:
    """Return the highest-priority active model for routing.

    Args:
        db: Async database session.
        provider: Optional provider filter.

    Returns:
        RegisteredModelResponse for the best available model, or None.
    """
    query = (
        select(RegisteredModel)
        .where(RegisteredModel.is_active == True)  # noqa: E712
        .order_by(RegisteredModel.routing_priority)
        .limit(1)
    )
    if provider:
        query = query.where(RegisteredModel.provider == provider)
    result = await db.execute(query)
    model = result.scalar_one_or_none()
    return RegisteredModelResponse.model_validate(model) if model else None
