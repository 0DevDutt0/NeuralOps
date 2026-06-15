"""Unit tests for the model registry service."""

import pytest

from neuralops.core.exceptions import ModelAlreadyRegisteredError, ModelNotFoundError
from neuralops.schemas.model_registry import RegisteredModelCreate, RegisteredModelUpdate
from neuralops.services import model_service


def _model_data(**kwargs) -> RegisteredModelCreate:
    defaults = dict(
        name="llama-3.1-8b-instant",
        provider="groq",
        display_name="Llama 3.1 8B Instant",
        description="Fast and efficient model",
        context_window=131072,
        cost_per_1k_input_tokens=0.00005,
        cost_per_1k_output_tokens=0.00008,
        routing_priority=1,
        capabilities=["chat", "code"],
    )
    defaults.update(kwargs)
    return RegisteredModelCreate(**defaults)


# ── register_model ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_register_model(db_session):
    data = _model_data()
    result = await model_service.register_model(db_session, data)
    assert result.id is not None
    assert result.name == "llama-3.1-8b-instant"
    assert result.provider == "groq"
    assert result.is_active is True


@pytest.mark.asyncio
async def test_register_model_duplicate_raises(db_session):
    data = _model_data()
    await model_service.register_model(db_session, data)
    with pytest.raises(ModelAlreadyRegisteredError):
        await model_service.register_model(db_session, data)


@pytest.mark.asyncio
async def test_register_same_name_different_provider(db_session):
    await model_service.register_model(db_session, _model_data(provider="groq"))
    result = await model_service.register_model(db_session, _model_data(provider="mistral"))
    assert result.provider == "mistral"


# ── get_model ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_model(db_session):
    created = await model_service.register_model(db_session, _model_data())
    fetched = await model_service.get_model(db_session, created.id)
    assert fetched.id == created.id
    assert fetched.name == created.name


@pytest.mark.asyncio
async def test_get_model_not_found(db_session):
    with pytest.raises(ModelNotFoundError):
        await model_service.get_model(db_session, "nonexistent-id")


# ── list_models ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_models_empty(db_session):
    result = await model_service.list_models(db_session)
    assert result == []


@pytest.mark.asyncio
async def test_list_models_returns_all(db_session):
    await model_service.register_model(db_session, _model_data(name="model-a", routing_priority=1))
    await model_service.register_model(db_session, _model_data(name="model-b", routing_priority=2))
    result = await model_service.list_models(db_session)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_list_models_active_only(db_session):
    m = await model_service.register_model(db_session, _model_data(name="active-m"))
    # Deactivate via update
    await model_service.update_model(
        db_session, m.id, RegisteredModelUpdate(is_active=False)
    )
    result = await model_service.list_models(db_session, active_only=True)
    assert all(r.is_active for r in result)
    assert not any(r.name == "active-m" for r in result)


@pytest.mark.asyncio
async def test_list_models_filter_by_provider(db_session):
    await model_service.register_model(db_session, _model_data(name="g-model", provider="groq"))
    await model_service.register_model(db_session, _model_data(name="m-model", provider="mistral"))
    result = await model_service.list_models(db_session, provider="groq")
    assert len(result) == 1
    assert result[0].provider == "groq"


# ── update_model ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_model_display_name(db_session):
    created = await model_service.register_model(db_session, _model_data())
    updated = await model_service.update_model(
        db_session, created.id, RegisteredModelUpdate(display_name="Updated Name")
    )
    assert updated.display_name == "Updated Name"
    assert updated.name == created.name  # unchanged


@pytest.mark.asyncio
async def test_update_model_is_active(db_session):
    created = await model_service.register_model(db_session, _model_data())
    updated = await model_service.update_model(
        db_session, created.id, RegisteredModelUpdate(is_active=False)
    )
    assert updated.is_active is False


@pytest.mark.asyncio
async def test_update_model_not_found(db_session):
    with pytest.raises(ModelNotFoundError):
        await model_service.update_model(
            db_session, "ghost-id", RegisteredModelUpdate(display_name="x")
        )


# ── delete_model ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_model(db_session):
    created = await model_service.register_model(db_session, _model_data())
    await model_service.delete_model(db_session, created.id)
    with pytest.raises(ModelNotFoundError):
        await model_service.get_model(db_session, created.id)


@pytest.mark.asyncio
async def test_delete_model_not_found(db_session):
    with pytest.raises(ModelNotFoundError):
        await model_service.delete_model(db_session, "no-such-id")


# ── get_best_model ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_best_model_returns_highest_priority(db_session):
    await model_service.register_model(db_session, _model_data(name="low", routing_priority=10))
    await model_service.register_model(db_session, _model_data(name="high", routing_priority=1))
    best = await model_service.get_best_model(db_session)
    assert best is not None
    assert best.name == "high"


@pytest.mark.asyncio
async def test_get_best_model_none_when_all_inactive(db_session):
    m = await model_service.register_model(db_session, _model_data())
    await model_service.update_model(db_session, m.id, RegisteredModelUpdate(is_active=False))
    best = await model_service.get_best_model(db_session)
    assert best is None


@pytest.mark.asyncio
async def test_get_best_model_with_provider_filter(db_session):
    await model_service.register_model(db_session, _model_data(name="g", provider="groq", routing_priority=1))
    await model_service.register_model(db_session, _model_data(name="m", provider="mistral", routing_priority=2))
    best = await model_service.get_best_model(db_session, provider="mistral")
    assert best is not None
    assert best.provider == "mistral"
