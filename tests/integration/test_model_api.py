"""Integration tests for the model registry API endpoints."""

import pytest
from httpx import AsyncClient


def _model_payload(**kwargs) -> dict:
    defaults = {
        "name": "llama-3.1-8b-instant",
        "provider": "groq",
        "display_name": "Llama 3.1 8B",
        "context_window": 131072,
        "cost_per_1k_input_tokens": 0.00005,
        "cost_per_1k_output_tokens": 0.00008,
        "routing_priority": 1,
    }
    defaults.update(kwargs)
    return defaults


@pytest.mark.asyncio
async def test_register_model(api_client: AsyncClient):
    resp = await api_client.post("/api/v1/models/", json=_model_payload())
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "llama-3.1-8b-instant"
    assert data["provider"] == "groq"
    assert data["is_active"] is True


@pytest.mark.asyncio
async def test_register_duplicate_model_returns_409(api_client: AsyncClient):
    await api_client.post("/api/v1/models/", json=_model_payload())
    resp = await api_client.post("/api/v1/models/", json=_model_payload())
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_list_models_empty(api_client: AsyncClient):
    resp = await api_client.get("/api/v1/models/")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_models_returns_registered(api_client: AsyncClient):
    await api_client.post("/api/v1/models/", json=_model_payload(name="m1"))
    await api_client.post("/api/v1/models/", json=_model_payload(name="m2", provider="mistral"))
    resp = await api_client.get("/api/v1/models/")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


@pytest.mark.asyncio
async def test_get_model_by_id(api_client: AsyncClient):
    created = (await api_client.post("/api/v1/models/", json=_model_payload())).json()
    resp = await api_client.get(f"/api/v1/models/{created['id']}")
    assert resp.status_code == 200
    assert resp.json()["id"] == created["id"]


@pytest.mark.asyncio
async def test_get_model_not_found(api_client: AsyncClient):
    resp = await api_client.get("/api/v1/models/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_model(api_client: AsyncClient):
    created = (await api_client.post("/api/v1/models/", json=_model_payload())).json()
    resp = await api_client.patch(
        f"/api/v1/models/{created['id']}",
        json={"display_name": "Updated Name", "routing_priority": 5},
    )
    assert resp.status_code == 200
    assert resp.json()["display_name"] == "Updated Name"


@pytest.mark.asyncio
async def test_delete_model(api_client: AsyncClient):
    created = (await api_client.post("/api/v1/models/", json=_model_payload())).json()
    resp = await api_client.delete(f"/api/v1/models/{created['id']}")
    assert resp.status_code == 204
    # Verify it's gone
    get_resp = await api_client.get(f"/api/v1/models/{created['id']}")
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_model_not_found(api_client: AsyncClient):
    resp = await api_client.delete("/api/v1/models/ghost-id")
    assert resp.status_code == 404
