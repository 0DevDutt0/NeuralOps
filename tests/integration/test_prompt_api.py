"""Integration tests for the prompt versioning API endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_prompt(api_client: AsyncClient):
    resp = await api_client.post("/api/v1/prompts/", json={"name": "my-prompt"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "my-prompt"
    assert "id" in data


@pytest.mark.asyncio
async def test_list_prompts_empty(api_client: AsyncClient):
    resp = await api_client.get("/api/v1/prompts/")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_prompts_after_create(api_client: AsyncClient):
    await api_client.post("/api/v1/prompts/", json={"name": "p1"})
    await api_client.post("/api/v1/prompts/", json={"name": "p2"})
    resp = await api_client.get("/api/v1/prompts/")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


@pytest.mark.asyncio
async def test_get_prompt_not_found(api_client: AsyncClient):
    resp = await api_client.get("/api/v1/prompts/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_and_get_prompt(api_client: AsyncClient):
    create_resp = await api_client.post(
        "/api/v1/prompts/",
        json={"name": "test-prompt", "description": "A test"},
    )
    prompt_id = create_resp.json()["id"]
    get_resp = await api_client.get(f"/api/v1/prompts/{prompt_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "test-prompt"


@pytest.mark.asyncio
async def test_create_version(api_client: AsyncClient):
    prompt_resp = await api_client.post("/api/v1/prompts/", json={"name": "vp"})
    prompt_id = prompt_resp.json()["id"]

    version_resp = await api_client.post(
        f"/api/v1/prompts/{prompt_id}/versions/",
        json={"version": "1.0.0", "content": "You are a helpful assistant."},
    )
    assert version_resp.status_code == 201
    assert version_resp.json()["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_create_duplicate_version_returns_409(api_client: AsyncClient):
    prompt_resp = await api_client.post("/api/v1/prompts/", json={"name": "dup"})
    prompt_id = prompt_resp.json()["id"]
    payload = {"version": "1.0.0", "content": "x"}
    await api_client.post(f"/api/v1/prompts/{prompt_id}/versions/", json=payload)
    resp = await api_client.post(f"/api/v1/prompts/{prompt_id}/versions/", json=payload)
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_invalid_semver_returns_422(api_client: AsyncClient):
    prompt_resp = await api_client.post("/api/v1/prompts/", json={"name": "sv"})
    prompt_id = prompt_resp.json()["id"]
    resp = await api_client.post(
        f"/api/v1/prompts/{prompt_id}/versions/",
        json={"version": "not-semver", "content": "x"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_activate_version(api_client: AsyncClient):
    prompt_resp = await api_client.post("/api/v1/prompts/", json={"name": "act"})
    prompt_id = prompt_resp.json()["id"]
    await api_client.post(
        f"/api/v1/prompts/{prompt_id}/versions/",
        json={"version": "1.0.0", "content": "x"},
    )
    resp = await api_client.post(f"/api/v1/prompts/{prompt_id}/activate/1.0.0")
    assert resp.status_code == 200
    assert resp.json()["is_active"] is True


@pytest.mark.asyncio
async def test_diff_two_versions(api_client: AsyncClient):
    prompt_resp = await api_client.post("/api/v1/prompts/", json={"name": "diff"})
    prompt_id = prompt_resp.json()["id"]
    await api_client.post(
        f"/api/v1/prompts/{prompt_id}/versions/",
        json={"version": "1.0.0", "content": "You are a helpful assistant."},
    )
    await api_client.post(
        f"/api/v1/prompts/{prompt_id}/versions/",
        json={"version": "2.0.0", "content": "You are an expert assistant."},
    )
    resp = await api_client.get(f"/api/v1/prompts/{prompt_id}/diff/1.0.0/2.0.0")
    assert resp.status_code == 200
    data = resp.json()
    assert "diff" in data
    assert data["additions"] >= 0


@pytest.mark.asyncio
async def test_rollback_version(api_client: AsyncClient):
    prompt_resp = await api_client.post("/api/v1/prompts/", json={"name": "rb"})
    prompt_id = prompt_resp.json()["id"]
    for ver in ["1.0.0", "2.0.0"]:
        await api_client.post(
            f"/api/v1/prompts/{prompt_id}/versions/",
            json={"version": ver, "content": f"Content {ver}"},
        )
    await api_client.post(f"/api/v1/prompts/{prompt_id}/activate/2.0.0")
    resp = await api_client.post(f"/api/v1/prompts/{prompt_id}/rollback/1.0.0")
    assert resp.status_code == 200
    assert resp.json()["new_active_version"] == "1.0.0"


@pytest.mark.asyncio
async def test_health_endpoint(api_client: AsyncClient):
    resp = await api_client.get("/health/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_readiness_endpoint(api_client: AsyncClient):
    resp = await api_client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True
