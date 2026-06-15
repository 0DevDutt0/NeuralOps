"""Shared pytest fixtures for the NeuralOps test suite.

All tests use in-memory SQLite (never PostgreSQL) and mocked LLM clients
so no real API keys or external services are required.
"""

import json

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from neuralops.api.dependencies import get_db
from neuralops.api.main import app
from neuralops.core.database import Base

# ── Database fixtures ─────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    """Yield a fresh in-memory SQLite session for each test.

    Creates all tables, yields the session, then tears down the engine.
    Each test gets a completely isolated database.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()


# ── API client fixture ────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def api_client(db_session: AsyncSession) -> AsyncClient:
    """Yield an httpx.AsyncClient wired to the FastAPI test app.

    Overrides the get_db dependency to use the in-memory SQLite session,
    ensuring tests never touch a real database.

    Args:
        db_session: In-memory database session from db_session fixture.

    Yields:
        AsyncClient configured for the test app.
    """
    app.dependency_overrides[get_db] = lambda: db_session
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
    app.dependency_overrides.clear()


# ── LLM mock fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def mock_groq_response():
    """Return a factory for building mock Groq API responses.

    Returns:
        Callable(content, prompt_tokens, completion_tokens) -> httpx.Response
    """
    def _build(
        content: str = "This is a mocked LLM response.",
        prompt_tokens: int = 15,
        completion_tokens: int = 20,
        model: str = "llama-3.1-8b-instant",
    ) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test-id",
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )
    return _build


@pytest.fixture
def mock_groq(mock_groq_response, respx_mock):
    """Mock all Groq API calls at the httpx transport level.

    Uses respx to intercept httpx requests before they reach the network.
    Compatible with the openai SDK (which uses httpx internally).

    Args:
        mock_groq_response: Response factory fixture.
        respx_mock: respx mock context from pytest-respx.

    Returns:
        respx_mock with Groq endpoint mocked.
    """
    respx_mock.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=mock_groq_response()
    )
    return respx_mock


@pytest.fixture
def mock_judge_response():
    """Mock Groq judge response as valid JSON scoring output.

    Returns:
        httpx.Response with structured judge JSON.
    """
    judge_json = json.dumps({
        "relevance": 8.5,
        "accuracy": 7.0,
        "clarity": 9.0,
        "safety": 10.0,
        "composite": 8.5,
        "reasoning": "The response directly addresses the question with good clarity.",
        "confidence": 0.9,
    })
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": judge_json}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 60, "total_tokens": 260},
            "model": "llama-3.3-70b-versatile",
        },
    )


# ── Sample data fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def sample_prompt_data() -> dict:
    """Return valid data for creating a test prompt."""
    return {"name": "test-assistant", "description": "A test prompt"}


@pytest.fixture
def sample_version_data() -> dict:
    """Return valid data for creating a prompt version."""
    return {
        "version": "1.0.0",
        "content": "You are a helpful assistant. Answer: {input}",
        "system_prompt": "Be concise and accurate.",
        "variables": {"input": "str"},
        "meta": {"author": "test", "tags": ["test"]},
    }
