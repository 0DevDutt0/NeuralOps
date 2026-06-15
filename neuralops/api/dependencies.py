"""FastAPI dependency injection: database sessions and service instances.

All dependencies are declared here and injected into route handlers
via FastAPI's Depends() mechanism.
"""

from collections.abc import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from neuralops.core.database import AsyncSessionLocal
from neuralops.engine.ab_runner import ABRunner
from neuralops.engine.judge import LLMJudge
from neuralops.services.llm_client import LLMClient


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session per request.

    Yields:
        AsyncSession: Active SQLAlchemy async session.
    """
    async with AsyncSessionLocal() as session:
        yield session


_shared_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return the shared LLMClient singleton.

    Returns:
        Shared LLMClient instance (created on first call).
    """
    global _shared_llm_client
    if _shared_llm_client is None:
        _shared_llm_client = LLMClient()
    return _shared_llm_client


def get_judge(llm_client: LLMClient = Depends(get_llm_client)) -> LLMJudge:
    """Return a LLMJudge using the shared LLM client.

    Args:
        llm_client: Injected shared LLMClient.

    Returns:
        LLMJudge instance.
    """
    return LLMJudge(llm_client=llm_client)


def get_ab_runner(
    llm_client: LLMClient = Depends(get_llm_client),
    judge: LLMJudge = Depends(get_judge),
) -> ABRunner:
    """Return an ABRunner using the shared LLM client and judge.

    Args:
        llm_client: Injected shared LLMClient.
        judge: Injected LLMJudge.

    Returns:
        ABRunner instance.
    """
    return ABRunner(llm_client=llm_client, judge=judge)
