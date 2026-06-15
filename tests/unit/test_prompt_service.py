"""Unit tests for the prompt version control service."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from neuralops.core.exceptions import (
    PromptNotFoundError,
    VersionAlreadyExistsError,
    VersionNotFoundError,
)
from neuralops.schemas.prompt import PromptCreate, PromptVersionCreate
from neuralops.services import prompt_service


async def _create_prompt_with_version(db: AsyncSession) -> tuple:
    """Helper: create a prompt and add version 1.0.0."""
    prompt = await prompt_service.create_prompt(db, PromptCreate(name="test-prompt"))
    version = await prompt_service.create_version(
        db,
        prompt.id,
        PromptVersionCreate(
            version="1.0.0",
            content="You are a helpful assistant. {input}",
        ),
    )
    return prompt, version


@pytest.mark.asyncio
async def test_create_prompt(db_session: AsyncSession):
    prompt = await prompt_service.create_prompt(
        db_session, PromptCreate(name="my-prompt", description="desc")
    )
    assert prompt.name == "my-prompt"
    assert prompt.description == "desc"
    assert prompt.id is not None


@pytest.mark.asyncio
async def test_list_prompts(db_session: AsyncSession):
    await prompt_service.create_prompt(db_session, PromptCreate(name="p1"))
    await prompt_service.create_prompt(db_session, PromptCreate(name="p2"))
    prompts = await prompt_service.list_prompts(db_session)
    assert len(prompts) == 2


@pytest.mark.asyncio
async def test_create_version_valid_semver(db_session: AsyncSession):
    prompt, version = await _create_prompt_with_version(db_session)
    assert version.version == "1.0.0"
    assert version.is_active is False


def test_create_version_invalid_semver():
    # The Pydantic schema validates semver before the service layer is reached
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        PromptVersionCreate(version="bad-version", content="x")
    assert "semver" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_create_duplicate_version_raises(db_session: AsyncSession):
    prompt, _ = await _create_prompt_with_version(db_session)
    with pytest.raises(VersionAlreadyExistsError):
        await prompt_service.create_version(
            db_session,
            prompt.id,
            PromptVersionCreate(version="1.0.0", content="different content"),
        )


@pytest.mark.asyncio
async def test_create_version_unknown_prompt(db_session: AsyncSession):
    with pytest.raises(PromptNotFoundError):
        await prompt_service.create_version(
            db_session,
            "nonexistent-id",
            PromptVersionCreate(version="1.0.0", content="x"),
        )


@pytest.mark.asyncio
async def test_activate_version(db_session: AsyncSession):
    prompt, version = await _create_prompt_with_version(db_session)
    activated = await prompt_service.activate_version(db_session, prompt.id, "1.0.0")
    assert activated.is_active is True


@pytest.mark.asyncio
async def test_activate_deactivates_previous(db_session: AsyncSession):
    prompt, _ = await _create_prompt_with_version(db_session)
    await prompt_service.create_version(
        db_session,
        prompt.id,
        PromptVersionCreate(version="2.0.0", content="updated content"),
    )
    await prompt_service.activate_version(db_session, prompt.id, "1.0.0")
    await prompt_service.activate_version(db_session, prompt.id, "2.0.0")

    v1 = await prompt_service.get_version(db_session, prompt.id, "1.0.0")
    v2 = await prompt_service.get_version(db_session, prompt.id, "2.0.0")
    assert v1.is_active is False
    assert v2.is_active is True


@pytest.mark.asyncio
async def test_diff_versions(db_session: AsyncSession):
    prompt, _ = await _create_prompt_with_version(db_session)
    await prompt_service.create_version(
        db_session,
        prompt.id,
        PromptVersionCreate(version="2.0.0", content="You are an expert assistant. {input}"),
    )
    diff = await prompt_service.diff_versions(db_session, prompt.id, "1.0.0", "2.0.0")
    assert diff.version_from == "1.0.0"
    assert diff.version_to == "2.0.0"
    assert "helpful" in diff.diff or "expert" in diff.diff
    assert diff.additions >= 0
    assert diff.deletions >= 0


@pytest.mark.asyncio
async def test_rollback_version(db_session: AsyncSession):
    prompt, _ = await _create_prompt_with_version(db_session)
    await prompt_service.create_version(
        db_session,
        prompt.id,
        PromptVersionCreate(version="2.0.0", content="v2 content"),
    )
    await prompt_service.activate_version(db_session, prompt.id, "2.0.0")
    rollback = await prompt_service.rollback_version(db_session, prompt.id, "1.0.0")
    assert rollback.new_active_version == "1.0.0"


@pytest.mark.asyncio
async def test_get_prompt_not_found(db_session: AsyncSession):
    with pytest.raises(PromptNotFoundError):
        await prompt_service.get_prompt(db_session, "does-not-exist")


@pytest.mark.asyncio
async def test_get_version_not_found(db_session: AsyncSession):
    prompt = await prompt_service.create_prompt(db_session, PromptCreate(name="p"))
    with pytest.raises(VersionNotFoundError):
        await prompt_service.get_version(db_session, prompt.id, "9.9.9")
