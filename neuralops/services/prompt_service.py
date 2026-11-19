"""Prompt version control service.

Implements Git-style versioning semantics: create, version, activate,
diff, and rollback. All database access goes through this service —
routers never touch SQLAlchemy directly.
"""

import difflib
import re
import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from neuralops.core.exceptions import (
    InvalidSemverError,
    PromptNotFoundError,
    VersionAlreadyExistsError,
    VersionNotFoundError,
)
from neuralops.core.logging import get_logger
from neuralops.models.prompt import Prompt, PromptVersion
from neuralops.schemas.prompt import (
    DiffResponse,
    PromptCreate,
    PromptDetailResponse,
    PromptResponse,
    PromptVersionCreate,
    PromptVersionResponse,
    RollbackResponse,
)

logger = get_logger(__name__)

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


async def create_prompt(db: AsyncSession, data: PromptCreate) -> PromptResponse:
    """Create a new prompt record.

    Args:
        db: Async database session.
        data: PromptCreate request body.

    Returns:
        PromptResponse for the newly created prompt.
    """
    prompt = Prompt(
        id=str(uuid.uuid4()),
        name=data.name,
        description=data.description,
    )
    db.add(prompt)
    await db.commit()
    await db.refresh(prompt)
    logger.info("Prompt created", prompt_id=prompt.id, name=prompt.name)
    return PromptResponse.model_validate(prompt)


async def get_prompt(db: AsyncSession, prompt_id: str) -> PromptDetailResponse:
    """Fetch a prompt with all its versions.

    Args:
        db: Async database session.
        prompt_id: UUID string of the prompt.

    Returns:
        PromptDetailResponse including all versions.

    Raises:
        PromptNotFoundError: If no prompt with that ID exists.
    """
    result = await db.execute(
        select(Prompt)
        .options(selectinload(Prompt.versions))
        .where(Prompt.id == prompt_id)
    )
    prompt = result.scalar_one_or_none()
    if prompt is None:
        raise PromptNotFoundError(f"Prompt '{prompt_id}' not found")
    return PromptDetailResponse.model_validate(prompt)


async def list_prompts(db: AsyncSession, skip: int = 0, limit: int = 50) -> list[PromptResponse]:
    """List all prompts with pagination.

    Args:
        db: Async database session.
        skip: Number of records to skip.
        limit: Maximum records to return.

    Returns:
        List of PromptResponse objects.
    """
    result = await db.execute(select(Prompt).offset(skip).limit(limit))
    prompts = result.scalars().all()
    return [PromptResponse.model_validate(p) for p in prompts]


async def create_version(
    db: AsyncSession, prompt_id: str, data: PromptVersionCreate
) -> PromptVersionResponse:
    """Add a new version to an existing prompt.

    Args:
        db: Async database session.
        prompt_id: UUID of the parent prompt.
        data: PromptVersionCreate request body.

    Returns:
        PromptVersionResponse for the new version.

    Raises:
        PromptNotFoundError: If the parent prompt doesn't exist.
        InvalidSemverError: If the version string is not valid semver.
        VersionAlreadyExistsError: If that version already exists on this prompt.
    """
    if not _SEMVER_RE.match(data.version):
        raise InvalidSemverError(f"Version '{data.version}' must be semver: MAJOR.MINOR.PATCH")

    prompt_result = await db.execute(select(Prompt).where(Prompt.id == prompt_id))
    prompt = prompt_result.scalar_one_or_none()
    if prompt is None:
        raise PromptNotFoundError(f"Prompt '{prompt_id}' not found")

    existing = await db.execute(
        select(PromptVersion).where(
            PromptVersion.prompt_id == prompt_id,
            PromptVersion.version == data.version,
        )
    )
    if existing.scalar_one_or_none() is not None:
        raise VersionAlreadyExistsError(
            f"Version '{data.version}' already exists on prompt '{prompt_id}'"
        )

    version = PromptVersion(
        id=str(uuid.uuid4()),
        prompt_id=prompt_id,
        version=data.version,
        content=data.content,
        system_prompt=data.system_prompt,
        variables=data.variables,
        meta=data.meta,
        is_active=False,
    )
    db.add(version)

    prompt.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(version)
    logger.info("Prompt version created", prompt_id=prompt_id, version=data.version)
    return PromptVersionResponse.model_validate(version)


async def list_versions(db: AsyncSession, prompt_id: str) -> list[PromptVersionResponse]:
    """List all versions of a prompt.

    Args:
        db: Async database session.
        prompt_id: UUID of the prompt.

    Returns:
        List of PromptVersionResponse, ordered by creation time.

    Raises:
        PromptNotFoundError: If the prompt doesn't exist.
    """
    prompt_result = await db.execute(select(Prompt).where(Prompt.id == prompt_id))
    if prompt_result.scalar_one_or_none() is None:
        raise PromptNotFoundError(f"Prompt '{prompt_id}' not found")

    result = await db.execute(
        select(PromptVersion)
        .where(PromptVersion.prompt_id == prompt_id)
        .order_by(PromptVersion.created_at)
    )
    versions = result.scalars().all()
    return [PromptVersionResponse.model_validate(v) for v in versions]


async def get_version(
    db: AsyncSession, prompt_id: str, version: str
) -> PromptVersionResponse:
    """Fetch a specific version of a prompt.

    Args:
        db: Async database session.
        prompt_id: UUID of the prompt.
        version: Semver string.

    Returns:
        PromptVersionResponse for the requested version.

    Raises:
        VersionNotFoundError: If that version doesn't exist.
    """
    result = await db.execute(
        select(PromptVersion).where(
            PromptVersion.prompt_id == prompt_id,
            PromptVersion.version == version,
        )
    )
    pv = result.scalar_one_or_none()
    if pv is None:
        raise VersionNotFoundError(
            f"Version '{version}' not found on prompt '{prompt_id}'"
        )
    return PromptVersionResponse.model_validate(pv)


async def activate_version(
    db: AsyncSession, prompt_id: str, version: str
) -> PromptVersionResponse:
    """Set a specific version as active (deactivates any other active version).

    Args:
        db: Async database session.
        prompt_id: UUID of the prompt.
        version: Semver string to activate.

    Returns:
        PromptVersionResponse for the now-active version.

    Raises:
        VersionNotFoundError: If the version doesn't exist.
    """
    # Deactivate any currently active version
    result = await db.execute(
        select(PromptVersion).where(
            PromptVersion.prompt_id == prompt_id,
            PromptVersion.is_active == True,  # noqa: E712
        )
    )
    currently_active = result.scalar_one_or_none()
    if currently_active:
        currently_active.is_active = False

    # Activate the requested version
    target_result = await db.execute(
        select(PromptVersion).where(
            PromptVersion.prompt_id == prompt_id,
            PromptVersion.version == version,
        )
    )
    target = target_result.scalar_one_or_none()
    if target is None:
        raise VersionNotFoundError(f"Version '{version}' not found on prompt '{prompt_id}'")

    target.is_active = True
    await db.commit()
    await db.refresh(target)
    logger.info("Prompt version activated", prompt_id=prompt_id, version=version)
    return PromptVersionResponse.model_validate(target)


async def diff_versions(
    db: AsyncSession, prompt_id: str, v1: str, v2: str
) -> DiffResponse:
    """Compute a unified diff between two prompt versions.

    Args:
        db: Async database session.
        prompt_id: UUID of the prompt.
        v1: First version semver string (from).
        v2: Second version semver string (to).

    Returns:
        DiffResponse with unified diff text and addition/deletion counts.

    Raises:
        VersionNotFoundError: If either version doesn't exist.
    """
    pv1 = await get_version(db, prompt_id, v1)
    pv2 = await get_version(db, prompt_id, v2)

    lines_v1 = pv1.content.splitlines(keepends=True)
    lines_v2 = pv2.content.splitlines(keepends=True)

    diff_lines = list(
        difflib.unified_diff(lines_v1, lines_v2, fromfile=f"v{v1}", tofile=f"v{v2}")
    )
    diff_text = "".join(diff_lines)
    additions = sum(1 for ln in diff_lines if ln.startswith("+") and not ln.startswith("+++"))
    deletions = sum(1 for ln in diff_lines if ln.startswith("-") and not ln.startswith("---"))

    return DiffResponse(
        prompt_id=prompt_id,
        version_from=v1,
        version_to=v2,
        diff=diff_text,
        additions=additions,
        deletions=deletions,
    )


async def rollback_version(
    db: AsyncSession, prompt_id: str, target_version: str
) -> RollbackResponse:
    """Rollback the active version of a prompt to a prior version.

    This activates the target version (which must already exist).

    Args:
        db: Async database session.
        prompt_id: UUID of the prompt.
        target_version: Semver string to roll back to.

    Returns:
        RollbackResponse confirming the rollback.

    Raises:
        VersionNotFoundError: If the target version doesn't exist.
    """
    activated = await activate_version(db, prompt_id, target_version)
    logger.info("Prompt rolled back", prompt_id=prompt_id, version=target_version)
    return RollbackResponse(
        prompt_id=prompt_id,
        rolled_back_to=target_version,
        new_active_version=activated.version,
        message=f"Successfully rolled back prompt '{prompt_id}' to version {target_version}",
    )
