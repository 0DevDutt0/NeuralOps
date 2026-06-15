"""Prompt CRUD and version control endpoints.

All 8 versioning operations:
    POST   /                    Create prompt
    GET    /                    List prompts
    GET    /{id}                Get prompt + versions
    POST   /{id}/versions/      Add new version
    GET    /{id}/versions/      List versions
    GET    /{id}/versions/{v}   Get specific version
    POST   /{id}/activate/{v}   Activate version
    GET    /{id}/diff/{v1}/{v2} Unified diff
    POST   /{id}/rollback/{v}   Rollback to version
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from neuralops.api.dependencies import get_db
from neuralops.core.exceptions import NeuralOpsError, to_http_exception
from neuralops.schemas.prompt import (
    DiffResponse,
    PromptCreate,
    PromptDetailResponse,
    PromptResponse,
    PromptVersionCreate,
    PromptVersionResponse,
    RollbackResponse,
)
from neuralops.services import prompt_service

router = APIRouter(tags=["prompts"])


@router.post("/", response_model=PromptResponse, status_code=201, summary="Create a prompt")
async def create_prompt(
    data: PromptCreate,
    db: AsyncSession = Depends(get_db),
) -> PromptResponse:
    """Create a new prompt record.

    Args:
        data: Name and optional description.
        db: Database session.

    Returns:
        The newly created prompt.
    """
    try:
        return await prompt_service.create_prompt(db, data)
    except NeuralOpsError as exc:
        raise to_http_exception(exc)


@router.get("/", response_model=list[PromptResponse], summary="List prompts")
async def list_prompts(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> list[PromptResponse]:
    """List all prompts with pagination.

    Args:
        skip: Records to skip.
        limit: Max records to return.
        db: Database session.

    Returns:
        List of prompts.
    """
    return await prompt_service.list_prompts(db, skip=skip, limit=limit)


@router.get("/{prompt_id}", response_model=PromptDetailResponse, summary="Get prompt with versions")
async def get_prompt(
    prompt_id: str,
    db: AsyncSession = Depends(get_db),
) -> PromptDetailResponse:
    """Fetch a prompt and all its versions.

    Args:
        prompt_id: UUID of the prompt.
        db: Database session.

    Returns:
        Prompt with full version history.
    """
    try:
        return await prompt_service.get_prompt(db, prompt_id)
    except NeuralOpsError as exc:
        raise to_http_exception(exc)


@router.post(
    "/{prompt_id}/versions/",
    response_model=PromptVersionResponse,
    status_code=201,
    summary="Create a new version",
)
async def create_version(
    prompt_id: str,
    data: PromptVersionCreate,
    db: AsyncSession = Depends(get_db),
) -> PromptVersionResponse:
    """Add a new semver version to an existing prompt.

    Args:
        prompt_id: UUID of the parent prompt.
        data: Version content and metadata.
        db: Database session.

    Returns:
        The newly created version.
    """
    try:
        return await prompt_service.create_version(db, prompt_id, data)
    except NeuralOpsError as exc:
        raise to_http_exception(exc)


@router.get(
    "/{prompt_id}/versions/",
    response_model=list[PromptVersionResponse],
    summary="List all versions",
)
async def list_versions(
    prompt_id: str,
    db: AsyncSession = Depends(get_db),
) -> list[PromptVersionResponse]:
    """List all versions of a prompt in chronological order.

    Args:
        prompt_id: UUID of the prompt.
        db: Database session.

    Returns:
        Ordered list of versions.
    """
    try:
        return await prompt_service.list_versions(db, prompt_id)
    except NeuralOpsError as exc:
        raise to_http_exception(exc)


@router.get(
    "/{prompt_id}/versions/{version}",
    response_model=PromptVersionResponse,
    summary="Get specific version",
)
async def get_version(
    prompt_id: str,
    version: str,
    db: AsyncSession = Depends(get_db),
) -> PromptVersionResponse:
    """Fetch a specific version of a prompt.

    Args:
        prompt_id: UUID of the prompt.
        version: Semver string e.g. "1.2.0".
        db: Database session.

    Returns:
        The requested version.
    """
    try:
        return await prompt_service.get_version(db, prompt_id, version)
    except NeuralOpsError as exc:
        raise to_http_exception(exc)


@router.post(
    "/{prompt_id}/activate/{version}",
    response_model=PromptVersionResponse,
    summary="Activate a version",
)
async def activate_version(
    prompt_id: str,
    version: str,
    db: AsyncSession = Depends(get_db),
) -> PromptVersionResponse:
    """Set a specific version as the active version for a prompt.

    Deactivates any currently active version first.

    Args:
        prompt_id: UUID of the prompt.
        version: Semver string to activate.
        db: Database session.

    Returns:
        The now-active version.
    """
    try:
        return await prompt_service.activate_version(db, prompt_id, version)
    except NeuralOpsError as exc:
        raise to_http_exception(exc)


@router.get(
    "/{prompt_id}/diff/{v1}/{v2}",
    response_model=DiffResponse,
    summary="Diff two versions",
)
async def diff_versions(
    prompt_id: str,
    v1: str,
    v2: str,
    db: AsyncSession = Depends(get_db),
) -> DiffResponse:
    """Compute a unified diff between two prompt versions.

    Args:
        prompt_id: UUID of the prompt.
        v1: Source version (from).
        v2: Target version (to).
        db: Database session.

    Returns:
        Unified diff with addition and deletion counts.
    """
    try:
        return await prompt_service.diff_versions(db, prompt_id, v1, v2)
    except NeuralOpsError as exc:
        raise to_http_exception(exc)


@router.post(
    "/{prompt_id}/rollback/{version}",
    response_model=RollbackResponse,
    summary="Rollback to a version",
)
async def rollback_version(
    prompt_id: str,
    version: str,
    db: AsyncSession = Depends(get_db),
) -> RollbackResponse:
    """Roll back the active version of a prompt to a prior version.

    Args:
        prompt_id: UUID of the prompt.
        version: Semver string to roll back to.
        db: Database session.

    Returns:
        Confirmation of the rollback.
    """
    try:
        return await prompt_service.rollback_version(db, prompt_id, version)
    except NeuralOpsError as exc:
        raise to_http_exception(exc)
