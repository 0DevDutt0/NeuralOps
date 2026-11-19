"""Pydantic v2 schemas for prompt and prompt version endpoints."""

import re
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


# ── Requests ──────────────────────────────────────────────────────────────────


class PromptCreate(BaseModel):
    """Request body for creating a new prompt."""

    name: str = Field(..., min_length=1, max_length=255, description="Unique prompt name")
    description: str | None = Field(None, description="Optional description")


class PromptVersionCreate(BaseModel):
    """Request body for adding a new version to an existing prompt."""

    version: str = Field(..., description="Semver string e.g. '1.0.0'")
    content: str = Field(..., min_length=1, description="The prompt text")
    system_prompt: str | None = Field(None, description="Optional system/context message")
    variables: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of variable names to expected types e.g. {'name': 'str'}",
    )
    meta: dict = Field(
        default_factory=dict, description="Arbitrary metadata (author, tags, etc.)"
    )

    @field_validator("version")
    @classmethod
    def validate_semver(cls, v: str) -> str:
        """Ensure version string is valid semver (MAJOR.MINOR.PATCH)."""
        if not _SEMVER_RE.match(v):
            raise ValueError(f"Version '{v}' must be semver format: MAJOR.MINOR.PATCH")
        return v


# ── Responses ─────────────────────────────────────────────────────────────────


class PromptVersionResponse(BaseModel):
    """Response schema for a single prompt version."""

    id: str
    prompt_id: str
    version: str
    content: str
    system_prompt: str | None
    variables: dict[str, str]
    meta: dict
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class PromptResponse(BaseModel):
    """Response schema for a prompt (without versions list)."""

    id: str
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PromptDetailResponse(PromptResponse):
    """Response schema for a prompt with all its versions."""

    versions: list[PromptVersionResponse] = []


class DiffResponse(BaseModel):
    """Response schema for a unified diff between two prompt versions."""

    prompt_id: str
    version_from: str
    version_to: str
    diff: str
    additions: int
    deletions: int


class RollbackResponse(BaseModel):
    """Response confirming a rollback operation."""

    prompt_id: str
    rolled_back_to: str
    new_active_version: str
    message: str
