"""SQLAlchemy ORM models for prompts and prompt versions.

Tables:
    prompts         — parent record (name, description).
    prompt_versions — immutable versioned snapshots of prompt content.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from neuralops.core.database import Base


class Prompt(Base):
    """Top-level prompt record.

    Attributes:
        id: UUID primary key.
        name: Human-readable name (unique).
        description: Optional description of the prompt's purpose.
        created_at: UTC timestamp of creation.
        updated_at: UTC timestamp of last modification.
        versions: Relationship to all PromptVersion records.
    """

    __tablename__ = "prompts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    versions: Mapped[list["PromptVersion"]] = relationship(
        "PromptVersion", back_populates="prompt", cascade="all, delete-orphan"
    )


class PromptVersion(Base):
    """Immutable versioned snapshot of a prompt.

    Attributes:
        id: UUID primary key.
        prompt_id: FK to the parent Prompt.
        version: Semver string e.g. "1.0.0".
        content: The actual prompt text.
        system_prompt: Optional system/context message.
        variables: JSON dict mapping variable names to their expected types.
        meta: JSON dict of arbitrary metadata (author, tags, etc.).
        is_active: Only one version per prompt should be active at a time.
        created_at: UTC timestamp of creation.
        prompt: Back-reference to parent Prompt.
    """

    __tablename__ = "prompt_versions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    prompt_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("prompts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    variables: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    meta: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    prompt: Mapped["Prompt"] = relationship("Prompt", back_populates="versions")
