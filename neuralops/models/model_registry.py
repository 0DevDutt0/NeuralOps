"""SQLAlchemy ORM model for the LLM model registry.

Table:
    registered_models — catalogue of LLM models with routing and cost metadata.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, String, Text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from neuralops.core.database import Base


class RegisteredModel(Base):
    """A registered LLM model in the NeuralOps model registry.

    Attributes:
        id: UUID primary key.
        name: Model identifier (e.g. "llama-3.1-8b-instant").
        provider: Backend provider ("groq" | "mistral" | "ollama").
        display_name: Human-readable label for the dashboard.
        description: Optional description of the model's strengths.
        context_window: Max context window in tokens.
        cost_per_1k_input_tokens: USD cost per 1 000 input tokens.
        cost_per_1k_output_tokens: USD cost per 1 000 output tokens.
        avg_latency_ms: Rolling average latency in milliseconds.
        is_active: Whether this model can receive routed requests.
        routing_priority: Lower = higher priority when auto-routing.
        capabilities: JSON list of capability tags (e.g. ["chat", "code"]).
        created_at: UTC timestamp of registration.
        updated_at: UTC timestamp of last update.
    """

    __tablename__ = "registered_models"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    context_window: Mapped[int] = mapped_column(default=8192, nullable=False)
    cost_per_1k_input_tokens: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    cost_per_1k_output_tokens: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    avg_latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    routing_priority: Mapped[int] = mapped_column(default=100, nullable=False)
    capabilities: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
