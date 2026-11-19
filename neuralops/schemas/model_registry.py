"""Pydantic v2 schemas for model registry endpoints."""

from datetime import datetime

from pydantic import BaseModel, Field

# ── Requests ──────────────────────────────────────────────────────────────────


class RegisteredModelCreate(BaseModel):
    """Request body for registering a new LLM model."""

    name: str = Field(..., description="Model identifier e.g. 'llama-3.1-8b-instant'")
    provider: str = Field(..., description="Backend: 'groq' | 'mistral' | 'ollama'")
    display_name: str = Field(..., description="Human-readable label")
    description: str | None = None
    context_window: int = Field(default=8192, gt=0)
    cost_per_1k_input_tokens: float = Field(default=0.0, ge=0)
    cost_per_1k_output_tokens: float = Field(default=0.0, ge=0)
    routing_priority: int = Field(default=100, ge=1)
    capabilities: list[str] = Field(default_factory=list)


class RegisteredModelUpdate(BaseModel):
    """Request body for updating a registered model."""

    display_name: str | None = None
    description: str | None = None
    cost_per_1k_input_tokens: float | None = None
    cost_per_1k_output_tokens: float | None = None
    is_active: bool | None = None
    routing_priority: int | None = None
    capabilities: list[str] | None = None


# ── Responses ─────────────────────────────────────────────────────────────────


class RegisteredModelResponse(BaseModel):
    """Response schema for a registered model."""

    id: str
    name: str
    provider: str
    display_name: str
    description: str | None
    context_window: int
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float
    avg_latency_ms: float
    is_active: bool
    routing_priority: int
    capabilities: list[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ModelCostSummary(BaseModel):
    """Cost summary for a model over a time window."""

    model_id: str
    model_name: str
    provider: str
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float
    request_count: int
