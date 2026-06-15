"""Custom exception hierarchy for NeuralOps.

All domain exceptions inherit from NeuralOpsError so callers can
catch the base class when they want to handle any platform error.
"""

from fastapi import HTTPException
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_409_CONFLICT,
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_502_BAD_GATEWAY,
)


class NeuralOpsError(Exception):
    """Base exception for all NeuralOps domain errors."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or message


# ── Prompt / Version exceptions ──────────────────────────────────────────────


class PromptNotFoundError(NeuralOpsError):
    """Raised when a prompt ID does not exist in the database."""


class VersionNotFoundError(NeuralOpsError):
    """Raised when a specific semver version of a prompt does not exist."""


class VersionAlreadyExistsError(NeuralOpsError):
    """Raised when attempting to create a duplicate version string."""


class InvalidSemverError(NeuralOpsError):
    """Raised when a version string does not conform to semver."""


# ── Experiment exceptions ─────────────────────────────────────────────────────


class ExperimentNotFoundError(NeuralOpsError):
    """Raised when an experiment ID does not exist."""


class ExperimentAlreadyCompletedError(NeuralOpsError):
    """Raised when trying to add trials to a completed experiment."""


# ── Model registry exceptions ─────────────────────────────────────────────────


class ModelNotFoundError(NeuralOpsError):
    """Raised when a registered model ID does not exist."""


class ModelAlreadyRegisteredError(NeuralOpsError):
    """Raised when a model name+provider combination already exists."""


# ── LLM client exceptions ─────────────────────────────────────────────────────


class LLMClientError(NeuralOpsError):
    """Base class for LLM provider errors."""


class LLMRateLimitError(LLMClientError):
    """Raised when a provider returns HTTP 429."""


class LLMTimeoutError(LLMClientError):
    """Raised when a provider call exceeds the configured timeout."""


class LLMProviderError(LLMClientError):
    """Raised for unexpected provider errors (5xx, malformed response)."""


class NoFallbackAvailableError(LLMClientError):
    """Raised when both primary and fallback providers fail."""


# ── Guardrail exceptions ──────────────────────────────────────────────────────


class GuardrailError(NeuralOpsError):
    """Base class for guardrail pipeline errors."""


class PIIDetectionError(GuardrailError):
    """Raised when Presidio fails to initialize or analyze."""


class ToxicityModelError(GuardrailError):
    """Raised when the toxicity model cannot load or infer."""


# ── Helper: convert domain errors to HTTP exceptions ─────────────────────────

_ERROR_STATUS_MAP: dict[type[NeuralOpsError], int] = {
    PromptNotFoundError: HTTP_404_NOT_FOUND,
    VersionNotFoundError: HTTP_404_NOT_FOUND,
    ModelNotFoundError: HTTP_404_NOT_FOUND,
    ExperimentNotFoundError: HTTP_404_NOT_FOUND,
    VersionAlreadyExistsError: HTTP_409_CONFLICT,
    ModelAlreadyRegisteredError: HTTP_409_CONFLICT,
    ExperimentAlreadyCompletedError: HTTP_409_CONFLICT,
    InvalidSemverError: HTTP_422_UNPROCESSABLE_ENTITY,
    LLMRateLimitError: HTTP_429_TOO_MANY_REQUESTS,
    LLMTimeoutError: HTTP_502_BAD_GATEWAY,
    LLMProviderError: HTTP_502_BAD_GATEWAY,
    NoFallbackAvailableError: HTTP_502_BAD_GATEWAY,
    GuardrailError: HTTP_500_INTERNAL_SERVER_ERROR,
}


def to_http_exception(exc: NeuralOpsError) -> HTTPException:
    """Convert a NeuralOpsError to the appropriate FastAPI HTTPException.

    Args:
        exc: The domain exception to convert.

    Returns:
        HTTPException with status code and detail message.
    """
    status_code = _ERROR_STATUS_MAP.get(type(exc), HTTP_400_BAD_REQUEST)
    return HTTPException(status_code=status_code, detail=exc.message)
