"""Unified LLM client supporting Groq, Mistral, and Ollama backends.

Usage:
    from neuralops.services.llm_client import LLMClient, LLMBackend

    client = LLMClient()
    response = await client.complete("Hello, world!", backend=LLMBackend.GROQ)
"""

import time
from dataclasses import dataclass
from enum import StrEnum

import httpx
from openai import AsyncOpenAI
from openai import RateLimitError as OpenAIRateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from neuralops.core.config import settings
from neuralops.core.exceptions import (
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    NoFallbackAvailableError,
)
from neuralops.core.logging import get_logger

logger = get_logger(__name__)

# Cost tables (USD per 1M tokens) — update from provider pricing pages
_GROQ_COSTS: dict[str, tuple[float, float]] = {
    # model: (input_per_1M, output_per_1M)
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama3-8b-8192": (0.05, 0.08),
    "llama3-70b-8192": (0.59, 0.79),
    "mixtral-8x7b-32768": (0.24, 0.24),
    "gemma2-9b-it": (0.20, 0.20),
}

_MISTRAL_COSTS: dict[str, tuple[float, float]] = {
    "mistral-small-latest": (0.20, 0.60),
    "mistral-medium-latest": (2.70, 8.10),
    "mistral-large-latest": (2.00, 6.00),
    "open-mistral-7b": (0.25, 0.25),
    "open-mixtral-8x7b": (0.70, 0.70),
}


class LLMBackend(StrEnum):
    """Supported LLM provider backends."""

    GROQ = "groq"
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    FIXTURE = "fixture"


@dataclass
class LLMResponse:
    """Structured response from any LLM backend.

    Attributes:
        content: The generated text content.
        model: Model identifier used.
        backend: Which backend was used.
        prompt_tokens: Number of input tokens consumed.
        completion_tokens: Number of output tokens generated.
        total_tokens: Sum of prompt and completion tokens.
        cost_usd: Estimated cost in USD.
        latency_ms: Time taken for the API call in milliseconds.
    """

    content: str
    model: str
    backend: LLMBackend
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float


def _count_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate missing usage locally at roughly four UTF-8 bytes per token."""
    return (len(text.encode("utf-8")) + 3) // 4


def _calculate_cost(
    model: str, backend: LLMBackend, prompt_tokens: int, completion_tokens: int
) -> float:
    """Calculate estimated USD cost for a request.

    Args:
        model: Model identifier string.
        backend: The LLM backend used.
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD.
    """
    if backend == LLMBackend.GROQ:
        costs = _GROQ_COSTS.get(model, (0.59, 0.79))
    elif backend == LLMBackend.MISTRAL:
        costs = _MISTRAL_COSTS.get(model, (2.00, 6.00))
    else:
        return 0.0  # Ollama is local/free
    input_cost = (prompt_tokens / 1_000_000) * costs[0]
    output_cost = (completion_tokens / 1_000_000) * costs[1]
    return round(input_cost + output_cost, 8)


class LLMClient:
    """Unified client for Groq, Mistral, and Ollama with retry and fallback.

    Attributes:
        _groq: AsyncOpenAI client pointed at Groq's API.
        _http: httpx.AsyncClient for Ollama requests.
    """

    def __init__(self) -> None:
        self._groq = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.groq_api_key or "placeholder",
            timeout=30.0,
        )
        self._http = httpx.AsyncClient(timeout=30.0)
        self._mistral_client = None

    def _get_mistral_client(self):
        """Lazily initialize Mistral client (avoids import error if key missing)."""
        if self._mistral_client is None:
            from mistralai import Mistral  # type: ignore[import]

            self._mistral_client = Mistral(api_key=settings.mistral_api_key or "placeholder")
        return self._mistral_client

    @retry(
        retry=retry_if_exception_type((LLMRateLimitError, LLMTimeoutError)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        backend: LLMBackend = LLMBackend.GROQ,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        stream: bool = False,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """Send a completion request to the specified backend.

        Args:
            prompt: User message content.
            system: Optional system message.
            model: Model identifier; defaults per-backend if None.
            backend: Which LLM provider to use.
            temperature: Sampling temperature (0.0–2.0).
            max_tokens: Maximum output tokens.
            stream: If True, stream tokens (not currently surfaced to callers).
            response_format: Optional format dict e.g. {"type": "json_object"}.

        Returns:
            LLMResponse with content, token counts, cost, and latency.

        Raises:
            LLMRateLimitError: Provider returned HTTP 429.
            LLMTimeoutError: Request timed out.
            LLMProviderError: Other provider-level error.
        """
        match backend:
            case LLMBackend.GROQ:
                return await self._complete_groq(
                    prompt, system, model, temperature, max_tokens, response_format
                )
            case LLMBackend.MISTRAL:
                return await self._complete_mistral(
                    prompt, system, model, temperature, max_tokens, response_format
                )
            case LLMBackend.OLLAMA:
                return await self._complete_ollama(prompt, system, model, temperature, max_tokens)
            case _:
                raise LLMProviderError(f"Provider '{backend}' has no live client")

    async def complete_with_fallback(
        self,
        prompt: str,
        primary: LLMBackend,
        fallback: LLMBackend,
        **kwargs,
    ) -> LLMResponse:
        """Try primary backend; fall back to secondary on rate limit or timeout.

        Args:
            prompt: User message content.
            primary: First backend to try.
            fallback: Backend to use if primary fails.
            **kwargs: Forwarded to complete().

        Returns:
            LLMResponse from whichever backend succeeded.

        Raises:
            NoFallbackAvailableError: Both backends failed.
        """
        try:
            return await self.complete(prompt, backend=primary, **kwargs)
        except (LLMRateLimitError, LLMTimeoutError, LLMProviderError) as primary_exc:
            logger.warning(
                "Primary backend failed, trying fallback",
                primary=primary.value,
                fallback=fallback.value,
                error=str(primary_exc),
            )
            try:
                return await self.complete(prompt, backend=fallback, **kwargs)
            except (LLMRateLimitError, LLMTimeoutError, LLMProviderError) as fallback_exc:
                raise NoFallbackAvailableError(
                    f"Both {primary.value} and {fallback.value} failed",
                    detail=f"Primary: {primary_exc.message}; Fallback: {fallback_exc.message}",
                ) from fallback_exc

    async def _complete_groq(
        self,
        prompt: str,
        system: str | None,
        model: str | None,
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
    ) -> LLMResponse:
        resolved_model = model or "llama-3.1-8b-instant"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": resolved_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        t0 = time.monotonic()
        try:
            response = await self._groq.chat.completions.create(**kwargs)
        except OpenAIRateLimitError as exc:
            raise LLMRateLimitError(f"Groq rate limit: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Groq timeout: {exc}") from exc
        except Exception as exc:
            raise LLMProviderError(f"Groq error: {exc}") from exc

        latency = (time.monotonic() - t0) * 1000
        usage = response.usage
        content = response.choices[0].message.content or ""
        prompt_tokens = usage.prompt_tokens if usage else _count_tokens((system or "") + prompt)
        completion_tokens = usage.completion_tokens if usage else _count_tokens(content)

        return LLMResponse(
            content=content,
            model=resolved_model,
            backend=LLMBackend.GROQ,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=_calculate_cost(resolved_model, LLMBackend.GROQ, prompt_tokens, completion_tokens),
            latency_ms=latency,
        )

    async def _complete_mistral(
        self,
        prompt: str,
        system: str | None,
        model: str | None,
        temperature: float,
        max_tokens: int,
        response_format: dict | None = None,
    ) -> LLMResponse:
        from mistralai.models import Messages, ResponseFormat, SystemMessage, UserMessage

        resolved_model = model or "mistral-small-latest"
        messages: list[Messages] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(UserMessage(content=prompt))

        t0 = time.monotonic()
        try:
            client = self._get_mistral_client()
            response = await client.chat.complete_async(
                model=resolved_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=ResponseFormat.model_validate(response_format) if response_format else None,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Mistral timeout: {exc}") from exc
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "rate limit" in msg.lower():
                raise LLMRateLimitError(f"Mistral rate limit: {exc}") from exc
            raise LLMProviderError(f"Mistral error: {exc}") from exc

        latency = (time.monotonic() - t0) * 1000
        usage = response.usage
        content = response.choices[0].message.content or ""
        prompt_tokens = usage.prompt_tokens if usage else _count_tokens((system or "") + prompt)
        completion_tokens = usage.completion_tokens if usage else _count_tokens(content)

        return LLMResponse(
            content=content,
            model=resolved_model,
            backend=LLMBackend.MISTRAL,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=_calculate_cost(resolved_model, LLMBackend.MISTRAL, prompt_tokens, completion_tokens),
            latency_ms=latency,
        )

    async def _complete_ollama(
        self,
        prompt: str,
        system: str | None,
        model: str | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        resolved_model = model or "llama3"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": resolved_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        t0 = time.monotonic()
        try:
            resp = await self._http.post(
                f"{settings.ollama_base_url}/api/chat", json=payload
            )
            resp.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Ollama timeout: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(f"Ollama HTTP {exc.response.status_code}: {exc}") from exc

        latency = (time.monotonic() - t0) * 1000
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        prompt_tokens = data.get("prompt_eval_count")
        if prompt_tokens is None:
            prompt_tokens = _count_tokens((system or "") + prompt)
        completion_tokens = data.get("eval_count")
        if completion_tokens is None:
            completion_tokens = _count_tokens(content)

        return LLMResponse(
            content=content,
            model=resolved_model,
            backend=LLMBackend.OLLAMA,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=0.0,
            latency_ms=latency,
        )

    async def close(self) -> None:
        """Close underlying HTTP clients."""
        await self._http.aclose()
        await self._groq.close()
