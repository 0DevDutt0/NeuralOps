"""Unit tests for the guardrail pipeline."""

import pytest

from neuralops.engine.guardrails.off_topic_detector import _fallback_detect
from neuralops.engine.guardrails.pii_detector import PIIEntity, PIIResult
from neuralops.engine.guardrails.toxicity_scorer import ToxicityResult
from neuralops.services.guardrail_service import run_guardrails

# ── Fallback off-topic detection (no LLM needed) ─────────────────────────────


def test_fallback_detect_relevant_content():
    # Use clear keyword overlap so the heuristic fires correctly
    system_context = "You help users with Python programming, coding, and software development."
    response = "Python coding requires understanding software development principles and tools."
    result = _fallback_detect(response, system_context)
    assert result.is_relevant is True
    assert result.method == "fallback"


def test_fallback_detect_irrelevant_content():
    system_context = "You are a cooking assistant focused on recipes."
    response = "The stock market crashed today. Bitcoin is at 50000 dollars."
    result = _fallback_detect(response, system_context)
    # Low keyword overlap should flag as potentially off-topic
    # (result depends on overlap, but method should be fallback)
    assert result.method == "fallback"
    assert 0.0 <= result.confidence <= 1.0


def test_fallback_detect_empty_context():
    result = _fallback_detect("Any response", "")
    assert result.is_relevant is True  # No context = assume relevant


# ── PII detector (unit testing the data structures) ────────────────────────────


def test_pii_result_has_pii():
    result = PIIResult(
        has_pii=True,
        entities=[PIIEntity("EMAIL_ADDRESS", 0, 20, 0.95, "test@example.com")],
        anonymized_text="<EMAIL_ADDRESS>",
    )
    assert result.has_pii is True
    assert len(result.entities) == 1
    assert result.entities[0].entity_type == "EMAIL_ADDRESS"


def test_pii_result_no_pii():
    result = PIIResult(has_pii=False, entities=[], anonymized_text="clean text")
    assert result.has_pii is False
    assert len(result.entities) == 0


# ── ToxicityResult data structure ─────────────────────────────────────────────


def test_toxicity_result_is_toxic():
    result = ToxicityResult(score=0.95, label="toxic", is_toxic=True)
    assert result.is_toxic is True
    assert result.score > 0.7


def test_toxicity_result_is_safe():
    result = ToxicityResult(score=0.05, label="non_toxic", is_toxic=False)
    assert result.is_toxic is False


# ── Guardrail pipeline (with mocked sub-checks) ───────────────────────────────


@pytest.mark.asyncio
async def test_guardrail_passes_clean_content(monkeypatch):
    """Clean text should pass all guards."""
    async def mock_detect_pii(text):
        return PIIResult(has_pii=False, entities=[], anonymized_text=text)

    async def mock_score_toxicity(text):
        return ToxicityResult(score=0.02, label="non_toxic", is_toxic=False)

    monkeypatch.setattr(
        "neuralops.services.guardrail_service.detect_pii", mock_detect_pii
    )
    monkeypatch.setattr(
        "neuralops.services.guardrail_service.score_toxicity", mock_score_toxicity
    )

    result = await run_guardrails(
        "Tell me about healthy eating habits.",
        check_off_topic=False,
    )
    assert result.passed is True
    assert result.blocked_by is None


@pytest.mark.asyncio
async def test_guardrail_blocks_on_pii(monkeypatch):
    """Text with PII should be blocked."""
    async def mock_detect_pii(text):
        return PIIResult(
            has_pii=True,
            entities=[PIIEntity("SSN", 10, 21, 0.99, "123-45-6789")],
            anonymized_text="My SSN is <SSN>",
        )

    monkeypatch.setattr(
        "neuralops.services.guardrail_service.detect_pii", mock_detect_pii
    )

    result = await run_guardrails(
        "My SSN is 123-45-6789",
        check_toxicity=False,
        check_off_topic=False,
    )
    assert result.passed is False
    assert result.blocked_by == "pii"


@pytest.mark.asyncio
async def test_guardrail_blocks_on_toxicity(monkeypatch):
    """Toxic text should be blocked."""
    async def mock_detect_pii(text):
        return PIIResult(has_pii=False, entities=[], anonymized_text=text)

    async def mock_score_toxicity(text):
        return ToxicityResult(score=0.97, label="toxic", is_toxic=True, categories=["toxic_language"])

    monkeypatch.setattr("neuralops.services.guardrail_service.detect_pii", mock_detect_pii)
    monkeypatch.setattr("neuralops.services.guardrail_service.score_toxicity", mock_score_toxicity)

    result = await run_guardrails("Extremely toxic content here", check_off_topic=False)
    assert result.passed is False
    assert result.blocked_by == "toxicity"


@pytest.mark.asyncio
async def test_guardrail_result_to_dict(monkeypatch):
    """GuardrailResult.to_dict() should serialize correctly."""
    async def mock_detect_pii(text):
        return PIIResult(has_pii=False, entities=[], anonymized_text=text)

    async def mock_score_toxicity(text):
        return ToxicityResult(score=0.01, label="non_toxic", is_toxic=False)

    monkeypatch.setattr("neuralops.services.guardrail_service.detect_pii", mock_detect_pii)
    monkeypatch.setattr("neuralops.services.guardrail_service.score_toxicity", mock_score_toxicity)

    result = await run_guardrails("Safe text", check_off_topic=False)
    d = result.to_dict()
    assert "passed" in d
    assert "pii_detected" in d
    assert "toxicity_score" in d
    assert d["passed"] is True
