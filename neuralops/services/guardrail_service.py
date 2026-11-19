"""Guardrail service — orchestrates the full guardrail pipeline.

Runs PII detection, toxicity scoring, and off-topic detection
in parallel, then aggregates the results into a single decision.
"""

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any

from neuralops.core.config import settings
from neuralops.core.logging import get_logger
from neuralops.engine.guardrails.off_topic_detector import OffTopicResult, detect_off_topic
from neuralops.engine.guardrails.pii_detector import PIIResult, detect_pii
from neuralops.engine.guardrails.toxicity_scorer import ToxicityResult, score_toxicity
from neuralops.services.llm_client import LLMClient

logger = get_logger(__name__)


@dataclass
class GuardrailResult:
    """Aggregated result from the full guardrail pipeline.

    Attributes:
        passed: True if all checks passed (no blocking issues found).
        blocked_by: Which check caused a block ("pii", "toxicity", "off_topic", or None).
        pii: PII detection result.
        toxicity: Toxicity scoring result.
        off_topic: Off-topic detection result.
        warnings: Non-blocking warning messages.
    """

    passed: bool
    blocked_by: str | None
    pii: PIIResult
    toxicity: ToxicityResult
    off_topic: OffTopicResult
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage."""
        return {
            "passed": self.passed,
            "blocked_by": self.blocked_by,
            "pii_detected": self.pii.has_pii,
            "pii_entity_types": [e.entity_type for e in self.pii.entities],
            "toxicity_score": self.toxicity.score,
            "is_toxic": self.toxicity.is_toxic,
            "is_relevant": self.off_topic.is_relevant,
            "relevance_confidence": self.off_topic.confidence,
            "warnings": self.warnings,
        }


async def run_guardrails(
    text: str,
    system_context: str = "",
    llm_client: LLMClient | None = None,
    check_pii: bool = True,
    check_toxicity: bool = True,
    check_off_topic: bool = True,
) -> GuardrailResult:
    """Run all guardrail checks in parallel on input text.

    PII detection short-circuits: if PII is found, the response is blocked
    immediately and the text is not sent to further checks.

    Args:
        text: The text to evaluate (typically LLM output or user input).
        system_context: System prompt context for off-topic detection.
        llm_client: Optional LLMClient for off-topic LLM check.
        check_pii: Whether to run PII detection.
        check_toxicity: Whether to run toxicity scoring.
        check_off_topic: Whether to run off-topic detection.

    Returns:
        GuardrailResult with aggregated pass/fail decision.
    """
    warnings: list[str] = []

    # PII check first — block immediately if found
    pii_result = PIIResult(has_pii=False, entities=[], anonymized_text=text)
    if check_pii and settings.pii_enabled:
        try:
            pii_result = await detect_pii(text)
        except Exception as exc:
            logger.warning("PII detection failed", error=str(exc))
            warnings.append(f"PII detection unavailable: {exc}")

    # Run toxicity and off-topic in parallel (only if PII didn't block)
    tox_result = ToxicityResult(score=0.0, label="unknown", is_toxic=False)
    off_topic_result = OffTopicResult(is_relevant=True, confidence=1.0, reason="skipped")

    checks: list[Coroutine[Any, Any, ToxicityResult | OffTopicResult]] = []
    if check_toxicity:
        checks.append(_safe_toxicity(text, warnings))
    if check_off_topic and system_context:
        checks.append(_safe_off_topic(text, system_context, llm_client, warnings))

    for result in await asyncio.gather(*checks):
        if isinstance(result, ToxicityResult):
            tox_result = result
        else:
            off_topic_result = result

    # Determine if any check blocks
    blocked_by: str | None = None
    if pii_result.has_pii:
        blocked_by = "pii"
    elif tox_result.is_toxic:
        blocked_by = "toxicity"
    elif not off_topic_result.is_relevant and off_topic_result.confidence >= 0.8:
        blocked_by = "off_topic"

    passed = blocked_by is None

    if not passed:
        logger.warning(
            "Guardrail blocked content",
            blocked_by=blocked_by,
            pii=pii_result.has_pii,
            toxicity_score=tox_result.score,
            is_relevant=off_topic_result.is_relevant,
        )

    return GuardrailResult(
        passed=passed,
        blocked_by=blocked_by,
        pii=pii_result,
        toxicity=tox_result,
        off_topic=off_topic_result,
        warnings=warnings,
    )


async def _safe_toxicity(text: str, warnings: list[str]) -> ToxicityResult:
    """Run toxicity check, catching errors gracefully."""
    try:
        return await score_toxicity(text)
    except Exception as exc:
        logger.warning("Toxicity scoring failed", error=str(exc))
        warnings.append(f"Toxicity scoring unavailable: {exc}")
        return ToxicityResult(score=0.0, label="error", is_toxic=False)


async def _safe_off_topic(
    text: str,
    system_context: str,
    llm_client: LLMClient | None,
    warnings: list[str],
) -> OffTopicResult:
    """Run off-topic check, catching errors gracefully."""
    try:
        return await detect_off_topic(text, system_context, llm_client)
    except Exception as exc:
        logger.warning("Off-topic detection failed", error=str(exc))
        warnings.append(f"Off-topic detection unavailable: {exc}")
        return OffTopicResult(is_relevant=True, confidence=0.5, reason="error — assumed relevant")
