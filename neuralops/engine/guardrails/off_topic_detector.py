"""Off-topic detection using a fast LLM (Groq llama-3.1-8b-instant).

Given a system prompt context and a candidate response, asks the LLM
whether the response is topically relevant. Falls back to a keyword
blocklist check if the LLM call fails.
"""

from dataclasses import dataclass

from neuralops.core.logging import get_logger
from neuralops.services.llm_client import LLMBackend, LLMClient

logger = get_logger(__name__)

_OFF_TOPIC_SYSTEM = """You are a relevance classifier. Given a system prompt context and a
candidate response, determine whether the response stays on topic.

Respond ONLY with valid JSON:
{"is_relevant": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}"""

_OFF_TOPIC_USER_TEMPLATE = """System prompt context:
{system_context}

Candidate response:
{response}

Is this response relevant to the system prompt context?"""


@dataclass
class OffTopicResult:
    """Result of off-topic detection.

    Attributes:
        is_relevant: True if the response is on-topic.
        confidence: Model confidence (0.0–1.0).
        reason: Brief explanation from the LLM.
        method: "llm" or "fallback" (which detection path was used).
    """

    is_relevant: bool
    confidence: float
    reason: str
    method: str = "llm"


async def detect_off_topic(
    response: str,
    system_context: str,
    llm_client: LLMClient | None = None,
) -> OffTopicResult:
    """Detect whether a response is off-topic relative to a system prompt.

    Uses Groq llama-3.1-8b-instant for fast, cheap classification.
    Falls back to simple heuristic if the LLM call fails.

    Args:
        response: The LLM-generated response to evaluate.
        system_context: The system prompt that defines the topic.
        llm_client: Optional LLMClient instance; creates one if not provided.

    Returns:
        OffTopicResult with relevance decision and confidence.
    """
    if not system_context.strip():
        # No context defined — cannot judge relevance
        return OffTopicResult(is_relevant=True, confidence=1.0, reason="No context defined")

    client = llm_client or LLMClient()
    prompt = _OFF_TOPIC_USER_TEMPLATE.format(
        system_context=system_context[:1000],
        response=response[:1000],
    )

    try:
        import json

        llm_response = await client.complete(
            prompt=prompt,
            system=_OFF_TOPIC_SYSTEM,
            model="llama-3.1-8b-instant",
            backend=LLMBackend.GROQ,
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        data = json.loads(llm_response.content)
        return OffTopicResult(
            is_relevant=bool(data.get("is_relevant", True)),
            confidence=float(data.get("confidence", 0.5)),
            reason=str(data.get("reason", "")),
            method="llm",
        )
    except Exception as exc:
        logger.warning("Off-topic LLM call failed, using fallback", error=str(exc))
        return _fallback_detect(response, system_context)


def _fallback_detect(response: str, system_context: str) -> OffTopicResult:
    """Simple keyword overlap heuristic as a fallback.

    Checks whether any significant words from the system context appear
    in the response. This is intentionally conservative (assumes relevant).
    """
    import re

    context_words = set(re.findall(r"\b[a-z]{4,}\b", system_context.lower()))
    response_words = set(re.findall(r"\b[a-z]{4,}\b", response.lower()))

    stop_words = {
        "this", "that", "with", "from", "have", "will", "they", "been",
        "also", "more", "some", "when", "what", "your", "which", "about",
    }
    context_words -= stop_words
    response_words -= stop_words

    if not context_words:
        return OffTopicResult(is_relevant=True, confidence=0.5, reason="No context keywords", method="fallback")

    overlap = context_words & response_words
    overlap_ratio = len(overlap) / len(context_words)
    is_relevant = overlap_ratio >= 0.1  # At least 10% keyword overlap

    return OffTopicResult(
        is_relevant=is_relevant,
        confidence=round(min(overlap_ratio * 2, 1.0), 2),
        reason=f"Keyword overlap: {len(overlap)}/{len(context_words)} context words found",
        method="fallback",
    )
