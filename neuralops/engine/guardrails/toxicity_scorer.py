"""Toxicity scoring using the unitary/toxic-bert HuggingFace model.

The pipeline is loaded once at first use and kept in memory.
Inference is run in a thread executor to avoid blocking the event loop.
On Windows without CUDA, torch defaults to CPU — this is slower but functional.
"""

import asyncio
from dataclasses import dataclass, field

from neuralops.core.config import settings
from neuralops.core.logging import get_logger

logger = get_logger(__name__)

_toxicity_pipeline = None
_MODEL_NAME = "unitary/toxic-bert"
_MAX_INPUT_LENGTH = 512  # BERT's context window


def _get_pipeline():
    """Lazily load toxic-bert pipeline on first call."""
    global _toxicity_pipeline
    if _toxicity_pipeline is not None:
        return _toxicity_pipeline

    try:
        import torch
        from transformers import pipeline  # type: ignore[import]

        device_str = settings.toxicity_device
        if device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device_str = "cpu"

        device = 0 if device_str == "cuda" else -1
        _toxicity_pipeline = pipeline(
            "text-classification",
            model=_MODEL_NAME,
            device=device,
            truncation=True,
            max_length=_MAX_INPUT_LENGTH,
        )
        logger.info("Toxicity pipeline loaded", model=_MODEL_NAME, device=device_str)
    except ImportError:
        logger.error("transformers or torch not installed — toxicity scoring disabled")
        _toxicity_pipeline = None

    return _toxicity_pipeline


@dataclass
class ToxicityResult:
    """Result of toxicity scoring.

    Attributes:
        score: Toxicity probability (0.0 = safe, 1.0 = toxic).
        label: Raw model label ("toxic" or "non_toxic").
        is_toxic: True if score >= configured threshold.
        categories: List of detected toxicity type strings (best-effort).
    """

    score: float
    label: str
    is_toxic: bool
    categories: list[str] = field(default_factory=list)


def _score_sync(text: str) -> ToxicityResult:
    """Run toxicity inference synchronously.

    Args:
        text: Text to score. Truncated to _MAX_INPUT_LENGTH tokens.

    Returns:
        ToxicityResult with score and label.
    """
    pipe = _get_pipeline()
    if pipe is None:
        return ToxicityResult(score=0.0, label="unknown", is_toxic=False)

    # Truncate by characters as a rough guard before tokenizer
    truncated = text[:2000]
    result = pipe(truncated)[0]

    label: str = result["label"]
    raw_score: float = result["score"]

    # The model outputs "toxic" with high score = bad; "non_toxic" with high score = safe
    # Normalize: toxicity_score = score if toxic, 1-score if non_toxic
    if label == "toxic":
        toxicity_score = raw_score
    else:
        toxicity_score = 1.0 - raw_score

    is_toxic = toxicity_score >= settings.toxicity_threshold

    return ToxicityResult(
        score=round(toxicity_score, 4),
        label=label,
        is_toxic=is_toxic,
        categories=["toxic_language"] if is_toxic else [],
    )


async def score_toxicity(text: str) -> ToxicityResult:
    """Score text for toxicity asynchronously.

    Runs the model in a thread executor to avoid blocking the event loop.

    Args:
        text: Text to score.

    Returns:
        ToxicityResult with probability score and threshold decision.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _score_sync, text)
