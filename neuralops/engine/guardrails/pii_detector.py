"""PII detection using Microsoft Presidio.

Loaded lazily at first use to avoid blocking startup. The AnalyzerEngine
holds a spaCy NLP model in memory; all inference is run in an executor
thread so it does not block the async event loop.

Prerequisites:
    python -m spacy download en_core_web_lg
"""

import asyncio
from dataclasses import dataclass

from neuralops.core.logging import get_logger

logger = get_logger(__name__)

_analyzer = None
_anonymizer = None


def _get_analyzer():
    """Lazily initialize Presidio AnalyzerEngine.

    Falls back from en_core_web_lg to en_core_web_sm if the large model
    is not installed.
    """
    global _analyzer
    if _analyzer is not None:
        return _analyzer

    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore[import]
        from presidio_analyzer.nlp_engine import NlpEngineProvider  # type: ignore[import]

        try:
            provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
            })
            nlp_engine = provider.create_engine()
        except OSError:
            logger.warning("en_core_web_lg not found, falling back to en_core_web_sm")
            provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            })
            nlp_engine = provider.create_engine()

        _analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        logger.info("Presidio AnalyzerEngine initialized")
    except ImportError:
        logger.error("presidio-analyzer not installed — PII detection disabled")
        _analyzer = None

    return _analyzer


def _get_anonymizer():
    """Lazily initialize Presidio AnonymizerEngine."""
    global _anonymizer
    if _anonymizer is not None:
        return _anonymizer
    try:
        from presidio_anonymizer import AnonymizerEngine  # type: ignore[import]

        _anonymizer = AnonymizerEngine()
    except ImportError:
        _anonymizer = None
    return _anonymizer


@dataclass
class PIIEntity:
    """A detected PII entity.

    Attributes:
        entity_type: e.g. "PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON".
        start: Start character offset in the analyzed text.
        end: End character offset.
        score: Confidence score (0.0–1.0).
        text: The actual detected text span.
    """

    entity_type: str
    start: int
    end: int
    score: float
    text: str


@dataclass
class PIIResult:
    """Result of a PII analysis pass.

    Attributes:
        has_pii: True if any PII was detected.
        entities: List of detected PIIEntity objects.
        anonymized_text: Input text with PII replaced by type tokens.
    """

    has_pii: bool
    entities: list[PIIEntity]
    anonymized_text: str


def _analyze_sync(text: str) -> PIIResult:
    """Run PII analysis synchronously (called in executor).

    Args:
        text: Text to analyze.

    Returns:
        PIIResult with detected entities and anonymized text.
    """
    analyzer = _get_analyzer()
    if analyzer is None:
        return PIIResult(has_pii=False, entities=[], anonymized_text=text)

    results = analyzer.analyze(text=text, language="en")
    entities = [
        PIIEntity(
            entity_type=r.entity_type,
            start=r.start,
            end=r.end,
            score=round(r.score, 4),
            text=text[r.start:r.end],
        )
        for r in results
    ]

    anonymized_text = text
    anonymizer = _get_anonymizer()
    if anonymizer and results:
        from presidio_anonymizer.entities import RecognizerResult  # type: ignore[import]

        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=[
                RecognizerResult(
                    entity_type=r.entity_type,
                    start=r.start,
                    end=r.end,
                    score=r.score,
                )
                for r in results
            ],
        )
        anonymized_text = anonymized.text

    return PIIResult(has_pii=bool(entities), entities=entities, anonymized_text=anonymized_text)


async def detect_pii(text: str) -> PIIResult:
    """Detect PII in text asynchronously.

    Runs Presidio analysis in a thread executor to avoid blocking the event loop.

    Args:
        text: Text to analyze for PII.

    Returns:
        PIIResult with detection results and anonymized text.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _analyze_sync, text)
