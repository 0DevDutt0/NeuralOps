"""Unit tests for the logging configuration module."""

import logging

from neuralops.core.logging import configure_logging, get_logger


def test_get_logger_returns_logger():
    logger = get_logger("test.module")
    assert logger is not None


def test_get_logger_different_names():
    a = get_logger("module.a")
    b = get_logger("module.b")
    # Both are valid logger objects
    assert a is not None
    assert b is not None


def test_configure_logging_json_mode():
    configure_logging(level="INFO", json_logs=True)
    root = logging.getLogger()
    assert root.level <= logging.INFO


def test_configure_logging_dev_mode():
    configure_logging(level="DEBUG", json_logs=False)
    root = logging.getLogger()
    assert root.level <= logging.DEBUG


def test_configure_logging_silences_noisy_loggers():
    configure_logging(level="INFO", json_logs=True)
    for name in ("uvicorn.access", "sqlalchemy.engine", "httpx", "httpcore"):
        assert logging.getLogger(name).level == logging.WARNING


def test_get_logger_usable():
    logger = get_logger("test")
    # Should not raise
    logger.info("test log message", key="value")
