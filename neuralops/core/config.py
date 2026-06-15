"""Application configuration via Pydantic Settings.

All runtime configuration is read from environment variables (or .env file).
Import the singleton `settings` object — never instantiate Settings directly.
"""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for NeuralOps.

    Attributes:
        llm_provider: Default LLM backend ("groq" | "mistral" | "ollama").
        groq_api_key: Groq API key (accessed via OpenAI SDK with base_url override).
        mistral_api_key: Mistral API key.
        ollama_base_url: Base URL for local Ollama server.
        database_url: SQLAlchemy database URL (sync or async driver).
        redis_url: Redis connection URL.
        neuralops_port: Port for the FastAPI server.
        neuralops_environment: "development" | "production" | "test".
        secret_key: Used for signing tokens/sessions.
        judge_model: Model name for LLM-as-Judge scoring.
        judge_backend: Backend for judge ("groq" | "mistral").
        experiment_significance_level: p-value threshold for A/B winner auto-promotion.
        drift_alert_threshold: Composite score below which drift alerts fire.
        drift_check_interval_minutes: How often the drift scheduler runs.
        toxicity_threshold: Float 0-1; scores above this are flagged.
        pii_enabled: Whether PII detection is active.
        toxicity_device: "cpu" or "cuda" for the transformers pipeline.
        telegram_bot_token: Telegram bot token for alerting.
        telegram_chat_id: Telegram chat ID to send alerts to.
        dashboard_api_url: URL the Streamlit dashboard uses to call the API.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    llm_provider: str = "groq"
    groq_api_key: str = ""
    mistral_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # Database
    database_url: str = "sqlite+aiosqlite:///./neuralops_dev.db"
    redis_url: str = "redis://localhost:6379"

    # Application
    neuralops_port: int = 8000
    neuralops_environment: str = "development"
    secret_key: str = "dev_secret_change_in_production"

    # Judge
    judge_model: str = "llama-3.3-70b-versatile"
    judge_backend: str = "groq"
    experiment_significance_level: float = 0.05

    # Drift
    drift_alert_threshold: float = 6.5
    drift_check_interval_minutes: int = 60

    # Guardrails
    toxicity_threshold: float = 0.7
    pii_enabled: bool = True
    toxicity_device: str = "cpu"

    # Alerting
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Dashboard
    dashboard_api_url: str = "http://localhost:8000"

    @property
    def async_database_url(self) -> str:
        """Return database URL with the correct async driver prefix."""
        url = self.database_url
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        if url.startswith("sqlite://"):
            return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
        return url

    @property
    def is_development(self) -> bool:
        """True when running in development mode."""
        return self.neuralops_environment == "development"

    @property
    def is_test(self) -> bool:
        """True when running under pytest."""
        return self.neuralops_environment == "test"

    @field_validator("experiment_significance_level")
    @classmethod
    def validate_significance(cls, v: float) -> float:
        """Ensure significance level is in (0, 1)."""
        if not 0 < v < 1:
            raise ValueError("experiment_significance_level must be between 0 and 1")
        return v

    @field_validator("toxicity_threshold")
    @classmethod
    def validate_toxicity_threshold(cls, v: float) -> float:
        """Ensure toxicity threshold is in [0, 1]."""
        if not 0 <= v <= 1:
            raise ValueError("toxicity_threshold must be between 0 and 1")
        return v


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    return Settings()


settings: Settings = get_settings()
