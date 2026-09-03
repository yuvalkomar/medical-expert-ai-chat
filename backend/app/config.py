from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    server_host: str = "0.0.0.0"
    server_port: int = Field(default=8000, ge=1, le=65535)
    database_url: str = "sqlite:///./data/medical_chat.db"

    llm_provider: str = "mock"
    llm_model: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    llm_temperature: float = Field(default=0.2, ge=0, le=1)
    llm_max_tokens: int = Field(default=1000, ge=1)

    max_retries: int = Field(default=3, ge=0)
    retry_delay: float = Field(default=2.0, ge=0)
    max_concurrency: int = Field(default=5, ge=1, le=100)
    shutdown_grace_period: float = Field(default=15.0, ge=0)

    log_level: str = "INFO"
    log_file: str = "./logs/medical_chat.log"

    aws_region: str = "us-east-1"
    aws_bedrock_endpoint_url: str | None = None

    mock_response_delay: float = Field(default=0.1, ge=0)
    mock_failures_before_success: int = Field(default=0, ge=0)

    @field_validator("llm_provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
        return normalized

    @model_validator(mode="after")
    def validate_provider_configuration(self) -> "Settings":
        if self.llm_provider not in {"mock", "bedrock"}:
            raise ValueError("LLM_PROVIDER must be either 'mock' or 'bedrock'")
        if self.llm_provider == "bedrock" and not self.llm_model.strip():
            raise ValueError("LLM_MODEL is required when LLM_PROVIDER=bedrock")
        if self.llm_provider == "bedrock" and not self.aws_region.strip():
            raise ValueError("AWS_REGION is required when LLM_PROVIDER=bedrock")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

