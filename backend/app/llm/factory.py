from backend.app.config import Settings
from backend.app.llm.base import LLMProvider
from backend.app.llm.bedrock import BedrockLLMProvider
from backend.app.llm.mock import MockLLMProvider


def create_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "mock":
        return MockLLMProvider(
            response_delay=settings.mock_response_delay,
            failures_before_success=settings.mock_failures_before_success,
        )
    if settings.llm_provider == "bedrock":
        return BedrockLLMProvider(
            model_id=settings.llm_model,
            region_name=settings.aws_region,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            endpoint_url=settings.aws_bedrock_endpoint_url,
        )
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

