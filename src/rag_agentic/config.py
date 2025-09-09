"""Configuration management using Pydantic settings."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables or `.env` file.

    Attributes
    ----------
    azure_openai_api_key: str
        Azure OpenAI API key.
    azure_openai_endpoint: str
        Azure OpenAI endpoint URL.
    azure_openai_deployment: str
        Deployment name for the GPT-5 chat model.
    azure_openai_embeddings_deployment: str
        Deployment name for the `text-embedding-ada-002` model.
    temperature: float
        Sampling temperature for generation.
    """

    azure_openai_api_key: str = Field("", validation_alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field("", validation_alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment: str = Field("gpt-5", validation_alias="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_embeddings_deployment: str = Field(
        "text-embedding-ada-002", validation_alias="AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"
    )
    temperature: float = Field(0.0, validation_alias="AZURE_OPENAI_TEMPERATURE")

    class Config:
        env_file = ".env"
        case_sensitive = False

    # lazy validate endpoint URL ensures correct schema
    @model_validator(mode="before")
    def _check_endpoint(cls, values: dict[str, object]) -> dict[str, object]:
        endpoint: Optional[str] = values.get("azure_openai_endpoint")  # type: ignore[assignment]
        if endpoint and not endpoint.startswith("https://"):
            values["azure_openai_endpoint"] = f"https://{endpoint}"
        return values


@lru_cache
def get_settings() -> Settings:  # noqa: D401
    """Return a cached `Settings` instance.

    BaseSettings 는 환경 변수에서 값을 자동으로 로드하므로
    인자를 전달하지 않고 인스턴스를 생성해도 안전합니다.
    정적 타입 체커(pyright/mypy)의 오경고를 무시하기 위해
    `type: ignore[call-arg]` 주석을 추가했습니다.
    """
    # NOTE: Static type checker does not understand BaseSettings' env loading.
    return Settings()  # type: ignore[call-arg]
