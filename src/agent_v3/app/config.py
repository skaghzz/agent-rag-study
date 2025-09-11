import os
from dataclasses import dataclass

# from dotenv import load_dotenv

# load_dotenv()

# ----------------------------------------
# 1) 언어 모델 전용 설정
# ----------------------------------------
@dataclass(frozen=True)
class AzureOpenAILanguageModelConfig:
    """Azure OpenAI GPT 계열 언어 모델 설정."""
    endpoint: str = os.getenv("AZURE_OPENAI_LANGUAGE_MODEL_ENDPOINT", "")
    api_key: str = os.getenv("AZURE_OPENAI_LANGUAGE_MODEL_API_KEY", "")
    api_version: str = os.getenv("AZURE_OPENAI_LANGUAGE_MODEL_API_VERSION", "2024-06-01")
    deployment: str = os.getenv("AZURE_OPENAI_LANGUAGE_MODEL_DEPLOYMENT", "gpt-4.1")

# ----------------------------------------
# 2) 임베딩 모델 전용 설정
# ----------------------------------------
@dataclass(frozen=True)
class AzureOpenAIEmbeddingModelConfig:
    """Azure OpenAI 임베딩 모델 설정."""
    endpoint: str = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", "")
    api_key: str = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY", "")
    api_version: str = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2023-05-15")
    deployment: str = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
    )

@dataclass(frozen=True)
class Flags:
    use_llm_router: bool = os.getenv("USE_LLM_ROUTER","true").lower()=="true"
    use_llm_grader: bool = os.getenv("USE_LLM_GRADER","true").lower()=="true"

@dataclass(frozen=True)
class AzureSearchConfig:
    """Configuration for Azure AI Search service."""

    endpoint: str = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    key: str = os.getenv("AZURE_SEARCH_KEY", "")
    index: str = os.getenv("AZURE_SEARCH_INDEX", "common-sense-index")


# ----------------------------------------
# 3) 전역 설정 인스턴스
# ----------------------------------------
# 언어 모델 & 임베딩 모델
llm_cfg = AzureOpenAILanguageModelConfig()
embedding_cfg = AzureOpenAIEmbeddingModelConfig()

flags = Flags()
azure_search_cfg = AzureSearchConfig()
