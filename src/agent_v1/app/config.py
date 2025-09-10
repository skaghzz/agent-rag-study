import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class AzureOpenAIConfig:
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT","")
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY","")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION","2024-06-01")
    deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT","gpt-5")
    embedding_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT","text-embedding-3-large")

cfg = AzureOpenAIConfig()
