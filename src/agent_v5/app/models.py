from functools import lru_cache
from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from agent_v5.app.config import embedding_cfg, llm_cfg


def _to_lc_messages(messages: List[Dict[str, str]]):
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


def chat(messages: List[Dict[str, str]]) -> str:
    """Simple chat helper returning raw content."""

    resp = get_llm().invoke(_to_lc_messages(messages))
    content = resp.content
    if isinstance(content, str):
        return content
    # LLM이 리스트나 기타 타입을 반환하는 엣지 케이스 대응
    return str(content)


# ---------------------------------------------------------------------------
# LLM singleton factory ------------------------------------------------------
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_llm() -> AzureChatOpenAI:
    """Return a cached AzureChatOpenAI instance."""

    return AzureChatOpenAI(
        azure_deployment=llm_cfg.deployment,
        azure_endpoint=llm_cfg.endpoint,
        api_key=llm_cfg.api_key,
        api_version=llm_cfg.api_version,
    )


# ---------------------------------------------------------------------------
# get_embeddings -------------------------------------------------------------
# ---------------------------------------------------------------------------


def get_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_deployment=embedding_cfg.deployment,
        openai_api_version=embedding_cfg.api_version,
        azure_endpoint=embedding_cfg.endpoint,
        api_key=embedding_cfg.api_key,
    )
