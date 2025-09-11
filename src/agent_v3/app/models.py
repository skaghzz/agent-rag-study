from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from agent_v3.app.config import embedding_cfg, llm_cfg


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


def get_llm(**kw) -> AzureChatOpenAI:
    return AzureChatOpenAI(azure_deployment=llm_cfg.deployment, azure_endpoint=llm_cfg.endpoint, api_key=llm_cfg.api_key, api_version=llm_cfg.api_version, **kw)


def chat(messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 800) -> str:
    llm = get_llm(temperature=temperature, max_tokens=max_tokens)
    resp = llm.invoke(_to_lc_messages(messages))
    return resp.content or ""


def get_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_deployment=embedding_cfg.deployment,
        openai_api_version=embedding_cfg.api_version,
        azure_endpoint=embedding_cfg.endpoint,
        api_key=embedding_cfg.api_key,
    )