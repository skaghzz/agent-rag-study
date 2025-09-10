from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from agent_v1.app.config import cfg


def _to_lc_messages(messages: list[dict[str, str]]):
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


def chat(messages: list[dict[str, str]], temperature: float = 0.2, max_tokens: int = 800) -> str:
    llm = AzureChatOpenAI(
        azure_deployment=cfg.deployment,
        azure_endpoint=cfg.endpoint,
        api_key=cfg.api_key,
        api_version=cfg.api_version,
    )
    resp = llm.invoke(_to_lc_messages(messages))
    content = resp.content
    if isinstance(content, str):
        return content
    # LLM이 리스트나 기타 타입을 반환하는 엣지 케이스 대응
    return str(content)
