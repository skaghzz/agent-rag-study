from typing import Any

from agent_v6.app.graph.state import GraphState

__all__ = ["last_user_text", "content_to_text"]


def content_to_text(content: Any) -> str:
    """Extract text from various content representations.

    Supports:
    1. plain ``str``
    2. list of dicts like ``[{"type": "text", "text": "..."}]`` (multi-modal)

    Args:
        content: Raw content value.

    Returns:
        Extracted plain text string.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        try:
            parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            return " ".join(t for t in parts if t)
        except Exception:
            # Fallback to string representation below
            pass

    return str(content or "")


def last_user_text(state: GraphState) -> str:
    """Return the most recent *user* text message from a ``GraphState``.

    The helper supports:
    1. OpenAI style message dicts: ``{"role": "user", "content": "..."}``
    2. LangChain ``HumanMessage`` objects (``m.type == "human"``)

    If no user text is found an empty string is returned.
    """
    messages = state.get("messages", []) or []

    for message in reversed(messages):
        # OpenAI style dict
        if isinstance(message, dict) and message.get("role") == "user":
            return content_to_text(message.get("content", ""))

        # LangChain message object
        m_type = getattr(message, "type", None)
        if m_type == "human":  # langchain_core.messages.HumanMessage
            return content_to_text(getattr(message, "content", ""))

    return ""
