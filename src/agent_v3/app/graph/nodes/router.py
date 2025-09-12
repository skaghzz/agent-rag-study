from typing import Any, Literal

from pydantic import BaseModel, Field

# Local imports --------------------------------------------------------------
from agent_v3.app.config import flags
from agent_v3.app.graph.state import GraphState
from agent_v3.app.models import get_llm

__all__ = [
    "RouterResult",
    "planner_router",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUTER_SYS_PROMPT: str = (
    "Return STRICT JSON with keys intent(one of 'followup','new_topic','ambiguous','task'), need_web(true/false)."
)


# ---------------------------------------------------------------------------
# Pydantic model for structured & validated output ---------------------------
# ---------------------------------------------------------------------------


class RouterResult(BaseModel):
    """Validated schema for LLM router output."""

    intent: Literal["followup", "new_topic", "ambiguous", "task"] = Field(
        default="new_topic", description="intent"
    )
    need_web: bool = Field(default=False, description="need to search web")


def _content_to_text(content: Any) -> str:
    """멀티모달 대비: str 또는 [{type:'text', text:'...'}] 지원."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        try:
            parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            return " ".join(t for t in parts if t)
        except Exception:
            pass
    return str(content or "")


def _last_user_text(state: GraphState) -> str:
    """MessagesState/원시 dict 모두 지원하여 '최근 사용자 텍스트' 추출."""
    msgs = state.get("messages", []) or []
    for m in reversed(msgs):
        # OpenAI 스타일 dict
        if isinstance(m, dict) and m.get("role") == "user":
            return _content_to_text(m.get("content", ""))
        # LangChain 메시지 객체
        mtype = getattr(m, "type", None)
        if mtype == "human":  # langchain_core.messages.HumanMessage
            return _content_to_text(getattr(m, "content", ""))
    return ""


# ---------------------------------------------------------------------------
# Public API -----------------------------------------------------------------
# ---------------------------------------------------------------------------


def planner_router(state: GraphState) -> RouterResult:  # noqa: D401
    """Return the user's *intent* and whether a **web** lookup is required.

    The function contains a fast-path that bypasses the LLM entirely when
    ``flags.use_llm_router`` is *false* or when the user's last chat message
    cannot be determined.
    """

    def _fallback() -> RouterResult:
        """Return a RouterResult using values already present in *state*."""

        return RouterResult(
            intent=state.get("intent", "new_topic"),
            need_web=state.get("need_web", False),
        )

    # Fast-path – skip the LLM call entirely if disabled in config.
    if not flags.use_llm_router:
        return _fallback()

    user_text: str = _last_user_text(state)
    if not user_text:
        return _fallback()

    try:
        llm = get_llm().with_structured_output(RouterResult)
        result = RouterResult.model_validate(
            llm.invoke(
                [
                    {"role": "system", "content": ROUTER_SYS_PROMPT},
                    {"role": "user", "content": user_text},
                ]
            )
        )

    except Exception:
        # Gracefully degrade to the fallback result in case *anything* goes
        # wrong – network issues, validation errors, etc.
        return _fallback()

    return RouterResult(intent=result.intent, need_web=result.need_web)
