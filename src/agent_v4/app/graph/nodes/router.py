from typing import Literal

from pydantic import BaseModel, Field

# Local imports --------------------------------------------------------------
from agent_v4.app.config import flags
from agent_v4.app.graph.state import GraphState
from agent_v4.app.models import get_llm
from agent_v4.app.utils.messages import last_user_text as _last_user_text

__all__ = [
    "RouterResult",
    "planner_router",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUTER_SYS_PROMPT: str = "Return STRICT JSON with keys intent(one of 'followup','new_topic','ambiguous','task'), need_web(true/false)."


# ---------------------------------------------------------------------------
# Pydantic model for structured & validated output ---------------------------
# ---------------------------------------------------------------------------


class RouterResult(BaseModel):
    """Validated schema for LLM router output."""

    intent: Literal["followup", "new_topic", "ambiguous", "task"] = Field(default="new_topic", description="intent")
    need_web: bool = Field(default=False, description="need to search web")


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
