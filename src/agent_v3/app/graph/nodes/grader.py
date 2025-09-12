import json
import logging
from typing import Any, Dict, List, Union, cast

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from agent_v3.app.config import flags
from agent_v3.app.graph.state import GraphState
from agent_v3.app.models import get_llm

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

REL_SYS = (
    "You judge if candidate passage is relevant. Return STRICT JSON with key 'relevant'(true/false)."
)


# ---------------------------------------------------------------------------
# Pydantic structured output schema ----------------------------------------
# ---------------------------------------------------------------------------


class _RelResult(BaseModel):
    """Validated schema for relevance classification."""

    relevant: bool = Field(default=True, description="whether evidence is relevant")

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

def _doc_to_dict(doc: Document) -> Dict[str, Any]:
    """Convert a langchain Document to a plain dict for downstream processing."""
    return {"content": getattr(doc, "page_content", ""), "metadata": getattr(doc, "metadata", {})}


def _llm_filter(question: str, evidences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter evidences using an LLM-based relevance classifier.

    Args:
        question: 사용자 질문 문자열.
        evidences: 검색된 후보 문서 리스트.

    Returns:
        relevance 판정 결과가 *True* 인 evidence 목록.
    """
    filtered: List[Dict[str, Any]] = []
    # Prepare LLM with structured output once per batch for efficiency.
    llm = get_llm().with_structured_output(_RelResult)

    for evidence in evidences:
        payload: Dict[str, str] = {
            "question": question,
            "candidate": evidence.get("content", ""),
        }

        try:
            result = _RelResult.model_validate(
                llm.invoke(
                    [
                        {"role": "system", "content": REL_SYS},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
                    ]
                )
            )
            relevant: bool = result.relevant
        except Exception as exc:
            # Gracefully degrade if anything goes wrong – network, validation etc.
            logger.exception("LLM filtering failed: %s", exc)
            relevant = True

        if relevant:
            filtered.append(evidence)
    return filtered


def evidence_grader(state: GraphState) ->  Dict[str, Any]:
    kb_raw = state.get("kb_docs", []) or []
    kb_docs: List[Union[Document, Dict[str, Any]]] = cast(List[Union[Document, Dict[str, Any]]], kb_raw)
    # Convert any Document objects to dicts for uniformity.
    ev: List[Dict[str, Any]] = [_doc_to_dict(d) if isinstance(d, Document) else d for d in kb_docs][:6]
    user_text = _last_user_text(state)
    if flags.use_llm_grader:
        ev = _llm_filter(user_text, ev)

    return {
        "evidence": ev
    }
