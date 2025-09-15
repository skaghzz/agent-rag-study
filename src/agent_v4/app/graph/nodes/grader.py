# from app.config import flags
# from app.models import chat
# import json
# REL_SYS = "You judge if candidate passage is relevant. Return JSON {'relevant':true|false}."
# def _llm_filter(q, docs):
#     out=[]
#     for d in docs:
#         payload={"question":q, "candidate":d.get("content","")[:1200]}
#         try:
#             res=chat([{"role":"system","content":REL_SYS},{"role":"user","content":json.dumps(payload)}],0.0,128)
#             data=json.loads(res)
#             if bool(data.get("relevant", True)): out.append(d)
#         except Exception:
#             out.append(d)
#     return out
# def evidence_grader(state):
#     kb=state.get("kb_docs",[]) or []
#     w1=state.get("web_docs_bing",[]) or []
#     w2=state.get("web_docs_google",[]) or []
#     docs=kb+w1+w2
#     if flags.use_llm_grader:
#         user=next((m for m in reversed(state.get("messages",[])) if m.get("role")=="user"),{"content":""})
#         docs=_llm_filter(user["content"], docs)
#     st=dict(state); st["evidence"]=docs[:6]; return st

import json
import logging
from typing import Any, Dict, List, Union, cast

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from agent_v4.app.config import flags
from agent_v4.app.graph.state import GraphState
from agent_v4.app.models import get_llm
from agent_v4.app.utils.messages import last_user_text as _last_user_text

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

REL_SYS = "You judge if candidate passage is relevant. Return STRICT JSON with key 'relevant'(true/false)."


# ---------------------------------------------------------------------------
# Pydantic structured output schema ----------------------------------------
# ---------------------------------------------------------------------------


class _RelResult(BaseModel):
    """Validated schema for relevance classification."""

    relevant: bool = Field(default=True, description="whether evidence is relevant")


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


def evidence_grader(state: GraphState) -> Dict[str, Any]:
    kb_raw = state.get("kb_docs", []) or []
    kb_docs: List[Union[Document, Dict[str, Any]]] = cast(List[Union[Document, Dict[str, Any]]], kb_raw)
    
    web_raw = state.get("web_docs", []) or []
    web_docs: List[Union[Document, Dict[str, Any]]] = cast(List[Union[Document, Dict[str, Any]]], web_raw)
    
    docs = kb_docs + web_docs
    # Convert any Document objects to dicts for uniformity.
    ev: List[Dict[str, Any]] = [_doc_to_dict(d) if isinstance(d, Document) else d for d in docs]
    user_text = _last_user_text(state)
    if flags.use_llm_grader:
        ev = _llm_filter(user_text, ev)

    return {"evidence": ev}
