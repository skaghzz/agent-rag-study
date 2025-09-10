import re
from typing import Any, Dict, List, Tuple

from agent_v2.app.graph.state import GraphState
from agent_v2.app.models import chat

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYS = (
    "You are a precise assistant. Answer in Korean if the question is Korean. "
    "Use ONLY the provided EVIDENCE. "
    "Cite inline with [KB:file] or [WEB:url] IMMEDIATELY after each claim. "
    "If no evidence supports a claim, write '근거 없음' for that part. "
    "If there is no evidence at all, say you lack evidence and avoid unverifiable facts."
)

MAX_SNIPPET = 500
MAX_ITEMS = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _content_to_text(content: Any) -> str:
    """str 또는 [{type:'text', text:'...'}] 멀티모달 대비."""
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
    """최근 사용자 메시지 텍스트만 안전하게 추출."""
    msgs = state.get("messages", []) or []
    for m in reversed(msgs):
        # OpenAI 스타일 dict
        if isinstance(m, dict) and m.get("role") == "user":
            return _content_to_text(m.get("content", ""))
        # LangChain 메시지 객체 (HumanMessage)
        if getattr(m, "type", None) == "human":
            return _content_to_text(getattr(m, "content", ""))
    return ""

def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[: n - 1] + "…")

def _split_source(src: str) -> Tuple[str, str]:
    """'KB:foo' 또는 'WEB:https://…' -> ('KB','foo') 형태로 분리."""
    if not src or ":" not in src:
        return ("", src or "")
    kind, rest = src.split(":", 1)
    return (kind.strip().upper(), rest.strip())

def _dedup_evidence(ev: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """source 기준 중복 제거(후행 항목으로 갱신)."""
    seen = {}
    for e in ev or []:
        seen[e.get("source", "")] = e
    return list(seen.values())

def _format_evidence(ev: List[Dict[str, Any]]) -> str:
    """
    모델이 읽기 쉬운 단문 목록으로 구성.
    각 항목: '- <TAG> <display>: <snippet>'
    TAG는 'KB' 또는 'WEB'
    """
    lines: List[str] = []
    for e in _dedup_evidence(ev)[:MAX_ITEMS]:
        src = e.get("source", "")
        kind, disp = _split_source(src)
        kind = "KB" if kind == "KB" else ("WEB" if kind == "WEB" else (kind or "SRC"))
        title = e.get("title") or ""
        content = e.get("content") or e.get("text") or ""
        # 제목이 있으면 표시, 없으면 display만
        head = title.strip() or disp
        # 과도한 공백 제거
        snippet = re.sub(r"\s+", " ", _clip(content, MAX_SNIPPET))
        lines.append(f"- {kind}:{disp} :: {head} :: {snippet}")
    return "\n".join(lines) if lines else "(none)"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
def generate(state: GraphState) -> Dict[str, Any]:
    """
    사용자 질문 + 증거 리스트를 기반으로 최종 답변 생성.
    반환: {'answer': <str>, 'step': <int>}
    """
    user_text = _last_user_text(state)
    evidence = state.get("evidence", []) or []

    evidence_block = _format_evidence(evidence)

    user_prompt = (
        "[QUESTION]\n"
        f"{user_text}\n\n"
        "[EVIDENCE]\n"
        f"{evidence_block}\n\n"
        "[INSTRUCTIONS]\n"
        "- Use ONLY the EVIDENCE above.\n"
        "- Cite each claim inline: [KB:file] or [WEB:url].\n"
        "- If no evidence exists for a point, write '근거 없음'.\n"
        "- Be concise and structured (bullets or short paragraphs)."
    )

    try:
        out = chat(
            [
                {"role": "system", "content": SYS},
                {"role": "user", "content": user_prompt},
            ]
        )
        answer = out if isinstance(out, str) else str(out)
    except Exception:
        # 모델 호출 실패 시 최소한의 폴백 제공
        if evidence:
            # 증거 요약만 간단히 노출
            answer = "죄송해요, 답변 생성 중 오류가 발생했습니다. 아래는 수집된 근거입니다:\n" + _format_evidence(evidence)
        else:
            answer = "죄송해요, 현재는 근거 자료가 없어 확답을 드리기 어려워요."

    return {
        "answer": answer.strip(),
    }
