import json
import re
from typing import Any, Dict, List, Optional

from agent_v2.app.graph.state import GraphState
from agent_v2.app.models import chat

SYS = (
    "You rewrite the user's last question into 2 search-friendly queries. "
    'Return a JSON array of exactly 2 short strings, e.g. ["...","..."] '
    "No explanations."
)

# ---- helpers ---------------------------------------------------------------


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


def _coerce_queries_from_json_str(s: str) -> Optional[List[str]]:
    """모델 출력이 JSON에 가깝지만 형식이 지저분한 경우까지 복구 시도."""
    s = s.strip()
    # 1) 정규 JSON 시도
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, dict) and "queries" in obj and isinstance(obj["queries"], list):
            return [str(x) for x in obj["queries"]]
    except Exception:
        pass
    # 2) 대괄호 슬라이스로 재시도 (불필요한 텍스트가 앞뒤에 붙은 경우)
    if "[" in s and "]" in s:
        s2 = s[s.find("[") : s.rfind("]") + 1]
        try:
            obj = json.loads(s2)
            if isinstance(obj, list):
                return [str(x) for x in obj]
        except Exception:
            pass
    return None


def _sanitize_queries(candidates: List[str], fallback: str, k: int = 2, max_len: int = 256) -> List[str]:
    """중복/공백 제거, 길이 제한, k개 보장(부족하면 fallback로 채움)."""
    cleaned = []
    seen = set()
    for q in candidates:
        q = (q or "").strip()
        if not q:
            continue
        q = re.sub(r"\s+", " ", q)[:max_len]
        if q and q.lower() not in seen:
            cleaned.append(q)
            seen.add(q.lower())
        if len(cleaned) >= k:
            break
    if not cleaned:
        cleaned = [fallback.strip()][:1]
    while len(cleaned) < k:
        cleaned.append(fallback.strip())
    return cleaned[:k]


# ---- node ------------------------------------------------------------------


def query_rewrite(state: GraphState) -> Dict[str, Any]:
    """
    - 마지막 사용자 입력을 찾아 2개의 검색친화 쿼리로 리라이트
    - JSON 강인 파싱 + 후처리
    - 반환: {'queries': [...], 'step': <증분>}
    """
    user_text = _last_user_text(state)
    # 모델 호출
    try:
        out = chat([{"role": "system", "content": SYS}, {"role": "user", "content": user_text}])
        raw = out if isinstance(out, str) else str(out)
    except Exception:
        raw = ""

    # 파싱 & 정리
    parsed = _coerce_queries_from_json_str(raw)
    if parsed is None:
        # 모델이 텍스트로 줄바꿈/불릿으로만 반환한 경우 분해
        lines = [ln.strip("-• \t") for ln in raw.splitlines() if ln.strip()]
        parsed = lines if lines else [user_text]

    queries = _sanitize_queries(parsed, fallback=user_text, k=2, max_len=256)

    updates: Dict[str, Any] = {
        "queries": queries,
        "step": int(state.get("step", 0)) + 1,
    }
    return updates
