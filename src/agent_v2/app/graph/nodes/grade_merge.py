from typing import Any, Dict

from agent_v2.app.graph.state import GraphState


def grade_merge(state: GraphState) -> Dict[str, Any]:
    kb = state.get("kb_docs", []) or []
    web = state.get("web_docs", []) or []
    ev = kb + web
    return {
        "evidence": ev
    }
