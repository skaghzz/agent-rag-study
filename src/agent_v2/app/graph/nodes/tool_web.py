from typing import Any, Dict, List

from ddgs import DDGS

from agent_v2.app.graph.state import GraphState


def retrieve_web(state: GraphState) -> Dict[str, Any]:
    if not state.get("need_web", True):
        return {"web_docs": []}
    
    queries: List[str] = state.get("queries") or []
    if not queries:
        return {"web_docs": []}
    
    result: List[Dict[str, Any]] = []
    
    try:
        with DDGS() as ddgs:
            for q in queries:
                try:
                    for r in ddgs.text(q, region="kr-kr", safesearch="moderate", max_results=5,):
                        result.append({"source": f"WEB:{r.get('href', '')}", "title":r.get("title"), "content": r.get("body", "")[:400], "query": q})
                except Exception:
                    continue
    except Exception:
        result = []
    return {"web_docs": result}
