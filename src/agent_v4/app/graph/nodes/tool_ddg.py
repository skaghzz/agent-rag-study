# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from typing import Any, Dict, List

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.documents import Document

# Local application imports
from agent_v4.app.graph.state import GraphState

# Public interface ----------------------------------------------------------------


def retrieve_ddg(state: GraphState) -> Dict[str, Any]:
    """Search the web via DuckDuckGo and store results in ``web_docs``.
    """

    # If the planner decided that we don't need a web search, skip.
    if not state.get("need_web", False):
        return {"web_docs": []}

    queries: List[str] = state.get("queries", []) or []
    if not queries:
        return {"web_docs": []}

    search_tool = DuckDuckGoSearchResults()

    collected: List[Document] = []
    # Limit to the 5 results per query to avoid latency.
    for query in queries:
        try:
            results = search_tool.invoke({"query": query, "max_results": 5})  # type: ignore[arg-type]
        except Exception:
            # Fail gracefully – return what we have so far.
            continue

        # DuckDuckGoSearchResults returns a list[dict]. Convert each entry to a Document.
        if isinstance(results, list):
            for hit in results:
                if isinstance(hit, dict):
                    content: str = hit.get("body") or hit.get("snippet") or ""
                    collected.append(Document(page_content=content, metadata={"source": hit.get("href", ""), "engine": "ddg"}))
                else:
                    collected.append(Document(page_content=str(hit), metadata={"engine": "ddg"}))
        else:
            # Unexpected format – store raw text.
            collected.append(Document(page_content=str(results), metadata={"engine": "ddg"}))

    return {"web_docs": collected}
