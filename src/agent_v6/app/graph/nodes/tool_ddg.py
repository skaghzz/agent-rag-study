# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from typing import Any, Dict, List

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.documents import Document

# Local application imports
from agent_v6.app.graph.state import GraphState

# Public interface ----------------------------------------------------------------


def retrieve_ddg(state: GraphState) -> Dict[str, Any]:
    """Run DuckDuckGo search queries decided by the planner.

    Args:
        state (GraphState): Current execution state. Expected keys:
            • need_web (bool): Whether a web search is required.
            • queries (List[str]): List of search queries.

    Returns:
        Dict[str, Any]: Dictionary containing ``web_docs`` – a list of
        langchain ``Document`` objects built from the search results.
    """

    # If the planner decided that we don't need a web search, skip.
    if not state.get("need_web", False):
        return {"web_docs": []}

    queries: List[str] = state.get("queries", []) or []
    if not queries:
        return {"web_docs": []}

    search_tool = DuckDuckGoSearchResults(output_format="list")

    collected: List[Document] = []
    # Limit to 5 results per query to avoid latency.
    for query in queries:
        try:
            results: List[Dict[str, Any]] = search_tool.invoke(
                {"query": query, "max_results": 5}  # type: ignore[arg-type]
            )
            # Transform raw results into langchain Document objects.
            for item in results:
                page_content: str = (
                    item.get("body")
                    or item.get("snippet")
                    or item.get("title", "")
                )
                metadata: Dict[str, Any] = {
                    "source": item.get("href") or item.get("link"),
                    "title": item.get("title"),
                    "query": query,
                }
                collected.append(
                    Document(page_content=page_content, metadata=metadata)
                )
        except Exception:
            # Fail gracefully – continue with the next query.
            continue

    return {"web_docs": collected}
