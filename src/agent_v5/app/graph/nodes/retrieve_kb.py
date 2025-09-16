from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document  # Added import

from agent_v5.app.graph.state import GraphState
from agent_v5.app.retrievers.aisearch_store import search_similar


def retrieve_kb(state: GraphState) -> Dict[str, Any]:
    # Safely extract the list of user queries, if available.
    queries: List[str] = state.get("queries", []) or []
    if not queries:
        # No queries yet – simply return the state with an empty ``kb_docs``.
        return {**state, "kb_docs": []}

    # Perform similarity search against the KB for **all** queries.
    kb_docs: list[dict[str, Any]] = []
    for query in queries:
        kb_docs.extend(search_similar(query, k=5))

    # Optionally, remove duplicates while preserving order.
    seen = set()
    deduped_docs = []
    for doc in kb_docs:
        content = doc["content"]
        if content not in seen:
            seen.add(content)
            deduped_docs.append(doc)

    # Convert dictionaries to `Document` instances expected by downstream nodes.
    documents: List[Document] = [
        Document(
            page_content=d["content"],
            metadata={k: v for k, v in d.items() if k != "content"},
        )
        for d in deduped_docs
    ]

    # Return a **new** state dict; avoid mutating the original state instance.
    return {"kb_docs": documents}
