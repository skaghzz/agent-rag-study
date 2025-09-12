from _operator import add
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.documents.base import Document
from langgraph.graph.message import MessagesState


class GraphState(MessagesState, total=False):
    queries: Annotated[List[str], add]
    kb_docs: Annotated[List[Document], add]
    web_docs: Annotated[List[Document], add]
    evidence: Annotated[List[Dict[str, Any]], add]
    answer: Optional[str]
    step: int
    need_web: bool
    intent: str