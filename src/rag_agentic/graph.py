from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt.tool_node import ToolNode, tools_condition

from rag_agentic.ingest import retriever_tool
from rag_agentic.nodes.generate_answer import generate_answer
from rag_agentic.nodes.generate_question import generate_query_or_respond
from rag_agentic.nodes.grade import grade_documents
from rag_agentic.nodes.rewrite_question import rewrite_question


def build_graph():
    """Return a compiled LangGraph runnable representing the agentic RAG."""
    workflow = StateGraph(MessagesState)
    
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)
    
    workflow.add_edge(START, "generate_query_or_respond")
    
    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        # Assess LLM decision (call `retriever_tool` tool or respond to the user)
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        grade_documents,
    )
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    # Compile
    graph = workflow.compile()
    
    return graph
