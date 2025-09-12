from langgraph.graph import END, StateGraph

from agent_v3.app.graph.nodes.generate import generate
from agent_v3.app.graph.nodes.grader import evidence_grader
from agent_v3.app.graph.nodes.retrieve_kb import retrieve_kb
from agent_v3.app.graph.nodes.rewrite import query_rewrite
from agent_v3.app.graph.nodes.router import planner_router
from agent_v3.app.graph.state import GraphState


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("router", planner_router)
    g.add_node("rewrite", query_rewrite)
    g.add_node("kb", retrieve_kb)
    g.add_node("grade", evidence_grader)
    g.add_node("generate", generate)
    
    g.set_entry_point("router")
    g.add_edge("router", "rewrite")
    g.add_edge("rewrite", "kb")
    g.add_edge("kb", "grade")
    g.add_edge("grade", "generate")
    g.add_edge("generate", END)
    
    return g.compile()
