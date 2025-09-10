from enum import StrEnum

from langgraph.graph import END, StateGraph

from agent_v2.app.graph.nodes.generate import generate
from agent_v2.app.graph.nodes.grade_merge import grade_merge
from agent_v2.app.graph.nodes.retrieve_kb import retrieve_kb
from agent_v2.app.graph.nodes.rewrite import query_rewrite
from agent_v2.app.graph.nodes.tool_web import retrieve_web
from agent_v2.app.graph.state import GraphState


class Nodes(StrEnum):
    REWRITE = "rewrite"
    KB = "kb"
    WEB = "web"
    GRADE = "grade"
    GENERATE = "generate"

def build_graph():
    g = StateGraph(GraphState)

    g.add_node(Nodes.REWRITE, query_rewrite)
    g.add_node(Nodes.KB, retrieve_kb)
    g.add_node(Nodes.WEB, retrieve_web)
    g.add_node(Nodes.GRADE, grade_merge)
    g.add_node(Nodes.GENERATE, generate)

    g.set_entry_point(Nodes.REWRITE)
    g.add_edge(Nodes.REWRITE, Nodes.KB)
    g.add_edge(Nodes.REWRITE, Nodes.WEB)
    g.add_edge(Nodes.KB, Nodes.GRADE)
    g.add_edge(Nodes.WEB, Nodes.GRADE)
    g.add_edge(Nodes.GRADE, Nodes.GENERATE)
    g.add_edge(Nodes.GENERATE, END)
    return g.compile()
