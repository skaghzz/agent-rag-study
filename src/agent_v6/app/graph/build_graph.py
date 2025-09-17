from langgraph.graph import END, StateGraph

from agent_v6.app.graph.nodes.faithfulness import faithfulness_check
from agent_v6.app.graph.nodes.generate import generate
from agent_v6.app.graph.nodes.grader import evidence_grader
from agent_v6.app.graph.nodes.retrieve_kb import retrieve_kb
from agent_v6.app.graph.nodes.rewrite import query_rewrite
from agent_v6.app.graph.nodes.router import planner_router
from agent_v6.app.graph.nodes.tool_ddg import retrieve_ddg
from agent_v6.app.graph.state import GraphState


def build_graph():
    def _route_after_faithfulness(state: GraphState) -> str:
        """Return 'retry' to loop, or 'end' to finish.

        - If faithful: end
        - If unfaithful and step < max_steps: retry
        - If unfaithful and no max or exceeded: end (safety)
        """
        faith = (state.get("faithfulness") or {}).get("faithful", True)
        if faith:
            return "end"
        step = int(state.get("step", 0) or 0)
        max_steps = state.get("max_steps")
        if max_steps is not None and step >= int(max_steps):
            return "end"
        # Otherwise retry
        return "retry"

    g = StateGraph(GraphState)
    g.add_node("router", planner_router)
    g.add_node("rewrite", query_rewrite)
    g.add_node("kb", retrieve_kb)
    g.add_node("ddg", retrieve_ddg)
    g.add_node("grade", evidence_grader)
    g.add_node("generate", generate)
    g.add_node("faithfulness", faithfulness_check)

    g.set_entry_point("router")
    g.add_edge("router", "rewrite")
    g.add_edge("rewrite", "kb")
    g.add_edge("rewrite", "ddg")
    g.add_edge("kb", "grade")
    g.add_edge("ddg", "grade")
    g.add_edge("grade", "generate")
    g.add_edge("generate", "faithfulness")
    # Faithfulness 결과에 따라 종료 또는 재질의 루프
    g.add_conditional_edges(
        "faithfulness",
        _route_after_faithfulness,
        {"retry": "rewrite", "end": END},
    )
    return g.compile()
