from operator import add
from typing import List, TypedDict

from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated


# 1) 상태 정의
class GraphState(TypedDict, total=False):
    queries: Annotated[List[str], add]

# 2) 노드 정의 (두 번에 걸쳐 같은 키 업데이트)
def n1(state: GraphState):
    return {"queries": ["A", "B"]}

def n2(state: GraphState):
    return {"queries": ["C", "D"]}

# 3) 그래프 구성 (중요: GraphState로 컴파일)
g = StateGraph(GraphState)
g.add_node("n1", n1)
g.add_node("n2", n2)
g.add_edge(START, "n1")
g.add_edge("n1", "n2")
g.add_edge("n2", END)

app = g.compile()

out = app.invoke({})
print(out["queries"])  # 기대: ["A", "B", "C"]
