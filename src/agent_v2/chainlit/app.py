import chainlit as cl

from agent_v2.app.graph.build_graph import build_graph
from agent_v2.app.graph.state import GraphState

graph = build_graph()


@cl.on_chat_start
async def start():
    await cl.Message(content="v2 Agentic Core: 질문을 입력하세요.").send()


@cl.on_message
async def on_msg(m: cl.Message):
    state = GraphState(messages=[{"role": "user", "content": m.content}], need_web=True, step=0)
    result = graph.invoke(state)

    await cl.Message(content=result.get("answer", "(no answer)")).send()
