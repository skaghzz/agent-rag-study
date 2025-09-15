import chainlit as cl
from chainlit.input_widget import Slider, Switch

from agent_v4.app.graph.build_graph import build_graph
from agent_v4.app.graph.state import GraphState

graph = build_graph()


@cl.on_chat_start
async def start():
    await cl.ChatSettings(
        [
            Switch(id="use_web", label="웹 검색 사용", initial=True),
            Slider(id="max_steps", label="최대 스텝(표시용)", initial=3, min=1, max=6, step=1),
        ]
    ).send()
    await cl.Message(content="v4: Bing/Google + Pydantic. 질문을 입력하세요.").send()


@cl.on_message
async def on_msg(m: cl.Message):
    state = GraphState(messages=[{"role": "user", "content": m.content}], need_web=True, step=0)
    result = graph.invoke(state)
    await cl.Message(content=result.get("answer", "(no answer)")).send()
