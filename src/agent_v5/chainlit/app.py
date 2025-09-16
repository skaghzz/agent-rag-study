import chainlit as cl
from chainlit.input_widget import Slider, Switch
from langchain_core.messages import HumanMessage

from agent_v5.app.graph.build_graph import build_graph
from agent_v5.app.graph.state import GraphState

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
    # 현재 세션의 ChatSettings 값을 가져온다.
    chat_settings: dict | None = cl.user_session.get("chat_settings")  # type: ignore[arg-type]
    use_web: bool = True
    max_steps: int | None = None

    if chat_settings:
        use_web = bool(chat_settings.get("use_web", True))
        max_steps = int(chat_settings.get("max_steps", 0) or 0) or None  # 0이면 None 처리

    # GraphState 생성 시 설정 반영
    state = GraphState(
        messages=[HumanMessage(content=m.content)],
        need_web=use_web,
        step=0,
        max_steps=max_steps,
    )
    result = graph.invoke(state)
    await cl.Message(content=result.get("answer", "(no answer)")).send()
