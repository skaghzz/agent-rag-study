import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage

from rag_agentic.graph import build_graph

# Build RAG graph once at startup
_GRAPH = build_graph()


@cl.on_chat_start
async def _on_chat_start() -> None:  # noqa: D401
    """Initialize a new chat session by storing an empty history."""

    cl.user_session.set("history", [])


@cl.on_message
async def _on_message(message: cl.Message) -> None:  # noqa: D401
    """Handle a user message, route through the RAG graph and stream the answer."""
    print('start _on_message')
    # history: list = cl.user_session.get("history") or []
    # history.append(HumanMessage(content=message.content))
    # print(history)
    
    
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")
    
    for msg, metadata in _GRAPH.stream({"messages": [HumanMessage(content=message.content)]}, stream_mode="messages", config=RunnableConfig(callbacks=[cb], **config)):
        if (
            msg.content
            and not isinstance(msg, HumanMessage)
            and metadata["langgraph_node"] in ("generate_answer", "generate_query_or_respond")
        ):
            await final_answer.stream_token(msg.content)
    # ai_msg = await _GRAPH.ainvoke(history)
    # history.append(ai_msg)

    # mypy stubs for Chainlit are incomplete, hence ignore type
    # await cl.Message(content=ai_msg.content).send()  # type: ignore[arg-type]
    print('end _on_message')
    await final_answer.send()
