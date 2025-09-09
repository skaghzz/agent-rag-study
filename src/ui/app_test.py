import chainlit as cl


@cl.on_chat_start
async def _on_chat_start():
    """Initialize a new chat session by storing an empty history."""
    cl.user_session.set("history", [])

@cl.on_message
async def main(message: cl.Message):
    history: list = cl.user_session.get("history") or []
    history.append(message.content)
    cl.user_session.set("history", history)
    print(history)
    # Send a response back to the user
    await cl.Message(
        # content=f"history : {history}\nReceived: {message.content}",
        content=f"Received: {message.content}",
    ).send()
