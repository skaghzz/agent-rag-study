import chainlit as cl

from agent_v1.app.models import chat
from agent_v1.app.retriever import keyword_retrieve

SYS = (
    "You are a concise assistant. "
    "Answer strictly based on the provided context. "
    "If the context is insufficient, respond with 'I don't know.' "
    "Cite as [KB::file::number]."
    "speak in Korean."
)

@cl.on_chat_start
async def start():
    await cl.Message(content="v1 RAG: 질문을 입력하세요.").send()

@cl.on_message
async def on_msg(m: cl.Message):
    q = m.content
    docs = keyword_retrieve(q, top_k=5)
    ctx = "\n".join([f"- {d['source']}: {d['content']}" for d in docs])
    out = chat([
        {"role":"system","content":SYS},
        {"role":"user","content": f"{q}\n\nContext:\n{ctx}"}
    ], temperature=0.2)
    await cl.Message(content=out).send()
