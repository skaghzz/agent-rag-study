from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState

from rag_agentic.ingest import retriever_tool


def generate_query_or_respond(state: MessagesState):
    response_model = init_chat_model("azure_openai:gpt-5", temperature=0)
    
    response = (
        response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    )
    return {"messages": [response]}