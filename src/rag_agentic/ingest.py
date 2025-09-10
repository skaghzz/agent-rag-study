"""Utilities to preprocess documents and build a vector store."""
from __future__ import annotations

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureOpenAIEmbeddings

from rag_agentic.config import get_settings

print('start ingest')

urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = [WebBaseLoader(url).load() for url in urls]

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = InMemoryVectorStore.from_documents(
documents=doc_splits, embedding=AzureOpenAIEmbeddings(
    azure_endpoint=get_settings().azure_openai_endpoint,
    azure_deployment=get_settings().azure_openai_embeddings_deployment,
    api_key=get_settings().azure_openai_api_key,
))
retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts.",
)

print('end ingest')