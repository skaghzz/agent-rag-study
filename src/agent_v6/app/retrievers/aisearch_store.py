import glob
import os
from typing import List

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.documents import Document
from langchain_text_splitters.character import CharacterTextSplitter

from agent_v6.app.config import azure_search_cfg
from agent_v6.app.models import get_embeddings

# ---------------------------------------------------------------------------- #
# Azure AI Search helpers
# ---------------------------------------------------------------------------- #


def load_vectorstore() -> AzureSearch:
    """Return an :class:`~langchain_community.vectorstores.azuresearch.AzureSearch` instance
    bound to the index configured in ``azure_search_cfg``.

    Vector-only search is preferred. If index-creation logic becomes necessary in
    the future it should be implemented explicitly (e.g. ``ensure_index_exists()``).
    """

    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=azure_search_cfg.endpoint,
        azure_search_key=azure_search_cfg.key,
        index_name=azure_search_cfg.index,
        embedding_function=get_embeddings().embed_query,
    )
    return vector_store


# ---------------------------------------------------------------------------- #
# Utility: ingest local markdown files into the vector store
# ---------------------------------------------------------------------------- #


def make_docs_from_folder(
    folder: str = "ground_docs",
    mask: str = "*.md",
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[Document]:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    data_dir = os.path.join(repo_root, folder)

    paths = glob.glob(os.path.join(data_dir, mask))
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        is_separator_regex=False,
    )

    docs: List[Document] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as fp:
                txt = fp.read()
        except Exception:
            continue

        # Split into single-line chunks and filter noise.
        chunks = [c.strip() for c in splitter.split_text(txt) if c.strip() and not c.strip().startswith("#")]
        docs.extend(Document(page_content=chunk, metadata={"source": os.path.basename(p)}) for chunk in chunks)

    return docs


# ---------------------------------------------------------------------------- #
# Public API: similarity search wrapper (vector first)
# ---------------------------------------------------------------------------- #


def search_similar(
    q: str,
    k: int = 5,
    *,
    score_threshold: float | None = 0.7,
) -> List[Document]:
    """Return top-*k* chunks from the KB most similar to *q*."""

    vector_store: AzureSearch = load_vectorstore()

    # res: list[Document] = vector_store.similarity_search(q, k=k, search_type="similarity")
    res: list[tuple[Document, float]] = vector_store.similarity_search_with_relevance_scores(q, k=k, score_threshold=score_threshold)

    # 반환 타입 일치: Document 객체만 추출
    return [doc for doc, _ in res]
