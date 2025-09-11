"""Azure AI Search 벡터 스토어에 문서를 적재하는 스크립트."""

from typing import List

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.documents import Document

from agent_v3.app.retrievers.aisearch_store import (
    load_vectorstore,
    make_docs_from_folder,
)


def main() -> None:
    """폴더의 문서를 분할 후 AzureSearch 벡터 스토어에 적재한다."""
    vector_store: AzureSearch = load_vectorstore()
    docs: List[Document] = make_docs_from_folder(chunk_size=1, chunk_overlap=0)

    if not docs:  # 엣지 케이스: 적재할 문서가 없는 경우
        print("Ingest 대상 문서가 없습니다.")
        return

    vector_store.add_documents(docs)
    print(f"Ingested {len(docs)} chunks.")

if __name__ == "__main__":
    main()
