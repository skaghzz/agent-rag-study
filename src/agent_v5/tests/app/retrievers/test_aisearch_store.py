"""`agent_v5.app.retrievers.aisearch_store` 테스트."""
from __future__ import annotations

from types import SimpleNamespace

from agent_v5.app.retrievers import aisearch_store as ai_mod


class _FakeVectorStore:
    def __init__(self):
        self._docs = [
            SimpleNamespace(page_content="내용", metadata={"source": "doc.md"})
        ]

    def hybrid_search(self, query: str, k: int = 5):  # noqa: D401, ARG002
        return self._docs


def test_search_similar(monkeypatch):
    monkeypatch.setattr(ai_mod, "load_vectorstore", lambda: _FakeVectorStore(), raising=False)

    res = ai_mod.search_similar("python", k=1)
    assert res[0]["source"].startswith("KB:")
    assert res[0]["content"] == "내용"
