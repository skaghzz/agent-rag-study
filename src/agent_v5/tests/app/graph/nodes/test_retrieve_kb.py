"""`retrieve_kb` 노드 테스트."""
from __future__ import annotations

from agent_v5.app.graph.nodes import retrieve_kb as rk_mod


def test_retrieve_kb_basic(monkeypatch):
    """`queries`가 주어지면 KB 문서가 반환되어야 한다."""

    def _fake_search_similar(_query: str, k: int = 5):  # noqa: D401, ARG002
        return [
            {"content": "문서1", "source": "KB:1", "title": "타이틀1"},
            {"content": "문서2", "source": "KB:2", "title": "타이틀2"},
        ]

    monkeypatch.setattr(rk_mod, "search_similar", _fake_search_similar, raising=False)

    state = {"queries": ["python"]}
    result = rk_mod.retrieve_kb(state)  # type: ignore

    kb_docs = result["kb_docs"]
    assert len(kb_docs) == 2
    # 중복 제거 확인: 동일 content가 하나만 남도록
    state2 = {"queries": ["python", "python"]}
    res2 = rk_mod.retrieve_kb(state2)  # type: ignore
    assert len(res2["kb_docs"]) == 2


def test_retrieve_kb_no_queries():
    """쿼리가 없으면 빈 리스트가 반환된다."""
    state = {}
    result = rk_mod.retrieve_kb(state)  # type: ignore
    assert result["kb_docs"] == []
