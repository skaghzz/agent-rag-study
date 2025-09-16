"""`retrieve_ddg` 노드 테스트."""
from __future__ import annotations

from typing import Any, Dict, List

from agent_v5.app.graph.nodes import tool_ddg as ddg_mod


class _FakeSearchTool:
    """DuckDuckGoSearchResults 대체 스텁."""

    def __init__(self, results: List[Dict[str, Any]]):
        self._results = results

    # pylint: disable=unused-argument
    def invoke(self, *_args, **_kwargs):  # noqa: D401
        return self._results


_FAKE_RESULTS = [
    {
        "body": "Python – Official Site",
        "href": "https://www.python.org",
        "title": "Python",
        "snippet": "Python is a programming language...",
        "link": "https://www.python.org",
    }
]


def test_retrieve_ddg_need_web_false():
    """need_web=False 이면 빈 리스트가 반환된다."""
    state = {"need_web": False, "queries": ["python"]}
    res = ddg_mod.retrieve_ddg(state)  # type: ignore[arg-type]
    assert res["web_docs"] == []


def test_retrieve_ddg_success(monkeypatch):
    """정상 호출 시 web_docs 가 생성된다."""
    monkeypatch.setattr(ddg_mod, "DuckDuckGoSearchResults", lambda *a, **k: _FakeSearchTool(_FAKE_RESULTS), raising=False)
    state = {"need_web": True, "queries": ["python"]}
    res = ddg_mod.retrieve_ddg(state)  # type: ignore[arg-type]
    docs = res["web_docs"]
    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata["source"] == "https://www.python.org"
