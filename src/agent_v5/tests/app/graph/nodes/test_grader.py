"""`evidence_grader` 노드 테스트."""
from __future__ import annotations

from typing import Any, Dict, List

from agent_v5.app.graph.nodes import grader as gr_mod

_DOC1 = {"content": "파이썬은 인터프리터 언어다", "source": "KB:1"}
_DOC2 = {"content": "자바는 컴파일 언어다", "source": "KB:2"}


def _make_state(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"messages": [{"role": "user", "content": "파이썬"}], "kb_docs": docs, "web_docs": []}


def test_grader_no_llm(monkeypatch):
    """LLM 필터 비활성화 시 모든 evidence 가 유지되어야 한다."""

    # flags.use_llm_grader = False
    patched_flags = gr_mod.flags.__class__(use_llm_router=True, use_llm_grader=False)  # type: ignore
    monkeypatch.setattr(gr_mod, "flags", patched_flags, raising=False)

    state = _make_state([_DOC1, _DOC2])
    res = gr_mod.evidence_grader(state)  # type: ignore
    ev = res["evidence"]
    assert len(ev) == 2


def test_grader_with_llm(monkeypatch):
    """LLM 필터가 특정 evidence 를 제거하는지 확인."""

    patched_flags = gr_mod.flags.__class__(use_llm_router=True, use_llm_grader=True)  # type: ignore
    monkeypatch.setattr(gr_mod, "flags", patched_flags, raising=False)

    def _fake_llm_filter(_q: str, evidences: List[Dict[str, Any]]):  # noqa: D401, ARG002
        # 첫 evidence 만 유지
        return evidences[:1]

    monkeypatch.setattr(gr_mod, "_llm_filter", _fake_llm_filter, raising=False)

    state = _make_state([_DOC1, _DOC2])
    res = gr_mod.evidence_grader(state)  # type: ignore
    ev = res["evidence"]
    assert len(ev) == 1
    assert ev[0]["content"] == _DOC1["content"]
