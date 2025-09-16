"""`generate` 노드 테스트."""
from __future__ import annotations

from typing import Any, Dict, List

from agent_v5.app.graph.nodes import generate as gen_mod


class _DummyChat:
    def __call__(self, *_a, **_kw):  # noqa: D401
        return "답변입니다."


def _make_state(ev: List[Dict[str, Any]] | None = None):
    return {"messages": [{"role": "user", "content": "파이썬?"}], "evidence": ev or []}


def test_generate_success(monkeypatch):
    monkeypatch.setattr(gen_mod, "chat", _DummyChat(), raising=False)

    res = gen_mod.generate(_make_state())  # type: ignore
    assert "답변" in res["answer"]


def test_generate_fallback(monkeypatch):
    """chat 호출이 예외를 던질 때 폴백 메시지가 포함되어야 한다."""

    class _ErrChat:
        def __call__(self, *_a, **_kw):  # noqa: D401
            raise RuntimeError("fail")

    monkeypatch.setattr(gen_mod, "chat", _ErrChat(), raising=False)

    ev = [
        {"content": "python is great", "source": "WEB:https://example.com", "title": "python"}
    ]
    res = gen_mod.generate(_make_state(ev))  # type: ignore
    assert "오류" in res["answer"] or "근거" in res["answer"]
