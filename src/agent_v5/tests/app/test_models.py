"""`agent_v5.app.models` 모듈 테스트."""
from __future__ import annotations

from types import SimpleNamespace

from agent_v5.app import models as mdl


class _FakeLLM:
    """`invoke` 호출 시 SimpleNamespace(content="pong") 반환."""

    def invoke(self, *_a, **_kw):  # noqa: D401
        return SimpleNamespace(content="pong")


def test_chat_returns_string(monkeypatch):
    """`chat` 헬퍼가 항상 문자열을 반환하는지 확인."""

    monkeypatch.setattr(mdl, "get_llm", lambda: _FakeLLM(), raising=False)

    msgs = [
        {"role": "user", "content": "ping"},
    ]
    out = mdl.chat(msgs)
    assert out == "pong"
