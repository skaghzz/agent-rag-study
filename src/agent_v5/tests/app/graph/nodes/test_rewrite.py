"""`query_rewrite` 노드에 대한 단위 테스트."""
from __future__ import annotations

from agent_v5.app.graph.nodes import rewrite as rw_mod


class _DummyChat:
    """`chat` 함수 스텁 – 항상 유효한 JSON 문자열을 반환."""

    def __call__(self, *_args, **_kwargs):  # noqa: D401
        return '["python", "파이썬"]'


def _make_state(msg: str) -> dict:
    return {"messages": [{"role": "user", "content": msg}], "step": 0}


def test_query_rewrite_basic(monkeypatch):
    """모델이 정상 JSON을 반환할 때 2개의 쿼리가 생성되어야 한다."""

    monkeypatch.setattr(rw_mod, "chat", _DummyChat(), raising=False)

    state = _make_state("파이썬 리스트 컴프리헨션 설명해줘")
    result = rw_mod.query_rewrite(state)  # type: ignore

    queries = result.get("queries")
    assert isinstance(queries, list)
    assert len(queries) == 2
    assert result["step"] == 1


def test_query_rewrite_invalid_json(monkeypatch):
    """모델이 잘못된 JSON을 반환하더라도 쿼리 2개가 보장되어야 한다."""

    class _BadChat:
        def __call__(self, *_a, **_kw):
            # 불완전한 JSON
            return "queries: 파이썬, python"

    monkeypatch.setattr(rw_mod, "chat", _BadChat(), raising=False)

    state = _make_state("python sort list")
    result = rw_mod.query_rewrite(state)  # type: ignore

    queries = result["queries"]
    assert len(queries) == 2
