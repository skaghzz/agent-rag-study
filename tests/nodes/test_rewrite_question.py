"""Tests for `rewrite_question` node."""

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from rag_agentic.nodes import rewrite_question as rw


def _msg(role: str, content: str):
    return SimpleNamespace(role=role, content=content)


def _make_state(question: str) -> Dict[str, List[Any]]:
    return {"messages": [_msg("user", question)]}


class DummyModel:
    def invoke(self, *_):
        return _msg("assistant", "Improved question?")


@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    monkeypatch.setattr(rw, "response_model", DummyModel())


def test_rewrite_question() -> None:
    state = _make_state("Where Eiffel?")
    result = rw.rewrite_question(state)

    assert "messages" in result
    msg = result["messages"][0]
    assert msg["role"] == "user"
    assert msg["content"] == "Improved question?"
