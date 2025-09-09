"""Tests for `generate_query_or_respond` node."""

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from rag_agentic.nodes import generate_question as gn


def _msg(role: str, content: str) -> SimpleNamespace:  # noqa: WPS430
    return SimpleNamespace(role=role, content=content)


def _make_state(question: str) -> Dict[str, List[Any]]:
    return {"messages": [_msg("user", question)]}


class DummyModel:
    """Mimics the LangChain chat model for testing."""

    def bind_tools(self, *_):  # noqa: D401 (simple returns self)
        return self

    def invoke(self, *_):  # noqa: D401
        return _msg("assistant", "mocked answer")


@pytest.fixture(autouse=True)
def patch_chat_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gn, "init_chat_model", lambda *_, **__: DummyModel())


def test_generate_query_or_respond_returns_structure() -> None:
    state = _make_state("Where is the Eiffel Tower?")
    result = gn.generate_query_or_respond(state)

    assert "messages" in result
    messages = result["messages"]
    assert isinstance(messages, list)
    assert messages[0].content == "mocked answer"
