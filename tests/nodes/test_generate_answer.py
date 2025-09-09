"""Tests for the `generate_answer` node."""

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from rag_agentic.nodes.generate_answer import generate_answer


def _make_state(question: str, context: str) -> Dict[str, List[Any]]:
    """Create a dummy `MessagesState`-like dict for testing.

    The real `MessagesState` contains message objects where the
    ``content`` field is accessed via attribute access (``message.content``).
    To mimic this, we wrap the payload in ``SimpleNamespace`` instead of
    plain ``dict``.
    """

    def _msg(role: str, content: str) -> SimpleNamespace:
        return SimpleNamespace(role=role, content=content)

    return {
        "messages": [
            _msg("user", question),
            _msg("system", context),
        ]
    }


class DummyModel(SimpleNamespace):
    """A dummy chat model to stand in for the real Azure OpenAI model."""

    def invoke(self, *_, **__) -> Dict[str, str]:
        # Always return the same answer for deterministic tests.
        return {"role": "assistant", "content": "This is a mocked answer."}


@pytest.fixture(autouse=True)
def patch_response_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the global `response_model` with a dummy implementation."""
    from rag_agentic.nodes import generate_answer as generate_answer_module  # type: ignore

    # Replace the `response_model` used in the generate_answer module.
    monkeypatch.setattr(generate_answer_module, "response_model", DummyModel())
    

def test_generate_answer_returns_expected_structure() -> None:
    """`generate_answer` should return the expected dict structure."""
    question_text = "What is the capital of France?"
    context_text = "Paris is the capital and most populous city of France."
    state = _make_state(question_text, context_text)

    result = generate_answer(state)

    assert "messages" in result, "Result should contain a 'messages' key."
    assert isinstance(result["messages"], list), "'messages' should be a list."

    # Validate the content of the assistant's answer.
    assistant_message = result["messages"][0]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["content"] == "This is a mocked answer."
