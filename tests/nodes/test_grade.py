"""Tests for `grade_documents` branching logic."""

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from rag_agentic.nodes import grade as gr


def _msg(role: str, content: str):
    return SimpleNamespace(role=role, content=content)


def _make_state(question: str, context: str) -> Dict[str, List[Any]]:
    return {"messages": [_msg("user", question), _msg("system", context)]}


class DummyGradeModel:
    """Fake model returning structured output."""

    def with_structured_output(self, _):
        return self

    def invoke(self, *_):
        return SimpleNamespace(binary_score=self._score)

    def set_score(self, score: str):  # helper
        self._score = score
        return self


@pytest.fixture(autouse=True)
def patch_grade_model(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = DummyGradeModel().set_score("yes")
    monkeypatch.setattr(gr, "grader_model", dummy)


def test_branch_generate_answer(monkeypatch):
    dummy = DummyGradeModel().set_score("yes")
    monkeypatch.setattr(gr, "grader_model", dummy)

    state = _make_state("q", "c")
    assert gr.grade_documents(state) == "generate_answer"


def test_branch_rewrite_question(monkeypatch):
    dummy = DummyGradeModel().set_score("no")
    monkeypatch.setattr(gr, "grader_model", dummy)

    state = _make_state("q", "c")
    assert gr.grade_documents(state) == "rewrite_question"
