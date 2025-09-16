"""`agent_v5.app.graph.nodes.router` 모듈의 `planner_router`에 대한 유닛 테스트.

이 테스트 스위트는 다음 사항을 검증합니다.

1. LLM 라우터가 비활성화된 경우 폴백 경로가 사용된다.
2. 마지막 사용자 메시지가 없을 때 폴백 경로가 사용된다.
3. LLM 호출이 성공했을 때 `need_web` 사용자의 명시적 선호도와 올바르게 병합된다.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

# Config classes ------------------------------------------------------------
from agent_v5.app.config import Flags as _ConfigFlags

# Module under test ---------------------------------------------------------
from agent_v5.app.graph.nodes import router as router_mod

# GraphState 타입 추가 임포트
from agent_v5.app.graph.state import GraphState

# ---------------------------------------------------------------------------
# Helper doubles & fixtures --------------------------------------------------
# ---------------------------------------------------------------------------


class _DummyLLM:
    """`planner_router`에서 사용하는 LangChain LLM 인터페이스를 최소한으로 흉내 내는 스텁."""

    def with_structured_output(self, _model):
        """실제 구현과 동일하게 자기 자신을 반환하여 체이닝을 가능하게 한다."""

        return self

    def invoke(self, _messages: list[Dict[str, Any]]):
        """라우터가 검증할 수 있도록 하드코딩된 응답을 반환한다."""

        return {"intent": "followup", "need_web": True}


@pytest.fixture(autouse=True)
def _restore_router_flags(monkeypatch):
    """각 테스트 후 원본 `flags` 객체를 복원한다."""

    original_flags = router_mod.flags
    yield monkeypatch.setattr(router_mod, "flags", original_flags, raising=False)


# ---------------------------------------------------------------------------
# Tests ---------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_fallback_when_llm_router_disabled(monkeypatch):
    """`use_llm_router`가 *False*인 경우 `planner_router`는 폴백 결과를 반환해야 한다."""

    # Arrange – patch *flags* with *use_llm_router* turned *off*.
    patched_flags = _ConfigFlags(use_llm_router=False, use_llm_grader=True)
    monkeypatch.setattr(router_mod, "flags", patched_flags, raising=False)

    # Provide a state that already contains intent / need_web values.
    state: GraphState = {
        "messages": [],  # minimal required field for GraphState
        "intent": "task",
        "need_web": True,
    }

    # Act
    result = router_mod.planner_router(state)

    # Assert
    assert result.intent == "task"
    assert result.need_web is True


def test_fallback_when_no_user_text(monkeypatch):
    """`last_user_text`가 빈 문자열 또는 `None`일 때 폴백을 사용해야 한다."""

    # Arrange – keep *use_llm_router* enabled but patch `_last_user_text` to return "".
    patched_flags = _ConfigFlags(use_llm_router=True, use_llm_grader=True)
    monkeypatch.setattr(router_mod, "flags", patched_flags, raising=False)
    monkeypatch.setattr(router_mod, "_last_user_text", lambda _state: "", raising=False)

    state: GraphState = {
        "messages": [],
        "intent": "followup",
        "need_web": False,
    }

    # Act
    result = router_mod.planner_router(state)

    # Assert – values should match the ones in *state* (fallback).
    assert result.intent == "followup"
    assert result.need_web is False


@pytest.mark.parametrize(
    "user_pref, expected_need_web",
    [  # fmt: off
        (None, True),  # No explicit preference – use the LLM's decision
        (False, False),  # User explicitly disabled web – override LLM
        (True, True),  # User explicitly enabled web – AND with LLM's *True*
    ],
)
def test_llm_success_merges_with_user_preference(monkeypatch, user_pref, expected_need_web):
    """LLM 결과와 사용자 `need_web` 선호 병합 로직을 검증한다."""

    # Arrange
    patched_flags = _ConfigFlags(use_llm_router=True, use_llm_grader=True)
    monkeypatch.setattr(router_mod, "flags", patched_flags, raising=False)
    monkeypatch.setattr(router_mod, "_last_user_text", lambda _state: "What is the capital of France?", raising=False)
    monkeypatch.setattr(router_mod, "get_llm", lambda: _DummyLLM(), raising=False)

    # Build state with / without explicit *need_web*.
    state: GraphState = {"messages": [], "intent": "new_topic"}
    if user_pref is not None:
        state["need_web"] = user_pref  # type: ignore[index]

    # Act
    result = router_mod.planner_router(state)

    # Assert – intent comes from the LLM, need_web follows merge rules.
    assert result.intent == "followup"
    assert result.need_web is expected_need_web
