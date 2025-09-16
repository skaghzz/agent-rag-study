"""`agent_v5.app.utils.messages` 모듈 테스트."""
from __future__ import annotations

from types import SimpleNamespace

from agent_v5.app.utils import messages as msg_mod


def test_content_to_text_variants():
    """여러 입력 형태가 올바르게 문자열로 변환되는지 확인."""

    assert msg_mod.content_to_text("abc") == "abc"

    list_form = [
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": "World"},
    ]
    assert msg_mod.content_to_text(list_form) == "Hello World"

    # 기타 타입은 str() 처리
    assert msg_mod.content_to_text(123) == "123"


def test_last_user_text_multi_formats():
    """dict 및 LangChain 스타일 객체 모두 인식하는지 확인."""

    st1 = {"messages": [{"role": "user", "content": "hi"}]}
    assert (lambda s: msg_mod.last_user_text(s))(st1) == "hi"

    # LangChain HumanMessage 대체: type 속성이 human 인 객체
    human = SimpleNamespace(type="human", content="hello")
    st2 = {"messages": [human]}
    assert (lambda s: msg_mod.last_user_text(s))(st2) == "hello"

    # 메시지가 없으면 빈 문자열
    assert msg_mod.last_user_text({"messages": []}) == ""
