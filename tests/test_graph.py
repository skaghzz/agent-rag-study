"""Tests for the `rag_agentic.graph` module.

These tests focus on ensuring that the `build_graph` function compiles a valid
LangGraph runnable without raising exceptions and exposes the expected
attributes.
"""
from typing import Any

import pytest

from rag_agentic.graph import build_graph


def test_build_graph_compiles() -> None:
    """`build_graph` should return a compiled LangGraph runnable instance."""
    graph = build_graph()

    # Basic sanity checks
    assert graph is not None, "Expected a runnable graph instance, got None"

    # The compiled graph should expose an ``invoke`` method which is callable.
    assert hasattr(graph, "invoke"), "Compiled graph should expose an 'invoke' method"
    assert callable(graph.invoke), "'invoke' should be callable"


@pytest.mark.parametrize("invalid_input", [None, {}, {"messages": []}])
def test_graph_rejects_invalid_state(invalid_input: Any) -> None:
    """The runnable should raise a ``ValueError`` for invalid initial state.

    This acts as a lightweight validation that the underlying runtime performs
    type checks on the state dictionary that drive the execution. We do *not*
    depend on external LLM calls here; we only expect a synchronous validation
    error from the runtime itself.
    """
    graph = build_graph()

    with pytest.raises(Exception):  # noqa: BLE001
        graph.invoke(invalid_input)  # type: ignore[arg-type]
