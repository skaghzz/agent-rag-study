"""CLI entry to run Agentic RAG v6 graph.

Usage:
  uv run python -m agent_v6.cli.run_agentic_graph "질문 내용"

Options:
  --web / --no-web        웹 검색 사용 여부(기본: --web)
  --max-steps N           Faithfulness 루프 최대 반복 수(기본: 3)

Env:
  USE_LLM_ROUTER=true|false
  USE_LLM_GRADER=true|false
"""
import argparse
import sys

from langchain_core.messages import HumanMessage

from agent_v6.app.graph.build_graph import build_graph
from agent_v6.app.graph.state import GraphState


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Agentic RAG v6 graph")
    p.add_argument("question", type=str, help="사용자 질문")
    p.add_argument("--web", dest="use_web", action="store_true", help="웹 검색 사용")
    p.add_argument("--no-web", dest="use_web", action="store_false", help="웹 검색 비활성화")
    p.set_defaults(use_web=True)
    p.add_argument("--max-steps", type=int, default=3, help="Faithfulness 루프 최대 반복 수")
    p.add_argument("--dump", action="store_true", help="최종 state를 JSON으로 출력")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    graph = build_graph()

    # 초기 상태 구성 (일반 dict로 작성해서 invoke에 전달)
    state = GraphState(  # type: ignore[assignment]
        messages=[HumanMessage(content=args.question)],
        need_web=bool(args.use_web),
        step=0,
        max_steps=int(args.max_steps) if args.max_steps is not None else None,
    )

    # 그래프 실행
    result = graph.invoke(state)

    # 출력
    answer = result.get("answer")
    faith = result.get("faithfulness") or {}
    faithful = faith.get("faithful")
    issues = faith.get("issues") or []

    print("=== Answer ===")
    print(answer or "(no answer)")

    if faithful is not None:
        print("\n=== Faithfulness ===")
        print(f"faithful: {faithful}")
        if issues:
            print("issues:")
            for i in issues:
                print(f" - {i}")

    if args.dump:
        print("\n=== RAW STATE ===")
        try:
            # ensure JSON serializable
            safe = {k: v for k, v in result.items() if k != "messages"}
            import json as _json

            print(_json.dumps(safe, ensure_ascii=False, indent=2))
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])

