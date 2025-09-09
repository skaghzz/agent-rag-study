# Agentic RAG Study

> Prototype implementation following ADR *Agentic RAG Architecture (LangGraph × Chainlit × Azure OpenAI GPT-5)*.

## Getting Started

```bash
# 1. Create a virtual env & install deps
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. Run a single-shot demo
python -m src.main "Explain LangGraph in one paragraph"
```

## Architecture

See `adr.md` for the full decision record. The current codebase only
implements a **stubbed end-to-end flow** so that the graph executes without
external services. Replace stub modules under `src/` with real integrations as
needed.

```text
src/
├─ graphs/          # LangGraph state machine
├─ agents/          # Planner, Synthesizer, Critic
├─ retrievers/      # Web & Vector search
├─ graders/         # Evidence grading
├─ hitl/            # Human-in-the-loop gate
├─ memory/          # Checkpointer & summarizer (TODO)
├─ services/        # LLM and utilities (TODO)
└─ ui/              # Chainlit app (TODO)
```

## Roadmap

1. Replace web/vector retrievers with Azure Bing & AI Search calls.
2. Implement proper evidence grading and confidence scoring.
3. Wire up Chainlit front-end with HITL actions.
4. Add Redis/SQLite checkpointer and conversation memory.
5. Containerize with Docker & add CI.
