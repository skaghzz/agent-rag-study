import json
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from agent_v6.app.config import flags
from agent_v6.app.graph.state import GraphState
from agent_v6.app.models import get_llm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
CITATION_PAT = re.compile(r"\[(KB|WEB):([^\]]+)\]")


def _extract_citations(text: str) -> List[Tuple[str, str]]:
    return [(m.group(1).upper(), m.group(2).strip()) for m in CITATION_PAT.finditer(text or "")]


def _evidence_sources(evidence: List[Dict[str, Any]]) -> List[str]:
    sources: List[str] = []
    for e in evidence or []:
        src = e.get("source") or (e.get("metadata") or {}).get("source") or ""
        if not isinstance(src, str):
            continue
        # Normalize to include KB:/WEB: prefix if present
        if src.startswith("KB:") or src.startswith("WEB:"):
            sources.append(src)
        else:
            # If missing prefix but looks like URL, call it WEB
            prefix = "WEB:" if (src.startswith("http://") or src.startswith("https://")) else "KB:"
            sources.append(f"{prefix}{src}")
    return sources


# ---------------------------------------------------------------------------
# LLM schema
# ---------------------------------------------------------------------------


class FaithfulnessResult(BaseModel):
    faithful: bool = Field(default=True, description="Does the answer strictly rely on evidence?")
    issues: List[str] = Field(default_factory=list, description="List of hallucination/unsupported claim descriptions")
    fixed_answer: Optional[str] = Field(default=None, description="If unfaithful, a corrected answer grounded ONLY on evidence")


SYS = (
    "You are a strict FAITHFULNESS judge. Check if the ASSISTANT ANSWER is fully grounded in the provided EVIDENCE. "
    "If any claim is not directly supported by EVIDENCE, mark faithful=false. "
    "If not faithful, rewrite a corrected answer using ONLY the EVIDENCE. Return STRICT JSON with keys: "
    "faithful (true/false), issues (array of strings), fixed_answer (string or null)."
)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def faithfulness_check(state: GraphState) -> Dict[str, Any]:
    answer: str = (state.get("answer") or "").strip()
    evidence: List[Dict[str, Any]] = state.get("evidence", []) or []

    # Heuristic signals -----------------------------------------------------
    citations = _extract_citations(answer)
    sources = _evidence_sources(evidence)

    cited_ok = 0
    for tag, ref in citations:
        # Compose normalized target with tag prefix
        # Evidence sources already include prefixes (KB:/WEB:)
        target1 = f"{tag}:{ref}"
        # Some models may include only file name for KB; accept substring match
        matched = any((s == target1) or (ref and ref in s) for s in sources)
        if matched:
            cited_ok += 1

    heuristics: Dict[str, Any] = {
        "has_evidence": bool(evidence),
        "citations": [f"{t}:{r}" for t, r in citations],
        "evidence_sources": sources[:20],
        "supported_citations": cited_ok,
    }

    # If no LLM grading desired/available, provide heuristic verdict
    if not flags.use_llm_grader:
        faithful = bool(evidence) and bool(citations) and cited_ok >= max(1, len(citations) // 2)
        return {
            "faithfulness": {
                "faithful": faithful,
                "issues": [] if faithful else ["Insufficient or mismatched citations against evidence"],
                "heuristics": heuristics,
            }
        }

    # LLM-based evaluation --------------------------------------------------
    try:
        llm = get_llm().with_structured_output(FaithfulnessResult)
        # Prepare compact evidence list
        ev_lines: List[str] = []
        for e in evidence[:12]:
            src = e.get("source") or (e.get("metadata") or {}).get("source") or "SRC"
            content = (e.get("content") or e.get("text") or "").strip()
            content = content[:600]
            ev_lines.append(f"- {src}: {content}")
        ev_block = "\n".join(ev_lines) if ev_lines else "(none)"

        payload = {
            "answer": answer,
            "evidence": ev_block,
        }
        result = FaithfulnessResult.model_validate(
            llm.invoke(
                [
                    {"role": "system", "content": SYS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ]
            )
        )
        faithful = result.faithful
        issues = list(result.issues or [])
        fixed = (result.fixed_answer or "").strip() or None

        updates: Dict[str, Any] = {
            "faithfulness": {
                "faithful": faithful,
                "issues": issues,
                "heuristics": heuristics,
            }
        }
        # If unfaithful and we have a corrected answer, update it
        if not faithful and fixed:
            updates["answer"] = fixed
        return updates
    except Exception:
        # Fallback to heuristic if LLM validation fails
        faithful = bool(evidence) and bool(citations) and cited_ok >= max(1, len(citations) // 2)
        return {
            "faithfulness": {
                "faithful": faithful,
                "issues": [] if faithful else ["LLM validation failed; heuristic used"],
                "heuristics": heuristics,
            }
        }
