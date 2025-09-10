import glob
import os
import re
from typing import Any, Dict

from langchain_text_splitters.character import CharacterTextSplitter

from agent_v2.app.graph.state import GraphState

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "ground_docs",
)


def retrieve_kb(state: GraphState) -> Dict[str, Any]:
    queries = state.get("queries", []) or []
    hits = []
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0, is_separator_regex=False)
    for p in glob.glob(os.path.join(DATA_DIR, "*.*")):
        print(f"retrieve processing ...[{p}]")
        try:
            txt = open(p, "r", encoding="utf-8").read()
        except Exception:
            continue
        chunks = splitter.split_text(txt)
        chunks = [c.strip() for c in chunks if c.strip() and not c.strip().startswith("#")]
        for i, chunk in enumerate(chunks):
            score = sum(len(re.findall(re.escape(q), chunk, flags=re.IGNORECASE)) for q in queries)
            if score > 0:
                hits.append({"source": f"KB:{os.path.basename(p)}", "content": txt, "score": score})

    hits.sort(key=lambda d: d["score"], reverse=True)
    return {"kb_docs": hits[:5]}
