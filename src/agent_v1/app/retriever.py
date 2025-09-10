import glob
import os
import re

from langchain_text_splitters import CharacterTextSplitter  # 각 행 단위 분할용

# 프로젝트 루트의 data/ 디렉터리를 가리키도록 수정
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "ground_docs",
)


def keyword_retrieve(query: str, top_k: int = 5):
    hits = []
    # 각 줄(리스트 항목)을 한 건으로 취급하도록 분할기 설정
    splitter = CharacterTextSplitter(
        separator="\n",  # 줄바꿈 기준
        chunk_size=1,  # 한 줄씩
        chunk_overlap=0,
        is_separator_regex=False,
    )
    for p in glob.glob(os.path.join(DATA_DIR, "*.*")):
        try:
            txt = open(p, "r", encoding="utf-8").read()
        except Exception:
            continue
        chunks = splitter.split_text(txt)
        # 공백·헤더 제거
        chunks = [c.strip() for c in chunks if c.strip() and not c.strip().startswith("#")]
        for i, chunk in enumerate(chunks):
            score = len(re.findall(re.escape(query), chunk, flags=re.IGNORECASE))
            if score > 0:
                hits.append((score, os.path.basename(p), i + 1, chunk))
    hits.sort(key=lambda x: x[0], reverse=True)
    return [{"source": f"KB:{h[1]}:{h[2]}", "content": h[3]} for h in hits[:top_k]]
