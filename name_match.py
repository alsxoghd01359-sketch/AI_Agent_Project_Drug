"""제품명 오타·축약형 대응 퍼지 매칭.

의미 검색(임베딩)이 아니라 문자열 유사도(rapidfuzz)를 쓴다.
"타이레놀" 같은 축약 입력이 "타이레놀정500밀리그람(아세트아미노펜)" 같은
정확한 등록명과 문자열로는 겹치는 부분이 많아 이 방식이 훨씬 잘 맞는다.
"""
import json
from pathlib import Path

from rapidfuzz import fuzz, process

DATA_DIR = Path("data")

_NAME_INDEX = None  # [(item_seq, item_name, source)], 지연 로딩


def _load_index():
    global _NAME_INDEX
    if _NAME_INDEX is not None:
        return _NAME_INDEX

    index = []
    with (DATA_DIR / "drug_prdt_prmsn_list.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            index.append((rec.get("ITEM_SEQ"), rec.get("ITEM_NAME"), "의약품"))

    with (DATA_DIR / "htfs_item.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            name = (rec.get("PRDUCT") or "").strip()
            if name:
                index.append((rec.get("STTEMNT_NO"), name, "건강기능식품"))

    _NAME_INDEX = index
    return index


def find_candidates(query: str, top_k: int = 5, min_score: float = 60.0):
    """query와 이름이 비슷한 제품 후보를 [(item_seq, item_name, source, score), ...]로 반환."""
    index = _load_index()
    names = [name for _, name, _ in index]

    results = process.extract(
        query, names, scorer=fuzz.WRatio, limit=top_k * 3,
    )

    seen = set()
    picked = []
    for name, score, idx in results:
        if score < min_score:
            continue
        item_seq, _, source = index[idx]
        key = (item_seq, name)
        if key in seen:
            continue
        seen.add(key)
        picked.append((item_seq, name, source, score))
        if len(picked) >= top_k:
            break

    return picked
