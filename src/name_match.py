"""제품명 오타·축약형 대응 퍼지 매칭.

의미 검색(임베딩)이 아니라 문자열 유사도(rapidfuzz)를 쓴다.
"타이레놀" 같은 축약 입력이 "타이레놀정500밀리그람(아세트아미노펜)" 같은
정확한 등록명과 문자열로는 겹치는 부분이 많아 이 방식이 훨씬 잘 맞는다.
"""
import json
import re
from pathlib import Path

from rapidfuzz import fuzz, process

DATA_DIR = Path("data")

_NAME_INDEX = None  # [(item_seq, item_name, source)], 지연 로딩
_NORMALIZED_NAMES = None  # _NAME_INDEX와 같은 순서의 단위 정규화된 이름


def normalize_units(text: str) -> str:
    """단위 표기를 하나로 통일한다. "mg" vs "밀리그람" vs "밀리그램" 표기가 섞여있으면
    실제로는 같은 제품인데도 퍼지 매칭 점수가 크게 갈리는 걸 확인해서 추가함.
    (예: "타이레돌정500mg"은 "타이레돌정500밀리그람"보다 훨씬 낮은 점수가 나왔음)
    """
    text = re.sub(r"(\d)\s*(mcg|μg|ug)\b", r"\1마이크로그램", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d)\s*mg\b", r"\1밀리그램", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d)\s*ml\b", r"\1밀리리터", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d)\s*g\b", r"\1그램", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d)\s*%", r"\1퍼센트", text)
    text = text.replace("그람", "그램")  # "밀리그람"/"마이크로그람"도 이 한 줄로 같이 통일됨
    return text


def _load_index():
    global _NAME_INDEX, _NORMALIZED_NAMES
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
    _NORMALIZED_NAMES = [normalize_units(name or "") for _, name, _ in index]
    return index


def find_candidates(query: str, top_k: int = 5, min_score: float = 60.0):
    """query와 이름이 비슷한 제품 후보를 [(item_seq, item_name, source, score), ...]로 반환."""
    index = _load_index()
    query = normalize_units(query)

    results = process.extract(
        query, _NORMALIZED_NAMES, scorer=fuzz.WRatio, limit=top_k * 3,
    )

    seen = set()
    picked = []
    for _normalized_name, score, idx in results:
        if score < min_score:
            continue
        item_seq, original_name, source = index[idx]
        key = (item_seq, original_name)
        if key in seen:
            continue
        seen.add(key)
        picked.append((item_seq, original_name, source, score))
        if len(picked) >= top_k:
            break

    return picked
