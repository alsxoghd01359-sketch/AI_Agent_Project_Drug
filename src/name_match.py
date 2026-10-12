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
_CLEAN_NAMES = None  # 숫자·기호를 뺀 이름(포함 여부 비교용)


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


def _clean_for_contains(text: str) -> str:
    return re.sub(r"[^가-힣a-zA-Z]", "", text)


def _load_index():
    global _NAME_INDEX, _NORMALIZED_NAMES
    if _NAME_INDEX is not None:
        return _NAME_INDEX

    index = []
    with (DATA_DIR / "drug_prdt_prmsn_list.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            index.append((rec.get("ITEM_SEQ"), rec.get("ITEM_NAME"), "의약품"))

    _NAME_INDEX = index
    global _CLEAN_NAMES
    _NORMALIZED_NAMES = [normalize_units(name or "") for _, name, _ in index]
    _CLEAN_NAMES = [_clean_for_contains(name or "") for _, name, _ in index]
    return index


def find_candidates(query: str, top_k: int = 5, min_score: float = 60.0):
    """query와 이름이 비슷한 제품 후보를 [(item_seq, item_name, source, score), ...]로 반환."""
    index = _load_index()
    raw_query = query
    query = normalize_units(query)

    results = process.extract(
        query, _NORMALIZED_NAMES, scorer=fuzz.WRatio, limit=top_k * 3,
    )

    # 줄임말 입력("쏘메토" -> "쏘메토320밀리그램연질캡슐")은 긴 정식 이름과 문자열 유사도가
    # 낮아서 무관한 짧은 이름들에 밀린다. 입력으로 시작하는 정식 이름은 최소 점수를 올린다.
    #
    # 치명적 버그(실측): "초당아스피린장용정100mg"으로 검색하면, 실제로 존재하는 정확한
    # 제품 "초당아스피린장용정100mg(아스피린)"의 WRatio 점수가 84.2에 그치는데, 전혀 다른
    # 제조사의 "한미아스피린장용정100밀리그램" 등은 글자 수가 같다는 이유만으로 87.5를
    # 받아서 더 높았다. 이 보정값이 85.0이라 여전히 87.5보다 낮아서, 정답 제품이 top_k=5
    # 안에 못 들고(11위) 완전히 무관한 다른 제조사 제품 5개만 후보로 나왔다. 입력이 정식
    # 이름의 접두사/부분문자열로 그대로 들어있는 경우는 "우연히 비슷한 다른 이름"보다
    # 항상 우선해야 하므로, 어떤 우연의 일치보다도 높도록 99.0으로 올린다(완전 일치는
    # WRatio 자체가 이미 100에 가까우므로 100은 그 몫으로 남겨둔다).
    clean_query = _clean_for_contains(raw_query)
    if len(clean_query) >= 2:
        best = {idx: score for _n, score, idx in results}
        for idx, clean_name in enumerate(_CLEAN_NAMES):
            if clean_name.startswith(clean_query) or clean_query in clean_name:
                best[idx] = max(best.get(idx, 0.0), 99.0)
        results = sorted(((n, s, i) for i, s in best.items() for n in [_NORMALIZED_NAMES[i]]),
                         key=lambda r: -r[1])

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
