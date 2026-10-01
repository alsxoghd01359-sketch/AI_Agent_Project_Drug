"""증상 기반 일반의약품 성분 후보 검색.

임베딩 의미검색 대신, 식약처 공식 약효분류(PRDUCT_TYPE) 구조화 데이터를 사용한다.
- 증상 키워드 -> 약효분류 코드로 매핑 (작고 통제 가능한 사전)
- 해당 분류의 일반의약품(전문의약품 제외)만 대상
- 주성분(mcpn_detail의 첫 번째 성분) 기준으로 그룹핑
- 사용자가 복용 중인 약의 성분과 DUR 병용금기/단일주의에 걸리는 성분군은 제외
- 성분당 실제 등록 제품 예시 2~3개를 그대로 보여줌 (인기도 판단 없음, "이거 드세요" 없음)
"""
import json
import re
from pathlib import Path

DATA_DIR = Path("data")

_FORMULATION_SUFFIXES = ["제피세립", "건조엑스", "연조엑스", "미분화", "과립", "세립", "분말", "펠렛", "제피"]

# 생약(한방 원료) 이름. 전부 나열하기보다 "양약/한방" 구분용으로만 쓰는 작은 사전.
_HERBAL_RAW_NAMES = {
    "갈근", "감초", "강활", "계지", "금은화", "당귀", "대추", "마황", "방기", "백출",
    "복령", "숙지황", "시호", "오가피", "위령선", "인삼", "작약", "조구등", "진피",
    "창출", "황금", "회향", "생강", "맥문동", "길경", "형개", "박하", "행인", "반하",
}

_HERBAL_SUFFIX_HINTS = ("엑스", "탕", "산", "환", "음", "단미")


def is_herbal(ingredient: str) -> bool:
    if ingredient in _HERBAL_RAW_NAMES:
        return True
    return ingredient.endswith(_HERBAL_SUFFIX_HINTS)


def _normalize_ingredient(name: str) -> str:
    """제형/농도 표기 차이로 같은 성분이 여러 그룹으로 쪼개지는 걸 합친다.
    예: '아세트아미노펜 과립', '아세트아미노펜(미분화)' -> '아세트아미노펜'
    """
    n = re.sub(r"[\(\[][^)\]]*[\)\]]", "", name)  # 괄호 안 내용 제거
    for suf in _FORMULATION_SUFFIXES:
        n = n.replace(suf, "")
    return n.strip()

# 증상 키워드 -> 약효분류코드(PRDUCT_TYPE의 대괄호 안 코드). 필요에 따라 계속 추가.
SYMPTOM_TO_CLASS_CODE = {
    "두통": "01140", "열": "01140", "발열": "01140", "근육통": "01140", "치통": "01140",
    "기침": "02220", "가래": "02220",
    "설사": "02360",
    "변비": "02370",
    "소화불량": "02390", "속쓰림": "02320",
    "콧물": "02610", "비염": "02610",
}


def _read_jsonl(name: str):
    with (DATA_DIR / f"{name}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def _load_primary_ingredient_by_item_seq() -> dict:
    """ITEM_SEQ -> 주성분(가장 앞 순번) 한글명."""
    result = {}
    for rec in _read_jsonl("drug_prdt_mcpn_detail"):
        seq = rec.get("ITEM_SEQ")
        sn = rec.get("MTRAL_SN")
        name = rec.get("MTRAL_NM")
        if not seq or not name:
            continue
        if seq not in result or (sn == "1"):
            result[seq] = name
    return result


def _load_dur_conflict_pairs() -> set:
    """(성분A, 성분B) 정렬된 튜플 집합. 병용금기 데이터 기준."""
    pairs = set()
    for rec in _read_jsonl("dur_usjnt_taboo"):
        a = rec.get("INGR_KOR_NAME")
        b = rec.get("MIXTURE_INGR_KOR_NAME")
        if a and b:
            pairs.add(tuple(sorted([a, b])))
    return pairs


def search_by_symptom(symptom: str, current_med_ingredients: list[str] | None = None,
                       examples_per_ingredient: int = 3):
    """증상 키워드로 일반의약품 성분 후보를 전부 찾아서 반환.

    current_med_ingredients: 사용자가 복용 중인 약의 성분 한글명 리스트.
    반환: [{"ingredient": str, "examples": [item_name, ...], "excluded_reason": str|None}, ...]
    excluded_reason이 있는 항목은 병용금기 때문에 제외 대상임을 표시(호출측에서 필터링해도 됨).
    """
    class_code = SYMPTOM_TO_CLASS_CODE.get(symptom)
    if not class_code:
        return None  # 매핑된 분류 없음 -> 호출측에서 "지원하지 않는 증상"으로 안내

    primary_ingr = _load_primary_ingredient_by_item_seq()
    current_med_ingredients = set(current_med_ingredients or [])

    groups: dict[str, list[str]] = {}
    for rec in _read_jsonl("drug_prdt_prmsn_list"):
        prduct_type = rec.get("PRDUCT_TYPE") or ""
        if not prduct_type.startswith(f"[{class_code}]"):
            continue
        if rec.get("SPCLTY_PBLC") != "일반의약품":
            continue
        if rec.get("CANCEL_NAME") != "정상":
            continue

        seq = rec.get("ITEM_SEQ")
        ingr = primary_ingr.get(seq)
        if not ingr:
            continue
        ingr = _normalize_ingredient(ingr)
        if not ingr:
            continue

        groups.setdefault(ingr, []).append(rec.get("ITEM_NAME"))

    conflict_pairs = _load_dur_conflict_pairs() if current_med_ingredients else set()

    results = []
    for ingr, item_names in groups.items():
        excluded_reason = None
        for med_ingr in current_med_ingredients:
            if tuple(sorted([ingr, med_ingr])) in conflict_pairs:
                excluded_reason = f"현재 복용 중인 '{med_ingr}' 성분과 DUR 병용금기 등록됨"
                break

        results.append({
            "ingredient": ingr,
            "category": "한방" if is_herbal(ingr) else "양약",
            "examples": item_names[:examples_per_ingredient],
            "excluded_reason": excluded_reason,
        })

    results.sort(key=lambda r: (r["category"], r["ingredient"]))
    return results
