"""제품명 검색(상세정보 조회)과 두 약 병용 확인 기능.

- search_product: 이름으로 후보를 찾고, 선택된 후보의 상세정보(효능효과/용법용량/
  주의사항/상호작용/이상반응)를 반환한다.
- check_interaction: 두 제품명을 받아 성분을 특정하고, DUR 병용금기 등록 여부와
  동일성분 중복 여부를 확인해서 사실 그대로 반환한다. "안전하다"는 판단은 하지 않는다.
"""
import json
from pathlib import Path

from name_match import find_candidates
from text_extract import doc_xml_to_text
from symptom_search import _normalize_ingredient

DATA_DIR = Path("data")

_detail_by_seq = None
_easy_by_seq = None
_list_by_seq = None
_ingredients_by_seq = None  # item_seq -> [MTRAL_NM, ...]
_taboo_pairs = None  # {(성분A, 성분B) 정렬됨: PROHBT_CONTENT}


def _read_jsonl(name: str):
    with (DATA_DIR / f"{name}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def _ensure_loaded():
    global _detail_by_seq, _easy_by_seq, _list_by_seq, _ingredients_by_seq, _taboo_pairs

    if _list_by_seq is None:
        _list_by_seq = {r["ITEM_SEQ"]: r for r in _read_jsonl("drug_prdt_prmsn_list")}

    if _detail_by_seq is None:
        _detail_by_seq = {r["ITEM_SEQ"]: r for r in _read_jsonl("drug_prdt_prmsn_detail")}

    if _easy_by_seq is None:
        _easy_by_seq = {r["itemSeq"]: r for r in _read_jsonl("easy_drug")}

    if _ingredients_by_seq is None:
        _ingredients_by_seq = {}
        for r in _read_jsonl("drug_prdt_mcpn_detail"):
            name = _normalize_ingredient(r["MTRAL_NM"] or "")
            if not name:
                continue
            names = _ingredients_by_seq.setdefault(r["ITEM_SEQ"], [])
            if name not in names:
                names.append(name)

    if _taboo_pairs is None:
        _taboo_pairs = {}
        for r in _read_jsonl("dur_usjnt_taboo"):
            a, b = r.get("INGR_KOR_NAME"), r.get("MIXTURE_INGR_KOR_NAME")
            if a and b:
                _taboo_pairs[tuple(sorted([a, b]))] = r.get("PROHBT_CONTENT")


def search_product(query: str, top_k: int = 5):
    """제품명으로 후보 검색. [{item_seq, item_name, source, score}, ...] 반환."""
    return [
        {"item_seq": seq, "item_name": name, "source": source, "score": round(score, 1)}
        for seq, name, source, score in find_candidates(query, top_k=top_k)
    ]


def get_product_detail(item_seq: str):
    """의약품 item_seq로 상세정보(효능효과/용법용량/주의사항/상호작용/이상반응) 조회."""
    _ensure_loaded()

    base = _list_by_seq.get(item_seq)
    if not base:
        return None

    detail = _detail_by_seq.get(item_seq, {})
    easy = _easy_by_seq.get(item_seq, {})

    efcy = easy.get("efcyQesitm") or doc_xml_to_text(detail.get("EE_DOC_DATA"))
    usemethod = easy.get("useMethodQesitm") or doc_xml_to_text(detail.get("UD_DOC_DATA"))
    caution = " ".join(filter(None, [easy.get("atpnWarnQesitm"), easy.get("atpnQesitm")])) \
        or doc_xml_to_text(detail.get("NB_DOC_DATA"))

    return {
        "item_seq": item_seq,
        "item_name": base.get("ITEM_NAME"),
        "entp_name": base.get("ENTP_NAME"),
        "spclty_pblc": base.get("SPCLTY_PBLC"),
        "cancel_name": base.get("CANCEL_NAME"),
        "cancel_date": base.get("CANCEL_DATE"),
        "ingredients": _ingredients_by_seq.get(item_seq, []),
        "효능효과": efcy,
        "용법용량": usemethod,
        "사용상주의사항": caution,
        "상호작용": easy.get("intrcQesitm"),
        "이상반응": easy.get("seQesitm"),
    }


def check_interaction(name_a: str, name_b: str):
    """두 제품명으로 병용 확인. 제품명이 애매하면 candidates를 함께 반환한다."""
    _ensure_loaded()

    cands_a = search_product(name_a, top_k=3)
    cands_b = search_product(name_b, top_k=3)

    if not cands_a or not cands_b:
        return {"error": "제품을 찾지 못했습니다.", "candidates_a": cands_a, "candidates_b": cands_b}

    # 가장 점수 높은 후보를 채택하되, 후보 전체도 같이 반환(호출측에서 애매하면 되물을 수 있게)
    best_a, best_b = cands_a[0], cands_b[0]

    ingr_a = _ingredients_by_seq.get(best_a["item_seq"], [])
    ingr_b = _ingredients_by_seq.get(best_b["item_seq"], [])

    # 1) 동일 성분 중복 (과다복용 위험)
    duplicate = sorted(set(ingr_a) & set(ingr_b))

    # 2) DUR 병용금기
    seen_pairs = set()
    taboo_hits = []
    for ia in ingr_a:
        for ib in ingr_b:
            key = tuple(sorted([ia, ib]))
            reason = _taboo_pairs.get(key)
            if reason and key not in seen_pairs:
                seen_pairs.add(key)
                taboo_hits.append({"ingredient_a": ia, "ingredient_b": ib, "reason": reason})

    return {
        "matched_a": {"item_seq": best_a["item_seq"], "item_name": best_a["item_name"], "candidates": cands_a},
        "matched_b": {"item_seq": best_b["item_seq"], "item_name": best_b["item_name"], "candidates": cands_b},
        "duplicate_ingredients": duplicate,
        "dur_taboo_matches": taboo_hits,
        "has_issue": bool(duplicate or taboo_hits),
    }
