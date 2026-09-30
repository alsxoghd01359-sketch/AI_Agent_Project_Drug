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


def check_multiple_drugs(names: list[str]):
    """2개 이상의 제품명을 받아 전체 조합에 대해 확인한다.

    - 성분 중복: 어떤 성분이든 2개 이상의 약에 공통으로 들어있으면 과다복용 위험으로 표시
      (병용금기와는 다른 문제 — 같은 성분을 이중으로 섭취하는 것 자체가 위험)
    - DUR 병용금기: 모든 약 쌍(pair) 조합에 대해 확인
    제품명이 애매하면 candidates를 함께 반환해서 호출측에서 되물을 수 있게 한다.
    """
    _ensure_loaded()

    resolved = []
    for name in names:
        cands = search_product(name, top_k=3)
        if not cands:
            resolved.append({"query": name, "item_seq": None, "item_name": None,
                              "candidates": [], "ingredients": []})
            continue
        best = cands[0]
        resolved.append({
            "query": name,
            "item_seq": best["item_seq"],
            "item_name": best["item_name"],
            "candidates": cands,
            "ingredients": _ingredients_by_seq.get(best["item_seq"], []),
        })

    unresolved = [r["query"] for r in resolved if r["item_seq"] is None]

    # 1) 성분 중복 (과다복용 위험) — 약 2개짜리 조합만이 아니라 전체 목록 기준
    ingredient_to_drugs: dict[str, set] = {}
    for r in resolved:
        for ingr in r["ingredients"]:
            ingredient_to_drugs.setdefault(ingr, set()).add(r["item_name"])

    duplicate_ingredients = [
        {"ingredient": ingr, "drugs": sorted(drugs)}
        for ingr, drugs in ingredient_to_drugs.items()
        if len(drugs) >= 2
    ]

    # 2) DUR 병용금기 — 모든 쌍 조합
    seen_pairs = set()
    taboo_hits = []
    for i in range(len(resolved)):
        for j in range(i + 1, len(resolved)):
            a, b = resolved[i], resolved[j]
            for ia in a["ingredients"]:
                for ib in b["ingredients"]:
                    key = tuple(sorted([ia, ib]))
                    reason = _taboo_pairs.get(key)
                    if reason and key not in seen_pairs:
                        seen_pairs.add(key)
                        taboo_hits.append({
                            "drug_a": a["item_name"], "ingredient_a": ia,
                            "drug_b": b["item_name"], "ingredient_b": ib,
                            "reason": reason,
                        })

    return {
        "resolved_drugs": resolved,
        "unresolved_queries": unresolved,
        "duplicate_ingredients": duplicate_ingredients,
        "dur_taboo_matches": taboo_hits,
        "has_issue": bool(duplicate_ingredients or taboo_hits),
    }
