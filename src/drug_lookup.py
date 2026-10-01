"""제품명 검색(상세정보 조회)과 두 약 병용 확인 기능.

- search_product: 이름으로 후보를 찾고, 선택된 후보의 상세정보(효능효과/용법용량/
  주의사항/상호작용/이상반응)를 반환한다.
- check_interaction: 두 제품명을 받아 성분을 특정하고, DUR 병용금기 등록 여부와
  동일성분 중복 여부를 확인해서 사실 그대로 반환한다. "안전하다"는 판단은 하지 않는다.
"""
import json
import pickle
import re
from pathlib import Path

from name_match import find_candidates, normalize_units
from text_extract import doc_xml_to_text
from symptom_search import _normalize_ingredient

_BRACKET_RE = re.compile(r"[\(\[][^)\]]*[\)\]]")


def _strip_brackets(name: str) -> str:
    return _BRACKET_RE.sub("", name or "").strip()

DATA_DIR = Path("data")
_CACHE_DIR = DATA_DIR / "_cache"

_detail_offsets = None  # item_seq -> drug_prdt_prmsn_detail.jsonl 안의 바이트 오프셋
_easy_by_seq = None
_list_by_seq = None
_ingredients_by_seq = None  # item_seq -> [MTRAL_NM, ...]
_taboo_pairs = None  # {(성분A, 성분B) 정렬됨: PROHBT_CONTENT}
_taboo_ingredients = None  # {병용금기 쌍에 한 번이라도 등장하는 성분명}

_ITEM_SEQ_RE = re.compile(rb'"ITEM_SEQ"\s*:\s*"([^"]+)"')


def _read_jsonl(name: str):
    with (DATA_DIR / f"{name}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def _build_detail_offsets():
    """drug_prdt_prmsn_detail.jsonl은 항목당 EE/UD/NB_DOC_DATA(XML 원문)가 들어있어서
    평균 수십 KB, 전체 2GB가 넘는다. 실측으로 확인: 이걸 통째로 dict에 올리면 14초
    넘게 걸리는데, 실제로는 세션당 한두 건만 조회한다. 그래서 전체를 json.loads로
    파싱하지 않고, 각 줄 맨 앞의 ITEM_SEQ만 정규식으로 빠르게 뽑아서 파일 바이트
    오프셋만 인덱싱해두고, 조회할 때 그 줄만 seek해서 읽는다.
    """
    global _detail_offsets
    cache_path = _CACHE_DIR / "detail_offsets.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as f:
            _detail_offsets = pickle.load(f)
        return

    offsets = {}
    path = DATA_DIR / "drug_prdt_prmsn_detail.jsonl"
    with path.open("rb") as f:
        offset = f.tell()
        for raw in f:
            m = _ITEM_SEQ_RE.search(raw[:200])
            if m:
                offsets[m.group(1).decode("utf-8")] = offset
            offset = f.tell()

    _detail_offsets = offsets
    _CACHE_DIR.mkdir(exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(offsets, f)


def _get_detail(item_seq: str) -> dict:
    if _detail_offsets is None:
        _build_detail_offsets()
    offset = _detail_offsets.get(item_seq)
    if offset is None:
        return {}
    path = DATA_DIR / "drug_prdt_prmsn_detail.jsonl"
    with path.open("rb") as f:
        f.seek(offset)
        line = f.readline()
    return json.loads(line)


def _load_taboo_pairs():
    """dur_usjnt_taboo.jsonl은 79만 줄(1.3GB)이지만 실제 고유 병용금기 쌍은
    870개뿐이다(중복 성분쌍이 대부분). 매번 1.3GB를 다 읽는 대신, 뽑아낸 결과를
    캐시 파일에 저장해두고 다음 실행부터는 그걸 바로 불러온다.
    """
    global _taboo_pairs, _taboo_ingredients
    cache_path = _CACHE_DIR / "taboo_pairs.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as f:
            _taboo_pairs = pickle.load(f)
        _taboo_ingredients = {ingr for pair in _taboo_pairs for ingr in pair}
        return

    pairs = {}
    for r in _read_jsonl("dur_usjnt_taboo"):
        a, b = r.get("INGR_KOR_NAME"), r.get("MIXTURE_INGR_KOR_NAME")
        if a and b:
            pairs[tuple(sorted([a, b]))] = r.get("PROHBT_CONTENT")

    _taboo_pairs = pairs
    _taboo_ingredients = {ingr for pair in pairs for ingr in pair}
    _CACHE_DIR.mkdir(exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(pairs, f)


def _ensure_loaded():
    global _easy_by_seq, _list_by_seq, _ingredients_by_seq

    if _list_by_seq is None:
        _list_by_seq = {r["ITEM_SEQ"]: r for r in _read_jsonl("drug_prdt_prmsn_list")}

    if _detail_offsets is None:
        _build_detail_offsets()

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
        _load_taboo_pairs()


def search_product(query: str, top_k: int = 5):
    """제품명으로 후보 검색. [{item_seq, item_name, source, score}, ...] 반환."""
    return [
        {"item_seq": seq, "item_name": name, "source": source, "score": round(score, 1)}
        for seq, name, source, score in find_candidates(query, top_k=top_k)
    ]


def _group_candidates_by_ingredient(candidates, ingr_sets):
    """후보들을 정확히 같은 성분 조합이 아니라, 성분이 겹치면 같은 "계열"로 묶는다.

    실측으로 확인: "게보린"은 제품마다 부성분(카페인, 비타민 등)이 조금씩 달라서
    전체 성분 조합이 똑같은 제품이 거의 없다. 그래서 완전히 같은 성분 조합으로
    묶으면 제품 5개가 그룹 5개로 쪼개져서 사용자에게 의미 없이 세세하게 되묻게 된다.
    실제로 사용자가 구분하고 싶어하는 건 "주성분이 같은가"이므로, 성분 하나라도
    겹치면 연결(union-find)해서 묶는다 (아세트아미노펜 계열 vs 이부프로펜 계열처럼).
    """
    n = len(candidates)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    empty_indices = [i for i in range(n) if not ingr_sets[i]]
    for i in empty_indices[1:]:
        union(empty_indices[0], i)

    for i in range(n):
        for j in range(i + 1, n):
            if ingr_sets[i] & ingr_sets[j]:
                union(i, j)

    component_indices: dict[int, list[int]] = {}
    for i in range(n):
        component_indices.setdefault(find(i), []).append(i)

    groups = []
    for idxs in component_indices.values():
        sets_in_group = [ingr_sets[i] for i in idxs]
        if all(sets_in_group):
            common = set.intersection(*sets_in_group)
        else:
            common = set()
        if not common:
            counter: dict[str, int] = {}
            for s in sets_in_group:
                for ingr in s:
                    counter[ingr] = counter.get(ingr, 0) + 1
            common = {max(counter, key=counter.get)} if counter else set()
        representative = sorted(common)[0] if common else "성분 정보 없음"
        all_ingredients = sorted(set().union(*sets_in_group)) if sets_in_group else []
        rest = [x for x in all_ingredients if x != representative]
        groups.append({
            "ingredients": [representative] + rest,
            "products": [candidates[i]["item_name"] for i in idxs],
        })
    return groups


def search_product_with_confidence(query: str, top_k: int = 5, overlap_threshold: float = 0.6,
                                    min_score: float = 75.0, min_candidates: int = 2):
    """제품명 검색 + 신뢰도 판정.

    오타가 2글자 이상이면 문자열 유사도 점수만으로는 완전히 다른 약과 구분이
    안 되는 걸 실측으로 확인함(예: "타이래논" -> 무관한 "타이록신캡슐"이 상위권).
    그래서 점수 대신 "상위 후보들이 같은 성분을 공유하는가"로 신뢰도를 판단한다.
    같은 제품군(예: 타이레놀 계열)은 브랜드명이 갈려도 성분이 겹치지만,
    완전히 다른 약이 우연히 비슷한 점수로 섞이면 성분이 겹치지 않기 때문이다.

    다만 성분 겹침만으로는 두 가지 거짓양성이 생기는 걸 실측으로 확인함.
    1) 후보가 1개뿐이면 자기 자신과만 비교해 겹침이 무조건 100%가 됨(예: "토아래눌")
    2) 오타 난 문자열이 우연히 전혀 다른 약 그룹과 더 많이 겹쳐서, 그 그룹 안에서는
       서로 성분이 일관되게 나오는 경우(예: "태이레논" -> 게피티니브 계열 항암제)
    두 경우 모두 최고 점수가 75 미만으로 낮았다는 공통점이 있어, 점수 하한과 최소
    후보 수 조건을 추가로 걸어서 걸러낸다.

    성분 겹침이 높아도 또 다른 문제가 있다는 걸 실측으로 확인함: "타이레놀"처럼
    브랜드명만 입력하면 성인용 정제/어린이용 산제·현탁액/8시간 서방정/감기 복합제
    (타이레놀콜드-에스정 등 아세트아미노펜 외 다른 성분이 추가된 제품)까지 전부
    최고 점수로 동점 처리된다. 이런 제품들은 같은 "계열"이라도 실제로는 서로
    다른 구체적 제품이고 용법·용량·주의사항이 다르므로, 점수가 동점인 후보가
    여럿이면(=쿼리만으로는 어떤 구체적 제품인지 구분이 안 됨) 성분이 같아도
    confident로 처리하지 않는다. "타이레놀정500mg"처럼 구체적으로 물으면
    최고 점수가 유일해져서 이 조건에 걸리지 않는다.

    반환: {"candidates": [...], "confident": bool, "overlap_ratio": float}
    confident=False면 특정 제품으로 단정하지 말고 재확인을 요청해야 한다.
    """
    _ensure_loaded()
    candidates = search_product(query, top_k=top_k)

    if not candidates:
        return {"candidates": [], "confident": False, "overlap_ratio": 0.0, "candidate_groups": []}

    ingr_sets = [set(_ingredients_by_seq.get(c["item_seq"], [])) for c in candidates]
    top_ingr = ingr_sets[0]
    overlap_count = sum(1 for s in ingr_sets if s & top_ingr)
    overlap_ratio = overlap_count / len(candidates)

    top_score = candidates[0]["score"]
    tied_names = {c["item_name"] for c in candidates if c["score"] == top_score}
    score_ambiguous = len(tied_names) >= 2

    # 괄호 성분표기·단위표기 차이를 무시하면 쿼리와 이름이 완전히 같은 후보가 "단
    # 하나"면, 다른 후보들의 성분 겹침과 무관하게 확정으로 본다. 실측으로 확인:
    # "판피린티정"(정확히 일치, 점수 100)이나 "탁센연질캡슐"(정확히 일치, 점수 90)
    # 처럼 명확한 쿼리에도, top_k로 같이 딸려온 전혀 무관한 저점 후보들(예: "엠티정",
    # "한솔나프록센연질캡슐")이 성분이 달라서 overlap_ratio를 끌어내려 거짓음성이
    # 발생했음. 정확히 일치하는 후보가 2개 이상이면(동명 제품) 이 지름길을 쓰지
    # 않고 아래 일반 로직으로 넘어간다.
    normalized_query_base = _strip_brackets(normalize_units(query))
    exact_matches = [
        c for c in candidates
        if _strip_brackets(normalize_units(c["item_name"])) == normalized_query_base
    ]
    exact_unique_match = len(exact_matches) == 1

    confident = exact_unique_match or (
        candidates[0]["score"] >= min_score
        and overlap_ratio >= overlap_threshold
        and len(candidates) >= min_candidates
        and not score_ambiguous
    )

    result = {
        "candidates": candidates,
        "confident": confident,
        "overlap_ratio": round(overlap_ratio, 2),
    }

    if not confident:
        # 후보들이 왜 하나로 안 좁혀지는지 LLM이 알 수 있게, 성분 기준으로 묶어서 보여준다.
        # (예: "게보린"은 이름은 같아도 아세트아미노펜 계열/이부프로펜 계열로 실제 성분이 갈림)
        candidate_groups = _group_candidates_by_ingredient(candidates, ingr_sets)
        result["candidate_groups"] = candidate_groups

        # 그룹 안에 제품이 여러 개면(예: 타이레놀 계열 안에 정/산제/현탁액/서방정/감기
        # 복합제가 다 섞여 있음) 대표 제품 하나만 물어보면 나머지 제품이 묻혀버리므로
        # 전부 나열한다. LLM한테 이 규칙을 글로만 지시하면 안정적으로 안 지키는 걸
        # 확인해서, 문장 자체를 코드에서 확정한다.
        questions = []
        for g in candidate_groups:
            if len(g["products"]) == 1:
                questions.append(f"{g['ingredients'][0]} 계열의 {g['products'][0]}을 말씀하시는 건가요?")
            else:
                product_list = ", ".join(g["products"])
                questions.append(
                    f"{g['ingredients'][0]} 계열에는 {product_list} 등 여러 제품이 있습니다. "
                    "그중 어떤 제품을 말씀하시는 건가요?"
                )
        result["clarification_question"] = " ".join(questions)

    return result


def get_product_detail(item_seq: str):
    """의약품 item_seq로 상세정보(효능효과/용법용량/주의사항/상호작용/이상반응) 조회."""
    _ensure_loaded()

    base = _list_by_seq.get(item_seq)
    if not base:
        return None

    detail = _get_detail(item_seq)
    easy = _easy_by_seq.get(item_seq, {})

    efcy = easy.get("efcyQesitm") or doc_xml_to_text(detail.get("EE_DOC_DATA"))
    usemethod = easy.get("useMethodQesitm") or doc_xml_to_text(detail.get("UD_DOC_DATA"))
    caution = " ".join(filter(None, [easy.get("atpnWarnQesitm"), easy.get("atpnQesitm")])) \
        or doc_xml_to_text(detail.get("NB_DOC_DATA"))

    ingredients = _ingredients_by_seq.get(item_seq, [])
    has_interaction_data = any(ingr in _taboo_ingredients for ingr in ingredients)

    return {
        "item_seq": item_seq,
        "item_name": base.get("ITEM_NAME"),
        "entp_name": base.get("ENTP_NAME"),
        "spclty_pblc": base.get("SPCLTY_PBLC"),
        "cancel_name": base.get("CANCEL_NAME"),
        "cancel_date": base.get("CANCEL_DATE"),
        "ingredients": ingredients,
        "효능효과": efcy,
        "용법용량": usemethod,
        "사용상주의사항": caution,
        "상호작용": easy.get("intrcQesitm"),
        "이상반응": easy.get("seQesitm"),
        "has_interaction_data": has_interaction_data,
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
        result = search_product_with_confidence(name, top_k=5)
        cands = result["candidates"]
        if not cands or not result["confident"]:
            # 신뢰 불가 -> 엉뚱한 약으로 조용히 진행하지 않고 미해결로 남긴다.
            resolved.append({"query": name, "item_seq": None, "item_name": None,
                              "candidates": cands, "ingredients": [],
                              "ambiguous": bool(cands),
                              "candidate_groups": result.get("candidate_groups", []),
                              "clarification_question": result.get("clarification_question")})
            continue
        best = cands[0]
        resolved.append({
            "query": name,
            "item_seq": best["item_seq"],
            "item_name": best["item_name"],
            "candidates": cands,
            "ingredients": _ingredients_by_seq.get(best["item_seq"], []),
            "ambiguous": False,
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
