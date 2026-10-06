"""질문에 조건이 두 개 이상 들어 있을 때, 조건마다 따로 근거를 찾아 합친다.

한 번에 임베딩하면 여러 조건 중 하나의 문단만 상위에 올라와서 다른 조건의 근거가 빠질 수 있다.
조건별로 따로 검색하면 각 조건이 자기 근거를 갖는다.
"""
import json
import re

from paragraph_search import search_paragraphs, _get_client

_EXTRACT_PROMPT = """사용자 질문: "{question}"

이 질문에 적힌 복용자의 상황을 하나씩 나눠서, 각 상황을 조회용 질문 한 문장으로 만드세요.
- 질문에 적힌 상황만 쓰고, 없는 상황을 추가하지 마세요.
- 약 이름은 넣지 마세요.
- 행동이면 "~해도 되나요?" 형태로 씁니다. 예: "운전해도 되나요?", "술을 마셔도 되나요?"
- 상태(질환, 나이 등)이면 "~인데 먹어도 되나요?" 형태로 씁니다. 예: "간이 안 좋은데 먹어도 되나요?"
- 상황이 하나면 하나만 넣으세요.
- 답은 JSON 배열만 출력하세요.

예시
질문: "심바로드정 나이가 많고 간도 안 좋은데 먹어도 되나요?" -> ["나이가 많은데 먹어도 되나요?", "간이 안 좋은데 먹어도 되나요?"]
질문: "타이레놀정500mg 먹고 술 마셔도 돼?" -> ["술을 마셔도 되나요?"]
질문: "에나폰정10밀리그램 녹내장이 있고 수술도 받을 예정인데 먹어도 되나요?" -> ["녹내장이 있는데 먹어도 되나요?", "수술을 받아도 되나요?"]
질문: "스무디핀정25밀리그램 운전하면서 술도 마시는데 먹어도 되나요?" -> ["운전해도 되나요?", "술을 마셔도 되나요?"]
"""


def extract_conditions(question, model):
    """질문의 조건 목록을 반환한다. 파싱에 실패하면 질문 전체를 하나의 조건으로 본다."""
    resp = _get_client().chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": _EXTRACT_PROMPT.format(question=question)}],
    )
    content = resp.choices[0].message.content or ""
    match = re.search(r"\[.*\]", content, flags=re.S)
    if not match:
        return [question]
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return [question]
    items = [str(x).strip() for x in items if str(x).strip()]
    return items or [question]


def search_by_conditions(item_seq, field, conditions, top_k=8):
    """조건마다 따로 검색하고, 같은 문단은 한 번만 남긴다. 각 항목에 근거가 된 조건을 붙인다."""
    merged = []
    seen = set()
    for condition in conditions:
        for p in search_paragraphs(item_seq, field, condition, top_k=top_k):
            if p["text"] in seen:
                continue
            seen.add(p["text"])
            merged.append({"text": p["text"], "score": p["score"], "condition": condition})
    return merged
