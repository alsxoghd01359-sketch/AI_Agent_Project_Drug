"""병용확인으로 넘어온 이름이 의약품인지 음식·음료인지 분류한다.

음식 종류는 목록으로 관리하기 어려우므로 LLM에게 묻는다.
병용확인 경로에서 이름이 넘어올 때만 호출하므로 추가 비용은 제한적이다.
"""
import json
import re

from paragraph_search import _get_client

_KIND_PROMPT = """다음 단어들이 각각 무엇인지 분류하세요.
- 의약품(약 이름)이면 "drug"
- 음식, 음료, 식품, 기호식품(예: 과일 주스, 커피, 술, 차)이면 "food"
- 판단이 어려우면 "drug"

단어 목록: {names}

답은 JSON 객체만 출력하세요. 예: {{"자몽주스": "food", "타이레놀": "drug"}}
"""


def classify_names(names, model):
    """{이름: "drug" 또는 "food"}를 반환한다. 파싱에 실패하면 모두 "drug"로 본다."""
    fallback = {n: "drug" for n in names}
    resp = _get_client().chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": _KIND_PROMPT.format(names=", ".join(names))}],
    )
    content = resp.choices[0].message.content or ""
    match = re.search(r"\{.*\}", content, flags=re.S)
    if not match:
        return fallback
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return fallback
    return {n: ("food" if str(parsed.get(n, "drug")).lower() == "food" else "drug") for n in names}
