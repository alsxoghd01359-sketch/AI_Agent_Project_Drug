"""자연어 질문 -> (응급 감지) -> LLM 도구 호출 -> 우리 함수 실행 -> 최종 답변.

응급 감지는 LLM 호출 전에 코드로 먼저 걸러낸다("코드가 먼저 가로챈다" 원칙).
그 외에는 OpenAI의 tool calling으로 어떤 함수를 어떤 인자로 부를지 LLM이
판단하게 하고, 우리는 실제 실행과 결과 전달만 담당한다.
"""
import html
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI

from emergency_detect import detect_emergency
from drug_lookup import check_multiple_drugs, search_product_with_confidence, get_product_detail, _get_detail
from text_extract import doc_xml_to_text
from symptom_search import search_by_symptom
from paragraph_search import search_paragraphs, _get_paragraphs
from multi_condition import extract_conditions
from name_kind import classify_names

load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHAT_MODEL = "gpt-4o-mini"
_OVERDOSE_KEYWORDS = ("과다", "과량", "한꺼번에")

# 채점용 진단 정보(검색 적중률/근거 표시 정확도/무응답 처리 참고 자료)를 답변 끝에
# 덧붙일지 여부. .env의 SCORE_DEBUG=true/false로 켜고 끈다(코드 수정 없이 토글).
# 기본값 false — 이 줄 또는 .env의 SCORE_DEBUG 설정만 지우면 이 기능은 완전히 비활성화된다.
SCORE_DEBUG = os.getenv("SCORE_DEBUG", "false").strip().lower() in ("1", "true", "yes")

_NO_ANSWER_MARKERS = ("찾을 수 없습니다", "등록되어 있지 않습니다", "등록된 정보가 없습니다")

_CITATION_FOLLOWUP_PROMPT = """질문: "{question}"

[후보 목록]
{candidates}

위 후보들을 보고 작성된 아래 답변을 보세요:
{answer}

이 답변이 실제로 근거로 사용한 후보 번호를 전부 쉼표로 구분해서만 답하세요(예: "1, 3").
근거로 쓴 후보가 없으면 "없음"이라고만 답하세요. 다른 말은 하지 마세요."""


def _detect_citations(question: str, lifestyle_result: dict | None, answer: str) -> list[int]:
    """메인 답변 생성 프롬프트(규칙이 이미 많음)에 "근거 번호를 같이 밝혀라"를
    끼워 넣었더니 LLM이 자주 생략하는 걸 확인했다 — 지시가 여러 규칙 속에 묻히면
    잘 안 지켜진다. 그래서 별도의 짧고 단순한 후속 질문으로 분리하니 안정적으로
    따랐다(채점 전용 스크립트에서 검증). SCORE_DEBUG가 꺼져 있으면 이 함수 자체가
    호출되지 않으므로 평소엔 추가 비용이 없다."""
    if not lifestyle_result:
        return []
    candidates = lifestyle_result.get("relevant_paragraphs", [])
    if not candidates:
        return []
    cand_text = "\n".join(f"{i + 1}. {c['text']}" for i, c in enumerate(candidates))
    prompt = _CITATION_FOLLOWUP_PROMPT.format(question=question, candidates=cand_text, answer=answer)
    resp = _client.chat.completions.create(
        model=CHAT_MODEL, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    reply = resp.choices[0].message.content or ""
    if "없음" in reply:
        return []
    return [n for n in (int(x) for x in re.findall(r"\d+", reply)) if 1 <= n <= len(candidates)]


def _truncate_for_debug(value, max_len=300):
    """채점용 진단 블록에 조회 데이터를 그대로 덤프하면 사용상주의사항 전문처럼
    긴 필드 때문에 가독성이 떨어진다. 사람이 읽는 화면에만 자르고, 실제 채점
    (_auto_grade)에는 원본 전체를 그대로 넘긴다."""
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + f" …(이하 생략, 전체 {len(value)}자)"
    if isinstance(value, list):
        return [_truncate_for_debug(v, max_len) for v in value]
    if isinstance(value, dict):
        return {k: _truncate_for_debug(v, max_len) for k, v in value.items()}
    return value


def _build_score_debug_block(last_tool_name, last_tool_result, lifestyle_result, cited_nums, answer_text):
    """채점 기준 4가지(검색 적중률/답 정확도/근거 표시 정확도/무응답 처리)에 맞춰
    참고 자료를 조립한다. "답 정확도"는 정답을 모르는 상태라 시스템이 스스로 판정할
    수 없으므로, 판단에 필요한 원재료(후보 전체·인용된 근거·무응답 여부)만 제공하고
    실제 채점은 사람이 하도록 한다."""
    lines = ["", "──────── [채점용 진단 정보 | SCORE_DEBUG] ────────"]
    if lifestyle_result is None:
        lines.append(f"※ 이번 턴은 ask_lifestyle_question(임베딩 검색)을 쓰지 않았습니다 "
                      f"— {last_tool_name or '(도구 호출 없음)'} 경로라 유사도/검색 적중률 지표가 없습니다.")
        if last_tool_result:
            lines.append("[조회된 데이터] 이 답변의 근거가 된 조회 결과(긴 필드는 일부만 표시):")
            data_str = json.dumps(_truncate_for_debug(last_tool_result), ensure_ascii=False, indent=2, default=str)
            lines.extend("  " + ln for ln in data_str.splitlines())
        else:
            lines.append("[조회된 데이터] 도구 호출 자체가 없었습니다(응급 안내, 되묻기 등).")
    else:
        candidates = lifestyle_result.get("relevant_paragraphs", [])
        lines.append(f"[검색 적중률] top-{len(candidates)} 후보 전체(유사도 내림차순) — "
                      "정답 근거가 이 안에 있는지 확인하세요:")
        for i, c in enumerate(candidates, 1):
            lines.append(f"  {i}. {c['score']:.4f} | {c['text'][:80]}")
        lines.append("")
        if cited_nums:
            lines.append("[근거 표시 정확도] 실제 답변에 인용됐다고 모델이 밝힌 후보:")
            for n in cited_nums:
                if 1 <= n <= len(candidates):
                    c = candidates[n - 1]
                    lines.append(f"  {n}번 ({c['score']:.4f}) | {c['text'][:80]}")
                else:
                    lines.append(f"  {n}번 — 범위를 벗어난 번호(모델 응답 오류 가능성)")
        elif not candidates:
            lines.append("[근거 표시 정확도] 후보 자체가 없었습니다(재랭킹 단계에서 전부 무관하다고 걸러짐).")
        else:
            lines.append("[근거 표시 정확도] 후보는 있었지만 모델이 그 중 어느 것도 "
                          "근거로 쓰지 않았다고 답했습니다(답변이 후보 내용을 벗어난 "
                          "방식으로 작성됐을 가능성 — 확인 필요).")
    no_answer = any(m in (answer_text or "") for m in _NO_ANSWER_MARKERS)
    lines.append("")
    lines.append(f"[없는 질문 처리] 이번 답변이 \"근거 없음\"으로 처리됨: {'예' if no_answer else '아니오'}")
    lines.append("[답 정확도] 정답은 이 시스템이 알 수 없으므로, 위 근거와 최종 답변을 보고 직접 판정하세요.")
    lines.append("──────────────────────────────────────────")
    return "\n".join(lines)


_GRADING_CRITERIA = [
    ("검색적중률", "검색 적중률"),
    ("답정확도", "답 정확도"),
    ("근거표시정확도", "근거 표시 정확도"),
    ("없는질문처리", "없는 질문 처리"),
]

_GRADING_PROMPT = """당신은 의약품 정보 챗봇의 답변을 채점하는 평가자입니다. 아래 4개 기준으로
각각 1.0(완전히 정확) / 0.5(부분적으로 정확) / 0.0(틀림) 중 하나를 매기고, 한 줄 이유를 쓰세요.

[질문]
{question}

[조회된 데이터] (도구: {tool_name})
{data}

[최종 답변]
{answer}

채점 기준:
1. 검색적중률: 조회된 데이터 안에 이 질문에 실제로 답이 되는 내용이 있었는가?
   (유사도 점수가 있으면 참고하되, 실제로 내용이 맞는지 직접 읽고 판단하세요.
   구조화 조회라 유사도가 없으면 그 데이터 안에 질문과 관련된 필드가 있었는지로 판단)
   이 질문이 애초에 데이터에 답이 없는 네거티브 케이스이고, 조회 결과도 실제로
   아무 근거를 못 찾았다면(후보 없음/전부 무관), 이는 검색 실패가 아니라 "없는
   것을 정확히 없다고 가려낸 것"이므로 1.0을 주세요.
2. 답정확도: 최종 답변의 내용이 조회된 데이터와 실제로 일치하는가? 데이터에 없는
   내용을 지어내거나 왜곡했으면 0.0, 일부만 왜곡/과장했으면 0.5, 완전히 일치하면 1.0.
   답변이 "찾을 수 없다/등록된 정보가 없다"인 경우, 실제로 데이터에 답이 없는 게
   맞다면 이것도 "데이터와 일치하는 정확한 답"이므로 1.0을 주세요(4번 기준과
   별개로, 여기서 0.0을 줄 이유가 없습니다) — 반대로 데이터에 답이 있는데도
   "찾을 수 없다"고 했다면 그때 0.0입니다.
3. 근거표시정확도: 답변이 구체적인 원문·근거를 명확히 제시했는가, 아니면 막연하게만
   설명했는가? 데이터 자체에 보여줄 근거가 애초에 없는 네거티브 케이스에서, 근거를
   지어내지 않고 정직하게 "찾을 수 없다"고만 답했다면 이것도 1.0입니다(보여줄 게
   없어서 안 보여준 것은 감점 사유가 아닙니다) — 반대로 데이터에 분명한 근거가
   있는데도 구체적 원문 인용 없이 막연하게만 설명했다면 그때 감점하세요.
4. 없는질문처리: 데이터에 답이 없으면 정직하게 "찾을 수 없다/등록된 정보가 없다"고
   했는가? 데이터에 답이 있는데 억지로 "없다"고 하거나, 반대로 없는데 답을 지어낸
   경우 0.0으로 매기세요.

이 점수는 참고용 자동 채점이며 사람의 최종 판단을 대체하지 않습니다. 그래도 최대한
근거 데이터를 꼼꼼히 읽고 신중하게 채점하세요.

반드시 이 JSON 형식으로만 답하세요(다른 텍스트 금지):
{{"검색적중률": {{"점수": 0, "이유": "..."}}, "답정확도": {{"점수": 0, "이유": "..."}},
  "근거표시정확도": {{"점수": 0, "이유": "..."}}, "없는질문처리": {{"점수": 0, "이유": "..."}}}}"""


def _auto_grade(question: str, tool_name: str | None, tool_result: dict | None, answer: str):
    """사용자가 준 4개 채점 기준(검색 적중률/답 정확도/근거 표시 정확도/없는 질문
    처리)을 LLM 심사위원에게 맡겨 0.0/0.5/1.0으로 자동 채점한다.

    한계: "답정확도"와 "검색적중률"은 원칙적으로 외부에서 정해진 정답(ground truth)과
    비교해야 하는데, 이 시스템은 정답지를 갖고 있지 않다. 그래서 실제로 채점하는 건
    "객관적 진실과 일치하는가"가 아니라 "조회된 데이터 자체와 앞뒤가 맞는가
    (grounded한가)"이다 — 데이터 자체가 틀렸거나 애매하면 이 자동 채점도 똑같이
    틀릴 수 있다. 참고용으로만 쓰고 최종 채점은 사람이 하는 걸 전제로 한다."""
    if not tool_result:
        return None
    data_str = json.dumps(tool_result, ensure_ascii=False, default=str)[:4000]
    prompt = _GRADING_PROMPT.format(question=question, tool_name=tool_name or "(없음)", data=data_str, answer=answer)
    resp = _client.chat.completions.create(
        model=CHAT_MODEL, temperature=0,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        return None


def _build_grading_block(grades):
    lines = ["", "[자동 채점 | LLM 심사위원, 참고용 — 사람 채점을 대체하지 않음]"]
    if not grades:
        lines.append("  채점에 쓸 조회 데이터가 없어 건너뜀.")
        return "\n".join(lines)
    total = 0.0
    for key, label in _GRADING_CRITERIA:
        g = grades.get(key, {}) if isinstance(grades, dict) else {}
        score = g.get("점수", 0) if isinstance(g, dict) else 0
        reason = g.get("이유", "") if isinstance(g, dict) else ""
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        total += score
        lines.append(f"  {label}: {score:g} — {reason}")
    lines.append(f"  합계: {total:g} / 4.0")
    return "\n".join(lines)

SYSTEM_PROMPT = """당신은 식약처 공공데이터를 조회해서 사실을 전달하는 복약 정보 도우미입니다.

절대 원칙:
1. 조회된 데이터에 있는 내용만 전달하세요. 의학적 판단이나 추천을 하지 마세요.
2. "안전합니다", "괜찮습니다", "문제없습니다" 같은 확신 표현을 쓰지 마세요.
   대신 "등록된 정보가 없습니다"처럼 사실 서술형으로만 답하세요.
3. 성분/효능/용법/주의사항/상호작용처럼 실제 조회된 데이터를 전달하는 답변에는
   [조회된 사실], [주의사항], [상담 권유] 순서를 지키세요. 섹션 사이에 화살표(->)는 쓰지 마세요.
   - [주의사항]은 "데이터의 한계로 추가 정보를 제공할 수 없습니다"처럼 우리 쪽 데이터가
     부족해서 못 알려준다는 식으로 쓰지 마세요. 대신 "이 정보는 일반적인 내용이며,
     개인의 체질이나 건강 상태, 복용 중인 다른 약에 따라 실제로 다르게 적용될 수
     있습니다"처럼, 원래 의약품 정보 자체가 개인마다 다르게 적용된다는 취지로 쓰세요.
   - [상담 권유]에는 "전문가"라는 두루뭉술한 표현 대신 "약사나 의사"라고 구체적으로
     쓰세요.
   단, 답변이 실제 데이터를 전달하는 게 아니라 제품을 특정해달라고 되묻기만 하는
   경우(5, 6, 7-(3)번 규칙에 해당하는 경우)에는 이 3단계 구조와 8번의 데이터 출처
   표기를 쓰지 마세요. 아직 아무 데이터도 조회해서 알려준 게 없는데 주의사항이나
   상담 권유를 붙이면 어색하므로, 되묻는 문장만 간결하게 전달하세요.
   도구 결과에 answer 필드가 있으면 이 3단계 구조를 직접 조립하지 말고 10번 규칙을
   따르세요 — answer 안에 이미 같은 구조(라벨, 글머리표, 데이터 출처까지)가 완성되어
   있으므로 그대로 출력하면 됩니다.
4. 병용금기가 등록되지 않았다고 해서 안전하다는 뜻이 아님을 항상 명시하세요.
   check_drug_interactions 결과가 answer 필드를 포함하면(제품이 모두 특정된 경우),
   그 안에 "(복용 후 주의)" bullet로 이미 포함되어 있으니 10번 규칙대로 그대로
   출력하세요.
5. 도구 결과의 resolved가 false이거나 confident가 false면, 절대 특정 제품으로
   단정하지 마세요. 도구 결과에 clarification_question 필드가 있으면, 그
   문장을 직접 새로 만들지 말고 반드시 그 문장을 그대로(고치거나 요약하지 말고)
   답변에 포함해서 되물으세요. 성분 목록이나 제품 목록을 따로 나열하지 마세요.
6. check_drug_interactions 결과에 unresolved_queries가 있으면(제품명이 특정
   안 된 약이 포함된 경우), 그 약이 "어떤 성분이면 괜찮고 어떤 성분이면
   안 되는지" 같은 조건부 추정이나 부분적 비교를 스스로 만들어내지 마세요.
   병용 여부 판단은 성분이 확정된 뒤에만 가능하므로, 해당 약의
   clarification_question을 그대로 전달해 제품부터 특정해달라고 요청하고,
   나머지 확정된 약들 사이의 결과만 사실대로 전달하세요. unresolved_queries에
   약이 2개 이상 있으면, "여러 제품이 있습니다"처럼 뭉뚱그리지 말고 각 약의
   clarification_question을 하나도 빠짐없이 전부 그대로 나열하세요.
7. 사용자가 특정 제품명을 언급하면서 "먹어도 되나요/복용해도 되나요/안전한가요/
   괜찮나요"라고 묻는 경우에만 아래 순서를 적용하세요. ("두통에 먹는 약 뭐
   있어?"처럼 제품명 없이 증상만 말하는 질문은 이 규칙이 아니라
   search_by_symptom을 바로 쓰세요.)
   (1) 질문 안에 다른 "의약품"(등록된 약) 이름이 이미 나와 있다 -> 되묻지
       말고 즉시 check_drug_interactions를 호출하세요. 주의: "A이랑 B 같이
       먹어도 되나요" 같은 문장 구조만 보고 B를 자동으로 "다른 약"이라고
       판단하지 마세요 — 질병명(우울증, 당뇨 등), 음식·음료(커피, 자몽주스 등),
       증상·일반명사(해열제, 감기약 등)는 의약품 이름이 아니므로 이 규칙 대상이
       아닙니다. 그런 경우는 (2)로 가세요.
   (2) 질문 안에 구체적인 생활 상황·동반 조건이 이미 나와 있다 -> 되묻지 말고
       즉시 ask_lifestyle_question을 호출하세요. 판단 기준은 "9번 규칙의
       dur_cautions 7개 유형(노인주의/특정연령대금기/용량주의/투여기간주의/
       서방정분할주의/임부금기/첨가제주의)에 해당하는가"입니다 — 해당하면
       (3)의 get_drug_info로, 해당하지 않으면 전부 이 경우입니다. 음주/공복/
       운전은 물론, 수술 예정/자몽주스/녹내장/전립선비대/간질환/신장질환/
       두드러기·알레르기 병력처럼 dur_cautions에 없는 주제는 모두 여기
       해당합니다. "음주/공복/운전 등"이라는 예시 몇 개에 없다고 해서 (3)으로
       보내지 마세요 — 예시는 전부가 아니라 일부일 뿐입니다.
   (3) 위 두 경우에 해당하지 않고 약 이름 하나만 있다 -> 되묻지 말고 바로
       get_drug_info를 호출하세요. 복용 중인 다른 약이 없는 사용자에게 매번
       "다른 약 드세요?"부터 묻는 건 불필요한 질문을 하나 더 받게 만드므로,
       먼저 정보부터 주는 게 맞습니다. 결과의 answer는 10번 규칙대로 그대로
       출력하면 되고, 병용금기 관련 안내가 필요하면 이미 answer 안에 들어가
       있습니다. 사용자가 나중에 다른 약 이름을 알려주면 그때
       check_drug_interactions를 호출하세요.
   (1)과 (2)에 해당하는 질문에는 절대로 "현재 다른 약을 복용하고 계신가요?"
   라고 되묻지 마세요 — 질문에 이미 답이 나와 있습니다.
8. 도구 결과에 data_source 필드가 있으면, 답변 마지막에 "데이터 출처: {그 값}"을
   그대로 쓰세요. 어떤 출처를 쓸지 직접 판단하거나 "DUR" 같은 다른 용어로
   바꾸지 말고, data_source 값을 그대로 옮기세요. 실제 데이터를 전달하지 않고
   되묻기만 하는 답변(3번 예외 참고)에는 이 출처 표기 자체를 넣지 마세요.
9. get_drug_info 결과의 cancel_name·cancel_date(판매 중지 여부), dur_cautions
   (노인주의/특정연령대금기/용량주의/투여기간주의/서방정분할주의/임부금기/
   첨가제주의 중 질문에 해당하는 유형)는 이제 Python이 질문에 맞게 골라서 answer
   안에 이미 넣어 두었습니다. 10번 규칙대로 answer를 그대로 출력하면 됩니다.
10. get_drug_info, ask_lifestyle_question, get_overdose_info, check_drug_interactions
    (제품이 모두 특정된 경우) 결과에 answer 필드가 있으면, 그 문자열을 글자 하나,
    줄바꿈이나 라벨 하나도 바꾸지 말고 통째로 그대로 출력하세요 — [조회된 사실]/
    [정리]/약사 상담/데이터 출처까지 이미 다 조립되어 있습니다. 요약하거나, 라벨을
    **굵게**나 다른 말로 바꾸거나, 문장을 합치거나 순서를 바꾸거나, 근거에 없는
    말을 덧붙이지 마세요.
11. 사용자가 과다복용·과량 복용 결과를 물으면(예: "타이레놀 과다복용하면 어떻게 돼?"),
    제품이 특정된 뒤 ask_lifestyle_question(NB)으로 과량투여 관련 원문을 찾아 그대로
    전달하세요. 제품이 특정되지 않았으면 제품부터 되물으세요. 지금 증상이 심하다고
    하면 119 연락을 먼저 안내하고, 그 다음에 원문 내용을 덧붙이세요.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_drug_interactions",
            "description": "2개 이상의 의약품을 함께 복용해도 되는지 확인한다. "
                            "의약품안전사용서비스 병용금기 등록 여부, 동일 성분 중복"
                            "(과다복용 위험), 효능군중복(성분은 달라도 효과가 겹치는 "
                            "경우) 여부를 반환한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "확인할 제품명 목록 (2개 이상)",
                    }
                },
                "required": ["product_names"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_drug_info",
            "description": "제품명으로 의약품을 검색해서 성분, 효능효과, 용법용량, "
                            "사용상주의사항, 상호작용, 이상반응, 판매상태를 조회한다. "
                            "용도는 두 가지뿐이다: (1) 질문에 아무 구체적 조건이 없는 "
                            "'이 약 뭐예요/먹어도 되나요' 같은 일반 정보 요청, "
                            "(2) 노인주의/특정연령대금기/용량주의/투여기간주의/"
                            "서방정분할주의/임부금기 7개 DUR 유형 전용 질문('노인이 "
                            "먹어도 되나요', '하루 몇 알까지', '쪼개 먹어도 되나요', "
                            "'임신 중인데 먹어도 되나요'). 이 두 경우가 아니라 음주/"
                            "운전/공복/수술/자몽주스/녹내장/전립선/간질환/신장질환/"
                            "알레르기처럼 구체적인 동반 상황이나 질환·물질이 언급되면, "
                            "이 도구에도 사용상주의사항 전문이 들어있다고 해서 쓰지 "
                            "말고 반드시 ask_lifestyle_question을 대신 쓴다 — 이 도구는 "
                            "문서가 길면 그 안에 묻힌 특정 조항을 놓칠 수 있어서 그런 "
                            "질문에는 신뢰도가 낮다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "조회할 제품명"}
                },
                "required": ["product_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_symptom",
            "description": "증상에 맞는 일반의약품 성분 후보를 공식 약효분류 기준으로 찾는다. "
                            "제품을 특정하지 않은 '두통에 먹는 약 뭐 있어?' 같은 질문에 사용. "
                            "지원 증상: 두통/열/발열/근육통/치통/기침/가래/설사/변비/소화불량/속쓰림/콧물/비염",
            "parameters": {
                "type": "object",
                "properties": {
                    "symptom": {"type": "string", "description": "증상 키워드"},
                    "current_medication_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "사용자가 현재 복용 중인 약 이름(있으면 병용금기 성분 제외)",
                    },
                },
                "required": ["symptom"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_lifestyle_question",
            "description": "이미 특정된 제품 하나에 대해, 7개 DUR 유형(노인주의/특정연령대금기/"
                            "용량주의/투여기간주의/서방정분할주의/임부금기)에 해당하지 않는 "
                            "구체적인 동반 상황·동반 질환·병용 물질이 언급된 질문에 답하기 위해 "
                            "전용 검색(임베딩+재랭킹)으로 관련 문단을 찾는다. 예: '술 마셔도 되나', "
                            "'운전해도 되나', '공복에 먹어야 하나', '수술 받아도 되나', '자몽주스 "
                            "마셔도 되나', '녹내장/전립선비대/간질환/신장질환 있는데 먹어도 되나', "
                            "'이 부작용 정상인가'. get_drug_info가 반환하는 사용상주의사항 전문은 "
                            "문서가 길면 특정 조항이 그 안에 묻혀 get_drug_info만으로는 놓칠 수 "
                            "있으므로, 이런 구체적 상황 질문은 '이미 사용상주의사항을 조회했으니 "
                            "됐다'고 넘기지 말고 반드시 이 도구를 추가로 호출한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"},
                    "question": {"type": "string", "description": "사용자의 원래 질문 문장"},
                    "field": {
                        "type": "string",
                        "enum": ["NB", "UD", "EE"],
                        "description": "NB=사용상주의사항(기본값), UD=용법용량, EE=효능효과",
                    },
                },
                "required": ["product_name", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_overdose_info",
            "description": "제품명으로 과다복용·과량 투여 관련 사용상주의사항 원문을 조회한다. "
                            "사용자가 과다복용, 과량, 한꺼번에 많이 먹은 경우의 결과를 물을 때 사용한다.",
            "parameters": {
                "type": "object",
                "properties": {"product_name": {"type": "string", "description": "조회할 제품명"}},
                "required": ["product_name"],
            },
        },
    },
]


_SYMPTOM_SAFETY_BULLET = (
    "- (복용 후 주의) 이 약들을 함께 복용한 후 평소와 다른 증상(어지러움, 메스꺼움, 두드러기, "
    "졸림 등)이 나타나면 즉시 복용을 중단하고 의사나 약사와 상담하세요. 증상이 심하면 119에 연락하세요."
)


def _interaction_facts(result):
    """병용금기·성분 중복·효능군 중복을 bullet로 나열한다. 어떤 종류인지 라벨을 붙여서
    한 줄로 뭉쳐 보여주던 것보다 어떤 조합이 왜 문제인지 구분해서 볼 수 있게 한다."""
    bullets = []
    for h in result.get("dur_taboo_matches", []):
        bullets.append(
            f"- (병용금기) {h['drug_a']}({h['ingredient_a']})과 {h['drug_b']}({h['ingredient_b']}): {h['reason']}"
        )
    for d in result.get("duplicate_ingredients", []):
        bullets.append(
            f"- (성분 중복) {d['ingredient']} 성분이 {', '.join(d['drugs'])}에 공통으로 들어 있어 과다복용 위험이 있습니다."
        )
    for e in result.get("effect_group_duplicates", []):
        bullets.append(
            f"- (효능군 중복) {e['drug_a']}과 {e['drug_b']}는 {e['effect_group']}에 속해 효과가 중복될 수 있습니다."
        )
    if not bullets:
        bullets.append("- 이 제품들의 조합은 의약품안전사용서비스에 등록된 병용금기가 없습니다.")
    bullets.append(_SYMPTOM_SAFETY_BULLET)
    return "\n".join(bullets)


def _interaction_summary(result):
    # 병용금기·성분 중복·효능군 중복은 서로 다른 성격의 위험이라, 가장 심각한 것부터
    # 하나만 골라 한 줄로 말한다(LLM에 맡기지 않고 Python이 정한다 — 이미 구조화된
    # 결과라 요약을 다시 판단할 필요가 없고, 매번 같은 결론이 나와야 한다).
    if result.get("dur_taboo_matches"):
        return "등록된 병용금기가 있으므로 함께 복용해서는 안 됩니다."
    if result.get("duplicate_ingredients"):
        return "같은 성분이 중복되어 과다복용 위험이 있습니다."
    if result.get("effect_group_duplicates"):
        return "효능이 겹치는 성분이 있어 과다복용과 비슷한 위험이 있습니다."
    return "등록된 병용금기가 없습니다."


def _tool_check_drug_interactions(product_names):
    result = check_multiple_drugs(product_names)

    unresolved = result.get("unresolved_queries") or []
    if unresolved:
        # 제품을 먼저 특정해야 비교할 수 있으므로, 비교·경고 필드는 아예 싣지 않는다.
        return {
            "unresolved_queries": unresolved,
            "clarification_questions": [
                r["clarification_question"] for r in result["resolved_drugs"]
                if r.get("clarification_question")
            ],
            "note": "제품이 특정되지 않아 병용 여부를 비교할 수 없습니다. "
                    "clarification_questions를 그대로 전달해 제품부터 특정해 달라고 요청하세요.",
        }

    item_seqs = {r["item_seq"] for r in result["resolved_drugs"] if r.get("item_seq")}
    if len(item_seqs) == 1:
        # 같은 제품을 여러 번 말한 경우: 병용이 아니라 과다복용 문제이므로 용법·용량 원문을 준다.
        seq = next(iter(item_seqs))
        detail = get_product_detail(seq)
        # 과량 문구는 e약은요 요약이 아니라 원본 NB 문서에만 있으므로 원본을 본다.
        raw_caution = doc_xml_to_text(_get_detail(seq).get("NB_DOC_DATA"))
        overdose_sentences = [s.strip() for s in re.split(r"\n|(?<=[.])\s+", raw_caution) if "과량" in s]
        facts = "\n".join(
            _split_field("용법용량", detail.get("용법용량") or "")
            + ([_plain_bullets(overdose_sentences, "과량투여")] if overdose_sentences else [])
        )
        summary = "같은 제품을 중복해서 드시는 것이므로, 병용이 아니라 과다복용에 주의해야 합니다."
        rx_closing = _PRESCRIPTION_NOTE if detail.get("spclty_pblc") == "전문의약품" else None
        return {
            "same_product_repeated": True,
            "item_name": detail.get("item_name"),
            "answer": _assemble_answer(facts, summary, "의약품 제품허가정보 기준", closing=rx_closing),
            "note": _ANSWER_NOTE,
        }

    # 병용 답변은 병용금기와 성분 중복만 다룬다. 제품별 연령·대상 주의는 병용과 무관하므로 싣지 않는다.
    rx_drugs = [
        r["item_name"] for r in result.get("resolved_drugs", [])
        if r.get("item_seq") and (get_product_detail(r["item_seq"]) or {}).get("spclty_pblc") == "전문의약품"
    ]
    rx_closing = (
        f"{', '.join(rx_drugs)}은 의사의 처방이 필요한 전문의약품입니다. 이 안내는 처방한 의사나 "
        "약사의 지침을 대신하지 않으니, 복용 여부와 방법은 처방 받은 지침을 따르세요."
    ) if rx_drugs else None
    for r in result.get("resolved_drugs", []):
        r.pop("dur_cautions", None)
    facts = _interaction_facts(result)
    summary = _interaction_summary(result)
    result["answer"] = _assemble_answer(facts, summary, "의약품안전사용서비스 기준", closing=rx_closing)
    result["note"] = _ANSWER_NOTE
    return result


_DUR_TYPE_KEYWORDS = {
    "임부금기": ("임신", "임산부", "임부", "수유"),
    "투여기간주의": ("며칠", "몇일", "얼마나 오래", "오래", "기간"),
    "서방정분할주의": ("쪼개", "나눠", "분할", "갈아"),
    "용량주의": ("하루 몇", "몇 알", "몇 정", "최대 용량", "용량"),
    "첨가제주의": ("첨가제", "부형제"),
}


def _dur_type_for_question(question):
    """질문이 7개 DUR 유형 중 하나를 콕 집어 물었으면 그 유형만, 아니면 None(전체 표시)."""
    for dtype, keywords in _DUR_TYPE_KEYWORDS.items():
        if any(k in question for k in keywords):
            return dtype
    return None


def _build_drug_info_answer(item_seq, detail, question, data_source):
    """get_drug_info 답변을 전부 Python에서 조립한다. 성분/효능/용법용량은 이미 깨끗한
    필드라 그대로 bullet로 쓰고, 사용상주의사항은 질문과 관련된 문단만 찾아 조항 이름을
    붙인다. dur_cautions는 질문이 특정 유형을 물었으면 그 유형만, 아니면 전체를 보여준다
    (어떤 유형을 보여줄지 LLM이 매번 다시 판단하게 하면 흔들린다)."""
    basic = []
    if detail.get("cancel_name") and detail["cancel_name"] != "정상":
        basic.append(
            f"- (판매 상태) 이 제품은 {detail.get('cancel_date')}자로 {detail['cancel_name']} 처리되어 "
            "더 이상 정상적으로 판매되지 않습니다. 복용 중이라면 약사나 의사와 상담하세요."
        )
    if detail.get("ingredients"):
        basic.append(f"- (성분) {', '.join(detail['ingredients'])}")
    basic += _split_field("효능효과", detail.get("효능효과") or "")
    basic += _split_field("용법용량", detail.get("용법용량") or "")

    paragraphs = search_paragraphs(item_seq, "NB", question or detail.get("item_name") or "", top_k=5)
    caution_md = _bullets_to_md(_render_bullets([p["text"] for p in paragraphs])) if paragraphs else ""

    dtype = _dur_type_for_question(question)
    dur_cautions = detail.get("dur_cautions", [])
    relevant = [d for d in dur_cautions if d.get("type") == dtype] if dtype else dur_cautions
    dur_lines = [
        f"- (DUR: {d['type']}) " + (
            d.get("content") or f"{d['type']}에 대한 주의가 필요한 성분/품목으로 등록되어 있으나 구체적인 문구는 없습니다."
        )
        for d in relevant
    ]
    if dtype and not relevant:
        dur_lines.append(f"- (DUR: {dtype}) 등록된 정보가 없습니다.")

    facts = "\n".join(filter(None, ["\n".join(basic), caution_md, "\n".join(dur_lines)])) or "등록된 정보가 없습니다."
    summary = _make_summary(facts)
    closings = []
    if detail.get("has_interaction_data"):
        closings.append(
            "이 약은 특정 약과 함께 복용 시 주의가 필요한 조합이 등록되어 있습니다. "
            "현재 복용 중인 다른 약이 있으면 알려주시면 추가로 확인해드리겠습니다."
        )
    if detail.get("spclty_pblc") == "전문의약품":
        closings.append(_PRESCRIPTION_NOTE)
    return _assemble_answer(facts, summary, data_source, closing=closings)


def _tool_get_drug_info(product_name, question="", build_answer=True):
    result = search_product_with_confidence(product_name)
    if not result["confident"]:
        return {
            "resolved": False,
            "candidate_groups": result.get("candidate_groups", []),
            "clarification_question": result.get("clarification_question"),
            "note": "제품명이 정확히 특정되지 않았습니다. clarification_question 문장만 그대로 답변으로 출력하세요. "
                    "'정보를 확인할 수 없다', '제품명이 정확히 특정되지 않았다' 같은 말을 덧붙이지 마세요.",
        }
    item_seq = result["candidates"][0]["item_seq"]
    detail = get_product_detail(item_seq, live_status=True)
    detail["resolved"] = True
    detail["data_source"] = (
        "의약품 제품허가정보 및 의약품안전사용서비스 기준"
        if detail.get("has_interaction_data")
        else "의약품 제품허가정보 기준"
    )
    # cancel_name이 "정상"이면 필드 자체를 안 보여준다. 프롬프트로 "정상이면
    # 언급하지 마라"고 아무리 명시해도 LLM이 "이 제품은 정상적으로 판매되고
    # 있습니다"를 계속 붙이는 걸 반복 확인함 — 볼 수 없는 값은 언급할 수 없으니
    # 아예 지워서 원천 차단한다.
    if detail.get("cancel_name") == "정상":
        detail.pop("cancel_name", None)
        detail.pop("cancel_date", None)
    if not build_answer:
        # get_age_dur_info처럼 dur_cautions 등 원본 필드만 필요한 내부 호출.
        # 매번 답변 문장까지 조립하면 search_paragraphs·LLM 호출이 낭비된다.
        return detail
    detail["answer"] = _build_drug_info_answer(item_seq, detail, question, detail["data_source"])
    detail["note"] = _ANSWER_NOTE
    return detail


def _tool_get_age_dur_info(product_name, question):
    detail = _tool_get_drug_info(product_name, build_answer=False)
    if not detail.get("resolved"):
        return detail
    group = _age_group(question)
    types = _AGE_DUR_TYPES.get(group, ())
    entries = [d for d in detail.get("dur_cautions", []) if d.get("type") in types]
    if not entries:
        # 연령 DUR 항목이 없으면 사용상주의사항 원문에서 질문과 관련된 문단을 찾는다.
        return _tool_ask_lifestyle_question(product_name, question)
    facts = "\n".join(
        f"- (DUR: {d['type']}) " + (
            d.get("content") or f"{d['type']}에 대한 주의가 필요한 성분/품목으로 등록되어 있으나 구체적인 문구는 없습니다."
        )
        for d in entries
    )
    rx_closing = _PRESCRIPTION_NOTE if detail.get("spclty_pblc") == "전문의약품" else None
    return {
        "resolved": True,
        "item_name": detail["item_name"],
        "answer": _assemble_answer(facts, None, detail["data_source"], closing=rx_closing),
        "note": _ANSWER_NOTE,
    }


def _tool_search_by_symptom(symptom, current_medication_names=None):
    ingredients = []
    for name in (current_medication_names or []):
        r = search_product_with_confidence(name)
        if r["confident"]:
            detail = get_product_detail(r["candidates"][0]["item_seq"])
            ingredients.extend(detail.get("ingredients", []))

    results = search_by_symptom(symptom, current_med_ingredients=ingredients)
    if results is None:
        return {
            "resolved": True,
            "answer": f"'{symptom}'은 지원하지 않는 증상 분류입니다. 다른 증상이나 구체적인 약 이름으로 다시 물어봐 주세요.",
            "note": _ANSWER_NOTE,
        }
    if not results:
        return {
            "resolved": True,
            "candidates": results,
            "answer": f"'{symptom}'에 해당하는 일반의약품 성분을 찾지 못했습니다.\n\n약사나 의사와 상담하시기 바랍니다.\n\n데이터 출처: 의약품 제품허가정보 기준",
            "note": _ANSWER_NOTE,
        }

    # 증상 하나에 매칭되는 성분이 수십~수백 개일 수 있다(예: 두통 80개 이상). 전부
    # bullet로 쏟아내면 읽기 어려우니 상위 몇 개만 보여주고 나머지는 개수만 알린다.
    # 병용금기로 제외해야 하는 성분은 안전과 직결되므로 cap과 무관하게 항상 보여준다.
    excluded = [r for r in results if r.get("excluded_reason")]
    others = [r for r in results if not r.get("excluded_reason")]
    max_shown = 12
    shown = excluded + others[:max(0, max_shown - len(excluded))]
    remaining = len(results) - len(shown)

    bullets = []
    for r in shown:
        line = f"- ({r['category']}) {r['ingredient']} — 예: {', '.join(r['examples'])}"
        if r.get("excluded_reason"):
            line += f" (주의: {r['excluded_reason']})"
        bullets.append(line)
    if remaining > 0:
        bullets.append(f"- 그 외 {remaining}개 성분이 더 있습니다. 구체적인 성분이나 제품명을 말씀해 주시면 확인해드리겠습니다.")
    facts = "\n".join(bullets)

    if excluded:
        names = ", ".join(r["ingredient"] for r in excluded)
        summary = f"{symptom}에 쓸 수 있는 성분 중 {names}은 현재 복용 중인 약과 병용금기가 등록되어 있어 피해야 합니다."
    else:
        summary = f"{symptom}에 쓸 수 있는 일반의약품 성분은 위와 같습니다. 구체적인 제품명을 알려주시면 더 확인해드리겠습니다."

    return {
        "resolved": True,
        "candidates": results,
        "answer": _assemble_answer(facts, summary, "의약품 제품허가정보 기준"),
        "note": _ANSWER_NOTE,
    }


_FOOD_CROSS_PROMPT = """복용자 상황: "{condition}"

1) 이 상황이 음식이나 음료에 관한 것인지 판단하세요. 음식·음료가 아니면(예: 운전, 질환, 나이, 다른 약 등) "해당없음"이라고만 답하세요.
2) 음식·음료에 관한 것이면, 아래 문단 중에 그 음식·음료를 직접 다루는 문단이 있는지 판단하세요.
   같은 뜻으로 쓰인 말도 인정합니다. 예: 술은 알코올, 커피는 카페인 등. "알코올섭취"는 술(음료)에 대한 내용이므로 있는 것으로 봅니다.
   약의 성분, 부형제, 수유 중 이행 같은 내용은 음식 자체에 대한 내용이 아닙니다.
   직접 다루는 문단이 있으면 "YES", 없으면 "NO"만 답하세요.

"해당없음", "YES", "NO" 중 하나만 답하세요.

[문단]
{paragraphs}
"""


def _food_cross_check(condition, paragraphs):
    """한 번의 호출로 "이 상황이 음식·음료에 관한 것인지"와 "문단이 그 음식을
    다루는지"를 같이 판단한다. 따로 부르면(이전에는 "음식 이름 추출" + "성분/음식
    분류" + "근거 확인" 최대 3번) 조건마다 호출이 늘어 느려진다."""
    text = "\n".join(f"- {p}" for p in paragraphs) if paragraphs else "(문단 없음)"
    resp = _client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": _FOOD_CROSS_PROMPT.format(condition=condition, paragraphs=text)}],
    )
    answer = (resp.choices[0].message.content or "").strip()
    if "해당없음" in answer:
        return "not_food"
    return "yes" if answer.upper().startswith("Y") else "no"




_CONDITION_PICK_PROMPT = """복용자 상황: "{condition}"

아래 문단 중에서 이 상황에 직접 해당하는 내용을 다루는 문단의 번호만 쉼표로 적으세요.
같은 뜻의 말은 같은 것으로 봅니다. 예: 콩팥과 신장, 신기능은 같은 뜻입니다. 음주와 알코올도 같은 뜻입니다.
상황의 대상(음식, 질환, 나이, 약 등)을 직접 언급하고 주의, 금기, 용량 조절 등을 말하는 문단을 고르세요.
대상이 단어만 같을 뿐 다른 내용을 말하는 문단은 제외합니다.
문단이 상황의 대상을 주제로 삼지 않고 다른 대상(예: 운전 상황에서 음주와의 병용, 신장 상황에서 다른 약과의 병용)을 주제로 삼으면 제외합니다.
이상반응 빈도 목록, 부작용 발현율 목록은 상황에 대한 주의가 아니므로 제외합니다.
해당하는 문단이 없으면 "없음"이라고만 답하세요.

[문단]
{paragraphs}
"""


def _pick_for_condition(condition, paragraphs):
    """조건에 직접 해당하는 문단만 고른다. 고른 문단은 원문 그대로 쓴다."""
    if not paragraphs:
        return []
    text = "\n".join(f"{i + 1}. {p['text']}" for i, p in enumerate(paragraphs))
    resp = _client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": _CONDITION_PICK_PROMPT.format(condition=condition, paragraphs=text)}],
    )
    numbers = {int(n) for n in re.findall(r"\d+", resp.choices[0].message.content or "")}
    return [p for i, p in enumerate(paragraphs, 1) if i in numbers]


def _pick_with_food_check(condition, raw):
    """조건에 맞는 문단을 고르고, 음식·음료 조건이면 고른 문단이 실제로 그 음식을
    다루는지 다시 확인한다. _pick_for_condition이 음식과 무관한 문단(예: "고추"
    조건에 음주 관련 문단)을 잘못 고르는 경우가 재현됐는데, 한 번 음식 근거
    판정을 통과하지 못하면 걸러낼 길이 없었다(이전에는 마지막 LLM 응답 단계에서
    가끔 걸러졌지만, 그 단계를 없앤 뒤로는 그대로 새어 나갔다)."""
    picked = _pick_for_condition(condition, raw)
    check_source = picked or raw
    if not check_source:
        return picked
    result = _food_cross_check(condition, [p["text"] for p in check_source[:3]])
    if result == "not_food":
        return picked
    if picked:
        return [] if result == "no" else picked
    return raw[:2] if result == "yes" else []


def _attach_body(picked, all_paragraphs):
    """짧은 제목 문단(예: "[11. 기타] ② 신기능 부전증")이 골라지면, 바로 뒤의 본문 문단을 함께 붙인다.
    본문은 같은 장 표시를 가지고 있고, 다음 소제목(짧은 문단)을 만나면 멈춘다."""
    texts = list(all_paragraphs)
    result = []
    seen = set()
    for p in picked:
        if p["text"] not in seen:
            seen.add(p["text"])
            result.append(p)
        if len(p["text"]) >= 60 or p["text"] not in texts:
            continue
        i = texts.index(p["text"])
        label = re.match(r"\[[^\]]*\]", p["text"])
        if not label:
            continue
        for j in range(i + 1, min(i + 3, len(texts))):
            nxt = texts[j]
            if not nxt.startswith(label.group(0)) or len(nxt) < 60:
                break
            if nxt not in seen:
                seen.add(nxt)
                result.append({"text": nxt, "score": p["score"], "condition": p.get("condition")})
    return result


_DURATION_OR_AMOUNT_RE = re.compile(r"(\d+|한|두|세|네|다섯)\s*(일|주|개월|달|알|정|회|번|시간)|넘게|하루|며칠|몇\s*(알|정|번|회|일)")

# 나이 질문은 DUR 항목의 원문을 코드에서 그대로 옮긴다. 사용상주의사항 전체를 LLM에 넘기면
# 여러 연령 문장이 섞여 "사용할 수 있다"처럼 원문에 없는 결론이 나왔다.
_AGE_DUR_TYPES = {"elderly": ("노인주의",), "child": ("특정연령대금기",)}


def _age_group(question):
    if re.search(r"노인|고령|어르신|(?<!\d)(6\d|[7-9]\d)\s*세", question):
        return "elderly"
    if re.search(r"소아|어린이|유아|아이(?:가|는|에게|한테|들|$| )|(?<!\d)1?\d\s*세", question):
        return "child"
    return None


_BULLET_PROMPT = """아래 문단을 번호 순서대로, 번호마다 한 문장씩 이해하기 쉬운 말로 바꾸세요.
문단에 없는 내용은 쓰지 마세요. 결론이나 주의를 말하는 표현(예: 무시해도 좋을 것으로 예상된다, 신중히 투여, 투여하지 말 것, 증강시킬 수 있다)은 원문 표현을 한 글자도 바꾸지 말고 그대로 옮기세요. 그 밖의 설명만 쉬운 말로 바꿀 수 있습니다.
번호와 줄 개수를 입력과 똑같이 유지하고 "1. 문장" 형식으로만 쓰세요. 다른 말은 덧붙이지 마세요.

{blocks}
"""


def _section_label(text):
    """문단 맨 앞의 [...] 표시(예: "[6. 상호작용]")를 조항 이름으로 뽑는다. 없으면 None.
    일부 제품은 이 표시 자체가 문단 내용과 거의 같은 긴 문장("[3. 다음과 같은 사람은
    ... 상의할 것]")이라, 그대로 라벨로 쓰면 본문과 중복돼 보인다. 짧은 조항명일 때만 쓴다."""
    m = re.match(r"\[([^\]]*)\]", text)
    if not m:
        return None
    label = m.group(1)
    return label if len(label) <= 22 else None


def _render_bullets(evidence_texts):
    """근거 문단마다 쉬운 말 한 문장 + 원문 조항 이름을 붙인 bullet 목록을 만든다.
    문단을 하나로 합쳐서 긴 문단으로 보여주면 가독성이 떨어지고 어느 조항에서 나온 내용인지
    알 수 없으므로, 문단 단위를 끝까지 유지하면서 문장만 쉬운 말로 바꾼다."""
    if not evidence_texts:
        return []
    labels = [_section_label(t) for t in evidence_texts]
    bodies = [re.sub(r"^\[[^\]]*\]\s*", "", t) for t in evidence_texts]
    numbered = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(bodies))
    resp = _client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": _BULLET_PROMPT.format(blocks=numbered)}],
    )
    rewritten = {}
    for line in (resp.choices[0].message.content or "").splitlines():
        m = re.match(r"\s*(\d+)[.\)]\s*(.+)", line)
        if m:
            rewritten[int(m.group(1))] = m.group(2).strip()
    return [
        {"label": labels[i], "text": rewritten.get(i + 1, bodies[i])}
        for i in range(len(evidence_texts))
    ]


def _bullets_to_md(bullets):
    lines = []
    for b in bullets:
        prefix = f"(출처: {b['label']}) " if b["label"] else ""
        lines.append(f"- {prefix}{b['text']}")
    return "\n".join(lines)


def _plain_bullets(sentences, tag):
    """[...] 라벨이 없는 문장 목록(과량투여 문구 등)을 bullet로 다듬는다.
    원문이 문장 단위로 쪼개져 있어서 "2) 과량투여시의 처치"처럼 제목만 있고 내용이
    없는 조각이 그대로 나오곤 했다. 번호를 떼고 LLM에게 한 문장씩 다시 쓰게 해서
    다른 bullet들과 톤을 맞춘다."""
    cleaned = [re.sub(r"^\d+[.)]\s*", "", s).strip() for s in sentences]
    cleaned = [s for s in cleaned if s]
    bullets = _render_bullets(cleaned)
    return "\n".join(f"- ({tag}) {b['text']}" for b in bullets)


def _split_field(tag, text, max_len=80):
    """효능효과·용법용량처럼 한 필드에 여러 항목이 들어있는 긴 텍스트를 항목별 bullet로
    나눈다. "1. ...  2. ..."나 "○ ..." 같은 표시가 있으면 그 단위로, 없으면 문장
    단위로 나눈다. 짧으면(그 자체로 한 항목이면) bullet 하나만 둔다."""
    text = " ".join(html.unescape(text or "").split())
    if not text:
        return []
    if len(text) <= max_len:
        return [f"- ({tag}) {text}"]
    parts = re.split(r"\s*(?=(?:\d+\.\s)|(?:\(\d+\)\s)|(?:○\s))", text)
    parts = [p.strip() for p in parts if len(p.strip()) > 1]
    if len(parts) <= 1:
        parts = [s.strip() for s in re.split(r"(?<=[.다])\s+", text) if len(s.strip()) > 1]
    return [f"- ({tag}) {p}" for p in parts] if parts else [f"- ({tag}) {text}"]


_ANSWER_NOTE = (
    "answer 필드를 글자 하나, 줄바꿈이나 라벨 하나도 바꾸지 말고 그대로 출력하세요. "
    "요약하거나, 라벨을 **굵게**로 바꾸거나, 설명을 더 붙이거나 빼지 마세요. 이게 전체 답변입니다."
)


def _assemble_answer(facts, summary, data_source, closing=None):
    """최종 답변 문자열을 Python에서 전부 완성한다. facts/summary를 따로 주고 메인 LLM에게
    조립을 맡기면, 라벨을 **굵게**로 바꾸거나 문단을 빠뜨리거나 근거에 없는 결론("운전은
    가능하다" 등)을 덧붙이는 경우가 있었다. 조립까지 끝낸 문자열 하나를 그대로 출력하게
    하면 이런 흔들림이 없다."""
    closings = [c for c in ([closing] if isinstance(closing, str) else (closing or [])) if c]
    # 전문의약품 안내 자체가 "처방한 의사·약사의 지침을 따르라"는 상담 권유를 이미
    # 포함하므로, 그때는 일반적인 "약사나 의사와 상담하시기 바랍니다."를 또 붙이지 않는다.
    tail = [] if any("처방 받은 지침을 따르세요" in c for c in closings) else ["약사나 의사와 상담하시기 바랍니다."]
    tail += closings
    tail_text = " ".join(tail)

    # [조회된 사실]의 bullet은 항목별로 분리된 근거라 그대로 두지만, 그 아래 정리·
    # 상담 권유·안내 문구는 전부 결론에 해당하는 말이라 blank line으로 뚝뚝 끊어
    # 보여주지 않고 한 문단으로 자연스럽게 이어 쓴다.
    parts = ["[조회된 사실]", facts, ""]
    if summary:
        parts += ["[정리]", f"{summary} {tail_text}".strip()]
    else:
        parts.append(tail_text)
    parts += ["", f"데이터 출처: {data_source}"]
    return "\n".join(parts)


_PRESCRIPTION_NOTE = (
    "이 약은 의사의 처방이 필요한 전문의약품입니다. 이 안내는 처방한 의사나 약사의 "
    "지침을 대신하지 않으니, 복용 여부와 방법은 처방 받은 지침을 따르세요."
)


def _prescription_closing(item_seq):
    """전문의약품이면 처방 지침이 우선이라는 안내를 덧붙인다. 일반의약품이면 None.
    "먹어도 되나요"는 전문의약품이라면 원래 처방한 의사가 판단할 문제다. 이미 처방받아
    복용 중인 환자가 상호작용 등을 확인하는 건 정당한 용도이므로 답변 자체를 거부하지는
    않지만, 이 안내가 처방을 대신하지 않는다는 것을 매번 분명히 한다."""
    detail = get_product_detail(item_seq)
    if detail and detail.get("spclty_pblc") == "전문의약품":
        return _PRESCRIPTION_NOTE
    return None


_SUMMARY_PROMPT = """아래 조회된 사실을 한 문장으로 짧게 정리하세요. 이것은 요약이지, 다시 말하는 것이 아닙니다.
- 조건마다 가장 중요한 결론 하나만 고르세요. 한 조건에 여러 주의사항이 있어도 전부 나열하지 말고 제일 중요한 것 하나만 쓰세요.
- 수치, 예외 상황, 부연 설명, 세부 조건은 빼세요.
- 조회된 사실에 있는 내용만 쓰세요. 추측이나 새 정보를 더하지 마세요.
- 조건이 여러 개면 조건마다 고른 결론을 이어 쓰되, 전체를 한 문장, 50자 안팎으로 짧게 끝내세요.
- 근거가 없다고 나온 조건은 정리에서 굳이 다시 언급하지 않아도 됩니다.
- 결론의 강도는 조회된 사실 그대로 옮기세요. '안전하다'는 말은 쓰지 마세요.

{facts}
"""


def _render_facts(conditions_out):
    """조건마다 따로 문단을 bullet로 바꾸고, Python이 조건 제목을 붙여 조립한다.
    근거가 있는지 없는지는 Python이 이미 알고 있으므로 그 판단은 LLM에게 다시 맡기지 않는다
    (맡기면 근거가 있어도 '관련 없음'이라고 잘못 쓰는 경우가 있었다). 조건마다 따로 조립하면
    가독성을 위해 조건 제목을 항상 붙일 수 있고, 한 조건의 문단이 다른 조건의 설명에 섞여
    들어가지 않는다."""
    def _render_one(c):
        if not c["evidence"]:
            return f"**{c['condition']}**\n- 이 조건과 관련된 주의사항은 없습니다."
        bullets = _render_bullets(c["evidence"])
        return f"**{c['condition']}**\n{_bullets_to_md(bullets)}"

    with ThreadPoolExecutor(max_workers=len(conditions_out)) as pool:
        parts = list(pool.map(_render_one, conditions_out))
    return "\n\n".join(parts)


def _make_summary(facts):
    resp = _client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": _SUMMARY_PROMPT.format(facts=facts)}],
    )
    return (resp.choices[0].message.content or "").strip()


def _tool_ask_lifestyle_question(product_name, question, field="NB"):
    result = search_product_with_confidence(product_name)
    if not result["confident"]:
        return {
            "resolved": False,
            "candidate_groups": result.get("candidate_groups", []),
            "clarification_question": result.get("clarification_question"),
            "note": "제품명이 정확히 특정되지 않았습니다. clarification_question 문장만 그대로 답변으로 출력하세요. "
                    "'정보를 확인할 수 없다', '제품명이 정확히 특정되지 않았다' 같은 말을 덧붙이지 마세요.",
        }
    item_seq = result["candidates"][0]["item_seq"]
    item_name = result["candidates"][0]["item_name"]
    rx_closing = _prescription_closing(item_seq)
    conditions = extract_conditions(question, CHAT_MODEL)
    if len(conditions) >= 2:
        # 조건마다 따로 근거를 판정한다. 근거가 없는 조건은 그 조건만 없다고 말한다.
        # 조건끼리는 서로 의존하지 않으므로(검색 -> 선별 -> 음식 확인 전부 조건
        # 하나짜리 데이터만 본다) 순서대로 돌리지 않고 동시에 돌려서 기다리는
        # 시간을 조건 개수만큼 늘리지 않는다.
        def _evidence_for(c):
            raw = search_paragraphs(item_seq, field, c, top_k=8)
            paras = _pick_with_food_check(c, raw)
            paras = _attach_body(paras, _get_paragraphs(item_seq, field))
            return c, paras

        with ThreadPoolExecutor(max_workers=len(conditions)) as pool:
            results = list(pool.map(_evidence_for, conditions))

        conditions_out = []
        found = []
        for c, paras in results:
            if paras:
                found.extend(paras)
            conditions_out.append({
                "condition": c,
                "evidence": [p["text"] for p in paras],
            })
        if not found:
            return _no_evidence_result(item_name, closing=rx_closing)
        facts = _render_facts(conditions_out)
        summary = _make_summary(facts)
        return {
            "resolved": True,
            "item_name": item_name,
            "answer": _assemble_answer(facts, summary, "의약품 제품허가정보 기준", closing=rx_closing),
            "conditions": conditions_out,
            "relevant_paragraphs": found,
            "note": _ANSWER_NOTE,
        }

    paragraphs = search_paragraphs(item_seq, field, question, top_k=8)
    if _DURATION_OR_AMOUNT_RE.search(question):
        # 용량·기간은 용법용량(UD) 항목에 있으므로 함께 찾는다.
        ud = search_paragraphs(item_seq, "UD", question, top_k=4)
        seen = {p["text"] for p in ud}
        paragraphs = ud + [p for p in paragraphs if p["text"] not in seen]
    picked = _pick_with_food_check(question, paragraphs)
    picked = _attach_body(picked, _get_paragraphs(item_seq, field))
    if not picked:
        return _no_evidence_result(item_name, closing=rx_closing)
    facts = _bullets_to_md(_render_bullets([p["text"] for p in picked]))
    summary = _make_summary(facts)
    return {
        "resolved": True,
        "item_name": item_name,
        "answer": _assemble_answer(facts, summary, "의약품 제품허가정보 기준", closing=rx_closing),
        "relevant_paragraphs": picked,
        "note": _ANSWER_NOTE,
    }


def _no_evidence_result(item_name, closing=None):
    # 근거가 없으면 LLM이 제각각 말하지 않도록 문장을 코드에서 고정한다.
    sentence = (f"{item_name}의 사용상주의사항에는 질문하신 내용과 관련된 주의사항이 없습니다. "
                "일반적으로 특별한 금기사항은 알려져 있지 않습니다. "
                "다만 개인의 상태에 따라 다를 수 있으니, 불편하시면 약사와 상담하시기 바랍니다.")
    text = f"{sentence} {closing}".strip() if closing else sentence
    return {
        "resolved": True,
        "item_name": item_name,
        "relevant_paragraphs": [],
        "answer": f"{text}\n\n데이터 출처: 의약품 제품허가정보 기준",
        "note": _ANSWER_NOTE,
    }


def _tool_get_overdose_info(product_name):
    result = search_product_with_confidence(product_name)
    if not result["confident"]:
        return {
            "resolved": False,
            "clarification_question": result.get("clarification_question"),
            "note": "제품명이 정확히 특정되지 않았습니다. clarification_question 문장만 그대로 답변으로 출력하세요. "
                    "'정보를 확인할 수 없다', '제품명이 정확히 특정되지 않았다' 같은 말을 덧붙이지 마세요.",
        }
    item_seq = result["candidates"][0]["item_seq"]
    item_name = result["candidates"][0]["item_name"]
    rx_closing = _prescription_closing(item_seq)
    raw_caution = doc_xml_to_text(_get_detail(item_seq).get("NB_DOC_DATA"))
    overdose_sentences = [s.strip() for s in re.split(r"\n|(?<=[.])\s+", raw_caution) if "과량" in s]
    if not overdose_sentences:
        return _no_evidence_result(item_name, closing=rx_closing)
    facts = _plain_bullets(overdose_sentences, "과량투여")
    summary = _make_summary(facts)
    return {
        "resolved": True,
        "item_name": item_name,
        "answer": _assemble_answer(facts, summary, "의약품 제품허가정보 기준", closing=rx_closing),
        "note": _ANSWER_NOTE,
    }


TOOL_DISPATCH = {
    "check_drug_interactions": _tool_check_drug_interactions,
    "get_drug_info": _tool_get_drug_info,
    "get_age_dur_info": _tool_get_age_dur_info,
    "search_by_symptom": _tool_search_by_symptom,
    "ask_lifestyle_question": _tool_ask_lifestyle_question,
    "get_overdose_info": _tool_get_overdose_info,
}


def warmup():
    """지연 로딩되는 데이터(의약품 상세정보, DUR, 병용금기 등)를 미리 불러온다.

    drug_lookup/paragraph_search의 데이터는 전부 "처음 쓰일 때 로딩"하는
    지연 로딩 방식이라, 아무 준비 없이 서비스를 띄우면 그 로딩 비용(실측
    27초 수준)을 맨 처음 질문한 사용자가 고스란히 떠안는다. 서버/앱을
    시작할 때(사용자 요청을 받기 전) 이 함수를 한 번 호출해서, 그 비용을
    "서버 시작 시간"쪽으로 옮긴다 — 서버 시작은 지켜보는 실제 사용자가
    없지만, 첫 질문에 27초가 걸리는 건 사용자가 직접 겪는 문제이기 때문.
    """
    from drug_lookup import _ensure_loaded as _ensure_drug_lookup_loaded
    from paragraph_search import _ensure_detail_loaded as _ensure_paragraph_detail_loaded
    from name_match import _load_index as _ensure_name_index_loaded

    _ensure_drug_lookup_loaded()
    _ensure_paragraph_detail_loaded()
    _ensure_name_index_loaded()


class Conversation:
    """대화 기록을 유지하는 멀티턴 세션.

    "약 먹어도 되나요?"처럼 되묻기(현재 다른 약 복용 중인지)가 필요한 질문은
    한 번의 호출로 끝나지 않으므로, 이전 턴의 messages를 계속 이어붙여야
    LLM이 되물은 질문에 대한 사용자의 답을 원래 맥락과 연결해서 이해할 수 있다.
    """

    def __init__(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def send(self, user_message: str, max_tool_rounds: int = 5) -> str:
        emergency = detect_emergency(user_message)
        if emergency:
            return emergency["message"]

        self.messages.append({"role": "user", "content": user_message})
        last_lifestyle_result = None
        last_tool_name = None
        last_tool_result = None

        overdose_intent = any(k in user_message for k in _OVERDOSE_KEYWORDS)
        overdose_forced = False

        for _ in range(max_tool_rounds):
            tool_choice = "auto"
            if overdose_intent and not overdose_forced:
                tool_choice = {"type": "function", "function": {"name": "get_overdose_info"}}
                overdose_forced = True
            response = _client.chat.completions.create(
                model=CHAT_MODEL,
                messages=self.messages,
                tools=TOOLS,
                tool_choice=tool_choice,
                temperature=0,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                content = msg.content
                self.messages.append({"role": "assistant", "content": content})
                if SCORE_DEBUG:
                    cited_nums = _detect_citations(user_message, last_lifestyle_result, content)
                    content += _build_score_debug_block(last_tool_name, last_tool_result, last_lifestyle_result, cited_nums, content)
                    grades = _auto_grade(user_message, last_tool_name, last_tool_result, content)
                    content += _build_grading_block(grades)
                return content

            self.messages.append(msg)
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                if fn_name == "check_drug_interactions" and len(fn_args.get("product_names", [])) < 2:
                    # 병용 확인은 제품이 2개 이상일 때만 의미가 있다. 1개뿐이면 질병·조건에 대한
                    # 질문이므로 생활질문 도구로 바꿔서 원래 질문 전체를 넘긴다.
                    fn_name = "ask_lifestyle_question"
                    fn_args = {"product_name": fn_args["product_names"][0], "question": user_message}
                if fn_name == "check_drug_interactions":
                    # 병용확인으로 넘어온 이름 중 음식·음료가 있으면 병용확인이 아니라 생활질문이다.
                    names = fn_args.get("product_names", [])
                    kinds = classify_names(names, CHAT_MODEL) if names else {}
                    foods = [n for n in names if kinds.get(n) == "food"]
                    if foods:
                        drugs = [n for n in names if n not in foods]
                        fn_name = "ask_lifestyle_question"
                        fn_args = {"product_name": (drugs or names)[0], "question": user_message}
                if fn_name == "ask_lifestyle_question" and _dur_type_for_question(user_message):
                    # 임부금기/서방정분할주의/첨가제주의 같은 DUR 유형 질문은 생활질문의 NB
                    # 원문 검색보다 DUR 항목으로 답하는 게 더 짧고 정확하다. "며칠"/"몇 알"처럼
                    # 기간·용량 질문과 겹치는 경우는 바로 아래에서 다시 생활질문으로 되돌아간다
                    # (그쪽이 NB·UD 원문에서 더 구체적인 답을 찾는다).
                    fn_name = "get_drug_info"
                    fn_args = {"product_name": fn_args["product_name"], "question": user_message}
                if fn_name == "get_drug_info" and (
                    len(extract_conditions(user_message, CHAT_MODEL)) >= 2
                    or _DURATION_OR_AMOUNT_RE.search(user_message)
                ):
                    # 조건이 두 개 이상이면 DUR 전문 조회로는 조건별 근거를 보장할 수 없다.
                    # 생활질문 경로에서 조건마다 따로 검색한다.
                    fn_name = "ask_lifestyle_question"
                    fn_args = {"product_name": fn_args["product_name"], "question": user_message}
                if fn_name == "get_drug_info" and _age_group(user_message):
                    # 나이 질문은 연령 DUR 항목 원문으로 답한다. 전체 조회 결과를 주면 원문에 없는 결론이 섞인다.
                    fn_name = "get_age_dur_info"
                    fn_args = {"product_name": fn_args["product_name"], "question": user_message}
                if fn_name == "get_drug_info":
                    # 질문 문맥에 따라 사용상주의사항 중 관련 문단과 DUR 유형을 고르려면 원래
                    # 질문이 필요하다. 모델이 실제로 넘긴 건 product_name뿐이라 직접 채운다.
                    fn_args = {"product_name": fn_args["product_name"], "question": user_message}
                fn = TOOL_DISPATCH.get(fn_name)
                result = fn(**fn_args) if fn else {"error": f"unknown tool {fn_name}"}
                if fn_name == "ask_lifestyle_question":
                    last_lifestyle_result = result
                last_tool_name = fn_name
                last_tool_result = result
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

            if len(msg.tool_calls) == 1 and isinstance(last_tool_result, dict) and "answer" in last_tool_result:
                # 도구가 이미 answer를 완성해 줬다. 한 번 더 모델을 불러 "그대로 출력"을
                # 시키면, 모델이 옮겨 적는 과정에서 줄바꿈이나 라벨이 가끔 흔들렸다.
                # 모델에게 다시 쓰게 하지 않고 Python이 그 문자열로 바로 끝낸다.
                content = last_tool_result["answer"]
                self.messages.append({"role": "assistant", "content": content})
                if SCORE_DEBUG:
                    cited_nums = _detect_citations(user_message, last_lifestyle_result, content)
                    content += _build_score_debug_block(last_tool_name, last_tool_result, last_lifestyle_result, cited_nums, content)
                    grades = _auto_grade(user_message, last_tool_name, last_tool_result, content)
                    content += _build_grading_block(grades)
                return content

        return "죄송합니다, 답변을 생성하는 데 문제가 발생했습니다."


def chat(user_message: str, max_tool_rounds: int = 5) -> str:
    """대화 기록이 필요 없는 단발성 질문용 편의 함수. 멀티턴 대화는 Conversation을 쓴다."""
    return Conversation().send(user_message, max_tool_rounds=max_tool_rounds)
