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
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from emergency_detect import detect_emergency
from drug_lookup import check_multiple_drugs, search_product_with_confidence, get_product_detail, _get_detail, _strip_brackets
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
    줄바꿈이나 라벨 하나도 바꾸지 말고 통째로 그대로 출력하세요 — 정리(요약)/
    [조회된 사실]/약사 상담/데이터 출처까지 이미 다 조립되어 있습니다. 요약하거나, 라벨을
    **굵게**나 다른 말로 바꾸거나, 문장을 합치거나 순서를 바꾸거나, 근거에 없는
    말을 덧붙이지 마세요.
11. 사용자가 과다복용·과량 복용 결과를 물으면(예: "타이레놀 과다복용하면 어떻게 돼?"),
    제품이 특정된 뒤 ask_lifestyle_question(NB)으로 과량투여 관련 원문을 찾아 그대로
    전달하세요. 제품이 특정되지 않았으면 제품부터 되물으세요. 지금 증상이 심하다고
    하면 119 연락을 먼저 안내하고, 그 다음에 원문 내용을 덧붙이세요.
12. 질문이 의약품·복약과 전혀 관련이 없는 "새로운 주제"면(예: 날씨, 시사·정치,
    스포츠, 연예인, 요리법, 코딩, 수학 문제, "너는 누구야"/"넌 AI야?" 같은 신상
    질문, 농담·잡담 요청 등) 어떤 도구도 호출하지 말고, 다른 말을 덧붙이지 말고
    정확히 이 문장만 그대로 답하세요: "복약 관련 정보만 도와드릴 수 있어요."
    이 규칙은 새로운 무관한 주제를 꺼낼 때만 적용되는 것이며, 아래처럼 직전
    대화에 대한 짧은 반응일 때는 절대 적용하지 마세요(이런 경우는 새 주제가
    아니라 방금 나눈 대화의 연속입니다):
    - "고마워", "감사합니다", "네 알겠습니다", "오 신기하네" 같은 인사·감사·
      맞장구: 가볍게 자연스러운 한 문장으로 응대하세요(도구 호출 불필요).
    - "그게 무슨 뜻이야?", "왜 그런거야?" 처럼 직전 답변 내용을 더 설명해
      달라는 질문: 이미 가진 정보로 자연스럽게 설명하세요.
    - 증상·질병·신체 상태·의학 용어의 뜻을 묻는 질문은 전부 예외입니다. 예:
      "현훈이 뭐야", "오한이 뭐야", "구안와사가 뭐야", "역류성식도염이 뭐야",
      "돌발성 난청이 뭐야" — 이런 용어는 아무리 낯설어도, 또 목록에 없는
      이름이어도 전부 신체 증상·질병이면 이 예외에 해당합니다(목록은 예시일
      뿐 전체가 아닙니다). 식약처 제품 데이터로 답할 질문이 아니라 사전적
      설명이므로 이 규칙 대상이 아니니, 아는 대로 짧게 설명하세요.
      (이 예외는 "이 용어가 뜻하는 게 뭐냐"는 질문에만 적용되고, 그 증상에
      먹을 약을 추천해달라는 질문은 여전히 위 도구들을 정상적으로 쓰세요.)
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
def _name_with_ingredient(drug_name, ingredient):
    # 제품명에 이미 성분이 들어있는 경우(예: "코니트라캡슐(이트라코나졸)(수출용)")
    # "(이트라코나졸)"을 또 붙이면 "...(이트라코나졸)(수출용)(이트라코나졸)"처럼
    # 중복 표기된다(실측). 이미 포함돼 있으면 괄호를 또 붙이지 않는다.
    if ingredient and ingredient in drug_name:
        return drug_name
    return f"{drug_name}({ingredient})" if ingredient else drug_name


def _interaction_facts(result):
    """병용금기·성분 중복·효능군 중복을 bullet로 나열한다. 어떤 종류인지 라벨을 붙여서
    한 줄로 뭉쳐 보여주던 것보다 어떤 조합이 왜 문제인지 구분해서 볼 수 있게 한다."""
    bullets = []
    for h in result.get("dur_taboo_matches", []):
        a = _name_with_ingredient(h['drug_a'], h['ingredient_a'])
        b = _name_with_ingredient(h['drug_b'], h['ingredient_b'])
        bullets.append(f"- (병용금기) {a}과 {b}: {h['reason']}")
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
    # 결과라 요약을 다시 판단할 필요가 없고, 매번 같은 결론이 나와야 한다). 다만
    # "등록된 병용금기가 있으므로 함께 복용해서는 안 됩니다"처럼 약 이름도 이유도 없는
    # 문장은 너무 짧아서 의도한 정리문처럼 안 느껴진다(실측 피드백) — 구조화된 결과에
    # 이미 약 이름과 이유가 있으므로 그대로 문장에 넣는다. 조합이 여러 개면(어느 걸
    # 대표로 보여줄지 애매하므로) 기존처럼 일반적인 문장으로 되돌아간다.
    taboo = result.get("dur_taboo_matches") or []
    if len(taboo) == 1:
        h = taboo[0]
        a = _name_with_ingredient(h['drug_a'], h['ingredient_a'])
        b = _name_with_ingredient(h['drug_b'], h['ingredient_b'])
        return f"{a}과 {b}는 함께 복용하면 {h['reason']} 위험이 있는 병용금기 조합으로 등록되어 있어, 함께 복용해서는 안 됩니다."
    if taboo:
        return "등록된 병용금기 조합이 여러 건 있으므로 함께 복용해서는 안 됩니다."
    dup = result.get("duplicate_ingredients") or []
    if len(dup) == 1:
        d = dup[0]
        return f"{', '.join(d['drugs'])}에 공통으로 {d['ingredient']} 성분이 들어 있어, 함께 복용하면 과다복용 위험이 있습니다."
    if dup:
        return "같은 성분이 중복되어 과다복용 위험이 있습니다."
    eff = result.get("effect_group_duplicates") or []
    if len(eff) == 1:
        e = eff[0]
        return f"{e['drug_a']}과 {e['drug_b']}는 성분은 다르지만 둘 다 {e['effect_group']}에 속해 효과가 중복될 수 있습니다."
    if eff:
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
        overdose_sentences = _extract_overdose_blocks(raw_caution)
        facts = "\n".join(
            _split_field("용법용량", detail.get("용법용량") or "")
            + ([_plain_bullets(overdose_sentences, "과량투여")] if overdose_sentences else [])
        )
        summary = "같은 제품을 중복해서 드시는 것이므로, 병용이 아니라 과다복용에 주의해야 합니다."
        rx_closing = _PRESCRIPTION_NOTE if detail.get("spclty_pblc") == "전문의약품" else None
        answer, footer = _assemble_answer(facts, summary, "의약품 제품허가정보 기준", closing=rx_closing, item_name=detail.get("item_name"))
        return {
            "same_product_repeated": True,
            "item_name": detail.get("item_name"),
            "answer": answer,
            "footer": footer,
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
    interaction_item_names = [r["item_name"] for r in result.get("resolved_drugs", []) if r.get("item_name")]
    result["answer"], result["footer"] = _assemble_answer(
        facts, summary, "의약품안전사용서비스 기준", closing=rx_closing, item_name=interaction_item_names
    )
    result["note"] = _ANSWER_NOTE
    return result


_DUR_TYPE_KEYWORDS = {
    "임부금기": ("임신", "임산부", "임부", "수유"),
    # "오랫동안"은 "오래"와 둘째 글자가 다르다(오+래 "동안"이 사이시옷으로 "오랫동안"이
    # 되면서 "래"가 "랫"으로 바뀜) — 그래서 "오래"만 있으면 "오랫동안 먹어도 되나요?"가
    # 매칭 안 됐다(실측: 맥페란정, 전체 덤프로 샘). "장기"도 원문 데이터 표현("장기
    # 복용", "장기간 복용")과 맞춰 추가한다.
    "투여기간주의": ("며칠", "몇일", "얼마나 오래", "오래", "오랫동안", "장기", "기간"),
    "서방정분할주의": ("쪼개", "나눠", "분할", "갈아"),
    "용량주의": ("하루 몇", "몇 알", "몇 정", "최대 용량", "용량"),
    "첨가제주의": ("첨가제", "부형제"),
}

# DUR 유형을 콕 집은 질문인데 구조화 DUR도, NB 원문도 아무것도 못 찾았을 때, "근거
# 없음" 안내에 어떤 주제를 못 찾았는지 자연어로 넣는다. "질문하신 내용"으로 뭉뚱그리지
# 않고 "임부·수유부 관련 주의사항이 없습니다"처럼 구체적으로 말하는 쪽이 더 친절하다.
_DUR_TYPE_TOPIC = {
    "임부금기": "임부·수유부",
    "투여기간주의": "복용 기간",
    "서방정분할주의": "분할 복용",
    "용량주의": "용량",
    "첨가제주의": "첨가제",
}


def _dur_type_for_question(question):
    """질문이 7개 DUR 유형 중 하나를 콕 집어 물었으면 그 유형만, 아니면 None(전체 표시)."""
    for dtype, keywords in _DUR_TYPE_KEYWORDS.items():
        if any(k in question for k in keywords):
            return dtype
    return None


_INFO_SCOPE_KEYWORDS = {
    "ingredients": ("성분", "주성분", "원료"),
    # "증상"을 추가한다 — "어떤 증상에 먹는 약이야?"처럼 "효능"/"효과"라는 단어 없이
    # 증상을 묻는 질문도 결국 효능효과 하나만 궁금한 것인데, 이 키워드가 없으면
    # scope가 None이 되어 성분·용법용량·DUR까지 전부 나온다(실측).
    "efficacy": ("효능", "효과", "증상에 먹는", "증상에 쓰", "무슨 병", "어디에 쓰"),
    "usage": ("용법", "용량", "복용법", "몇 알", "몇 정", "어떻게 먹", "먹는 법"),
    # "중지"는 빼둔다 — "복용 중지해야 하나요?"처럼 약을 그만 먹는다는 뜻으로도 많이
    # 쓰여서, "판매 중지" 의도가 아닌데도 이 범위로 잘못 좁혀질 위험이 있다. "판매"
    # 자체가 "판매중지"/"판매 중지"/"판매되고 있어" 전부를 이미 포함하므로 충분하다.
    "sale_status": ("판매", "취하", "회수", "구할 수", "살 수 있"),
}


def _msg_role(m):
    return m.get("role") if isinstance(m, dict) else getattr(m, "role", None)


def _msg_content(m):
    content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
    return content or ""


def _is_clarification_reply(messages):
    """messages의 마지막이 사용자 메시지이고, 그 직전 assistant 턴이 제품을
    특정해달라는 되묻기("...말씀하시는 건가요?")였으면 True. _effective_question과
    Conversation.send()의 브랜드명 고정(핀) 로직이 같은 기준을 쓴다."""
    return (
        len(messages) >= 2
        and _msg_role(messages[-2]) == "assistant"
        and "말씀하시는 건가요?" in _msg_content(messages[-2])
    )


def _effective_question(messages, current_message):
    """제품을 특정해달라는 되묻기("...말씀하시는 건가요?")에 사용자가 제품명만 답하면,
    현재 메시지에는 원래 질문("성분이 뭐야" 등)의 의도가 전혀 남지 않는다. 직전
    assistant 턴이 되묻기였으면 그 이전 사용자 질문을 이어받아 현재 메시지와 합쳐서
    돌려준다. messages에는 이미 현재 사용자 메시지가 append된 상태로 들어오고, 그
    앞에는 도구 호출 라운드(assistant tool_call -> tool 결과)가 끼어 있을 수 있어서
    바로 앞이 아니라 역순으로 가장 가까운 user 메시지를 찾는다."""
    if _is_clarification_reply(messages):
        for m in reversed(messages[:-2]):
            if _msg_role(m) == "user":
                prior_text = _msg_content(m)
                if prior_text:
                    return f"{prior_text} {current_message}"
                break
    return current_message


def _info_scope(question):
    """질문이 성분·효능효과·용법용량 중 하나만 콕 집어 물었으면 그 범위만 보여주고,
    범위를 특정하지 않았거나("먹어도 되나요" 등 안전을 묻는 질문) 범위 키워드가 없으면
    None(전체 표시)을 돌려준다. "성분이 뭐야"처럼 단순 정보 질문에 경고·DUR까지 전부
    보여주면 묻지 않은 걸 끼워 넣는 셈이라 불필요하게 느껴진다."""
    for scope, keywords in _INFO_SCOPE_KEYWORDS.items():
        if any(k in question for k in keywords):
            return scope
    return None


def _build_drug_info_answer(item_seq, detail, question, data_source):
    """get_drug_info 답변을 전부 Python에서 조립한다. 성분/효능/용법용량은 이미 깨끗한
    필드라 그대로 bullet로 쓰고, 사용상주의사항은 질문과 관련된 문단만 찾아 조항 이름을
    붙인다. dur_cautions는 질문이 특정 유형을 물었으면 그 유형만, 아니면 전체를 보여준다
    (어떤 유형을 보여줄지 LLM이 매번 다시 판단하게 하면 흔들린다)."""
    scope = _info_scope(question)
    # DUR 유형(임부금기/노인주의/용량주의 등)을 콕 집어 물은 질문도 "성분이 뭐야"처럼
    # 범위를 좁혀 물은 것과 같다. dtype이 있으면 scope가 None이어도(= "성분"/"효능"/
    # "용법" 키워드가 없어도) 성분·효능효과·용법용량이나 사용상주의사항 전문 검색까지
    # 같이 보여주면 묻지 않은 걸 끼워 넣는 셈이다(실측: "임산부인데 먹어도 되나요?"에
    # 성분·효능효과·용법용량·전체 경고까지 다 나와서 "임부금기" 외의 내용이 끼어들었다).
    dtype = _dur_type_for_question(question)
    basic = []
    if detail.get("cancel_name") and detail["cancel_name"] != "정상":
        # 판매 중지 여부는 뭘 물었든 항상 먼저 알린다(안전과 직결).
        basic.append(
            f"- (판매 상태) 이 제품은 {detail.get('cancel_date')}자로 {detail['cancel_name']} 처리되어 "
            "더 이상 정상적으로 판매되지 않습니다. 복용 중이라면 약사나 의사와 상담하세요."
        )
    elif scope == "sale_status":
        # "판매 중지됐어?"처럼 판매 상태를 콕 집어 물었는데 정상 판매 중이면, 범위를
        # 좁혀 물은 것과 같으니 그 사실만 답하고 아래 성분/효능효과/용법용량/주의사항
        # 전부는 보여주지 않는다(실측: 판매 상태만 물었는데 혈압약 용법용량·DUR
        # 임부금기까지 전부 나왔다). 정상이 아니면 위 분기에서 이미 처리했으므로
        # 이 분기는 "정상인데 콕 집어 물은 경우"만 해당한다.
        basic.append("- (판매 상태) 이 제품은 정상적으로 판매되고 있습니다.")
    if scope in (None, "ingredients") and not dtype and detail.get("ingredients"):
        basic.append(f"- (성분) {', '.join(detail['ingredients'])}")
    if scope in (None, "ingredients", "efficacy") and not dtype:
        basic += _split_field("효능효과", detail.get("효능효과") or "")
    if scope in (None, "usage") and not dtype:
        basic += _split_field("용법용량", detail.get("용법용량") or "")

    # 사용상주의사항 원문 검색과 DUR 전체 목록은 "먹어도 되나요"처럼 범위를 특정하지
    # 않은 안전 질문에만 보여준다. "성분이 뭐야"처럼 단순 정보 질문에 경고·DUR까지
    # 끼워 넣으면 묻지 않은 걸 같이 알려주는 셈이라 불필요하게 느껴진다. 다만 질문이
    # 특정 DUR 유형(예: 용량주의)을 콕 집었으면 범위를 좁혀 물은 것과 같으니 보여준다.
    caution_md = ""
    dur_lines = []
    if scope is None or dtype:
        if scope is None:
            paragraphs = search_paragraphs(item_seq, "NB", question or detail.get("item_name") or "", top_k=5)
            if dtype:
                # DUR 유형을 콕 집은 질문이면 top-5 중 그 유형과 무관한 다른 경고까지
                # 전부 보여주면 묻지 않은 걸 끼워 넣는 셈이다(실측: "임산부인데
                # 먹어도 되나요?"에 간장애·글루타치온·최대용량 경고까지 다 나왔다).
                # 하지만 아예 다 빼버리면 구조화 DUR 데이터에 없는 임신 관련 문구가
                # NB 원문에만 있는 경우(실측: 타이레놀 임부금기 DUR은 미등록이지만
                # NB에는 "임부 또는 수유부는... 상의할 것"이 있음) 그 유일한 근거까지
                # 사라진다. ask_lifestyle_question과 같은 방식으로 실제 관련 있는
                # 문단만 추려서 보여준다.
                paragraphs = _pick_with_relevance_check(question, paragraphs)
            caution_md = _bullets_to_md(_render_bullets([p["text"] for p in paragraphs])) if paragraphs else ""
        dur_cautions = detail.get("dur_cautions", [])
        relevant = [d for d in dur_cautions if d.get("type") == dtype] if dtype else dur_cautions
        dur_lines = [
            f"- (DUR: {d['type']}) " + (
                d.get("content") or f"{d['type']}에 대한 주의가 필요한 성분/품목으로 등록되어 있으나 구체적인 문구는 없습니다."
            )
            for d in relevant
        ]
        if dtype and not relevant and not caution_md:
            # 구조화된 DUR 데이터도 없고 NB 원문 검색(caution_md)도 못 찾았으면, 그
            # "(DUR: 유형) 등록된 정보가 없습니다." 한 줄짜리 facts를 그대로
            # _make_summary에 넘기면 "~에 따라, 등록된 정보가 없습니다" 같은 어색한
            # 문장이 나온다. 병용금기 없음/노인·소아 근거 없음과 똑같이, 이 경우도
            # Python이 고정 문장으로 바로 답한다 — 주제를 구체적으로 밝혀서
            # "질문하신 내용"으로 뭉뚱그리지 않는다.
            no_evidence = _no_evidence_result(
                detail.get("item_name"),
                closing=_PRESCRIPTION_NOTE if detail.get("spclty_pblc") == "전문의약품" else None,
                topic=_DUR_TYPE_TOPIC.get(dtype, dtype),
                data_source=data_source,
            )
            return no_evidence["answer"], no_evidence["footer"]

    facts = "\n".join(filter(None, ["\n".join(basic), caution_md, "\n".join(dur_lines)])) or "등록된 정보가 없습니다."
    summary = _make_summary(facts, item_name=detail.get("item_name"), data_source=data_source)
    closings = []
    if scope is None and not dtype and detail.get("has_interaction_data"):
        closings.append(
            "이 약은 특정 약과 함께 복용 시 주의가 필요한 조합이 등록되어 있습니다. "
            "현재 복용 중인 다른 약이 있으면 알려주시면 추가로 확인해드리겠습니다."
        )
    if detail.get("spclty_pblc") == "전문의약품":
        closings.append(_PRESCRIPTION_NOTE)
    return _assemble_answer(facts, summary, data_source, closing=closings, item_name=detail.get("item_name"))


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
    if not build_answer:
        # get_age_dur_info처럼 dur_cautions 등 원본 필드만 필요한 내부 호출.
        # 매번 답변 문장까지 조립하면 search_paragraphs·LLM 호출이 낭비된다.
        return detail
    # _build_drug_info_answer는 "판매 중지됐어?"처럼 판매 상태를 콕 집어 물은
    # 경우 cancel_name이 "정상"이어도 그 사실을 답에 넣어야 하므로, 아직 지우지
    # 않은 상태로 넘긴다. answer를 만든 뒤에야(cancel_name이 "정상"이면) 지운다 —
    # 프롬프트로 "정상이면 언급하지 마라"고 아무리 명시해도 LLM이 "이 제품은
    # 정상적으로 판매되고 있습니다"를 다른 질문에까지 계속 붙이는 걸 반복
    # 확인함(이 shortcut이 꺼져서 두 번째 LLM 라운드가 원본 JSON을 다시 보는
    # 경우). 볼 수 없는 값은 언급할 수 없으니 answer 조립이 끝나면 지운다.
    detail["answer"], detail["footer"] = _build_drug_info_answer(item_seq, detail, question, detail["data_source"])
    if detail.get("cancel_name") == "정상":
        detail.pop("cancel_name", None)
        detail.pop("cancel_date", None)
    detail["note"] = _ANSWER_NOTE
    return detail


def _tool_get_age_dur_info(product_name, question):
    detail = _tool_get_drug_info(product_name, build_answer=False)
    if not detail.get("resolved"):
        return detail
    group = _age_group(_strip_product_name(question, product_name))
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
    answer, footer = _assemble_answer(facts, None, detail["data_source"], closing=rx_closing, item_name=detail.get("item_name"))
    return {
        "resolved": True,
        "item_name": detail["item_name"],
        "answer": answer,
        "footer": footer,
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
            "answer": f"'{symptom}'에 해당하는 일반의약품 성분을 찾지 못했습니다.",
            "footer": f"약사나 의사와 상담하시기 바랍니다.\n\n{_data_source_line('의약품 제품허가정보 기준')}",
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

    answer, footer = _assemble_answer(facts, summary, "의약품 제품허가정보 기준")
    return {
        "resolved": True,
        "candidates": results,
        "answer": answer,
        "footer": footer,
        "note": _ANSWER_NOTE,
    }


_RELEVANCE_FILTER_PROMPT = """복용자 상황: "{condition}"

이 상황이 구체적으로 묻는 대상이 무엇인지 생각하세요(예: 운전, 콩팥, 커피, 자몽주스, 나이 등).
아래 문단 중에서 그 대상을 주된 내용으로 다루는 문단의 번호만 쉼표로 적으세요.
같은 뜻으로 쓰인 말도 인정합니다. 예: 콩팥=신장=신기능, 술=알코올, 커피=카페인.
다른 약물이나 물질(알코올, 다른 약 등)과의 상호작용을 다루는 문단은, 상황이 그 약물·물질 자체를 묻는 게 아니라면 제외하세요.
예: "운전해도 되나요"에는 운전·기계 사용 자체를 다루는 문단만 포함합니다. 알코올과 함께 복용했을 때 혈압이나 어지럼증이 변한다는 문단은, 그 문단의 진짜 주제가 알코올과의 상호작용이므로, 운전 상황에는 포함하지 않습니다(알코올에 대해 물을 때만 포함합니다).
약의 성분, 부형제, 수유 중 이행 같은 내용도 음식·음료 자체를 다루는 것이 아니므로 제외합니다.
해당하는 문단이 없으면 "없음"이라고만 답하세요.

[문단]
{paragraphs}
"""


def _relevance_filter(condition, paragraphs):
    """문단 목록 중 조건이 실제로 묻는 대상을 주된 내용으로 다루는 것만 골라
    인덱스로 돌려준다. _pick_for_condition이 상황과 다른 대상을 다루는 문단을
    잘못 고르는 경우가 반복적으로 재현됐다(예: "운전" 조건에 음주 문단, "고추"
    조건에 음주 문단). 처음에는 "그 대상을 다루는 문단이 있는가"를 묶음으로
    물었는데, 맞는 문단과 틀린 문단이 같은 묶음에 있으면 묶음 전체가 YES로
    통과해서 틀린 문단이 그대로 섞여 나왔다. 문단마다 번호로 다시 골라야 이
    문제가 없다."""
    if not paragraphs:
        return []
    text = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(paragraphs))
    resp = _client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": _RELEVANCE_FILTER_PROMPT.format(condition=condition, paragraphs=text)}],
    )
    numbers = {int(n) for n in re.findall(r"\d+", resp.choices[0].message.content or "")}
    return [i for i in range(len(paragraphs)) if (i + 1) in numbers]




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
    """조건에 직접 해당하는 문단만 고른다. 고른 문단은 원문 그대로 쓴다.
    순서는 임베딩 검색 점수(raw 순서)가 아니라, 프롬프트가 요청한 "조건 없이 그대로
    맞아떨어지는 문단 먼저" 순서를 따른다 — 임베딩 유사도는 문장이 길고 키워드가
    많이 겹치면 점수가 높아지는 경향이 있어서, "매일 세 잔 이상 마시는 사람" 같은
    좁은 조건문이 "알코올을 복용한 사람" 같은 더 직접적인 문구보다 먼저 나오는
    경우가 있었다. 사용자가 실제로 물은 대상에 조건 없이 바로 해당하는 근거를
    먼저 보여줘야 자연스럽다."""
    if not paragraphs:
        return []
    text = "\n".join(f"{i + 1}. {p['text']}" for i, p in enumerate(paragraphs))
    resp = _client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": _CONDITION_PICK_PROMPT.format(condition=condition, paragraphs=text)}],
    )
    order = [int(n) for n in re.findall(r"\d+", resp.choices[0].message.content or "")]
    seen = set()
    picked = []
    for n in order:
        if n not in seen and 1 <= n <= len(paragraphs):
            seen.add(n)
            picked.append(paragraphs[n - 1])
    return picked


def _pick_with_relevance_check(condition, raw):
    """조건에 맞는 문단을 고르고, 고른 문단이 조건이 실제로 묻는 대상을 다루는지
    다시 확인한다. _pick_for_condition이 상황과 다른 대상을 다루는 문단을 잘못
    고르는 경우가 재현됐는데(예: "운전" 조건에 음주 문단, "고추" 조건에 음주
    문단), 한 번 판정을 통과하지 못하면 걸러낼 길이 없었다(이전에는 마지막 LLM
    응답 단계에서 가끔 걸러졌지만, 그 단계를 없앤 뒤로는 그대로 새어 나갔다).
    고른 게 없을 때는 상위 문단 중에 실제로 대상을 다루는 게 있는지 봐서,
    _pick_for_condition이 너무 엄격했던 경우를 보완한다."""
    picked = _pick_for_condition(condition, raw)
    if picked:
        # 이미 고른 문단 전체를 다시 확인한다(일부만 보면 뒤쪽에 있는 정상적인
        # 문단을 잘라낼 수 있다).
        keep = _relevance_filter(condition, [p["text"] for p in picked])
        return _order_by_directness(condition, [picked[i] for i in keep])
    if not raw:
        return []
    # 고른 게 없으면 상위 몇 개만 느슨하게 다시 본다(_pick_for_condition이
    # 너무 엄격했던 경우를 보완하는 마지막 fallback이라 비용을 제한한다).
    check_source = raw[:3]
    keep = _relevance_filter(condition, [p["text"] for p in check_source])
    return _order_by_directness(condition, [check_source[i] for i in keep][:2])


_DIRECTNESS_ORDER_PROMPT = """아래 문단들은 모두 사용자의 상황 "{condition}"에 해당한다고 이미
확인됐습니다. 문단마다 이 상황에 조건 없이 그대로 직접 해당하는지, 아니면 양·빈도·정도 같은
특정 조건이 붙어야 해당하는 더 좁은 경우인지 판단하세요.

{paragraphs}

조건 없이 직접 해당하는 문단 번호부터, 조건이 붙은 문단 번호를 그 다음에 쉼표로, 전부
빠짐없이 답하세요(예: "2, 1").
"""


def _order_by_directness(condition, paragraphs):
    """조건에 맞는다고 이미 고른 문단들을 "조건 없이 그대로 해당" -> "조건부로만 해당"
    순서로 재배열한다. _pick_for_condition에 선별과 순서 지정을 한꺼번에 시키면(같은
    프롬프트에 순서 기준을 추가해도) 결국 입력 순서를 그대로 베끼는 경우가 실측으로
    반복됐다(예: 타이레놀 "술" 조건에서 "매일 세 잔 이상 마시는 사람"(조건부)이 "알코올을
    복용한 사람"(조건 없음)보다 임베딩 점수가 낮아 먼저 입력됐는데, 선별+순서를 같이
    시키면 늘 입력 순서 그대로 나왔다). 순서 재배열만 따로 물어도, "조건 없는 문단을
    먼저 써라"는 규칙만 추상적으로 주면(8/8 실패) 여전히 입력 순서를 베꼈다. 문단마다
    "조건 없이 직접 해당하는가, 조건이 붙는가"를 먼저 판단하게 시키니(비교·판단을
    명시적으로 요구) 안정적으로 맞혔다(조건 문구가 짧은 명사("술")든 긴 질문문
    ("술을 마셔도 되나요?")이든 8/8). 문단이 1개 이하면 재배열할 필요가 없다."""
    if len(paragraphs) < 2:
        return paragraphs
    text = "\n".join(f"문단{i + 1}: {p['text']}" for i, p in enumerate(paragraphs))
    resp = _client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": _DIRECTNESS_ORDER_PROMPT.format(condition=condition, paragraphs=text)}],
    )
    order = [int(n) for n in re.findall(r"\d+", resp.choices[0].message.content or "")]
    seen = set()
    result = []
    for n in order:
        if n not in seen and 1 <= n <= len(paragraphs):
            seen.add(n)
            result.append(paragraphs[n - 1])
    result += [p for i, p in enumerate(paragraphs, 1) if i not in seen]
    return result


def _attach_body(picked, all_paragraphs, condition=None):
    """짧은 제목 문단(예: "[11. 기타] ② 신기능 부전증")이 골라지면, 바로 뒤의 본문 문단을 함께 붙인다.
    본문은 같은 장 표시를 가지고 있고, 다음 소제목(짧은 문단)을 만나면 멈춘다.

    다만 짧은 문단이 "제목+본문이 분리된 경우"만 있는 게 아니라, "9) 고령자(노인)",
    "10) 임부 또는 수유부", "11) 와파린... 복용 환자"처럼 같은 조항 번호 목록 안에서
    서로 다른 대상을 나열한 항목인 경우도 있다(실측: "80세 노인인데 먹어도 되나요?"에
    바로 뒤에 붙은 "임부 또는 수유부", "와파린 복용 환자" 항목까지 같이 끼어들어서
    묻지도 않은 내용이 나왔다). 이런 항목은 길이·같은 라벨이라는 조건만으로는 진짜
    본문 이어붙이기와 구분이 안 되므로, condition을 주면 새로 붙이려는 문단이 실제로
    그 조건과 관련 있는지 다시 확인하고 관련 없으면 붙이지 않는다. 원래 picked에
    있던 문단은(이미 한 번 관련성 확인을 거쳤으므로) 다시 거르지 않는다."""
    texts = list(all_paragraphs)
    result = []
    seen = set()
    extra = []
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
                entry = {"text": nxt, "score": p["score"], "condition": p.get("condition")}
                result.append(entry)
                extra.append(entry)
    if condition and extra:
        keep_idx = _relevance_filter(condition, [e["text"] for e in extra])
        keep_texts = {extra[i]["text"] for i in keep_idx}
        extra_texts = {e["text"] for e in extra}
        result = [r for r in result if r["text"] not in extra_texts or r["text"] in keep_texts]
    return result


_DURATION_OR_AMOUNT_RE = re.compile(r"(\d+|한|두|세|네|다섯)\s*(일|주|개월|달|알|정|회|번|시간)|넘게|하루|며칠|몇\s*(알|정|번|회|일)")

# 나이 질문은 DUR 항목의 원문을 코드에서 그대로 옮긴다. 사용상주의사항 전체를 LLM에 넘기면
# 여러 연령 문장이 섞여 "사용할 수 있다"처럼 원문에 없는 결론이 나왔다.
_AGE_DUR_TYPES = {"elderly": ("노인주의",), "child": ("특정연령대금기",)}


def _strip_product_name(text, product_name):
    """나이 등 키워드 판단 전에 질문에서 제품명을 떼어낸다. question에는 제품명이
    그대로 포함돼 있는데(사용자가 "OO정 ~"처럼 제품명과 질문을 같이 치므로),
    제품명 자체에 우연히 판단 키워드와 같은 글자가 들어있으면(실측: "티엘트레노인
    20연질캡슐"의 "트레노인" 안에 "노인"이 부분 문자열로 들어있어 나이 질문이
    아닌데도 "노인" 질문으로 오판했다) 질문 내용이 아니라 제품명 때문에 키워드가
    잘못 발동한다. product_name을 알고 있으면 그 부분을 지우고 나머지만 검사한다."""
    if not product_name:
        return text
    return text.replace(product_name, "")


def _age_group(question):
    if re.search(r"노인|고령|어르신|(?<!\d)(6\d|[7-9]\d)\s*세", question):
        return "elderly"
    if re.search(r"소아|어린이|유아|아이(?:가|는|에게|한테|들|$| )|(?<!\d)1?\d\s*세", question):
        return "child"
    return None


_BULLET_PROMPT = """아래 문단을 번호 순서대로, 번호마다 한 문장씩 이해하기 쉬운 말로 바꾸세요.
문단에 없는 내용은 쓰지 마세요. 결론이나 주의를 말하는 표현(예: 무시해도 좋을 것으로 예상된다, 신중히 투여, 투여하지 말 것, 증강시킬 수 있다)은 원문 표현을 한 글자도 바꾸지 말고 그대로 옮기세요. 그 밖의 설명만 쉬운 말로 바꿀 수 있습니다.
문단이 "제목: 9) 대상"처럼 제목과 원문 목록 번호가 섞인 형태면, "9)" 같은 목록 번호는
지우고 제목과 대상을 자연스러운 한 문장으로 합치세요(예: "다음과 같은 사람은 이 약을 복용하지
말 것: 9) 알코올을 복용한 사람" -> "알코올을 복용한 사람은 이 약을 복용하지 말 것"). 목록
번호를 지우는 것과 결론 표현의 글자를 바꾸는 것은 다릅니다 — 번호만 지우고 결론 표현("복용하지
말 것" 등)은 그대로 쓰세요.
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

    def _body_for_rewrite(text, label):
        # 헤딩이 짧으면(라벨로 따로 보여줄 수 있으면) 본문에서는 그냥 뗀다 — 헤딩은
        # "(출처: ...)" 인용으로 이미 보여준다. 헤딩이 길어서 라벨로 못 보여주면
        # (label is None) 그냥 버리지 않는다 — 헤딩 자체가 지시문인 경우가 있다
        # (예: "[다음과 같은 사람은 이 약을 복용하지 말 것] 9) 알코올을 복용한 사람"처럼
        # 본문은 대상만 나열하고 "복용하지 말 것"이라는 결론은 헤딩에만 있다. 헤딩을
        # 버리면 그 결론이 통째로 사라져서 "알코올을 복용한 사람"처럼 뭘 하지 말라는
        # 건지 빠진 문장이 나왔다). 그래서 헤딩이 길면 "헤딩: 본문" 형태로 합쳐서
        # 다시 쓰기 단계가 지시문을 문장에 녹이게 한다.
        m = re.match(r"^\[([^\]]*)\]\s*(.*)$", text, re.S)
        if not m:
            return text
        heading, body = m.group(1), m.group(2)
        if label is not None:
            return body or heading
        # 헤딩 자체에 "2. 다음과 같은 사람은..."처럼 조항 번호가 붙어 있으면,
        # 바깥에서 또 번호를 붙일 때("1. 2. 다음과 같은...") 숫자가 겹쳐 다시 쓰기
        # LLM이 원문 번호 매기기로 착각해서 손대지 않고 그대로 베끼는 문제가 있었다
        # (실측). 헤딩의 조항 번호는 여기서는 필요 없으니 뗀다.
        heading = re.sub(r"^\d+[.)]\s*", "", heading)
        return f"{heading}: {body}" if body else heading

    bodies = [_body_for_rewrite(t, labels[i]) for i, t in enumerate(evidence_texts)]
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
    # 조항 출처("(출처: 1. 경고)" 등)는 더 이상 문장 앞에 안 붙인다 — 사용자는 어느
    # 조항에서 나왔는지보다 문장 자체를 바로 읽는 게 더 중요하다는 피드백. 출처 정보는
    # 답변 맨 아래 데이터 출처 줄(_data_source_line)에서 제품 단위로 한 번만 보여준다.
    return "\n".join(f"- {b['text']}" for b in bullets)


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


def _file_date(path):
    """파일을 마지막으로 받아온 날짜(수정 시각)를 YYYY-MM-DD로. 식약처 API의 레코드별
    갱신일이 아니라, 이 로컬 스냅샷을 언제 내려받았는지를 뜻한다 — 정적 데이터라
    식약처가 그 뒤 갱신해도 반영되지 않을 수 있다는 걸 사용자가 알 수 있게 한다."""
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")
    except OSError:
        return None


_PRMSN_UPDATED = _file_date("data/drug_prdt_prmsn_list.jsonl")
_DUR_UPDATED = _file_date("data/dur_usjnt_taboo.jsonl")


def _data_source_line(data_source, item_name=None):
    """제품명 + 출처 종류 + 갱신일을 한 줄로 보여준다(예: "타이레놀정500밀리그람
    (의약품안전사용서비스 기준). 갱신일: 2026-09-29"). data_source 문구에
    "제품허가정보"나 "안전사용서비스"가 있으면 그에 맞는 날짜를, 둘 다 없으면
    제품허가정보 날짜를 쓴다. item_name이 리스트/튜플이면(병용 확인처럼 제품이
    여럿인 경우) 쉼표로 묶는다. item_name이 없으면(증상 검색처럼 특정 제품 하나로
    좁혀지지 않는 경우) 예전처럼 "데이터 출처: ..."로만 쓴다."""
    dates = set()
    if "제품허가정보" in data_source and _PRMSN_UPDATED:
        dates.add(_PRMSN_UPDATED)
    if "안전사용서비스" in data_source and _DUR_UPDATED:
        dates.add(_DUR_UPDATED)
    if not dates and _PRMSN_UPDATED:
        dates.add(_PRMSN_UPDATED)
    if isinstance(item_name, (list, tuple)):
        item_name = ", ".join(_strip_brackets(n) for n in item_name if n)
    elif item_name:
        # 원본 ITEM_NAME 필드에 "(아세트아미노펜)" 같은 성분 표기가 그대로 붙어 있으면
        # "제품명(성분)(출처 기준)"처럼 괄호가 두 번 겹쳐 보인다. 출처 괄호와 안 겹치게
        # 성분 표기는 뗀다(요약 문장에서 제품명 쓸 때도 같은 이유로 떼고 있다).
        item_name = _strip_brackets(item_name)
    if item_name:
        suffix = f" 갱신일: {max(dates)}" if dates else ""
        return f"{item_name}({data_source}).{suffix}"
    suffix = f" (갱신일: {max(dates)})" if dates else ""
    return f"데이터 출처: {data_source}{suffix}"


def _assemble_answer(facts, summary, data_source, closing=None, item_name=None):
    """답변을 본문(body)과 footer로 나눠서 (body, footer) 튜플로 돌려준다.
    facts/summary를 따로 주고 메인 LLM에게 조립을 맡기면, 라벨을 **굵게**로 바꾸거나
    문단을 빠뜨리거나 근거에 없는 결론("운전은 가능하다" 등)을 덧붙이는 경우가
    있었다. 조립까지 끝낸 문자열을 그대로 출력하게 하면 이런 흔들림이 없다.
    body는 조회된 사실·정리처럼 근거에 따라 매번 달라지는 핵심 내용이고, footer는
    상담 권유·전문의약품 안내·데이터 출처처럼 거의 고정된 문구다. Streamlit에서
    footer를 작은 회색 글씨로 따로 보여주기 위해 나눈다."""
    # 정리(요약)를 먼저 보여줘서 핵심만 간단히 전달하고, 자세한 근거가 필요하면 아래
    # [조회된 사실]에서 확인하도록 순서를 바꿨다(상담 받는 느낌으로 먼저 결론,
    # 그 다음 근거). "[정리]" 라벨 자체는 기계적인 느낌을 줘서 빼고 문장만 보여준다.
    body_parts = []
    if summary:
        body_parts += [summary, ""]
    body_parts += ["[조회된 사실]", facts]
    body = "\n".join(body_parts)

    # "약사나 의사와 상담하시기 바랍니다"는 더 이상 매 답변마다 붙이지 않는다 — 매번
    # 반복되면 부자연스럽고, 채팅 UI 하단에 고정 안내문으로 한 번만 보여주는 쪽이
    # 낫다(화면 쪽 구현). 여기서는 특정 상황에서만 의미 있는 안내(전문의약품,
    # 병용금기 등록 등)만 footer에 남긴다.
    closings = [c for c in ([closing] if isinstance(closing, str) else (closing or [])) if c]
    footer = " ".join(closings)
    footer = (footer + "\n\n" + _data_source_line(data_source, item_name)) if footer else _data_source_line(data_source, item_name)
    return body, footer


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


_SUMMARY_PROMPT = """아래 "조회된 사실"만 근거로, 약사가 환자에게 직접 설명해주듯 자연스러운
상담 답변으로 풀어 쓰세요. 항목을 기계적으로 나열하지 말고, 결론부터 말한 뒤 조회된 사실 문장
안에 이유나 조건이 이미 들어있으면 그것도 이어지는 문장으로 자연스럽게 설명하세요.
문장을 여러 개 쓸 때는 각 문장을 뚝뚝 끊어서 나란히 두지 말고, 문장 사이 관계에 맞는 접속어로
자연스럽게 이어 붙이세요(예: 뒤 문장이 조건이면 "만약 ~라면", 예외나 반대면 "하지만"/"다만",
같은 맥락을 덧붙이면 "그리고"/"또한"). 접속어 없이 문장만 붙여서 뚝뚝 끊기는 느낌이 나게
쓰지 마세요. 단, 접속어는 조회된 사실에 실제로 있는 서로 다른 문장·조건들 사이의 관계를
연결하는 데만 쓰세요. 출처(어디에 근거했는지)는 첫 문장에서 한 번만 밝히면 됩니다(아래
제품명 규칙 참고) — "그렇게 명시되어 있기 때문이에요"처럼 뒤에 또 이유로 반복하지 마세요.
"왜 금기인지"에 대한 구체적인 의학적 기전(예: 효과가 감소해서, 상호작용이 발생해서, 간이
손상돼서)은 조회된 사실 문장에 그 기전이 직접 적혀 있을 때만 쓰세요. 조항 이름(예:
"상호작용")만 보고 구체적 기전을 추측해서 만들어내지 마세요 — 그 규칙은 아래 [가장 중요한
규칙]이 여기보다 항상 우선합니다.

[가장 중요한 규칙] 이 약이나 이 상황에 대해 당신이 원래 알고 있는 의학 지식은 전부 무시하세요.
아래 "조회된 사실"에 실제로 적힌 문장에만 근거해서 쓰세요. "일반적으로 잘 알려진 사실"이라도
조회된 사실에 그 내용이 없으면 절대 언급하지 마세요. 조회된 사실에 없는 약·성분·상황(예: 질문과
무관한 음주, 다른 약 이름 등)을 새로 끌어오지 마세요.
"~때문에", "~때문이에요", "~해서" 같은 이유 표현은 조회된 사실 문장에 그 기전이 실제로
적혀 있을 때만 쓰세요(예: 사실에 "이러한 사람이 이 약을 복용하면 간손상이 유발될 수 있다"가
있으면 그 기전을 그대로 설명해도 됩니다). 사실에 없는 구체적 의학적 기전(예: 효과가
감소해서, 상호작용이 발생해서, 부작용이 발생할 수 있어서, 안전성을 보장할 수 없어서)을
추측해서 이유로 붙이지 마세요. 출처(어디에 근거했는지)는 첫 문장에서 제품명과 함께 한 번만
밝히고, "그렇게 명시되어 있기 때문이에요"처럼 뒤에 또 이유로 반복하지 마세요 — 이 규칙은
답변에 등장하는 모든 문장에 똑같이 적용됩니다. 예를 들어 사실이 "A약, B약, 술과 함께
사용하지 않는다"처럼 여러 항목을 한 문장에 나열만 하고 있으면(각 항목에 대한 개별 이유가
없음), "술과 함께 사용하지 않아야 해요. A약, B약과도 마찬가지예요."처럼 이유 표현 없이
결론만 나열하고, A약·B약 쪽에 "부작용이 발생할 수 있어서" 같은 새 기전을 따로 지어내
붙이지 마세요. 아래 표현들은 조회된 사실 문장에 그 글자 그대로 적혀 있지 않으면 절대 쓰지
마세요: "예상치 못한 부작용이 발생할 수 있어서/있으니", "효과가 감소해서", "상호작용이
발생할 수 있어서", "안전성을 보장할 수 없어서", "그렇게 명시되어 있기 때문이에요". 이런
표현이 쓰고 싶어지면, 그건 지어내고 있거나 출처를 반복하고 있다는 신호이니 이유 표현 자체를
빼세요. 마찬가지로 조회된 사실이
단순 정보(성분·효능효과 등)만 있고 주의사항이 전혀 없으면, "사용 시 주의가 필요해요" 같은
일반적인 주의 문구를 임의로 덧붙이지 마세요 — 사실에 없으면 아예 언급하지 않는 쪽을 선택하세요.
- 조회된 사실 안에 위험도가 다른 여러 조건이 구분되어 있으면(예: 특정 조건에서만 주의하라는
  문장과, 더 강한 금지 문장이 같이 있는 경우) 그 구분을 살려서 설명하세요. 그런 구분이 조회된
  사실에 없으면 지어내지 마세요.
- 조회된 사실의 bullet는 이미 사용자 질문에 더 가깝고 직접적인 근거가 먼저, 조건이 붙어
  더 좁은 근거가 그 다음 순서로 정렬되어 있습니다. 답변 문장도 이 순서를 따라, 먼저 나온
  근거의 결론을 먼저 말하고 그 다음 근거는 뒤에 보충하듯 이어 쓰세요. bullet 순서를
  무시하고 두 번째 이후 근거를 먼저 말하지 마세요.
- 조건마다 핵심 결론을 설명하세요. 한 조건에 여러 주의사항이 있으면 전부 나열하지 말고 가장
  중요한 내용 위주로 자연스럽게 이어 쓰세요.
- 분량은 짧게 끝날 수 있으면 한두 문장으로 쓰고, 조회된 사실 자체에 설명할 만한 조건이나
  이유가 있을 때만 3~4문장으로 늘리세요. 조회된 사실에 없는 내용으로 문장 수를 채우지 마세요.
- 근거가 없다고 나온 조건은 굳이 다시 언급하지 않아도 됩니다.
- 조회된 사실에 **조건명** 형태의 굵은 제목이 있으면, 그 제목이 사용자가 실제로 물어본 대상입니다. 근거 문장에 다른 항목이 함께 나열되어 있어도(예: "기침약, 가래약, ..., 술 등과 함께 사용하지 않는다"), 목록의 다른 항목이 아니라 반드시 그 제목(술)을 설명의 중심(주어/대상)으로 쓰세요.
- 결론의 강도는 조회된 사실 그대로 옮기세요. '안전하다'는 말은 쓰지 마세요. 조회된 사실이
  특정 조건에서만 주의·상담을 요구하는 조건부 표현이면, 그 조건을 모든 경우로 넓혀서 전면
  금지처럼 과장하지 마세요. 조회된 사실이 실제로 전면 금지("~하지 않는다" 등)라고 할 때만
  그렇게 쓰세요.
- 약사나 의사가 환자에게 직접 설명해주는 말투로, 문장을 "~요"로 끝내세요(예: "~사용돼요", "~주의가 필요해요", "~복용하지 않아야 해요"). "~다"나 "~습니다"로 끝내지 마세요.{product_rule}

조회된 사실:
{facts}
"""

_SUMMARY_PRODUCT_RULE = """
- 문장은 반드시 "{item_name}{josa} {source}에 따라, "로 시작하세요. 그 뒤에 조회된
  사실에 맞는 결론을 자연스럽게 이어 쓰세요(예시는 강도가 아니라 "제품명+출처로 시작한다"는
  형식만 참고하세요: "{item_name}{josa} {source}에 따라, 특정 상황에서 주의가 필요해요"
  또는 "{item_name}{josa} {source}에 따라, 이 경우 복용하지 않아야 해요" 등 사실의 실제
  강도에 맞게 쓰세요). "이 약은" 같은 일반적인 표현 대신 제품명을 직접 쓰세요. 이렇게 첫
  문장에서 출처를 한 번 밝혔으니, 뒤에 "또한"으로 덧붙이는 문장에는 "그렇게 명시되어 있기
  때문이에요"처럼 출처를 다시 이유로 대지 마세요 — 반복입니다."""


def _josa_eun_neun(word):
    """단어 마지막 글자의 받침 유무로 '은'/'는'을 고른다. 은/는 선택을 LLM에 맡기면
    가끔 틀리므로(받침 판단은 기계적으로 정해지는 문제라 LLM이 판단할 필요가 없다)
    Python에서 미리 정해서 프롬프트에 완성된 조사를 박아 넣는다. 한글이 아닌 글자로
    끝나면(숫자·영문 등) 받침 유무를 판단하기 어려워 '는'으로 처리한다."""
    if not word:
        return "는"
    code = ord(word[-1]) - 0xAC00
    if 0 <= code <= 11171:
        return "은" if code % 28 != 0 else "는"
    return "는"


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


def _make_summary(facts, item_name=None, data_source=None):
    product_rule = ""
    if item_name:
        display_name = _strip_brackets(item_name)
        # footer의 "데이터 출처: ..."와 같은 값을 재사용한다. 문장 안에 녹일 때는 끝의
        # " 기준"을 떼서 "~기준에 따라"처럼 어색하게 겹치지 않게 한다("~에 따라" 자체가
        # 이미 "기준으로"라는 뜻이라 "기준"이 중복된다).
        source = re.sub(r"\s*기준\s*$", "", data_source) if data_source else "식약처 등록 정보"
        product_rule = _SUMMARY_PRODUCT_RULE.format(
            item_name=display_name, josa=_josa_eun_neun(display_name), source=source
        )
    prompt = _SUMMARY_PROMPT.format(facts=facts, product_rule=product_rule)
    resp = _client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
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
            paras = _pick_with_relevance_check(c, raw)
            paras = _attach_body(paras, _get_paragraphs(item_seq, field), condition=c)
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
        summary = _make_summary(facts, item_name=item_name, data_source="의약품 제품허가정보 기준")
        answer, footer = _assemble_answer(facts, summary, "의약품 제품허가정보 기준", closing=rx_closing, item_name=item_name)
        return {
            "resolved": True,
            "item_name": item_name,
            "answer": answer,
            "footer": footer,
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
    # 조건 선별에는 원문 질문 그대로("에나폰정10밀리그램 먹을 때 고추 마셔도 되나요?")가
    # 아니라 extract_conditions가 뽑아낸 깨끗한 조건 문장을 쓴다. 약 이름과 "먹을 때"
    # 같은 말이 섞여 있으면, 음식과 무관한 "약물 등 복용 시 주의" 문단을 음식 조건에
    # 맞는 것처럼 잘못 고르는 경우가 재현됐다(조건별 경로는 이미 깨끗한 문장을 쓴다).
    condition_text = conditions[0] if conditions else question
    picked = _pick_with_relevance_check(condition_text, paragraphs)
    picked = _attach_body(picked, _get_paragraphs(item_seq, field), condition=condition_text)
    if not picked:
        return _no_evidence_result(item_name, closing=rx_closing)
    # 근거 문장 하나에 여러 항목이 나열된 경우(예: "기침약, 가래약, ..., 술 등과
    # 함께 사용하지 않는다"), 조건 제목 없이 요약을 시키면 LLM이 사용자가 실제로
    # 물어본 항목(술)이 아니라 목록의 첫 항목(기침약)을 주어로 골라버리는 문제가
    # 재현됐다. 조건별 경로(_render_facts)와 똑같이 **조건명** 제목을 붙여서 요약이
    # 반드시 그 조건을 주어로 삼게 한다.
    facts = f"**{condition_text}**\n{_bullets_to_md(_render_bullets([p['text'] for p in picked]))}"
    summary = _make_summary(facts, item_name=item_name, data_source="의약품 제품허가정보 기준")
    answer, footer = _assemble_answer(facts, summary, "의약품 제품허가정보 기준", closing=rx_closing, item_name=item_name)
    return {
        "resolved": True,
        "item_name": item_name,
        "answer": answer,
        "footer": footer,
        "relevant_paragraphs": picked,
        "note": _ANSWER_NOTE,
    }


def _no_evidence_result(item_name, closing=None, topic="질문하신 내용", data_source="의약품 제품허가정보 기준"):
    # 근거가 없으면 LLM이 제각각 말하지 않도록 문장을 코드에서 고정한다. topic을 주면
    # ("임부·수유부" 등) 뭘 못 찾았는지 구체적으로 밝히고, 안 주면 기존처럼 "질문하신
    # 내용"으로 뭉뚱그린다.
    sentence = (f"{item_name}의 사용상주의사항에는 {topic} 관련 주의사항이 없습니다만, "
                "개인의 상태에 따라 다를 수 있으니 복용 후 평소와 다른 증상이 나타나면 "
                "의사나 약사와 상담하시기 바랍니다.")
    footer = f"{closing}\n\n" if closing else ""
    footer += _data_source_line(data_source, item_name)
    return {
        "resolved": True,
        "item_name": item_name,
        "relevant_paragraphs": [],
        "answer": sentence,
        "footer": footer,
        "note": _ANSWER_NOTE,
    }


_OVERDOSE_BOUNDARY_RE = re.compile(r"^\d+[.)]")


def _extract_overdose_blocks(raw_caution):
    """원문에서 "과량"/"과다"가 들어간 줄만 뽑으면 "13) 이 약의 과량 복용 시 다음과
    같은 이상반응이 나타날 수 있다." 같은 제목 줄만 남고, 그 아래 번호로 나열된 실제
    증상·처치 내용(실측: "(1) 귀와 미로 이상: 어지러움" 등 8개 줄, "과량"이라는 글자가
    안 들어간 문구 하나하나)은 전부 걸러져 사라진다(실측: 부루펜정200밀리그램).
    그래서 "과량"/"과다"가 들어간 줄을 찾으면 그 줄부터, 다음 번호 항목("6.", "7)" 등
    줄머리의 숫자+점/괄호)이 나오기 전까지를 통째로 한 블록으로 가져온다."""
    lines = raw_caution.split("\n")
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        if any(k in lines[i] for k in ("과량", "과다")):
            block = [lines[i].strip()]
            j = i + 1
            while j < n and not _OVERDOSE_BOUNDARY_RE.match(lines[j].strip()):
                if lines[j].strip():
                    block.append(lines[j].strip())
                j += 1
            blocks.append(" ".join(block))
            i = j
        else:
            i += 1
    return blocks


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
    overdose_sentences = _extract_overdose_blocks(raw_caution)
    if not overdose_sentences:
        return _no_evidence_result(item_name, closing=rx_closing)
    facts = _plain_bullets(overdose_sentences, "과량투여")
    summary = _make_summary(facts, item_name=item_name, data_source="의약품 제품허가정보 기준")
    answer, footer = _assemble_answer(facts, summary, "의약품 제품허가정보 기준", closing=rx_closing, item_name=item_name)
    return {
        "resolved": True,
        "item_name": item_name,
        "answer": answer,
        "footer": footer,
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
        # 브랜드명(예: "타이레놀") -> 되묻기로 확정된 구체적 제품명(예: "타이레놀정500밀리그람").
        # 한 번 확정되면 대화가 끝날 때까지 그 브랜드명이 다시 나올 때마다 매번 다시
        # 되묻지 않고 이 제품으로 바로 해석한다. 사용자가 같은 계열의 다른 구체적
        # 제품명을 직접 말하면 그 이름은 이 사전의 키와 다른 문자열이라 그대로
        # 통과하고, 이 사전 자체는 바뀌지 않는다.
        self.product_pins = {}
        # 바로 직전 도구 호출이 제품을 특정하지 못해 되물은 경우, 그때 쓰인(치환 전)
        # 브랜드명. 사용자가 그 되묻기에 바로 답해서 이번 턴에 제품이 특정되면 이
        # 브랜드명을 product_pins에 등록한다. 되묻기에 답하지 않고 다른 질문으로
        # 넘어가면(= 이번 턴이 되묻기 응답이 아니면) 버린다.
        self._pending_unresolved_name = None

    def send(self, user_message: str, max_tool_rounds: int = 5) -> dict:
        """반환값은 {"text": 핵심 답변, "footer": 상담 권유·안내·데이터 출처(없으면
        None)}다. Streamlit 등에서 footer를 본문과 다르게(작은 회색 글씨로) 보여줄
        수 있게 나눈다. 대화 기록(self.messages)에는 둘을 합친 전체 텍스트를
        남겨서, 다음 턴에 모델이 이전에 말한 내용을 전부 보게 한다."""
        emergency = detect_emergency(user_message)
        if emergency:
            return {"text": emergency["message"], "footer": None}

        self.messages.append({"role": "user", "content": user_message})
        last_lifestyle_result = None
        last_tool_name = None
        last_tool_result = None

        # 되묻기에 제품명만 답한 경우, 원래 질문 의도를 이어받은 텍스트. 도구 인자의
        # question을 채우거나 질문 의도를 키워드로 판단할 때는 이걸 쓰고, 감정·긴급
        # 상황 감지처럼 "지금 이 문장 자체"가 중요한 곳에는 user_message를 그대로 쓴다.
        question_context = _effective_question(self.messages, user_message)

        # 이번 턴이 직전 되묻기("...말씀하시는 건가요?")에 대한 답인지. 맞으면 이번
        # 턴에 제품이 특정될 때 self._pending_unresolved_name을 브랜드명 고정(핀)으로
        # 등록한다. 아니면(사용자가 되묻기를 무시하고 다른 질문으로 넘어간 경우)
        # 남아있던 pending은 더 이상 유효하지 않으므로 버린다.
        is_clarification_reply = _is_clarification_reply(self.messages)
        if not is_clarification_reply:
            self._pending_unresolved_name = None

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
                footer = None
                if SCORE_DEBUG:
                    cited_nums = _detect_citations(user_message, last_lifestyle_result, content)
                    debug = _build_score_debug_block(last_tool_name, last_tool_result, last_lifestyle_result, cited_nums, content)
                    grades = _auto_grade(user_message, last_tool_name, last_tool_result, content)
                    footer = (debug + _build_grading_block(grades)).strip() or None
                return {"text": content, "footer": footer}

            self.messages.append(msg)
            dispatch_cache = {}
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                if fn_name == "check_drug_interactions" and len(fn_args.get("product_names", [])) < 2:
                    # 병용 확인은 제품이 2개 이상일 때만 의미가 있다. 1개뿐이면 질병·조건에 대한
                    # 질문이므로 생활질문 도구로 바꿔서 원래 질문 전체를 넘긴다.
                    fn_name = "ask_lifestyle_question"
                    fn_args = {"product_name": fn_args["product_names"][0], "question": question_context}
                if fn_name == "check_drug_interactions":
                    # 병용확인으로 넘어온 이름 중 음식·음료가 있으면 병용확인이 아니라 생활질문이다.
                    names = fn_args.get("product_names", [])
                    kinds = classify_names(names, CHAT_MODEL) if names else {}
                    foods = [n for n in names if kinds.get(n) == "food"]
                    if foods:
                        drugs = [n for n in names if n not in foods]
                        fn_name = "ask_lifestyle_question"
                        fn_args = {"product_name": (drugs or names)[0], "question": question_context}
                if fn_name == "ask_lifestyle_question" and _dur_type_for_question(question_context):
                    # 임부금기/서방정분할주의/첨가제주의 같은 DUR 유형 질문은 생활질문의 NB
                    # 원문 검색보다 DUR 항목으로 답하는 게 더 짧고 정확하다. "며칠"/"몇 알"처럼
                    # 기간·용량 질문과 겹치는 경우는 바로 아래에서 다시 생활질문으로 되돌아간다
                    # (그쪽이 NB·UD 원문에서 더 구체적인 답을 찾는다).
                    fn_name = "get_drug_info"
                    fn_args = {"product_name": fn_args["product_name"], "question": question_context}
                if fn_name == "get_drug_info":
                    _conditions = extract_conditions(question_context, CHAT_MODEL)
                if fn_name == "get_drug_info" and (
                    len(_conditions) >= 2
                    or _DURATION_OR_AMOUNT_RE.search(question_context)
                ):
                    # 조건이 두 개 이상이면 DUR 전문 조회로는 조건별 근거를 보장할 수 없다.
                    # 생활질문 경로에서 조건마다 따로 검색한다.
                    fn_name = "ask_lifestyle_question"
                    fn_args = {"product_name": fn_args["product_name"], "question": question_context}
                if fn_name == "get_drug_info" and _age_group(_strip_product_name(question_context, fn_args["product_name"])):
                    # 나이 질문은 연령 DUR 항목 원문으로 답한다. 전체 조회 결과를 주면 원문에 없는 결론이 섞인다.
                    fn_name = "get_age_dur_info"
                    fn_args = {"product_name": fn_args["product_name"], "question": question_context}
                if (
                    fn_name == "get_drug_info"
                    and _info_scope(question_context) is None
                    and _conditions
                    and not _dur_type_for_question(question_context)
                ):
                    # 범위를 특정하지 않은 일반 질문인데 구체적인 생활 조건(술 등)이 하나라도
                    # 있으면 전체 덤프가 아니라 ask_lifestyle_question이 맞다(7-(2) 규칙).
                    # 되묻기에 제품명만 답한 경우, 모델이 원래 조건을 다시 반영해 호출하지
                    # 않고 get_drug_info를 직접 부르는 경우가 재현됐다(예: "게보린
                    # 술마시고 먹어도 돼?" -> 되묻기 -> "게보린정"만 답하면 모델이
                    # get_drug_info(게보린정)을 불러서, scope=None이라 전체 덤프가 나오고
                    # "술" 맥락이 묻혔다). 조건이 이미 2개 이상이거나 나이 질문이면 위에서
                    # 먼저 처리되므로 여기 오지 않는다. _dur_type_for_question이 맞으면
                    # 바로 위(1472줄)의 reroute가 "DUR 항목이 NB 전문 검색보다 더 정확하다"는
                    # 이유로 일부러 get_drug_info에 머물게 한 것이므로, 여기서 다시
                    # ask_lifestyle_question으로 되돌리면 그 reroute가 무의미해진다.
                    fn_name = "ask_lifestyle_question"
                    fn_args = {"product_name": fn_args["product_name"], "question": question_context}
                if fn_name == "get_drug_info":
                    # 질문 문맥에 따라 사용상주의사항 중 관련 문단과 DUR 유형을 고르려면 원래
                    # 질문이 필요하다. 모델이 실제로 넘긴 건 product_name뿐이라 직접 채운다.
                    fn_args = {"product_name": fn_args["product_name"], "question": question_context}
                if fn_name == "ask_lifestyle_question":
                    # 질문에 조건이 여러 개 섞여 있으면("운전해도 되고 임산부도 먹어도 돼?")
                    # 모델이 조건별로 ask_lifestyle_question을 따로따로(한 번에 여러
                    # tool_call) 호출하는 경우가 재현됐다. 도구 내부(extract_conditions)가
                    # 이미 원래 질문 하나로 조건을 전부 나눠 처리하므로, 항상
                    # question_context(원래 질문 전체)로 통일해서 모델이 자기 방식대로
                    # 쪼갠 질문 대신 도구가 한 번에 전부 처리하게 한다.
                    fn_args = {"product_name": fn_args["product_name"], "question": question_context}
                # 브랜드명(예: "타이레놀")이 이미 product_pins에 구체적 제품으로 고정돼
                # 있으면, 다시 되묻지 않고 그 제품으로 바로 치환한다. 치환 전 이름은
                # 아래 핀 등록에 쓰려고 따로 남겨 둔다.
                orig_product_name = fn_args.get("product_name")
                if orig_product_name is not None:
                    pinned = self.product_pins.get(orig_product_name.strip())
                    if pinned:
                        fn_args["product_name"] = pinned
                if "product_names" in fn_args:
                    fn_args["product_names"] = [
                        self.product_pins.get(n.strip(), n) for n in fn_args["product_names"]
                    ]

                # 위 통일 때문에 조건이 여러 개면 서로 다른 tool_call들이 결국 같은
                # (fn_name, fn_args)로 수렴한다. 그대로 두면 똑같은 조회를 여러 번
                # 반복하고, 호출이 여럿이라는 이유만으로 아래 shortcut이 꺼져서 결과를
                # 다시 LLM이 조립하다가 굵은 제목 같은 형식이 흔들린다. 라운드 안에서
                # (fn_name, fn_args)가 같으면 한 번만 실제로 호출하고 결과를 재사용한다.
                dispatch_key = (fn_name, json.dumps(fn_args, sort_keys=True, ensure_ascii=False))
                if dispatch_key in dispatch_cache:
                    result = dispatch_cache[dispatch_key]
                else:
                    fn = TOOL_DISPATCH.get(fn_name)
                    result = fn(**fn_args) if fn else {"error": f"unknown tool {fn_name}"}
                    dispatch_cache[dispatch_key] = result

                # 브랜드명 고정(핀) 갱신. product_name(단일 제품 도구)에만 적용한다 —
                # check_drug_interactions처럼 product_names(목록)인 경우는 어떤
                # 브랜드명이 어떤 제품으로 풀렸는지 1:1로 대응시키기 애매해서 범위를
                # 좁힌다.
                if orig_product_name is not None and isinstance(result, dict):
                    if result.get("item_name"):
                        if is_clarification_reply and self._pending_unresolved_name:
                            self.product_pins[self._pending_unresolved_name.strip()] = result["item_name"]
                        self._pending_unresolved_name = None
                    elif result.get("clarification_question") or result.get("candidate_groups"):
                        self._pending_unresolved_name = orig_product_name

                if fn_name == "ask_lifestyle_question":
                    last_lifestyle_result = result
                last_tool_name = fn_name
                last_tool_result = result
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

            if len(dispatch_cache) == 1 and isinstance(last_tool_result, dict) and "answer" in last_tool_result:
                # 도구가 이미 answer(+footer)를 완성해 줬다. 한 번 더 모델을 불러
                # "그대로 출력"을 시키면, 모델이 옮겨 적는 과정에서 줄바꿈이나 라벨이
                # 가끔 흔들렸다. 모델에게 다시 쓰게 하지 않고 Python이 그 문자열로
                # 바로 끝낸다.
                text = last_tool_result["answer"]
                footer = last_tool_result.get("footer")
                combined = f"{text}\n\n{footer}" if footer else text
                self.messages.append({"role": "assistant", "content": combined})
                if SCORE_DEBUG:
                    cited_nums = _detect_citations(user_message, last_lifestyle_result, combined)
                    debug = _build_score_debug_block(last_tool_name, last_tool_result, last_lifestyle_result, cited_nums, combined)
                    grades = _auto_grade(user_message, last_tool_name, last_tool_result, combined)
                    debug += _build_grading_block(grades)
                    footer = f"{footer}\n\n{debug}" if footer else debug
                return {"text": text, "footer": footer}

        return {"text": "죄송합니다, 답변을 생성하는 데 문제가 발생했습니다.", "footer": None}


def chat(user_message: str, max_tool_rounds: int = 5) -> dict:
    """대화 기록이 필요 없는 단발성 질문용 편의 함수. 반환값은 Conversation.send()와
    같다({"text":, "footer":}). 멀티턴 대화는 Conversation을 쓴다."""
    return Conversation().send(user_message, max_tool_rounds=max_tool_rounds)
