"""자연어 질문 -> (응급 감지) -> LLM 도구 호출 -> 우리 함수 실행 -> 최종 답변.

응급 감지는 LLM 호출 전에 코드로 먼저 걸러낸다("코드가 먼저 가로챈다" 원칙).
그 외에는 OpenAI의 tool calling으로 어떤 함수를 어떤 인자로 부를지 LLM이
판단하게 하고, 우리는 실제 실행과 결과 전달만 담당한다.
"""
import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from emergency_detect import detect_emergency
from drug_lookup import check_multiple_drugs, search_product_with_confidence, get_product_detail, _get_detail
from text_extract import doc_xml_to_text
from symptom_search import search_by_symptom
from paragraph_search import search_paragraphs

load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHAT_MODEL = "gpt-4o-mini"

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
   [조회된 사실] -> [주의사항] -> [상담 권유] 순서를 지키세요.
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
4. 병용금기가 등록되지 않았다고 해서 안전하다는 뜻이 아님을 항상 명시하세요.
   check_drug_interactions 결과에 symptom_safety_note 필드가 있으면, 그 문장을
   고치거나 요약하지 말고 그대로 답변에 포함하세요(등록된 병용금기가 있는지
   없는지와 무관하게 항상 포함).
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
       즉시 ask_lifestyle_question을 호출하세요. 판단 기준은 "10번 규칙의
       dur_cautions 7개 유형(노인주의/특정연령대금기/용량주의/투여기간주의/
       서방정분할주의/임부금기/첨가제주의)에 해당하는가"입니다 — 해당하면
       (3)의 get_drug_info로, 해당하지 않으면 전부 이 경우입니다. 음주/공복/
       운전은 물론, 수술 예정/자몽주스/녹내장/전립선비대/간질환/신장질환/
       두드러기·알레르기 병력처럼 dur_cautions에 없는 주제는 모두 여기
       해당합니다. "음주/공복/운전 등"이라는 예시 몇 개에 없다고 해서 (3)으로
       보내지 마세요 — 예시는 전부가 아니라 일부일 뿐입니다.
   (3) 위 두 경우에 해당하지 않고 약 이름 하나만 있다 -> 되묻지 말고 바로
       get_drug_info를 호출해서 성분/효능/주의사항을 전달하세요. 복용 중인
       다른 약이 없는 사용자에게 매번 "다른 약 드세요?"부터 묻는 건 불필요한
       질문을 하나 더 받게 만드므로, 먼저 정보부터 주는 게 맞습니다. 도구
       결과의 has_interaction_data가 true이면(이 약이 등록된 병용금기 쌍에
       포함돼 있다는 뜻), 답변 끝에 "이 약은 특정 약과 함께 복용 시 주의가
       필요한 조합이 등록되어 있습니다. 현재 복용 중인 다른 약이 있으면
       알려주시면 추가로 확인해드리겠습니다"라는 취지의 문장을 덧붙이세요.
       has_interaction_data가 false면 이 문장을 붙이지 마세요. 사용자가
       나중에 다른 약 이름을 알려주면 그때 check_drug_interactions를
       호출하세요.
   (1)과 (2)에 해당하는 질문에는 절대로 "현재 다른 약을 복용하고 계신가요?"
   라고 되묻지 마세요 — 질문에 이미 답이 나와 있습니다.
8. 도구 결과에 data_source 필드가 있으면, 답변 마지막에 "데이터 출처: {그 값}"을
   그대로 쓰세요. 어떤 출처를 쓸지 직접 판단하거나 "DUR" 같은 다른 용어로
   바꾸지 말고, data_source 값을 그대로 옮기세요. 실제 데이터를 전달하지 않고
   되묻기만 하는 답변(3번 예외 참고)에는 이 출처 표기 자체를 넣지 마세요.
9. get_drug_info 결과의 cancel_name이 "정상"이 아니면(예: "취소", "취하", "폐업",
   "유효기간만료" 등), 이 제품은 더 이상 정상적으로 판매되지 않는다는 뜻이므로
   절대 빠뜨리지 말고 [조회된 사실] 맨 처음에 가장 먼저 알리세요(예: "이 제품은
   {cancel_date}자로 {cancel_name} 처리되어 더 이상 정상적으로 판매되지 않습니다.").
   이미 가지고 있는 사용자를 위해 효능/주의사항 등 나머지 정보는 계속 전달하되,
   이 약을 계속 복용 중이라면 약사나 의사와 상담하라고 안내하세요. (cancel_name이
   정상인 경우 도구 결과에 그 필드 자체가 없으니, 판매 상태는 비정상일 때만
   언급하면 됩니다.)
10. get_drug_info나 check_drug_interactions 결과의 dur_cautions 목록에는 노인주의/
    특정연령대금기/용량주의/투여기간주의/서방정분할주의/임부금기/첨가제주의가 섞여
    있을 수 있습니다. 사용자 질문이 "노인이 먹어도 되나요", "소아가 먹어도
    되나요", "하루 몇 번까지", "며칠까지 먹어도 되나요", "쪼개 먹어도 되나요",
    "임신 중인데 먹어도 되나요"처럼 특정 대상·측면을 콕 집어 물었으면, dur_cautions
    중 그 질문에 해당하는 유형 하나만 답하고 나머지 무관한 유형(예: 노인 질문에
    임부금기나 첨가제주의)은 답변에 꺼내지 마세요 — 묻지 않은 걸 같이 알려주면
    오히려 헷갈립니다. 해당 유형이 목록에 없으면 "등록된 정보가 없습니다"라고만
    답하세요. 반대로 "이 약 먹어도 되나요"처럼 범위를 특정하지 않고 물었다면,
    이때는 dur_cautions의 모든 항목을 [주의사항]에 포함해도 됩니다.
    content가 있으면 그 문구를 그대로 전달하고, content가 없으면("해당 유형으로
    등록되어 있으나 구체적인 문구는 없음") "~에 대한 주의가 필요한 성분/품목으로
    등록되어 있습니다" 정도로만 언급하고 구체적인 제한 수치나 대상 연령을 지어내지
    마세요. 등록된 항목이 없으면 "등록된 정보가 없습니다"라고만 답하고
    안전하다고 단정하지 마세요.
11. check_drug_interactions 결과에 effect_group_duplicates가 있으면(서로 다른
    성분이지만 같은 효능군에 속해 효과가 중복될 수 있는 경우), duplicate_ingredients와
    마찬가지로 과다복용과 비슷한 위험으로 안내하세요(예: "두 약 모두
    {effect_group}에 속해 효과가 중복될 수 있습니다").
12. ask_lifestyle_question 결과의 relevant_paragraphs는 유사도 순으로 정렬된
    후보 문단 목록입니다(score가 높을수록 질문과 관련 있을 가능성이 높지만,
    낮은 순위에 실제 정답이 있을 수도 있으니 목록 전체를 검토하세요). 이 중
    질문에 실제로 답이 되는 내용이 있는지 직접 판단한 뒤:
    - 답이 되는 문단이 있으면 "사용상주의사항에 따르면"으로 시작해 그 문단의
      원문을 거의 그대로(의역하거나 결론을 지어내지 말고) 근거로 제시하세요.
      추측이나 일반 상식으로 답을 보충하지 말고, 그 문단에 실제로 쓰여 있는
      내용만 전달하세요.
    - 여러 문단에 걸쳐 관련 내용이 나뉘어 있으면 그것들을 종합해서 답해도
      되지만, 역시 각 문단에 실제로 쓰여 있는 내용 범위를 벗어나지 마세요.
    - 후보 중 실제로 질문에 답이 되는 내용이 없으면 "관련된 사용상주의사항을
      찾을 수 없습니다"라고 답하세요. 억지로 가장 비슷해 보이는 문단을 답인
      것처럼 포장하지 마세요.
13. check_drug_interactions 결과에 same_product_repeated가 true이면, 사용자가
    같은 제품을 여러 번 말한 것입니다. 이는 병용 문제가 아니므로 "병용", "함께
    복용 시 주의" 같은 표현을 쓰지 마세요. 대신 "같은 제품을 중복해서 드시는
    것이므로 과다복용에 주의해야 합니다"라고 안내하고, usage_info의 용법용량 원문과
    과량투여 관련 원문을 그대로 전달하세요(몇 시간 간격인지, 성인·소아 기준 몇 정인지
    등). 원문에 없는 내용은 지어내지 마세요.
14. 사용자가 과다복용·과량 복용 결과를 물으면(예: "타이레놀 과다복용하면 어떻게 돼?"),
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
]


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
        return {
            "same_product_repeated": True,
            "item_name": detail.get("item_name"),
            "usage_info": {
                "용법용량": detail.get("용법용량"),
                "과량투여_관련_사용상주의사항": overdose_sentences,
            },
            "data_source": "의약품 제품허가정보 기준",
        }

    result["data_source"] = "의약품안전사용서비스 기준"
    result["symptom_safety_note"] = (
        "이 약들을 함께 복용한 후 평소와 다른 증상(어지러움, 메스꺼움, 두드러기, "
        "졸림 등)이 나타나면 즉시 복용을 중단하고 의사나 약사와 상담하세요. "
        "증상이 심하면 119에 연락하세요."
    )
    return result


def _tool_get_drug_info(product_name):
    result = search_product_with_confidence(product_name)
    if not result["confident"]:
        return {
            "resolved": False,
            "candidate_groups": result.get("candidate_groups", []),
            "clarification_question": result.get("clarification_question"),
            "note": "제품명이 정확히 특정되지 않았습니다. clarification_question 문장을 "
                    "그대로 답변에 포함해 사용자에게 되물으세요.",
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
    return detail


def _tool_search_by_symptom(symptom, current_medication_names=None):
    ingredients = []
    for name in (current_medication_names or []):
        r = search_product_with_confidence(name)
        if r["confident"]:
            detail = get_product_detail(r["candidates"][0]["item_seq"])
            ingredients.extend(detail.get("ingredients", []))

    results = search_by_symptom(symptom, current_med_ingredients=ingredients)
    if results is None:
        return {"error": f"'{symptom}'은 지원하지 않는 증상 분류입니다."}
    return {"candidates": results, "data_source": "의약품 제품허가정보 기준"}


def _tool_ask_lifestyle_question(product_name, question, field="NB"):
    result = search_product_with_confidence(product_name)
    if not result["confident"]:
        return {
            "resolved": False,
            "candidate_groups": result.get("candidate_groups", []),
            "clarification_question": result.get("clarification_question"),
            "note": "제품명이 정확히 특정되지 않았습니다. clarification_question 문장을 "
                    "그대로 답변에 포함해 사용자에게 되물으세요.",
        }
    item_seq = result["candidates"][0]["item_seq"]
    item_name = result["candidates"][0]["item_name"]
    paragraphs = search_paragraphs(item_seq, field, question, top_k=8)
    return {
        "resolved": True,
        "item_name": item_name,
        "relevant_paragraphs": paragraphs,
        "data_source": "의약품 제품허가정보 기준",
    }


TOOL_DISPATCH = {
    "check_drug_interactions": _tool_check_drug_interactions,
    "get_drug_info": _tool_get_drug_info,
    "search_by_symptom": _tool_search_by_symptom,
    "ask_lifestyle_question": _tool_ask_lifestyle_question,
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

        for _ in range(max_tool_rounds):
            response = _client.chat.completions.create(
                model=CHAT_MODEL,
                messages=self.messages,
                tools=TOOLS,
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

        return "죄송합니다, 답변을 생성하는 데 문제가 발생했습니다."


def chat(user_message: str, max_tool_rounds: int = 5) -> str:
    """대화 기록이 필요 없는 단발성 질문용 편의 함수. 멀티턴 대화는 Conversation을 쓴다."""
    return Conversation().send(user_message, max_tool_rounds=max_tool_rounds)
