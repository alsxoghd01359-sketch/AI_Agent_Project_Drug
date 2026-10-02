"""자연어 질문 -> (응급 감지) -> LLM 도구 호출 -> 우리 함수 실행 -> 최종 답변.

응급 감지는 LLM 호출 전에 코드로 먼저 걸러낸다("코드가 먼저 가로챈다" 원칙).
그 외에는 OpenAI의 tool calling으로 어떤 함수를 어떤 인자로 부를지 LLM이
판단하게 하고, 우리는 실제 실행과 결과 전달만 담당한다.
"""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from emergency_detect import detect_emergency
from drug_lookup import check_multiple_drugs, search_product_with_confidence, get_product_detail
from symptom_search import search_by_symptom
from paragraph_search import search_paragraphs

load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHAT_MODEL = "gpt-4o-mini"

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
   (1) 질문 안에 다른 약 이름이 이미 나와 있다 -> 되묻지 말고 즉시
       check_drug_interactions를 호출하세요.
   (2) 질문 안에 음주/공복/운전 등 구체적인 생활 상황이 이미 나와 있다 ->
       되묻지 말고 즉시 ask_lifestyle_question을 호출하세요.
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
                            "노인/소아 등 특정연령대 주의, 용량주의, 투여기간주의, "
                            "서방정분할주의, 임부금기 등록 여부도 함께 반환하므로 "
                            "'노인이 먹어도 되나요', '하루 몇 알까지', '쪼개 먹어도 "
                            "되나요', '임신 중인데 먹어도 되나요' 같은 질문에도 이 "
                            "도구를 쓴다.",
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
            "description": "이미 특정된 제품 하나에 대해 '술 마셔도 되나', '공복에 먹어야 하나', "
                            "'이 부작용 정상인가' 같은 생활 밀착 질문에 답하기 위해 관련 문단을 검색한다. "
                            "의약품안전사용서비스 조회로 답할 수 없는, 약 하나에 대한 질문에만 사용.",
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

        for _ in range(max_tool_rounds):
            response = _client.chat.completions.create(
                model=CHAT_MODEL,
                messages=self.messages,
                tools=TOOLS,
                temperature=0,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                self.messages.append({"role": "assistant", "content": msg.content})
                return msg.content

            self.messages.append(msg)
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                fn = TOOL_DISPATCH.get(fn_name)
                result = fn(**fn_args) if fn else {"error": f"unknown tool {fn_name}"}
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

        return "죄송합니다, 답변을 생성하는 데 문제가 발생했습니다."


def chat(user_message: str, max_tool_rounds: int = 5) -> str:
    """대화 기록이 필요 없는 단발성 질문용 편의 함수. 멀티턴 대화는 Conversation을 쓴다."""
    return Conversation().send(user_message, max_tool_rounds=max_tool_rounds)
