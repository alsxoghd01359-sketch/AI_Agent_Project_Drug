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
3. 모든 답변은 [조회된 사실] -> [데이터의 한계 고지] -> [전문가 상담 권유] 순서를 지키세요.
4. 병용금기가 등록되지 않았다고 해서 안전하다는 뜻이 아님을 항상 명시하세요.
5. 도구 결과의 resolved가 false이거나 confident가 false면, 절대 특정 제품으로
   단정하지 마세요. 도구 결과에 clarification_question 필드가 있으면, 그
   문장을 직접 새로 만들지 말고 반드시 그 문장을 그대로(고치거나 요약하지 말고)
   답변에 포함해서 되물으세요. 성분 목록이나 제품 목록을 따로 나열하지 마세요.
6. 답변 마지막에 데이터 출처를 명시하세요(예: "DUR 데이터 기준", "의약품 제품허가정보 기준").
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_drug_interactions",
            "description": "2개 이상의 의약품을 함께 복용해도 되는지 확인한다. "
                            "DUR 병용금기 등록 여부와 동일 성분 중복(과다복용 위험) 여부를 반환한다.",
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
                            "사용상주의사항, 상호작용, 이상반응, 판매상태를 조회한다.",
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
                            "DUR로 답할 수 없는, 약 하나에 대한 질문에만 사용.",
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
    return check_multiple_drugs(product_names)


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
    detail = get_product_detail(item_seq)
    detail["resolved"] = True
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
    return {"candidates": results}


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
    paragraphs = search_paragraphs(item_seq, field, question, top_k=3)
    return {"resolved": True, "item_name": item_name, "relevant_paragraphs": paragraphs}


TOOL_DISPATCH = {
    "check_drug_interactions": _tool_check_drug_interactions,
    "get_drug_info": _tool_get_drug_info,
    "search_by_symptom": _tool_search_by_symptom,
    "ask_lifestyle_question": _tool_ask_lifestyle_question,
}


def chat(user_message: str, max_tool_rounds: int = 5) -> str:
    emergency = detect_emergency(user_message)
    if emergency:
        return emergency["message"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for _ in range(max_tool_rounds):
        response = _client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOLS,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content

        messages.append(msg)
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            fn = TOOL_DISPATCH.get(fn_name)
            result = fn(**fn_args) if fn else {"error": f"unknown tool {fn_name}"}
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

    return "죄송합니다, 답변을 생성하는 데 문제가 발생했습니다."
