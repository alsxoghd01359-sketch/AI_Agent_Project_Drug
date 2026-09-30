"""응급 신호 감지 — LLM 판단이 아니라 코드가 먼저 가로챈다.

정보 조회(병용확인/성분조회 등)로 들어가기 전에 반드시 이 함수를 먼저 거쳐야 한다.
감지되면 즉시 고정된 응급 안내 메시지를 반환하고, 그 뒤로는 DUR 조회든 뭐든
넘어가지 않고 응답을 종료한다.

주의: 개별 단어("어지럽다" 등)만으로 판단하면 "이 약 먹으면 어지러울 수 있나요?" 같은
일반 정보성 질문까지 오탐하므로, 실제로 "지금 증상을 겪고 있다"는 뉘앙스의
구체적인 문구(예: "숨을 못 쉬겠어요", "한꺼번에 다 먹었어요")로 패턴을 잡는다.
"""
import re

EMERGENCY_PATTERNS = {
    "호흡곤란": [r"숨[을이]?\s*(못|안)\s*쉬", r"호흡\s*곤란", r"숨이?\s*막혀", r"숨쉬기\s*힘들"],
    "의식저하": [r"의식이?\s*없", r"정신을?\s*잃", r"쓰러졌", r"기절"],
    "경련·발작": [r"경련", r"발작"],
    "과다복용": [
        r"많이\s*먹었", r"한꺼번에\s*(다\s*)?먹었", r"다\s*먹어버렸",
        r"약을?\s*왕창", r"과다\s*복용", r"한\s*통을?\s*다\s*먹",
    ],
    "심한알레르기반응": [
        r"목이?\s*부어", r"얼굴이?\s*부어", r"입술이?\s*부어",
        r"두드러기.{0,10}(전신|온몸)", r"쇼크",
    ],
    "출혈": [r"피를?\s*토", r"토혈", r"코피.{0,5}(멈추지|안\s*멈춰)"],
    "자해위험": [r"죽고\s*싶", r"자살", r"극단적\s*선택", r"목숨을?\s*끊"],
}

_GENERAL_MESSAGE = (
    "지금은 정보 조회보다 즉시 대응이 우선입니다.\n"
    "증상이 심하거나 악화되면 119에 연락하거나 가까운 응급실로 가세요.\n"
    "119 신고 시 어떤 약을 얼마나 먹었는지 말씀하시면 중독·과다복용 관련 대응 안내도 받을 수 있습니다.\n"
    "복용한 약의 포장이나 설명서를 챙겨서 병원에 함께 가져가세요."
)

_SELF_HARM_MESSAGE = (
    "지금 많이 힘드신 것 같습니다. 혼자 견디지 않으셔도 됩니다.\n"
    "생명이 위급한 상황이면 119에 즉시 연락하세요.\n"
    "자살예방상담전화 1393(24시간), 정신건강상담전화 1577-0199로도 연결하실 수 있습니다."
)

_CATEGORY_MESSAGE = {"자해위험": _SELF_HARM_MESSAGE}


def detect_emergency(text: str):
    """감지되면 {"category":, "matched_pattern":, "message":} 반환, 아니면 None.

    "자해위험"은 다른 카테고리(과다복용 등)와 같이 언급되는 경우가 많고
    (예: "죽고 싶어요 약을 많이 먹었어요"), 이때는 반드시 자살예방상담전화
    안내가 빠지면 안 되므로 다른 카테고리보다 먼저 확인한다.
    """
    for pat in EMERGENCY_PATTERNS["자해위험"]:
        if re.search(pat, text):
            return {"category": "자해위험", "matched_pattern": pat, "message": _SELF_HARM_MESSAGE}

    for category, patterns in EMERGENCY_PATTERNS.items():
        if category == "자해위험":
            continue
        for pat in patterns:
            if re.search(pat, text):
                return {
                    "category": category,
                    "matched_pattern": pat,
                    "message": _CATEGORY_MESSAGE.get(category, _GENERAL_MESSAGE),
                }
    return None
