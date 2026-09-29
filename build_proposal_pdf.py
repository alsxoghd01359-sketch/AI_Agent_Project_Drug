# -*- coding: utf-8 -*-
"""AI_복약_정보_도우미_제안서.pdf 수정본 생성.

원본 대비 수정 사항:
1. 시나리오1 예시: 와파린+이부프로펜(실제 DUR 데이터에 없음) -> 이트라코나졸+심바스타틴(실제 등록된 사례)
2. "병용주의" -> "병용금기" (실제 DUR 유형 명칭에 맞게 수정)
3. "한국중독정보센터(1339)" 삭제 (2012년 119로 통합되어 현재 미운영) -> 119로 통일
4. "API 5종" 표현에 각주 추가: 실제로는 승인 서비스 4개(허가정보+DUR가 한 서비스, 호출한도 공유)
5. 시스템 구성도에 사전 수집/로컬 DB 적재 단계 추가 (실시간 호출로는 호출한도 초과)
6. 리스크 #4: ITEM_SEQ가 4개 카테고리에서 공통 식별자로 쓰임을 확인 -> 심각도 하향, 내용 갱신
7. 리스크 #9: 건강기능식품 INTAKE_HINT1에 실제 의약품 상호작용 경고 문구 확인됨 -> 내용 갱신
8. API 개요 표: e약은요의 상호작용(intrcQesitm) 필드 명시
9. 개발 단계에 "사전 데이터 수집·로컬 DB 구축" 단계 추가
"""
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
)

# CID 표준폰트(HYSMyeongJo 등)는 뷰어의 폰트 대체에 의존하는데, 이 환경의 렌더러가
# 큰 글자(제목/헤더)에서 대체를 제대로 못 해 글자가 안 보이는 문제가 있었다.
# 실제 트루타입 폰트 파일을 PDF에 직접 임베드해서 렌더러에 상관없이 보이도록 한다.
pdfmetrics.registerFont(TTFont("Malgun", r"C:\Windows\Fonts\malgun.ttf"))
pdfmetrics.registerFont(TTFont("Malgun-Bold", r"C:\Windows\Fonts\malgunbd.ttf"))
pdfmetrics.registerFontFamily("Malgun", normal="Malgun", bold="Malgun-Bold")

FONT_BOLD = "Malgun-Bold"
FONT_REGULAR = "Malgun"
EMPH_COLOR = "#1F4E79"


def fix_emphasis(text: str) -> str:
    return text

NAVY = colors.HexColor("#1F4E79")
LIGHT_BLUE = colors.HexColor("#DCE6F1")
GRAY_TEXT = colors.HexColor("#444444")
RED_BG = colors.HexColor("#FBE4E4")
GREEN_BG = colors.HexColor("#E6F4E6")
RED_TEXT = colors.HexColor("#B00000")
GREEN_TEXT = colors.HexColor("#1B7A1B")

styles = {
    "title": ParagraphStyle("title", fontName=FONT_BOLD, fontSize=22, leading=28,
                             textColor=NAVY, alignment=1, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", fontName=FONT_BOLD, fontSize=13, leading=18,
                                textColor=NAVY, alignment=1, spaceAfter=2),
    "supersub": ParagraphStyle("supersub", fontName=FONT_REGULAR, fontSize=9.5, leading=13,
                                textColor=GRAY_TEXT, alignment=1, spaceAfter=14),
    "kicker": ParagraphStyle("kicker", fontName=FONT_REGULAR, fontSize=9.5, leading=13,
                              textColor=GRAY_TEXT, alignment=1, spaceAfter=4),
    "h2": ParagraphStyle("h2", fontName=FONT_BOLD, fontSize=14.5, leading=20,
                          textColor=NAVY, spaceBefore=16, spaceAfter=8),
    "h3": ParagraphStyle("h3", fontName=FONT_BOLD, fontSize=11.5, leading=16,
                          textColor=colors.HexColor("#222222"), spaceBefore=10, spaceAfter=6),
    "body": ParagraphStyle("body", fontName=FONT_REGULAR, fontSize=9.7, leading=15,
                            textColor=colors.HexColor("#222222")),
    "bodyBold": ParagraphStyle("bodyBold", fontName=FONT_BOLD, fontSize=9.7, leading=15,
                                textColor=colors.HexColor("#222222")),
    "bullet": ParagraphStyle("bullet", fontName=FONT_REGULAR, fontSize=9.7, leading=15,
                              textColor=colors.HexColor("#222222"), leftIndent=12,
                              bulletIndent=0, spaceAfter=4),
    "cell": ParagraphStyle("cell", fontName=FONT_REGULAR, fontSize=9, leading=13.5,
                            textColor=colors.HexColor("#222222")),
    "cellBold": ParagraphStyle("cellBold", fontName=FONT_BOLD, fontSize=9, leading=13.5,
                                textColor=colors.HexColor("#222222")),
    "cellHead": ParagraphStyle("cellHead", fontName=FONT_BOLD, fontSize=9.3, leading=13,
                                textColor=colors.white),
    "note": ParagraphStyle("note", fontName=FONT_REGULAR, fontSize=8.6, leading=13,
                            textColor=colors.HexColor("#555555"), leftIndent=6,
                            borderColor=colors.HexColor("#AAAAAA")),
    "scenarioLabelBad": ParagraphStyle("scenarioLabelBad", fontName=FONT_BOLD, fontSize=9.7,
                                        leading=14, textColor=RED_TEXT),
    "scenarioLabelGood": ParagraphStyle("scenarioLabelGood", fontName=FONT_BOLD, fontSize=9.7,
                                         leading=14, textColor=GREEN_TEXT),
    "scenarioBad": ParagraphStyle("scenarioBad", fontName=FONT_REGULAR, fontSize=9.5,
                                   leading=14.5, textColor=colors.HexColor("#7a1f1f")),
    "scenarioGood": ParagraphStyle("scenarioGood", fontName=FONT_REGULAR, fontSize=9.5,
                                    leading=14.5, textColor=colors.HexColor("#1f5c2a")),
    "mono": ParagraphStyle("mono", fontName=FONT_REGULAR, fontSize=9, leading=14,
                            textColor=colors.HexColor("#222222"), backColor=colors.HexColor("#F2F2F2")),
}


def hr():
    return HRFlowable(width="100%", thickness=1.1, color=NAVY, spaceAfter=10)


def p(text, style="body"):
    return Paragraph(fix_emphasis(text), styles[style])


def make_table(header, rows, col_widths, header_bg=NAVY):
    data = [[Paragraph(fix_emphasis(h), styles["cellHead"]) for h in header]]
    for row in rows:
        data.append([Paragraph(fix_emphasis(str(c)), styles["cell"]) for c in row])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#B9C6D6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FC")]),
    ]))
    return t


def scenario_box(label, body_text, good: bool):
    bg = GREEN_BG if good else RED_BG
    mark = "✓ 올바른 답" if good else "✗ 잘못된 답"
    lbl_style = "scenarioLabelGood" if good else "scenarioLabelBad"
    txt_style = "scenarioGood" if good else "scenarioBad"
    cell = [Paragraph(mark, styles[lbl_style]), Spacer(1, 3),
            Paragraph(fix_emphasis(body_text), styles[txt_style])]
    t = Table([[cell]], colWidths=[160 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 0.6, bg),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


story = []

# ---------------- 표지/개요 ----------------
story.append(Spacer(1, 6))
story.append(p("프로젝트 제안서", "kicker"))
story.append(p("AI 복약 정보 도우미", "title"))
story.append(p("공공 의약품 데이터 기반의 안전한 복약 정보 질의응답", "subtitle"))
story.append(p("식품의약품안전처 공공데이터 활용 (정보 카테고리 5종 · 승인 서비스 4개)", "supersub"))

story.append(p("1. 프로젝트 개요", "h2"))
story.append(hr())
story.append(p(
    "사용자가 복용 중인 약의 정보를 묻거나, 여러 약을 함께 복용했을 때 문제가 없는지 자연어로 질의할 수 "
    "있는 AI 서비스를 만든다. 식품의약품안전처가 제공하는 공공데이터를 데이터 기반으로 삼고, "
    "LLM은 데이터를 찾아 설명하는 역할만 하며 의학적 판단은 하지 않는다.", "body"))
story.append(Spacer(1, 8))

api_header = ["활용 API", "제공 정보"]
api_rows = [
    ["의약품 제품 허가정보", "제품명, 성분, 효능·효과, 허가 상태 등 기본 정보"],
    ["의약품안전사용서비스(DUR) 품목정보",
     "병용금기, 특정연령대금기, 임부금기, 용량주의 등 상호작용 정보<br/>"
     "※ 의약품 제품 허가정보와 같은 서비스(base URL) 소속 — 일일 호출 한도를 공유함"],
    ["의약품 낱알식별 정보", "알약의 색상·모양·각인 등을 통한 식별 정보"],
    ["의약품개요정보(e약은요)",
     "일반인 대상 복약 안내(효능, 용법, 주의사항)와 별도의 상호작용(intrcQesitm) 자연어 설명 제공 — "
     "DUR에 등록되지 않은 조합에 대한 보조 근거로 활용 가능"],
    ["건강기능식품정보",
     "건강기능식품의 기능성 원료, 섭취 관련 정보. 섭취 시 주의사항 필드에 의약품 병용 경고 문구가 "
     "포함된 사례를 실제 데이터에서 확인함"],
]
story.append(make_table(api_header, api_rows, [55 * mm, 105 * mm]))
story.append(Spacer(1, 6))
story.append(p(
    "※ 실제 API 활용신청 스펙을 확인한 결과, <b>의약품 제품 허가정보와 DUR 품목정보는 하나의 서비스 "
    "단위(활용신청)에 속한 오퍼레이션들</b>이다. 즉 정보 카테고리는 5종이지만 실제 승인 서비스는 4개이며, "
    "허가정보와 DUR은 일일 호출 한도(10,000건)를 서로 공유한다.", "note"))

story.append(Spacer(1, 10))
story.append(p("2. 서비스 취약점 분석", "h2"))
story.append(hr())

vuln_header = ["#", "취약점", "심각도", "설명"]
vuln_rows = [
    ["1", "\"정보 없음\"의 안전 오인", "최고",
     "DUR 데이터는 알려진·문서화된 상호작용만 담고 있다. 조회에 걸리지 않았다고 실제로 안전하다는 "
     "보장은 없다. 이를 \"안전함\"으로 표현하면 가장 위험한 오해를 낳는다."],
    ["2", "LLM의 일반화·추론 오류", "최고",
     "LLM이 조회된 사실을 넘어 \"비슷한 계열이니 괜찮을 것\" 같은 유추를 하면 근거 없는 판단이 된다. "
     "LLM의 역할은 조회 결과를 설명하는 것으로 엄격히 제한해야 한다."],
    ["3", "응급 상황 미대응", "높음",
     "과다복용, 급성 부작용 증상 등은 정보 조회가 아니라 즉시 응급 대응 안내가 필요한 상황이다. "
     "이 분기가 없으면 위험하다."],
    ["4", "API 간 식별자 불일치", "중간",
     "<b>[실측 결과로 갱신]</b> ITEM_SEQ가 의약품 제품허가정보·DUR·e약은요·낱알식별정보 4개 카테고리에서 "
     "공통 식별자로 쓰이는 것을 확인했다(건강기능식품만 별도 체계인 STTEMNT_NO 사용). 우려했던 코드 "
     "불일치 문제는 크지 않다. 다만 동명이인 약(성분·용량이 다른 동일 제품명)을 잘못 매칭하는 문제는 "
     "여전히 유효하므로 제품명만이 아니라 성분·함량까지 함께 확인해야 한다."],
    ["5", "범위 확장(의료 상담화)", "높음",
     "용법·용량 조언, 증상 기반 판단, 복용 중단 권유 등으로 범위가 넓어지면 사실상 의료 상담이 된다. "
     "범위를 공식 데이터 조회·설명으로 좁혀야 한다."],
    ["6", "낱알식별 오인식", "중간",
     "사진이나 설명 기반 식별은 오인식 가능성이 있고, 오인식이 그대로 잘못된 상호작용 정보로 이어진다. "
     "후보를 여러 개 제시하고 약사 확인을 안내하는 보조 기능으로만 제한한다."],
    ["7", "민감 개인정보 처리", "중간",
     "복용 중인 약 목록은 민감한 건강정보다. 저장 여부·기간·처리방침을 명시해야 한다."],
    ["8", "법적·규제 측면", "중간",
     "의약품 정보 제공은 약사법·의료법과 맞닿아 있다. 진단·처방을 대체하지 않는다는 고지가 필요하며, "
     "정도에 따라 법적 검토가 필요할 수 있다. (법률 전문가 검토 별도 필요)"],
    ["9", "API 자체의 한계", "중간",
     "<b>[실측 결과로 갱신]</b> 의약품 제품허가정보와 DUR 품목정보가 같은 서비스로 묶여 일일 10,000건을 "
     "공유한다(실측: 병용금기 전체 79만 6천여 건 수집에만 약 1,600회 호출 소요). 데이터 갱신 주기(리콜·"
     "판매중지 반영 여부)는 별도 확인이 필요하다. 건강기능식품-의약품 상호작용 데이터는 우려와 달리 "
     "INTAKE_HINT1 필드에 실제 경고 문구가 포함된 사례를 확인했다(예: \"의약품(당뇨치료제, "
     "혈액항응고제) 복용 시 섭취에 주의\")."],
]
story.append(make_table(vuln_header, vuln_rows, [8 * mm, 32 * mm, 15 * mm, 105 * mm]))

story.append(PageBreak())

# ---------------- 응답 설계 원칙 ----------------
story.append(p("3. 응답 설계 원칙", "h2"))
story.append(hr())
story.append(p(
    "시스템 프롬프트에 \"의사와 상담하라\"는 면책 문구를 넣는 것만으로는 충분하지 않다. 프롬프트는 강한 "
    "가이드라인이지 강제 규칙이 아니라서, LLM이 확신에 찬 어조로 답할 가능성이 여전히 남는다. 또한 "
    "답을 \"두루뭉술하게\" 흐리는 것도 해법이 아니다. 모호함은 오해를 낳는다. <b>필요한 것은 애매함이 아니라, "
    "데이터에 있는 것과 없는 것을 정확히 구분해서 말하는 것이다.</b>", "body"))

story.append(p("3.1 답변 구조 고정", "h3"))
bullets = [
    "모든 답변은 <b>[조회된 사실] → [데이터의 한계 고지] → [전문가 상담 권유]</b> 순서를 지킨다. "
    "LLM의 자유 서술에 맡기면 이 구조가 누락될 수 있으므로 템플릿으로 강제한다.",
    "<b>확신 표현을 금지어로 처리한다.</b> \"안전합니다\", \"괜찮습니다\", \"문제없습니다\" 같은 단정적 "
    "표현 대신 \"등록된 정보가 없습니다\"처럼 사실 서술형만 사용한다.",
    "응급 신호는 LLM 판단이 아니라 코드가 먼저 가로챈다. \"어지럽다\", \"숨쉬기 힘들다\" 같은 키워드가 "
    "감지되면 조회 이전에 응급 대응 안내를 고정 출력하고 종료한다.",
    "매 답변에 <b>데이터 출처와 갱신일</b>을 표시해 정보의 신뢰 근거를 명확히 한다.",
]
for b in bullets:
    story.append(Paragraph(fix_emphasis("• " + b), styles["bullet"]))

story.append(PageBreak())

# ---------------- 시나리오 ----------------
story.append(p("4. 응답 시나리오 예시", "h2"))
story.append(hr())
story.append(p(
    "아래 네 가지는 답변 설계의 기준이 되는 대표 시나리오다. 실제 구현 후 이 패턴을 지키는지 평가셋으로 "
    "검증한다.", "body"))
story.append(Spacer(1, 6))

story.append(p("시나리오 1. DUR에 병용금기가 등록된 경우", "h3"))
story.append(scenario_box("bad", "\"음, 그 두 약은 같이 먹으면 조금 주의가 필요할 수도 있어요. "
                                  "확실친 않지만 조심하시는 게 좋을 것 같아요.\"", good=False))
story.append(Spacer(1, 6))
story.append(scenario_box("good",
    "DUR(의약품안전사용서비스) 데이터에 <b>이트라코나졸과 심바스타틴의 병용금기</b>가 등록되어 있습니다.<br/>"
    "등록된 사유: <b>횡문근융해증</b> 위험 (DUR 데이터 기준)<br/>"
    "이건 데이터베이스에 등록된 정보를 그대로 전달한 것이며, 실제로 복용해도 되는지는 현재 건강 상태에 "
    "따라 다릅니다. 처방한 의사나 약사에게 반드시 확인하세요.", good=True))
story.append(p("(※ 실제 DUR 병용금기 데이터로 검증된 사례로 교체함 — 원안의 와파린·이부프로펜 조합은 "
                "실제 병용금기 데이터셋 79만여 건 중 단 한 건도 없어 예시로 부적절함)", "note"))

story.append(Spacer(1, 10))
story.append(p("시나리오 2. DUR에 아무것도 걸리지 않은 경우 (가장 중요)", "h3"))
story.append(scenario_box("bad", "\"타이레놀과 오메가3는 특별히 문제가 되는 조합은 아니에요. "
                                  "편하게 드셔도 될 것 같습니다.\"", good=False))
story.append(Spacer(1, 6))
story.append(scenario_box("good",
    "DUR 데이터베이스에서 조회한 결과, 타이레놀(성분: 아세트아미노펜)과 오메가3 사이에 등록된 "
    "<b>병용금기·병용주의 정보는 없습니다.</b><br/>"
    "다만 이건 \"이 조합이 안전하다고 확인됐다\"는 뜻이 아니라, DUR에 등록된 알려진 위험 사례가 없다는 "
    "뜻입니다. 개인의 질환에 따라 상황이 다를 수 있으니, 정확한 판단은 약사나 의사에게 확인하시길 "
    "권합니다.", good=True))

story.append(Spacer(1, 10))
story.append(p("시나리오 3. 응급 신호가 섞인 경우 (코드가 먼저 가로챔)", "h3"))
story.append(scenario_box("bad", "\"말씀하신 약의 DUR 정보를 조회해보겠습니다...\" (조회 결과 서술로 진행)",
                           good=False))
story.append(Spacer(1, 6))
story.append(scenario_box("good",
    "지금은 정보 조회보다 <b>즉시 대응이 우선</b>입니다.<br/>"
    "증상이 심하거나 악화되면 <b>119</b>에 연락하거나 가까운 응급실로 가세요.<br/>"
    "119 신고 시 어떤 약을 얼마나 먹었는지 말씀하시면 중독·과다복용 관련 대응 안내도 받을 수 있습니다.<br/>"
    "복용한 약의 포장이나 설명서를 챙겨서 병원에 함께 가져가세요.<br/>"
    "(이 뒤에는 DUR 정보 조회로 넘어가지 않고 응답을 종료)", good=True))
story.append(p("(※ 과거 안내되던 \"한국중독정보센터(1339)\"는 2012년 119로 완전히 통합되어 현재 별도로 "
                "운영되지 않는 번호이므로 삭제하고 119로 통일함)", "note"))

story.append(Spacer(1, 10))
story.append(p("시나리오 4. 낱알식별처럼 애초에 확신할 수 없는 요청", "h3"))
story.append(scenario_box("bad", "\"설명하신 특징으로 보면 타이레놀일 가능성이 높아요.\"", good=False))
story.append(Spacer(1, 6))
story.append(scenario_box("good",
    "말씀하신 정도의 정보(색상, 모양)만으로는 정확한 식별이 어렵습니다. 흰색 원형 알약은 매우 많은 "
    "종류가 있어서, 잘못 짚으면 위험할 수 있습니다.<br/>"
    "각인이 흐릿하더라도 사진을 밝은 곳에서 다시 찍어 확인해보시거나, 가까운 약국에 가져가시면 가장 "
    "정확하게 확인할 수 있습니다.", good=True))

story.append(PageBreak())

# ---------------- 핵심 기능 ----------------
story.append(p("5. 핵심 기능", "h2"))
story.append(hr())
features = [
    "<b>약 정보 조회:</b> 제품명으로 성분, 효능·효과, 용법·용량(e약은요 기준) 안내",
    "<b>병용 확인:</b> 복용 중인 약 목록을 입력하면 DUR 데이터로 병용금기·병용주의 여부 조회 및 사실 "
    "서술형 안내",
    "<b>낱알식별(검토 대상):</b> 알약 특징이나 사진으로 식별 지원. 오인식 위험이 있어 단정적 식별 대신 "
    "후보 목록 제시 + 약사 확인 안내로만 제공",
    "<b>건강기능식품 교차 확인:</b> 의약품과 건강기능식품 간 주의사항 안내. 건강기능식품 상세정보의 "
    "섭취 시 주의사항 필드에 의약품 병용 경고가 포함된 사례를 실제로 확인함",
    "<b>응급 대응 분기:</b> 과다복용·급성 증상 키워드 감지 시 정보 조회 없이 응급 안내로 즉시 전환",
]
for f in features:
    story.append(Paragraph(fix_emphasis("• " + f), styles["bullet"]))

story.append(Spacer(1, 8))
story.append(p("6. 시스템 구성(안)", "h2"))
story.append(hr())
diagram = (
    "[사전 배치 작업]<br/>"
    "5종 정보(API 4개 서비스) 전체 데이터 수집 → 구조화 DB(정확한 조회용) + 벡터DB(설명 텍스트 RAG용) "
    "로컬 적재 · 주기적 재수집(예: 월 1회)<br/>"
    "─────────────────────────<br/>"
    "[사용자] ↔ [채팅 UI]<br/>"
    "　↓<br/>"
    "[의도 분류 · 응급 키워드 분기(코드)]<br/>"
    "　↓ (응급 아님)<br/>"
    "[약품명 → 품목코드 매칭(코드, 로컬 DB 조회)]<br/>"
    "　↓<br/>"
    "[로컬 구조화DB/벡터DB 조회 · 결과 취합(코드)]<br/>"
    "　↓<br/>"
    "[LLM: 답변 구조 템플릿에 따라 서술]<br/>"
    "　↓<br/>"
    "[출처·갱신일·전문가 상담 권유 고정 삽입]"
)
story.append(Paragraph(diagram, styles["mono"]))
story.append(Spacer(1, 6))
story.append(p(
    "※ [수정] 사용자 질문마다 5종 API를 실시간 호출하는 구조 대신, <b>사전에 전체 데이터를 로컬 DB에 "
    "적재해두고 조회하는 구조로 변경</b>했다. DUR 병용금기 데이터만 79만 건 이상이라 실시간 호출로는 "
    "허가정보와 공유하는 일일 호출 한도(10,000건)를 금방 소진한다.", "note"))

story.append(Spacer(1, 10))
story.append(p("7. 평가 계획", "h2"))
story.append(hr())
eval_header = ["평가 항목", "방법"]
eval_rows = [
    ["확신 표현 사용 여부", "\"안전합니다\" 등 금지 표현이 답변에 등장하는지 검사"],
    ["정보 없음/안전함 구분", "DUR 미등록 조합 질문에 \"안전하다\"로 오인될 표현을 쓰는지 확인"],
    ["응급 분기 정확도", "응급 신호 포함 질문 세트에서 코드 분기가 제대로 작동하는지"],
    ["품목 매칭 정확도", "동명이인 약 등 매칭 오류 사례로 정확한 코드 매칭 여부 확인"],
    ["전문가 상담 권유 포함", "모든 답변에 상담 권유 문구가 누락 없이 포함되는지"],
]
story.append(make_table(eval_header, eval_rows, [55 * mm, 105 * mm]))

story.append(PageBreak())

# ---------------- 리스크/개발단계/기대효과 ----------------
story.append(p("8. 리스크와 제한", "h2"))
story.append(hr())
risks = [
    "<b>서비스 범위:</b> 공식 데이터 조회·설명으로 한정하며, 용법·용량 조언, 증상 기반 진단, 복용 중단 "
    "권유는 제공하지 않는다.",
    "<b>법적 고지:</b> 이 서비스는 의학적 진단·처방을 대체하지 않으며, 최종 판단은 의료 전문가에게 "
    "있음을 모든 화면에 명시한다.",
    "<b>개인정보:</b> 복용 약물 목록은 민감정보로 취급하며, 저장 여부와 기간을 명확히 정책화한다.",
    "<b>데이터 신뢰성:</b> API의 갱신 주기와 최신성을 확인하고, 오래된 정보에는 경고를 표시한다.",
    "<b>법률 검토:</b> 약사법·의료법 관련 저촉 여부는 법률 전문가 검토가 필요하다. (본 제안서는 법률 "
    "자문이 아님)",
]
for r in risks:
    story.append(Paragraph(fix_emphasis("• " + r), styles["bullet"]))

story.append(Spacer(1, 8))
story.append(p("9. 개발 단계(안)", "h2"))
story.append(hr())
dev_header = ["단계", "내용"]
dev_rows = [
    ["1", "5종 정보(API 4개 서비스) 실제 응답 구조 확인, 전체 데이터 사전 수집 및 로컬 구조화DB·벡터DB 구축"],
    ["2", "품목코드(ITEM_SEQ) 매칭 로직 설계 — 제품명뿐 아니라 성분·함량까지 함께 확인해 동명이인 약 방지"],
    ["3", "응답 구조 템플릿과 금지 표현 필터 구현"],
    ["4", "응급 키워드 분기 로직 구현"],
    ["5", "약 정보 조회·병용 확인 기본 기능 구현"],
    ["6", "평가셋 작성 및 검증(실제 등록 데이터 기반 사례 포함), 개선 반복"],
    ["7", "낱알식별 등 부가 기능 도입 여부 재검토"],
]
story.append(make_table(dev_header, dev_rows, [15 * mm, 145 * mm]))

story.append(Spacer(1, 8))
story.append(p("10. 기대 효과", "h2"))
story.append(hr())
effects = [
    "복약 관련 정보를 자연어로 손쉽게 확인할 수 있다.",
    "데이터의 한계를 정직하게 전달해 잘못된 안전 인식을 방지한다.",
    "안전이 중요한 도메인에서 LLM의 역할을 엄격히 제한하는 설계 경험을 포트폴리오로 보여줄 수 있다.",
]
for e in effects:
    story.append(Paragraph(fix_emphasis("• " + e), styles["bullet"]))

doc = SimpleDocTemplate(
    "AI_복약_정보_도우미_제안서_수정본.pdf",
    pagesize=A4,
    topMargin=20 * mm, bottomMargin=18 * mm, leftMargin=18 * mm, rightMargin=18 * mm,
    title="AI 복약 정보 도우미 - 프로젝트 제안서 (수정본)",
)
doc.build(story)
print("done")
