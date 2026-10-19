"""AI 복약 정보 도우미 채팅 UI.

실행: 프로젝트 루트(C:\\AI_agent_project_2)에서
    streamlit run src/streamlit_app.py
데이터 경로가 상대경로("data/...")라서 반드시 프로젝트 루트에서 실행해야 한다.
"""
import time

import streamlit as st

from orchestrator import Conversation

st.set_page_config(page_title="AI 복약 정보 도우미", page_icon="💊")

# 배경·포인트 색(진한 티얼~블루 그라데이션)은 로딩 화면과 채팅 화면 둘 다에서
# 똑같이 쓴다 — 로딩 화면에서 봤던 색이 채팅 화면으로 넘어갈 때 안 바뀌어야
# "같은 앱"이라는 느낌이 유지된다.
_BG_DARK = "#050a09"
_BG_RADIAL = "#0d1f1c"
_RING_START = "#2F8FFF"
_RING_END = "#30E3B3"
_TEXT = "#EAF6F3"

# 이모지 대신 로딩 화면과 같은 스타일(알약/사람 실루엣)로 직접 그린 SVG를 쓴다.
# src/assets/에 있고, 앱은 프로젝트 루트에서 실행되므로 이 상대경로로 찾는다.
_USER_AVATAR = "src/assets/user_avatar.svg"
_ASSISTANT_AVATAR = "src/assets/pill_avatar.svg"

_SHARED_CSS = f"""
<style>
.stApp {{
    background: radial-gradient(circle at 50% 35%, {_BG_RADIAL} 0%, {_BG_DARK} 70%);
}}
/* 채팅 말풍선: AI 쪽은 배경 없이(원래 Streamlit 느낌 그대로), 사용자 쪽만
   은은한 포인트색 배경으로 구분한다. 아바타 testid(stChatMessageAvatarUser 등)는
   커스텀 이미지 아바타를 쓰면 안 붙는다는 걸 실측으로 확인했다(기본 이모지
   아바타 전용) — 대신 내용 쪽에 항상 붙는 aria-label(예: "Chat message from
   user")로 부모(stChatMessage)를 고른다. 이것도 Streamlit 내부 DOM이라 버전이
   바뀌면 깨질 수 있지만, 셀렉터가 안 맞아도 이 블록만 무시되고 기본 모양으로
   보일 뿐이다. */
[data-testid="stChatMessage"] {{
    border-radius: 16px;
    padding: 4px 6px;
    margin-bottom: 4px;
}}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {{
    background: #13293f;
    border: 1px solid {_RING_START}55;
    border-right: 3px solid {_RING_START};
}}
</style>
"""
st.markdown(_SHARED_CSS, unsafe_allow_html=True)


def _md_safe(text):
    """조회된 데이터 원문의 ~(물결, "1~2정"처럼 범위 표시)·`(백틱)·_(밑줄)이 마크다운
    문법(취소선·코드·기울임)으로 잘못 해석되는 걸 막는다. "1~2정"이 "1"에 취소선이
    그어진 것처럼 보이는 문제가 실제로 있었다. 우리가 직접 쓰는 서식(글머리표 "- ",
    굵게 "**..**", "[...]" 라벨)에는 이 세 글자를 쓰지 않으므로 그대로 이스케이프해도
    안전하다."""
    for ch in ("\\", "`", "_", "~"):
        text = text.replace(ch, "\\" + ch)
    return text


# SVG 원형 링(stroke-dasharray/dashoffset 방식) 치수. conic-gradient 방식은 끝이
# 각지게 잘려서(stroke-linecap 같은 게 없음) 사용자가 지적한 "부자연스러움"의
# 원인이었다 — SVG circle + stroke-linecap:round로 바꾸면 끝이 둥글게 마감된다.
_RING_R = 92
_RING_CIRC = 2 * 3.141592653589793 * _RING_R  # 둘레(=stroke-dasharray 전체 길이)


def _ring_svg(offset_expr: str) -> str:
    """offset_expr: stroke-dashoffset에 들어갈 CSS 값(고정값 또는 keyframe 변수명).

    모든 줄을 들여쓰기 없이(칸 0부터) 이어 붙인다 — st.markdown은 내부적으로
    마크다운 파서를 거치는데, 줄 앞에 공백이 4칸 이상 있으면 HTML이 아니라
    "들여쓰기된 코드 블록"으로 해석해서 태그가 그대로 글자로 보이는 문제가
    실제로 있었다(알약 div가 텍스트로 그대로 노출된 원인)."""
    return (
        f'<svg width="220" height="220" viewBox="0 0 220 220" style="transform: rotate(-90deg);">'
        f'<defs><linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
        f'<stop offset="0%" stop-color="{_RING_START}" />'
        f'<stop offset="100%" stop-color="{_RING_END}" />'
        f'</linearGradient></defs>'
        f'<circle cx="110" cy="110" r="{_RING_R}" fill="none" stroke="#ffffff14" stroke-width="16" />'
        f'<circle class="ring-progress" cx="110" cy="110" r="{_RING_R}" fill="none" stroke="url(#ringGrad)" '
        f'stroke-width="16" stroke-linecap="round" '
        f'stroke-dasharray="{_RING_CIRC:.2f}" stroke-dashoffset="{offset_expr}" />'
        f'</svg>'
    )


def _loading_html(phase: str) -> str:
    """로딩 화면 한 단계를 그린다.

    phase="fill": 실제 로딩이 정확히 몇 초 걸릴지는 알 수 없으니, 약 20초에 걸쳐
    95%까지만 서서히(ease-out) 채우고 거기서 멈춘다 — 100%까지 미리 채워버리면
    "다 끝났다"는 거짓 신호가 되기 때문에 95%를 상한으로 둔다.
    phase="done": 실제로 전부 끝난 시점에 95% -> 100%를 짧게(0.5초) 채워서
    "방금 끝났다"는 걸 보여준 뒤 채팅 화면으로 넘어간다.
    """
    if phase == "fill":
        start_offset = _RING_CIRC
        end_offset = _RING_CIRC * 0.05  # 95% 채움
        duration = "20s"
        timing = "cubic-bezier(0.15, 0.6, 0.25, 1)"  # 갈수록 느려짐(ease-out), 자연스러운 감속
    else:
        start_offset = _RING_CIRC * 0.05
        end_offset = 0  # 100% 채움
        duration = "0.7s"
        # easeOutExpo류 곡선: 처음엔 휙 빠르게 움직이다 끝에서 부드럽게 멎는다.
        # "ease-out"(선형에 가까움)이었을 땐 거의 순간이동처럼 "띡" 채워지는
        # 느낌이었는데, 이 곡선은 움직임 자체가 눈에 보이면서도 급정거하지 않는다.
        timing = "cubic-bezier(0.16, 1, 0.3, 1)"

    return f"""
<style>
.loading-wrap {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 70vh;
}}
.ring-box {{
    position: relative;
    width: 220px;
    height: 220px;
    filter: drop-shadow(0 0 14px {_RING_END}55);
}}
.ring-progress {{
    animation: fillRing {duration} {timing} forwards;
}}
@keyframes fillRing {{
    0%   {{ stroke-dashoffset: {start_offset:.2f}; }}
    100% {{ stroke-dashoffset: {end_offset:.2f}; }}
}}
.pill-overlay {{
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}}
.pill {{
    width: 72px;
    height: 34px;
    border-radius: 17px;
    background: linear-gradient(135deg, {_RING_END} 0%, {_RING_END} 48%, #ffffff 52%, #ffffff 100%);
    box-shadow: 0 0 10px {_RING_END}88;
    transform: rotate(-45deg);
}}
.loading-text {{
    margin-top: 28px;
    color: {_TEXT};
    font-size: 1.05rem;
    letter-spacing: 0.02em;
}}
.dots span {{
    display: inline-block;
    opacity: 0;
    animation: dotfade 1.4s infinite;
}}
.dots span:nth-child(1) {{ animation-delay: 0s; }}
.dots span:nth-child(2) {{ animation-delay: 0.25s; }}
.dots span:nth-child(3) {{ animation-delay: 0.5s; }}
@keyframes dotfade {{
    0%, 80%, 100% {{ opacity: 0; }}
    35% {{ opacity: 1; }}
}}
</style>
<div class="loading-wrap"><div class="ring-box">{_ring_svg(f"{start_offset:.2f}")}<div class="pill-overlay"><div class="pill"></div></div></div><div class="loading-text">의약품 정보를 불러오고 있습니다<span class="dots"><span>.</span><span>.</span><span>.</span></span></div></div>
"""


if "app_ready" not in st.session_state:
    st.session_state.app_ready = False

if not st.session_state.app_ready:
    # drug_lookup/paragraph_search/name_match의 지연 로딩 데이터를 여기서 미리
    # 불러온다(실측 약 27초). orchestrator.warmup()과 같은 일을 하지만, 로딩
    # 화면을 띄운 채로 불러와야 해서 그 안의 세 호출을 여기서 직접 부른다.
    from drug_lookup import _ensure_loaded as _load_drug_lookup
    from paragraph_search import _ensure_detail_loaded as _load_paragraph_detail
    from name_match import _load_index as _load_name_index

    placeholder = st.empty()

    placeholder.markdown(_loading_html("fill"), unsafe_allow_html=True)
    _load_drug_lookup()
    _load_paragraph_detail()
    _load_name_index()

    placeholder.markdown(_loading_html("done"), unsafe_allow_html=True)
    time.sleep(0.5)  # 100% 채워진 모습을 짧게라도 보여준 뒤 채팅 화면으로 넘어간다

    st.session_state.app_ready = True
    st.rerun()

st.title("AI 복약 정보 도우미")
st.caption(
    "식품의약품안전처 공공데이터를 조회해서 설명합니다. 의학적 진단·처방을 대체하지 않으며, "
    "최종 판단은 의사나 약사에게 있습니다. 응급 상황이면 지금 바로 119에 연락하세요."
)

if "conversation" not in st.session_state:
    st.session_state.conversation = Conversation()
if "history" not in st.session_state:
    st.session_state.history = []  # [{"role": "user"|"assistant", "text": str, "footer": str|None}]

# st.chat_input은 스크립트에서 어디에 적든 항상 화면 맨 아래에 고정 렌더링된다
# (Streamlit의 특수 동작). 그래서 여기서 먼저 입력값만 받아두고, 화면에 보여줄
# 내용(대화 내역 + 안내문)은 이 아래에서 순서대로 쌓는다 — 안내문이 chat_input보다
# 먼저 호출돼도, 실제로는 입력창이 항상 그 아래(화면 맨 아래)에 고정된다.
user_input = st.chat_input("약에 대해 궁금한 점을 물어보세요")

for msg in st.session_state.history:
    avatar = _USER_AVATAR if msg["role"] == "user" else _ASSISTANT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(_md_safe(msg["text"]))
        if msg.get("footer"):
            # 전문의약품 안내·병용금기 등록 안내·데이터 출처처럼 그 답변에만 해당하는
            # 부가 정보는 본문보다 작고 흐린 글씨(st.caption)로 따로 보여준다.
            st.caption(_md_safe(msg["footer"]))

if user_input:
    st.session_state.history.append({"role": "user", "text": user_input, "footer": None})
    with st.chat_message("user", avatar=_USER_AVATAR):
        st.markdown(_md_safe(user_input))

    with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
        with st.spinner("데이터를 조회하는 중입니다..."):
            reply = st.session_state.conversation.send(user_input)
        st.markdown(_md_safe(reply["text"]))
        if reply.get("footer"):
            st.caption(_md_safe(reply["footer"]))
    st.session_state.history.append(
        {"role": "assistant", "text": reply["text"], "footer": reply.get("footer")}
    )

# 상담 권유를 답변마다 반복하는 대신, 대화 내역 맨 아래(= 질문·답변이 쌓일수록 계속
# 밀려 내려가는 위치)에 고정 안내문으로 한 번만 보여준다.
st.divider()
st.caption("AI 복약 정보 도우미는 의사가 아닙니다. 정확한 진단과 처방은 의사나 약사와 상담하세요.")
