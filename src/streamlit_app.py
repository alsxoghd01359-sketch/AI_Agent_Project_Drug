"""AI 복약 정보 도우미 채팅 UI.

실행: 프로젝트 루트(C:\\AI_agent_project_2)에서
    streamlit run src/streamlit_app.py
데이터 경로가 상대경로("data/...")라서 반드시 프로젝트 루트에서 실행해야 한다.
"""
import streamlit as st

from orchestrator import Conversation, warmup

st.set_page_config(page_title="AI 복약 정보 도우미", page_icon="💊")


def _md_safe(text):
    """조회된 데이터 원문의 ~(물결, "1~2정"처럼 범위 표시)·`(백틱)·_(밑줄)이 마크다운
    문법(취소선·코드·기울임)으로 잘못 해석되는 걸 막는다. "1~2정"이 "1"에 취소선이
    그어진 것처럼 보이는 문제가 실제로 있었다. 우리가 직접 쓰는 서식(글머리표 "- ",
    굵게 "**..**", "[...]" 라벨)에는 이 세 글자를 쓰지 않으므로 그대로 이스케이프해도
    안전하다."""
    for ch in ("\\", "`", "_", "~"):
        text = text.replace(ch, "\\" + ch)
    return text


@st.cache_resource
def _warmup():
    # drug_lookup/paragraph_search/name_match의 지연 로딩 데이터를 앱 시작 시 한 번만
    # 불러온다(실측 약 20초). st.cache_resource로 감싸서 Streamlit이 스크립트를 다시
    # 실행할 때마다(위젯 조작 등) 매번 다시 부르지 않게 한다.
    warmup()
    return True


_warmup()

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
    with st.chat_message(msg["role"]):
        st.markdown(_md_safe(msg["text"]))
        if msg.get("footer"):
            # 전문의약품 안내·병용금기 등록 안내·데이터 출처처럼 그 답변에만 해당하는
            # 부가 정보는 본문보다 작고 흐린 글씨(st.caption)로 따로 보여준다.
            st.caption(_md_safe(msg["footer"]))

if user_input:
    st.session_state.history.append({"role": "user", "text": user_input, "footer": None})
    with st.chat_message("user"):
        st.markdown(_md_safe(user_input))

    with st.chat_message("assistant"):
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
