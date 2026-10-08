"""AI 복약 정보 도우미 채팅 UI.

실행: 프로젝트 루트(C:\\AI_agent_project_2)에서
    streamlit run src/streamlit_app.py
데이터 경로가 상대경로("data/...")라서 반드시 프로젝트 루트에서 실행해야 한다.
"""
import streamlit as st

from orchestrator import Conversation, warmup

st.set_page_config(page_title="AI 복약 정보 도우미", page_icon="💊")


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

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        if msg.get("footer"):
            # 상담 권유·전문의약품 안내·데이터 출처처럼 거의 고정된 문구는 본문보다
            # 작고 흐린 글씨(st.caption)로 따로 보여준다.
            st.caption(msg["footer"])

user_input = st.chat_input("약에 대해 궁금한 점을 물어보세요")
if user_input:
    st.session_state.history.append({"role": "user", "text": user_input, "footer": None})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("데이터를 조회하는 중입니다..."):
            reply = st.session_state.conversation.send(user_input)
        st.markdown(reply["text"])
        if reply.get("footer"):
            st.caption(reply["footer"])
    st.session_state.history.append(
        {"role": "assistant", "text": reply["text"], "footer": reply.get("footer")}
    )
