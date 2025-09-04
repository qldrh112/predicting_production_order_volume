import streamlit as st

def render_ui():
    st.sidebar.header("데이터 입력")
    uploaded_file = st.sidebar.file_uploader("엑셀 파일 업로드", type=["xlsx"])
    user_prompt = st.text_area("분석 요청 입력", placeholder="예: 지난달 생산량 트렌드를 분석해줘")

    return uploaded_file, user_prompt
