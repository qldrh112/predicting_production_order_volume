import streamlit as st

def render_ui():
    """
    streamlit 서버를 통해 사용자로부터 xlsx 파일과 프롬프트를 입력받아 시스템 내부로 반환하는 함수
    """
    st.sidebar.header("파일 추가")
    uploaded_file = st.sidebar.file_uploader("xlsx 파일 추가", type=["xlsx"])
    user_prompt = st.text_area("xlsx 파일으로 무엇을 할까요?", placeholder="무엇이든 물어보세요.") #placeholder는 chatgpt를 참고함

    return uploaded_file, user_prompt
