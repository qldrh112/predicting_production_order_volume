import streamlit as st
from app.data_handler import handle_uploaded_file
from app.ui import render_ui
from app.workflow import run_workflow

def main():
    st.title("📊 생산 데이터 분석 AI Agent")

    # UI (파일 업로드 + 프롬프트 입력)
    uploaded_file, user_prompt = render_ui()

    if uploaded_file and user_prompt:
        # 데이터 처리
        df = handle_uploaded_file(uploaded_file)

        # 워크플로우 실행
        result = run_workflow(user_prompt, df)

        # 결과 표시
        st.subheader("🔎 분석 결과")
        st.write(result)

if __name__ == "__main__":
    main()
