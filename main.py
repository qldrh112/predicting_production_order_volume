import streamlit as st
from app.data_handler import handle_uploaded_xlsx
from app.ui import render_ui
from app.workflow import run_workflow

def main():
    st.title("📊 생산 데이터 분석 AI Agent")

    # UI를 통해 'xlsx' 파일과 사용자의 프롬프트를 전달받음
    uploaded_xlsx, user_prompt = render_ui()

    # 프롬프트만 입력받는 것은 기능상 제한할 것인가??
    if uploaded_xlsx and user_prompt:   # xlsx 파일과 사용자 프롬프트가 존재하면

        # xlsx 파일을 dataframe 형태로 가공하고 중요 데이터 마스킹
        df = handle_uploaded_xlsx(uploaded_xlsx)

        # 워크플로우 실행
        result = run_workflow(user_prompt, df)

        # 응답 결과를 UI를 통해 노출
        # 응답 결과를 UI를 통해 노출해야 하는지는 요구사항 확인 필요
        st.subheader("🔎 분석 결과")
        st.write(result)

        # 결과에 대해 무조건 xlsx로 받는다고 하면, main.py에 xslx 반환 로직을 추가하고
        # 그렇지 않다고 하면 st.write를 하

if __name__ == "__main__":
    main()
