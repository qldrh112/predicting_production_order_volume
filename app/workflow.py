from app.tools import analyze_dataframe
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import MODEL, GOOGLE_API_KEY

def run_workflow(prompt, df):
    """LLM + DataFrame 분석 워크플로우"""
    # LLM 초기화
    llm = ChatGoogleGenerativeAI(model=MODEL, api_key=GOOGLE_API_KEY)

    # 데이터 분석
    insights = analyze_dataframe(df, prompt)

    # LLM에 분석 결과 전달
    response = llm.invoke(f"""
    사용자 요청: {prompt}
    데이터 분석 결과: {insights}
    최종적으로 사용자에게 제공할 해석을 작성해줘.
    """)

    return response.content
