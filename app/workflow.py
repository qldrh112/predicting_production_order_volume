from langchain_google_genai import ChatGoogleGenerativeAI
from app.kernel_manger import JupyterKernel
from app.tools import analyze_dataframe
from config.settings import MODEL, GOOGLE_API_KEY

def run_workflow(prompt, df):
    """DataFrame 분석 결과를 Jupyter 커널을 통해 실행한 뒤, LLM에 전달하여 응답을 반환하는 워크플로우를 실행"""

    # 사용자가 입력한 프롬프트에 따라 적절한 DataFrame 조작을 한 뒤, 반환
    code_to_run = analyze_dataframe(df, prompt)

    # LLM 및 Jupyter Kernel 초기화
    llm = ChatGoogleGenerativeAI(model=MODEL, api_key=GOOGLE_API_KEY)
    kernel = JupyterKernel()

    try:
        execution_result = kernel.run_code(code_to_run)
    finally:
        kernel.shutdown()

    # LLM에 분석 결과 전달
    response = llm.invoke(f"""
    사용자 요청: {prompt}
    데이터 분석 결과: {execution_result}
    최종적으로 사용자에게 제공할 해석을 작성해줘.
    """)

    return response.content
