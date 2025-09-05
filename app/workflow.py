from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from app.kernel_manger import JupyterKernel
from app.tools import analyze_dataframe
from config.settings import MODEL, GOOGLE_API_KEY

def run_workflow(prompt, df):
    """DataFrame 분석 결과를 Jupyter 커널을 통해 실행한 뒤, LLM에 전달하여 응답을 반환하는 워크플로우를 실행"""

    # 사용자가 입력한 프롬프트에 따라 적절하게 DataFrame을 조작하고, 프롬프트를 입혀서 반환
    # (고도화 방안): 먼저 프롬프트를 분석하고, 프롬프트에 따라서 agent가 계획을 수립함, 그리고 analyze_dataframe 함수는 그 계획에 따라서 동적으로 코드를 생성하고, 결과를 kernel에서 실행함
    # analysis_prompt = analyze_dataframe(df, prompt)

    analysis_prompt = f"""
    다음은 사용자가 업로드한 xlsx을 데이터프레임으로 변환한 것입니다.

    {df}

    사용자의 질문: {prompt}
    """

    # LLM, Jupyter Kernel, 메모리 초기화
    kernel = JupyterKernel()
    llm = ChatGoogleGenerativeAI(model=MODEL, api_key=GOOGLE_API_KEY)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 툴 정의
    tools = [
        Tool(
            name="execute_code",
            func=kernel.execute_code,
            description="주피터 노트북에서 Python 코드를 실행합니다."
        ),
        Tool(
            name="get_variable",
            func=kernel.get_variable,
            description="실행된 코드의 변수 값을 조회합니다."
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type="zero-shot-react-description",
        memory=memory,
        verbose=True
    )

    # agent에 프롬프트를 전달
    try:
        response = agent.invoke(analysis_prompt)
        return response
    except Exception as e:
        return f'분석 중 오류가 발생했습니다: {str(e)}'