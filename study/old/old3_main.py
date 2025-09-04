import os
from typing import List, Dict, Any

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
import google.generativeai as genai
from jupyter_client.manager import KernelManager
from queue import Empty
import json
import jupyter_client

# Google API 키 설정
os.environ["GOOGLE_API_KEY"] = "your-google-api-key"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

class DataAnalysisAgent:
    def __init__(self):
        # Jupyter 커널 초기화
        self.km = KernelManager()
        self.km.start_kernel()
        self.kernel_client = self.km.client()
        self.kernel_client.start_channels()
        
        # 메모리 초기화
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 도구 설정
        self.tools = [
            Tool(
                name="execute_code",
                func=self.execute_code,
                description="주피터 노트북에서 Python 코드를 실행합니다."
            ),
            Tool(
                name="get_variable",
                func=self.get_variable,
                description="실행된 코드의 변수 값을 조회합니다."
            )
        ]
        
        # Agent 설정
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0,
            convert_system_message_to_human=True
        )
        self.agent = initialize_agent(
            tools=self.tools,
            llm=llm,
            agent_type="zero-shot-react-description",
            memory=self.memory,
            verbose=True
        )
        
    def execute_code(self, code: str) -> str:
        """Python 코드를 실행하고 결과를 반환합니다."""
        try:
            msg_id = self.kernel_client.execute(code)
            while True:
                try:
                    msg = self.kernel_client.get_iopub_msg(timeout=10)
                    if msg['parent_header']['msg_id'] == msg_id:
                        if msg['msg_type'] == 'execute_result':
                            return str(msg['content']['data'].get('text/plain', ''))
                        elif msg['msg_type'] == 'stream':
                            return msg['content']['text']
                        elif msg['msg_type'] == 'error':
                            return f"Error: {msg['content']['evalue']}"
                except Empty:
                    break
            return "실행 완료"
        except Exception as e:
            return f"실행 중 오류 발생: {str(e)}"
            
    def get_variable(self, name: str) -> str:
        """특정 변수의 값을 조회합니다."""
        code = f"print({name})"
        return self.execute_code(code)

    def analyze(self, query: str) -> str:
        """
        사용자 쿼리를 처리하고 분석을 수행합니다.
        
        Args:
            query: 사용자의 분석 요청 쿼리
            
        Returns:
            분석 결과 문자열
        """
        try:
            result = self.agent.run(query)
            return result
        except Exception as e:
            return f"분석 중 오류가 발생했습니다: {str(e)}"

def main():
    st.title("데이터 분석 에이전트")
    
    # 세션 상태 초기화
    if "agent" not in st.session_state:
        st.session_state.agent = DataAnalysisAgent()
    
    # 사용자 입력
    query = st.text_input("분석하고 싶은 내용을 입력하세요:")
    
    if st.button("분석 시작"):
        if query:
            with st.spinner("분석 중..."):
                result = st.session_state.agent.analyze(query)
                st.write(result)
        else:
            st.warning("분석할 내용을 입력해주세요.")

if __name__ == "__main__":
    main()