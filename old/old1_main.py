import os
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
import google.generativeai as genai
from jupyter_client.manager import KernelManager
from queue import Empty

from pydantic import SecretStr


# reportlab (PDF)
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# 환경 변수 로드
load_dotenv()
api_key = SecretStr(os.getenv("MODEL_CONNECT_API_KEY")) if os.getenv("MODEL_CONNECT_API_KEY") else None
model = os.getenv("MODEL")

# LLM 초기화
llm = ChatGoogleGenerativeAI(
    model=model,
    temperature=0,
    api_key=api_key
)

# --- PDF + XLSX 리포트 생성 함수 ---
def create_report(payload: Dict) -> str:
    """리포트 파일 생성: pdf 또는 xlsx"""
    from io import StringIO
    fmt = payload.get("format", "pdf").lower()
    filename = payload.get("filename", f"report.{fmt}")
    content = payload.get("content", "")
    table_csv = payload.get("table_csv")

    out_path = Path(filename)

    if fmt == "xlsx":
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            if table_csv:
                df = pd.read_csv(StringIO(table_csv))
                df.to_excel(writer, index=False, sheet_name="data")
            else:
                pd.DataFrame({"content":[content]}).to_excel(writer, index=False, sheet_name="report")

    elif fmt == "pdf":
        # ✅ 한글 폰트 등록
        pdfmetrics.registerFont(UnicodeCIDFont('HYSMyeongJo-Medium'))

        doc = SimpleDocTemplate(str(out_path), pagesize=A4)  # ✅ 수정됨
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="Korean", fontName="HYSMyeongJo-Medium", fontSize=11, leading=14))

        story = []
        story.append(Paragraph("AI 분석 보고서", styles['Korean']))
        story.append(Spacer(1, 16))

        # 본문 내용
        for line in content.split("\n"):
            story.append(Paragraph(line, styles['Korean']))
            story.append(Spacer(1, 6))

        # 표가 있으면 추가
        if table_csv:
            df = pd.read_csv(StringIO(table_csv))
            table_data = [list(df.columns)] + df.values.tolist()
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'HYSMyeongJo-Medium'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ]))
            story.append(table)

        doc.build(story)

    return str(out_path)


# --- 보고서 요청 감지 ---
def detect_report_request(query: str):
    import re
    fmt = re.search(r"format\s*[:=]?\s*(pdf|xlsx)", query, re.I)
    fname = re.search(r"(?:filename|파일명)\s*[:=]?\s*([\w.\-]+)", query, re.I)  # 수정됨
    if fmt and fname:
        return fmt.group(1).lower(), fname.group(1)
    return None, None

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

# --- Streamlit UI ---
def main():
    st.title("데이터 분석 에이전트 (PDF/XLSX 보고서 지원)")

    # 세션 상태 초기화
    if "agent" not in st.session_state:
        st.session_state.agent = DataAnalysisAgent()

    uploaded = st.file_uploader("엑셀 파일 업로드", type=["xlsx"])
    query = st.text_area("분석/질문 입력")

    if st.button("분석 실행"):
        if not query:
            st.warning("질문을 입력하세요.")
            return

        # 엑셀 맥락
        context = ""
        if uploaded:
            df = pd.read_excel(uploaded).to_csv(index=False)  # 샘플만 사용
            context = f"데이터 샘플:\\n{df}\\n\\n"

        # LLM 실행
        full_prompt = context + f"질문: {query}"
        result = llm.invoke(full_prompt).content
        st.write("LLM 응답:", result)

        # 리포트 요청 감지
        fmt, fname = detect_report_request(query)
        if fmt and fname:
            path = create_report({"format": fmt, "filename": fname, "content": result})
            with open(path, "rb") as f:
                if fmt == "pdf":
                    st.download_button("📥 PDF 다운로드", f.read(), file_name=fname, mime="application/pdf")
                else:
                    st.download_button("📥 Excel 다운로드", f.read(), file_name=fname, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    main()
