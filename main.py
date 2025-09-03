import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict
from queue import Empty

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from jupyter_client.manager import KernelManager

# PDF & Excel 리포트
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


# ----------------------------
# 환경 변수 & LLM 초기화
# ----------------------------
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL", "gemini-2.5-pro"),
    temperature=0,
    api_key=os.getenv("MODEL_CONNECT_API_KEY")
)


# ----------------------------
# 리포트 생성 함수
# ----------------------------
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
                pd.DataFrame({"content": [content]}).to_excel(writer, index=False, sheet_name="report")

    elif fmt == "pdf":
        pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
        doc = SimpleDocTemplate(str(out_path), pagesize=A4)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="Korean", fontName="HYSMyeongJo-Medium", fontSize=11, leading=14))

        story = [Paragraph("AI 분석 보고서", styles["Korean"]), Spacer(1, 16)]
        for line in content.split("\n"):
            story.append(Paragraph(line, styles["Korean"]))
            story.append(Spacer(1, 6))

        if table_csv:
            df = pd.read_csv(StringIO(table_csv))
            table_data = [list(df.columns)] + df.values.tolist()
            table = Table(table_data)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, -1), "HYSMyeongJo-Medium"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ]))
            story.append(table)

        doc.build(story)

    return str(out_path)


# ----------------------------
# DataAnalysisAgent
# ----------------------------
class DataAnalysisAgent:
    def __init__(self):
        # Jupyter 커널
        self.km = KernelManager()
        self.km.start_kernel()
        self.kernel_client = self.km.client()
        self.kernel_client.start_channels()

        # 메모리
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # 도구
        self.tools = [
            Tool(
                name="execute_code",
                func=self.execute_code,
                description="Python 코드를 실행합니다."
            ),
            Tool(
                name="get_variable",
                func=self.get_variable,
                description="실행된 코드의 변수 값을 조회합니다."
            ),
            Tool(
                name="create_report",
                func=create_report,
                description="분석 내용을 PDF 또는 Excel 리포트로 저장합니다."
            )
        ]

        # Agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=llm,
            agent_type="zero-shot-react-description",
            memory=self.memory,
            verbose=True
        )

    def execute_code(self, code: str) -> str:
        """Python 코드 실행"""
        try:
            msg_id = self.kernel_client.execute(code)
            while True:
                try:
                    msg = self.kernel_client.get_iopub_msg(timeout=5)
                    if msg["parent_header"]["msg_id"] == msg_id:
                        if msg["msg_type"] == "execute_result":
                            return str(msg["content"]["data"].get("text/plain", ""))
                        elif msg["msg_type"] == "stream":
                            return msg["content"]["text"]
                        elif msg["msg_type"] == "error":
                            return f"Error: {msg['content']['evalue']}"
                except Empty:
                    break
            return "실행 완료"
        except Exception as e:
            return f"실행 오류: {str(e)}"

    def get_variable(self, name: str) -> str:
        """특정 변수 조회"""
        return self.execute_code(f"print({name})")

    def analyze(self, query: str, context: str = "") -> str:
        """분석 실행"""
        try:
            return self.agent.run(context + query)
        except Exception as e:
            return f"분석 중 오류: {str(e)}"


# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.title("데이터 분석 에이전트 (PDF/XLSX 보고서 지원)")

    if "agent" not in st.session_state:
        st.session_state.agent = DataAnalysisAgent()

    uploaded = st.file_uploader("엑셀 파일 업로드", type=["xlsx"])
    query = st.text_area("분석/질문 입력")

    if st.button("분석 실행"):
        if not query:
            st.warning("질문을 입력하세요.")
            return

        # 파일 업로드 시 데이터 샘플 컨텍스트 제공
        context = ""
        if uploaded:
            df = pd.read_excel(uploaded).to_csv(index=False)
            context = f"데이터 샘플:\n{df}\n\n"

        result = st.session_state.agent.analyze(query, context)
        st.write("LLM 응답:", result)


if __name__ == "__main__":
    main()
