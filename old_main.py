import os
from dotenv import load_dotenv
from typing import List, Dict, Any

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from langchain.agents import AgentExecutor, initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
import google.generativeai as genai
from jupyter_client.manager import KernelManager
from queue import Empty
import tempfile
from pathlib import Path
import io
import json
import jupyter_client

# Google API 키 설정
load_dotenv()
# genai.configure(api_key=os.getenv('MODEL_CONNECT_API_KEY'))

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
        
        # 세션 전용 작업 디렉토리 (생성된 파일을 격리)
        self.session_dir = Path(tempfile.mkdtemp(prefix="langgraph_"))

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
            ),
            Tool(
                name="create_report",
                func=self.create_report,
                description=("간단한 리포트 파일을 생성합니다. 입력은 JSON 문자열로, 예: "
                             "{'format':'pdf'|'xlsx', 'filename':'out.pdf', 'content':'텍스트', 'table_csv':'csv 문자열(선택)'}")
            )
        ]
        
        # Agent 및 LLM 설정
        raw_key = os.getenv("MODEL_CONNECT_API_KEY")
        api_key = SecretStr(raw_key) if raw_key else None

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0,
            api_key=api_key
        )
        # handle_parsing_errors=True 로 설정하여 출력 파싱 실패 시 에이전트가 재시도하도록 함
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent_type="zero-shot-react-description",
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
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

    def create_report(self, json_input: str) -> str:
        """Agent가 호출할 수 있는 리포트 생성 도구.

        json_input은 JSON 문자열이어야 하며 다음 키를 가질 수 있습니다:
          - format: 'pdf' 또는 'xlsx'
          - filename: 생성할 파일명 (권장)
          - content: 보고서 텍스트
          - table_csv: csv 문자열 (선택)

        반환값: 생성된 파일의 절대 경로(문자열) 또는 에러 메시지
        """
        try:
            import json
            import pandas as pd
            from io import StringIO

            payload = json.loads(json_input) if isinstance(json_input, str) else (json_input if isinstance(json_input, dict) else {})
            fmt = payload.get('format', 'pdf')
            filename = payload.get('filename') or f'report.{fmt}'
            content = payload.get('content', '')
            table_csv = payload.get('table_csv')

            out_path = self.session_dir / filename

            if fmt.lower() == 'pdf':
                # If table_csv provided, convert to df and use generate_pdf_report
                if table_csv:
                    df = pd.read_csv(StringIO(table_csv))
                    pdf_bytes = generate_pdf_report(df)
                else:
                    # create a single-page PDF with text content
                    from reportlab.lib.pagesizes import A4
                    from reportlab.pdfgen import canvas
                    from reportlab.lib.units import mm
                    from reportlab.pdfbase import pdfmetrics
                    from reportlab.pdfbase.ttfonts import TTFont

                    # try common system fonts that support Korean
                    font_registered = False
                    font_name = 'KoreanFont'
                    try_paths = [
                        r'C:\Windows\Fonts\malgun.ttf',
                        r'C:\Windows\Fonts\malgunbd.ttf',
                        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'
                    ]

                    for p in try_paths:
                        try:
                            if Path(p).exists():
                                pdfmetrics.registerFont(TTFont(font_name, p))
                                font_registered = True
                                break
                        except Exception:
                            continue

                    buffer = io.BytesIO()
                    c = canvas.Canvas(buffer, pagesize=A4)
                    width, height = A4
                    # 제목
                    try:
                        if font_registered:
                            c.setFont(font_name, 12)
                        else:
                            c.setFont('Helvetica-Bold', 12)
                    except Exception:
                        c.setFont('Helvetica-Bold', 12)

                    c.drawString(20 * mm, height - 20 * mm, 'Agent-generated report')

                    # 본문 (한글 지원 폰트가 등록되어 있으면 사용)
                    try:
                        text_obj = c.beginText(20 * mm, height - 30 * mm)
                        if font_registered:
                            text_obj.setFont(font_name, 10)
                        else:
                            text_obj.setFont('Helvetica', 10)
                        for line in str(content).split('\n'):
                            text_obj.textLine(line[:120])
                        c.drawText(text_obj)
                    except Exception:
                        # fallback: draw simple ASCII-only
                        c.setFont('Helvetica', 10)
                        y = height - 30 * mm
                        for line in str(content).split('\n'):
                            c.drawString(20 * mm, y, line[:120])
                            y -= 6 * mm

                    c.save()
                    buffer.seek(0)
                    pdf_bytes = buffer.read()

                with open(out_path, 'wb') as fh:
                    fh.write(pdf_bytes)
                return str(out_path)

            elif fmt.lower() in ('xlsx', 'xls'):
                # write table_csv if present, otherwise write content to a sheet
                with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
                    if table_csv:
                        df = pd.read_csv(StringIO(table_csv))
                        df.to_excel(writer, index=False, sheet_name='data')
                    else:
                        # write content to a single-cell sheet
                        import pandas as pd

                        df = pd.DataFrame({'content': [content]})
                        df.to_excel(writer, index=False, sheet_name='report')
                return str(out_path)

            else:
                return f'Unsupported format: {fmt}'
        except Exception as e:
            return f'create_report error: {str(e)}'

    def analyze(self, query: str) -> str:
        """
        사용자 쿼리를 처리하고 분석을 수행합니다.
        
        Args:
            query: 사용자의 분석 요청 쿼리
            
        Returns:
            분석 결과 문자열
        """
        try:
            # 안전하게 에이전트를 호출 (ResourceExhausted 대응 등)
            result = run_agent_safe(self.agent, query)
            # run_agent_safe은 에러를 문자열로 반환할 수 있으므로 그 케이스도 그대로 반환
            # 특정한 파싱/입력 누락 오류는 더 친절한 안내로 변환
            if isinstance(result, str) and ("Missing some input keys" in result or "'format'" in result):
                return (
                    "분석 중 오류가 발생했습니다: 에이전트가 리포트 생성 도구를 호출할 때 필수 입력(format 등)이 누락된 것 같습니다.\n"
                    "프롬프트에서 'PDF로 저장해줘, 파일명: report.pdf'처럼 출력 형식(format)과 파일명을 명확히 지정해 주세요."
                )
            return result
        except Exception as e:
            return f"분석 중 오류가 발생했습니다: {str(e)}"

def main():
    st.title("데이터 분석 에이전트")

    # 세션 상태 초기화
    if "agent" not in st.session_state:
        st.session_state.agent = DataAnalysisAgent()

    st.header("1) 엑셀 파일 업로드 및 마스킹 설정")
    uploaded_file = st.file_uploader("엑셀 파일(.xlsx) 업로드", type=["xlsx"])

    mask_map_store_key = "mask_map"

    if uploaded_file is not None:
        import pandas as pd

        # 엑셀 파일의 시트 목록 표시
        xl = pd.ExcelFile(uploaded_file)
        sheets = xl.sheet_names
        sheet = st.selectbox("시트 선택", sheets)

        df = xl.parse(sheet)
        st.write("선택한 시트 미리보기 (처음 5행)")
        st.dataframe(df.head())

        # 컬럼 선택하여 마스킹
        candidate_cols = list(df.columns)
        cols_to_mask = st.multiselect("마스킹할 컬럼 선택", candidate_cols)

        mask_type = st.selectbox("마스킹 방식 선택", ["tokenize", "hash", "partial"], index=0)

        if st.button("마스킹 생성 및 적용"):
            mask_map = create_mask_map(df, cols_to_mask, method=mask_type)
            st.session_state[mask_map_store_key] = mask_map
            st.session_state['original_df'] = df
            st.session_state['masked_df'] = mask_dataframe(df, mask_map)
            st.success("마스킹 적용 완료 — 세션에 저장됨")

            # PDF 리포트 생성 버튼은 제거. 프롬프트로 생성 요청하면 에이전트가 파일을 만들 수 있음.

        # 마스킹이 이미 세션에 존재하면 미리보기 및 분석 컨트롤을 보여줌
        if st.session_state.get('masked_df') is not None:
            st.subheader("마스킹된 데이터 (세션에 저장됨)")
            st.dataframe(st.session_state['masked_df'].head())

            if st.checkbox("마스킹된 데이터로 분석 실행 (샘플 20행)", key='mask_analysis_checkbox'):
                masked_df = st.session_state.get('masked_df')
                if masked_df is None:
                    st.warning("마스킹된 데이터가 없습니다. 먼저 마스킹을 생성하세요.")
                else:
                    small = dataframe_summary_and_sample(masked_df, sample_n=5)
                    user_query = st.session_state.get('global_query') or '요약해줘.'

    st.header("2) 자유 텍스트 분석")
    query = st.text_area("분석하고 싶은 텍스트를 입력하세요:")

    if st.button("분석 시작 (텍스트)" ):
        if query:
            # 만약 마스킹 맵이 존재하면 쿼리 내 토큰을 언마스킹해서 보여주도록 전달
            with st.spinner("분석 중..."):
                # 세션에 마스킹된 데이터가 있으면 컨텍스트로 포함
                masked_df = st.session_state.get('masked_df')
                if masked_df is not None:
                    small = dataframe_summary_and_sample(masked_df, sample_n=5)
                    full_prompt = (
                        f"아래는 분석 컨텍스트로 사용할 마스킹된 데이터의 요약과 샘플입니다.\n\n"
                        f"요약:\n{small['summary']}\n\n샘플 CSV:\n{small['sample_csv']}\n\n질문: {query}"
                    )
                else:
                    full_prompt = query
                # 먼저 format/filename 패턴이 있으면 create_report 직접 호출
                handled, msg = parse_report_request_and_execute(full_prompt, st.session_state.agent, masked_df=masked_df)
                if handled:
                    st.write(msg)
                else:
                    result = run_agent_safe(st.session_state.agent, full_prompt)
                    # 언마스킹 처리
                    unmasked = unmask_text(result, st.session_state.get(mask_map_store_key, {}))
                    st.write(unmasked)
                # 커널/에이전트가 생성한 PDF 파일이 있으면 다운로드 버튼 제공 (세션 디렉토리 우선)
                try:
                    agent_obj = st.session_state.get('agent')
                    base = agent_obj.session_dir if agent_obj is not None else Path(os.getcwd())
                    pdfs = [p for p in base.iterdir() if p.suffix.lower() == '.pdf']
                    if pdfs:
                        st.write("생성된 PDF 파일:")
                        for p in pdfs:
                            with open(p, 'rb') as fh:
                                st.download_button(f"{p.name} 다운로드", data=fh.read(), file_name=p.name, mime='application/pdf')
                except Exception:
                    pass
        else:
            st.warning("분석할 내용을 입력해주세요.")


def create_mask_map(df, cols: List[str], method: str = "tokenize") -> Dict[str, Dict[Any, str]]:
    """데이터프레임의 지정된 컬럼들에 대해 원본값->마스킹 토큰 매핑을 생성합니다.

    반환값은 {'col_name': {original_value: token, ...}, ...} 형태입니다.
    """
    mask_map: Dict[str, Dict[Any, str]] = {}
    for col in cols:
        col_map: Dict[Any, str] = {}
        uniq_vals = pd_unique_values_safe(df, col)
        for i, val in enumerate(uniq_vals, start=1):
            if pd_is_nan(val):
                continue
            if method == "tokenize":
                token = f"{col[:3].upper()}_T{i}"
            elif method == "hash":
                import hashlib

                token = hashlib.sha256(str(val).encode()).hexdigest()[:12]
            else:  # partial
                s = str(val)
                token = s[0] + "*" * max(0, len(s) - 2) + s[-1] if len(s) > 2 else "**"
            col_map[val] = token
        mask_map[col] = col_map
    return mask_map


def mask_dataframe(df, mask_map: Dict[str, Dict[Any, str]]):
    import pandas as pd

    result_df = df.copy()
    for col, cmap in mask_map.items():
        if col in result_df.columns:
            result_df[col] = result_df[col].apply(lambda v: cmap.get(v, v))
    return result_df


def unmask_text(text: str, mask_map: Dict[str, Dict[Any, str]]) -> str:
    """LLM 결과에서 토큰을 실제 값으로 대체합니다. 간단하게 토큰->원값 매핑을 뒤집어 치환합니다."""
    # invert maps
    reverse_map = {}
    for col, cmap in mask_map.items():
        for orig, token in cmap.items():
            reverse_map[str(token)] = str(orig)

    # 간단 토큰 치환
    out = text
    for token, orig in reverse_map.items():
        out = out.replace(token, orig)
    return out


def pd_unique_values_safe(df, col: str):
    """pandas에 의존하지 않는 안전한 유일값 획득기"""
    try:
        import pandas as pd

        vals = pd.unique(df[col].dropna())
        return list(vals)
    except Exception:
        # fallback
        seen = []
        for v in df[col].tolist():
            if v not in seen and not pd_is_nan(v):
                seen.append(v)
        return seen


def pd_is_nan(x):
    try:
        import math

        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return x is None


def dataframe_summary_and_sample(df, sample_n: int = 3) -> Dict[str, str]:
    """작은 컨텍스트용 요약을 생성합니다: 컬럼, 행 수, 간단 통계, 그리고 소샘플 CSV.

    반환값: {'summary': str, 'sample_csv': str}
    """
    import pandas as pd

    try:
        nrows = len(df)
        cols = list(df.columns)
        dtypes = df.dtypes.apply(lambda x: str(x)).to_dict()

        # 기본 통계(수치형 컬럼만)를 텍스트로
        try:
            stats = df.describe().to_string()
        except Exception:
            stats = ''

        sample_csv = df.head(sample_n).to_csv(index=False)

        summary_lines = [f'rows: {nrows}', f'columns: {cols}', f'dtypes: {dtypes}']
        if stats:
            summary_lines.append('stats:\n' + stats[:1000])

        summary = '\n'.join(summary_lines)
        return {'summary': summary, 'sample_csv': sample_csv}
    except Exception as e:
        return {'summary': f'unable to summarize: {str(e)}', 'sample_csv': df.head(1).to_csv(index=False)}

def parse_report_request_and_execute(prompt: str, agent_obj, masked_df=None, analysis_result: str = None):
    """
    리포트 요청이 있는지 감지만 하고,
    analysis_result가 주어졌을 때만 create_report를 실행.
    """
    import re, json
    fmt_m = re.search(r"format\s*(?:[:=]|\s)\s*(pdf|xlsx|xls)", prompt, re.IGNORECASE)
    fname_m = re.search(r"(?:filename|파일명)\s*(?:[:=]|\s)*([\w\-. ]+)", prompt, re.IGNORECASE)
    if not fmt_m or not fname_m:
        return False, ""

    fmt = fmt_m.group(1).lower()
    filename = fname_m.group(1).strip()

    if analysis_result is None:
        # 보고서 요청이 있음을 알리고, 아직 생성은 안함
        return True, f"리포트 요청 감지됨 → 분석 후 {filename}로 저장 예정"

    payload = {
        "format": fmt,
        "filename": filename,
        "content": analysis_result
    }

    if masked_df is not None:
        from io import StringIO
        small = dataframe_summary_and_sample(masked_df, sample_n=5)
        payload["table_csv"] = small["sample_csv"]

    res = agent_obj.create_report(json.dumps(payload))
    return True, res


def run_agent_safe(agent_obj, prompt: str) -> str:
    """AgentExecutor나 DataAnalysisAgent wrapper를 안전하게 호출합니다."""
    try:
        # AgentExecutor는 run 메서드, DataAnalysisAgent는 analyze 메서드
        if hasattr(agent_obj, 'run'):
            return agent_obj.run(prompt)
        if hasattr(agent_obj, 'analyze'):
            return agent_obj.analyze(prompt)
        return str(agent_obj(prompt))
    except Exception as e:
        msg = str(e)
        # Google Gen AI ResourceExhausted는 컨텍스트가 너무 클 때 발생 가능
        if 'ResourceExhausted' in msg or 'resource exhausted' in msg.lower():
            # 짧은 폴백 프롬프트로 한 번만 재시도
            try:
                fallback = "요약해줘. (간결하게 3문장 이내)"
                if hasattr(agent_obj, 'run'):
                    return agent_obj.run(fallback)
                if hasattr(agent_obj, 'analyze'):
                    return agent_obj.analyze(fallback)
                return str(agent_obj(fallback))
            except Exception as e2:
                return f"리소스 부족으로 분석 실패(재시도 실패): {str(e2)}"
        # Missing some input keys 같이 도구 호출시 필수 입력 누락 오류는 사용자 친화적 안내로 대체
        if 'Missing some input keys' in msg or "'format'" in msg:
            return (
                "분석 중 오류가 발생했습니다: 에이전트가 리포트 생성 도구를 호출할 때 일부 필수 입력이 누락된 것 같습니다.\n"
                "프롬프트에서 출력 형식(format)을 'pdf' 또는 'xlsx'로 명시하고, 원하는 파일명(filename)을 지정해 주세요.\n"
                "예시: '마스킹된 데이터를 바탕으로 PDF 리포트를 생성해줘. format: pdf, filename: student_report.pdf'"
            )
        return f"분석 도중 에러 발생: {msg}"


def generate_pdf_report(df) -> bytes:
    """간단한 테이블과 통계 정보를 포함한 PDF 바이트를 생성합니다."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import io

        # register korean-capable font if available
        font_registered = False
        font_name = 'KoreanFont'
        try_paths = [
            r'C:\Windows\Fonts\malgun.ttf',
            r'C:\Windows\Fonts\malgunbd.ttf',
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'
        ]
        for p in try_paths:
            try:
                if Path(p).exists():
                    pdfmetrics.registerFont(TTFont(font_name, p))
                    font_registered = True
                    break
            except Exception:
                continue

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        # 제목
        if font_registered:
            c.setFont(font_name, 14)
        else:
            c.setFont("Helvetica-Bold", 14)
        c.drawString(20 * mm, height - 20 * mm, "데이터 분석 리포트 (마스킹된 데이터)")

        # 테이블 헤더 출력 (최대 6컬럼만 보여줌)
        c.setFont("Helvetica", 10)
        cols = list(df.columns)[:6]
        y = height - 30 * mm
        x = 20 * mm
        for col in cols:
            c.drawString(x, y, str(col))
            x += 30 * mm

        # 몇 행 출력
        y -= 7 * mm
        for _, row in df.head(10).iterrows():
            x = 20 * mm
            for col in cols:
                c.drawString(x, y, str(row.get(col, ''))[:25])
                x += 30 * mm
            y -= 6 * mm
            if y < 20 * mm:
                c.showPage()
                y = height - 20 * mm

        # 통계 요약
        c.showPage()
        c.setFont("Helvetica-Bold", 12)
        c.drawString(20 * mm, height - 20 * mm, "기본 통계 요약")
        c.setFont("Helvetica", 10)
        stats_text = df.describe(include='all').to_string()
        text_obj = c.beginText(20 * mm, height - 30 * mm)
        for line in stats_text.split('\n'):
            text_obj.textLine(line[:120])
        c.drawText(text_obj)

        c.save()
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        raise


if __name__ == "__main__":
    main()