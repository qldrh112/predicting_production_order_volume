# 데이터 분석 에이전트 (LangGraph 기반)

이 프로젝트는 **엑셀(xlsx) 데이터**를 업로드하면  
AI가 데이터를 분석하고, PDF 또는 Excel 형태의 보고서를 만들어주는 서비스입니다.  

Streamlit을 이용해 웹 화면에서 간단히 사용할 수 있으며,  
LangChain 기반의 **에이전트 구조**로 설계되어 확장이 쉽습니다.  

## 🚀 주요 기능
1. **데이터 업로드**
   - 사용자가 엑셀 파일(xlsx)을 업로드합니다.
   - AI는 데이터를 요약하거나 분석에 활용합니다.

2. **자연어 질의**
   - "이번 달 생산량 추세를 알려줘" 와 같은 질문을 입력하면
   - AI가 데이터를 분석한 결과를 보여줍니다.

3. **리포트 생성**
   - 분석 결과를 PDF 또는 Excel 보고서로 다운로드할 수 있습니다.

4. **확장 가능한 구조**
   - 데이터베이스 연결, API 연동, 추가 분석 도구를 붙이는 것이 쉽습니다.

## 🔧 아키텍처

본 프로젝트는 LangChain 기반 AI Agent로, 생산 데이터(XLSX)를 분석하고 인사이트를 제공합니다.

Streamlit (app/main.py): 사용자 인터페이스 제공

LangChain (app/workflow.py): LLM 워크플로우 관리

Jupyter KernelManager (app/kernel_manager.py): LLM이 안전하게 Python 코드를 실행할 수 있도록 지원

Data Handler (app/data_handler.py): 업로드된 데이터 전처리 및 마스킹

Tools (app/tools.py): 데이터프레임 분석 함수 모음

## 📊 실행 흐름

사용자가 Streamlit에서 XLSX 파일 업로드 + 분석 요청 입력

data_handler가 파일을 읽고 민감 데이터 마스킹

LangChain 워크플로우가 LLM을 통해 분석 지시 생성

Jupyter KernelManager가 실제 Python 코드 실행 (예: pandas, matplotlib)

결과를 LLM이 해석하고 사용자에게 반환

## 📂 폴더 구조
```
predicting_production_order_volume/
project/
├── main.py              # Streamlit 실행 진입점
├── app/
│   ├── __init__.py
│   ├── ui.py                # UI 관련 코드
│   ├── workflow.py          # LangChain workflow (LLM 연결 + Jupyter 커널 호출)
│   ├── tools.py             # 엑셀 데이터 분석 로직
│   ├── data_handler.py      # 데이터 업로드, 마스킹, 전처리
│   └── kernel_manager.py    # Jupyter 커널 관리 (KernelManager 래퍼)
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── utils/
│   ├── __init__.py
│   └── pdf_utils.py
│
├── requirements.txt
└── README.md
```


## ⚙️ 설치 방법

1. 저장소 복제
```bash
git clone https://github.com/qldrh112/predicting_production_order_volume.git
cd predicting_production_order_volume
```

2. 가상환경 생성 및 활성화

``` bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

3. 패키지 설치

``` bash
코드 복사
pip install -r requirements.txt

``` 
4. 환경 변수 설정 (.env 파일 생성)
```ini
MODEL_CONNECT_API_KEY=your_api_key_here
MODEL=사용하고자 하는 LLM 모델 명칭
```

## ▶️ 실행 방법
``` bash
streamlit run main.py
```

브라우저(기본: http://localhost:8501)에서 서비스를 사용할 수 있습니다.


## 📊 예시 화면
1. 엑셀 파일 업로드
2. "이번 달 생산량 추이를 알려줘" 입력
3. 분석 결과 확인
4. "보고서를 pdf로 저장해줘" → PDF 다운로드

## 🔧 확장 아이디어
- DB 연결: 엑셀 업로드 대신 DB에서 데이터 직접 조회
- 자동 리포트 생성: 매일/매주 분석 보고서를 자동 저장
- 대시보드 연동: 분석 결과를 시각화하여 웹 대시보드 제공

## 👥 대상 독자
- 현업 데이터 담당자
- 비전공자: 엑셀만 다룰 줄 알아도 사용 가능
- 개발자: 구조가 단순해 기능 확장이 용이