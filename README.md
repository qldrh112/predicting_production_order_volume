# 데이터 분석 에이전트 (Langchain 기반)

이 프로젝트는 **엑셀(xlsx) 데이터**를 업로드하면  
AI가 데이터를 분석하고, Excel 형태의 보고서를 만들어주는 서비스입니다.  

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
   - 분석 결과를 Excel 보고서로 다운로드할 수 있습니다.

4. **확장 가능한 구조**
   - 데이터베이스 연결, API 연동, 추가 분석 도구를 붙이는 것이 쉽습니다.

## 🔧 아키텍처

본 프로젝트는 LangChain 기반 AI Agent로, 생산 데이터(XLSX)를 분석하고 인사이트를 제공합니다.
![software_architecture](https://github-production-user-asset-6210df.s3.amazonaws.com/69291489/486480411-91e277ce-0837-496b-aeb6-14f8d52f72e4.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250907%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250907T042919Z&X-Amz-Expires=300&X-Amz-Signature=2fd882d0beb068bf884663034913212a68aca91d308db1bc3b018bf25c8b4295&X-Amz-SignedHeaders=host)
```
[사용자] 
    │
    ▼
[Streamlit UI] ──> [Langchain Agent] ──┐
    │                                  │
    │                          분석 계획 / 도구 호출
    │                                  │
    ▼                                  ▼
[Raw Data DB] ───────────> [Jupyter MCP Server] ──> Python 코드 실행
    │                                  │
    ▼                                  ▼
   마스킹/전처리                 분석 결과 / 계산 값
    │                                  │
    └─────────── LLM 검토 & 도구 호출 ─┘
                    │
                    ▼
         [xlsx 조작 도구 호출 / 최종 산출물]
                    │
                    ▼
            [사용자에게 반환]
```

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
├── .venv/                   # 가상 환경
├── app/
│   ├── __init__.py
│   ├── data_handler.py      # Xlsx 전처리 및 마스킹
│   ├── kernel_manager.py    # Jupyter 커널 관리
│   ├── tools.py             # DataFrame 분석/조작 함수
│   ├── ui.py                # Streamlit UI 로직 분리
│   └── workflow.py          # LLM + Agent + memory + tool orchestration
│
├── config/
│   ├── __init__.py
│   └── settings.py          # 환경 변수 불러오기
│
├── docs/                    # 문서
│   ├── meeting_note         # 회의록
│   └── Softward_requirements_Specification.xlsx          # 요구사항명세서
│
├── output/                  # 결과물
│
├── utils/
│   ├── __init__.py
│   └── pdf_utils.py         # pdf 변환 로직
│
├── .env                     # 환경 변수 관리
├── .gitginore               # git으로 통제하지 않는 객체 관리
├── main.py                  # Streamlit 실행 진입점 및 UI endpoint
├── requirements.txt         # 의존성 관리
└── README.md                
```


## ⚙️ 개발 환경

- **Python 버전**: 3.13 이상 (권장)  
- **가상환경**: `virtualenv` 사용  
- **패키지 관리**: `pip` + `requirements.txt`  
- **OS**: Windows / macOS / Linux 모두 가능  

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
pip install -r requirements.txt
``` 

4. 환경 변수 설정 (.env 파일 생성)
```ini
MODEL_CONNECT_API_KEY=당신의 API KEY
MODEL=사용하고자 하는 LLM 모델 명칭
```

5. 커널 등록
현재 가상 환경을 Jupyter에서 사용할 수 있도록 커널을 등록함
``` bash
python -m ipykernel install --user --name=python3 --display-name "Python 3 (.venv)"
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
4. "보고서를 xlsx로 저장해줘" → xlsx 다운로드

## 💡 프로젝트 관리
### 프로젝트 일정 관리

- **형상 관리 & 이슈 관리**: GitHub  
- **일정 관리 & 태스크 관리**: Jira (무료 요금제, 10 팀 이내)  
  - Epic → Story → Task 구조로 관리  
  - 시작일/마감일 설정, 칸반 보드 기반으로 진행  
- **GitHub ↔ Jira 연동**
  - 커밋/PR 메시지에 Jira 이슈 키 포함 (예: `PROJ-12`)
  - Smart Commit (`#done`, `#in-progress`) 으로 Jira 상태 자동 변경
- **회의록 관리**
  - GitHub 저장소에 `docs/meeting_notes/` 하위에 `.md`로 관리

### 협업 방식

Git 브랜치 전략: GitFlow 또는 Feature Branch 기반
- main / develop / feature/* / hotfix/* 등
- PR(풀 리퀘스트) 리뷰   
- 최소 1명 이상 코드 리뷰 필수  
- 코드 스타일/포맷팅 확인 (PEP8, black 등)  

#### 커밋 메시지 규칙
- feat: 새로운 기능 추가
- fix: 버그 수정
- refactor: 코드 구조 개선
- docs: 문서 변경
- comment: 주석
- chore: 빌드, 패키지, 환경 설정 등

## 🔧 추후 개선 사항
- DB 연결: 엑셀 업로드 대신 DB에서 데이터 직접 조회
- 자동 리포트 생성: 매일/매주 분석 보고서를 자동 저장
- 대시보드 연동: 분석 결과를 시각화하여 웹 대시보드 제공

## 📌 주의사항
- API Key, 개인정보 등 민감 데이터는 절대로 GitHub 등 외부에 노출 금지
- LLM 호출 시 대용량 데이터는 성능에 영향 가능 → 필요시 샘플링