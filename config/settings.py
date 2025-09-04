import os
from dotenv import load_dotenv
from pathlib import Path

# 프로젝트 루트 기준으로 .env 로드
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MODEL = os.getenv("MODEL", "gemini-2.5-pro")   # 기본값 지정
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")