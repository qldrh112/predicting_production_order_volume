import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 기준으로 .env 로드
# settings.py의 위치가 바뀌면 해당 코드도 수정해야 함
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
# predicting_production_order_volume/config/settings.py의 할아버지는 predicting_production_order_volume/
# = load_dotenv("predicting_production_order_volume/.env")

MODEL = os.getenv("MODEL", "gemini-2.5-pro")    # .env 파일에 MODEL 변수가 할당되어 있지 않으면, gemini-2.5-pro라는 값을 가진다.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")