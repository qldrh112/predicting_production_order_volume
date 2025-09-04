import pandas as pd

def handle_uploaded_file(uploaded_file):
    """업로드된 파일 읽기 + 데이터 마스킹"""
    df = pd.read_excel(uploaded_file)

    # 간단한 데이터 마스킹 예시
    if "고객명" in df.columns:
        df["고객명"] = df["고객명"].apply(lambda x: str(x)[:1] + "***")

    return df
