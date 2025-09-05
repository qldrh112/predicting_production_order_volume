import pandas as pd

def handle_uploaded_xlsx(uploaded_xlsx) -> pd.DataFrame:
    """사용자가 업로드한 xlsx 파일을 받아 DataFrame으로 변환하고, 민감 데이터를 마스킹하는 함수"""
    df = pd.read_excel(uploaded_xlsx)

    # 엑셀 파일에 어떤 데이터가 들어 있는지 알 수가 없음
    # 그래서 함부로 사용자가 업로드한 xlsx 파일을 LLM에 전송하면 안 됨
    # 방법1. 마스킹할 컬럼이나 정보를 하드코딩한다.

    return df
