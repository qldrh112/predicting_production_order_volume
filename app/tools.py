import pandas as pd

def analyze_dataframe(df, prompt):
    """사용자 프롬프트에 따라서 DataFrame에 적절한 조작을 가하여 프롬프트를 반환하는 함수"""

    # 간단한 예시 (추후 고도화 가능)
    if "평균" in prompt:
        return f"{str(df.mean(numeric_only=True))}이야. {prompt}"
    elif "합계" in prompt:
        return f"{str(df.sum(numeric_only=True))}이야. {prompt}"
    else:
        return f"데이터 크기: {df.shape}, 컬럼: {list(df.columns)}이야. {prompt}"
