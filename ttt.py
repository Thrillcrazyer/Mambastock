import pandas as pd
import glob

def combine_csv_to_series_column(file_path_pattern):
    # glob을 사용해 파일 경로 패턴에 맞는 모든 파일을 가져옵니다.
    file_paths = glob.glob(file_path_pattern)

    # 빈 DataFrame을 초기화합니다.
    combined_df = pd.DataFrame()

    for file_path in file_paths:
        # CSV 파일 읽기
        data = pd.read_csv(file_path, index_col=None)

        # 각 CSV 데이터를 새로운 열로 추가
        combined_df[file_path] = data["0"]
    combined_df.index=data["Unnamed: 0"]
    return combined_df

# 예제 사용법
# 파일 경로 패턴을 지정 (예: data/*.csv)
file_path_pattern = "result/*.csv"
result_df = combine_csv_to_series_column(file_path_pattern)

# 결과 출력
result_df.to_csv("total.csv")
print(result_df.loc["Return [%]"])
