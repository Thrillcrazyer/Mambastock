import FinanceDataReader as fdr
import time
import os

def is_file_exist(file_path):
    """
    특정 파일이 존재하는지 확인합니다.

    :param file_path: 확인할 파일의 전체 경로
    :return: 파일이 존재하면 True, 아니면 False
    """
    return os.path.isfile(file_path)

def get_price_data(market):
    folder=f'./data/eval'
    total=len(market)
    error_ticker=[]
    
    if not os.path.isdir(folder):
        os.mkdir(folder)
        
    for i, stock in enumerate(market):
        name= stock
        save_path=f'./data/eval/{name}.csv'
        try:
            if is_file_exist(save_path):
                continue
            print(save_path)
            data=fdr.DataReader(name)
            data.to_csv(save_path)
            print(f"{name} 다운로드 완료! 진행도: {i+1} / {total}\n")
        except Exception as e:
            error_ticker.append(name)
            print(e)
    print("오류 티커", error_ticker)
            
       
            

if __name__ == '__main__':
    yahoo_index_symbol_map = (
            'DJI', 'IXIC', 'US500', 'S&P500',
            'RUT', 'VIX', 'N225', 'SSEC',
            'FTSE', 'HSI', 'FCHI', 'GDAXI',
            'US5YT', 'US10YT', 'US30YT', # US Treasury Bonds
        )
    get_price_data(yahoo_index_symbol_map)
    #get_stock_price_data("KOSDAQ")
    
    
