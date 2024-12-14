import FinanceDataReader as fdr
from pykrx import stock
import time
import os

def is_file_exist(file_path):
    """
    특정 파일이 존재하는지 확인합니다.

    :param file_path: 확인할 파일의 전체 경로
    :return: 파일이 존재하면 True, 아니면 False
    """
    return os.path.isfile(file_path)

def get_stock_price_data(market):
    stocks = fdr.StockListing(market)
    folder=f'./data/{market}'
    total=len(stocks)
    print(stocks)
    error_ticker=[]
    #print(stocks)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    for i,stock in stocks.iterrows():
        print(stock)
        try:
            종목=stock["Name"]
            save_path=f'./data/{market}/{종목}.csv'
            
            if is_file_exist(save_path):
                continue
            
            data=fdr.DataReader(stock["Code"])
            data.to_csv(save_path)
            print(f"{종목} 다운로드 완료!\n사이즈: {len(data)}\n진행도: {i+1} / {total}\n") 
        except Exception as e:  
            error_ticker.append(종목)
            print(e)  

if __name__ == '__main__':
    #get_stock_price_data("KOSPI")
    get_stock_price_data("KOSDAQ")
    
