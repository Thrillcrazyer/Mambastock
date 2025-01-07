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

def get_stock_price_data(market):
    stocks = fdr.StockListing(market)
    folder=f'./data/{market}'
    total=len(stocks)
    error_ticker=[]
    
    if not os.path.isdir(folder):
        os.mkdir(folder)
        
    for i, stock in stocks.iterrows():
       # if i<1597:
        #    continue
        name= stock['Name'].replace("/" , "")
        ticker= stock['Symbol']
        save_path=f'./data/{market}/{ticker}.csv'
        try:
            if is_file_exist(save_path):
                continue
            print(save_path)
            data=fdr.DataReader(f'YAHOO:{ticker}')
            data.to_csv(save_path)
            print(f"{name}({ticker}) 다운로드 완료!\n사이즈: {total}\n진행도: {i+1} / {total}\n남은시간: {(total-i)/2:.2f}분")
        except Exception as e:
            error_ticker.append(ticker)
            print(e)
    print("오류 티커", error_ticker)
            
       
            

if __name__ == '__main__':
    #get_stock_price_data('S&P500')
    name='NASDAQ'
    get_stock_price_data(name)
    get_stock_price_data("NYSE")
    