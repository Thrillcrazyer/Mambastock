
import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pandas as pd
from binance import Client
from dotenv import load_dotenv
import datetime
import time

def get_client():
    load_dotenv()
    api_key = os.getenv('API_KEY')
    api_secret = os.getenv('API_SECRET')
    client = Client(api_key, api_secret)
    return client

def get_USDT_symbols(client):
    tickers = client.get_orderbook_tickers()
    df = pd.DataFrame(tickers)
    df= df["symbol"]
    df=pd.DataFrame(df)
    ndf=df[df["symbol"].str[-4:]=="USDT"]
    return ndf

def formatting(candles):
    """
    market data 를 api에서 수신받으면, pd.Dataframe에 맞게 수정하는 코드
    col = ['open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'kline_close_time', 'quote_asset_volume', 'number_of_trades', 'base', 'quote', 'dummy']
    """
    col = ['open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'kline_close_time', 'quote_asset_volume', 'number_of_trades', 'base', 'quote', 'dummy']
    data = pd.DataFrame(candles, columns=col)
    data = data.drop('dummy', axis=1)
    data=data.astype(float)
    data['date'] = data['open_time'].apply(lambda x: datetime.date.fromtimestamp(x / 1000))
    return data

def collecting_data_1day(client, save_path='./test_dataset/1day.csv',symbol='BTCUSDT',size=3000):
    """
    바이낸스에서 1시간봉의 모든 데이터를 가져오는 코드.
    interval 의 max=1000이다.
    """
    interval='1d'
    start_time = datetime.datetime.now() - datetime.timedelta(days=size)
    print(start_time)
    start_str = start_time.strftime("%d %b, %Y %H:%M:%S")
    klines = client.get_historical_klines(symbol, interval, start_str=start_str)
    data=formatting(klines)
    data.to_csv(save_path)
    print(data)
    
def collecting_data_all_1hour(client,save_path='./test_dataset/zzz.csv',interval='1h',symbol='BTCUSDT'):
    start_time = 1502928000000.0
    # 밀리초를 초로 변환
    start_time_in_seconds = start_time / 1000.0
    start_time = datetime.datetime.utcfromtimestamp(start_time_in_seconds)
    print(start_time)
    start_str = start_time.strftime("%d %b, %Y %H:%M:%S")
    klines = client.get_historical_klines(symbol, interval, start_str=start_str)
    data=formatting(klines)
    data.to_csv(save_path)
    print(f"다운로드된 사이즈: {len(data)}")

def collecting():
    client=get_client()
    datas= get_USDT_symbols(client)
    save_path='./data/crypto_1hour'
    total=len(datas["symbol"])
    for i,symbol in enumerate(datas["symbol"]):
        try:
            path=save_path+f'/{symbol}.csv'
            print(path)
            collecting_data_all_1hour(client=client,save_path=path,symbol=symbol)
            print(f"{i+1}/{total} {symbol}다운로드 완료!")
            time.sleep(5)
        except Exception as e:  
            print(e)  

if __name__ == '__main__':
    #collecting()
    path='./data/HBARUSDT_15min.csv'
    client=get_client()
    #collecting_data_1day(client=client,save_path=path,symbol="BTCUSDT")
    collecting_data_all_1hour(client=client,save_path=path,interval='15m',symbol="1MBABYDOGEUSDT")

