import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pandas as pd
import torch
from tqdm import tqdm

from dataset.dataset_iter import data_formatting
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

data_path="/workspace/stock/data/BTCUSDT.csv"
device="cuda"
d_type=torch.float32

def MambaModel():
    from model.mamba_stock import Mambamodeling
    pretrained_path="/workspace/stock/result/Thrillcrazyer/Mambastocks_ver0.5/Mamba2_2048/0000085000"
    model=Mambamodeling(
            d_model=768,
            d_inermediate=2048,
            n_layer=8,
            layer="Mamba2",
            pretrained_path=pretrained_path
        ).to(device)
    model.eval()
    return model

def TransformerModel():
    from model.qwen_stock import QwenStock
    from transformers import AutoConfig
    
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    
    qwenconfig=AutoConfig.from_pretrained(model_name,
                    intermediate_size= 2048,
                    max_position_embeddings= 128,
                    num_hidden_layers=8,
                    vocab_size= 5
                    )

    model=QwenStock(qwenconfig).from_pretrained("/workspace/stock/result/Thrillcrazyer/Mambastocks_ver0.5/Qwen2/0000085000").to(device)
    return model

data=pd.read_csv(data_path).dropna()
data = data[~(data == 0).any(axis=1)]
data['date'] = pd.to_datetime(data['date'])

def predict(data,model,seq_length):
    stock_predictions=[]
    diff_pred=[]
    total= len(data)-seq_length-1
    
    for start_idx in tqdm(range(total), desc="Prediction", ncols=80, ascii=True):
        open=data.loc[data.index[start_idx], "Open"]
        close=data.loc[data.index[start_idx+seq_length], "Close"]
        input=data_formatting(data,start_idx,1,seq_length,d_type)
        with torch.no_grad():
            pred=model(input)[0,-1,:]
        price_pred=pred[3]*open/100
        stock_predictions.append(price_pred)
        diff_pred.append((price_pred-close)/close*100)
        if start_idx%200==0:
            torch.cuda.empty_cache()
            
    stock_predictions=torch.stack(stock_predictions, dim=0).cpu().numpy()
    data["predict"]=np.concatenate((np.zeros(seq_length+1), stock_predictions))
    diff_pred=torch.stack(diff_pred, dim=0).cpu().numpy()
    data["diff_pred"]=np.concatenate((np.zeros(seq_length+1), diff_pred))
    data['Return (%)'] = data['Close'].pct_change() * 100
    return data
        
def eval(df,seqlen):
    import scipy.stats as stats
    df=df.iloc[seqlen:].copy()
    #RMSE 측정
    mse = np.square(np.mean((df['Close'] - df['predict']) ** 2))
    print("Root Mean Squared Error (RMSE):", mse)
    
    
    #승률 측정
    df['Is_Increased_Real'] = df['Return (%)'] > 0
    df['Is_Increased_Pred'] = df['diff_pred'] > 0
    df['Rate']=df['Is_Increased_Pred']^df['Is_Increased_Real']
    true_count = (df['Rate'] == True).sum()
    false_count = (df['Rate'] == False).sum()
    print(f"Win RATE: {true_count/(true_count+false_count)}")
    
    
    data['error']=(data['predict']-data['Close'])/data['Close']
    
    # ShapiroTest 측정
    stat, p_value = stats.shapiro(data['error'])
    if p_value > 0.05:
        print("데이터는 정규 분포를 따릅니다. (p > 0.05)")
    else:
        print("데이터는 정규 분포를 따르지 않습니다. (p <= 0.05)")
        
    


def plotting(data,seqlen=128,folder_path='./df'):
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    predict_plot=os.path.join(folder_path,'predict.jpg')
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], data['Close'], label='Close')
    plt.plot(data['date'], data['predict'], label='Predict')
    plt.title("Stock Price Chart")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.savefig(predict_plot)
    
    predict_tail_plot=os.path.join(folder_path,'predict_tail.jpg')
    datat=data.tail()
    plt.figure(figsize=(12, 6))
    plt.plot(datat['date'], datat['Close'], label='Close')
    plt.plot(datat['date'], datat['predict'], label='Predict')
    plt.title("Stock Price Chart")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.savefig(predict_tail_plot)
    
    data=data.iloc[seqlen:].copy()
    error_plot=os.path.join(folder_path,'error.jpg')
    plt.figure(figsize=(12, 6))
    data['error']=(data['predict']-data['Close'])/data['Close']*100
    plt.hist(data['error'],bins=200)
    plt.title("Predict ERROR Chart")
    plt.xlabel("ERROR(%)")
    plt.ylabel("Freq")
    plt.savefig(error_plot)
    
    predict_tail_plot=os.path.join(folder_path,'predict_return.jpg')
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], data['Return (%)'], label='Return (%)')
    plt.plot(data['date'], data["diff_pred"], label='Retrun Pred')
    plt.title("Stock Return Chart")
    plt.xlabel("Date")
    plt.ylabel("Return (%)")
    plt.savefig(predict_tail_plot)


if __name__ == "__main__":
    #model=TransformerModel()
    model=MambaModel()
    predict(data,model,127)
    plotting(data,folder_path='plots/Mamba2_2048')
    eval(data,128)