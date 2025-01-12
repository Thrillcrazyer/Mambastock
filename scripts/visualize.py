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
device="cuda"
d_type=torch.float32

def Mamba2Model():
    from model.mamba_stock import Mambamodeling
    pretrained_path="/workspace/weight/Mamba2"
    model=Mambamodeling(
            d_model=768,
            d_inermediate=2048,
            n_layer=8,
            layer="Mamba2",
            pretrained_path=pretrained_path
        ).to(device)
    model.eval()
    return model

def Mamba1Model():
    from model.mamba_stock import Mambamodeling
    pretrained_path="/workspace/weight/Mamba1"
    model=Mambamodeling(
            d_model=768,
            d_inermediate=2048,
            n_layer=8,
            layer="Mamba1",
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

    model=QwenStock(qwenconfig).from_pretrained("/workspace/weight/QWEN").to(device)
    return model

def xLSTMModel():
    from model.xlstm_stock import xLSTMStockModeling
    model=xLSTMStockModeling(pretrained_path="/workspace/weight/xLSTM2").to("cuda")
    
    return model



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
    
    data=data.iloc[seq_length+1:]    
    return data
        
def eval(df,folder_path):
    import scipy.stats as stats
    #MSE,MAE 측정
    file_path=os.path.join(folder_path,"result.txt")
    with open(file_path, "w") as file:
        
        data=(df['Close'] - df['predict'])/df['Close']
        error = (np.mean(np.square(data)))
    
        print("MSE:", error)
        file.write(f"MSE: {error}\n")
        print("MAE: ",(np.mean(np.abs(data))))
        file.write(f"MAE: {(np.mean(np.abs(data)))}\n")

        # ShapiroTest 측정
        stat, p_value = stats.shapiro(data)
        print(f"Shapiro Test: {p_value}")
        file.write(f"Shapiro Test: {p_value}\n")
    

def plotting(data,folder_path='./df'):
    
    predict_plot=os.path.join(folder_path,'predict.jpg')
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], data['Close'], label='Close')
    plt.plot(data['date'], data['predict'], label='Predict')
    plt.title("Stock Price Chart")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.savefig(predict_plot)
    
    error_plot=os.path.join(folder_path,'error_hist.jpg')
    plt.figure(figsize=(12, 6))
    data['error']=(data['predict']-data['Close'])/data['Close']*100
    plt.hist(data['error'],bins=200)
    plt.title("Predict ERROR Chart")
    plt.xlabel("ERROR(%)")
    plt.ylabel("Freq")
    plt.savefig(error_plot)
    
    predict_tail_plot=os.path.join(folder_path,'error.jpg')
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], data["error"], label='Error')
    plt.title("Stock Pred_ERROR Chart")
    plt.xlabel("Date")
    plt.ylabel("ERROR")
    plt.savefig(predict_tail_plot)

def visualize(data,model_name,stock_name):
    print(f"Evaluate {model_name}>>>>>>>>>>>>>>>")
    folder_path=os.path.join('plots',stock_name,model_name)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    if model_name=="Mamba2":
        model=Mamba2Model()
    if model_name=="Mamba1":
        model=Mamba1Model()
    if model_name=="Transformer":
        model=TransformerModel()
    if model_name=="xLSTM":
        model=xLSTMModel()
        
    data=predict(data,model,127)
    plotting(data,folder_path=folder_path)
    eval(data,folder_path=folder_path)
    

if __name__ == "__main__":
    #data_path="/workspace/data/BTCUSDT_hour.csv"
    #data=pd.read_csv(data_path).dropna()
    #data = data[~(data == 0).any(axis=1)]
    #data['date'] = pd.to_datetime(data['date'])
    from backtesting.test import EURUSD,GOOG
    data=GOOG
    data['date']=data.index
    
    #for model_name in ('Transformer','Mamba1','Mamba2','xLSTM'):
    #    visualize(data,model_name,"./GOOG")
    visualize(data,'xLSTM',"./GOOG")
    