from transformers import AutoConfig
from model.mamba_stock import Mambamodeling
#from model.qwen_stock import QwenStock
#from model.lstm_stock import LSTM_Mixer,LSTMConfig
from train.trainer import Trainer
from argparse import Namespace
import argparse
import json

parser = argparse.ArgumentParser(description='StockTrain')
parser.add_argument('-c', '--config', default='/workspace/stock/configs/train_config.json', type=str)
args = parser.parse_args()


def main(config):
    
    #lstmconfig=LSTMConfig(d_intermediate=2048,hidden_size=768,num_layer=2)
    #model=LSTM_Mixer(lstmconfig).to("cuda")
    
    model=Mambamodeling(
        d_model=1024,
        d_inermediate=4098,
        n_layer=12,
        layer="Mamba2"
    )
    
    project_name="Thrillcrazyer/Mambastocks_ver0.5/Mamba2_latest"

    trainer=Trainer(model,project_name,config)
    
    trainer.train()
    
    return
    

if __name__ == '__main__':
    config = json.load(open(args.config))
    config = Namespace(**config)
    main(config)
    