from model.mamba_stock import Mambamodeling
from dataset.dataset_iter import ConstantLengthDataset,ConstantAllLengthDataset
from argparse import Namespace
from accelerate import Accelerator
from torch.optim import AdamW
import transformers,datasets,torch
from mamba_ssm.models.config_mamba import MambaConfig
from transformers import get_scheduler
from huggingface_hub import Repository, get_full_repo_name

import logging,wandb
from torch.utils.tensorboard import SummaryWriter
import os

accelerator = Accelerator()


def eval(eval_dataloader, model, mse):
    model.eval()
    losses = []
    inf_data=0
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            loss = get_loss(model, mse, batch)
        if torch.isinf(loss) or loss.item()>10000:
            inf_data+=1
        else:
            losses.append(loss.item())
        print("loss 값: ",loss)
    
    
    # losses를 Tensor로 변환 후 평균 계산
    loss = torch.tensor(losses).mean()
    
    return loss.item(),losses


def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"],weight_decay = 0.1):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]
    
def get_loss(model,mse,batch):
    input=batch[:,:-1,:]
    ans=batch[:,-1,:3]
    pred=model(input)[:,-1,:3]
    loss=torch.sqrt(mse(pred,ans))
    return loss




def main():
    
    #데이터셋 불러오기
    eval_dataset = ConstantLengthDataset(folder_path='./data/KOSPI',
                                          seq_length=129,
                                          batch=1)
    
    # 모델 정의
    model=Mambamodeling(
        d_model=256,
        d_inermediate=768,
        n_layer=32,
        layer="Mamba1",
        pretrained_path="./result/Thrillcrazyer/Mambastocks_ver0.5/0000085000"
    )
    

    MSE = torch.nn.MSELoss()

    model, eval_dataset = accelerator.prepare(model ,eval_dataset)
    model.eval()
    
    # Test
    eval_loss,_= eval(eval_dataset,model,MSE)
    print("eval_loss: ",eval_loss)
   

if __name__ == '__main__':
    main()
    
