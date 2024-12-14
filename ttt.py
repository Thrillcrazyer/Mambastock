from model.mamba_stock import MambaStockHeadModel
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
config = {"train_batch_size": 512, # 12
          "valid_batch_size": 128, # 12
          "weight_decay": 0.1,
          "shuffle_buffer": 1000,
          "learning_rate": 5e-4, # 5e-4
          "lr_scheduler_type": "cosine",
          "num_warmup_steps": 750, # 2000
          "gradient_accumulation_steps": 1, # 1
          "max_train_steps": 300000, # 150000
          "max_eval_steps": -1,
          "seq_length": 129,
          "save_checkpoint_steps": 5000} # 15000
    
    
args = Namespace(**config)
accelerator = Accelerator()
samples_per_step = accelerator.state.num_processes * args.train_batch_size


#Logger 함수 정의
def setup_logging(project_name):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO, handlers=[
        logging.FileHandler(f"log/debug_{accelerator.process_index}.log"),
        logging.StreamHandler()])
    if accelerator.is_main_process: # 로깅을 한 번만 설정합니다.
        wandb.init(project=project_name, config=args)
        run_name = wandb.run.name
        tb_writer = SummaryWriter()
        tb_writer.add_hparams(vars(args), {'0': 0})
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_debug()
        transformers.utils.logging.set_verbosity_info()
    else:
        tb_writer = None
        run_name = ''
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    return logger, tb_writer, run_name


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
    
    # losses를 Tensor로 변환 후 평균 계산
    loss = torch.tensor(losses).mean()
    
    return loss.item(),inf_data


def create_dataloaders(batch,seq_length):
    train_data = './data'
    valid_data = './eval'
    train_dataset = ConstantAllLengthDataset(folder_path=train_data,
                                          seq_length=seq_length,
                                          batch=batch
                                          )
    valid_dataset = ConstantLengthDataset(folder_path=valid_data,
                                          seq_length=seq_length,
                                          batch=batch)
    
    
    return train_dataset,valid_dataset

def Mambamodeling():
    device = "cuda"
    dtype = torch.float32

    config = MambaConfig(
        d_model=256,
        d_intermediate=768,
        n_layer=32,
        vocab_size=5,
        ssm_cfg=dict(layer="Mamba1"),
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=5,
    )
    
    model = MambaStockHeadModel(config, device=device, dtype=dtype)
    return model

def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"],weight_decay = args.weight_decay):
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
    ans=batch[:,1:,:]
    loss=torch.sqrt(mse(model(input),ans))
    return loss



def log_metrics(step, metrics):
    logger.info(f"Step {step}: {metrics}")
    if accelerator.is_main_process:
        wandb.log(metrics)
        [tb_writer.add_scalar(k, v, step) for k, v in metrics.items()]

#허깅페이스 저장 장소와 로컬 저장 장소 지정
project_name="Thrillcrazyer/Mambastocks_ver0.5"
output_dir='./result/'+project_name

if not os.path.exists(output_dir):  
        os.makedirs(output_dir)      


# 로깅
logger, tb_writer, run_name = setup_logging(project_name.split("/")[1])
logger.info(accelerator.state)

def main():
    
    def get_lr():
        return optimizer.param_groups[0]['lr']
    
    #데이터셋 불러오기
    train_dataloader, eval_dataloader = create_dataloaders(batch=args.train_batch_size,seq_length=128)
    
    # 모델 정의
    model=Mambamodeling()
    
    #학습 스케쥴러 정의
    optimizer = AdamW(get_grouped_params(model), lr=args.learning_rate)
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer,
                                num_warmup_steps=args.num_warmup_steps,
                                num_training_steps=args.max_train_steps,)
    MSE = torch.nn.MSELoss()

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)
    model.train()
    
    completed_steps = 0
    device= "cuda"
    
    
    for step, batch in enumerate(train_dataloader,start=1):
        
        batch=batch.to(device)
        loss=torch.sqrt(get_loss(model,MSE,batch))
        loss = loss / args.gradient_accumulation_steps
        accelerator.backward(loss)
        
        if step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
                        
        log_metrics(step, {'lr': get_lr(), 'samples': step*samples_per_step,
                       'steps': completed_steps, 'loss/train': loss.item()})
        
        if step % args.save_checkpoint_steps == 0:
            logger.info('Evaluating and saving model checkpoint')
            eval_loss,inf_error= eval(eval_dataloader,model,MSE)
            log_metrics(step, {'loss/eval': eval_loss, 'inf_error': inf_error})
            #TODO
            #- eval function is error.
            #- dataset_iter is not good. redesign it.
            accelerator.wait_for_everyone()
            
            unwrapped_model = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                outdir= os.path.join(output_dir,str(step).zfill(10))
                unwrapped_model.save_pretrained(outdir)
                
            model.train()

        
        if completed_steps >= args.max_train_steps:
            break

if __name__ == '__main__':
    main()
    
