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


    

class Trainer:
    def __init__(self,model,project_name,args):
        self.args=args
        self.device="cuda"
        self.accelerator=Accelerator()
        
        
        self.output_dir='./result/'+project_name
        if not os.path.exists(self.output_dir):  
            os.makedirs(self.output_dir)      

        #로깅
        self.logger, self.tb_writer, self.run_name = self.setup_logging(project_name.split("/")[1])
        self.logger.info(self.accelerator.state)
        
        train_dataloader, eval_dataloader=self.create_dataloaders(
                                                            train_data=self.args.train_dataset_path,
                                                            valid_data=self.args.valid_dataset_path,
                                                            seq_length=self.args.seq_length)
        
        
        self.samples_per_step = self.accelerator.state.num_processes * self.args.train_batch_size
        self.gradient_accumulation_steps=args.gradient_accumulation_steps
        self.save_checkpoint_steps=args.save_checkpoint_steps
        self.max_train_steps=args.max_train_steps
        
        optimizer = AdamW(self.get_grouped_params(model,weight_decay=args.weight_decay), lr=args.learning_rate)
        self.lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer,
                                    num_warmup_steps=args.num_warmup_steps,
                                    num_training_steps=args.max_train_steps,)
        
        mt=args.metric
        if mt == "RMSE":
            self.criterion = torch.nn.MSELoss()
        elif mt =="L1":
            self.criterion = torch.nn.L1Loss()
        elif mt =="Huber":
            self.criterion =torch.nn.HuberLoss()
        else:
            self.criterion = torch.nn.MSELoss()
        
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)
        
        return None
    
    def get_grouped_params(self,model,weight_decay, no_decay=["bias", "LayerNorm.weight"]):
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
    
    def get_loss(self,criterion,batch):
            input=batch[:,:-1,:]
            ans=batch[:,1:,:]
            

            if self.args.metric == "RMSE" or self.args.metric == None:
                predict=self.model(input)
                loss=torch.sqrt(criterion(predict,ans))
            else:
                loss=criterion(self.model(input),ans)
            return loss
        
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    def log_metrics(self,step, metrics):
        self.logger.info(f"Step {step}: {metrics}")
        if self.accelerator.is_main_process:
            wandb.log(metrics)
            [self.tb_writer.add_scalar(k, v, step) for k, v in metrics.items()]
    
    def create_dataloaders(self,train_data,valid_data,seq_length):
        train_data = './data'
        valid_data = './eval'
        train_dataset = ConstantAllLengthDataset(folder_path=train_data,
                                            seq_length=seq_length,
                                            batch=self.args.train_batch_size,
                                            is_multi_feature=self.args.multi_feature
                                            )
        valid_dataset = ConstantLengthDataset(folder_path=valid_data,
                                            seq_length=seq_length,
                                            batch=self.args.valid_batch_size,
                                            is_multi_feature=self.args.multi_feature)
        return train_dataset,valid_dataset
    
    def setup_logging(self,project_name):
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO, handlers=[
            logging.FileHandler(f"log/debug_{self.accelerator.process_index}.log"),
            logging.StreamHandler()])
        if self.accelerator.is_main_process: # 로깅을 한 번만 설정합니다.
            wandb.init(project=project_name, config=self.args)
            run_name = wandb.run.name
            tb_writer = SummaryWriter()
            print(vars(self.args))
            tb_writer.add_hparams(vars(self.args), {'0': 0})
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
    
    def eval(self):
        self.model.eval()
        losses=[]
        inf_data=0
        MSE=torch.nn.MSELoss()
        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                input=batch[:,:-1,:]
                ans=batch[:,1:,:]
                loss=torch.sqrt(MSE(self.model(input),ans))
            if torch.isinf(loss) or loss.item()>10000:
                inf_data+=1
            else:
                losses.append(loss.item())
        # losses를 Tensor로 변환 후 평균 계산
        loss = torch.tensor(losses).mean()
        
        return loss.item(),inf_data
    
    def train(self):
        self.model.train()
        completed_steps = 0
        
        for step, batch in enumerate(self.train_dataloader,start=1):
        
            batch=batch.to(self.device)
            loss=torch.sqrt(self.get_loss(self.criterion,batch))
            loss = loss / self.gradient_accumulation_steps
            self.accelerator.backward(loss)
            
            if step % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                completed_steps += 1
                            
            self.log_metrics(step, {'lr': self.get_lr(),'steps': completed_steps, 'loss/train': loss.item()})
            
            if step % self.save_checkpoint_steps == 0:
                self.logger.info('Evaluating and saving model checkpoint')
                eval_loss,inf_error= self.eval()
                self.log_metrics(step, {'loss/eval': eval_loss, 'inf_error': inf_error})
 
                self.accelerator.wait_for_everyone()
                
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                if self.accelerator.is_main_process:
                    outdir= os.path.join(self.output_dir,str(step).zfill(10))
                    unwrapped_model.save_pretrained(outdir)
                    
                self.model.train()
           
            if completed_steps >= self.max_train_steps:
                break
            
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if self.accelerator.is_main_process:
                outdir= os.path.join(self.output_dir,str(step).zfill(10))
                unwrapped_model.save_pretrained(outdir)
        print("Done!")

    