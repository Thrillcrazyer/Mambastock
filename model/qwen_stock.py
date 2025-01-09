from transformers import AutoConfig,Qwen2Model,Qwen2PreTrainedModel
import torch
from torch import nn

class QwenStock(Qwen2PreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.model=Qwen2Model(config)
        self.model.embed_tokens = torch.nn.Linear(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self,inputs):
        output_qwen = self.model(inputs)[0]
        outputs=self.lm_head(output_qwen)
        return outputs

def count_parameters_in_millions(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/ 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/ 1e6:.2f}M")
    return total_params / 1e6, trainable_params / 1e6


if __name__ == '__main__':

    batch = 5
    seqlen = 1000
    device = "cpu"
    dtype = torch.float16
    ohlcv=5
    
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    
    qwenconfig=AutoConfig.from_pretrained(model_name,
                    intermediate_size= 2048,
                    max_position_embeddings= 128,
                    num_hidden_layers=8,
                    vocab_size= 5
                    )

    x = torch.randint(0, 5, (batch,seqlen+1,ohlcv))
    
    model=QwenStock(qwenconfig)
    input=x[:,:-1,:]
    ans=x[:,1:,:]
    MSE = nn.MSELoss()
    
    count_parameters_in_millions(model)
    
    predict=model(input)
    loss=MSE(predict,ans)
    
    
    
    print(loss)