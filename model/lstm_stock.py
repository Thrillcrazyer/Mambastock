import torch
from torch import nn
import os
import json
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from dataclasses import dataclass, field


@dataclass
class LSTMConfig:

    d_model: int = 2560
    d_intermediate: int = 0
    num_layer: int = 64
    hidden_size: int = 128
    ohlcv: int =5
    bidirectional: bool = False
    dropout=0.0



class LSTM_RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        """
        Args:
            dim (int): Normalization dimension (usually feature dimension).
            eps (float): A small constant for numerical stability.
        """
        super(LSTM_RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))  # Learnable scale parameter

    def forward(self, x):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).
        
        Returns:
            torch.Tensor: RMS-normalized tensor with the same shape as input.
        """
        # Compute the RMS value along the last dimension
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize input using RMS and apply learnable scale
        return x / rms * self.scale

class LSTM_MLP(nn.Module):
    def __init__(self, hidden_size,intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

class LSTM_Decoder_Layer(nn.Module):
    def __init__(self,hidden_size,d_intermediate,bidirectional=False, dropout=0.0,batch_first=True):
        super().__init__()
        self.hidden_size=hidden_size
        self.input_rms_norm = LSTM_RMSNorm(dim=hidden_size)
        self.post_lstm_layernorm=LSTM_RMSNorm(dim=hidden_size)
        self.mlp=LSTM_MLP(hidden_size=hidden_size,intermediate_size=d_intermediate)
        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        
    def forward(self,hidden_states):
        residual=hidden_states
        hidden_states = self.input_rms_norm(hidden_states) 
        h_0 = torch.zeros(1, hidden_states.shape[0], self.hidden_size).to("cuda")
        c_0 = torch.zeros(1, hidden_states.shape[0], self.hidden_size).to("cuda")      
        hidden_states,_ = self.lstm(hidden_states,(h_0, c_0))
        hidden_states=residual+hidden_states
        residual=hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
        

class LSTM_Mixer(nn.Module):
    def __init__(self,config:LSTMConfig):
        super().__init__()
        ohlcv_size=config.ohlcv
        hidden_size=config.hidden_size
        d_intermediate=config.d_intermediate
        bidirectional=config.bidirectional
        num_layers=config.num_layer
        dropout=config.dropout
        self.config=config
        self.embed_tokens=nn.Linear(ohlcv_size, hidden_size)

        self.lstm_layers = nn.ModuleList(
            LSTM_Decoder_Layer(
                hidden_size=hidden_size,
                d_intermediate=d_intermediate,
                dropout=dropout,
                bidirectional=bidirectional
            )
            for i in range(num_layers)
        )

        self.lm_head = nn.Linear(hidden_size, ohlcv_size, bias=False)
   
    def forward(self,inputs):
        outputs = self.embed_tokens(inputs)
        for layer in self.lstm_layers:  
            outputs = layer(outputs)
        outputs= self.lm_head(outputs)
        return outputs
     
    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = LSTMConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)


def count_parameters_in_millions(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/ 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/ 1e6:.2f}M")
    return total_params / 1e6, trainable_params / 1e6

if __name__ == '__main__':

    batch = 512
    seqlen = 128
    device = "cuda"
    dtype = torch.float16
    ohlcv=5

    x = torch.rand(batch,seqlen+1,ohlcv, dtype = torch.float32, device=device)
    
    lstmconfig=LSTMConfig(d_intermediate=2048,hidden_size=768,num_layer=8)
    model=LSTM_Mixer(lstmconfig).to("cuda")
    
    
    input=x[:,:-1,:]
    ans=x[:,1:,:]
    print(input.shape)
    print(ans.shape)
    MSE = nn.MSELoss()
    
    count_parameters_in_millions(model)
    
    predict=model(input)
    
    loss=MSE(predict,ans)
    
    
    
    print("loss: ",loss)   
        
        
  